
import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))

import cv2
import tensorflow as tf
import numpy as np
from preprocess.transforms import read_image
from preprocess.transforms import RandomCropTransform
from preprocess.transforms import ResizeTransform
from preprocess.utils import draw_keypoints

class DataLoader(object):
    def __init__(self, **kwargs):
        self.num_of_joints = kwargs.get('num_of_joints', 17)
        self.img_height = kwargs.get('height', 256)
        self.img_width = kwargs.get('width', 256)
        self.image_size = np.array([self.img_height, self.img_width])
        self.hp_height = kwargs.get('heatmap_height', 64)
        self.hp_width = kwargs.get('heatmap_width', 64)
        self.heatmap_size = np.array([self.hp_height, self.hp_width])
        self.sigma = kwargs.get('sigma', 2)
        self.transform_method = kwargs.get('transform_method', 'resize')
        self.visual = True

    def __tensor2list(self, tensor_data):
        list_data = []
        length = tensor_data.shape[0]
        for i in range(length):
            list_data.append(bytes.decode(tensor_data[i].numpy(), encoding="utf-8"))
        return list_data

    def __get_one_human_instance_keypoints(self, line_keypoints):
        '''
            读取txt file中每一行, 提取图片路径, bbox框, 关键点, 并将关键点转为x,y,confidence
            返回 从原图中, 截取, 缩放为目标大小, 标准化 的图片, 所有的关键点数列, 存在的关键点数列
        '''
        # step 1: 读取txt file中每一行, 提取图片路径, bbox框, 关键点, 并将关键点转为x,y,confidence
        line_keypoints = line_keypoints.strip()
        split_line = line_keypoints.split(" ")
        image_file = split_line[0]
        _, bbox = self.__convert_string_to_float_and_int(split_line[3:7]) # bbox 为 int
        keypoints, _ = self.__convert_string_to_float_and_int(split_line[7:]) # keypoints 为 float 
        keypoints_tensor = tf.convert_to_tensor(value=keypoints, dtype=tf.dtypes.float32)
        keypoints_tensor = tf.reshape(keypoints_tensor, shape=(-1, 3))

        # step 2:  返回还有人体的截取图片, 同时缩放截取图片以及关键点
        # Resize the image, and change the coordinates of the keypoints accordingly.
        image_tensor, keypoints = self.__image_and_keypoints_process(image_file, keypoints_tensor, bbox)

        # step 3: 删除坐标为 0 的 关键点
        # keypoints_3d: 所有关键点包括不存在的
        # keypoints_3d_exist: 存在的关键点
        keypoints_3d, keypoints_3d_exist = self.__get_keypoints_3d(keypoints)
        return image_tensor, keypoints_3d, keypoints_3d_exist

    def __convert_string_to_float_and_int(self, string_list):
        float_list = []
        int_list = []
        for data_string in string_list:
            data_float = float(data_string)
            data_int = int(data_float)
            float_list.append(data_float)
            int_list.append(data_int)
        return float_list, int_list
    
    def __image_and_keypoints_process(self, image_dir, keypoints, bbox):
        # 输入 图片路径, 关键点, bbox框
        # 图片被标准化
        image_tensor = read_image(image_dir)
        if self.transform_method == "random crop":
            raise NotImplementedError("Not available temporarily.")
            # transform = RandomCropTransform(image=image_tensor, keypoints=keypoints, bbox=bbox, resize_h=self.image_size[0], resize_w=self.image_size[1], num_of_joints=self.num_of_joints)
            # resized_image, resize_ratio, crop_rect = transform.image_transform()
            # keypoints = transform.keypoints_transform(resize_ratio, crop_rect)
            # return resized_image, keypoints
        elif self.transform_method == "resize":
            transform = ResizeTransform(image=image_tensor, 
                                        keypoints=keypoints, 
                                        bbox=bbox, 
                                        resize_h=self.image_size[0], 
                                        resize_w=self.image_size[1], 
                                        num_of_joints=self.num_of_joints,
                                        visual=False)
            # 将图片中有人的部分根据给的框分割
            # 将分割后的图片reisze成目标大小(256, 256, 3)
            # 返回 resize 后的图片, resize_ratio, 以及left, top的点
            resized_image, resize_ratio, left_top = transform.image_transform()
            # 将选中框内人体关键点 转变为 对应大小图片 keypoints shape: (17, 3) 
            keypoints = transform.keypoints_transform(resize_ratio, left_top)

            # if self.visual:
            #     draw_keypoints(resized_image, keypoints)

            return resized_image, keypoints
        else:
            raise ValueError("Invalid TRANSFORM_METHOD.")

    def __get_keypoints_3d(self, keypoints):
        keypoints_3d_list = []
        keypoints_3d_exist_list = []
        for i in range(self.num_of_joints):
            keypoints_3d_list.append(tf.convert_to_tensor([keypoints[i, 0], keypoints[i, 1], 0], dtype=tf.dtypes.float32))
            exist_value = keypoints[i, 2]
            if exist_value > 1:
                exist_value = 1
            # exist_value: (1: exist , 0: not exist)
            keypoints_3d_exist_list.append(tf.convert_to_tensor([exist_value, exist_value, 0], dtype=tf.dtypes.float32))
        
        keypoints_3d = tf.stack(values=keypoints_3d_list, axis=0)  # shape: (self.num_of_joints, 3)
        keypoints_3d_exist = tf.stack(values=keypoints_3d_exist_list, axis=0)   # shape: (self.num_of_joints, 3)
        return keypoints_3d, keypoints_3d_exist

    def __generate_target(self, keypoints_3d, keypoints_3d_exist):
        target_weight = np.ones((self.num_of_joints, 1), dtype=np.float32)
        target_weight[:, 0] = keypoints_3d_exist[:, 0]

        target = np.zeros((self.num_of_joints, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        temp_size = self.sigma * 3
        for joint_id in range(self.num_of_joints):
            feature_stride = self.image_size / self.heatmap_size
            mu_x = int(keypoints_3d[joint_id][0] / feature_stride[1] + 0.5)
            mu_y = int(keypoints_3d[joint_id][1] / feature_stride[0] + 0.5)
            upper_left = [int(mu_x - temp_size), int(mu_y - temp_size)]
            bottom_right = [int(mu_x + temp_size + 1), int(mu_y + temp_size + 1)]
            if upper_left[0] >= self.heatmap_size[1] or upper_left[1] >= self.heatmap_size[0] or bottom_right[0] < 0 or bottom_right[1] < 0:
                # Set the joint invisible.
                target_weight[joint_id] = 0
                continue
            size = 2 * temp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]   # shape : (size, 1)
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
            g_x = max(0, -upper_left[0]), min(bottom_right[0], self.heatmap_size[1]) - upper_left[0]
            g_y = max(0, -upper_left[1]), min(bottom_right[1], self.heatmap_size[0]) - upper_left[1]
            img_x = max(0, upper_left[0]), min(bottom_right[0], self.heatmap_size[1])
            img_y = max(0, upper_left[1]), min(bottom_right[1], self.heatmap_size[0])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        target = tf.convert_to_tensor(target, tf.dtypes.float32)
        target = tf.transpose(a=target, perm=[1, 2, 0])    # shape : (self.heatmap_size[0], self.heatmap_size[1], self.num_of_joints)
        target_weight = tf.convert_to_tensor(target_weight, tf.dtypes.float32)    # shape : (self.num_of_joints, 1)

        # if self.visual:
            
        #     print('target: {}, {} - {}, {}'.format(np.shape(target), np.min(target), np.max(target), target.dtype))
        #     target = np.array(target, dtype=np.uint8)
        #     copy_target = target.copy()
        #     copy_target = np.argmax(copy_target, axis=-1)
        #     copy_target = np.expand_dims(copy_target, axis=-1)
        #     copy_target = cv2.resize(copy_target, (500, 500), interpolation=cv2.INTER_CUBIC)
        #     copy_target = copy_target*16
        #     print(np.max(copy_target))
        #     cv2.imshow('target', copy_target)
        #     cv2.waitKey(1000)

        return target, target_weight

    def get_ground_truth(self, batch_data):
        batch_target = []
        batch_target_weight = []
        batch_images = []
        self.batch_keypoints_list = self.__tensor2list(batch_data)
        for item in self.batch_keypoints_list:
            # item: image_path, height, width, x, y, w, h, pose
            image, keypoints_3d, keypoints_3d_exist = self.__get_one_human_instance_keypoints(line_keypoints=item)
            # 返回 heatmaps, weight list
            target, target_weight = self.__generate_target(keypoints_3d.numpy(), keypoints_3d_exist.numpy())
            batch_images.append(image)
            batch_target.append(target)
            batch_target_weight.append(target_weight)
        batch_images_tensor = tf.stack(values=batch_images, axis=0)  # (batch_size, image_height, image_width, channels)
        batch_target_tensor = tf.stack(values=batch_target, axis=0)    # (batch_size, heatmap_height, heatmap_width, num_of_joints)
        batch_target_weight_tensor = tf.stack(values=batch_target_weight, axis=0)  # (batch_size, num_of_joints, 1)
        return batch_images_tensor, batch_target_tensor, batch_target_weight_tensor


if __name__ == '__main__':
    from preprocess.coco_dataset import CocoDataset
    batch_size = 8
    txt_path = '../preparation/data_txt/coco/coco_train.txt'
    coco_train = CocoDataset(txt_path, dataset_type='train')
    dataset, dataset_length = coco_train.generate_dataset(batch_size)
    print('Number of training dataset samples: ', dataset_length)

    train_params = {'num_of_joints': 17,
                    'height': 256,
                    'width': 256,
                    'heatmap_height': 64, 
                    'heatmap_width': 64,
                    'sigma': 2, 
                    'transform_method': 'resize'}

    train_loader = DataLoader(**train_params)

    for batch_data in dataset:
        batch_images, batch_target, batch_target_weight = train_loader.get_ground_truth(batch_data)
        print('batch_images: ', np.shape(batch_images))
        print('batch_target: ', np.shape(batch_target))
        print('batch_target_weight: ', np.shape(batch_target_weight))
        print('')
