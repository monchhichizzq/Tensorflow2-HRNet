import tensorflow as tf


class CocoDataset(object):
    def __init__(self, txt_path, dataset_type):
        if dataset_type == "train":
            self.data_dir = txt_path
        elif dataset_type == "valid":
            self.data_dir = txt_path
        else:
            raise ValueError("Invalid dataset_type name!")

    def __get_length_of_dataset(self, dataset):
        count = 0
        for _ in dataset:
            count += 1
        return count

    def generate_dataset(self, batch_size):
        dataset = tf.data.TextLineDataset(filenames=self.data_dir)
        dataset_length = self.__get_length_of_dataset(dataset)
        dataset = dataset.batch(batch_size=batch_size)
        return dataset, dataset_length


if __name__ == '__main__':
    batch_size = 8
    txt_path = '../preparation/data_txt/coco/coco_train.txt'
    coco_train = CocoDataset(txt_path, dataset_type='train')
    dataset, dataset_length = coco_train.generate_dataset(batch_size)
    print('Number of training dataset samples: ', dataset_length)
    for batch_data in dataset:
        print(batch_data)
