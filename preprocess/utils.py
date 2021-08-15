import cv2
import numpy as np

CocoPairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
             [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], 
             [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def draw_keypoints(image, keypoints):
    
    image = np.array(image, dtype=np.uint8)

    for i, points in enumerate(keypoints):

        # point
        px, py, pc = points
        color = CocoColors[i]
        point = (px, py)
        print(i, px, py, pc)

        # pairs
        pair_points = CocoPairs[i]
        point_1_idx, point_2_idx = pair_points
        point1 = (keypoints[point_1_idx-1][0], keypoints[point_1_idx-1][1])
        point2 = (keypoints[point_2_idx-1][0], keypoints[point_2_idx-1][1])
        
        if (point1[0] + point1[1]) != 0 and (point2[0] + point2[1]) != 0:
            cv2.line(image, point1, point2, color, 3)
        cv2.circle(image, point, 5, color, -1)
        cv2.putText(image,str(i), point,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

    cv2.imshow('pose', image)
    cv2.waitKey(1000)