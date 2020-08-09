# from gluoncv import model_zoo, data, utils
# from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
# from gluoncv.data.transforms.presets.ssd import transform_test
#
# import mxnet as mx
import numpy as np
import cv2
from openpose_wrapper import openpose_25_kp

#
# class HumanDetector:
#     def __init__(self):
#         self.detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
#         self.detector.reset_class(["person"], reuse_weights=['person'])
#         self.pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)
#
#     def detect(self, img_or_img_name):
#         if isinstance(img_or_img_name, str):
#             img_or_img_name = cv2.imread(img_or_img_name)
#         pred_coords = openpose_25_kp(img_or_img_name)
#         return pred_coords


class LocConvert:
    def __init__(self, P):
        self.P = P

    def solve_3d_point_with_z(self, image_point):
        P = self.P
        x, y, known_z = image_point

        b = np.asarray(
            [[- known_z * P[0, 2] - P[0, 3]], [- known_z * P[1, 2] - P[1, 3]], [- known_z * P[2, 2] - P[2, 3]]])
        a = np.array(
            [
                [P[0, 0], P[0, 1], -x],
                [P[1, 0], P[1, 1], -y],
                [P[2, 0], P[2, 1], -1]
            ]
        )

        known_x, known_y, s = np.linalg.solve(a, b)
        known_x, known_y, s = float(known_x), float(known_y), float(s)
        return known_x, known_y, known_z

    def p_to_loc(self, point):
        img_x, img_y = point
        pos_3d = self.solve_3d_point_with_z([img_x, img_y, 0])
        return pos_3d[:2]
