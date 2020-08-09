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
    def __init__(self, M):
        self.M = M

    def bbox_to_loc(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        foot_pos = [(xmin + xmax) / 2, ymax]
        loc = self.PerspectiveTransform(foot_pos)
        return loc

    def pose_to_loc(self, pose):
        ankle_p = pose[-1]
        loc = self.PerspectiveTransform(ankle_p)
        return loc

    def p_to_loc(self, point):
        loc = self.PerspectiveTransform(point)
        return loc

    def PerspectiveTransform(self, obj):
        """
        :param obj:
        :param M:
        :return:
        """
        x, y = obj
        x_new, y_new, t = np.matrix(self.M) * np.matrix([x, y, 1]).T
        x_new, y_new = x_new / t, y_new / t
        return [round(float(x_new), 1), round(float(y_new), 1)]
