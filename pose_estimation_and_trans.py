
from config import Config
import numpy as np

from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.data.transforms.presets.ssd import transform_test

from openpose_wrapper import openpose_25_kp

import mxnet as mx
import cv2


class HumanDetector:
    def __init__(self):
        self.detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
        self.detector.reset_class(["person"], reuse_weights=['person'])
        self.pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

    def detect(self, img_or_img_name):
        if isinstance(img_or_img_name, str):
            img_or_img_name = cv2.imread(img_or_img_name)
        pred_coords = openpose_25_kp(img_or_img_name)
        return pred_coords


class Mp4VideoWriter:
    def __init__(self, video_path, height, width, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter(
            video_path,
            fourcc,
            float(fps),
            (int(width), int(height)), True
        )

    def write(self, frame):
        self.video.write(frame)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def locs_in_image(img_or_img_path, detector, out_image_path):

    if not out_image_path:
        out_image_path = 'debug.png'

    if isinstance(img_or_img_path, str):
        img_or_img_path = cv2.imread(img_or_img_path)

    img = img_or_img_path
    img_save = np.copy(img)
    src_height, src_width = img.shape[:2]

    src = np.array(
                [
                 [0, src_height],
                 [src_width, 0],
                 [src_width, src_height]], dtype=np.float32
    )

    dst_height, dst_width = (512, 512 / src_height * src_width)


    dst = np.array(
                [
                 [0, dst_height],
                 [dst_width, 0],
                 [dst_width, dst_height]], dtype=np.float32
            )

    trans = cv2.getAffineTransform(dst, src)
    poses = detector.detect(img)

    ret = []
    for i, pose in enumerate(poses):
        new_poses = []
        for j in range(len(pose)):
            joint = pose[j]
            if joint[-1] > 0.3:
                score = joint[-1]
                joint_pos = joint[:2]

                dst_joint = affine_transform(joint_pos, trans)

                img_save = cv2.circle(img_save, (int(dst_joint[0]), int(dst_joint[1])), 4, (0, 0, 255))

                if j == len(pose) - 1:
                    img_save = cv2.putText(img_save, str(f"{i}"), (int(dst_joint[0]), int(dst_joint[1])),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
                new_poses.append([*dst_joint, score])
        ret.append(new_poses)

    cv2.imwrite(out_image_path, img_save)

    return np.asarray(ret, dtype=np.float32)



if __name__ == '__main__':
    human_detector = HumanDetector()

    img_path = ''  #此处为需要进行人员检测的图片路径
    out_img_path = '' #检测可视化结果存储路径
    poses = locs_in_image(img_path, human_detector, out_img_path)
