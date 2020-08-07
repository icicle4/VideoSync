import sys

# 运行之前，你需要安装openpose，并把 params["model_folder"] 的值替换为你的值，同样的将 sys.path.append 中的路径替换为你的本机路径。
# 注意：替换的两个文件路径均在你的openpose安装路径下

try:
    params = dict()
    params["model_folder"] = "/home/icicle/openpose/models"
    # params['net_resolution'] = "-1x720"
    sys.path.append('/home/icicle/openpose/build/python/')
    from openpose import pyopenpose as op
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
except ImportError as e:
    print(e, 'you need a openpose, ')
    exit(0)

import copy
import os
import json
import numpy as np


def static_openpose_25_kp(skeleton_dir, image_name):
    with open(os.path.join(skeleton_dir, '{}.json'.format(image_name)), 'r') as f:
        poses = np.asarray(json.load(f), dtype=np.float32)

    return poses


def openpose_25_kp(image):
    datum = op.Datum()
    image_to_process = copy.copy(image)
    datum.cvInputData = image_to_process
    opWrapper.emplaceAndPop([datum])
    pose_key_points = datum.poseKeypoints
    return pose_key_points