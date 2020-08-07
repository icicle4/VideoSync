from pose_info import HumanDetector, LocConvert
from config import Config
import cv2
import numpy as np
import os
from match_cost import video_loc_match_score

cfg = Config()


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


def locs_in_image(img_or_img_path, detector, loc_converter, debug, writer):
    if isinstance(img_or_img_path, str):
        img_or_img_path = cv2.imread(img_or_img_path)

    img = img_or_img_path
    img_save = np.copy(img)
    #src_height, src_width = img.shape[:2]

    # src = np.array(
    #             [
    #              [0, src_height],
    #              [src_width, 0],
    #              [src_width, src_height]], dtype=np.float32
    # )
    #
    # dst_height, dst_width = (512, 512 / src_height * src_width)
    #
    #
    # dst = np.array(
    #             [
    #              [0, dst_height],
    #              [dst_width, 0],
    #              [dst_width, dst_height]], dtype=np.float32
    #         )
    #
    # trans = cv2.getAffineTransform(dst, src)
    poses = detector.detect(img)

    locs = []
    for pose in poses:
        right_ankle = pose[19]

        if right_ankle[-1] > 0.3:
            score = right_ankle[-1]

            # foot_p = right_ankle[:2]
            #
            # dst_foot_p = affine_transform(foot_p, trans)

            loc = loc_converter.p_to_loc(right_ankle[:2])

            img_save = cv2.circle(img_save, (int(right_ankle[0]), int(right_ankle[1])), 4, (0, 0, 255))
            img_save = cv2.putText(img_save, str(f"{loc[0]} {loc[1]}"), (int(right_ankle[0]), int(right_ankle[1])),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
            locs.append([*loc, score])

    writer.write(img_save)
    return locs


def early_locs_in_video(video_path, duration, detector, loc_converter, debug, debug_video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    process_frame_count = int(fps * duration)

    writer = Mp4VideoWriter(debug_video_path, height, width, fps)
    print(video_path, debug_video_path, height, width, fps)

    video_locs = []
    f = 0
    while f < process_frame_count:
        ret, frame = cap.read()

        if not ret:
            break

        frame_locs = locs_in_image(frame, detector, loc_converter, debug, writer)
        video_locs.append(frame_locs)
        f += 1
    return video_locs

def debug_video_path(video_path):
    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    debug_video_path = os.path.join(video_dir, f'{video_name}_debug.mp4')
    return debug_video_path


if __name__ == '__main__':
    human_detector = HumanDetector()

    video1 = '/Users/linda/Downloads/video_sync_test/前.mp4'
    video2 = '/Users/linda/Downloads/video_sync_test/侧中.mp4'
    # 我们假设视频的对齐误差为2s

    cam1_loc_converter = LocConvert(cfg.front_M)
    cam2_loc_converter = LocConvert(cfg.mid_M)

    debug = True
    # 从每段视频的开始处，获取4s内人的位置分布情况
    video1_locs = early_locs_in_video(video1, duration=4.0, detector=human_detector,
                                      loc_converter=cam1_loc_converter, debug=debug,
                                      debug_video_path=debug_video_path(video1))
    video2_locs = early_locs_in_video(video2, duration=4.0, detector=human_detector,
                                      loc_converter=cam2_loc_converter, debug=debug,
                                      debug_video_path=debug_video_path(video2))

    match_scores = []

    for i in range(100):
        video1_to_match_locs = video1_locs[i:i+100]
        video2_to_match_locs = video2_locs[:100]
        match_scores.append(video_loc_match_score(video1_to_match_locs, video2_to_match_locs))

    for i in range(100):
        video1_to_match_locs = video1_locs[:100]
        video2_to_match_locs = video2_locs[i:i+100]
        match_scores.append(video_loc_match_score(video1_to_match_locs, video2_to_match_locs))

    print(match_scores)
    idx = np.argmax(match_scores)
    print(idx)
    if idx > 100:
        offset = idx - 100
    else:
        offset = -idx

    print(offset)