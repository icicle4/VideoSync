import numpy as np
import cv2


class Config:
    def __init__(self):
        self.side_front_image_objs = np.array(
            [[678, 651],
             [718, 1010],
             [1673, 841],
             [1410, 586]], dtype=np.float32
        )

        self.side_front_world_objs = np.array(
            [[0, 0],
             [0, 900],
             [600, 900],
             [900, 0]], dtype=np.float32
        )

        self.side_front_M = cv2.getPerspectiveTransform(self.side_front_image_objs, self.side_front_world_objs)

        self.mid_image_objs = np.array(
            [[426, 741],
             [700, 379],
             [1536, 694],
             [1695, 353]], dtype=np.float32
        )

        self.mid_world_objs = np.array(
            [[600, 900],
             [600, 0],
             [1200, 900],
             [1800, 0]], dtype=np.float32
        )

        self.mid_M = cv2.getPerspectiveTransform(self.mid_image_objs, self.mid_world_objs)

        self.front_image_objs = np.array(
            [[89, 798],
             [1828, 796],
             [462, 518],
             [1452, 517 ]], dtype=np.float32
        )

        self.front_world_objs = np.array(
            [
                [0, 0],
                [0, 900],
                [900, 0],
                [900, 900]
            ], dtype=np.float32
        )
        self.front_M = cv2.getPerspectiveTransform(self.front_image_objs, self.front_world_objs)
