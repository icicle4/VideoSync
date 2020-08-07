import numpy as np
from munkres import Munkres


def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp


def video_loc_match_score(video1_locs, video2_locs):
    match_scores = []
    for video1_loc, video2_loc in zip(video1_locs, video2_locs):
        match_scores.append(
            view_loc_match_score(video1_loc, video2_loc)
        )
    return sum(match_scores)


def view_loc_match_score(view1_loc, view2_loc):

    dis_matrix = np.zeros((len(view1_loc), len(view2_loc)))
    for i in range(len(view1_loc)):
        for j in range(len(view2_loc)):
            loc1 = view1_loc[i][:2]
            loc2 = view2_loc[j][:2]

            dis_matrix[i, j] = float(np.linalg.norm(np.asarray(loc1) - np.asarray(loc2)))

    dis_matrix_save = np.copy(dis_matrix)
    if len(view1_loc) > len(view2_loc):
        dis_matrix = np.concatenate(
            (
                dis_matrix,
                np.zeros((len(view1_loc), len(view1_loc) - len(view2_loc))) + 1e5
            ), axis=1
        )

    pairs = py_max_match(dis_matrix)

    dises = []
    for row, col in pairs:
        if row < len(view1_loc) and col < len(view2_loc) and dis_matrix_save[row, col] < 200:
            dises.append(
                dis_matrix_save[row, col]
            )
    return len(dises) / (sum(dises) / 100 + 1e-5)
