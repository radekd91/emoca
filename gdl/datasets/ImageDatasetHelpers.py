"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""


import numpy as np
from skimage.transform import estimate_transform, warp


def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0
        center_y =  bottom - (bottom - top) / 2.0
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0 + old_size * 0.12
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    elif type == "mediapipe":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0
        # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    else:
        raise NotImplementedError(f" bbox2point not implemented for {type} ")
    if isinstance(center_x, np.ndarray):
        center = np.stack([center_x, center_y], axis=1)
    else: 
        center = np.array([center_x, center_y])
    return old_size, center


def point2bbox(center, size):
    size2 = size / 2

    src_pts = np.array(
        [[center[0] - size2, center[1] - size2], [center[0] - size2, center[1] + size2],
         [center[0] + size2, center[1] - size2]])
    return src_pts


def point2transform(center, size, target_size_height, target_size_width):
    target_size_width = target_size_width or target_size_height
    src_pts = point2bbox(center, size)
    dst_pts = np.array([[0, 0], [0, target_size_width - 1], [target_size_height - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform


def bbpoint_warp(image, center, size, target_size_height, target_size_width=None, output_shape=None, inv=True, landmarks=None, 
        order=3 # order of interpolation, bicubic by default
        ):
    target_size_width = target_size_width or target_size_height
    tform = point2transform(center, size, target_size_height, target_size_width)
    tf = tform.inverse if inv else tform
    output_shape = output_shape or (target_size_height, target_size_width)
    dst_image = warp(image, tf, output_shape=output_shape, order=order)
    if landmarks is None:
        return dst_image
    # points need the matrix
    if isinstance(landmarks, np.ndarray):
        assert isinstance(landmarks, np.ndarray)
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = tf_lmk(landmarks[:, :2])
    elif isinstance(landmarks, list): 
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = [] 
        for i in range(len(landmarks)):
            dst_landmarks += [tf_lmk(landmarks[i][:, :2])]
    elif isinstance(landmarks, dict): 
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = {}
        for key, value in landmarks.items():
            dst_landmarks[key] = tf_lmk(landmarks[key][:, :2])
    else: 
        raise ValueError("landmarks must be np.ndarray, list or dict")
    return dst_image, dst_landmarks