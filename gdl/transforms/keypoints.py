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


import torch
import numpy as np


class KeypointTransform(torch.nn.Module):

    def __init__(self, scale_x=1., scale_y=1.):
        super().__init__()
        self.scale_x = scale_x
        self.scale_y = scale_y

    def set_scale(self, scale_x=1., scale_y=1.):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def forward(self, points):
        raise NotImplementedError()

class KeypointScale(KeypointTransform):

    def __init__(self, scale_x=1., scale_y=1.):
        super().__init__(scale_x, scale_y)

    def forward(self, points):
        points_ = points.clone()
        points_[..., 0] *= self.scale_x
        points_[..., 1] *= self.scale_y
        return points_


class KeypointNormalization(KeypointTransform):

    def __init__(self, scale_x=1., scale_y=1.):
        super().__init__(scale_x, scale_y)

    def forward(self, points):
        # normalization the way EMOCA uses it.
        # the keypoints are not used in image space but in normalized space
        # for loss computation
        # the normalization is as follows:
        if isinstance(points, torch.Tensor):
            points_ = points.clone()
        elif isinstance(points, np.ndarray):
            points_ = points.copy()
        else:
            raise ValueError(f"Invalid type of points {str(type(points))}")
        points_[..., 0] -= self.scale_x/2
        points_[..., 0] /= self.scale_x/2
        points_[..., 1] -= self.scale_y/2
        points_[..., 1] /= self.scale_y/2
        return points_

    def inv(self, points):
        if isinstance(points, torch.Tensor):
            points_ = points.clone()
        elif isinstance(points, np.ndarray):
            points_ = points.copy()
        else:
            raise ValueError(f"Invalid type of points {str(type(points))}")
        points_[..., 0] *= self.scale_x / 2
        points_[..., 0] += self.scale_x / 2
        points_[..., 1] *= self.scale_y / 2
        points_[..., 1] += self.scale_y / 2
        return points_