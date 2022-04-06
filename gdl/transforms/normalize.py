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


class NormalizeGeometricData(object):

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def _init(self, dtype, device):
        assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')
        self.mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        self.std = torch.as_tensor(self.std, dtype=dtype, device=device)

    def __call__(self, data):
        self._init(data.x.dtype, data.x.device)
        data.x = (data.x - self.mean)/self.std
        data.y = (data.y - self.mean)/self.std
        return data

    def inv(self, data):
        self._init(data.x.dtype, data.x.device)
        data.x = (data.x*self.std) + self.mean
        data.y = (data.y*self.std) + self.mean
        return data
