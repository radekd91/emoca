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
import torch.nn.functional as F
from gdl.layers.losses.EmonetLoader import get_emonet


class EmoNetRegressor(torch.nn.Module):

    def __init__(self, outsize, last_op=None):
        super().__init__()
        self.emonet = get_emonet().eval()
        # self.emonet.eval()
        # self.emonet = self.emonet.requires_grad_(False)
        # self.transforms = Resize((256, 256))
        self.input_image_size = (256, 256) # for now, emonet is pretrained for this particual image size (the official impl)

        self.feature_to_use = 'emo_feat_2'

        if self.feature_to_use == 'emo_feat_2':
            self.emonet_feature_size = 256
            self.fc_size = 256
        else:
            raise NotImplementedError(f"Not yet implemented for feature '{self.feature_to_use}'")

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.emonet_feature_size, self.fc_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_size, outsize)
        )
        self.last_op = last_op

    def forward(self, images):
        images = F.interpolate(images, self.input_image_size, mode='bilinear')
        out = self.emonet(images, intermediate_features=True)
        # out has the following keys: 'heatmap', 'expression' 'valence', 'arousal', 'emo_feat', 'emo_feat_2'
        out = self.layers(out[self.feature_to_use])
        return out


class EmonetRegressorStatic(EmoNetRegressor):

    def __init__(self, outsize, last_op=None):
        super().__init__(outsize, last_op)
        self.emonet.requires_grad_(False)
        self.emonet.eval()

    def train(self, mode=True):
        # this one only trains the FC layers
        self.emonet.eval()
        self.layers.train(mode)
        return self

