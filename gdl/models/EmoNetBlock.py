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
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.InstanceNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.InstanceNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.InstanceNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class EmoNetHead(nn.Module):

    def __init__(self, num_modules=2, n_expression=8, n_reg=2, n_blocks=4, attention=True, input_image_only=False, temporal_smoothing=False):
        super().__init__()
        self.num_modules = num_modules
        self.n_expression = n_expression
        self.n_reg = n_reg
        self.n_blocks = n_blocks
        self.input_image_only = input_image_only
        self.attention = attention
        if input_image_only and attention:
            raise ValueError("Options 'input_image_only' and 'attention' cannot be both activated")
        self.temporal_smoothing = temporal_smoothing
        self.init_smoothing = False

        if self.temporal_smoothing:
            self.n_temporal_states = 5
            self.init_smoothing = True
            self.temporal_weights = torch.Tensor([0.1,0.1,0.15,0.25,0.4]).unsqueeze(0).unsqueeze(2).cuda() #Size (1,5,1)

        self._create_Emo()


    def _create_Emo(self):
        if self.input_image_only:
            n_in_features = 3
        elif self.attention:
            n_in_features = 256 * (self.num_modules + 1)  # Heatmap is applied hence no need to have it
        else:
            n_in_features = 256 * (self.num_modules + 1) + 68  # 68 for the heatmap

        n_features = [(256, 256)] * (self.n_blocks)

        self.emo_convs = []
        self.conv1x1_input_emo_2 = nn.Conv2d(n_in_features, 256, kernel_size=1, stride=1, padding=0)
        for in_f, out_f in n_features:
            self.emo_convs.append(ConvBlock(in_f, out_f))
            self.emo_convs.append(nn.MaxPool2d(2, 2))
        self.emo_net_2 = nn.Sequential(*self.emo_convs)
        self.avg_pool_2 = nn.AvgPool2d(4)
        self.emo_fc_2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                      nn.Linear(128, self.n_expression + self.n_reg))

        # Separate eomotion parameters from FAN parameters
        self.emo_parameters = list(set(self.parameters()).difference(set(self.fan_parameters)))
        self.emo_modules = list(set(self.modules()).difference(set(self.fan_modules)))


    def forward(self, x, hg_features=None, tmp_out=None, reset_smoothing=False, intermediate_features=False):
        # Resets the temporal smoothing
        if self.init_smoothing:
            self.init_smoothing = False
            self.temporal_state = torch.zeros(x.size(0), self.n_temporal_states, self.n_expression + self.n_reg).cuda()
        if reset_smoothing:
            self.temporal_state = self.temporal_state.zeros_()

        hg_features = torch.cat(tuple(hg_features), dim=1)

        if self.input_image_only:
            assert hg_features is None and tmp_out is None
            emo_feat = x
        elif self.attention:
            mask = torch.sum(tmp_out, dim=1, keepdim=True)
            hg_features *= mask
            emo_feat = torch.cat((x, hg_features), dim=1)
        else:
            emo_feat = torch.cat([x, hg_features, tmp_out], dim=1)

        emo_feat_conv1D = self.conv1x1_input_emo_2(emo_feat)
        final_features = self.emo_net_2(emo_feat_conv1D)
        final_features = self.avg_pool_2(final_features)
        batch_size = final_features.shape[0]
        final_features = final_features.view(batch_size, final_features.shape[1])
        if intermediate_features:
            emo_feat2 = final_features
        final_features = self.emo_fc_2(final_features)

        if self.temporal_smoothing:
            with torch.no_grad():
                self.temporal_state[:, :-1, :] = self.temporal_state[:, 1:, :]
                self.temporal_state[:, -1, :] = final_features
                final_features = torch.sum(self.temporal_weights * self.temporal_state, dim=1)

        res = {'heatmap': tmp_out, 'expression': final_features[:, :-2], 'valence': final_features[:, -2],
               'arousal': final_features[:, -1]}

        if intermediate_features:
            res['emo_feat'] = emo_feat
            res['emo_feat_2'] = emo_feat2

        return res
