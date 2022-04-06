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


from collections import OrderedDict

from torch import nn as nn
from torch.hub import load_state_dict_from_url


## COPY-PASTED and modified from torchvision
class VGG19(nn.Module):

    def __init__(self, layer_activation_indices, batch_norm=False):
        super().__init__()
        self.layer_activation_indices = layer_activation_indices
        self.blocks = _vgg('vgg19', 'E', batch_norm=batch_norm, pretrained=True, progress=True)
        self.conv_block_indices = []

        self.layers = []
        for bi, block in enumerate(self.blocks):
            for layer in block:
                self.layers += [layer]
                if isinstance(layer, nn.Conv2d):
                    self.conv_block_indices += [bi]

        if len(self.layer_activation_indices) != len(set(layer_activation_indices).intersection(set(self.conv_block_indices))):
            raise ValueError("The specified layer indices are not of a conv block")

        self.net = nn.Sequential(*self.layers)
        self.net.eval()
        self.net.requires_grad_(False)

    def requires_grad_(self, requires_grad: bool = True):
        return super().requires_grad_(False)

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, x):
        layer_outputs = {}
        for bi, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)
            if bi in self.layer_activation_indices:
                layer_outputs[bi] = x
        layer_outputs['final'] = x
        return layer_outputs


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [ nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2)])]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [ nn.ModuleList([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])]
            else:
                layers += [ nn.ModuleList([conv2d, nn.ReLU(inplace=True)])]
            in_channels = v
    return nn.ModuleList(layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    layers = make_layers(cfgs[cfg], batch_norm=batch_norm)
    if pretrained:
        archname = arch
        if batch_norm:
            archname += "_bn"
        state_dict = load_state_dict_from_url(model_urls[archname],
                                              progress=progress)
        state_dict2 = OrderedDict()
        for key in state_dict.keys():
            if "features" in key:
                # hack layer names
                state_dict2[key[len("features."):]] = state_dict[key]
        layers_ = []
        for bi, block in enumerate(layers):
            for layer in block:
                # layer.name = "features." + layer.name
                layers_ += [layer]
        net = nn.Sequential(*layers_)
        net.load_state_dict(state_dict2)
    return layers


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}