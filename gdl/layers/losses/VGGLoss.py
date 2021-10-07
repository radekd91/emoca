"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import torch
import torch.nn as nn
from gdl.models.VGG import VGG19

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class VGG19Loss(nn.Module):

    def __init__(self, layer_activation_indices_weights, diff=torch.nn.functional.l1_loss, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.vgg19 = VGG19(sorted(layer_activation_indices_weights.keys()), batch_norm=batch_norm)
        self.layer_activation_indices_weights = layer_activation_indices_weights
        self.diff = diff

    def forward(self, x, y):
        feat_x = self.vgg19(x)
        feat_y = self.vgg19(y)

        out = {}
        loss = 0
        for idx, weight in self.layer_activation_indices_weights.items():
            d = self.diff(feat_x[idx], feat_y[idx])
            out[idx] = d
            loss += d*weight
        return loss, out

