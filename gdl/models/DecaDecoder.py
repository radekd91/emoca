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

Parts of the code were adapted from the original DECA release: 
https://github.com/YadiraF/DECA/ 
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1, out_scale=1, sample_mode='bilinear'):
        super(Generator, self).__init__()
        self.out_scale = out_scale

        self.init_size = 32 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 64
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 128
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 256
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img * self.out_scale


from gdl.layers.AdaIN import AdaIN


class AdaInUpConvBlock(nn.Module):

    def __init__(self, dim_in, dim_out, cond_dim, kernel_size=3, scale_factor=2, sample_mode='bilinear'):
        super().__init__()
        self.norm = AdaIN(cond_dim, dim_in)
        self.actv = nn.LeakyReLU(0.2, inplace=True)
        if scale_factor > 0:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=sample_mode)
        else:
            self.upsample = None
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride=1, padding=1)


    def forward(self, x, condition):
        x = self.norm(x, condition)
        x = self.actv(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.conv(x)
        return x


class GeneratorAdaIn(nn.Module):
    def __init__(self, latent_dim, condition_dim, out_channels=1, out_scale=1, sample_mode='bilinear'):
        super().__init__()
        self.out_scale = out_scale

        self.init_size = 32 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        # self.conv_blocks = nn.Sequential(
        #     # nn.BatchNorm2d(128),
        #     # nn.Upsample(scale_factor=2, mode=sample_mode),  # 16
        #     # nn.Conv2d(128, 128, 3, stride=1, padding=1),
        #     AdaInUpConvBlock(128,128, condition_dim),
        #     # nn.BatchNorm2d(128, 0.8),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Upsample(scale_factor=2, mode=sample_mode),  # 32
        #     # nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     AdaInUpConvBlock(128, 64, condition_dim),
        #     # nn.BatchNorm2d(64, 0.8),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Upsample(scale_factor=2, mode=sample_mode),  # 64
        #     # nn.Conv2d(64, 64, 3, stride=1, padding=1),
        #     AdaInUpConvBlock(64, 64, condition_dim),
        #     # nn.BatchNorm2d(64, 0.8),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Upsample(scale_factor=2, mode=sample_mode),  # 128
        #     # nn.Conv2d(64, 32, 3, stride=1, padding=1),
        #     AdaInUpConvBlock(64, 32, condition_dim),
        #     # nn.BatchNorm2d(32, 0.8),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Upsample(scale_factor=2, mode=sample_mode),  # 256
        #     # nn.Conv2d(32, 16, 3, stride=1, padding=1),
        #     AdaInUpConvBlock(32, 16, condition_dim),
        #     # nn.BatchNorm2d(16, 0.8),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Conv2d(16, out_channels, 3, stride=1, padding=1),
        #     AdaInUpConvBlock(16, out_channels, condition_dim, scale_factor=0)
        #     nn.Tanh(),
        # )
        self.conv_block1 = AdaInUpConvBlock(128,128, condition_dim, sample_mode=sample_mode) # 16
        self.conv_block2 = AdaInUpConvBlock(128, 64, condition_dim, sample_mode=sample_mode) # 32
        self.conv_block3 = AdaInUpConvBlock(64, 64, condition_dim, sample_mode=sample_mode)  # 64
        self.conv_block4 = AdaInUpConvBlock(64, 32, condition_dim, sample_mode=sample_mode)  # 128
        self.conv_block5 = AdaInUpConvBlock(32, 16, condition_dim, sample_mode=sample_mode) #  256
        self.conv_block6 = AdaInUpConvBlock(16, out_channels, condition_dim, scale_factor=0) # 256
        self.conv_blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4,
                            self.conv_block5, self.conv_block6]
        self.out_actv = nn.Tanh()


    def forward(self, z, cond):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        for i, block in enumerate(self.conv_blocks):
            out = block(out, cond)
        img = self.out_actv(out)
        return img * self.out_scale

