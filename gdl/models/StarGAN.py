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


import stargan
import torch
from pathlib import Path
from stargan.core.model import build_style_encoder, build_generator, build_FAN
from stargan.core.checkpoint import CheckpointIO
from omegaconf import DictConfig, OmegaConf
from munch import Munch
import torch.nn.functional as F


class StarGANWrapper(torch.nn.Module):

    def __init__(self, cfg, stargan_repo=None):
        super().__init__()

        if isinstance(cfg, (str, Path)):
            self.args = OmegaConf.load(cfg)
            if self.args.wing_path is not None:
                self.args.wing_path = str(Path(stargan_repo ) / self.args.wing_path)
                self.args.checkpoint_dir = str(Path(stargan_repo ) / self.args.checkpoint_dir)
        else:
            self.args = cfg

        generator = build_generator(self.args)
        style_encoder = build_style_encoder(self.args)

        generator.requires_grad_(False)
        style_encoder.requires_grad_(False)

        self.nets_ema = Munch(generator=generator,
                         # mapping_network=mapping_network_ema,
                         style_encoder=style_encoder)
        fan = build_FAN(self.args)
        if fan is not None:
            self.nets_ema.fan = fan
        self._load_checkpoint('latest')

    @property
    def background_mode(self):
        return self.args.deca_background

    def _load_checkpoint(self, step):
        self.ckptios = [
            CheckpointIO(str(Path(self.args.checkpoint_dir) / '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        if isinstance(step, str):
            if step == 'latest':
                path = Path(self.ckptios[0].fname_template)
                ckpts = list(path.parent.glob("*.ckpt"))
                ckpts.sort(reverse=True)
                found = False
                for ckpt in ckpts:
                # split_name = ckpts[0].name.split("_")[0]
                    split_name = ckpt.name.split("_")
                    if len(split_name) < 1:
                        print(f"Skipping checkpoint '{ckpt}'")
                        continue
                    num = split_name[0]
                    step = int(num)
                    print(f"Loading Stargan from {ckpt}")
                    found = True
                    break
                if not found:
                    raise RuntimeError(f"Checkpoint not found in '{path.parent}'")
            else:
                raise ValueError(f"Invalid resume_iter value: '{step}'")

        if step is not None and not isinstance(step, int):
            raise ValueError(f"Invalid resume_iter value: '{step}' or type: '{type(step)}'")

        for ckptio in self.ckptios:
            ckptio.load(step)

    def _normalize(self, img, mean, std, max_pixel_value=1.0):
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32, device = img.device)
        mean *= max_pixel_value
        if img.ndim == 4:
            mean = mean.view(1, mean.numel(), 1 , 1)
        else:
            mean = mean.view(mean.numel(), 1, 1)

        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32, device = img.device)
        std *= max_pixel_value
        if img.ndim == 4:
            std = std.view(1, std.numel(), 1 , 1)
        else:
            std = std.view(std.size(), 1, 1)
            # std = std.view(1, *std.shape)
        mean = mean.to(device=img.device)
        std = std.to(device=img.device)

        denominator = torch.reciprocal(std)

        img = img.to(torch.float32,)
        img = img - mean
        img = img * denominator
        return img

    def _denormalize(self, img,  mean, std, max_pixel_value=1.0):
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32, device=img.device)
        mean *= max_pixel_value
        if img.ndim == 4:
            mean = mean.view(1, mean.numel(), 1 , 1)
        else:
            mean = mean.view(mean.numel(), 1, 1)

        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32, device=img.device)
        std *= max_pixel_value
        if img.ndim == 4:
            std = std.view(1, std.numel(), 1 , 1)
        else:
            std = std.view(std.size(), 1, 1)
            # std = std.view(1, *std.shape)

        denominator = torch.reciprocal(std)
        img = img / denominator
        img = img + mean
        return img
        # out = (x + 1) / 2
        # return out.clamp_(0, 1)

    def forward(self, sample):
        # images come in range [0,1] and will get transfered from [-1,1] to StarGAN
        input_image = sample["input_image"]
        input_image = F.interpolate(input_image, (self.args.img_size, self.args.img_size), mode='bilinear')
        image = self._normalize(input_image,
                                    mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
        input_ref_image = F.interpolate(sample["ref_image"],
                                        (self.args.img_size, self.args.img_size),
                                        mode='bilinear')
        ref_image = self._normalize(input_ref_image,
                                    mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

        target_domain_label = sample["target_domain"]
        target_style = self.nets_ema.style_encoder(ref_image, target_domain_label)

        if hasattr(self.nets_ema, 'fan'):
            masks = self.nets_ema.fan.get_heatmap(image)
        else:
            masks = None

        translated_image = self.nets_ema.generator(image, target_style, masks=masks)
        translated_image = self._denormalize(translated_image,
                                    mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
        output_size = sample["input_image"].shape[2:4]
        translated_image = F.interpolate(translated_image,
                      output_size,
                      mode='bilinear')

        return translated_image
