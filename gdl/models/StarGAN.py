import stargan
import torch
from pathlib import Path
from stargan.core.model import build_style_encoder, build_generator, build_FAN
from stargan.core.checkpoint import CheckpointIO
from omegaconf import DictConfig, OmegaConf
from munch import Munch


class StarGANWrapper(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        if isinstance(cfg, (str, Path)):
            self.args = OmegaConf.load(cfg)
        else:
            self.args = cfg

        generator = build_generator(self.args)
        style_encoder = build_style_encoder(self.args)

        self.nets_ema = Munch(generator=generator,
                         # mapping_network=mapping_network_ema,
                         style_encoder=style_encoder)
        fan = build_FAN(self.args)
        if fan is not None:
            self.nets_ema.fan = fan
        self._load_checkpoint('latest')

    def _load_checkpoint(self, step):
        self.ckptios = [
            CheckpointIO(str(Path(self.args.checkpoint_dir) / '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        if isinstance(step, str):
            if step == 'latest':
                path = Path(self.ckptios[0].fname_template)
                ckpts = list(path.parent.glob("*.ckpt"))
                ckpts.sort(reverse=True)
                num = ckpts[0].name.split("_")[0]
                step = int(num)
            else:
                raise ValueError(f"Invalid resume_iter value: {step}")

        if step is not None and not isinstance(step, int):
            raise ValueError(f"Invalid resume_iter value: {step}")

        for ckptio in self.ckptios:
            ckptio.load(step)

    def _normalize(self, img, mean, std, max_pixel_value=1.0):
        mean = torch.tensor(mean, dtype=torch.float32)
        mean *= max_pixel_value
        if img.ndim == 4:
            mean = mean.view(1, mean.numel(), 1 , 1)
        else:
            mean = mean.view(mean.numel(), 1, 1)

        std = torch.tensor(std, dtype=torch.float32)
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
        if img.ndim == 4:
            mean = mean.view(1, mean.numel(), 1 , 1)
        else:
            mean = mean.view(mean.numel(), 1, 1)

        std = torch.tensor(std, dtype=torch.float32)
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
        image = self._normalize(sample["input_image"],
                                    mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
        ref_image = self._normalize(sample["ref_image"],
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
        return translated_image
