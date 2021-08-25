import torch
from gdl.layers.losses.EmonetLoader import get_emonet
from pathlib import Path
import torch.nn.functional as F
from gdl.models.EmoNetModule import EmoNetModule
from gdl.models.EmoSwinModule import EmoSwinModule
from gdl.models.EmoCnnModule import EmoCnnModule
from gdl.models.IO import get_checkpoint_with_kwargs
from gdl.utils.other import class_from_str
import sys


def emo_network_from_path(path):
    print(f"Loading trained EmoNet from: '{path}'")

    def load_configs(run_path):
        from omegaconf import OmegaConf
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)
        return conf

    cfg = load_configs(path)
    checkpoint_mode = 'best'
    stages_prefixes = ""

    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, stages_prefixes,
                                                               checkpoint_mode=checkpoint_mode,
                                                               # relative_to=relative_to_path,
                                                               # replace_root=replace_root_path
                                                               )
    checkpoint_kwargs = checkpoint_kwargs or {}

    if 'emodeca_type' in cfg.model.keys():
        module_class = class_from_str(cfg.model.emodeca_type, sys.modules[__name__])
    else:
        module_class = EmoNetModule

    emonet_module = module_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False,
                                                      **checkpoint_kwargs)
    return emonet_module


def create_emo_loss(device, emoloss = None, trainable=False):
    if emoloss is None:
        return EmoNetLoss(device, emonet=emoloss)
    if isinstance(emoloss, str):
        path = Path(emoloss)
        if path.is_dir():
            emo_loss = emo_network_from_path(path)

            if isinstance(emo_loss, EmoNetModule):
                emonet = emo_loss.emonet
                return EmoNetLoss(device, emonet=emonet, trainable=trainable)
            else:
                return EmoBackboneLoss(device, emo_loss, trainable=trainable)
        else:
            raise ValueError("Please specify the directory which contains the config of the trained Emonet.")


class EmoLossBase(torch.nn.Module):

    def __init__(self, trainable=False):
        super().__init__()
        self.emo_feat_loss = F.l1_loss
        self.valence_loss = F.l1_loss
        self.arousal_loss = F.l1_loss
        # self.expression_loss = F.kl_div
        self.expression_loss = F.l1_loss
        self.input_emotion = None
        self.output_emotion = None
        self.trainable = trainable


    @property
    def input_emo(self):
        return self.input_emotion

    @property
    def output_emo(self):
        return self.output_emotion

    def compute_loss(self, input_images, output_images):
        # input_emotion = None
        # self.output_emotion = None

        # there is no need to keep gradients for input (even if we're finetuning, which we don't, it's the output image we'd wannabe finetuning on)
        with torch.no_grad():
            input_emotion = self(input_images)

        output_emotion = self(output_images)
        self.input_emotion = input_emotion
        self.output_emotion = output_emotion

        if 'emo_feat' in input_emotion.keys():
            emo_feat_loss_1 = self.emo_feat_loss(input_emotion['emo_feat'], output_emotion['emo_feat'])
        else:
            emo_feat_loss_1 = None
        emo_feat_loss_2 = self.emo_feat_loss(input_emotion['emo_feat_2'], output_emotion['emo_feat_2'])
        valence_loss = self.valence_loss(input_emotion['valence'], output_emotion['valence'])
        arousal_loss = self.arousal_loss(input_emotion['arousal'], output_emotion['arousal'])
        if 'expression' in input_emotion.keys():
            expression_loss = self.expression_loss(input_emotion['expression'], output_emotion['expression'])
        elif 'expr_classification' in input_emotion.keys():
            expression_loss = self.expression_loss(input_emotion['expr_classification'], output_emotion['expr_classification'])
        else:
            raise ValueError("Missing expression")
        return emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss


    def _get_trainable_params(self):
        if self.trainable:
            return list(self.parameters())
        return []


class EmoNetLoss(EmoLossBase):
# class EmoNetLoss(object):

    def __init__(self, device, emonet=None, trainable=False):
        super().__init__(trainable)
        if emonet is None:
            self.emonet = get_emonet(device).eval()
        # elif isinstance(emonet, str):
        #     path = Path(emonet)
        #     if path.is_dir():
        #         print(f"Loading trained EmoNet from: '{path}'")
        #         def load_configs(run_path):
        #             from omegaconf import OmegaConf
        #             with open(Path(run_path) / "cfg.yaml", "r") as f:
        #                 conf = OmegaConf.load(f)
        #             return conf
        #
        #         cfg = load_configs(path)
        #         checkpoint_mode = 'best'
        #         stages_prefixes = ""
        #
        #         checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, stages_prefixes,
        #                                                                    checkpoint_mode=checkpoint_mode,
        #                                                                    # relative_to=relative_to_path,
        #                                                                    # replace_root=replace_root_path
        #                                                                    )
        #         checkpoint_kwargs = checkpoint_kwargs or {}
        #         emonet_module = EmoNetModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
        #         self.emonet = emonet_module.backbone
        #     else:
        #         raise ValueError("Please specify the directory which contains the config of the trained Emonet.")

        else:
            self.emonet = emonet
        if not trainable:
            self.emonet.eval()
            self.emonet.requires_grad_(False)
        else:
            self.emonet.train()
            self.emonet.emo_parameters_requires_grad(True)

        # self.emonet.eval()
        # self.emonet = self.emonet.requires_grad_(False)
        # self.transforms = Resize((256, 256))
        self.size = (256, 256)
        self.emo_feat_loss = F.l1_loss
        self.valence_loss = F.l1_loss
        self.arousal_loss = F.l1_loss
        # self.expression_loss = F.kl_div
        self.expression_loss = F.l1_loss
        self.input_emotion = None
        self.output_emotion = None

    @property
    def network(self):
        return self.emonet

    def to(self, *args, **kwargs):
        self.emonet = self.emonet.to(*args, **kwargs)
        # self.emonet = self.emonet.requires_grad_(False)
        # for p in self.emonet.parameters():
        #     p.requires_grad = False

    def eval(self):
        self.emonet = self.emonet.eval()
        # self.emonet = self.emonet.requires_grad_(False)
        # for p in self.emonet.parameters():
        #     p.requires_grad = False

    def train(self, mode: bool = True):
        # super().train(mode)
        if hasattr(self, 'emonet'):
            self.emonet = self.emonet.eval() # evaluation mode no matter what, it's just a loss function
            # self.emonet = self.emonet.requires_grad_(False)
            # for p in self.emonet.parameters():
            #     p.requires_grad = False

    def forward(self, images):
        return self.emonet_out(images)

    def emonet_out(self, images):
        images = F.interpolate(images, self.size, mode='bilinear')
        # images = self.transform(images)
        return self.emonet(images, intermediate_features=True)


    def _get_trainable_params(self):
        if self.trainable:
            return self.emonet.emo_parameters
        return []

class EmoBackboneLoss(EmoLossBase):
# class EmoNetLoss(object):

    # def __init__(self, device, backbone=None):
    def __init__(self, device, backbone, trainable=False):
        super().__init__(trainable)
        # if backbone is None:
        #     self.backbone = get_emonet(device).eval()
        # el
        # if isinstance(backbone, str):
        #     path = Path(backbone)
        #     if path.is_dir():
        #         print(f"Loading trained EmoNet from: '{path}'")
        #         def load_configs(run_path):
        #             from omegaconf import OmegaConf
        #             with open(Path(run_path) / "cfg.yaml", "r") as f:
        #                 conf = OmegaConf.load(f)
        #             return conf
        #
        #         cfg = load_configs(path)
        #         checkpoint_mode = 'best'
        #         stages_prefixes = ""
        #
        #         checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, stages_prefixes,
        #                                                                    checkpoint_mode=checkpoint_mode,
        #                                                                    # relative_to=relative_to_path,
        #                                                                    # replace_root=replace_root_path
        #                                                                    )
        #         checkpoint_kwargs = checkpoint_kwargs or {}
        #
        #         from gdl.utils.other import class_from_str
        #         module_class = class_from_str(config.model.emodeca_type)
        #
        #         emonet_module = module_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
        #         self.backbone = emonet_module.backbone
        #     else:
        #         raise ValueError("Please specify the directory which contains the config of the trained Emonet.")
        #
        # else:
        self.backbone = backbone
        if trainable:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        else:
            self.backbone.requires_grad_(True)

    def forward(self, images):
        return self.backbone._forward(images)
        # return self.backbone(images)

#
# if __name__ == "__main__":
#     net = get_emonet(load_pretrained=False)