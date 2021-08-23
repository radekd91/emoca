import torch
from gdl.layers.losses.EmonetLoader import get_emonet
from pathlib import Path
import torch.nn.functional as F
from gdl.models.EmoNetModule import EmoNetModule
from gdl.models.IO import get_checkpoint_with_kwargs


class EmoNetLoss(torch.nn.Module):
# class EmoNetLoss(object):

    def __init__(self, device, emonet=None):
        super().__init__()
        if emonet is None:
            self.emonet = get_emonet(device).eval()
        elif isinstance(emonet, str):
            path = Path(emonet)
            if path.is_dir():
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
                emonet_module = EmoNetModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
                self.emonet = emonet_module.emonet
            else:
                raise ValueError("Please specify the directory which contains the config of the trained Emonet.")

        else:
            self.emonet = emonet
        self.emonet.requires_grad_(False)
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

    def compute_loss(self, input_images, output_images):
        # input_emotion = None
        # self.output_emotion = None
        input_emotion = self.emonet_out(input_images)
        output_emotion = self.emonet_out(output_images)
        self.input_emotion = input_emotion
        self.output_emotion = output_emotion

        emo_feat_loss_1 = self.emo_feat_loss(input_emotion['emo_feat'], output_emotion['emo_feat'])
        emo_feat_loss_2 = self.emo_feat_loss(input_emotion['emo_feat_2'], output_emotion['emo_feat_2'])
        valence_loss = self.valence_loss(input_emotion['valence'], output_emotion['valence'])
        arousal_loss = self.arousal_loss(input_emotion['arousal'], output_emotion['arousal'])
        expression_loss = self.expression_loss(input_emotion['expression'], output_emotion['expression'])
        return emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss

    @property
    def input_emo(self):
        return self.input_emotion

    @property
    def output_emo(self):
        return self.output_emotion

#
# if __name__ == "__main__":
#     net = get_emonet(load_pretrained=False)