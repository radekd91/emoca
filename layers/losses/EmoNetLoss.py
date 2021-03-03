import torch
from torchvision.transforms.transforms import Resize
from pathlib import Path
import sys
import inspect
import torch.nn.functional as F


def get_emonet(device=None):
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path_to_emonet = Path(__file__).absolute().resolve().parent.parent.parent.parent / "emonet"
    if not(str(path_to_emonet) in sys.path  or str(path_to_emonet.absolute()) in sys.path):
        print(f"Adding EmoNet path '{path_to_emonet}'")
        sys.path += [str(path_to_emonet)]

    from emonet.models import EmoNet
    # n_expression = 5
    n_expression = 8

    # Loading the model

    state_dict_path = Path(inspect.getfile(EmoNet)).parent.parent.parent /'pretrained' / f'emonet_{n_expression}.pth'

    print(f'Loading the EmoNet model from {state_dict_path}.')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    net = EmoNet(n_expression=n_expression).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net


class EmoNetLoss(torch.nn.Module):
# class EmoNetLoss(object):

    def __init__(self, device):
        super().__init__()
        self.emonet = get_emonet(device).eval()
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
