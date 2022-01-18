import inspect
import sys
from pathlib import Path
from gdl.utils.other import get_path_to_externals
import torch


def get_emonet(device=None, load_pretrained=True):
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path_to_emonet = get_path_to_externals() / "emonet"
    if not(str(path_to_emonet) in sys.path  or str(path_to_emonet.absolute()) in sys.path):
        # print(f"Adding EmoNet path '{path_to_emonet}'")
        sys.path += [str(path_to_emonet)]

    from emonet.models import EmoNet
    # n_expression = 5
    n_expression = 8

    # Create the model
    net = EmoNet(n_expression=n_expression).to(device)

    # if load_pretrained:
    state_dict_path = Path(
        inspect.getfile(EmoNet)).parent.parent.parent / 'pretrained' / f'emonet_{n_expression}.pth'
    print(f'Loading the EmoNet model from {state_dict_path}.')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=False)
    if not load_pretrained:
        print("Created an untrained EmoNet instance")
        net.reset_emo_parameters()

    net.eval()
    return net