import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional, Linear, LeakyReLU, Sequential
import torch.nn.functional as F
from utils.other import class_from_str


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_size : int,
        out_size: int,
        hidden_layer_sizes : list,
        hidden_activation = None
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_layer_sizes = hidden_layer_sizes
        hidden_activation = hidden_activation or LeakyReLU(0.2)
        self.hidden_activation = hidden_activation
        self._build_network()

    def _build_network(self):
        layers = []
        layers += [Linear(self.in_size, self.hidden_layer_sizes[0])]
        layers += [self.hidden_activation]
        for i in range(1, len(self.hidden_layer_sizes)):
            layers += [
                Linear(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i])
            ]
            layers += [self.hidden_activation]
        layers += [Linear(self.hidden_layer_sizes[-1], self.out_size)]
        self.model = Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y
