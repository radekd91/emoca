import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional, Linear, LeakyReLU, Sequential
import torch.nn.functional as F
from gdl.utils.other import class_from_str


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_size : int,
        out_size: int,
        hidden_layer_sizes : list,
        hidden_activation = None,
        batch_norm = None
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.batch_norm = batch_norm
        self.hidden_layer_sizes = hidden_layer_sizes
        hidden_activation = hidden_activation or LeakyReLU(0.2)
        self.hidden_activation = hidden_activation
        self._build_network()

    def _build_network(self):
        layers = []
        # layers += [Linear(self.in_size, self.hidden_layer_sizes[0])]
        # layers += [self.hidden_activation]
        layer_sizes = [self.in_size] + self.hidden_layer_sizes
        for i in range(1, len(layer_sizes)):
            layers += [
                Linear(layer_sizes[i - 1], layer_sizes[i])
            ]
            if self.batch_norm is not None:
                layers += [self.batch_norm(layer_sizes[i])]
            layers += [self.hidden_activation]
        layers += [Linear(layer_sizes[-1], self.out_size)]
        self.model = Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y
