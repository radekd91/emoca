import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional, Linear, LeakyReLU, Sequential

from utils.other import class_from_str


class MLP(torch.nn.Module):
    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__()
        self.in_size = config.in_size
        self.out_size = config.out_size
        self.metric = config.metric
        self.hidden_layer_sizes = OmegaConf.to_container(config.hidden_layer_sizes)
        self.loss = class_from_str(config.loss, F)
        self._build_network()

    def _build_network(self):
        layers = []
        layers += [Linear(self.in_size, self.hidden_layer_sizes[0])]
        layers += [LeakyReLU(0.2)]
        for i in range(1, len(self.hidden_layer_sizes)):
            layers += [
                Linear(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i])
            ]
            layers += [LeakyReLU(0.2)]
        layers += [Linear(self.hidden_layer_sizes[-1], self.out_size)]
        self.model = Sequential(*layers)

    def forward(self, sample):
        # print(sample.keys())
        x = sample[self.metric]
        y = self.model(x)
        return y
