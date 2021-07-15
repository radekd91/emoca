import os, sys
import torch
from torch.optim import Adam
from torch.nn import BatchNorm1d, BatchNorm2d, Conv1d, Conv2d, ReLU, Module, Sequential, Linear, Tanh
from torch.nn import MultiheadAttention, GRU, LSTM, RNNBase
import torch.nn.functional as F
from gdl.datasets.MeshDataset import EmoSpeechDataModule
from collections import OrderedDict

from pytorch_lightning.core import LightningModule

from facenet_pytorch import InceptionResnetV1


class CNNBackbone(Module):

    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    @property
    def output_size(self):
        raise NotImplementedError()

    def forward(self, x):
        return self.cnn(x)


class FaceNetIncreptionResnet(CNNBackbone):

    def __init__(self, pretrained='vggface2', num_classes=None):
        if num_classes is None:
            self.out_size = 512
        else:
            self.out_size = num_classes
        cnn = InceptionResnetV1(pretrained=pretrained, num_classes=num_classes)
        super().__init__(cnn)

    @property
    def output_size(self):
        return self.out_size


class EmotionRecognition(LightningModule):

    def __init__(self, dm,
                 feature_extractor='FaceNetIncreptionResnet',
                 recurrent_layers='GRU',
                 recurrent_num_hidden=512):
        super().__init__()
        if feature_extractor == 'FaceNetIncreptionResnet':
            self.feature_extractor = FaceNetIncreptionResnet(pretrained='vggface2')
        else:
            raise ValueError("Invalid CNN specifier '%s'." % feature_extractor)
        if recurrent_layers is None:
            self.recurrent = None
        else:
            self.recurrent = RNNBase(recurrent_layers, self.feature_extractor.output_size,
                                     recurrent_num_hidden)
        self.attention = None
        self.outputs = OrderedDict()

    def forward(self, x):
        feat = self.feature_extractor(x)

        if self.recurrent is not None:
            assert self.attention is not None
            feat = self.recurrent(feat)
            feat = self.attention(feat)

        outputs = OrderedDict()
        for name, output_layer in self.outputs:
            outputs[name] = output_layer(feat)

        return outputs

#
# class RecurrentModule(Module):
#
#     def __init__(self, layer_size_list, hidden_activations, output_activation):
#         super().__init__()
#         self.layer_size_list = []
#         self.layers = []
#         self.hidden_activations = hidden_activations
#         self.activations = []
#         self.nl = len(layer_size_list)
#         assert (len(hidden_activations))
#         for i in range(self.nl -1):
#             self.layers += [self.layer_type(layer_size_list[i], layer_size_list[i+1])]
#             if i == self.nl - 2:
#                 self.activations += [output_activation]
#             else:
#                 self.activations += [hidden_activations]
#
#     @property
#     def layer_type(self, idx=None):
#         raise NotImplementedError
#
#
# class GRUState(RecurrentModule):
#
#     def layer_type(self, idx=None):
#         return GRU
#
#
# class LSTMState(RecurrentModule):
#
#     def layer_type(self, idx=None):
#         return LSTM


class MLP(Module):

    def __init__(self, layer_size_list, hidden_activations, output_activation):
        super().__init__()
        self.layer_size_list = []
        self.layers = []
        self.hidden_activations = hidden_activations
        self.activations = []
        self.nl = len(layer_size_list)
        assert (len(hidden_activations))
        for i in range(self.nl -1):
            self.layers += [Linear(layer_size_list[i], layer_size_list[i+1])]
            if i == self.nl - 2:
                self.activations += [output_activation]
            else:
                self.activations += [hidden_activations]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            activation = self.activations[i]
            if activation is not None:
                x = self.activations[i](x)


class ActionUnitClassifier(MLP):

    def __init__(self):
        layer_size_list = [20, 20]
        nl = len(layer_size_list)
        hidden_activation = [F.relu]*(nl-1)
        output_activation = F.binary_cross_entropy
        super().__init__(layer_size_list, hidden_activation, output_activation)


class ValenceArousalRegressor(MLP):

    def __init__(self):
        layer_size_list = [20, 20]
        nl = len(layer_size_list)
        hidden_activation = [F.relu]*(nl-1)
        output_activation = F.mse_loss
        super().__init__(layer_size_list, hidden_activation, output_activation)


class ExpressionClassifier(MLP):

    def __init__(self):
        layer_size_list = [20, 20]
        nl = len(layer_size_list)
        hidden_activation = [F.relu]*(nl-1)
        output_activation = F.cross_entropy
        super().__init__(layer_size_list, hidden_activation, output_activation)
