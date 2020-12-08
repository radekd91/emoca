import os, sys
# from fastai.torch_core import Module
# from fastai.layers import ConvLayer, BatchNorm

from torch.nn import BatchNorm1d, Conv1d, Conv2d, ReLU, Module, Sequential, Linear, Tanh
import torch.nn.functional as F

#@delegates()


class SpeechEncoder(Module):

    def __init__(self, config):
        super().__init__()

        # self.bn = BatchNorm()

        num_feats = 5
        size_factor = 1
        speech_encoding_dim = 1

        self._speech_encoding_dim = config['expression_dim']
        self._condition_speech_features = config['condition_speech_features']
        self._speech_encoder_size_factor = config['speech_encoder_size_factor']
        self._num_training_styles = config['num_training_subjects']

        self.bn = BatchNorm1d(eps=1e-5, momentum=0.9, num_features=num_feats)
        # self.conv1_time = Conv1d(in_channels=1e-5, kernel_size=5, num_feat)
        self.conv1_time = Conv2d(kernel_size=(3, 1), stride=(2,1), in_channels=num_feats, out_channels=32*size_factor)
        self.conv2_time = Conv2d(kernel_size=(3, 1), stride=(2,1), in_channels=num_feats, out_channels=32*size_factor)
        self.conv3_time = Conv2d(kernel_size=(3, 1), stride=(2,1), in_channels=num_feats, out_channels=64*size_factor)
        self.conv4_time = Conv2d(kernel_size=(3, 1), stride=(2,1), in_channels=num_feats, out_channels=64*size_factor)

        self.fc1 = Linear(in_features= , out_features=128)
        self.fc2 = Linear(in_features=128, out_features=speech_encoding_dim)



    def forward(self, x, cond=None):
        x = F.relu(self.bn(x))

        if self._condition_speech_features:
            if cond is None:
                raise RuntimeError("Speech conditiong is set but no flag is passed in")



        x = F.relu(self.conv1_time(x))
        x = F.relu(self.conv2_time(x))
        x = F.relu(self.conv3_time(x))
        x = F.relu(self.conv4_time(x))

        x = x.view(-1, x.shape(1) * x.shape(2) * x.shape(3))

        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x





class Voca(Module):

    def __init__(self):
        super().__init__()

        self.con


    def forward(self):
        ConvLayer()


    pass