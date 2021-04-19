# THIS FILE HAS BEEN COPIED FROM THE DECA TRAINING REPOSITORY

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import models.ResNet as resnet


class ResnetEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model()  # out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        self.last_op = last_op

    def forward(self, inputs, output_features=False):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        if not output_features:
            return parameters
        return parameters, features


class SecondHeadResnet(nn.Module):

    def __init__(self, resnet : ResnetEncoder, outsize, last_op=None):
        super().__init__()
        self.resnet = resnet
        feature_size = 2048
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        if last_op == 'same':
            self.last_op = self.resnet.last_op
        else:
            self.last_op = last_op


    def forward(self, inputs):
        out1, features = self.resnet(inputs, output_features=True)
        out2 = self.layers(features)
        if self.last_op:
            out2 = self.last_op(out2)
        return out1, out2


    def train(self, mode: bool = True):
        #here we NEVER modify the eval/train status of the resnet backbone, only the FC layers of the second head
        self.layers.train(mode)
        return self


# class ResnetEncoder(nn.Module):
#     def __init__(self, append_layers = None):
#         super(ResnetEncoder, self).__init__()
#         # feature_size = 2048
#         self.feature_dim = 2048
#         self.encoder = resnet.load_ResNet50Model() #out: 2048
#         ### regressor
#         self.append_layers = append_layers
#         ## for normalize input images
#         MEAN = [0.485, 0.456, 0.406]
#         STD = [0.229, 0.224, 0.225]
#         self.register_buffer('MEAN', torch.tensor(MEAN)[None,:,None,None])
#         self.register_buffer('STD', torch.tensor(STD)[None,:,None,None])

#     def forward(self, inputs):
#         inputs = (inputs - self.MEAN)/self.STD
#         features = self.encoder(inputs)
#         if self.append_layers:
#             features = self.last_op(features)
#         return features

# class MLP(nn.Module):
#     def __init__(self, channels = [2048, 1024, 1], last_op = None):
#         super(MLP, self).__init__()
#         layers = []

#         for l in range(0, len(channels) - 1):
#             layers.append(nn.Linear(channels[l], channels[l+1]))
#             if l < len(channels) - 2:
#                 layers.append(nn.ReLU())
#         if last_op:
#             layers.append(last_op)

#         self.layers = nn.Sequential(*layers)

#     def forward(self, inputs):
#         outs = self.layers(inputs)
#         return outs
