"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""


import torch
import pytorch_lightning as pl
import numpy as np
from gdl.utils.other import class_from_str
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from gdl.layers.losses.EmonetLoader import get_emonet
from gdl.utils.emotion_metrics import *
from torch.nn.functional import mse_loss, cross_entropy, nll_loss, l1_loss, log_softmax
import sys
import adabound
from gdl.datasets.EmotioNetDataModule import ActionUnitTypes
from torch.nn import BatchNorm1d, InstanceNorm1d
from .EmotionRecognitionModuleBase import EmotionRecognitionBaseModule
from gdl.models.MLP import MLP


class EmoMLP(EmotionRecognitionBaseModule):
    """
    EmoMLPmodule that was designed to predict emotion from precomputed 3DMM parameters of other 3D reconsturction methods 
    such as MGCNet or ExpNet. It takes the 3DMM parameters on the input and feeds it throu an MLP to predict emotion.
    """

    def __init__(self, config):
        super().__init__(config)

        in_size = 0
        if self.config.model.use_identity:
            in_size += self.config.model.identity_dim
        if self.config.model.use_expression:
            in_size +=  self.config.model.exp_dim
        # if self.config.model.use_global_pose:
        #     # in_size += 3
        #     in_size += 12 # 3x3 matrix + 3 translation

        if 'mlp_dimension_factor' in self.config.model.keys():
            dim_factor = self.config.model.mlp_dimension_factor
            dimension = in_size * dim_factor
        elif 'mlp_dim' in self.config.model.keys():
            dimension = self.config.model.mlp_dim
        else:
            dimension = in_size
        hidden_layer_sizes = config.model.num_mlp_layers * [dimension]

        out_size = 0
        if self.predicts_expression():
            self.num_classes = self.config.data.n_expression if 'n_expression' in self.config.data.keys() else 9
            out_size += self.num_classes
        if self.predicts_valence():
            out_size += 1
        if self.predicts_arousal():
            out_size += 1
        if self.predicts_AUs():
            out_size += self.predicts_AUs()

        if "use_mlp" not in self.config.model.keys() or self.config.model.use_mlp:
            if 'mlp_norm_layer' in self.config.model.keys():
                batch_norm = class_from_str(self.config.model.mlp_norm_layer, sys.modules[__name__])
            else:
                batch_norm = None
            self.mlp = MLP(in_size, out_size, hidden_layer_sizes, batch_norm=batch_norm)
        else:
            raise NotImplementedError("Other options are not supported. ")

    def forward(self, batch):
        shapecode = batch['shapecode']
        # texcode = values['texcode']
        expcode = batch['expcode']
        # posecode = values['posecode']
        values = {}
        if self.mlp is not None:
            input_list = []

            if self.config.model.use_identity:
                input_list += [shapecode]

            if self.config.model.use_expression:
                input_list += [expcode]

            # if self.config.model.use_global_pose:
            #     input_list += [global_pose]

            input = torch.cat(input_list, dim=1)
            output = self.mlp(input)

            out_idx = 0
            if self.predicts_expression():
                expr_classification = output[:, out_idx:(out_idx + self.num_classes)]
                if self.exp_activation is not None:
                    expr_classification = self.exp_activation(output[:, out_idx:(out_idx + self.num_classes)], dim=1)
                out_idx += self.num_classes
            else:
                expr_classification = None

            if self.predicts_valence():
                valence = output[:, out_idx:(out_idx+1)]
                if self.v_activation is not None:
                    valence = self.v_activation(valence)
                out_idx += 1
            else:
                valence = None

            if self.predicts_arousal():
                arousal = output[:, out_idx:(out_idx+1)]
                if self.a_activation is not None:
                    arousal = self.a_activation(output[:, out_idx:(out_idx + 1)])
                out_idx += 1
            else:
                arousal = None

            if self.predicts_AUs():
                num_AUs = self.config.model.predict_AUs
                AUs = output[:, out_idx:(out_idx + num_AUs)]
                if self.AU_activation is not None:
                    AUs = self.AU_activation(AUs)
                out_idx += num_AUs
            else:
                AUs = None

            values["valence"] = valence
            values["arousal"] = arousal
            values["expr_classification"] = expr_classification
            values["AUs"] = AUs
        return values

    def _get_trainable_parameters(self):
        return list(self.parameters())

    def _compute_loss(self,
                     pred, gt,
                     class_weight,
                     training=True,
                     **kwargs):

        if self.mlp is not None:
            losses_mlp, metrics_mlp = super()._compute_loss(pred, gt, class_weight, training, **kwargs)
        else:
            raise NotImplementedError("")
            # losses_mlp, metrics_mlp = {}, {}
        return losses_mlp, metrics_mlp

    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        pass
