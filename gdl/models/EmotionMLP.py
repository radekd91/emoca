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


# from .EMOCA import DecaModule, instantiate_deca, DecaMode
from .EmotionRecognitionModuleBase import \
    EmotionRecognitionBaseModule, loss_from_cfg, _get_step_loss_weights, va_loss, v_or_a_loss, exp_loss
from .MLP import MLP
import torch
import pytorch_lightning as pl
import numpy as np
from gdl.utils.other import class_from_str
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from gdl.layers.losses.EmonetLoader import get_emonet
import sys

# important for class_from_str to work
from torch.nn.functional import mse_loss, cross_entropy, nll_loss, l1_loss, log_softmax
from torch.nn import BatchNorm1d, LayerNorm, InstanceNorm1d


def add_cfg_if_missing(cfg, name, default):
    if name not in cfg.keys():
        cfg[name] = default
    return cfg


class EmotionMLP(torch.nn.Module):

    def __init__(self, config, deca_cfg):
        super().__init__()
        self.config = config
        in_size = 0
        if self.config.use_identity:
            in_size += deca_cfg.n_shape
        if self.config.use_expression:
            in_size += deca_cfg.n_exp
        if self.config.use_global_pose:
            in_size += 3
        if self.config.use_jaw_pose:
            in_size += 3
        if self.config.use_detail_code:
            self.n_detail = deca_cfg.n_detail
            in_size += deca_cfg.n_detail
        else:
            self.n_detail = None
        if 'use_detail_emo_code' in self.config.keys() and self.config.use_detail_emo_code:
            self.n_detail_emo = deca_cfg.n_detail_emo
            in_size += deca_cfg.n_detail_emo
        else:
            self.n_detail_emo = None

        hidden_layer_sizes = config.num_mlp_layers * [in_size]

        out_size = 0
        if self.config.predict_expression:
            self.num_classes =  self.config.data.n_expression if 'n_expression' in self.config.data.keys() else 9
            out_size += self.num_classes
        if self.config.predict_valence:
            out_size += 1
        if self.config.predict_arousal:
            out_size += 1

        # if "use_mlp" not in self.config.keys() or self.config.use_mlp:
        if 'mlp_norm_layer' in self.config.keys():
            batch_norm = class_from_str(self.config.mlp_norm_layer, sys.modules[__name__])
        else:
            batch_norm = None
        self.mlp = MLP(in_size, out_size, hidden_layer_sizes, batch_norm=batch_norm)
        # else:
        #     self.mlp = None

        if 'v_activation' in config.keys():
            self.v_activation = class_from_str(self.config.v_activation, sys.modules[__name__])
        else:
            self.v_activation = None

        if 'a_activation' in config.keys():
            self.a_activation = class_from_str(self.config.a_activation, sys.modules[__name__])
        else:
            self.a_activation = None

        if 'exp_activation' in config.keys():
            self.exp_activation = class_from_str(self.config.exp_activation, sys.modules[__name__])
        else:
            self.exp_activation = F.log_softmax

        self.va_loss = loss_from_cfg(config, 'va_loss')
        self.v_loss = loss_from_cfg(config, 'v_loss')
        self.a_loss = loss_from_cfg(config, 'a_loss')
        self.exp_loss = loss_from_cfg(config, 'exp_loss')

        # config backwards compatibility
        self.config = add_cfg_if_missing(self.config, 'detach_shape', False)
        self.config = add_cfg_if_missing(self.config, 'detach_expression', False)
        self.config = add_cfg_if_missing(self.config, 'detach_detailcode', False)
        self.config = add_cfg_if_missing(self.config, 'detach_jaw', False)
        self.config = add_cfg_if_missing(self.config, 'detach_global_pose', False)


    def forward(self, values, result_prefix=""):
        shapecode = values['shapecode']

        if self.config.detach_shape:
            shapecode = shapecode.detach()

        # texcode = values['texcode']
        expcode = values['expcode']

        if self.config.detach_expression:
            expcode = expcode.detach()

        posecode = values['posecode']
        if self.config.use_detail_code:
            if 'detailcode' in values.keys() and values['detailcode'] is not None:
                detailcode = values['detailcode']
                if self.config.detach_detailcode:
                    detailcode = detailcode.detach()
            else:
                detailcode = torch.zeros((posecode.shape[0], self.n_detail), dtype=posecode.dtype, device=posecode.device )
        else:
            detailcode = None

        if 'use_detailemo_code' in self.config.keys() and self.config.use_detailemo_code:
            if 'detailemocode' in values.keys() and values['detailemocode'] is not None:
                detailemocode = values['detailemocode']
                if 'detach_detailemocode' in self.config.keys() and self.config.detach_detailemocode:
                    detailemocode = detailemocode.detach()
            else:
                detailemocode = torch.zeros((posecode.shape[0], self.n_detail_emo), dtype=posecode.dtype, device=posecode.device )
        else:
            detailemocode = None


        global_pose = posecode[:, :3]
        if self.config.detach_global_pose:
            global_pose = global_pose.detach()

        jaw_pose = posecode[:, 3:]
        if self.config.detach_jaw:
            jaw_pose = jaw_pose.detach()

        input_list = []

        if self.config.use_identity:
            input_list += [shapecode]

        if self.config.use_expression:
            input_list += [expcode]

        if self.config.use_global_pose:
            input_list += [global_pose]

        if self.config.use_jaw_pose:
            input_list += [jaw_pose]

        if self.config.use_detail_code:
            input_list += [detailcode]

        if 'use_detail_emo_code' in self.config.keys() and self.config.use_detail_emo_code:
            input_list += [detailemocode]

        input = torch.cat(input_list, dim=1)
        output = self.mlp(input)

        out_idx = 0
        if self.config.predict_expression:
            expr_classification = output[:, out_idx:(out_idx + self.num_classes)]
            if self.exp_activation is not None:
                expr_classification = self.exp_activation(output[:, out_idx:(out_idx + self.num_classes)], dim=1)
            out_idx += self.num_classes
        else:
            expr_classification = None

        if self.config.predict_valence:
            valence = output[:, out_idx:(out_idx+1)]
            if self.v_activation is not None:
                valence = self.v_activation(valence)
            out_idx += 1
        else:
            valence = None

        if self.config.predict_arousal:
            arousal = output[:, out_idx:(out_idx+1)]
            if self.a_activation is not None:
                arousal = self.a_activation(output[:, out_idx:(out_idx + 1)])
            out_idx += 1
        else:
            arousal = None

        values[result_prefix + "valence"] = valence
        values[result_prefix + "arousal"] = arousal
        values[result_prefix + "expr_classification"] = expr_classification

        return values


    def compute_loss(self, pred, batch, training, pred_prefix=""):
        valence_gt = pred["va"][:, 0:1]
        arousal_gt = pred["va"][:, 1:2]
        expr_classification_gt = pred["affectnetexp"]
        if "expression_weight" in pred.keys():
            class_weight = pred["expression_weight"][0]
        else:
            class_weight = None

        gt = {}
        gt["valence"] = valence_gt
        gt["arousal"] = arousal_gt
        gt["expr_classification"] = expr_classification_gt

        # TODO: this is not ugly enough
        scheme = None if 'va_loss_scheme' not in self.config.keys() else self.config.va_loss_scheme
        loss_term_weights = _get_step_loss_weights(self.v_loss, self.a_loss, self.va_loss, scheme, training)

        valence_sample_weight = batch["valence_sample_weight"] if "valence_sample_weight" in batch.keys() else None
        arousal_sample_weight = batch["arousal_sample_weight"] if "arousal_sample_weight" in batch.keys() else None
        va_sample_weight = batch["va_sample_weight"] if "va_sample_weight" in batch.keys() else None
        expression_sample_weight = batch["expression_sample_weight"] if "expression_sample_weight" in batch.keys() else None

        if 'continuous_va_balancing' in self.config.keys():
            if self.config.continuous_va_balancing == '1d':
                v_weight = valence_sample_weight
                a_weight = arousal_sample_weight
            elif self.config.continuous_va_balancing == '2d':
                v_weight = va_sample_weight
                a_weight = va_sample_weight
            elif self.config.continuous_va_balancing == 'expr':
                v_weight = expression_sample_weight
                a_weight = expression_sample_weight
            else:
                raise RuntimeError(f"Invalid continuous affect balancing"
                                   f" '{self.config.continuous_va_balancing}'")
            if len(v_weight.shape) > 1:
                v_weight = v_weight.view(-1)
            if len(a_weight.shape) > 1:
                a_weight = a_weight.view(-1)
        else:
            v_weight = None
            a_weight = None

        losses, metrics = {}, {}
        # print(metrics.keys())
        losses, metrics = v_or_a_loss(self.v_loss, pred, gt, loss_term_weights, metrics, losses, "valence",
                                      pred_prefix=pred_prefix, permit_dropping_corr=not training,
                                      sample_weights=v_weight)
        losses, metrics = v_or_a_loss(self.a_loss, pred, gt, loss_term_weights, metrics, losses, "arousal",
                                      pred_prefix=pred_prefix, permit_dropping_corr=not training,
                                      sample_weights=a_weight)
        losses, metrics = va_loss(self.va_loss, pred, gt, loss_term_weights, metrics, losses,
                                  pred_prefix=pred_prefix,  permit_dropping_corr=not training)
        losses, metrics = exp_loss(self.exp_loss, pred, gt, class_weight, metrics, losses,
                                   self.config.expression_balancing, self.num_classes, pred_prefix=pred_prefix, )

        return losses, metrics
