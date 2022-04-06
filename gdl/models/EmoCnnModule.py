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


import sys

import torch
import pytorch_lightning as pl
import numpy as np
from gdl.utils.other import class_from_str
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from gdl.datasets.AffectNetDataModule import AffectNetExpressions
from gdl.datasets.AffWild2Dataset import Expression7
from pathlib import Path
from gdl.utils.lightning_logging import _log_array_image, _log_wandb_image, _torch_image2np
from gdl.models.EmotionRecognitionModuleBase import EmotionRecognitionBaseModule
import torchvision.models.vgg as vgg # vgg19, vgg11, vgg13, vgg16, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from gdl.layers.losses.FRNet import resnet50, load_state_dict
from torch.nn import Linear
import pytorch_lightning.plugins.environments.lightning_environment as le


class EmoCnnModule(EmotionRecognitionBaseModule):
    """
    Emotion Recognitition module which uses a conv net as its backbone. Currently Resnet-50 and VGG are supported. 
    ResNet-50 based emotion recognition trained on AffectNet is the network used for self-supervising emotion in EMOCA.
    """
    def __init__(self, config):
        super().__init__(config)
        # self.n_expression = 9  # we use all affectnet classes (included none) for now
        self.n_expression = config.data.n_expression if 'n_expression' in config.data.keys() else 9

        self.num_outputs = 0
        if self.config.model.predict_expression:
            self.num_outputs += self.n_expression
            self.num_classes = self.n_expression

        if self.config.model.predict_valence:
            self.num_outputs += 1

        if self.config.model.predict_arousal:
            self.num_outputs += 1

        if 'predict_AUs' in self.config.model.keys() and self.config.model.predict_AUs:
            self.num_outputs += self.config.model.predict_AUs

        if config.model.backbone == "resnet50":
            self.backbone = resnet50(num_classes=8631, include_top=False)
            if config.model.load_pretrained:
                # checkpoint = '/ps/scratch/face2d3d/ringnetpp/eccv/data/resnet50_ft_weight.pkl'
                load_state_dict(self.backbone, config.model.pretrained_weights)
            self.last_feature_size = 2048
            self.linear = Linear(self.last_feature_size, self.num_outputs) # 2048 is the output of  the resnet50 backbone without the MLP "top"
        elif config.model.backbone[:3] == "vgg":
            vgg_constructor = getattr(vgg, config.model.backbone)
            self.backbone = vgg_constructor(pretrained=bool(config.model.load_pretrained), progress=True)
            self.last_feature_size = 1000
            self.linear = Linear(self.last_feature_size, self.num_outputs) #1000 is the number of imagenet classes so the dim of the output of the vgg backbone
        else:
            raise ValueError(f"Invalid backbone: '{self.config.model.backbone}'")

    def get_last_feature_size(self):
        return self.last_feature_size

    def _forward(self, images):
        output = self.backbone(images)
        emo_feat_2 = output
        output = self.linear(output.view(output.shape[0], -1))

        out_idx = 0
        if self.predicts_expression():
            expr_classification = output[:, out_idx:(out_idx + self.n_expression)]
            if self.exp_activation is not None:
                expr_classification = self.exp_activation(expr_classification, dim=1)
            out_idx += self.n_expression
        else:
            expr_classification = None

        if self.predicts_valence():
            valence = output[:, out_idx:(out_idx + 1)]
            if self.v_activation is not None:
                valence = self.v_activation(valence)
            out_idx += 1
        else:
            valence = None

        if self.predicts_arousal():
            arousal = output[:, out_idx:(out_idx + 1)]
            if self.a_activation is not None:
                arousal = self.a_activation(arousal)
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

        assert out_idx == output.shape[1]

        values = {}
        values["emo_feat_2"] = emo_feat_2
        values["valence"] = valence
        values["arousal"] = arousal
        values["expr_classification"] = expr_classification
        values["AUs"] = AUs
        return values


    def forward(self, batch):
        images = batch['image']

        if len(images.shape) == 5:
            K = images.shape[1]
        elif len(images.shape) == 4:
            K = 1
        else:
            raise RuntimeError("Invalid image batch dimensions.")

        # print("Batch size!")
        # print(images.shape)
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        emotion = self._forward(images)

        valence = emotion['valence']
        arousal = emotion['arousal']

        # emotion['expression'] = emotion['expression']

        # classes_probs = F.softmax(emotion['expression'])
        # expression = self.exp_activation(emotion['expr_classification'], dim=1)

        values = {}
        if self.predicts_valence():
            values['valence'] = valence.view(-1,1)
        if self.predicts_arousal():
            values['arousal'] = arousal.view(-1,1)
        # values['expr_classification'] = expression
        values['expr_classification'] = emotion['expr_classification']
        if self.predicts_AUs():
            values['AUs'] = emotion['AUs']

        # TODO: WARNING: HACK
        if 'n_expression' not in self.config.data:
            if self.n_expression == 8:
                raise NotImplementedError("This here should not be called")
                values['expr_classification'] = torch.cat([
                    values['expr_classification'], torch.zeros_like(values['expr_classification'][:, 0:1])
                                                   + 2*values['expr_classification'].min()],
                    dim=1)

        return values

    def _get_trainable_parameters(self):
        return list(self.backbone.parameters())

    ## we can leave the default implementation
    # def train(self, mode=True):
    #     pass

    def _vae_2_str(self, valence=None, arousal=None, affnet_expr=None, expr7=None, prefix=""):
        caption = ""
        if len(prefix) > 0:
            prefix += "_"
        if valence is not None and not np.isnan(valence).any():
            caption += prefix + "valence= %.03f\n" % valence
        if arousal is not None and not np.isnan(arousal).any():
            caption += prefix + "arousal= %.03f\n" % arousal
        if affnet_expr is not None and not np.isnan(affnet_expr).any():
            caption += prefix + "expression= %s \n" % AffectNetExpressions(affnet_expr).name
        if expr7 is not None and not np.isnan(expr7).any():
            caption += prefix +"expression= %s \n" % Expression7(expr7).name
        return caption

    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        return None
        batch_size = input_batch['image'].shape[0]

        visdict = {}
        if 'image' in input_batch.keys():
            visdict['inputs'] = input_batch['image']

        valence_pred = output_values["valence"]
        arousal_pred = output_values["arousal"]
        expr_classification_pred = output_values["expr_classification"]

        valence_gt = input_batch["va"][:, 0:1]
        arousal_gt = input_batch["va"][:, 1:2]
        expr_classification_gt = input_batch["affectnetexp"]

        # visdict = self.deca._create_visualizations_to_log("test", visdict, output_values, batch_idx,
        #                                                   indices=0, dataloader_idx=dataloader_idx)
        stage = "test"
        vis_dict = {}

        if self.trainer.is_global_zero:
            if isinstance(self.logger, WandbLogger):
                caption = self._vae_2_str(
                    valence=valence_pred.detach().cpu().numpy()[0],
                    arousal=arousal_pred.detach().cpu().numpy()[0],
                    affnet_expr=torch.argmax(expr_classification_pred, dim=1).detach().cpu().numpy().astype(np.int32)[0],
                    expr7=None, prefix="pred")
                caption += self._vae_2_str(
                    valence=valence_gt.cpu().numpy()[0],
                    arousal=arousal_gt.cpu().numpy()[0],
                    affnet_expr=expr_classification_gt.cpu().numpy().astype(np.int32)[0],
                    expr7=None, prefix="gt")

            i = 0 # index of sample in batch to log
            for key in visdict.keys():
                images = _torch_image2np(visdict[key])
                savepath = Path(
                    f'{self.config.inout.full_run_dir}/{stage}/{key}/{self.current_epoch:04d}_{batch_idx:04d}_{i:02d}.png')
                image = images[i]
                # im2log = Image(image, caption=caption)
                if isinstance(self.logger, WandbLogger):
                    im2log = _log_wandb_image(savepath, image, caption)
                elif self.logger is not None:
                    im2log = _log_array_image(savepath, image, caption)
                else:
                    im2log = _log_array_image(None, image, caption)
                name = stage + "_" + key
                if dataloader_idx is not None:
                    name += "/dataloader_idx_" + str(dataloader_idx)
                vis_dict[name] = im2log

            if isinstance(self.logger, WandbLogger):
                self.logger.log_metrics(vis_dict)
                # self.log_dict(vis_dict, sync_dist=True)
        return vis_dict