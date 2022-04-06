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
import numpy as np
import torch.nn.functional as F
# from gdl_apps.EMOCA.train_deca_modular import get_checkpoint
from pytorch_lightning.loggers import WandbLogger
from gdl.layers.losses.EmonetLoader import get_emonet
from gdl.datasets.AffectNetDataModule import AffectNetExpressions
from gdl.datasets.AffWild2Dataset import Expression7
from pathlib import Path
from gdl.utils.lightning_logging import _log_array_image, _log_wandb_image, _torch_image2np
from gdl.models.EmotionRecognitionModuleBase import EmotionRecognitionBaseModule
import pytorch_lightning.plugins.environments.lightning_environment as le


class EmoNetModule(EmotionRecognitionBaseModule):
    """
    Emotion analysis using the EmoNet architecture. 
    https://github.com/face-analysis/emonet 
    """

    def __init__(self, config):
        super().__init__(config)
        self.emonet = get_emonet(load_pretrained=config.model.load_pretrained_emonet)
        if not config.model.load_pretrained_emonet:
            n_expression = config.data.n_expression if 'n_expression' in config.data.keys() else 9
            self.emonet.n_expression = n_expression # we use all affectnet classes (included none) for now
            self.n_expression = n_expression# we use all affectnet classes (included none) for now
            self.emonet._create_Emo() # reinitialize
        else:
            self.n_expression = 8
        self.num_classes = self.n_expression

        self.size = (256, 256) # predefined input image size

    def emonet_out(self, images, intermediate_features=False):
        images = F.interpolate(images, self.size, mode='bilinear')
        return self.emonet(images, intermediate_features=intermediate_features)

    def _forward(self, images):
        if len(images.shape) == 5:
            K = images.shape[1]
        elif len(images.shape) == 4:
            K = 1
        else:
            raise RuntimeError("Invalid image batch dimensions.")

        # print("Batch size!")
        # print(images.shape)
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        emotion = self.emonet_out(images, intermediate_features=True)

        valence = emotion['valence']
        arousal = emotion['arousal']

        # emotion['expression'] = emotion['expression']

        # classes_probs = F.softmax(emotion['expression'])
        if self.exp_activation is not None:
            expression = self.exp_activation(emotion['expression'], dim=1)

        values = {}
        values['valence'] = valence.view(-1,1)
        values['arousal'] = arousal.view(-1,1)
        values['expr_classification'] = expression

        # TODO: WARNING: HACK
        if 'n_expression' not in self.config.data:
            if self.n_expression == 8:
                values['expr_classification'] = torch.cat([
                    values['expr_classification'], torch.zeros_like(values['expr_classification'][:, 0:1])
                                                   + 2*values['expr_classification'].min()],
                    dim=1)

        return values

    def forward(self, batch):
        images = batch['image']
        return self._forward(images)


    def _get_trainable_parameters(self):
        return list(self.emonet.parameters())

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


        stage = "test"
        vis_dict = {}
        if self.trainer.is_global_zero:
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
            #     # self.log_dict(visdict, sync_dist=True)
        return vis_dict
