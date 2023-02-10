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


from .DECA import DecaModule, instantiate_deca, DecaMode
from .EmotionRecognitionModuleBase import EmotionRecognitionBaseModule, loss_from_cfg
from .MLP import MLP
import torch
from torch.nn import BatchNorm1d, LayerNorm, InstanceNorm1d
import pytorch_lightning as pl
import numpy as np
from gdl.utils.other import class_from_str
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from gdl.layers.losses.EmonetLoader import get_emonet
import sys
import pytorch_lightning.plugins.environments.lightning_environment as le


class EmoDECA(EmotionRecognitionBaseModule):
    """
    EmoDECA loads a pretrained DECA-based face reconstruction net and uses it to predict emotion
    """

    def __init__(self, config):
        super().__init__(config)
        # model_params = config.model.deca_cfg.model
        # learning_params = config.model.deca_cfg.learning
        # inout_params = config.model.deca_cfg.inout
        deca_checkpoint = config.model.deca_checkpoint
        deca_stage = config.model.deca_stage
        config.model.deca_cfg.model.background_from_input = False
        deca_checkpoint_kwargs = {
            "model_params": config.model.deca_cfg.model,
            "learning_params": config.model.deca_cfg.learning,
            "inout_params": config.model.deca_cfg.inout,
            "stage_name": "testing",
        }
        # instantiate the face net
        if bool(deca_checkpoint):
            self.deca = instantiate_deca(config.model.deca_cfg, deca_stage , "test", deca_checkpoint, deca_checkpoint_kwargs)
            self.deca.inout_params.full_run_dir = config.inout.full_run_dir
            self._setup_deca(False)
        else: 
            self.deca = None

        # which latent codes are being used
        in_size = 0
        if self.config.model.use_identity:
            in_size += config.model.deca_cfg.model.n_shape
        if self.config.model.use_expression:
            in_size += config.model.deca_cfg.model.n_exp
        if self.config.model.use_global_pose:
            in_size += 3
        if self.config.model.use_jaw_pose:
            in_size += 3
        if self.config.model.use_detail_code:
            in_size += config.model.deca_cfg.model.n_detail
            
        if 'use_detail_emo_code' in self.config.model.keys() and self.config.model.use_detail_emo_code:
            # deprecated
            in_size += config.model.deca_cfg.model.n_detail_emo

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
            # self.num_classes = 9
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
            self.mlp = None

        if "use_emonet" in self.config.model.keys() and self.config.model.use_emonet:
            self.emonet = get_emonet(load_pretrained=config.model.load_pretrained_emonet)
            if not config.model.load_pretrained_emonet:
                self.emonet.n_expression = self.num_classes  # we use all affectnet classes (included none) for now
                self.emonet._create_Emo()  # reinitialize
        else:
            self.emonet = None

    def _get_trainable_parameters(self):
        trainable_params = []
        if self.config.model.finetune_deca:
            trainable_params += self.deca._get_trainable_parameters()
        if self.mlp is not None:
            trainable_params += list(self.mlp.parameters())
        if self.emonet is not None:
            trainable_params += list(self.emonet.parameters())
        return trainable_params

    def _setup_deca(self, train : bool):
        if self.config.model.finetune_deca:
            self.deca.train(train)
            self.deca.requires_grad_(True)
        else:
            self.deca.train(False)
            self.deca.requires_grad_(False)

    def train(self, mode=True):
        self._setup_deca(mode)
        if self.mlp is not None:
            self.mlp.train(mode)
        if self.emonet is not None:
            self.emonet.train(mode)

    def emonet_out(self, images):
        images = F.interpolate(images, (256, 256), mode='bilinear')
        return self.emonet(images, intermediate_features=False)

    def forward_emonet(self, values, values_decoded, mode):
        if mode == 'detail':
            image_name = 'predicted_detailed_image'
        elif mode == 'coarse':
            image_name = 'predicted_images'
        else:
            raise ValueError(f"Invalid image mode '{mode}'")

        emotion = self.emonet_out(values_decoded[image_name])
        if self.v_activation is not None:
            emotion['valence'] = self.v_activation(emotion['valence'])
        if self.a_activation is not None:
            emotion['arousal'] = self.a_activation(emotion['arousal'])
        if self.exp_activation is not None:
            emotion['expression'] = self.exp_activation(emotion['expression'])
        values[f"emonet_{mode}_valence"] = emotion['valence'].view(-1,1)
        values[f"emonet_{mode}_arousal"] = emotion['arousal'].view(-1,1)
        values[f"emonet_{mode}_expr_classification"] = emotion['expression']
        return values

    def forward(self, batch):
        values = self.deca.encode(batch, training=False)
        shapecode = values['shapecode']
        # texcode = values['texcode']
        expcode = values['expcode']
        posecode = values['posecode']
        if self.config.model.use_detail_code:
            assert self.deca.mode == DecaMode.DETAIL
            detailcode = values['detailcode']
            detailemocode = values['detailemocode']
        else:
            detailcode = None
            detailemocode = None

        global_pose = posecode[:, :3]
        jaw_pose = posecode[:, 3:]

        if self.mlp is not None:
            input_list = []

            if self.config.model.use_identity:
                input_list += [shapecode]

            if self.config.model.use_expression:
                input_list += [expcode]

            if self.config.model.use_global_pose:
                input_list += [global_pose]

            if self.config.model.use_jaw_pose:
                input_list += [jaw_pose]

            if self.config.model.use_detail_code:
                input_list += [detailcode]

            if 'use_detail_emo_code' in self.config.model.keys() and self.config.model.use_detail_emo_code:
                input_list += [detailemocode]

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

        if self.emonet is not None:
            # deprecated
            ## NULLIFY VALUES
            values2decode = { **values }
            if self.config.model.unpose_global_emonet:
                values2decode["posecode"] = torch.cat([torch.zeros_like(global_pose), jaw_pose], dim=1)

            if self.config.model.static_light:
                lightcode = torch.zeros_like(values["lightcode"][0:1])
                # lightcode[0, 0, :] = 3.5
                lightcode[0, 0, :] = self.config.model.static_light
                lightcode = lightcode.repeat(values["lightcode"].shape[0], 1, 1)
                values2decode["lightcode"] = lightcode

            if self.config.model.static_cam_emonet:
                values2decode["cam"] = torch.tensor([[self.config.model.static_cam_emonet[0],
                                                      self.config.model.static_cam_emonet[1],
                                                      self.config.model.static_cam_emonet[2]]],
                                                    dtype=values["cam"].dtype,
                                                    device=values["cam"].device).\
                    repeat(values["cam"].shape[0], 1)
            # values2decode["cam"] = None # TODO: set a meaningful camera
            values_decoded = self.deca.decode(values2decode)

            if self.config.model.use_coarse_image_emonet:
                # deprecated
                values = self.forward_emonet(values, values_decoded, 'coarse')

            if self.config.model.use_detail_image_emonet:
                # deprecated
                values = self.forward_emonet(values, values_decoded, 'detail')

        # emotion['expression'] = emotion['expression']

        # classes_probs = F.softmax(emotion['expression'])
        # expression = self.exp_activation(emotion['expression'], dim=1)

        return values

    def _compute_loss(self,
                     pred, gt,
                     class_weight,
                     training=True,
                     **kwargs):
        if self.mlp is not None:
            losses_mlp, metrics_mlp = super()._compute_loss(pred, gt, class_weight, training, **kwargs)
        else:
            losses_mlp, metrics_mlp = {}, {}

        if self.emonet is not None:
            if self.config.model.use_coarse_image_emonet:
                losses_emonet_c, metrics_emonet_c = super()._compute_loss(pred, gt, class_weight, training,
                                                                      pred_prefix="emonet_coarse_", **kwargs)
            else:
                losses_emonet_c, metrics_emonet_c = {}, {}

            if self.config.model.use_detail_image_emonet:
                losses_emonet_d, metrics_emonet_d = super()._compute_loss(pred, gt, class_weight, training,
                                                                      pred_prefix="emonet_detail_", **kwargs)
            else:
                losses_emonet_d, metrics_emonet_d = {}, {}
            losses_emonet = {**losses_emonet_c, **losses_emonet_d}
            metrics_emonet = {**metrics_emonet_c, **metrics_emonet_d}
        else:
            losses_emonet, metrics_emonet = {}, {}

        losses = {**losses_emonet, **losses_mlp}
        metrics = {**metrics_emonet, **metrics_mlp}

        return losses, metrics


    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        return None
        valence_pred = output_values["valence"]
        arousal_pred = output_values["arousal"]
        expr_classification_pred = output_values["expr_classification"]

        valence_gt = input_batch["va"][:, 0:1]
        arousal_gt = input_batch["va"][:, 1:2]
        expr_classification_gt = input_batch["affectnetexp"]

        with torch.no_grad():
            values = self.deca.decode(output_values)

        # self.deca.logger = self.logger # old version of PL, now logger is a property and retreived logger from the trainer
        self.deca.trainer = self.trainer # new version of PL

        mode_ = str(self.deca.mode.name).lower()

        if "uv_detail_normals" in values.keys():
            uv_detail_normals = values["uv_detail_normals"]
        else:
            uv_detail_normals = None

        values[f"{mode_}_valence_gt"] = valence_gt
        values[f"{mode_}_arousal_gt"] = arousal_gt
        values[f"{mode_}_expression_gt"] = expr_classification_gt
        values["affectnetexp"] = expr_classification_gt


        visualizations, grid_image = self.deca._visualization_checkpoint(values['verts'], values['trans_verts'],
                                                                         values['ops'],
                                                                         uv_detail_normals, values, self.global_step,
                                                                         "test", "")

        visdict = {}
        if self.trainer.is_global_zero:
            indices = 0
            visdict = self.deca._create_visualizations_to_log("test", visualizations, values, batch_idx,
                                                              indices=indices, dataloader_idx=dataloader_idx)
            if f"{mode_}_test_landmarks_gt" in visdict.keys():
                del visdict[f"{mode_}_test_landmarks_gt"]
            if f"{mode_}_test_landmarks_predicted" in visdict.keys():
                del visdict[f"{mode_}_test_landmarks_predicted"]
            if f"{mode_}_test_mask" in visdict.keys():
                del visdict[f"{mode_}_test_mask"]
            if f"{mode_}_test_albedo" in visdict.keys():
                del visdict[f"{mode_}_test_albedo"]
            if f"{mode_}_test_mask" in visdict.keys():
                del visdict[f"{mode_}_test_mask"]
            if f"{mode_}_test_uv_detail_normals" in visdict.keys():
                del visdict[f"{mode_}_test_uv_detail_normals"]
            if f"{mode_}_test_uv_texture_gt" in visdict.keys():
                del visdict[f"{mode_}_test_uv_texture_gt"]

            if isinstance(self.logger, WandbLogger):
                caption = self.deca.vae_2_str(
                    valence=valence_pred.detach().cpu().numpy()[indices, ...],
                    arousal=arousal_pred.detach().cpu().numpy()[indices, ...],
                    affnet_expr=torch.argmax(expr_classification_pred, dim=1).detach().cpu().numpy().astype(np.int32)[indices, ...],
                    expr7=None, prefix="pred")
                if f"{mode_}_test_geometry_coarse" in visdict.keys():
                    visdict[f"{mode_}_test_geometry_coarse"]._caption += caption
                if f"{mode_}_test_geometry_detail" in visdict.keys():
                    visdict[f"{mode_}_test_geometry_detail"]._caption += caption

            if isinstance(self.logger, WandbLogger):
            #     # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            #     env = le.LightningEnvironment()
            #     if env.global_rank() == 0:
                self.logger.log_metrics(visdict)
                # self.log_dict(visdict, sync_dist=True)
        return visdict

    # def test_step(self, batch, batch_idx, dataloader_idx=None):
    #     values = self.forward(batch)
    #     valence_pred = values["valence"]
    #     arousal_pred = values["arousal"]
    #     expr_classification_pred = values["expr_classification"]
    #     if "expression_weight" in batch.keys():
    #         class_weight = batch["expression_weight"][0]
    #     else:
    #         class_weight = None
    #
    #     if "va" in batch.keys():
    #         valence_gt = batch["va"][:, 0:1]
    #         arousal_gt = batch["va"][:, 1:2]
    #         expr_classification_gt = batch["affectnetexp"]
    #         losses, metrics = self.compute_loss(valence_pred, arousal_pred, expr_classification_pred,
    #                                             valence_gt, arousal_gt, expr_classification_gt, class_weight)
    #
    #
    #     if self.config.learning.test_vis_frequency > 0:
    #         if batch_idx % self.config.learning.test_vis_frequency == 0:
    #             self._test_visalization(values, batch, batch_idx, dataloader_idx=dataloader_idx)
    #
    #         self._log_losses_and_metrics(losses, metrics, "test")
    #         self.logger.log_metrics({f"test_step": batch_idx})


    # def _log_losses_and_metrics(self, losses, metrics, stage):
    #     if stage in ["train", "val"]:
    #         on_epoch = True
    #         on_step = False
    #         self.log_dict({f"{stage}_loss_" + key: value for key, value in losses.items()}, on_epoch=on_epoch,
    #                       on_step=on_step)
    #         self.log_dict({f"{stage}_metric_" + key: value for key, value in metrics.items()}, on_epoch=on_epoch,
    #                       on_step=on_step)
    #     else:
    #         on_epoch = False
    #         on_step = True
    #
    #         self.logger.log_metrics({f"{stage}_loss_" + key: value for key,value in losses.items()})#, on_epoch=on_epoch, on_step=on_step)
    #         #
    #         self.logger.log_metrics({f"{stage}_metric_" + key: value for key,value in metrics.items()})#, on_epoch=on_epoch, on_step=on_step)
    #

