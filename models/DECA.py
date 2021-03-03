import os, sys
import torch
import torchvision
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from layers.losses.EmoNetLoss import EmoNetLoss
import numpy as np
# from time import time
from skimage.io import imread
import cv2
from pathlib import Path
from skimage.io import imsave

# add DECA's repo
# sys.path += [str(Path(__file__).parent.parent.parent.absolute() / 'DECA-training')]
# from lib.utils.renderer import SRenderY
# from lib.models.encoders import ResnetEncoder
# from lib.models.decoders import Generator
# from lib.models.FLAME import FLAME, FLAMETex
from .Renderer import SRenderY
from .DecaEncoder import ResnetEncoder
from .DecaDecoder import Generator
from .DecaFLAME import FLAME, FLAMETex

# from lib.utils import lossfunc, util
# from . import datasets
# from lib.datasets.datasets import VoxelDataset, TestData
# import lib.utils.util as util
# import lib.utils.lossfunc as lossfunc

import layers.losses.DecaLosses as lossfunc
import utils.DecaUtils as util
from wandb import Image
from datasets.FaceVideoDataset import AffectNetExpressions, Expression7, expr7_to_affect_net
torch.backends.cudnn.benchmark = True
from enum import Enum


class DecaMode(Enum):
    COARSE = 1
    DETAIL = 2


class DecaModule(LightningModule):

    def __init__(self, model_params, learning_params, inout_params, stage_name = ""):
        super().__init__()
        self.learning_params = learning_params
        self.inout_params = inout_params
        self.deca = DECA(config=model_params)
        self.mode = DecaMode[str(model_params.mode).upper()]
        self.stage_name = stage_name
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        if 'emonet_weight' in self.deca.config.keys():
            self.emonet_loss = EmoNetLoss(self.device)
        else:
            self.emonet_loss = None

    def reconfigure(self, model_params, inout_params, stage_name="", downgrade_ok=False, train=True):
        if (self.mode == DecaMode.DETAIL and model_params.mode != DecaMode.DETAIL) and not downgrade_ok:
            raise RuntimeError("You're switching the DECA mode from DETAIL to COARSE. Is this really what you want?!")
        self.inout_params = inout_params
        self.deca._reconfigure(model_params)
        self.stage_name = stage_name
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        self.mode = DecaMode[str(model_params.mode).upper()]
        self.train(mode=train)
        print(f"DECA MODE RECONFIGURED TO: {self.mode}")

    # should not be necessary now that the the buffers are registered
    # def _move_extra_params_to_correct_device(self):
    #     if self.deca.uv_face_eye_mask.device != self.device:
    #         self.deca.uv_face_eye_mask = self.deca.uv_face_eye_mask.to(self.device)
    #     if self.deca.fixed_uv_dis.device != self.device:
    #         self.deca.fixed_uv_dis = self.deca.fixed_uv_dis.to(self.device)
    #     if self.emonet_loss is not None:
    #         self.emonet_loss.to(device=self.device)

    def train(self, mode: bool = True):
        # super().train(mode) # not necessary
        if mode:
            if self.mode == DecaMode.COARSE:
                self.deca.E_flame.train()
                # print("Setting E_flame to train")
                self.deca.E_detail.eval()
                # print("Setting E_detail to eval")
                self.deca.D_detail.eval()
                # print("Setting D_detail to eval")
            elif self.mode == DecaMode.DETAIL:
                if self.deca.config.train_coarse:
                    # print("Setting E_flame to train")
                    self.deca.E_flame.train()
                else:
                    # print("Setting E_flame to eval")
                    self.deca.E_flame.eval()
                self.deca.E_detail.train()
                # print("Setting E_detail to train")
                self.deca.D_detail.train()
                # print("Setting D_detail to train")
        else:
            self.deca.E_flame.eval()
            # print("Setting E_flame to eval")
            self.deca.E_detail.eval()
            # print("Setting E_detail to eval")
            self.deca.D_detail.eval()
            # print("Setting D_detail to eval")

        # these are set to eval no matter what, they're never being trained
        if self.emonet_loss is not None:
            self.emonet_loss.eval()

        self.deca.flame.eval()
        self.deca.flametex.eval()

        self.deca.perceptual_loss.eval()
        self.deca.id_loss.eval()

        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # if 'device' in kwargs.keys():
        # self._move_extra_params_to_correct_device()
        return self

    def cuda(self, device=None):
        super().cuda(device)
        # self._move_extra_params_to_correct_device()
        return self

    def cpu(self):
        super().cpu()
        # self._move_extra_params_to_correct_device()
        return self

    # def forward(self, image):
    #     codedict = self.deca.encode(image)
    #     opdict, visdict = self.deca.decode(codedict)
    #     opdict = dict_tensor2npy(opdict)


    def _encode(self, batch, training=True) -> dict:
        codedict = {}
        original_batch_size = batch['image'].shape[0]

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
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

        if 'landmark' in batch.keys():
            lmk = batch['landmark']
            lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        if 'mask' in batch.keys():
            masks = batch['mask']
            masks = masks.view(-1, images.shape[-2], images.shape[-1])

        if 'va' in batch:
            va = batch['va']
            va = va.view(-1, va.shape[-1])
        else:
            va = None

        if 'expr7' in batch:
            expr7 = batch['expr7']
            expr7 = expr7.view(-1, expr7.shape[-1])
        else:
            expr7 = None

        if self.mode == DecaMode.COARSE or \
                (self.mode == DecaMode.DETAIL and self.deca.config.train_coarse):
            parameters = self.deca.E_flame(images)
        elif self.mode == DecaMode.DETAIL:
            with torch.no_grad():
                parameters = self.deca.E_flame(images)
        else:
            raise ValueError(f"Invalid DECA Mode {self.mode}")

        code_list = self.deca.decompose_code(parameters)
        shapecode, texcode, expcode, posecode, cam, lightcode = code_list

        # #TODO: figure out if we want to keep this code block:
        # if self.config.model.jaw_type == 'euler':
        #     # if use euler angle
        #     euler_jaw_pose = posecode[:, 3:].clone()  # x for yaw (open mouth), y for pitch (left ang right), z for roll
        #     # euler_jaw_pose[:,0] = 0.
        #     # euler_jaw_pose[:,1] = 0.
        #     # euler_jaw_pose[:,2] = 30.
        #     posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)

        if training:
            if self.mode == DecaMode.COARSE:
                ### shape constraints
                if self.deca.config.shape_constrain_type == 'same':
                    # reshape shapecode => [B, K, n_shape]
                    # shapecode_idK = shapecode.view(self.batch_size, self.deca.K, -1)
                    shapecode_idK = shapecode.view(original_batch_size, K, -1)
                    # get mean id
                    shapecode_mean = torch.mean(shapecode_idK, dim=[1])
                    # shapecode_new = shapecode_mean[:, None, :].repeat(1, self.deca.K, 1)
                    shapecode_new = shapecode_mean[:, None, :].repeat(1, K, 1)
                    shapecode = shapecode_new.view(-1, self.deca.config.model.n_shape)
                elif self.deca.config.shape_constrain_type == 'exchange':
                    '''
                    make sure s0, s1 is something to make shape close
                    the difference from ||so - s1|| is 
                    the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
                    '''
                    # new_order = np.array([np.random.permutation(self.deca.config.train_K) + i * self.deca.config.train_K for i in range(self.deca.config.batch_size_train)])
                    # new_order = np.array([np.random.permutation(self.deca.config.train_K) + i * self.deca.config.train_K for i in range(original_batch_size)])
                    new_order = np.array([np.random.permutation(K) + i * K for i in range(original_batch_size)])
                    new_order = new_order.flatten()
                    shapecode_new = shapecode[new_order]
                    # import ipdb; ipdb.set_trace()
                    ## append new shape code data
                    shapecode = torch.cat([shapecode, shapecode_new], dim=0)
                    texcode = torch.cat([texcode, texcode], dim=0)
                    expcode = torch.cat([expcode, expcode], dim=0)
                    posecode = torch.cat([posecode, posecode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## append gt
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    masks = torch.cat([masks, masks], dim=0)

                    if va is not None:
                        va = torch.cat([va, va], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7], dim=0)


        # -- detail
        if self.mode == DecaMode.DETAIL:
            detailcode = self.deca.E_detail(images)

            if training:
                if self.deca.config.detail_constrain_type == 'exchange':
                    '''
                    make sure s0, s1 is something to make shape close
                    the difference from ||so - s1|| is 
                    the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
                    '''
                    # new_order = np.array(
                    #     [np.random.permutation(self.deca.config.K) + i * self.deca.config.K for i in range(self.deca.config.effective_batch_size)])
                    new_order = np.array(
                        # [np.random.permutation(self.deca.config.train_K) + i * self.deca.config.train_K for i in range(original_batch_size)])
                        [np.random.permutation(K) + i * K for i in range(original_batch_size)])
                    new_order = new_order.flatten()
                    detailcode_new = detailcode[new_order]
                    # import ipdb; ipdb.set_trace()
                    detailcode = torch.cat([detailcode, detailcode_new], dim=0)
                    ## append new shape code data
                    shapecode = torch.cat([shapecode, shapecode], dim=0)
                    texcode = torch.cat([texcode, texcode], dim=0)
                    expcode = torch.cat([expcode, expcode], dim=0)
                    posecode = torch.cat([posecode, posecode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## append gt
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    masks = torch.cat([masks, masks], dim=0)

                    if va is not None:
                        va = torch.cat([va, va], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7], dim=0)


        codedict['shapecode'] = shapecode
        codedict['texcode'] = texcode
        codedict['expcode'] = expcode
        codedict['posecode'] = posecode
        codedict['cam'] = cam
        codedict['lightcode'] = lightcode
        if self.mode == DecaMode.DETAIL:
            codedict['detailcode'] = detailcode
        codedict['images'] = images
        if 'mask' in batch.keys():
            codedict['masks'] = masks
        if 'landmark' in batch.keys():
            codedict['lmk'] = lmk

        if 'va' in batch.keys():
            codedict['va'] = va
        if 'expr7' in batch.keys():
            codedict['expr7'] = expr7
        return codedict


    def _decode(self, codedict, training=True) -> dict:
        shapecode = codedict['shapecode']
        expcode = codedict['expcode']
        posecode = codedict['posecode']
        texcode = codedict['texcode']
        cam = codedict['cam']
        lightcode = codedict['lightcode']
        images = codedict['images']
        if 'masks' in codedict.keys():
            masks = codedict['masks']
        else:
            masks = None

        effective_batch_size = images.shape[0]  # this is the current batch size after all training augmentations modifications

        # FLAME - world space
        verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                          pose_params=posecode)
        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]

        albedo = self.deca.flametex(texcode)

        # ------ rendering
        ops = self.deca.render(verts, trans_verts, albedo, lightcode)
        # mask
        mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(effective_batch_size, -1, -1, -1),
                                      ops['grid'].detach(),
                                      align_corners=False)
        # images
        predicted_images = ops['images'] * mask_face_eye * ops['alpha_images']
        # predicted_images_no_mask = ops['images'] #* mask_face_eye * ops['alpha_images']
        if self.deca.config.useSeg and (masks is not None ):
            masks = masks[:, None, :, :]
        else:
            masks = mask_face_eye * ops['alpha_images']

        if self.deca.config.background_from_input:
            predicted_images = (1. - masks) * images + masks * predicted_images
        else:
            predicted_images = masks * predicted_images

        if self.mode == DecaMode.DETAIL:
            detailcode = codedict['detailcode']
            uv_z = self.deca.D_detail(torch.cat([posecode[:, 3:], expcode, detailcode], dim=1))
            # render detail
            uv_detail_normals, uv_coarse_vertices = self.deca.displacement2normal(uv_z, verts, ops['normals'])
            uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
            uv_texture = albedo.detach() * uv_shading
            predicted_detailed_image = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)
            if self.deca.config.background_from_input:
                predicted_detailed_image = (1. - masks) * images + masks*predicted_detailed_image
            else:
                predicted_detailed_image = masks * predicted_detailed_image

            # --- extract texture
            uv_pverts = self.deca.render.world2uv(trans_verts).detach()
            uv_gt = F.grid_sample(torch.cat([images, masks], dim=1), uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2],
                                  mode='bilinear')
            uv_texture_gt = uv_gt[:, :3, :, :].detach()
            uv_mask_gt = uv_gt[:, 3:, :, :].detach()
            # self-occlusion
            normals = util.vertex_normals(trans_verts, self.deca.render.faces.expand(effective_batch_size, -1, -1))
            uv_pnorm = self.deca.render.world2uv(normals)

            uv_mask = (uv_pnorm[:, -1, :, :] < -0.05).float().detach()
            uv_mask = uv_mask[:, None, :, :]
            ## combine masks
            uv_vis_mask = uv_mask_gt * uv_mask * self.deca.uv_face_eye_mask
        else:
            uv_detail_normals = None
            predicted_detailed_image = None

        # populate the value dict for metric computation/visualization
        codedict['predicted_images'] = predicted_images
        codedict['predicted_detailed_image'] = predicted_detailed_image
        codedict['verts'] = verts
        codedict['albedo'] = albedo
        codedict['mask_face_eye'] = mask_face_eye
        codedict['landmarks2d'] = landmarks2d
        codedict['landmarks3d'] = landmarks3d
        codedict['predicted_landmarks'] = predicted_landmarks
        codedict['trans_verts'] = trans_verts
        codedict['ops'] = ops
        codedict['masks'] = masks

        if self.mode == DecaMode.DETAIL:
            codedict['uv_texture_gt'] = uv_texture_gt
            codedict['uv_texture'] = uv_texture
            codedict['uv_detail_normals'] = uv_detail_normals
            codedict['uv_z'] = uv_z
            codedict['uv_shading'] = uv_shading
            codedict['uv_vis_mask'] = uv_vis_mask

        return codedict

    def _compute_emotion_loss(self, images, predicted_images, loss_dict, metric_dict, prefix, va=None, expr7=None):
        if self.deca.config.use_emonet_loss:
            d = loss_dict
            emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss = \
                self.emonet_loss.compute_loss(images, predicted_images)
        else:
            d = metric_dict
            # with torch.no_grad():
            emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss = \
                self.emonet_loss.compute_loss(images, predicted_images)

        def loss_or_metric(name, loss, is_loss):
            if not is_loss:
                metric_dict[name] = loss
            else:
                loss_dict[name] = loss

        # EmoNet self-consistency loss terms
        loss_or_metric(prefix + '_emonet_feat_1_L1', emo_feat_loss_1 * self.deca.config.emonet_weight,
                       self.deca.config.use_emonet_feat_1 and self.deca.config.use_emonet_loss)
        loss_or_metric(prefix + '_emonet_feat_2_L1', emo_feat_loss_2 * self.deca.config.emonet_weight,
                       self.deca.config.use_emonet_feat_2 and self.deca.config.use_emonet_loss)
        loss_or_metric(prefix + '_emonet_valence_L1', valence_loss * self.deca.config.emonet_weight,
                       self.deca.config.use_emonet_valence and self.deca.config.use_emonet_loss)
        loss_or_metric(prefix + '_emonet_arousal_L1', arousal_loss * self.deca.config.emonet_weight,
                       self.deca.config.use_emonet_arousal and self.deca.config.use_emonet_loss)
        # loss_or_metric(prefix + 'emonet_expression_KL', expression_loss * self.deca.config.emonet_weight) # KL seems to be causing NaN's
        loss_or_metric(prefix + '_emonet_expression_L1',expression_loss * self.deca.config.emonet_weight,
                       self.deca.config.use_emonet_expression and self.deca.config.use_emonet_loss)
        loss_or_metric(prefix + '_emonet_combined', (emo_feat_loss_1 + emo_feat_loss_2 + valence_loss + arousal_loss + expression_loss) * self.deca.config.emonet_weight,
                       self.deca.config.use_emonet_combined and self.deca.config.use_emonet_loss)

        # Log also the VA
        metric_dict[prefix + "_valence_input"] = self.emonet_loss.input_emotion['valence'].mean().detach()
        metric_dict[prefix + "_valence_output"] = self.emonet_loss.output_emotion['valence'].mean().detach()
        metric_dict[prefix + "_arousal_input"] = self.emonet_loss.input_emotion['arousal'].mean().detach()
        metric_dict[prefix + "_arousal_output"] = self.emonet_loss.output_emotion['arousal'].mean().detach()

        input_ex = self.emonet_loss.input_emotion['expression'].detach().cpu().numpy()
        input_ex = np.argmax(input_ex, axis=1).mean()
        output_ex = self.emonet_loss.output_emotion['expression'].detach().cpu().numpy()
        output_ex = np.argmax(output_ex, axis=1).mean()
        metric_dict[prefix + "_expression_input"] = torch.tensor(input_ex, device=self.device)
        metric_dict[prefix + "_expression_output"] = torch.tensor(output_ex, device=self.device)

        # GT emotion loss terms
        if self.deca.config.use_gt_emotion_loss:
            d = loss_dict
        else:
            d = metric_dict

        if va is not None:
            d[prefix + 'emo_sup_val_L1'] = F.l1_loss(self.emonet_loss.output_emotion['valence'], va[:, 0]) \
                                           * self.deca.config.gt_emotion_reg
            d[prefix + 'emo_sup_ar_L1'] = F.l1_loss(self.emonet_loss.output_emotion['arousal'], va[:, 1]) \
                                          * self.deca.config.gt_emotion_reg

            metric_dict[prefix + "_valence_gt"] = va[:, 0].mean().detach()
            metric_dict[prefix + "_arousal_gt"] = va[:, 1].mean().detach()

        if expr7 is not None:
            affectnet_gt = [expr7_to_affect_net(int(expr7[i])).value for i in range(len(expr7))]
            affectnet_gt = torch.tensor(np.array(affectnet_gt), device=self.device, dtype=torch.long)
            d[prefix + '_emo_sup_expr_CE'] = F.cross_entropy(self.emonet_loss.output_emotion['expression'], affectnet_gt) * self.deca.config.gt_emotion_reg
            metric_dict[prefix + "_expr_gt"] = affectnet_gt.mean().detach()



    def _compute_loss(self, codedict, training=True) -> (dict, dict):
        #### ----------------------- Losses
        losses = {}
        metrics = {}

        predicted_landmarks = codedict["predicted_landmarks"]
        if "lmk" in codedict.keys():
            lmk = codedict["lmk"]
        else:
            lmk = None

        if "masks" in codedict.keys():
            masks = codedict["masks"]
        else:
            masks = None

        predicted_images = codedict["predicted_images"]
        images = codedict["images"]
        lightcode = codedict["lightcode"]
        albedo = codedict["albedo"]
        mask_face_eye = codedict["mask_face_eye"]
        shapecode = codedict["shapecode"]
        expcode = codedict["expcode"]
        texcode = codedict["texcode"]
        ops = codedict["ops"]

        if 'va' in codedict:
            va = codedict['va']
            va = va.view(-1, va.shape[-1])
        else:
            va = None

        if 'expr7' in codedict:
            expr7 = codedict['expr7']
            expr7 = expr7.view(-1, expr7.shape[-1])
        else:
            expr7 = None

        if self.mode == DecaMode.DETAIL:
            uv_texture = codedict["uv_texture"]
            uv_texture_gt = codedict["uv_texture_gt"]


        ## COARSE loss only
        if self.mode == DecaMode.COARSE or (self.mode == DecaMode.DETAIL and self.deca.config.train_coarse):

            # landmark losses (only useful if coarse model is being trained
            if training or lmk is not None:
                if self.deca.config.use_landmarks:
                    d = losses
                else:
                    d = metrics

                if self.deca.config.useWlmk:
                    d['landmark'] = lossfunc.weighted_landmark_loss(predicted_landmarks,
                                                                              lmk) * self.deca.config.lmk_weight
                else:
                    d['landmark'] = lossfunc.landmark_loss(predicted_landmarks, lmk) * self.deca.config.lmk_weight
                # losses['eye_distance'] = lossfunc.eyed_loss(predicted_landmarks, lmk) * self.deca.config.lmk_weight * 2
                d['eye_distance'] = lossfunc.eyed_loss(predicted_landmarks, lmk) * self.deca.config.eyed
                d['lip_distance'] = lossfunc.eyed_loss(predicted_landmarks, lmk) * self.deca.config.lipd
                #TODO: fix this on the next iteration lipd_loss
                # d['lip_distance'] = lossfunc.lipd_loss(predicted_landmarks, lmk) * self.deca.config.lipd

            # photometric loss
            if training or masks is not None:
                if self.deca.config.use_photometric:
                    d = losses
                else:
                    d = metrics
                d['photometric_texture'] = (masks * (predicted_images - images).abs()).mean() * self.deca.config.photow

            if self.deca.config.idw > 1e-3:
                shading_images = self.deca.render.add_SHlight(ops['normal_images'], lightcode.detach())
                albedo_images = F.grid_sample(albedo.detach(), ops['grid'], align_corners=False)
                overlay = albedo_images * shading_images * mask_face_eye + images * (1 - mask_face_eye)
                losses['identity'] = self.deca.id_loss(overlay, images) * self.deca.config.idw

            losses['shape_reg'] = (torch.sum(shapecode ** 2) / 2) * self.deca.config.shape_reg
            losses['expression_reg'] = (torch.sum(expcode ** 2) / 2) * self.deca.config.exp_reg
            losses['tex_reg'] = (torch.sum(texcode ** 2) / 2) * self.deca.config.tex_reg
            losses['light_reg'] = ((torch.mean(lightcode, dim=2)[:, :,
                                    None] - lightcode) ** 2).mean() * self.deca.config.light_reg

            if self.emonet_loss is not None:
                # with torch.no_grad():
                self._compute_emotion_loss(images, predicted_images, losses, metrics, "coarse", va, expr7)
                codedict["coarse_valence_input"] = self.emonet_loss.input_emotion['valence']
                codedict["coarse_arousal_input"] = self.emonet_loss.input_emotion['arousal']
                codedict["coarse_expression_input"] = self.emonet_loss.input_emotion['expression']
                codedict["coarse_valence_output"] = self.emonet_loss.output_emotion['valence']
                codedict["coarse_arousal_output"] = self.emonet_loss.output_emotion['arousal']
                codedict["coarse_expression_output"] = self.emonet_loss.output_emotion['expression']

                if va is not None:
                    codedict["coarse_valence_gt"] = va[:, 0]
                    codedict["coarse_arousal_gt"] = va[:, 1]
                if expr7 is not None:
                    codedict["coarse_expression_gt"] = expr7


        ## DETAIL loss only
        if self.mode == DecaMode.DETAIL:
            predicted_detailed_image = codedict["predicted_detailed_image"]
            uv_z = codedict["uv_z"] # UV displacement map
            uv_shading = codedict["uv_shading"]
            uv_vis_mask = codedict["uv_vis_mask"] # uv_mask of what is visible

            photometric_detailed = (masks * (
                    predicted_detailed_image - images).abs()).mean() * self.deca.config.photow

            if self.deca.config.use_detailed_photo:
                losses['photometric_detailed_texture'] = photometric_detailed
            else:
                metrics['photometric_detailed_texture'] = photometric_detailed

            if self.emonet_loss is not None:
                self._compute_emotion_loss(images, predicted_detailed_image, losses, metrics, "detail")
                codedict["detail_valence_input"] = self.emonet_loss.input_emotion['valence']
                codedict["detail_arousal_input"] = self.emonet_loss.input_emotion['arousal']
                codedict["detail_expression_input"] = self.emonet_loss.input_emotion['expression']
                codedict["detail_valence_output"] = self.emonet_loss.output_emotion['valence']
                codedict["detail_arousal_output"] = self.emonet_loss.output_emotion['arousal']
                codedict["detail_expression_output"] = self.emonet_loss.output_emotion['expression']

                if va is not None:
                    codedict["detail_valence_gt"] = va[:,0]
                    codedict["detail_arousal_gt"] = va[:,1]
                if expr7 is not None:
                    codedict["detail_expression_gt"] = expr7

            for pi in range(3):  # self.deca.face_attr_mask.shape[0]):
                # if pi==0:
                new_size = 256
                # else:
                #     new_size = 128
                # if self.deca.config.uv_size != 256:
                #     new_size = 128
                uv_texture_patch = F.interpolate(
                    uv_texture[:, :, self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                    self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]],
                    [new_size, new_size], mode='bilinear')
                uv_texture_gt_patch = F.interpolate(
                    uv_texture_gt[:, :, self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                    self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]], [new_size, new_size],
                    mode='bilinear')
                uv_vis_mask_patch = F.interpolate(
                    uv_vis_mask[:, :, self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                    self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]],
                    [new_size, new_size], mode='bilinear')

                detail_l1 = (uv_texture_patch * uv_vis_mask_patch - uv_texture_gt_patch * uv_vis_mask_patch).abs().mean() * \
                                                    self.deca.config.sfsw[pi]
                if self.deca.config.use_detail_l1:
                    losses['detail_l1_{}'.format(pi)] = detail_l1
                else:
                    metrics['detail_l1_{}'.format(pi)] = detail_l1

                mrf = self.deca.perceptual_loss(uv_texture_patch * uv_vis_mask_patch,
                                                                               uv_texture_gt_patch * uv_vis_mask_patch) * \
                                                     self.deca.config.sfsw[pi] * self.deca.config.mrfwr
                if self.deca.config.use_detail_mrf:
                    losses['detail_mrf_{}'.format(pi)] = mrf
                else:
                    metrics['detail_mrf_{}'.format(pi)] = mrf

                # if pi == 2:
                #     uv_texture_gt_patch_ = uv_texture_gt_patch
                #     uv_texture_patch_ = uv_texture_patch
                #     uv_vis_mask_patch_ = uv_vis_mask_patch

            losses['z_reg'] = torch.mean(uv_z.abs()) * self.deca.config.zregw
            losses['z_diff'] = lossfunc.shading_smooth_loss(uv_shading) * self.deca.config.zdiffw
            nonvis_mask = (1 - util.binary_erosion(uv_vis_mask))
            losses['z_sym'] = (nonvis_mask * (uv_z - torch.flip(uv_z, [-1]).detach()).abs()).sum() * self.deca.config.zsymw

        # else:
        #     uv_texture_gt_patch_ = None
        #     uv_texture_patch_ = None
        #     uv_vis_mask_patch_ = None

        return losses, metrics

    def compute_loss(self, values, training=True) -> (dict, dict):
        losses, metrics = self._compute_loss(values, training=training)

        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        # losses['all_loss'] = all_loss
        losses = {'loss_' + key : value for key, value in losses.items()} # add prefix loss for better logging
        losses['loss'] = all_loss

        # add metrics that do not effect the loss function (if any)
        for key in metrics.keys():
            losses['metric_' + key] = metrics[key]
        return losses

    def _val_to_be_logged(self, d):
        if not hasattr(self, 'val_dict_list'):
            self.val_dict_list = []
        self.val_dict_list += [d]

    def _train_to_be_logged(self, d):
        if not hasattr(self, 'train_dict_list'):
            self.train_dict_list = []
        self.train_dict_list += [d]

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        with torch.no_grad():
            values = self._encode(batch, training=False)
            values = self._decode(values, training=False)
            losses_and_metrics = self.compute_loss(values, training=False)
        #### self.log_dict(losses_and_metrics, on_step=False, on_epoch=True)
        # prefix = str(self.mode.name).lower()
        prefix = self._get_logging_prefix()

        # if dataloader_idx is not None:
        #     dataloader_str = str(dataloader_idx) + "_"
        # else:
        dataloader_str = ''

        stage_str = dataloader_str + 'val_'

        # losses_and_metrics_to_log = {prefix + dataloader_str +'_val_' + key: value.detach().cpu() for key, value in losses_and_metrics.items()}
        losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach() for key, value in losses_and_metrics.items()}
        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = self.current_epoch
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        # log val_loss also without any prefix for a model checkpoint to track it
        losses_and_metrics_to_log[stage_str + 'loss'] = losses_and_metrics_to_log[prefix + '_' + stage_str + 'loss']

        losses_and_metrics_to_log[prefix + '_' + stage_str + 'step'] = self.global_step
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'batch_idx'] = batch_idx
        losses_and_metrics_to_log[stage_str + 'step'] = self.global_step
        losses_and_metrics_to_log[stage_str + 'batch_idx'] = batch_idx
        # self._val_to_be_logged(losses_and_metrics_to_log)
        if self.global_step % self.deca.config.val_vis_frequency == 0:
            uv_detail_normals = None
            if 'uv_detail_normals' in values.keys():
                uv_detail_normals = values['uv_detail_normals']
            visualizations, grid_image = self._visualization_checkpoint(values['verts'], values['trans_verts'], values['ops'],
                                           uv_detail_normals, values, batch_idx, stage_str[:-1], prefix)
            vis_dict = self._log_visualizations(stage_str[:-1], visualizations, values, batch_idx, indices=0, dataloader_idx=dataloader_idx)
            # image = Image(grid_image, caption="full visualization")
            # vis_dict[prefix + '_val_' + "visualization"] = image
            if isinstance(self.logger, WandbLogger):
                self.logger.log_metrics(vis_dict)
            # self.logger.experiment.log(vis_dict) #, step=self.global_step)

        self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch # recommended
        # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=False) # log per step
        # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True) # log per both
        # return losses_and_metrics
        return None

    def _get_logging_prefix(self):
        prefix = self.stage_name + str(self.mode.name).lower()
        return prefix

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = self._get_logging_prefix()
        losses_and_metrics_to_log = {}

        # if dataloader_idx is not None:
        #     dataloader_str = str(dataloader_idx) + "_"
        # else:
        dataloader_str = ''
        stage_str = dataloader_str + 'test_'

        with torch.no_grad():
            values = self._encode(batch, training=False)
            values = self._decode(values, training=False)
            if 'mask' in batch.keys():
                losses_and_metrics = self.compute_loss(values, training=False)
                # losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach().cpu() for key, value in losses_and_metrics.items()}
                losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach() for key, value in losses_and_metrics.items()}
            else:
                losses_and_metric = None

        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = self.current_epoch
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'step'] = torch.tensor(self.global_step, device=self.device)
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'batch_idx'] = torch.tensor(batch_idx, device=self.device)
        losses_and_metrics_to_log[stage_str + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        losses_and_metrics_to_log[stage_str + 'step'] = torch.tensor(self.global_step, device=self.device)
        losses_and_metrics_to_log[stage_str + 'batch_idx'] = torch.tensor(batch_idx, device=self.device)
        # if self.global_step % 200 == 0:
        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']

        if batch_idx % self.deca.config.test_vis_frequency == 0:
            visualizations, grid_image = self._visualization_checkpoint(values['verts'], values['trans_verts'], values['ops'],
                                           uv_detail_normals, values, self.global_step, stage_str[:-1], prefix)
            visdict = self._log_visualizations(stage_str[:-1], visualizations, values, batch_idx, indices=0, dataloader_idx=dataloader_idx)
            # image = Image(grid_image, caption="full visualization")
            # visdict[ prefix + '_' + stage_str + "visualization"] = image
            if isinstance(self.logger, WandbLogger):
                self.logger.log_metrics(visdict)#, step=self.global_step)
        if self.logger is not None:
            self.logger.log_metrics(losses_and_metrics_to_log)
        return None

    def training_step(self, batch, batch_idx): #, debug=True):
        values = self._encode(batch, training=True)
        values = self._decode(values, training=True)
        losses_and_metrics = self.compute_loss(values, training=True)

        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']

        # prefix = str(self.mode.name).lower()
        prefix = self._get_logging_prefix()
        # losses_and_metrics_to_log = {prefix + '_train_' + key: value.detach().cpu() for key, value in losses_and_metrics.items()}
        losses_and_metrics_to_log = {prefix + '_train_' + key: value.detach() for key, value in losses_and_metrics.items()}
        losses_and_metrics_to_log[prefix + '_train_' + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        losses_and_metrics_to_log[prefix + '_train_' + 'step'] = self.global_step
        losses_and_metrics_to_log['train_' + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        losses_and_metrics_to_log['train_' + 'step'] = self.global_step

        # log loss also without any prefix for a model checkpoint to track it
        losses_and_metrics_to_log['loss'] = losses_and_metrics_to_log[prefix + '_train_loss']

        if self.global_step % self.deca.config.train_vis_frequency == 0:
            visualizations, grid_image = self._visualization_checkpoint(values['verts'], values['trans_verts'], values['ops'],
                                           uv_detail_normals, values, batch_idx, "train", prefix)
            visdict = self._log_visualizations('train', visualizations, values, batch_idx, indices=0)
            # image = Image(grid_image, caption="full visualization")
            # visdict[prefix + '_test_' + "visualization"] = image
            if isinstance(self.logger, WandbLogger):
                self.logger.log_metrics(visdict)#, step=self.global_step)


        self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
        # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=False) # log per step
        # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True) # log per both
        # return losses_and_metrics
        return losses_and_metrics['loss']

    ### STEP ENDS ARE PROBABLY NOT NECESSARY BUT KEEP AN EYE ON THEM IF MULI-GPU TRAINING DOESN'T WORKs
    # def training_step_end(self, batch_parts):
    #     return self._step_end(batch_parts)
    #
    # def validation_step_end(self, batch_parts):
    #     return self._step_end(batch_parts)
    #
    # def _step_end(self, batch_parts):
    #     # gpu_0_prediction = batch_parts.pred[0]['pred']
    #     # gpu_1_prediction = batch_parts.pred[1]['pred']
    #     N = len(batch_parts)
    #     loss_dict = {}
    #     for key in batch_parts[0]:
    #         for i in range(N):
    #             if key not in loss_dict.keys():
    #                 loss_dict[key] = batch_parts[i]
    #             else:
    #                 loss_dict[key] = batch_parts[i]
    #         loss_dict[key] = loss_dict[key] / N
    #     return loss_dict


    def _torch_image2np(self, torch_image):
        image = torch_image.detach().cpu().numpy()
        if len(image.shape) == 4:
            image = image.transpose([0, 2, 3, 1])
        elif len(image.shape) == 3:
            image = image.transpose([1, 2, 0])
        return image


    def vae_2_str(self, valence=None, arousal=None, affnet_expr=None, expr7=None, prefix=""):
        caption = ""
        if len(prefix) > 0:
            prefix += "_"
        if valence is not None:
            caption += prefix + "valence= %.03f\n" % valence
        if arousal is not None:
            caption += prefix + "arousal= %.03f\n" % arousal
        if affnet_expr is not None:
            caption += prefix + "expression= %s \n" % AffectNetExpressions(affnet_expr).name
        if expr7 is not None:
            caption += prefix +"expression= %s \n" % Expression7(expr7).name
        return caption

    def _fix_image(self, image):
        if image.max() < 30.:
            image = image * 255.
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _log_wandb_image(self, path, image, caption=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = self._fix_image(image)
        imsave(path, image)
        if caption is not None:
            caption_file = Path(path).parent / (Path(path).stem + ".txt")
            with open(caption_file, "w") as f:
                f.write(caption)
        wandb_image = Image(str(path), caption=caption)
        return wandb_image

    def _log_array_image(self, path, image, caption=None):
        image = self._fix_image(image)
        if path is not None:
            imsave(path, image)
        return image

    def _log_visualizations(self, stage, visdict, values, step, indices=None, dataloader_idx=None):
        mode_ = str(self.mode.name).lower()
        prefix = self._get_logging_prefix()

        log_dict = {}
        for key in visdict.keys():
            images = self._torch_image2np(visdict[key])
            if images.dtype == np.float32 or images.dtype == np.float64 or images.dtype == np.float16:
                images = np.clip(images, 0, 1)
            if indices is None:
                indices = np.arange(images.shape[0])
            if isinstance(indices, int):
                indices = [indices,]
            if isinstance(indices, str) and indices == 'all':
                image = np.concatenate([images[i] for i in range(images.shape[0])], axis=1)
                savepath = Path(f'{self.inout_params.full_run_dir}/{prefix}_{stage}/{key}/{self.current_epoch:04d}_{step:04d}_all.png')
                # im2log = Image(image, caption=key)
                if isinstance(self.logger, WandbLogger):
                    im2log = self._log_wandb_image(savepath, image)
                else:
                    im2log = self._log_array_image(savepath, image)
                name = prefix + "_" + stage + "_" + key
                if dataloader_idx is not None:
                    name += "/dataloader_idx_" + str(dataloader_idx)
                log_dict[name] = im2log
            else:
                for i in indices:
                    caption = key + f" batch_index={step}\n"
                    caption += key + f" index_in_batch={i}\n"
                    if self.emonet_loss is not None:
                        if key == 'inputs':
                            if mode_ + "_valence_input" in values.keys():
                                caption += self.vae_2_str(values[mode_ + "_valence_input"][i].detach().cpu().item(),
                                                                 values[mode_ + "_arousal_input"][i].detach().cpu().item(),
                                                                 np.argmax(values[mode_ + "_expression_input"][i].detach().cpu().numpy()),
                                                                 prefix="emonet") + "\n"
                            if 'va' in values.keys():
                                caption += self.vae_2_str(
                                    values[mode_ + "_valence_gt"][i].detach().cpu().item(),
                                    values[mode_ + "_arousal_gt"][i].detach().cpu().item(),
                                    prefix="gt") + "\n"
                            if 'expr7' in values.keys():
                                caption += "\n" + self.vae_2_str(
                                    expr7=values[mode_ + "_expression_gt"][i].detach().cpu().numpy(),
                                    prefix="gt") + "\n"
                        elif key == 'output_images_' + mode_:
                            if mode_ + "_valence_output" in values.keys():
                                caption += self.vae_2_str(values[mode_ + "_valence_output"][i].detach().cpu().item(),
                                                                 values[mode_ + "_arousal_output"][i].detach().cpu().item(),
                                                                 np.argmax(values[mode_ + "_expression_output"][i].detach().cpu().numpy())) + "\n"
                        # elif key == 'output_images_detail':
                        #     caption += "\n" + self.vae_2_str(values["detail_output_valence"][i].detach().cpu().item(),
                        #                                  values["detail_output_valence"][i].detach().cpu().item(),
                        #                                  np.argmax(values["detail_output_expression"][
                        #                                                i].detach().cpu().numpy()))
                    savepath = Path(f'{self.inout_params.full_run_dir}/{prefix}_{stage}/{key}/{self.current_epoch:04d}_{step:04d}_{i:02d}.png')
                    image = images[i]
                    # im2log = Image(image, caption=caption)
                    if isinstance(self.logger, WandbLogger):
                        im2log = self._log_wandb_image(savepath, image, caption)
                    elif self.logger is not None:
                        im2log = self._log_array_image(savepath, image, caption)
                    else:
                        im2log = self._log_array_image(None, image, caption)
                    name = prefix + "_" + stage + "_" + key
                    if dataloader_idx is not None:
                        name += "/dataloader_idx_" + str(dataloader_idx)
                    log_dict[name] = im2log
        # self.log_dict(log_dict, on_step=on_step, on_epoch=on_epoch)
        # if on_step:
        #     step = self.global_step
        # if on_epoch:
        #     step = self.current_epoch
        # self.logger.experiment.log(log_dict, step=step)
        # self.logger.experiment.log(log_dict)#, step=step)
        return log_dict

    def _visualization_checkpoint(self, verts, trans_verts, ops, uv_detail_normals, additional, batch_idx, stage, prefix,
                                  save=True):
        batch_size = verts.shape[0]
        visind = np.arange(batch_size)
        shape_images = self.deca.render.render_shape(verts, trans_verts)
        if uv_detail_normals is not None:
            detail_normal_images = F.grid_sample(uv_detail_normals.detach(), ops['grid'].detach(),
                                                 align_corners=False)
            shape_detail_images = self.deca.render.render_shape(verts, trans_verts,
                                                           detail_normal_images=detail_normal_images)
        else:
            shape_detail_images = None

        visdict = {}
        if 'images' in additional.keys():
            visdict['inputs'] = additional['images'][visind]

        if 'images' in additional.keys() and 'lmk' in additional.keys():
            visdict['landmarks_gt'] = util.tensor_vis_landmarks(additional['images'][visind], additional['lmk'][visind])

        if 'images' in additional.keys() and 'predicted_landmarks' in additional.keys():
            visdict['landmarks_predicted'] = util.tensor_vis_landmarks(additional['images'][visind],
                                                                     additional['predicted_landmarks'][visind])

        if 'predicted_images' in additional.keys():
            visdict['output_images_coarse'] = additional['predicted_images'][visind]

        visdict['geometry_coarse'] = shape_images[visind]
        if shape_detail_images is not None:
            visdict['geometry_detail'] = shape_detail_images[visind]

        if 'albedo_images' in additional.keys():
            visdict['albedo_images'] = additional['albedo_images'][visind]

        if 'masks' in additional.keys():
            visdict['mask'] = additional['masks'].repeat(1, 3, 1, 1)[visind]
        if 'albedo' in additional.keys():
            visdict['albedo'] = additional['albedo'][visind]

        if 'predicted_detailed_image' in additional.keys() and additional['predicted_detailed_image'] is not None:
            visdict['output_images_detail'] = additional['predicted_detailed_image'][visind]

        if 'shape_detail_images' in additional.keys():
            visdict['shape_detail_images'] = additional['shape_detail_images'][visind]

        if 'uv_detail_normals' in additional.keys():
            visdict['uv_detail_normals'] = additional['uv_detail_normals'][visind] * 0.5 + 0.5

        if 'uv_texture_patch' in additional.keys():
            visdict['uv_texture_patch'] = additional['uv_texture_patch'][visind]

        if 'uv_texture_gt' in additional.keys():
            visdict['uv_texture_gt'] = additional['uv_texture_gt'][visind]

        if 'uv_vis_mask_patch' in additional.keys():
            visdict['uv_vis_mask_patch'] = additional['uv_vis_mask_patch'][visind]

        if save:
            savepath = f'{self.inout_params.full_run_dir}/{prefix}_{stage}/combined/{self.current_epoch:04d}_{batch_idx:04d}.png'
            Path(savepath).parent.mkdir(exist_ok=True, parents=True)
            visualization_image = self.deca.visualize(visdict, savepath)
            return visdict, visualization_image[..., [2, 1, 0]]
        else:
            visualization_image = None
            return visdict, None


    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print("Configuring optimizer")
        trainable_params = []
        if self.mode == DecaMode.COARSE:
            trainable_params += list(self.deca.E_flame.parameters())
            print("Add E_flame.parameters() to the optimizer")
        elif self.mode == DecaMode.DETAIL:
            if self.deca.config.train_coarse:
                trainable_params += list(self.deca.E_flame.parameters())
                print("Add E_flame.parameters() to the optimizer")
            trainable_params += list(self.deca.E_detail.parameters())
            print("Add E_detail.parameters() to the optimizer")
            trainable_params += list(self.deca.D_detail.parameters())
            print("Add D_detail.parameters() to the optimizer")
        else:
            raise ValueError(f"Invalid deca mode: {self.mode}")

        if self.learning_params.optimizer == 'Adam':
            self.deca.opt = torch.optim.Adam(
                trainable_params,
                lr=self.learning_params.learning_rate,
                amsgrad=False)

        elif self.learning_params.optimizer == 'SGD':
            self.deca.opt = torch.optim.SGD(
                trainable_params,
                lr=self.learning_params.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: '{self.learning_params.optimizer}'")

        optimizers = [self.deca.opt]
        schedulers = []
        if 'learning_rate_decay' in self.learning_params.keys():
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.deca.opt, gamma=self.learning_params.learning_rate_decay)
            schedulers += [scheduler]
        if len(schedulers) == 0:
            return self.deca.opt

        return optimizers, schedulers


class DECA(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self._reconfigure(config)
        self._reinitialize()

    def _reconfigure(self, config):
        self.config = config
        self.n_param = config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        self.n_detail = config.n_detail
        self.n_cond = 3 + config.n_exp

    def _reinitialize(self):
        self._create_model()
        self._setup_renderer()

        self.perceptual_loss = lossfunc.IDMRFLoss().eval()
        self.id_loss = lossfunc.VGGFace2Loss(self.config.pretrained_vgg_face_path).eval()
        self.face_attr_mask = util.load_local_mask(image_size=self.config.uv_size, mode='bbx')

    def _setup_renderer(self):
        self.render = SRenderY(self.config.image_size, obj_filename=self.config.topology_path,
                               uv_size=self.config.uv_size)  # .to(self.device)
        # face mask for rendering details
        mask = imread(self.config.face_mask_path).astype(np.float32) / 255.;
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        mask = imread(self.config.face_eye_mask_path).astype(np.float32) / 255.;
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        # self.uv_face_eye_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        uv_face_eye_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)

        ## displacement correct
        if os.path.isfile(self.config.fixed_displacement_path):
            fixed_dis = np.load(self.config.fixed_displacement_path)
            # self.fixed_uv_dis = torch.tensor(fixed_dis).float()
            fixed_uv_dis = torch.tensor(fixed_dis).float()
        else:
            fixed_uv_dis = torch.zeros([512, 512]).float()
        self.register_buffer('fixed_uv_dis', fixed_uv_dis)

    def _create_model(self):
        # coarse shape
        self.E_flame = ResnetEncoder(outsize=self.n_param)
        self.flame = FLAME(self.config)
        self.flametex = FLAMETex(self.config)
        # detail modeling
        self.E_detail = ResnetEncoder(outsize=self.n_detail)
        self.D_detail = Generator(latent_dim=self.n_detail + self.n_cond, out_channels=1, out_scale=0.01,
                                  sample_mode='bilinear')

        if self.config.resume_training:
            model_path = self.config.pretrained_modelpath
            print('trained model found. load {}'.format(model_path))
            checkpoint = torch.load(model_path)
            # model
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            # util.copy_state_dict(self.opt.state_dict(), checkpoint['opt']) # deprecate
            # detail model
            if 'E_detail' in checkpoint.keys():
                util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
                util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
            # training state
            self.start_epoch = 0  # checkpoint['epoch']
            self.start_iter = 0  # checkpoint['iter']
        else:
            print('Start training from scratch')
            self.start_epoch = 0
            self.start_iter = 0

    def decompose_code(self, code):
        '''
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        '''
        code_list = []
        num_list = [self.config.n_shape, self.config.n_tex, self.config.n_exp, self.config.n_pose, self.config.n_cam,
                    self.config.n_light]
        start = 0
        for i in range(len(num_list)):
            code_list.append(code[:, start:start + num_list[i]])
            start = start + num_list[i]
        # shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        return code_list

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()

        uv_z = uv_z * self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z * uv_coarse_normals + self.fixed_uv_dis[None, None, :,
                                                                             :] * uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        # uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1-self.uv_face_eye_mask)
        # uv_detail_normals = util.gaussian_blur(uv_detail_normals)
        return uv_detail_normals, uv_coarse_vertices

    def visualize(self, visdict, savepath):
        grids = {}
        for key in visdict:
            # print(key)
            if visdict[key] is None:
                continue
            grids[key] = torchvision.utils.make_grid(
                F.interpolate(visdict[key], [self.config.image_size, self.config.image_size])).detach().cpu()
        grid = torch.cat(list(grids.values()), 1)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        cv2.imwrite(savepath, grid_image)
        return grid_image
