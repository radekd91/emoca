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

Parts of the code were adapted from the original DECA release: 
https://github.com/YadiraF/DECA/ 
"""


import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as F_v
import adabound
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from gdl.layers.losses.EmoNetLoss import EmoNetLoss, create_emo_loss, create_au_loss
import numpy as np
# from time import time
from skimage.io import imread
from skimage.transform import resize
import cv2
from pathlib import Path

from gdl.models.Renderer import SRenderY
from gdl.models.DecaEncoder import ResnetEncoder, SecondHeadResnet, SwinEncoder
from gdl.models.DecaDecoder import Generator, GeneratorAdaIn
from gdl.models.DecaFLAME import FLAME, FLAMETex, FLAME_mediapipe
from gdl.models.EmotionMLP import EmotionMLP

import gdl.layers.losses.DecaLosses as lossfunc
import gdl.layers.losses.MediaPipeLandmarkLosses as lossfunc_mp
import gdl.utils.DecaUtils as util
from gdl.datasets.AffWild2Dataset import Expression7
from gdl.datasets.AffectNetDataModule import AffectNetExpressions
from gdl.utils.lightning_logging import _log_array_image, _log_wandb_image, _torch_image2np

torch.backends.cudnn.benchmark = True
from enum import Enum
from gdl.utils.other import class_from_str, get_path_to_assets
from gdl.layers.losses.VGGLoss import VGG19Loss
from omegaconf import OmegaConf, open_dict

import pytorch_lightning.plugins.environments.lightning_environment as le


class DecaMode(Enum):
    COARSE = 1 # when switched on, only coarse part of DECA-based networks is used
    DETAIL = 2 # when switched on, only coarse and detail part of DECA-based networks is used 


class DecaModule(LightningModule):
    """
    DecaModule is a PL module that implements DECA-inspired face reconstruction networks. 
    """

    def __init__(self, model_params, learning_params, inout_params, stage_name = ""):
        """
        :param model_params: a DictConfig of parameters about the model itself
        :param learning_params: a DictConfig of parameters corresponding to the learning process (such as optimizer, lr and others)
        :param inout_params: a DictConfig of parameters about input and output (where checkpoints and visualizations are saved)
        """
        super().__init__()
        self.learning_params = learning_params
        self.inout_params = inout_params

        # detail conditioning - what is given as the conditioning input to the detail generator in detail stage training
        if 'detail_conditioning' not in model_params.keys():
            # jaw, expression and detail code by default
            self.detail_conditioning = ['jawpose', 'expression', 'detail'] 
            OmegaConf.set_struct(model_params, True)
            with open_dict(model_params):
                model_params.detail_conditioning = self.detail_conditioning
        else:
            self.detail_conditioning = model_params.detail_conditioning

        # deprecated and is not used
        if 'detailemo_conditioning' not in model_params.keys():
            self.detailemo_conditioning = []
            OmegaConf.set_struct(model_params, True)
            with open_dict(model_params):
                model_params.detailemo_conditioning = self.detailemo_conditioning
        else:
            self.detailemo_conditioning = model_params.detailemo_conditioning

        supported_conditioning_keys = ['identity', 'jawpose', 'expression', 'detail', 'detailemo']
        
        for c in self.detail_conditioning:
            if c not in supported_conditioning_keys:
                raise ValueError(f"Conditioning on '{c}' is not supported. Supported conditionings: {supported_conditioning_keys}")
        for c in self.detailemo_conditioning:
            if c not in supported_conditioning_keys:
                raise ValueError(f"Conditioning on '{c}' is not supported. Supported conditionings: {supported_conditioning_keys}")

        # which type of DECA network is used
        if 'deca_class' not in model_params.keys() or model_params.deca_class is None:
            print(f"Deca class is not specified. Defaulting to {str(DECA.__class__.__name__)}")
            # vanilla DECA by default (not EMOCA)
            deca_class = DECA
        else:
            # other type of DECA-inspired networks possible (such as ExpDECA, which is what EMOCA)
            deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])

        # instantiate the network
        self.deca = deca_class(config=model_params)

        self.mode = DecaMode[str(model_params.mode).upper()]
        self.stage_name = stage_name
        if self.stage_name is None:
            self.stage_name = ""
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        
        # initialize the emotion perceptual loss (used for EMOCA supervision)
        self.emonet_loss = None
        self._init_emotion_loss()
        
        # initialize the au perceptual loss (not currently used in EMOCA)
        self.au_loss = None
        self._init_au_loss()

        # initialize the lip reading perceptual loss (not currently used in original EMOCA)
        self.lipread_loss = None
        self._init_lipread_loss()

        # MPL regressor from the encoded space to emotion labels (not used in EMOCA but could be used for direct emotion supervision)
        if 'mlp_emotion_predictor' in self.deca.config.keys():
            # self._build_emotion_mlp(self.deca.config.mlp_emotion_predictor)
            self.emotion_mlp = EmotionMLP(self.deca.config.mlp_emotion_predictor, model_params)
        else:
            self.emotion_mlp = None

    def get_input_image_size(self): 
        return (self.deca.config.image_size, self.deca.config.image_size)

    def _instantiate_deca(self, model_params):
        """
        Instantiate the DECA network.
        """
        # which type of DECA network is used
        if 'deca_class' not in model_params.keys() or model_params.deca_class is None:
            print(f"Deca class is not specified. Defaulting to {str(DECA.__class__.__name__)}")
            # vanilla DECA by default (not EMOCA)
            deca_class = DECA
        else:
            # other type of DECA-inspired networks possible (such as ExpDECA, which is what EMOCA)
            deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])

        # instantiate the network
        self.deca = deca_class(config=model_params)


    def _init_emotion_loss(self):
        """
        Initialize the emotion perceptual loss (used for EMOCA supervision)
        """
        if 'emonet_weight' in self.deca.config.keys() and bool(self.deca.config.emonet_model_path):
            if self.emonet_loss is not None:
                emoloss_force_override = True if 'emoloss_force_override' in self.deca.config.keys() and self.deca.config.emoloss_force_override else False
                if self.emonet_loss.is_trainable():
                    if not emoloss_force_override:
                        print("The old emonet loss is trainable and will not be overrided or replaced.")
                        return
                        # raise NotImplementedError("The old emonet loss was trainable. Changing a trainable loss is probably now "
                        #                       "what you want implicitly. If you need this, use the '`'emoloss_force_override' config.")
                    else:
                        print("The old emonet loss is trainable but override is set so it will be replaced.")
                else:
                    print("The old emonet loss is not trainable. It will be replaced.")

            if 'emonet_model_path' in self.deca.config.keys():
                emonet_model_path = self.deca.config.emonet_model_path
            else:
                emonet_model_path=None
            # self.emonet_loss = EmoNetLoss(self.device, emonet=emonet_model_path)
            emoloss_trainable = True if 'emoloss_trainable' in self.deca.config.keys() and self.deca.config.emoloss_trainable else False
            emoloss_dual = True if 'emoloss_dual' in self.deca.config.keys() and self.deca.config.emoloss_dual else False
            normalize_features = self.deca.config.normalize_features if 'normalize_features' in self.deca.config.keys() else None
            emo_feat_loss = self.deca.config.emo_feat_loss if 'emo_feat_loss' in self.deca.config.keys() else None
            old_emonet_loss = self.emonet_loss

            self.emonet_loss = create_emo_loss(self.device, emoloss=emonet_model_path, trainable=emoloss_trainable,
                                               dual=emoloss_dual,
                                               normalize_features=normalize_features,
                                               emo_feat_loss=emo_feat_loss)

            if old_emonet_loss is not None and type(old_emonet_loss) != self.emonet_loss:
                print(f"The old emonet loss {old_emonet_loss.__class__.__name__} is replaced during reconfiguration by "
                      f"new emotion loss {self.emonet_loss.__class__.__name__}")

        else:
            self.emonet_loss = None

    def _init_au_loss(self):
        """
        Initialize the au perceptual loss (not currently used in EMOCA)
        """
        if 'au_loss' in self.deca.config.keys():
            if self.au_loss is not None:
                force_override = True if 'force_override' in self.deca.config.au_loss.keys() \
                                         and self.deca.config.au_loss.force_override else False
                if self.au_loss.is_trainable():
                    if not force_override:
                        print("The old AU loss is trainable and will not be overrided or replaced.")
                        return
                        # raise NotImplementedError("The old emonet loss was trainable. Changing a trainable loss is probably now "
                        #                       "what you want implicitly. If you need this, use the '`'emoloss_force_override' config.")
                    else:
                        print("The old AU loss is trainable but override is set so it will be replaced.")
                else:
                    print("The old AU loss is not trainable. It will be replaced.")

            old_au_loss = self.emonet_loss
            self.au_loss = create_au_loss(self.device, self.deca.config.au_loss)
        else:
            self.au_loss = None

    def _init_lipread_loss(self):
        """
        Initialize the au perceptual loss (not currently used in EMOCA)
        """
        if 'lipread_loss' in self.deca.config.keys() and self.deca.config.lipread_loss.get('load', True):
            if self.lipread_loss is not None:
                force_override = True if 'force_override' in self.deca.config.lipread_loss.keys() \
                                         and self.deca.config.lipread_loss.force_override else False
                assert self.lipread_loss.is_trainable(), "Trainable lip reading loss is not supported yet."
                if self.lipread_loss.is_trainable():
                    if not force_override:
                        print("The old lip reading loss is trainable and will not be overrided or replaced.")
                        return
                        # raise NotImplementedError("The old emonet loss was trainable. Changing a trainable loss is probably now "
                        #                       "what you want implicitly. If you need this, use the '`'emoloss_force_override' config.")
                    else:
                        print("The old lip reading loss is trainable but override is set so it will be replaced.")
                else:
                    print("The old lip reading loss is not trainable. It will be replaced.")

            # old_lipread_loss = self.emonet_loss
            from gdl.models.temporal.external.LipReadingLoss import LipReadingLoss
            self.lipread_loss = LipReadingLoss(self.device, self.deca.config.lipread_loss.lipread_loss)
            self.lipread_loss.eval()
            self.lipread_loss.requires_grad_(False)
        else:
            self.lipread_loss = None

    def reconfigure(self, model_params, inout_params, learning_params, stage_name="", downgrade_ok=False, train=True):
        """
        Reconfigure the model. Usually used to switch between detail and coarse stages (which have separate configs)
        """
        if (self.mode == DecaMode.DETAIL and model_params.mode != DecaMode.DETAIL) and not downgrade_ok:
            raise RuntimeError("You're switching the EMOCA mode from DETAIL to COARSE. Is this really what you want?!")
        self.inout_params = inout_params
        self.learning_params = learning_params

        if self.deca.__class__.__name__ != model_params.deca_class:
            old_deca_class = self.deca.__class__.__name__
            state_dict = self.deca.state_dict()
            if 'deca_class' in model_params.keys():
                deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])
            else:
                deca_class = DECA
            self.deca = deca_class(config=model_params)

            diff = set(state_dict.keys()).difference(set(self.deca.state_dict().keys()))
            if len(diff) > 0:
                raise RuntimeError(f"Some values from old state dict will not be used. This is probably not what you "
                                   f"want because it most likely means that the pretrained model's weights won't be used. "
                                   f"Maybe you messed up backbone compatibility (i.e. SWIN vs ResNet?) {diff}")
            ret = self.deca.load_state_dict(state_dict, strict=False)
            if len(ret.unexpected_keys) > 0:
                raise print(f"Unexpected keys: {ret.unexpected_keys}")
            missing_modules = set([s.split(".")[0] for s in ret.missing_keys])
            print(f"Missing modules when upgrading from {old_deca_class} to {model_params.deca_class}:")
            print(missing_modules)
        else:
            self.deca._reconfigure(model_params)

        self._init_emotion_loss()
        self._init_au_loss()

        self.stage_name = stage_name
        if self.stage_name is None:
            self.stage_name = ""
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        self.mode = DecaMode[str(model_params.mode).upper()]
        self.train(mode=train)
        print(f"EMOCA MODE RECONFIGURED TO: {self.mode}")

        if 'shape_contrain_type' in self.deca.config.keys() and str(self.deca.config.shape_constrain_type).lower() != 'none':
            shape_constraint = self.deca.config.shape_constrain_type
        else:
            shape_constraint = None
        if 'expression_constrain_type' in self.deca.config.keys() and str(self.deca.config.expression_constrain_type).lower() != 'none':
            expression_constraint = self.deca.config.expression_constrain_type
        else:
            expression_constraint = None

        if shape_constraint is not None and expression_constraint is not None:
            raise ValueError("Both shape constraint and expression constraint are active. This is probably not what we want.")

    def uses_texture(self):
        """
        Check if the model uses texture
        """
        return self.deca.uses_texture()

    def visualize(self, visdict, savepath, catdim=1):
        return self.deca.visualize(visdict, savepath, catdim)

    def train(self, mode: bool = True):
        # super().train(mode) # not necessary
        self.deca.train(mode)

        if self.emotion_mlp is not None:
            self.emotion_mlp.train(mode)

        if self.emonet_loss is not None:
            self.emonet_loss.eval()

        if self.deca.perceptual_loss is not None:
            self.deca.perceptual_loss.eval()
        if self.deca.id_loss is not None:
            self.deca.id_loss.eval()

        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        return self

    def cpu(self):
        super().cpu()
        return self

    def forward(self, batch):
        values = self.encode(batch, training=False)
        values = self.decode(values, training=False)
        return values

    def _unwrap_list(self, codelist): 
        shapecode, texcode, expcode, posecode, cam, lightcode = codelist
        return shapecode, texcode, expcode, posecode, cam, lightcode

    
    def _unwrap_list_to_dict(self, codelist): 
        shapecode, texcode, expcode, posecode, cam, lightcode = codelist
        return {'shape': shapecode, 'tex': texcode, 'exp': expcode, 'pose': posecode, 'cam': cam, 'light': lightcode}
        # return shapecode, texcode, expcode, posecode, cam, lightcode

    def _encode_flame(self, images):
        if self.mode == DecaMode.COARSE or \
                (self.mode == DecaMode.DETAIL and self.deca.config.train_coarse):
            # forward pass with gradients (for coarse stage (used), or detail stage with coarse training (not used))
            parameters = self.deca._encode_flame(images)
        elif self.mode == DecaMode.DETAIL:
            # in detail stage, the coarse forward pass does not need gradients
            with torch.no_grad():
                parameters = self.deca._encode_flame(images)
        else:
            raise ValueError(f"Invalid EMOCA Mode {self.mode}")
        code_list, original_code = self.deca.decompose_code(parameters)
        # shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        # return shapecode, texcode, expcode, posecode, cam, lightcode, original_code
        return code_list, original_code

    def _expression_ring_exchange(self, original_batch_size, K,
                                  expcode, posecode, shapecode, lightcode, texcode,
                                  images, cam, lmk, masks, va, expr7, affectnetexp,
                                  detailcode=None, detailemocode=None, exprw=None, lmk_mp=None):
        """
        Deprecated. Expression ring exchange is not used in EMOCA (nor DECA).
        """
        new_order = np.array([np.random.permutation(K) + i * K for i in range(original_batch_size)])
        new_order = new_order.flatten()
        expcode_new = expcode[new_order]
        ## append new shape code data
        expcode = torch.cat([expcode, expcode_new], dim=0)
        texcode = torch.cat([texcode, texcode], dim=0)
        shapecode = torch.cat([shapecode, shapecode], dim=0)

        globpose = posecode[..., :3]
        jawpose = posecode[..., 3:]

        if self.deca.config.expression_constrain_use_jaw_pose:
            jawpose_new = jawpose[new_order]
            jawpose = torch.cat([jawpose, jawpose_new], dim=0)
        else:
            jawpose = torch.cat([jawpose, jawpose], dim=0)

        if self.deca.config.expression_constrain_use_global_pose:
            globpose_new = globpose[new_order]
            globpose = torch.cat([globpose, globpose_new], dim=0)
        else:
            globpose = torch.cat([globpose, globpose], dim=0)

        if self.deca.config.expression_constrain_use_jaw_pose or self.deca.config.expression_constrain_use_global_pose:
            posecode = torch.cat([globpose, jawpose], dim=-1)
            # posecode_new = torch.cat([globpose, jawpose], dim=-1)
        else:
            # posecode_new = posecode
            # posecode_new = posecode
            posecode = torch.cat([posecode, posecode], dim=0)

        cam = torch.cat([cam, cam], dim=0)
        lightcode = torch.cat([lightcode, lightcode], dim=0)
        ## append gt
        images = torch.cat([images, images],
                           dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        if lmk_mp is not None:
            lmk_mp = torch.cat([lmk_mp, lmk_mp], dim=0)
        masks = torch.cat([masks, masks], dim=0)


        # NOTE:
        # Here we could think about what makes sense to exchange
        # 1) Do we exchange all emotion GT (VA and expression) within the ring?
        # 2) Do we exchange only the GT on which the ring is constructed (AffectNet ring based on binned VA or expression or Emonet feature?)
        # note: if we use EmoMLP that goes from (expression, jawpose, detailcode) -> (v,a,expr) and we exchange
        # ALL of these, the EmoMLP prediction will of course be the same. The output image still changes,
        # so EmoNet loss (if used) would be different. Same for the photometric/landmark losses.

        # TODO:
        # For now I decided to exchange everything but this should probably be experimented with
        # I would argue though, that exchanging the GT is the right thing to do
        if va is not None:
            va = torch.cat([va, va[new_order]], dim=0)
        if expr7 is not None:
            expr7 = torch.cat([expr7, expr7[new_order]], dim=0)
        if affectnetexp is not None:
            affectnetexp = torch.cat([affectnetexp, affectnetexp[new_order]], dim=0)
        if exprw is not None:
            exprw = torch.cat([exprw, exprw[new_order]], dim=0)

        if detailcode is not None:
            #TODO: to exchange or not to exchange, that is the question, the answer is probably NO
            detailcode = torch.cat([detailcode, detailcode], dim=0)
            # detailcode = torch.cat([detailcode, detailcode[new_order]], dim=0)
        if detailemocode is not None:
            # TODO: to exchange or not to exchange, that is the question, the answer is probably YES
            detailemocode = torch.cat([detailemocode, detailemocode[new_order]], dim=0)

        return expcode, posecode, shapecode, lightcode, texcode, images, cam, lmk, masks, va, expr7, affectnetexp, \
               detailcode, detailemocode, exprw, lmk_mp

        # return expcode, posecode, shapecode, lightcode, texcode, images, cam, lmk, masks, va, expr7


    def encode(self, batch, training=True) -> dict:
        """
        Forward encoding pass of the model. Takes a batch of images and returns the corresponding latent codes for each image.
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        For a testing pass, the images suffice. 
        :param training: Whether the forward pass is for training or testing.
        """
        codedict = {}
        original_batch_size = batch['image'].shape[0]

        images = batch['image']

        if len(images.shape) == 5:
            K = images.shape[1]
        elif len(images.shape) == 4:
            K = 1
        else:
            raise RuntimeError("Invalid image batch dimensions.")

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        if 'landmark' in batch.keys():
            lmk = batch['landmark']
            lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        
        if 'landmark_mediapipe' in batch.keys():
            lmk_mp = batch['landmark_mediapipe']
            lmk_mp = lmk_mp.view(-1, lmk_mp.shape[-2], lmk_mp.shape[-1])
        else:
            lmk_mp = None

        if 'mask' in batch.keys():
            masks = batch['mask']
            masks = masks.view(-1, images.shape[-2], images.shape[-1])

        # valence / arousal - not necessary unless we want to use VA for supervision (not done in EMOCA)
        if 'va' in batch:
            va = batch['va']
            va = va.view(-1, va.shape[-1])
        else:
            va = None

        # 7 basic expression - not necessary unless we want to use expression for supervision (not done in EMOCA or DECA)
        if 'expr7' in batch:
            expr7 = batch['expr7']
            expr7 = expr7.view(-1, expr7.shape[-1])
        else:
            expr7 = None

        # affectnet basic expression - not necessary unless we want to use expression for supervision (not done in EMOCA or DECA)
        if 'affectnetexp' in batch:
            affectnetexp = batch['affectnetexp']
            affectnetexp = affectnetexp.view(-1, affectnetexp.shape[-1])
        else:
            affectnetexp = None

        # expression weights if supervising by expression is used (to balance the classification loss) - not done in EMOCA or DECA
        if 'expression_weight' in batch:
            exprw = batch['expression_weight']
            exprw = exprw.view(-1, exprw.shape[-1])
        else:
            exprw = None


        # 1) COARSE STAGE
        # forward pass of the coarse encoder
        # shapecode, texcode, expcode, posecode, cam, lightcode = self._encode_flame(images)
        code, original_code = self._encode_flame(images)
        shapecode, texcode, expcode, posecode, cam, lightcode = self._unwrap_list(code)
        if original_code is not None:
            original_code = self._unwrap_list_to_dict(original_code)

        if training:
            # If training, we employ the disentanglement strategy
            if self.mode == DecaMode.COARSE:

                if self.deca.config.shape_constrain_type == 'same':
                    ## Enforce that all identity shape codes within ring are the same. The batch is duplicated 
                    ## and the duplicated part's shape codes are shuffled.

                    # reshape shapecode => [B, K, n_shape]
                    # shapecode_idK = shapecode.view(self.batch_size, self.deca.K, -1)
                    shapecode_idK = shapecode.view(original_batch_size, K, -1)
                    # get mean id
                    shapecode_mean = torch.mean(shapecode_idK, dim=[1])
                    # shapecode_new = shapecode_mean[:, None, :].repeat(1, self.deca.K, 1)
                    shapecode_new = shapecode_mean[:, None, :].repeat(1, K, 1)
                    shapecode = shapecode_new.view(-1, self.deca._get_num_shape_params())

                    # do the same for the original code dict
                    shapecode_orig = original_code['shape']
                    shapecode_orig_idK = shapecode_orig.view(original_batch_size, K, -1)
                    shapecode_orig_mean = torch.mean(shapecode_orig_idK, dim=[1])
                    shapecode_orig_new = shapecode_orig_mean[:, None, :].repeat(1, K, 1)
                    original_code['shape'] = shapecode_orig_new.view(-1, self.deca._get_num_shape_params())

                elif self.deca.config.shape_constrain_type == 'exchange':
                    ## Shuffle identitys shape codes within ring (they should correspond to the same identity)
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
                    if lmk_mp is not None:
                        lmk_mp = torch.cat([lmk_mp, lmk_mp], dim=0)
                    masks = torch.cat([masks, masks], dim=0)

                    if va is not None:
                        va = torch.cat([va, va], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7], dim=0)

                    # do the same for the original code dict
                    shapecode_orig = original_code['shape']
                    shapecode_orig_new = shapecode_orig[new_order]
                    original_code['shape'] = torch.cat([shapecode_orig, shapecode_orig_new], dim=0)
                    original_code['tex'] = torch.cat([original_code['tex'], original_code['tex']], dim=0)
                    original_code['exp'] = torch.cat([original_code['exp'], original_code['exp']], dim=0)
                    original_code['pose'] = torch.cat([original_code['pose'], original_code['pose']], dim=0)
                    original_code['cam'] = torch.cat([original_code['cam'], original_code['cam']], dim=0)
                    original_code['light'] = torch.cat([original_code['light'], original_code['light']], dim=0)


                elif self.deca.config.shape_constrain_type == 'shuffle_expression':
                    assert original_code is not None
                    ## DEPRECATED, NOT USED IN EMOCA OR DECA
                    new_order = np.random.permutation(K*original_batch_size)
                    old_order = np.arange(K*original_batch_size)
                    while (new_order == old_order).any(): # ugly hacky way of assuring that every element is permuted
                        new_order = np.random.permutation(K * original_batch_size)
                    codedict['new_order'] = new_order
                    # exchange expression
                    expcode_new = expcode[new_order]
                    expcode = torch.cat([expcode, expcode_new], dim=0)

                    # exchange jaw pose (but not global pose)
                    global_pose = posecode[:, :3]
                    jaw_pose = posecode[:, 3:]
                    jaw_pose_new = jaw_pose[new_order]
                    jaw_pose = torch.cat([jaw_pose, jaw_pose_new], dim=0)
                    global_pose = torch.cat([global_pose, global_pose], dim=0)
                    posecode = torch.cat([global_pose, jaw_pose], dim=1)

                    ## duplicate the rest
                    shapecode = torch.cat([shapecode, shapecode], dim=0)
                    texcode = torch.cat([texcode, texcode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## duplicate gt if any
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    print(f"TRAINING: {training}")
                    if lmk is not None:
                        lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    if lmk_mp is not None:
                        lmk_mp = torch.cat([lmk_mp, lmk_mp], dim=0)
                    masks = torch.cat([masks, masks], dim=0)

                    ref_images_identity_idxs = np.concatenate([old_order, old_order])
                    ref_images_expression_idxs = np.concatenate([old_order, new_order])
                    codedict["ref_images_identity_idxs"] = ref_images_identity_idxs
                    codedict["ref_images_expression_idxs"] = ref_images_expression_idxs

                    if va is not None:
                        va = torch.cat([va, va[new_order]], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7[new_order]], dim=0)

                    # do the same for the original code dict
                    original_code['shape'] = torch.cat([original_code['shape'], original_code['shape']], dim=0)
                    original_code['tex'] = torch.cat([original_code['tex'], original_code['tex']], dim=0)
                    original_code['exp'] = torch.cat([original_code['exp'], original_code['exp'][new_order]], dim=0)
                    original_global_pose = original_code['pose'][:, :3]
                    original_jaw_pose = original_code['pose'][:, 3:]
                    original_jaw_pose = torch.cat([original_jaw_pose, original_jaw_pose[new_order]], dim=0)
                    original_global_pose = torch.cat([original_global_pose, original_global_pose], dim=0)
                    original_code['pose'] = torch.cat([original_global_pose, original_jaw_pose], dim=1)
                    original_code['cam'] = torch.cat([original_code['cam'], original_code['cam']], dim=0)
                    original_code['light'] = torch.cat([original_code['light'], original_code['light']], dim=0)

                elif self.deca.config.shape_constrain_type == 'shuffle_shape':
                    ## The shape codes are shuffled without duplication
                    new_order = np.random.permutation(K*original_batch_size)
                    old_order = np.arange(K*original_batch_size)
                    while (new_order == old_order).any(): # ugly hacky way of assuring that every element is permuted
                        new_order = np.random.permutation(K * original_batch_size)
                    codedict['new_order'] = new_order
                    shapecode_new = shapecode[new_order]
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
                    if lmk is not None:
                        lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    masks = torch.cat([masks, masks], dim=0)

                    ref_images_identity_idxs = np.concatenate([old_order, new_order])
                    ref_images_expression_idxs = np.concatenate([old_order, old_order])
                    codedict["ref_images_identity_idxs"] = ref_images_identity_idxs
                    codedict["ref_images_expression_idxs"] = ref_images_expression_idxs

                    if va is not None:
                        va = torch.cat([va, va], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7], dim=0)

                    # do the same for the original code dict
                    shapecode_orig = original_code['shape']
                    shapecode_orig_new = shapecode_orig[new_order]
                    original_code['shape'] = torch.cat([shapecode_orig, shapecode_orig_new], dim=0)
                    original_code['tex'] = torch.cat([original_code['tex'], original_code['tex']], dim=0)
                    original_code['exp'] = torch.cat([original_code['exp'], original_code['exp']], dim=0)
                    original_code['pose'] = torch.cat([original_code['pose'], original_code['pose']], dim=0)
                    original_code['cam'] = torch.cat([original_code['cam'], original_code['cam']], dim=0)
                    original_code['light'] = torch.cat([original_code['light'], original_code['light']], dim=0)
                    original_code['ref_images_identity_idxs'] = ref_images_identity_idxs
                    original_code['ref_images_expression_idxs'] = ref_images_expression_idxs

                elif 'expression_constrain_type' in self.deca.config.keys() and \
                        self.deca.config.expression_constrain_type == 'same':
                    ## NOT USED IN EMOCA OR DECA, deprecated

                    # reshape shapecode => [B, K, n_shape]
                    # shapecode_idK = shapecode.view(self.batch_size, self.deca.K, -1)
                    expcode_idK = expcode.view(original_batch_size, K, -1)
                    # get mean id
                    expcode_mean = torch.mean(expcode_idK, dim=[1])
                    # shapecode_new = shapecode_mean[:, None, :].repeat(1, self.deca.K, 1)
                    expcode_new = expcode_mean[:, None, :].repeat(1, K, 1)
                    expcode = expcode_new.view(-1, self.deca._get_num_shape_params())

                    # do the same thing for the original code dict
                    expcode_idK = original_code['exp'].view(original_batch_size, K, -1)
                    expcode_mean = torch.mean(expcode_idK, dim=[1])
                    expcode_new = expcode_mean[:, None, :].repeat(1, K, 1)
                    original_code['exp'] = expcode_new.view(-1, self.deca._get_num_shape_params())

                elif 'expression_constrain_type' in self.deca.config.keys() and \
                        self.deca.config.expression_constrain_type == 'exchange':
                    ## NOT USED IN EMOCA OR DECA, deprecated
                    expcode, posecode, shapecode, lightcode, texcode, images, cam, lmk, masks, va, expr7, affectnetexp, _, _, exprw, lmk_mp = \
                        self._expression_ring_exchange(original_batch_size, K,
                                  expcode, posecode, shapecode, lightcode, texcode,
                                  images, cam, lmk, masks, va, expr7, affectnetexp, None, None, exprw, lmk_mp)
                    # (self, original_batch_size, K,
                    #                                   expcode, posecode, shapecode, lightcode, texcode,
                    #                                   images, cam, lmk, masks, va, expr7, affectnetexp,
                    #                                   detailcode=None, detailemocode=None, exprw=None):

        # 2) DETAIL STAGE
        if self.mode == DecaMode.DETAIL:
            all_detailcode = self.deca.E_detail(images)

            # identity-based detail code
            detailcode = all_detailcode[:, :self.deca.n_detail]

            # detail emotion code is deprecated and will be empty
            detailemocode = all_detailcode[:, self.deca.n_detail:(self.deca.n_detail + self.deca.n_detail_emo)]

            if training:
                # If training, we employ the disentanglement strategy
                if self.deca.config.detail_constrain_type == 'exchange':
                    # Identity within the same ring should be the same, so they should have the same code. 
                    # This can be enforced by shuffling. The batch is duplicated and the duplicated part's code shuffled
                    '''
                    make sure s0, s1 is something to make shape close
                    the difference from ||so - s1|| is 
                    the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
                    '''
                    # this creates a per-ring random permutation. The detail exchange happens ONLY between the same
                    # identities (within the ring) but not outside (no cross-identity detail exchange)
                    new_order = np.array(
                        # [np.random.permutation(self.deca.config.train_K) + i * self.deca.config.train_K for i in range(original_batch_size)])
                        [np.random.permutation(K) + i * K for i in range(original_batch_size)])
                    new_order = new_order.flatten()
                    detailcode_new = detailcode[new_order]
                    detailcode = torch.cat([detailcode, detailcode_new], dim=0)
                    detailemocode = torch.cat([detailemocode, detailemocode], dim=0)
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

                elif self.deca.config.detail_constrain_type == 'shuffle_expression':
                    ## Deprecated and not used in EMOCA or DECA
                    new_order = np.random.permutation(K*original_batch_size)
                    old_order = np.arange(K*original_batch_size)
                    while (new_order == old_order).any(): # ugly hacky way of assuring that every element is permuted
                        new_order = np.random.permutation(K * original_batch_size)
                    codedict['new_order'] = new_order
                    # exchange expression
                    expcode_new = expcode[new_order]
                    expcode = torch.cat([expcode, expcode_new], dim=0)

                    # exchange emotion code, but not (identity-based) detailcode
                    detailemocode_new = detailemocode[new_order]
                    detailemocode = torch.cat([detailemocode, detailemocode_new], dim=0)
                    detailcode = torch.cat([detailcode, detailcode], dim=0)

                    # exchange jaw pose (but not global pose)
                    global_pose = posecode[:, :3]
                    jaw_pose = posecode[:, 3:]
                    jaw_pose_new = jaw_pose[new_order]
                    jaw_pose = torch.cat([jaw_pose, jaw_pose_new], dim=0)
                    global_pose = torch.cat([global_pose, global_pose], dim=0)
                    posecode = torch.cat([global_pose, jaw_pose], dim=1)


                    ## duplicate the rest
                    shapecode = torch.cat([shapecode, shapecode], dim=0)
                    texcode = torch.cat([texcode, texcode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## duplicate gt if any
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    print(f"TRAINING: {training}")
                    if lmk is not None:
                        lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    masks = torch.cat([masks, masks], dim=0)

                    ref_images_identity_idxs = np.concatenate([old_order, old_order])
                    ref_images_expression_idxs = np.concatenate([old_order, new_order])
                    codedict["ref_images_identity_idxs"] = ref_images_identity_idxs
                    codedict["ref_images_expression_idxs"] = ref_images_expression_idxs

                    if va is not None:
                        va = torch.cat([va, va[new_order]], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7[new_order]], dim=0)

                elif self.deca.config.detail_constrain_type == 'shuffle_shape':
                    ## Shuffles teh shape code without duplicating the batch
                    new_order = np.random.permutation(K*original_batch_size)
                    old_order = np.arange(K*original_batch_size)
                    while (new_order == old_order).any(): # ugly hacky way of assuring that every element is permuted
                        new_order = np.random.permutation(K * original_batch_size)
                    codedict['new_order'] = new_order
                    shapecode_new = shapecode[new_order]
                    ## append new shape code data
                    shapecode = torch.cat([shapecode, shapecode_new], dim=0)

                    # exchange (identity-based) detailcode, but not emotion code
                    detailcode_new = detailcode[new_order]
                    detailcode = torch.cat([detailcode, detailcode_new], dim=0)
                    detailemocode = torch.cat([detailemocode, detailemocode], dim=0)

                    texcode = torch.cat([texcode, texcode], dim=0)
                    expcode = torch.cat([expcode, expcode], dim=0)
                    posecode = torch.cat([posecode, posecode], dim=0)
                    cam = torch.cat([cam, cam], dim=0)
                    lightcode = torch.cat([lightcode, lightcode], dim=0)
                    ## append gt
                    images = torch.cat([images, images],
                                       dim=0)  # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    if lmk is not None:
                        lmk = torch.cat([lmk, lmk], dim=0)  # lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
                    masks = torch.cat([masks, masks], dim=0)

                    ref_images_identity_idxs = np.concatenate([old_order, new_order])
                    ref_images_expression_idxs = np.concatenate([old_order, old_order])
                    codedict["ref_images_identity_idxs"] = ref_images_identity_idxs
                    codedict["ref_images_expression_idxs"] = ref_images_expression_idxs

                    if va is not None:
                        va = torch.cat([va, va], dim=0)
                    if expr7 is not None:
                        expr7 = torch.cat([expr7, expr7], dim=0)

                elif 'expression_constrain_type' in self.deca.config.keys() and \
                        self.deca.config.expression_constrain_type == 'exchange':
                    expcode, posecode, shapecode, lightcode, texcode, images, cam, lmk, masks, va, expr7, affectnetexp, detailcode, detailemocode, exprw = \
                        self._expression_ring_exchange(original_batch_size, K,
                                  expcode, posecode, shapecode, lightcode, texcode,
                                  images, cam, lmk, masks, va, expr7, affectnetexp, detailcode, detailemocode, exprw)


        codedict['shapecode'] = shapecode
        codedict['texcode'] = texcode
        codedict['expcode'] = expcode
        codedict['posecode'] = posecode
        codedict['cam'] = cam
        codedict['lightcode'] = lightcode
        if self.mode == DecaMode.DETAIL:
            codedict['detailcode'] = detailcode
            codedict['detailemocode'] = detailemocode
        codedict['images'] = images
        if 'mask' in batch.keys():
            codedict['masks'] = masks
        if 'landmark' in batch.keys():
            codedict['lmk'] = lmk
        if lmk_mp is not None:
            codedict['lmk_mp'] = lmk_mp

        if 'va' in batch.keys():
            codedict['va'] = va
        if 'expr7' in batch.keys():
            codedict['expr7'] = expr7
        if 'affectnetexp' in batch.keys():
            codedict['affectnetexp'] = affectnetexp

        if 'expression_weight' in batch.keys():
            codedict['expression_weight'] = exprw

        if original_code is not None:
            codedict['original_code'] = original_code

        return codedict


    def _create_conditioning_lists(self, codedict, condition_list):
        detail_conditioning_list = []
        if 'globalpose' in condition_list:
            detail_conditioning_list += [codedict["posecode"][:, :3]]
        if 'jawpose' in condition_list:
            detail_conditioning_list += [codedict["posecode"][:, 3:]]
        if 'identity' in condition_list:
            detail_conditioning_list += [codedict["shapecode"]]
        if 'expression' in condition_list:
            detail_conditioning_list += [codedict["expcode"]]

        if isinstance(self.deca.D_detail, Generator):
            # the detail codes might be excluded from conditioning based on the Generator architecture (for instance
            # for AdaIn Generator)
            if 'detail' in condition_list:
                detail_conditioning_list += [codedict["detailcode"]]
            if 'detailemo' in condition_list:
                detail_conditioning_list += [codedict["detailemocode"]]

        return detail_conditioning_list


    def decode(self, codedict, training=True, render=True, **kwargs) -> dict:
        """
        Forward decoding pass of the model. Takes the latent code predicted by the encoding stage and reconstructs and renders the shape.
        :param codedict: Batch dict of the predicted latent codes
        :param training: Whether the forward pass is for training or testing.
        """
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

        # 1) Reconstruct the face mesh
        # FLAME - world space
        if not isinstance(self.deca.flame, FLAME_mediapipe):
            verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                          pose_params=posecode)
            landmarks2d_mediapipe = None
        else:
            verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = self.deca.flame(shapecode, expcode, posecode)
        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]

        if landmarks2d_mediapipe is not None:
            predicted_landmarks_mediapipe = util.batch_orth_proj(landmarks2d_mediapipe, cam)[:, :, :2]
            predicted_landmarks_mediapipe[:, :, 1:] = - predicted_landmarks_mediapipe[:, :, 1:]

        if self.uses_texture():
            albedo = self.deca.flametex(texcode)
        else: 
            # if not using texture, default to gray
            albedo = torch.ones([effective_batch_size, 3, self.deca.config.uv_size, self.deca.config.uv_size], device=images.device) * 0.5

        # 2) Render the coarse image
        if render:
            ops = self.deca.render(verts, trans_verts, albedo, lightcode)
            # mask
            mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(effective_batch_size, -1, -1, -1),
                                        ops['grid'].detach(),
                                        align_corners=False)
            # images
            predicted_images = ops['images']
            # predicted_images = ops['images'] * mask_face_eye * ops['alpha_images']
            # predicted_images_no_mask = ops['images'] #* mask_face_eye * ops['alpha_images']
            segmentation_type = None
            if isinstance(self.deca.config.useSeg, bool):
                if self.deca.config.useSeg:
                    segmentation_type = 'gt'
                else:
                    segmentation_type = 'rend'
            elif isinstance(self.deca.config.useSeg, str):
                segmentation_type = self.deca.config.useSeg
            else:
                raise RuntimeError(f"Invalid 'useSeg' type: '{type(self.deca.config.useSeg)}'")

            if segmentation_type not in ["gt", "rend", "intersection", "union"]:
                raise ValueError(f"Invalid segmentation type for masking '{segmentation_type}'")

            if masks is None: # if mask not provided, the only mask available is the rendered one
                segmentation_type = 'rend'

            elif masks.shape[-1] != predicted_images.shape[-1] or masks.shape[-2] != predicted_images.shape[-2]:
                # resize masks if need be (this is only done if configuration was changed at some point after training)
                dims = masks.ndim == 3
                if dims:
                    masks = masks[:, None, :, :]
                masks = F.interpolate(masks, size=predicted_images.shape[-2:], mode='bilinear')
                if dims:
                    masks = masks[:, 0, ...]

            # resize images if need be (this is only done if configuration was changed at some point after training)
            if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                ## special case only for inference time if the rendering image sizes have been changed
                images_resized = F.interpolate(images, size=predicted_images.shape[-2:], mode='bilinear')
            else:
                images_resized = images

            # what type of segmentation we use
            if segmentation_type == "gt": # GT stands for external segmetnation predicted by face parsing or similar
                masks = masks[:, None, :, :]
            elif segmentation_type == "rend": # mask rendered as a silhouette of the face mesh
                masks = mask_face_eye * ops['alpha_images']
            elif segmentation_type == "intersection": # intersection of the two above
                masks = masks[:, None, :, :] * mask_face_eye * ops['alpha_images']
            elif segmentation_type == "union": # union of the first two options
                masks = torch.max(masks[:, None, :, :],  mask_face_eye * ops['alpha_images'])
            else:
                raise RuntimeError(f"Invalid segmentation type for masking '{segmentation_type}'")


            if self.deca.config.background_from_input in [True, "input"]:
                if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                    ## special case only for inference time if the rendering image sizes have been changed
                    predicted_images = (1. - masks) * images_resized + masks * predicted_images
                else:
                    predicted_images = (1. - masks) * images + masks * predicted_images
            elif self.deca.config.background_from_input in [False, "black"]:
                predicted_images = masks * predicted_images
            elif self.deca.config.background_from_input in ["none"]:
                predicted_images = predicted_images
            else:
                raise ValueError(f"Invalid type of background modification {self.deca.config.background_from_input}")

        # 3) Render the detail image
        if self.mode == DecaMode.DETAIL:
            detailcode = codedict['detailcode']
            detailemocode = codedict['detailemocode']

            # a) Create the detail conditioning lists
            detail_conditioning_list = self._create_conditioning_lists(codedict, self.detail_conditioning)
            detailemo_conditioning_list = self._create_conditioning_lists(codedict, self.detailemo_conditioning)
            final_detail_conditioning_list = detail_conditioning_list + detailemo_conditioning_list


            # b) Pass the detail code and the conditions through the detail generator to get displacement UV map
            if isinstance(self.deca.D_detail, Generator):
                uv_z = self.deca.D_detail(torch.cat(final_detail_conditioning_list, dim=1))
            elif isinstance(self.deca.D_detail, GeneratorAdaIn):
                uv_z = self.deca.D_detail(z=torch.cat([detailcode, detailemocode], dim=1),
                                          cond=torch.cat(final_detail_conditioning_list, dim=1))
            else:
                raise ValueError(f"This class of generarator is not supported: '{self.deca.D_detail.__class__.__name__}'")

            # if there is a displacement mask, apply it (DEPRECATED and not USED in DECA or EMOCA)
            if hasattr(self.deca, 'displacement_mask') and self.deca.displacement_mask is not None:
                if 'apply_displacement_masks' in self.deca.config.keys() and self.deca.config.apply_displacement_masks:
                    uv_z = uv_z * self.deca.displacement_mask

            # uv_z = self.deca.D_detail(torch.cat([posecode[:, 3:], expcode, detailcode], dim=1))
            # render detail
            if render:
                detach_from_coarse_geometry = not self.deca.config.train_coarse
                uv_detail_normals, uv_coarse_vertices = self.deca.displacement2normal(uv_z, verts, ops['normals'],
                                                                                    detach=detach_from_coarse_geometry)
                uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
                uv_texture = albedo.detach() * uv_shading

                # batch size X image_rows X image_cols X 2
                # you can query the grid for UV values of the face mesh at pixel locations
                grid = ops['grid']
                if detach_from_coarse_geometry:
                    # if the grid is detached, the gradient of the positions of UV-values in image space won't flow back to the geometry
                    grid = grid.detach()
                predicted_detailed_image = F.grid_sample(uv_texture, grid, align_corners=False)
                if self.deca.config.background_from_input in [True, "input"]:
                    if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                        ## special case only for inference time if the rendering image sizes have been changed
                        # images_resized = F.interpolate(images, size=predicted_images.shape[-2:], mode='bilinear')
                        ## before bugfix
                        # predicted_images = (1. - masks) * images_resized + masks * predicted_images
                        ## after bugfix
                        predicted_detailed_image = (1. - masks) * images_resized + masks * predicted_detailed_image
                    else:
                        predicted_detailed_image = (1. - masks) * images + masks * predicted_detailed_image
                elif self.deca.config.background_from_input in [False, "black"]:
                    predicted_detailed_image = masks * predicted_detailed_image
                elif self.deca.config.background_from_input in ["none"]:
                    predicted_detailed_image = predicted_detailed_image
                else:
                    raise ValueError(f"Invalid type of background modification {self.deca.config.background_from_input}")


                # --- extract texture
                uv_pverts = self.deca.render.world2uv(trans_verts).detach()
                uv_gt = F.grid_sample(torch.cat([images_resized, masks], dim=1), uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2],
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


        ## 4) (Optional) NEURAL RENDERING - not used in neither DECA nor EMOCA
        # If neural rendering is enabled, the differentiable rendered synthetic images are translated using an image translation net (such as StarGan)
        predicted_translated_image = None
        predicted_detailed_translated_image = None
        translated_uv_texture = None

        if render:
            if self.deca._has_neural_rendering():
                predicted_translated_image = self.deca.image_translator(
                    {
                        "input_image" : predicted_images,
                        "ref_image" : images,
                        "target_domain" : torch.tensor([0]*predicted_images.shape[0],
                                                    dtype=torch.int64, device=predicted_images.device)
                    }
                )

                if self.mode == DecaMode.DETAIL:
                    predicted_detailed_translated_image = self.deca.image_translator(
                            {
                                "input_image" : predicted_detailed_image,
                                "ref_image" : images,
                                "target_domain" : torch.tensor([0]*predicted_detailed_image.shape[0],
                                                            dtype=torch.int64, device=predicted_detailed_image.device)
                            }
                        )
                    translated_uv = F.grid_sample(torch.cat([predicted_detailed_translated_image, masks], dim=1), uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2],
                                        mode='bilinear')
                    translated_uv_texture = translated_uv[:, :3, :, :].detach()

                else:
                    predicted_detailed_translated_image = None

                    translated_uv_texture = None
                    # no need in coarse mode
                    # translated_uv = F.grid_sample(torch.cat([predicted_translated_image, masks], dim=1), uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2],
                    #                       mode='bilinear')
                    # translated_uv_texture = translated_uv_gt[:, :3, :, :].detach()

        if self.emotion_mlp is not None:
            codedict = self.emotion_mlp(codedict, "emo_mlp_")

        # populate the value dict for metric computation/visualization
        if render:
            codedict['predicted_images'] = predicted_images
            codedict['predicted_detailed_image'] = predicted_detailed_image
            codedict['predicted_translated_image'] = predicted_translated_image
            codedict['ops'] = ops
            codedict['normals'] = ops['normals']
            codedict['mask_face_eye'] = mask_face_eye
        
        codedict['verts'] = verts
        codedict['albedo'] = albedo
        codedict['landmarks2d'] = landmarks2d
        codedict['landmarks3d'] = landmarks3d
        codedict['predicted_landmarks'] = predicted_landmarks
        if landmarks2d_mediapipe is not None:
            codedict['predicted_landmarks_mediapipe'] = predicted_landmarks_mediapipe
        codedict['trans_verts'] = trans_verts
        codedict['masks'] = masks

        if self.mode == DecaMode.DETAIL:
            if render:
                codedict['predicted_detailed_translated_image'] = predicted_detailed_translated_image
                codedict['translated_uv_texture'] = translated_uv_texture
                codedict['uv_texture_gt'] = uv_texture_gt
                codedict['uv_texture'] = uv_texture
                codedict['uv_detail_normals'] = uv_detail_normals
                codedict['uv_shading'] = uv_shading
                codedict['uv_vis_mask'] = uv_vis_mask
                codedict['uv_mask'] = uv_mask
            codedict['uv_z'] = uv_z
            codedict['displacement_map'] = uv_z + self.deca.fixed_uv_dis[None, None, :, :]

        return codedict

    def _compute_emotion_loss(self, images, predicted_images, loss_dict, metric_dict, prefix, va=None, expr7=None, with_grad=True,
                              batch_size=None, ring_size=None):
        def loss_or_metric(name, loss, is_loss):
            if not is_loss:
                metric_dict[name] = loss
            else:
                loss_dict[name] = loss

        # if self.deca.config.use_emonet_loss:
        if with_grad:
            d = loss_dict
            emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss, au_loss = \
                self.emonet_loss.compute_loss(images, predicted_images, batch_size=batch_size, ring_size=ring_size)
        else:
            d = metric_dict
            with torch.no_grad():
                emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss, au_loss = \
                    self.emonet_loss.compute_loss(images, predicted_images, batch_size=batch_size, ring_size=ring_size)



        # EmoNet self-consistency loss terms
        if emo_feat_loss_1 is not None:
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
        loss_or_metric(prefix + '_emonet_combined', ((emo_feat_loss_1 if emo_feat_loss_1 is not None else 0)
                                                     + emo_feat_loss_2 + valence_loss + arousal_loss + expression_loss) * self.deca.config.emonet_weight,
                       self.deca.config.use_emonet_combined and self.deca.config.use_emonet_loss)

        # Log also the VA
        metric_dict[prefix + "_valence_input"] = self.emonet_loss.input_emotion['valence'].mean().detach()
        metric_dict[prefix + "_valence_output"] = self.emonet_loss.output_emotion['valence'].mean().detach()
        metric_dict[prefix + "_arousal_input"] = self.emonet_loss.input_emotion['arousal'].mean().detach()
        metric_dict[prefix + "_arousal_output"] = self.emonet_loss.output_emotion['arousal'].mean().detach()

        input_ex = self.emonet_loss.input_emotion['expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification'].detach().cpu().numpy()
        input_ex = np.argmax(input_ex, axis=1).mean()
        output_ex = self.emonet_loss.output_emotion['expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification'].detach().cpu().numpy()
        output_ex = np.argmax(output_ex, axis=1).mean()
        metric_dict[prefix + "_expression_input"] = torch.tensor(input_ex, device=self.device)
        metric_dict[prefix + "_expression_output"] = torch.tensor(output_ex, device=self.device)

        # # GT emotion loss terms
        # if self.deca.config.use_gt_emotion_loss:
        #     d = loss_dict
        # else:
        #     d = metric_dict

        # TODO: uncomment this after you handle the case when certain entries are NaN (GT missing, not a bug)
        # if va is not None:
        #     d[prefix + 'emo_sup_val_L1'] = F.l1_loss(self.emonet_loss.output_emotion['valence'], va[:, 0]) \
        #                                    * self.deca.config.gt_emotion_reg
        #     d[prefix + 'emo_sup_ar_L1'] = F.l1_loss(self.emonet_loss.output_emotion['arousal'], va[:, 1]) \
        #                                   * self.deca.config.gt_emotion_reg
        #
        #     metric_dict[prefix + "_valence_gt"] = va[:, 0].mean().detach()
        #     metric_dict[prefix + "_arousal_gt"] = va[:, 1].mean().detach()
        #
        # if expr7 is not None:
        #     affectnet_gt = [expr7_to_affect_net(int(expr7[i])).value for i in range(len(expr7))]
        #     affectnet_gt = torch.tensor(np.array(affectnet_gt), device=self.device, dtype=torch.long)
        #     d[prefix + '_emo_sup_expr_CE'] = F.cross_entropy(self.emonet_loss.output_emotion['expression'], affectnet_gt) * self.deca.config.gt_emotion_reg
        #     metric_dict[prefix + "_expr_gt"] = affectnet_gt.mean().detach()


    def _compute_au_loss(self, images, predicted_images, loss_dict, metric_dict, prefix, au=None, with_grad=True):
        def loss_or_metric(name, loss, is_loss):
            if not is_loss:
                metric_dict[name] = loss
            else:
                loss_dict[name] = loss

        # if self.deca.config.use_emonet_loss:
        if with_grad:
            d = loss_dict
            au_feat_loss_1, au_feat_loss_2, _, _, _, au_loss = \
                self.au_loss.compute_loss(images, predicted_images)
        else:
            d = metric_dict
            with torch.no_grad():
                au_feat_loss_1, au_feat_loss_2, _, _, _, au_loss = \
                    self.au_loss.compute_loss(images, predicted_images)



        # EmoNet self-consistency loss terms
        if au_feat_loss_1 is not None:
            loss_or_metric(prefix + '_au_feat_1_L1', au_feat_loss_1 * self.deca.config.au_loss.au_weight,
                           self.deca.config.au_loss.use_feat_1 and self.deca.config.au_loss.use_as_loss)
        loss_or_metric(prefix + '_au_feat_2_L1', au_feat_loss_2 * self.deca.config.au_loss.au_weight,
                       self.deca.config.au_loss.use_feat_2 and self.deca.config.au_loss.use_as_loss)
        loss_or_metric(prefix + '_au_loss', au_loss * self.deca.config.au_loss.au_weight,
                       self.deca.config.au_loss.use_aus and self.deca.config.au_loss.use_as_loss)
        # loss_or_metric(prefix + '_au_losses_L1', arousal_loss * self.deca.config.au_loss.au_weight,
        #                self.deca.config.au_loss.use_emonet_arousal and self.deca.config.au_loss.use_as_loss)
        # loss_or_metric(prefix + 'emonet_expression_KL', expression_loss * self.deca.config.au_loss.au_weight) # KL seems to be causing NaN's

        # # Log also the VA
        # metric_dict[prefix + "_valence_input"] = self.emonet_loss.input_emotion['valence'].mean().detach()
        # metric_dict[prefix + "_valence_output"] = self.emonet_loss.output_emotion['valence'].mean().detach()
        # metric_dict[prefix + "_arousal_input"] = self.emonet_loss.input_emotion['arousal'].mean().detach()
        # metric_dict[prefix + "_arousal_output"] = self.emonet_loss.output_emotion['arousal'].mean().detach()

        # input_ex = self.emonet_loss.input_emotion['expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification'].detach().cpu().numpy()
        # input_ex = np.argmax(input_ex, axis=1).mean()
        # output_ex = self.emonet_loss.output_emotion['expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification'].detach().cpu().numpy()
        # output_ex = np.argmax(output_ex, axis=1).mean()
        # metric_dict[prefix + "_expression_input"] = torch.tensor(input_ex, device=self.device)
        # metric_dict[prefix + "_expression_output"] = torch.tensor(output_ex, device=self.device)

        # # GT emotion loss terms
        # if self.deca.config.use_gt_emotion_loss:
        #     d = loss_dict
        # else:
        #     d = metric_dict


    def _cut_mouth_vectorized(self, images, landmarks, convert_grayscale=True):
        # mouth_window_margin = 12
        mouth_window_margin = 1 # not temporal
        mouth_crop_height = 96
        mouth_crop_width = 96
        mouth_landmark_start_idx = 48
        mouth_landmark_stop_idx = 68
        B, T = images.shape[:2]

        landmarks = landmarks.to(torch.float32)

        with torch.no_grad():
            image_size = images.shape[-1] / 2

            landmarks = landmarks * image_size + image_size
            # #1) smooth the landmarks with temporal convolution
            # landmarks are of shape (T, 68, 2) 
            # reshape to (T, 136) 
            landmarks_t = landmarks.reshape(*landmarks.shape[:2], -1)
            # make temporal dimension last 
            landmarks_t = landmarks_t.permute(0, 2, 1)
            # change chape to (N, 136, T)
            # landmarks_t = landmarks_t.unsqueeze(0)
            # smooth with temporal convolution
            temporal_filter = torch.ones(mouth_window_margin, device=images.device) / mouth_window_margin
            # pad the the landmarks 
            landmarks_t_padded = F.pad(landmarks_t, (mouth_window_margin // 2, mouth_window_margin // 2), mode='replicate')
            # convolve each channel separately with the temporal filter
            num_channels = landmarks_t.shape[1]
            if temporal_filter.numel() > 1:
                smooth_landmarks_t = F.conv1d(landmarks_t_padded, 
                    temporal_filter.unsqueeze(0).unsqueeze(0).expand(num_channels,1,temporal_filter.numel()), 
                    groups=num_channels, padding='valid'
                )
                smooth_landmarks_t = smooth_landmarks_t[..., 0:landmarks_t.shape[-1]]
            else:
                smooth_landmarks_t = landmarks_t

            # reshape back to the original shape 
            smooth_landmarks_t = smooth_landmarks_t.permute(0, 2, 1).view(landmarks.shape)
            smooth_landmarks_t = smooth_landmarks_t + landmarks.mean(dim=2, keepdims=True) - smooth_landmarks_t.mean(dim=2, keepdims=True)

            # #2) get the mouth landmarks
            mouth_landmarks_t = smooth_landmarks_t[..., mouth_landmark_start_idx:mouth_landmark_stop_idx, :]
            
            # #3) get the mean of the mouth landmarks
            mouth_landmarks_mean_t = mouth_landmarks_t.mean(dim=-2, keepdims=True)
        
            # #4) get the center of the mouth
            center_x_t = mouth_landmarks_mean_t[..., 0]
            center_y_t = mouth_landmarks_mean_t[..., 1]

            # #5) use grid_sample to crop the mouth in every image 
            # create the grid
            height = mouth_crop_height//2
            width = mouth_crop_width//2

            torch.arange(0, mouth_crop_width, device=images.device)

            grid = torch.stack(torch.meshgrid(torch.linspace(-height, height, mouth_crop_height).to(images.device) / (images.shape[-2] /2),
                                            torch.linspace(-width, width, mouth_crop_width).to(images.device) / (images.shape[-1] /2) ), 
                                            dim=-1)
            grid = grid[..., [1, 0]]
            grid = grid.unsqueeze(0).unsqueeze(0).repeat(*images.shape[:2], 1, 1, 1)

            center_x_t -= images.shape[-1] / 2
            center_y_t -= images.shape[-2] / 2

            center_x_t /= images.shape[-1] / 2
            center_y_t /= images.shape[-2] / 2

            grid = grid + torch.cat([center_x_t, center_y_t ], dim=-1).unsqueeze(-2).unsqueeze(-2)

        images = images.view(B*T, *images.shape[2:])
        grid = grid.view(B*T, *grid.shape[2:])

        if convert_grayscale: 
            images = F_v.rgb_to_grayscale(images)

        image_crops = F.grid_sample(
            images, 
            grid,  
            align_corners=True, 
            padding_mode='zeros',
            mode='bicubic'
            )
        image_crops = image_crops.view(B, T, *image_crops.shape[1:])

        if convert_grayscale:
            image_crops = image_crops#.squeeze(1)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(image_crops[0, 0].permute(1,2,0).cpu().numpy())
        # plt.show()

        # plt.figure()
        # plt.imshow(image_crops[0, 10].permute(1,2,0).cpu().numpy())
        # plt.show()

        # plt.figure()
        # plt.imshow(image_crops[0, 20].permute(1,2,0).cpu().numpy())
        # plt.show()

        # plt.figure()
        # plt.imshow(image_crops[1, 0].permute(1,2,0).cpu().numpy())
        # plt.show()

        # plt.figure()
        # plt.imshow(image_crops[1, 10].permute(1,2,0).cpu().numpy())
        # plt.show()

        # plt.figure()
        # plt.imshow(image_crops[1, 20].permute(1,2,0).cpu().numpy())
        # plt.show()
        return image_crops


    def _compute_lipread_loss(self, images, predicted_images, landmarks, predicted_landmarks, loss_dict, metric_dict, prefix, with_grad=True): 
        def loss_or_metric(name, loss, is_loss):
            if not is_loss:
                metric_dict[name] = loss
            else:
                loss_dict[name] = loss

        # shape of images is: (B, R, C, H, W)
        # convert to (B * R, 1, H, W, C)
        images = images.unsqueeze(1)
        predicted_images = predicted_images.unsqueeze(1)
        landmarks = landmarks.unsqueeze(1)
        predicted_landmarks = predicted_landmarks.unsqueeze(1)

        # cut out the mouth region


        images_mouth = self._cut_mouth_vectorized(images, landmarks)
        predicted_images_mouth  = self._cut_mouth_vectorized(predicted_images, predicted_landmarks)

        # make sure that the lip reading net interprests  things with depth=1, 

        # if self.deca.config.use_emonet_loss:
        if with_grad:
            d = loss_dict
            loss = self.lipread_loss.compute_loss(images_mouth, predicted_images_mouth)
        else:
            d = metric_dict
            with torch.no_grad():
                loss = self.lipread_loss.compute_loss(images_mouth, predicted_images_mouth)

        d[prefix + '_lipread'] = loss * self.deca.config.lipread_loss.weight


    def _metric_or_loss(self, loss_dict, metric_dict, is_loss):
        if is_loss:
            d = loss_dict
        else:
            d = metric_dict
        return d


    def _compute_id_loss(self, codedict, batch, training, testing, losses, batch_size,
                                                       ring_size):
        # if self.deca.config.idw > 1e-3:
        if self.deca.id_loss is not None:

            images = codedict["images"]

            ops = codedict["ops"]
            mask_face_eye = codedict["mask_face_eye"]

            shading_images = self.deca.render.add_SHlight(ops['normal_images'], codedict["lightcode"].detach())
            albedo_images = F.grid_sample(codedict["albedo"].detach(), ops['grid'], align_corners=False)

            # TODO: get to the bottom of this weird overlay thing - why is it there?
            # answer: This renders the face and takes background from the image
            overlay = albedo_images * shading_images * mask_face_eye + images * (1 - mask_face_eye)

            if self.global_step >= self.deca.id_loss_start_step:
                if 'id_metric' in self.deca.config.keys() and 'barlow_twins' in self.deca.config.id_metric:
                    assert ring_size == 1 or ring_size == 2

                effective_bs = images.shape[0]
                # losses['identity'] = self.deca.id_loss(overlay, images, batch_size=batch_size,
                #                                        ring_size=ring_size) * self.deca.config.idw

                if "ref_images_identity_idxs" in codedict.keys():
                    # in case there was shuffling, this ensures that the proper images are used for identity loss
                    images_ = images[codedict["ref_images_identity_idxs"]]
                else:
                    images_ = images
                losses['identity'] = self.deca.id_loss(overlay, images_, batch_size=effective_bs,
                                                       ring_size=1) * self.deca.config.idw
                if 'id_contrastive' in self.deca.config.keys() and bool(self.deca.config.id_contrastive):
                    if ring_size == 2:
                        assert effective_bs % 2 == 0
                        assert self.deca.id_loss.trainable
                        has_been_shuffled = 'new_order' in codedict.keys()

                        idxs_a = torch.arange(0, images.shape[0], 2)  # indices of first images within the ring
                        idxs_b = torch.arange(1, images.shape[0], 2)  # indices of second images within the ring

                        # WARNING - this assumes the ring is identity-based
                        if self.deca.config.id_contrastive in [True, "real", "both"]:
                            # we are taking this from the original batch dict because we do not care about the
                            # shuffled, duplicated samples (they don't have to be doubled)
                            images_0 = batch["image"][:, 0, ...]
                            images_1 = batch["image"][:, 1, ...]
                            losses['identity_contrastive_real'] = self.deca.id_loss(
                                images_0,  # first images within the ring
                                images_1,  # second images within the ring
                                batch_size=images_0.shape[0],
                                ring_size=1) * self.deca.config.idw * 2
                        if self.deca.config.id_contrastive in [True, "synth", "both"]:

                            if self.deca.config.shape_constrain_type in ['exchange', 'same']:
                                # we can take all when identity has been exchange within rings
                                overlay_0 = overlay[idxs_a]
                                overlay_1 = overlay[idxs_b]
                            else:
                                #if the batch was double otherwise (global shuffling) we only take the first half
                                # if batch_size * ring_size < effective_bs:
                                overlay_0 = overlay[0:batch_size * ring_size:2]
                                overlay_1 = overlay[1:batch_size * ring_size:2]

                            losses['identity_contrastive_synthetic'] = self.deca.id_loss(
                                overlay_0,  # first images within the ring
                                overlay_1,  # second images within the ring
                                batch_size=overlay_0.shape[0],
                                ring_size=1) * self.deca.config.idw


                        if has_been_shuffled:
                            new_order = codedict['new_order']

                            # TODO: compare the idxs to these:
                            # codedict["ref_images_identity_idxs"]

                            if self.deca.config.shape_constrain_type == 'shuffle_expression':
                                idxs_a_synth = np.arange(new_order.shape[0])  # first half of the batch
                                idxs_b_synth = np.arange(new_order.shape[0],
                                                         2 * new_order.shape[0])  # second half of the batch
                            elif self.deca.config.shape_constrain_type == 'shuffle_shape':
                                idxs_a_synth = new_order  # shuffled first half of the batch
                                idxs_b_synth = np.arange(new_order.shape[0],
                                                         2 * new_order.shape[0])  # second half of the batch
                            else:
                                raise NotImplementedError("Unexpected shape consistency value ")

                            # if this doesn't go through, something went wrong with the shuffling indexations
                            assert codedict["shapecode"][idxs_a_synth].allclose(codedict["shapecode"][idxs_b_synth])

                            losses['identity_contrastive_synthetic_shuffled'] = self.deca.id_loss(
                                overlay[idxs_a_synth],  # synthetic images of identities with reconstructed expressions
                                overlay[idxs_b_synth],  # synthetic images of identities with shuffled expressions
                                batch_size=idxs_a_synth.size,
                                ring_size=1) * self.deca.config.idw

                            losses['identity_contrastive_synthetic2real_shuffled'] = self.deca.id_loss(
                                images[idxs_a_synth],  # synthetic images of identities with reconstructed expressions
                                overlay[idxs_b_synth],  # synthetic images of identities with shuffled expressions
                                batch_size=idxs_a_synth.size,
                                ring_size=1) * self.deca.config.idw
                    elif ring_size > 2:
                        raise NotImplementedError("Contrastive loss does not support ring sizes > 2.")
        return losses


    def _compute_emonet_loss_wrapper(self, codedict, batch, training, testing, losses, metrics, prefix, image_key,
                                     with_grad, batch_size, ring_size):

        if self.emonet_loss is not None:

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

            # with torch.no_grad():
            # TODO: if expression shuffled, this needs to be changed, the input images no longer correspond

            images = codedict["images"]
            predicted_images = codedict[image_key]
            effective_bs = images.shape[0]

            if "ref_images_expression_idxs" in codedict.keys():
                # in case there was shuffling, this ensures that the proper images are used for emotion loss
                images_ = images[codedict["ref_images_expression_idxs"]]
            else:
                images_ = images
            effective_bs = images.shape[0]
            self._compute_emotion_loss(images_, predicted_images, losses, metrics, f"{prefix}",
                                       va, expr7,
                                       with_grad=with_grad,
                                       batch_size=effective_bs, ring_size=1)

            codedict[f"{prefix}_valence_input"] = self.emonet_loss.input_emotion['valence']
            codedict[f"{prefix}_arousal_input"] = self.emonet_loss.input_emotion['arousal']
            codedict[f"{prefix}_expression_input"] = self.emonet_loss.input_emotion[
                'expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification']
            codedict[f"{prefix}_valence_output"] = self.emonet_loss.output_emotion['valence']
            codedict[f"{prefix}_arousal_output"] = self.emonet_loss.output_emotion['arousal']
            codedict[f"{prefix}_expression_output"] = self.emonet_loss.output_emotion[
                'expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification']

            if 'emo_contrastive' in self.deca.config.keys() and self.deca.config.emo_contrastive:
                assert ring_size == 2 or ring_size == 1

                assert self.emonet_loss.trainable or (
                            hasattr(self.emonet_loss, 'clone_is_trainable') and self.emonet_lossclone_is_trainable)

                has_been_shuffled = 'new_order' in codedict.keys()

                # if self.deca.config.shape_constrain_type == 'shuffle_expression' and has_been_shuffled:
                #     new_order = codedict['new_order']
                #

                if self.deca.config.emo_contrastive in [True, "real", "both"]:
                    if ring_size == 2:

                        assert effective_bs % 2 == 0

                        if not isinstance(self.deca, ExpDECA):
                            raise NotImplementedError("Cross-ring emotion contrast means the ring has to be "
                                                      "expression based, not identity based. This is not guaranteed "
                                                      "for vanilla EMOCA (or its datasets).")
                        # we are taking this from the original batch dict because we do not care about the
                        # shuffled, duplicated samples (they don't have to be doubled)
                        images_0 = batch["image"][:, 0, ...]
                        images_1 = batch["image"][:, 1, ...]
                        self._compute_emotion_loss(images_0,  # real images of first expressions in the ring
                                                   images_1,  # real images of second expressions in the ring
                                                   losses, metrics, f"{prefix}_contrastive_real",
                                                   va, expr7, with_grad=self.deca.config.use_emonet_loss,
                                                   batch_size=images_0.shape[0], ring_size=1)
                    else:
                        print("[WARNING] Cannot compute real contrastive emotion loss because there is no ring!")

                if self.deca.config.emo_contrastive in [True, "synth", "both"]:

                    if ring_size == 2:
                        assert effective_bs % 2 == 0

                        idxs_a = torch.arange(0, images.shape[0], 2) # indices of first expressions within a ring
                        idxs_b = torch.arange(1, images.shape[0], 2) # indices of second expressions within a ring

                        if 'expression_constrain_type' in self.deca.config.keys() and \
                                self.deca.config.expression_constrain_type in ['exchange', 'same']:
                            # we can take all when identity has been exchange within rings
                            predicted_images_0 = predicted_images[idxs_a]
                            predicted_images_1 = predicted_images[idxs_b]
                            raise RuntimeError("This should work but it was never tested or intended. Make sure this works.")
                        else:
                            # if the batch was double otherwise (global shuffling) we only take the first half
                            # if batch_size * ring_size < effective_bs:
                            predicted_images_0 = predicted_images[0:batch_size * ring_size:2]
                            predicted_images_1 = predicted_images[1:batch_size * ring_size:2]

                        if not isinstance(self.deca, ExpDECA):
                            raise NotImplementedError("Cross-ring emotion contrast means the ring has to be "
                                                      "expression based, not identity based. This is not guaranteed "
                                                      "for vanilla EMOCA.")

                        self._compute_emotion_loss(predicted_images_0,
                                                   # rec images of first expressions in the ring
                                                   predicted_images_1,
                                                   # rec images of second expressions in the ring
                                                   losses, metrics, f"{prefix}_contrastive_synth",
                                                   va, expr7, with_grad=self.deca.config.use_emonet_loss,
                                                   batch_size=predicted_images_1.shape[0], ring_size=1)
                    else:
                        print("[WARNING] Cannot compute synthetic contrastive emotion loss because there is no ring!")

                    if has_been_shuffled:
                        new_order = codedict['new_order']
                        if self.deca.config.shape_constrain_type == 'shuffle_expression':
                            # this gets tricky, in this case the images are not duplicates -> we need all, but the second
                            # half's order is shuffled, so we need to be careful here
                            idxs_a_synth = new_order  # shuffled first half of the batch
                            idxs_b_synth = np.arange(new_order.shape[0],
                                                     2 * new_order.shape[0])  # second half of the batch
                        elif self.deca.config.shape_constrain_type == 'shuffle_shape':
                            idxs_a_synth = np.arange(new_order.shape[0])  # first half of the batch
                            idxs_b_synth = np.arange(new_order.shape[0],
                                                     2 * new_order.shape[0])  # second half of the batch

                        # if this doesn't go through, something went wrong with the shuffling indexations
                        assert codedict["expcode"][idxs_a_synth].allclose(codedict["expcode"][idxs_b_synth])

                        # the expressions at corresponding index positions of idxs_a_synth and idxs_b_synth should match now
                        self._compute_emotion_loss(predicted_images[idxs_a_synth],
                                                   # synthetic images of reconstructed expressions and corresponding identities
                                                   predicted_images[idxs_b_synth],
                                                   # synthetic images of reconstructed expressions and shuffled identities
                                                   losses, metrics, f"{prefix}_contrastive_synth_shuffled",
                                                   va, expr7,
                                                   with_grad=self.deca.config.use_emonet_loss and not self.deca._has_neural_rendering(),
                                                   batch_size=idxs_a_synth.size, ring_size=1)
                        
                        self._compute_emotion_loss(images[idxs_a_synth],
                                                   # synthetic images of reconstructed expressions and corresponding identities
                                                   predicted_images[idxs_b_synth],
                                                   # synthetic images of reconstructed expressions and shuffled identities
                                                   losses, metrics, f"{prefix}_contrastive_synth2real_shuffled",
                                                   va, expr7,
                                                   with_grad=self.deca.config.use_emonet_loss and not self.deca._has_neural_rendering(),
                                                   batch_size=idxs_a_synth.size,
                                                   ring_size=1)
                        

            if va is not None:
                codedict[f"{prefix}_valence_gt"] = va[:, 0]
                codedict[f"{prefix}_arousal_gt"] = va[:, 1]
            if expr7 is not None:
                codedict[f"{prefix}_expression_gt"] = expr7

            if self.deca._has_neural_rendering():
                assert 'emo_contrastive' not in self.deca.config.keys() or self.deca.config.emo_contrastive is False
                # TODO possible to make this more GPU efficient by not recomputing emotion for input image
                self._compute_emotion_loss(images, predicted_translated_image, losses, metrics, f"{prefix}_translated",
                                           va, expr7,
                                           with_grad=self.deca.config.use_emonet_loss and self.deca._has_neural_rendering(),
                                           batch_size=bs,
                                           ring_size=1)

                # codedict[f"{prefix}_valence_input"] = self.emonet_loss.input_emotion['valence']
                # codedict[f"{prefix}_arousal_input"] = self.emonet_loss.input_emotion['arousal']
                # codedict[f"{prefix}_expression_input"] = self.emonet_loss.input_emotion['expression']
                codedict[f"{prefix}_translated_valence_output"] = self.emonet_loss.output_emotion['valence']
                codedict[f"{prefix}_translated_arousal_output"] = self.emonet_loss.output_emotion['arousal']
                codedict[f"{prefix}_translated_expression_output"] = self.emonet_loss.output_emotion[
                    'expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification']
        return losses, metrics, codedict


    def _compute_loss(self, codedict, batch, training=True, testing=False) -> (dict, dict):
        #### ----------------------- Losses
        losses = {}
        metrics = {}

        predicted_landmarks = codedict["predicted_landmarks"]
        predicted_landmarks_mediapipe = codedict.get("predicted_landmarks_mediapipe", None)
        if "lmk" in codedict.keys():
            lmk = codedict["lmk"]
        else:
            lmk = None
        
        if "lmk_mp" in codedict.keys():
            lmk_mp = codedict["lmk_mp"]
        else:
            lmk_mp = None

        if "masks" in codedict.keys():
            masks = codedict["masks"]
        else:
            masks = None

        batch_size = codedict["predicted_images"].shape[0]

        use_geom_losses = 'use_geometric_losses_expression_exchange' in self.deca.config.keys() and \
            self.deca.config.use_geometric_losses_expression_exchange

        if training and ('expression_constrain_type' in self.deca.config.keys() \
            and ('expression_constrain_type' in self.deca.config.keys() and self.deca.config.expression_constrain_type == 'exchange') or
                         ( 'shape_constrain_type' in self.deca.config.keys() and
                           self.deca.config.shape_constrain_type in ['shuffle_expression', 'shuffle_shape'])) \
            and (self.deca.mode == DecaMode.COARSE or self.deca.config.train_coarse) \
            and (not use_geom_losses):
            if batch_size % 2 != 0:
                raise RuntimeError("The batch size should be even because it should have "
                                   f"got doubled in expression ring exchange. Instead it was odd: {batch_size}")
            # THIS IS DONE BECAUSE LANDMARK AND PHOTOMETRIC LOSSES MAKE NO SENSE FOR EXPRESSION EXCHANGE
            geom_losses_idxs = batch_size // 2

        else:
            geom_losses_idxs = batch_size

        predicted_images = codedict["predicted_images"]
        images = codedict["images"]
        lightcode = codedict["lightcode"]
        albedo = codedict["albedo"]
        mask_face_eye = codedict["mask_face_eye"]
        shapecode = codedict["shapecode"]
        expcode = codedict["expcode"]
        texcode = codedict["texcode"]
        ops = codedict["ops"]


        if self.mode == DecaMode.DETAIL:
            uv_texture = codedict["uv_texture"]
            uv_texture_gt = codedict["uv_texture_gt"]


        # this determines the configured batch size that is currently used (training, validation or testing)
        # the reason why this is important is because of potential multi-gpu training and loss functions (such as Barlow Twins)
        # that might need the full size of the batch (not just the chunk of the current GPU).
        if training:
            bs = self.learning_params.batch_size_train
            rs = self.learning_params.train_K
        else:
            if not testing:
                bs = self.learning_params.batch_size_val
                rs = self.learning_params.val_K
            else:
                bs = self.learning_params.batch_size_test
                rs = self.learning_params.test_K


        ## COARSE loss only
        if self.mode == DecaMode.COARSE or (self.mode == DecaMode.DETAIL and self.deca.config.train_coarse):

            # landmark losses (only useful if coarse model is being trained
            # if training or lmk is not None:
            if lmk is not None:
                # if self.deca.config.use_landmarks:
                #     d = losses
                # else:
                #     d = metrics
                d = self._metric_or_loss(losses, metrics, self.deca.config.use_landmarks)



                if self.deca.config.useWlmk:
                    d['landmark'] = \
                        lossfunc.weighted_landmark_loss(predicted_landmarks[:geom_losses_idxs, ...], lmk[:geom_losses_idxs, ...]) * self.deca.config.lmk_weight
                else:
                    d['landmark'] = \
                        lossfunc.landmark_loss(predicted_landmarks[:geom_losses_idxs, ...], lmk[:geom_losses_idxs, ...]) * self.deca.config.lmk_weight

                d = self._metric_or_loss(losses, metrics, 'use_eye_distance' not in self.deca.config.keys() or
                                         self.deca.config.use_eye_distance)
                # losses['eye_distance'] = lossfunc.eyed_loss(predicted_landmarks, lmk) * self.deca.config.lmk_weight * 2
                d['eye_distance'] = lossfunc.eyed_loss(predicted_landmarks[:geom_losses_idxs, ...],
                                                       lmk[:geom_losses_idxs, ...]) * self.deca.config.eyed
                d = self._metric_or_loss(losses, metrics, 'use_lip_distance' not in self.deca.config.keys() or
                                         self.deca.config.use_lip_distance)
                d['lip_distance'] = lossfunc.lipd_loss(predicted_landmarks[:geom_losses_idxs, ...],
                                                       lmk[:geom_losses_idxs, ...]) * self.deca.config.lipd

                d = self._metric_or_loss(losses, metrics, 'use_mouth_corner_distance' in self.deca.config.keys() and
                                         self.deca.config.use_mouth_corner_distance)
                d['mouth_corner_distance'] = lossfunc.mouth_corner_loss(predicted_landmarks[:geom_losses_idxs, ...],
                                                       lmk[:geom_losses_idxs, ...]) * self.deca.config.lipd

                if predicted_landmarks_mediapipe is not None and lmk_mp is not None:
                    use_mediapipe_landmarks = self.deca.config.get('use_mediapipe_landmarks', False) 
                    d = self._metric_or_loss(losses, metrics, use_mediapipe_landmarks)
                    d['landmark_mediapipe'] =lossfunc_mp.landmark_loss(predicted_landmarks_mediapipe[:geom_losses_idxs, ...], lmk_mp[:geom_losses_idxs, ...]) * self.deca.config.lmk_weight_mp

                    d = self._metric_or_loss(losses, metrics, self.deca.config.get('use_eye_distance_mediapipe', False) )
                    d['eye_distance_mediapipe'] = lossfunc_mp.eyed_loss(predicted_landmarks_mediapipe[:geom_losses_idxs, ...],
                                                        lmk_mp[:geom_losses_idxs, ...]) * self.deca.config.eyed_mp
                    d = self._metric_or_loss(losses, metrics,  self.deca.config.get('use_lip_distance_mediapipe', False) )
                    d['lip_distance_mediapipe'] = lossfunc_mp.lipd_loss(predicted_landmarks_mediapipe[:geom_losses_idxs, ...],
                                                        lmk_mp[:geom_losses_idxs, ...]) * self.deca.config.lipd_mp

                    d = self._metric_or_loss(losses, metrics, self.deca.config.get('use_mouth_corner_distance_mediapipe', False))
                    d['mouth_corner_distance_mediapipe'] = lossfunc_mp.mouth_corner_loss(predicted_landmarks_mediapipe[:geom_losses_idxs, ...],
                                                        lmk_mp[:geom_losses_idxs, ...]) * self.deca.config.lipd_mp


                #TODO: fix this on the next iteration lipd_loss
                # d['lip_distance'] = lossfunc.lipd_loss(predicted_landmarks, lmk) * self.deca.config.lipd

            # photometric loss
            # if training or masks is not None:
            if masks is not None:
                # if self.deca.config.use_photometric:
                #     d = losses
                # else:
                #     d = metrics
                # d['photometric_texture'] = (masks * (predicted_images - images).abs()).mean() * self.deca.config.photow

                photometric = masks[:geom_losses_idxs, ...] * ((predicted_images[:geom_losses_idxs, ...] - images[:geom_losses_idxs, ...]).abs())

                if 'photometric_normalization' not in self.deca.config.keys() or self.deca.config.photometric_normalization == 'mean':
                    photometric = photometric.mean()
                elif self.deca.config.photometric_normalization == 'rel_mask_value':
                    photometric = photometric * masks[:geom_losses_idxs, ...].mean(dim=tuple(range(1,masks.ndim)), keepdim=True)
                    photometric = photometric.mean()
                elif self.deca.config.photometric_normalization == 'neg_rel_mask_value':
                    mu = 1. - masks[:geom_losses_idxs, ...].mean(dim=tuple(range(1,masks.ndim)), keepdim=True)
                    photometric = photometric * mu
                    photometric = photometric.mean()
                elif self.deca.config.photometric_normalization == 'inv_rel_mask_value':
                    mu = 1./ masks[:geom_losses_idxs, ...].mean(dim=tuple(range(1,masks.ndim)), keepdim=True)
                    photometric = photometric * mu
                    photometric = photometric.mean()
                elif self.deca.config.photometric_normalization == 'abs_mask_value':
                    photometric = photometric * masks[:geom_losses_idxs, ...].sum(dim=tuple(range(1,masks.ndim)), keepdim=True)
                    photometric = photometric.mean()
                else:
                    raise ValueError(f"Invalid photometric loss normalization: '{self.deca.config.photometric_normalization}'")

                self._metric_or_loss(losses, metrics, self.deca.config.use_photometric)['photometric_texture'] = \
                    photometric * self.deca.config.photow

                if self.deca.vgg_loss is not None:
                    vggl, _ = self.deca.vgg_loss(
                        masks[:geom_losses_idxs, ...] * images[:geom_losses_idxs, ...], # masked input image
                        masks[:geom_losses_idxs, ...] * predicted_images[:geom_losses_idxs, ...], # masked output image
                    )
                    self._metric_or_loss(losses, metrics, self.deca.config.use_vgg)['vgg'] = vggl * self.deca.config.vggw

                if self.deca._has_neural_rendering():
                    predicted_translated_image = codedict["predicted_translated_image"]
                    photometric_translated = (masks[:geom_losses_idxs, ...] * (
                            predicted_translated_image[:geom_losses_idxs, ...] -
                            images[:geom_losses_idxs, ...]).abs()).mean() * self.deca.config.photow
                    if self.deca.config.use_photometric:
                        losses['photometric_translated_texture'] = photometric_translated
                    else:
                        metrics['photometric_translated_texture'] = photometric_translated

                    if self.deca.vgg_loss is not None:
                        vggl, _ = self.deca.vgg_loss(
                            masks[:geom_losses_idxs, ...] * images[:geom_losses_idxs, ...],  # masked input image
                            masks[:geom_losses_idxs, ...] * predicted_translated_image[:geom_losses_idxs, ...],
                            # masked output image
                        )
                        self._metric_or_loss(losses, metrics, self.deca.config.use_vgg)['vgg_translated'] = vggl * self.deca.config.vggw

            else:
                raise ValueError("Is this line ever reached?")


            losses = self._compute_id_loss(codedict, batch, training, testing, losses, batch_size=bs, ring_size=rs)

            losses['shape_reg'] = (torch.sum(shapecode ** 2) / 2) * self.deca.config.shape_reg
            losses['expression_reg'] = (torch.sum(expcode ** 2) / 2) * self.deca.config.exp_reg
            losses['tex_reg'] = (torch.sum(texcode ** 2) / 2) * self.deca.config.tex_reg
            losses['light_reg'] = ((torch.mean(lightcode, dim=2)[:, :,
                                    None] - lightcode) ** 2).mean() * self.deca.config.light_reg

            if 'original_code' in codedict.keys():
                # original jaw pose regularization
                if self.deca.config.get('exp_deca_jaw_pose', False) and \
                    'deca_jaw_reg' in self.deca.config.keys() and self.deca.config.deca_jaw_reg > 0:
                    jaw_pose_orig = codedict['original_code']['pose'][:, 3:]
                    jaw_pose = codedict['posecode'][..., 3:]
                    deca_jaw_pose_reg = (torch.sum((jaw_pose - jaw_pose_orig) ** 2) / 2) * self.deca.config.deca_jaw_reg
                    losses['deca_jaw_pose_reg'] = deca_jaw_pose_reg

                if self.deca.config.get('exp_deca_global_pose', False) and \
                    'deca_global_reg' in self.deca.config.keys() and self.deca.config.deca_global_reg > 0:
                    global_pose_orig = codedict['original_code']['pose'][:, :3]
                    global_pose = codedict['posecode'][..., :3]
                    global_pose_reg = (torch.sum((global_pose - global_pose_orig) ** 2) / 2) * self.deca.config.deca_global_reg
                    losses['deca_global_pose_reg'] = global_pose_reg

                # original expression regularization
                if 'deca_expression_reg' in self.deca.config.keys() and self.deca.config.deca_expression_reg > 0:
                    expression_orig = codedict['original_code']['exp']
                    expression = codedict['expcode']
                    deca_expression_reg = (torch.sum((expression - expression_orig) ** 2) / 2) * self.deca.config.deca_expression_reg
                    losses['deca_expression_reg'] = deca_expression_reg


            losses, metrics, codedict = self._compute_emonet_loss_wrapper(codedict, batch, training, testing, losses, metrics,
                                                                 prefix="coarse", image_key="predicted_images",
                                                                with_grad=self.deca.config.use_emonet_loss and not self.deca._has_neural_rendering(),
                                                                batch_size=bs, ring_size=rs)
            if self.deca._has_neural_rendering():
                losses, metrics, codedict = self._compute_emonet_loss_wrapper(codedict, batch, training, testing, losses, metrics,
                                                                     prefix="coarse_translated", image_key="predicted_translated_image",
                                                                     with_grad=self.deca.config.use_emonet_loss and self.deca._has_neural_rendering(),
                                                                     batch_size=bs, ring_size=rs
                                                                     )

            if self.au_loss is not None:
                # with torch.no_grad():

                self._compute_au_loss(images, predicted_images, losses, metrics, "coarse",
                                      au=None,
                                      with_grad=self.deca.config.au_loss.use_as_loss and not self.deca._has_neural_rendering())
                if self.deca._has_neural_rendering():
                    self._compute_au_loss(images, predicted_translated_image, losses, metrics, "coarse",
                                          au=None,
                                          with_grad=self.deca.config.au_loss.use_as_loss and self.deca._has_neural_rendering())

            if self.lipread_loss is not None:
                # with torch.no_grad():

                self._compute_lipread_loss(images, predicted_images, lmk, predicted_landmarks, losses, metrics, "coarse",
                                      with_grad=self.deca.config.lipread_loss.use_as_loss and not self.deca._has_neural_rendering())
                if self.deca._has_neural_rendering():
                    self._compute_lipread_loss(images, predicted_translated_image, 
                                        lmk, predicted_landmarks,
                                          losses, metrics, "coarse",
                                          with_grad=self.deca.config.lipread_loss.use_as_loss and self.deca._has_neural_rendering())

        ## DETAIL loss only
        if self.mode == DecaMode.DETAIL:
            predicted_detailed_image = codedict["predicted_detailed_image"]
            uv_z = codedict["uv_z"] # UV displacement map
            uv_shading = codedict["uv_shading"]
            uv_vis_mask = codedict["uv_vis_mask"] # uv_mask of what is visible

            photometric_detailed = (masks[:geom_losses_idxs, ...] * (
                    predicted_detailed_image[:geom_losses_idxs, ...] -
                    images[:geom_losses_idxs, ...]).abs()).mean() * self.deca.config.photow

            if self.deca.config.use_detailed_photo:
                losses['photometric_detailed_texture'] = photometric_detailed
            else:
                metrics['photometric_detailed_texture'] = photometric_detailed

            if self.deca.vgg_loss is not None:
                vggl, _ = self.deca.vgg_loss(
                    masks[:geom_losses_idxs, ...] * images[:geom_losses_idxs, ...],  # masked input image
                    masks[:geom_losses_idxs, ...] * predicted_detailed_image[:geom_losses_idxs, ...],
                    # masked output image
                )
                self._metric_or_loss(losses, metrics, self.deca.config.use_vgg)['vgg_detailed'] = vggl * self.deca.config.vggw

            if self.deca._has_neural_rendering():
                predicted_detailed_translated_image = codedict["predicted_detailed_translated_image"]
                photometric_detailed_translated = (masks[:geom_losses_idxs, ...] * (
                        predicted_detailed_translated_image[:geom_losses_idxs, ...] - images[:geom_losses_idxs,
                                                                           ...]).abs()).mean() * self.deca.config.photow
                if self.deca.config.use_detailed_photo:
                    losses['photometric_translated_detailed_texture'] = photometric_detailed_translated
                else:
                    metrics['photometric_translated_detailed_texture'] = photometric_detailed_translated

                if self.deca.vgg_loss is not None:
                    vggl, _ = self.deca.vgg_loss(
                        masks[:geom_losses_idxs, ...] * images[:geom_losses_idxs, ...],  # masked input image
                        masks[:geom_losses_idxs, ...] * predicted_detailed_translated_image[:geom_losses_idxs, ...],
                        # masked output image
                    )
                    self._metric_or_loss(losses, metrics, self.deca.config.use_vgg)[
                        'vgg_detailed_translated'] =  vggl * self.deca.config.vggw


            losses, metrics, codedict = self._compute_emonet_loss_wrapper(codedict, batch, training, testing, losses, metrics,
                                                                 prefix="detail", image_key = "predicted_detailed_image",
                                                                 with_grad=self.deca.config.use_emonet_loss and not self.deca._has_neural_rendering(),
                                                                 batch_size=bs, ring_size=rs)
            if self.deca._has_neural_rendering():
                losses, metrics, codedict = self._compute_emonet_loss_wrapper(codedict, batch, training, testing, losses, metrics,
                                                                     prefix="detail_translated",
                                                                     image_key="predicted_detailed_translated_image",
                                                                     with_grad=self.deca.config.use_emonet_loss and self.deca._has_neural_rendering(),
                                                                     batch_size=bs, ring_size=rs)

            # if self.emonet_loss is not None:
            #     self._compute_emotion_loss(images, predicted_detailed_image, losses, metrics, "detail",
            #                                with_grad=self.deca.config.use_emonet_loss and not self.deca._has_neural_rendering(),
            #                                batch_size=bs, ring_size=rs)
                # codedict["detail_valence_input"] = self.emonet_loss.input_emotion['valence']
                # codedict["detail_arousal_input"] = self.emonet_loss.input_emotion['arousal']
                # codedict["detail_expression_input"] = self.emonet_loss.input_emotion['expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification']
                # codedict["detail_valence_output"] = self.emonet_loss.output_emotion['valence']
                # codedict["detail_arousal_output"] = self.emonet_loss.output_emotion['arousal']
                # codedict["detail_expression_output"] = self.emonet_loss.output_emotion['expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification']
                #
                # if va is not None:
                #     codedict["detail_valence_gt"] = va[:,0]
                #     codedict["detail_arousal_gt"] = va[:,1]
                # if expr7 is not None:
                #     codedict["detail_expression_gt"] = expr7


                # if self.deca._has_neural_rendering():


                    # #TODO possible to make this more GPU efficient by not recomputing emotion for input image
                    # self._compute_emotion_loss(images, predicted_detailed_translated_image,
                    #                            losses, metrics, "detail_translated",
                    #                            va, expr7,
                    #                            with_grad= self.deca.config.use_emonet_loss and self.deca._has_neural_rendering(),
                    #                            batch_size=bs, ring_size=rs)
                    #
                    # # codedict["coarse_valence_input"] = self.emonet_loss.input_emotion['valence']
                    # # codedict["coarse_arousal_input"] = self.emonet_loss.input_emotion['arousal']
                    # # codedict["coarse_expression_input"] = self.emonet_loss.input_emotion['expression']
                    # codedict["detail_translated_valence_output"] = self.emonet_loss.output_emotion['valence']
                    # codedict["detail_translated_arousal_output"] = self.emonet_loss.output_emotion['arousal']
                    # codedict["detail_translated_expression_output"] = self.emonet_loss.output_emotion['expression' if 'expression' in self.emonet_loss.input_emotion.keys() else 'expr_classification']

            if self.au_loss is not None:
                self._compute_au_loss(images, predicted_images, losses, metrics, "detail",
                                      au=None,
                                      with_grad=self.deca.config.au_loss.use_as_loss and not self.deca._has_neural_rendering())

                if self.deca._has_neural_rendering():
                    self._compute_au_loss(images, predicted_detailed_translated_image, losses, metrics, "detail",
                                          au=None,
                                          with_grad=self.deca.config.au_loss.use_as_loss and self.deca._has_neural_rendering())

            for pi in range(3):  # self.deca.face_attr_mask.shape[0]):
                if self.deca.config.sfsw[pi] != 0:
                    # if pi==0:
                    new_size = 256
                    # else:
                    #     new_size = 128
                    # if self.deca.config.uv_size != 256:
                    #     new_size = 128
                    uv_texture_patch = F.interpolate(
                        uv_texture[:geom_losses_idxs, :, self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                        self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]],
                        [new_size, new_size], mode='bilinear')
                    uv_texture_gt_patch = F.interpolate(
                        uv_texture_gt[:geom_losses_idxs, :, self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                        self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]], [new_size, new_size],
                        mode='bilinear')
                    uv_vis_mask_patch = F.interpolate(
                        uv_vis_mask[:geom_losses_idxs, :, self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                        self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]],
                        [new_size, new_size], mode='bilinear')

                    detail_l1 = (uv_texture_patch * uv_vis_mask_patch - uv_texture_gt_patch * uv_vis_mask_patch).abs().mean() * \
                                                        self.deca.config.sfsw[pi]
                    if self.deca.config.use_detail_l1 and not self.deca._has_neural_rendering():
                        losses['detail_l1_{}'.format(pi)] = detail_l1
                    else:
                        metrics['detail_l1_{}'.format(pi)] = detail_l1

                    if self.deca.config.use_detail_mrf and not self.deca._has_neural_rendering():
                        mrf = self.deca.perceptual_loss(uv_texture_patch * uv_vis_mask_patch,
                                                        uv_texture_gt_patch * uv_vis_mask_patch) * \
                                                        self.deca.config.sfsw[pi] * self.deca.config.mrfwr
                        losses['detail_mrf_{}'.format(pi)] = mrf
                    else:
                        with torch.no_grad():
                            mrf = self.deca.perceptual_loss(uv_texture_patch * uv_vis_mask_patch,
                                                            uv_texture_gt_patch * uv_vis_mask_patch) * \
                                  self.deca.config.sfsw[pi] * self.deca.config.mrfwr
                            metrics['detail_mrf_{}'.format(pi)] = mrf

                    if self.deca._has_neural_rendering():
                        # raise NotImplementedError("Gotta implement the texture extraction first.")
                        translated_uv_texture = codedict["translated_uv_texture"]
                        translated_uv_texture_patch = F.interpolate(
                            translated_uv_texture[:geom_losses_idxs, :,
                            self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                            self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]],
                            [new_size, new_size], mode='bilinear')

                        translated_detail_l1 = (translated_uv_texture_patch * uv_vis_mask_patch
                                     - uv_texture_gt_patch * uv_vis_mask_patch).abs().mean() * \
                                    self.deca.config.sfsw[pi]

                        if self.deca.config.use_detail_l1:
                            losses['detail_translated_l1_{}'.format(pi)] = translated_detail_l1
                        else:
                            metrics['detail_translated_l1_{}'.format(pi)] = translated_detail_l1

                        if self.deca.config.use_detail_mrf:
                            translated_mrf = self.deca.perceptual_loss(translated_uv_texture_patch * uv_vis_mask_patch,
                                                            uv_texture_gt_patch * uv_vis_mask_patch) * \
                                  self.deca.config.sfsw[pi] * self.deca.config.mrfwr
                            losses['detail_translated_mrf_{}'.format(pi)] = translated_mrf
                        else:
                            with torch.no_grad():
                                mrf = self.deca.perceptual_loss(translated_uv_texture_patch * uv_vis_mask_patch,
                                                                uv_texture_gt_patch * uv_vis_mask_patch) * \
                                      self.deca.config.sfsw[pi] * self.deca.config.mrfwr
                                metrics['detail_translated_mrf_{}'.format(pi)] = mrf
                # Old piece of debug code. Good to delete.
                # if pi == 2:
                #     uv_texture_gt_patch_ = uv_texture_gt_patch
                #     uv_texture_patch_ = uv_texture_patch
                #     uv_vis_mask_patch_ = uv_vis_mask_patch

            losses['z_reg'] = torch.mean(uv_z.abs()) * self.deca.config.zregw
            losses['z_diff'] = lossfunc.shading_smooth_loss(uv_shading) * self.deca.config.zdiffw
            nonvis_mask = (1 - util.binary_erosion(uv_vis_mask))
            losses['z_sym'] = (nonvis_mask * (uv_z - torch.flip(uv_z, [-1]).detach()).abs()).sum() * self.deca.config.zsymw

        if self.emotion_mlp is not None:# and not testing:
            mlp_losses, mlp_metrics = self.emotion_mlp.compute_loss(
                codedict, batch, training=training, pred_prefix="emo_mlp_")
            for key in mlp_losses.keys():
                if key in losses.keys():
                    raise RuntimeError(f"Duplicate loss label {key}")
                losses[key] = self.deca.config.mlp_emotion_predictor_weight * mlp_losses[key]
            for key in mlp_metrics.keys():
                if key in metrics.keys():
                    raise RuntimeError(f"Duplicate metric label {key}")
                # let's report the metrics (which are a superset of losses when it comes to EmoMLP) without the weight,
                # it's hard to plot the metrics otherwise
                metrics[key] = mlp_metrics[key]
                # metrics[key] = self.deca.config.mlp_emotion_predictor_weight * mlp_metrics[key]

        # else:
        #     uv_texture_gt_patch_ = None
        #     uv_texture_patch_ = None
        #     uv_vis_mask_patch_ = None

        return losses, metrics

    def compute_loss(self, values, batch, training=True, testing=False) -> dict:
        """
        The function used to compute the loss on a training batch.
        :
        training should be set to true when calling from training_step only
        """
        losses, metrics = self._compute_loss(values, batch, training=training, testing=testing)

        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        # losses['all_loss'] = all_loss
        losses = {'loss_' + key: value for key, value in losses.items()} # add prefix loss for better logging
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
        """
        Training step override of pytorch lightning module. It makes the encoding, decoding passes, computes the loss and logs the losses/visualizations. 
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        :batch_idx batch index
        """
        with torch.no_grad():
            training = False
            values = self.encode(batch, training=training)
            values = self.decode(values, training=training)
            losses_and_metrics = self.compute_loss(values, batch, training=training)
        #### self.log_dict(losses_and_metrics, on_step=False, on_epoch=True)
        # prefix = str(self.mode.name).lower()
        prefix = self._get_logging_prefix()

        # if dataloader_idx is not None:
        #     dataloader_str = str(dataloader_idx) + "_"
        # else:
        dataloader_str = ''

        stage_str = dataloader_str + 'val_'

        # losses_and_metrics_to_log = {prefix + dataloader_str +'_val_' + key: value.detach().cpu() for key, value in losses_and_metrics.items()}
        # losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach() for key, value in losses_and_metrics.items()}
        losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach().cpu().item() for key, value in losses_and_metrics.items()}
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = self.current_epoch
        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        # log val_loss also without any prefix for a model checkpoint to track it
        losses_and_metrics_to_log[stage_str + 'loss'] = losses_and_metrics_to_log[prefix + '_' + stage_str + 'loss']

        losses_and_metrics_to_log[prefix + '_' + stage_str + 'step'] = self.global_step
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'batch_idx'] = batch_idx
        losses_and_metrics_to_log[stage_str + 'step'] = self.global_step
        losses_and_metrics_to_log[stage_str + 'batch_idx'] = batch_idx

        losses_and_metrics_to_log[prefix + '_' + stage_str + 'mem_usage'] = self.process.memory_info().rss
        losses_and_metrics_to_log[stage_str + 'mem_usage'] = self.process.memory_info().rss
        # self._val_to_be_logged(losses_and_metrics_to_log)


        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch # recommended

        if self.trainer.is_global_zero:
            if self.deca.config.val_vis_frequency > 0:
                if batch_idx % self.deca.config.val_vis_frequency == 0:
                    uv_detail_normals = None
                    if 'uv_detail_normals' in values.keys():
                        uv_detail_normals = values['uv_detail_normals']
                    visualizations, grid_image = self._visualization_checkpoint(values['verts'], values['trans_verts'], values['ops'],
                                                   uv_detail_normals, values, batch_idx, stage_str[:-1], prefix)
                    vis_dict = self._create_visualizations_to_log(stage_str[:-1], visualizations, values, batch_idx, indices=0, dataloader_idx=dataloader_idx)
                    # image = Image(grid_image, caption="full visualization")
                    # vis_dict[prefix + '_val_' + "visualization"] = image
                    if isinstance(self.logger, WandbLogger):
                        self.logger.log_metrics(vis_dict)

        return None

    def _get_logging_prefix(self):
        prefix = self.stage_name + str(self.mode.name).lower()
        return prefix

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Testing step override of pytorch lightning module. It makes the encoding, decoding passes, computes the loss and logs the losses/visualizations
        without gradient  
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        :batch_idx batch index
        """
        prefix = self._get_logging_prefix()
        losses_and_metrics_to_log = {}

        # if dataloader_idx is not None:
        #     dataloader_str = str(dataloader_idx) + "_"
        # else:
        dataloader_str = ''
        stage_str = dataloader_str + 'test_'

        with torch.no_grad():
            training = False
            testing = True
            values = self.encode(batch, training=training)
            values = self.decode(values, training=training)
            if 'mask' in batch.keys():
                losses_and_metrics = self.compute_loss(values, batch, training=False, testing=testing)
                # losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach().cpu() for key, value in losses_and_metrics.items()}
                losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach().cpu().item() for key, value in losses_and_metrics.items()}
            else:
                losses_and_metric = None

        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = self.current_epoch
        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'step'] = torch.tensor(self.global_step, device=self.device)
        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'batch_idx'] = torch.tensor(batch_idx, device=self.device)
        # losses_and_metrics_to_log[stage_str + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        # losses_and_metrics_to_log[stage_str + 'step'] = torch.tensor(self.global_step, device=self.device)
        # losses_and_metrics_to_log[stage_str + 'batch_idx'] = torch.tensor(batch_idx, device=self.device)
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'epoch'] = self.current_epoch
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'step'] = self.global_step
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'batch_idx'] = batch_idx
        losses_and_metrics_to_log[prefix + '_' + stage_str + 'mem_usage'] = self.process.memory_info().rss
        losses_and_metrics_to_log[stage_str + 'epoch'] = self.current_epoch
        losses_and_metrics_to_log[stage_str + 'step'] = self.global_step
        losses_and_metrics_to_log[stage_str + 'batch_idx'] = batch_idx
        losses_and_metrics_to_log[stage_str + 'mem_usage'] = self.process.memory_info().rss

        if self.logger is not None:
            # self.logger.log_metrics(losses_and_metrics_to_log)
            self.log_dict(losses_and_metrics_to_log, sync_dist=True, on_step=False, on_epoch=True)

        # if self.global_step % 200 == 0:
        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']

        if self.deca.config.test_vis_frequency > 0:
            # Log visualizations every once in a while
            if batch_idx % self.deca.config.test_vis_frequency == 0:
                # if self.trainer.is_global_zero:
                visualizations, grid_image = self._visualization_checkpoint(values['verts'], values['trans_verts'], values['ops'],
                                               uv_detail_normals, values, self.global_step, stage_str[:-1], prefix)
                visdict = self._create_visualizations_to_log(stage_str[:-1], visualizations, values, batch_idx, indices=0, dataloader_idx=dataloader_idx)
                self.logger.log_metrics(visdict)
        return None

    @property
    def process(self):
        if not hasattr(self,"process_"):
            import psutil
            self.process_ = psutil.Process(os.getpid())
        return self.process_


    def training_step(self, batch, batch_idx, *args, **kwargs): #, debug=True):
        """
        Training step override of pytorch lightning module. It makes the encoding, decoding passes, computes the loss and logs the losses/visualizations. 
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        :batch_idx batch index
        """
        values = self.encode(batch, training=True)
        values = self.decode(values, training=True)
        losses_and_metrics = self.compute_loss(values, batch, training=True)

        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']

        # prefix = str(self.mode.name).lower()
        prefix = self._get_logging_prefix()
        # losses_and_metrics_to_log = {prefix + '_train_' + key: value.detach().cpu() for key, value in losses_and_metrics.items()}
        # losses_and_metrics_to_log = {prefix + '_train_' + key: value.detach() for key, value in losses_and_metrics.items()}
        losses_and_metrics_to_log = {prefix + '_train_' + key: value.detach().cpu().item() for key, value in losses_and_metrics.items()}
        # losses_and_metrics_to_log[prefix + '_train_' + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        losses_and_metrics_to_log[prefix + '_train_' + 'epoch'] = self.current_epoch
        losses_and_metrics_to_log[prefix + '_train_' + 'step'] = self.global_step
        losses_and_metrics_to_log[prefix + '_train_' + 'batch_idx'] = batch_idx
        losses_and_metrics_to_log[prefix + '_' + "train_" + 'mem_usage'] = self.process.memory_info().rss

        # losses_and_metrics_to_log['train_' + 'epoch'] = torch.tensor(self.current_epoch, device=self.device)
        losses_and_metrics_to_log['train_' + 'epoch'] = self.current_epoch
        losses_and_metrics_to_log['train_' + 'step'] = self.global_step
        losses_and_metrics_to_log['train_' + 'batch_idx'] = batch_idx

        losses_and_metrics_to_log["train_" + 'mem_usage'] = self.process.memory_info().rss

        # log loss also without any prefix for a model checkpoint to track it
        losses_and_metrics_to_log['loss'] = losses_and_metrics_to_log[prefix + '_train_loss']

        if self.logger is not None:
            self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        if self.deca.config.train_vis_frequency > 0:
            if self.global_step % self.deca.config.train_vis_frequency == 0:
                if self.trainer.is_global_zero:
                    visualizations, grid_image = self._visualization_checkpoint(values['verts'], values['trans_verts'], values['ops'],
                                                   uv_detail_normals, values, batch_idx, "train", prefix)
                    visdict = self._create_visualizations_to_log('train', visualizations, values, batch_idx, indices=0)

                    if isinstance(self.logger, WandbLogger):
                        self.logger.log_metrics(visdict)#, step=self.global_step)
                        # self.log_dict(visdict, sync_dist=True)

 
        # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=False) # log per step
        # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True) # log per both
        # return losses_and_metrics
        return losses_and_metrics['loss']


    ### STEP ENDS ARE PROBABLY NOT NECESSARY BUT KEEP AN EYE ON THEM IF MULI-GPU TRAINING DOESN'T WORK
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


    def vae_2_str(self, valence=None, arousal=None, affnet_expr=None, expr7=None, prefix=""):
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


    def _create_visualizations_to_log(self, stage, visdict, values, step, indices=None,
                                      dataloader_idx=None, output_dir=None):
        mode_ = str(self.mode.name).lower()
        prefix = self._get_logging_prefix()

        output_dir = output_dir or self.inout_params.full_run_dir

        log_dict = {}
        for key in visdict.keys():
            images = _torch_image2np(visdict[key])
            if images.dtype == np.float32 or images.dtype == np.float64 or images.dtype == np.float16:
                images = np.clip(images, 0, 1)
            if indices is None:
                indices = np.arange(images.shape[0])
            if isinstance(indices, int):
                indices = [indices,]
            if isinstance(indices, str) and indices == 'all':
                image = np.concatenate([images[i] for i in range(images.shape[0])], axis=1)
                savepath = Path(f'{output_dir}/{prefix}_{stage}/{key}/{self.current_epoch:04d}_{step:04d}_all.png')
                # im2log = Image(image, caption=key)
                if isinstance(self.logger, WandbLogger):
                    im2log = _log_wandb_image(savepath, image)
                else:
                    im2log = _log_array_image(savepath, image)
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
                                caption += self.vae_2_str(
                                    values[mode_ + "_valence_input"][i].detach().cpu().item(),
                                    values[mode_ + "_arousal_input"][i].detach().cpu().item(),
                                    np.argmax(values[mode_ + "_expression_input"][i].detach().cpu().numpy()),
                                    prefix="emonet") + "\n"
                            if 'va' in values.keys() and mode_ + "valence_gt" in values.keys():
                                # caption += self.vae_2_str(
                                #     values[mode_ + "_valence_gt"][i].detach().cpu().item(),
                                #     values[mode_ + "_arousal_gt"][i].detach().cpu().item(),
                                caption += self.vae_2_str(
                                    values[mode_ + "valence_gt"][i].detach().cpu().item(),
                                    values[mode_ + "arousal_gt"][i].detach().cpu().item(),
                                    prefix="gt") + "\n"
                            if 'expr7' in values.keys() and mode_ + "_expression_gt" in values.keys():
                                caption += "\n" + self.vae_2_str(
                                    expr7=values[mode_ + "_expression_gt"][i].detach().cpu().numpy(),
                                    prefix="gt") + "\n"
                            if 'affectnetexp' in values.keys() and mode_ + "_expression_gt" in values.keys():
                                caption += "\n" + self.vae_2_str(
                                    affnet_expr=values[mode_ + "_expression_gt"][i].detach().cpu().numpy(),
                                    prefix="gt") + "\n"
                        elif 'geometry_detail' in key:
                            if "emo_mlp_valence" in values.keys():
                                caption += self.vae_2_str(
                                    values["emo_mlp_valence"][i].detach().cpu().item(),
                                    values["emo_mlp_arousal"][i].detach().cpu().item(),
                                    prefix="mlp")
                            if 'emo_mlp_expr_classification' in values.keys():
                                caption += "\n" + self.vae_2_str(
                                    affnet_expr=values["emo_mlp_expr_classification"][i].detach().cpu().argmax().numpy(),
                                    prefix="mlp") + "\n"
                        elif key == 'output_images_' + mode_:
                            if mode_ + "_valence_output" in values.keys():
                                caption += self.vae_2_str(values[mode_ + "_valence_output"][i].detach().cpu().item(),
                                                                 values[mode_ + "_arousal_output"][i].detach().cpu().item(),
                                                                 np.argmax(values[mode_ + "_expression_output"][i].detach().cpu().numpy())) + "\n"

                        elif key == 'output_translated_images_' + mode_:
                            if mode_ + "_translated_valence_output" in values.keys():
                                caption += self.vae_2_str(values[mode_ + "_translated_valence_output"][i].detach().cpu().item(),
                                                                 values[mode_ + "_translated_arousal_output"][i].detach().cpu().item(),
                                                                 np.argmax(values[mode_ + "_translated_expression_output"][i].detach().cpu().numpy())) + "\n"


                        # elif key == 'output_images_detail':
                        #     caption += "\n" + self.vae_2_str(values["detail_output_valence"][i].detach().cpu().item(),
                        #                                  values["detail_output_valence"][i].detach().cpu().item(),
                        #                                  np.argmax(values["detail_output_expression"][
                        #                                                i].detach().cpu().numpy()))
                    savepath = Path(f'{output_dir}/{prefix}_{stage}/{key}/{self.current_epoch:04d}_{step:04d}_{i:02d}.png')
                    image = images[i]
                    # im2log = Image(image, caption=caption)
                    if isinstance(self.logger, WandbLogger):
                        im2log = _log_wandb_image(savepath, image, caption)
                    elif self.logger is not None:
                        im2log = _log_array_image(savepath, image, caption)
                    else:
                        im2log = _log_array_image(None, image, caption)
                    name = prefix + "_" + stage + "_" + key
                    if dataloader_idx is not None:
                        name += "/dataloader_idx_" + str(dataloader_idx)
                    log_dict[name] = im2log
        return log_dict

    def _visualization_checkpoint(self, verts, trans_verts, ops, uv_detail_normals, additional, batch_idx, stage, prefix,
                                  save=False):
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

        if 'predicted_translated_image' in additional.keys() and additional['predicted_translated_image'] is not None:
            visdict['output_translated_images_coarse'] = additional['predicted_translated_image'][visind]

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

        if 'predicted_detailed_translated_image' in additional.keys() and additional['predicted_detailed_translated_image'] is not None:
            visdict['output_translated_images_detail'] = additional['predicted_detailed_translated_image'][visind]

        if 'shape_detail_images' in additional.keys():
            visdict['shape_detail_images'] = additional['shape_detail_images'][visind]

        if 'uv_detail_normals' in additional.keys():
            visdict['uv_detail_normals'] = additional['uv_detail_normals'][visind] * 0.5 + 0.5

        if 'uv_texture_patch' in additional.keys():
            visdict['uv_texture_patch'] = additional['uv_texture_patch'][visind]

        if 'uv_texture_gt' in additional.keys():
            visdict['uv_texture_gt'] = additional['uv_texture_gt'][visind]

        if 'translated_uv_texture' in additional.keys() and additional['translated_uv_texture'] is not None:
            visdict['translated_uv_texture'] = additional['translated_uv_texture'][visind]

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

    def _get_trainable_parameters(self):
        trainable_params = []
        if self.mode == DecaMode.COARSE:
            trainable_params += self.deca._get_coarse_trainable_parameters()
        elif self.mode == DecaMode.DETAIL:
            trainable_params += self.deca._get_detail_trainable_parameters()
        else:
            raise ValueError(f"Invalid deca mode: {self.mode}")

        if self.emotion_mlp is not None:
            trainable_params += list(self.emotion_mlp.parameters())

        if self.emonet_loss is not None:
            trainable_params += self.emonet_loss._get_trainable_params()

        if self.deca.id_loss is not None:
            trainable_params += self.deca.id_loss._get_trainable_params()

        return trainable_params


    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print("Configuring optimizer")

        trainable_params = self._get_trainable_parameters()

        if self.learning_params.optimizer == 'Adam':
            self.deca.opt = torch.optim.Adam(
                trainable_params,
                lr=self.learning_params.learning_rate,
                amsgrad=False)
        elif self.config.learning.optimizer == 'AdaBound':
            self.deca.opt = adabound.AdaBound(
                trainable_params,
                lr=self.config.learning.learning_rate,
                final_lr=self.config.learning.final_learning_rate
            )
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
    """
    The original DECA class which contains the encoders, FLAME decoder and the detail decoder.
    """

    def __init__(self, config):
        """
        :config corresponds to a model_params from DecaModule
        """
        super().__init__()
        
        # ID-MRF perceptual loss (kept here from the original DECA implementation)
        self.perceptual_loss = None
        
        # Face Recognition loss
        self.id_loss = None

        # VGG feature loss
        self.vgg_loss = None
        
        self._reconfigure(config)
        self._reinitialize()

    def get_input_image_size(self): 
        return (self.config.image_size, self.config.image_size)

    def _reconfigure(self, config):
        self.config = config
        
        self.n_param = config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        # identity-based detail code 
        self.n_detail = config.n_detail
        # emotion-based detail code (deprecated, not use by DECA or EMOCA)
        self.n_detail_emo = config.n_detail_emo if 'n_detail_emo' in config.keys() else 0

        # count the size of the conidition vector
        if 'detail_conditioning' in self.config.keys():
            self.n_cond = 0
            if 'globalpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'jawpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'identity' in self.config.detail_conditioning:
                self.n_cond += config.n_shape
            if 'expression' in self.config.detail_conditioning:
                self.n_cond += config.n_exp
        else:
            self.n_cond = 3 + config.n_exp

        self.mode = DecaMode[str(config.mode).upper()]
        self._create_detail_generator()
        self._init_deep_losses()
        self._setup_neural_rendering()

    def _reinitialize(self):
        self._create_model()
        self._setup_renderer()
        self._init_deep_losses()
        self.face_attr_mask = util.load_local_mask(image_size=self.config.uv_size, mode='bbx')

    def _get_num_shape_params(self): 
        return self.config.n_shape

    def _init_deep_losses(self):
        """
        Initialize networks for deep losses
        """
        # TODO: ideally these networks should be moved out the DECA class and into DecaModule, 
        # but that would break backwards compatility with the original DECA and would not be able to load DECA's weights
        if 'mrfwr' not in self.config.keys() or self.config.mrfwr == 0:
            self.perceptual_loss = None
        else:
            if self.perceptual_loss is None:
                self.perceptual_loss = lossfunc.IDMRFLoss().eval()
                self.perceptual_loss.requires_grad_(False)  # TODO, move this to the constructor

        if 'idw' not in self.config.keys() or self.config.idw == 0:
            self.id_loss = None
        else:
            if self.id_loss is None:
                id_metric = self.config.id_metric if 'id_metric' in self.config.keys() else None
                id_trainable = self.config.id_trainable if 'id_trainable' in self.config.keys() else False
                self.id_loss_start_step = self.config.id_loss_start_step if 'id_loss_start_step' in self.config.keys() else 0
                self.id_loss = lossfunc.VGGFace2Loss(self.config.pretrained_vgg_face_path, id_metric, id_trainable)
                self.id_loss.freeze_nontrainable_layers()

        if 'vggw' not in self.config.keys() or self.config.vggw == 0:
            self.vgg_loss = None
        else:
            if self.vgg_loss is None:
                vgg_loss_batch_norm = 'vgg_loss_batch_norm' in self.config.keys() and self.config.vgg_loss_batch_norm
                self.vgg_loss = VGG19Loss(dict(zip(self.config.vgg_loss_layers, self.config.lambda_vgg_layers)), batch_norm=vgg_loss_batch_norm).eval()
                self.vgg_loss.requires_grad_(False) # TODO, move this to the constructor

    def _setup_renderer(self):
        self.render = SRenderY(self.config.image_size, obj_filename=self.config.topology_path,
                               uv_size=self.config.uv_size)  # .to(self.device)
        # face mask for rendering details
        mask = imread(self.config.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        mask = imread(self.config.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)

        # displacement mask is deprecated and not used by DECA or EMOCA
        if 'displacement_mask' in self.config.keys():
            displacement_mask_ = 1-np.load(self.config.displacement_mask).astype(np.float32)
            # displacement_mask_ = np.load(self.config.displacement_mask).astype(np.float32)
            displacement_mask_ = torch.from_numpy(displacement_mask_)[None, None, ...].contiguous()
            displacement_mask_ = F.interpolate(displacement_mask_, [self.config.uv_size, self.config.uv_size])
            self.register_buffer('displacement_mask', displacement_mask_)

        ## displacement correct
        if os.path.isfile(self.config.fixed_displacement_path):
            fixed_dis = np.load(self.config.fixed_displacement_path)
            fixed_uv_dis = torch.tensor(fixed_dis).float()
        else:
            fixed_uv_dis = torch.zeros([512, 512]).float()
            print("Warning: fixed_displacement_path not found, using zero displacement")
        self.register_buffer('fixed_uv_dis', fixed_uv_dis)

    def uses_texture(self): 
        if 'use_texture' in self.config.keys():
            return self.config.use_texture
        return True # true by default

    def _disable_texture(self, remove_from_model=False): 
        self.config.use_texture = False
        if remove_from_model:
            self.flametex = None

    def _enable_texture(self):
        self.config.use_texture = True

    def _has_neural_rendering(self):
        return hasattr(self.config, "neural_renderer") and bool(self.config.neural_renderer)

    def _setup_neural_rendering(self):
        if self._has_neural_rendering():
            if self.config.neural_renderer.class_ == "StarGAN":
                from .StarGAN import StarGANWrapper
                print("Creating StarGAN neural renderer")
                self.image_translator = StarGANWrapper(self.config.neural_renderer.cfg, self.config.neural_renderer.stargan_repo)
            else:
                raise ValueError(f"Unsupported neural renderer class '{self.config.neural_renderer.class_}'")

            if self.image_translator.background_mode == "input":
                if self.config.background_from_input not in [True, "input"]:
                    raise NotImplementedError("The background mode of the neural renderer and deca is not synchronized. "
                                              "Background should be inpainted from the input")
            elif self.image_translator.background_mode == "black":
                if self.config.background_from_input not in [False, "black"]:
                    raise NotImplementedError("The background mode of the neural renderer and deca is not synchronized. "
                                              "Background should be black.")
            elif self.image_translator.background_mode == "none":
                if self.config.background_from_input not in ["none"]:
                    raise NotImplementedError("The background mode of the neural renderer and deca is not synchronized. "
                                              "The background should not be handled")
            else:
                raise NotImplementedError(f"Unsupported mode of the neural renderer backroungd: "
                                          f"'{self.image_translator.background_mode}'")

    def _create_detail_generator(self):
        #backwards compatibility hack:
        if hasattr(self, 'D_detail'):
            if (not "detail_conditioning_type" in self.config.keys() or  self.config.detail_conditioning_type == "concat") \
                and isinstance(self.D_detail, Generator):
                return
            if self.config.detail_conditioning_type == "adain" and isinstance(self.D_detail, GeneratorAdaIn):
                return
            print("[WARNING]: We are reinitializing the detail generator!")
            del self.D_detail # just to make sure we free the CUDA memory, probably not necessary

        if not "detail_conditioning_type" in self.config.keys() or str(self.config.detail_conditioning_type).lower() == "concat":
            # concatenates detail latent and conditioning (this one is used by DECA/EMOCA)
            print("Creating classic detail generator.")
            self.D_detail = Generator(latent_dim=self.n_detail + self.n_detail_emo + self.n_cond, out_channels=1, out_scale=0.01,
                                      sample_mode='bilinear')
        elif str(self.config.detail_conditioning_type).lower() == "adain":
            # conditioning passed in through adain layers (this one is experimental and not currently used)
            print("Creating AdaIn detail generator.")
            self.D_detail = GeneratorAdaIn(self.n_detail + self.n_detail_emo,  self.n_cond, out_channels=1, out_scale=0.01,
                                      sample_mode='bilinear')
        else:
            raise NotImplementedError(f"Detail conditioning invalid: '{self.config.detail_conditioning_type}'")

    def _create_model(self):
        # 1) build coarse encoder
        e_flame_type = 'ResnetEncoder'
        if 'e_flame_type' in self.config.keys():
            e_flame_type = self.config.e_flame_type

        if e_flame_type == 'ResnetEncoder':
            self.E_flame = ResnetEncoder(outsize=self.n_param)
        elif e_flame_type[:4] == 'swin':
            self.E_flame = SwinEncoder(outsize=self.n_param, img_size=self.config.image_size, swin_type=e_flame_type)
        else:
            raise ValueError(f"Invalid 'e_flame_type' = {e_flame_type}")

        import copy 
        flame_cfg = copy.deepcopy(self.config)
        flame_cfg.n_shape = self._get_num_shape_params()
        if 'flame_mediapipe_lmk_embedding_path' not in flame_cfg.keys():
            self.flame = FLAME(flame_cfg)
        else:
            self.flame = FLAME_mediapipe(flame_cfg)

        if self.uses_texture():
            self.flametex = FLAMETex(self.config)
        else: 
            self.flametex = None

        # 2) build detail encoder
        e_detail_type = 'ResnetEncoder'
        if 'e_detail_type' in self.config.keys():
            e_detail_type = self.config.e_detail_type

        if e_detail_type == 'ResnetEncoder':
            self.E_detail = ResnetEncoder(outsize=self.n_detail + self.n_detail_emo)
        elif e_flame_type[:4] == 'swin':
            self.E_detail = SwinEncoder(outsize=self.n_detail + self.n_detail_emo, img_size=self.config.image_size, swin_type=e_detail_type)
        else:
            raise ValueError(f"Invalid 'e_detail_type'={e_detail_type}")
        self._create_detail_generator()
        # self._load_old_checkpoint()

    def _get_coarse_trainable_parameters(self):
        print("Add E_flame.parameters() to the optimizer")
        return list(self.E_flame.parameters())

    def _get_detail_trainable_parameters(self):
        trainable_params = []
        if self.config.train_coarse:
            trainable_params += self._get_coarse_trainable_parameters()
            print("Add E_flame.parameters() to the optimizer")
        trainable_params += list(self.E_detail.parameters())
        print("Add E_detail.parameters() to the optimizer")
        trainable_params += list(self.D_detail.parameters())
        print("Add D_detail.parameters() to the optimizer")
        return trainable_params

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.mode == DecaMode.COARSE:
                self.E_flame.train()
                # print("Setting E_flame to train")
                self.E_detail.eval()
                # print("Setting E_detail to eval")
                self.D_detail.eval()
                # print("Setting D_detail to eval")
            elif self.mode == DecaMode.DETAIL:
                if self.config.train_coarse:
                    # print("Setting E_flame to train")
                    self.E_flame.train()
                else:
                    # print("Setting E_flame to eval")
                    self.E_flame.eval()
                self.E_detail.train()
                # print("Setting E_detail to train")
                self.D_detail.train()
                # print("Setting D_detail to train")
            else:
                raise ValueError(f"Invalid mode '{self.mode}'")
        else:
            self.E_flame.eval()
            # print("Setting E_flame to eval")
            self.E_detail.eval()
            # print("Setting E_detail to eval")
            self.D_detail.eval()
            # print("Setting D_detail to eval")

        # these are set to eval no matter what, they're never being trained (the FLAME shape and texture spaces are pretrained)
        self.flame.eval()
        if self.flametex is not None:
            self.flametex.eval()
        return self


    def _load_old_checkpoint(self):
        """
        Loads the DECA model weights from the original DECA implementation: 
        https://github.com/YadiraF/DECA 
        """
        if self.config.resume_training:
            model_path = self.config.pretrained_modelpath
            print(f"Loading model state from '{model_path}'")
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

    def _encode_flame(self, images):
        return self.E_flame(images)

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
        return code_list, None

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals, detach=True):
        """
        Converts the displacement uv map (uv_z) and coarse_verts to a normal map coarse_normals. 
        """
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts)#.detach()
        if detach:
            uv_coarse_vertices = uv_coarse_vertices.detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals)#.detach()
        if detach:
            uv_coarse_normals = uv_coarse_normals.detach()

        uv_z = uv_z * self.uv_face_eye_mask

        # detail vertices = coarse vertice + predicted displacement*normals + fixed displacement*normals
        uv_detail_vertices = uv_coarse_vertices + \
                             uv_z * uv_coarse_normals + \
                             self.fixed_uv_dis[None, None, :,:] * uv_coarse_normals #.detach()

        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        # uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1-self.uv_face_eye_mask)
        # uv_detail_normals = util.gaussian_blur(uv_detail_normals)
        return uv_detail_normals, uv_coarse_vertices

    def visualize(self, visdict, savepath, catdim=1):
        grids = {}
        for key in visdict:
            # print(key)
            if visdict[key] is None:
                continue
            grids[key] = torchvision.utils.make_grid(
                F.interpolate(visdict[key], [self.config.image_size, self.config.image_size])).detach().cpu()
        grid = torch.cat(list(grids.values()), catdim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        if savepath is not None:
            cv2.imwrite(savepath, grid_image)
        return grid_image

    def create_mesh(self, opdict, dense_template):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        if 'uv_texture_gt' in opdict.keys():
            texture = util.tensor2image(opdict['uv_texture_gt'][i])
        else:
            texture = None
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        if 'uv_detail_normals' in opdict.keys():
            normal_map = util.tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
            # upsample mesh, save detailed mesh
            texture = texture[:, :, [2, 1, 0]]
            normals = opdict['normals'][i].cpu().numpy()
            displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
            dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces,
                                                                           displacement_map, texture, dense_template)
        else:
            normal_map = None
            dense_vertices = None
            dense_colors  = None
            dense_faces  = None

        return vertices, faces, texture, uvcoords, uvfaces, normal_map, dense_vertices, dense_faces, dense_colors


    def save_obj(self, filename, opdict, dense_template, mode ='detail'):
        if mode not in ['coarse', 'detail', 'both']:
            raise ValueError(f"Invalid mode '{mode}. Expected modes are: 'coarse', 'detail', 'both'")

        vertices, faces, texture, uvcoords, uvfaces, normal_map, dense_vertices, dense_faces, dense_colors \
            = self.create_mesh(opdict, dense_template)

        if mode == 'both':
            if isinstance(filename, list):
                filename_coarse = filename[0]
                filename_detail = filename[1]
            else:
                filename_coarse = filename
                filename_detail = filename.replace('.obj', '_detail.obj')
        elif mode == 'coarse':
            filename_coarse = filename
        else:
            filename_detail = filename

        if mode in ['coarse', 'both']:
            util.write_obj(str(filename_coarse), vertices, faces,
                            texture=texture,
                            uvcoords=uvcoords,
                            uvfaces=uvfaces,
                            normal_map=normal_map)

        if mode in ['detail', 'both']:
            util.write_obj(str(filename_detail),
                            dense_vertices,
                            dense_faces,
                            colors = dense_colors,
                            inverse_face_order=True)


from gdl.models.EmoNetRegressor import EmoNetRegressor, EmonetRegressorStatic


class ExpDECA(DECA):
    """
    This is the EMOCA class (previously ExpDECA). This class derives from DECA and add EMOCA-related functionality. 
    Such as a separate expression decoder and related.
    """

    def _create_model(self):
        # 1) Initialize DECA
        super()._create_model()
        # E_flame should be fixed for expression EMOCA
        self.E_flame.requires_grad_(False)
        
        # 2) add expression decoder
        if self.config.expression_backbone == 'deca_parallel':
            ## a) Attach a parallel flow of FCs onto the original DECA coarse backbone. (Only the second FC head is trainable)
            self.E_expression = SecondHeadResnet(self.E_flame, self.n_exp_param, 'same')
        elif self.config.expression_backbone == 'deca_clone':
            ## b) Clones the original DECA coarse decoder (and the entire decoder will be trainable) - This is in final EMOCA.
            #TODO this will only work for Resnet. Make this work for the other backbones (Swin) as well.
            self.E_expression = ResnetEncoder(self.n_exp_param)
            # clone parameters of the ResNet
            self.E_expression.encoder.load_state_dict(self.E_flame.encoder.state_dict())
        elif self.config.expression_backbone == 'emonet_trainable':
            # Trainable EmoNet instead of Resnet (deprecated)
            self.E_expression = EmoNetRegressor(self.n_exp_param)
        elif self.config.expression_backbone == 'emonet_static':
            # Frozen EmoNet with a trainable head instead of Resnet (deprecated)
            self.E_expression = EmonetRegressorStatic(self.n_exp_param)
        else:
            raise ValueError(f"Invalid expression backbone: '{self.config.expression_backbone}'")
        
        if self.config.get('zero_out_last_enc_layer', False):
            self.E_expression.reset_last_layer() 

    def _get_coarse_trainable_parameters(self):
        print("Add E_expression.parameters() to the optimizer")
        return list(self.E_expression.parameters())

    def _reconfigure(self, config):
        super()._reconfigure(config)
        self.n_exp_param = self.config.n_exp

        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            self.n_exp_param += self.config.n_pose
        elif self.config.exp_deca_global_pose or self.config.exp_deca_jaw_pose:
            self.n_exp_param += 3

    def _encode_flame(self, images):
        if self.config.expression_backbone == 'deca_parallel':
            #SecondHeadResnet does the forward pass for shape and expression at the same time
            return self.E_expression(images)
        # other regressors have to do a separate pass over the image
        deca_code = super()._encode_flame(images)
        exp_deca_code = self.E_expression(images)
        return deca_code, exp_deca_code

    def decompose_code(self, code):
        deca_code = code[0]
        expdeca_code = code[1]

        deca_code_list, _ = super().decompose_code(deca_code)
        # shapecode, texcode, expcode, posecode, cam, lightcode = deca_code_list
        exp_idx = 2
        pose_idx = 3

        # deca_exp_code = deca_code_list[exp_idx]
        # deca_global_pose_code = deca_code_list[pose_idx][:3]
        # deca_jaw_pose_code = deca_code_list[pose_idx][3:6]

        deca_code_list_copy = deca_code_list.copy()

        # self.E_mica.cfg.model.n_shape

        #TODO: clean this if-else block up
        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            exp_code = expdeca_code[:, :self.config.n_exp]
            pose_code = expdeca_code[:, self.config.n_exp:]
            deca_code_list[exp_idx] = exp_code
            deca_code_list[pose_idx] = pose_code
        elif self.config.exp_deca_global_pose:
            # global pose from ExpDeca, jaw pose from EMOCA
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_exp_deca, pose_code_deca[:,3:]], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        elif self.config.exp_deca_jaw_pose:
            # global pose from EMOCA, jaw pose from ExpDeca
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_deca[:, :3], pose_code_exp_deca], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        else:
            exp_code = expdeca_code
            deca_code_list[exp_idx] = exp_code

        return deca_code_list, deca_code_list_copy

    def train(self, mode: bool = True):
        super().train(mode)

        # for expression deca, we are not training the resnet feature extractor plus the identity/light/texture regressor
        self.E_flame.eval()

        if mode:
            if self.mode == DecaMode.COARSE:
                self.E_expression.train()
                # print("Setting E_expression to train")
                self.E_detail.eval()
                # print("Setting E_detail to eval")
                self.D_detail.eval()
                # print("Setting D_detail to eval")
            elif self.mode == DecaMode.DETAIL:
                if self.config.train_coarse:
                    # print("Setting E_flame to train")
                    self.E_expression.train()
                else:
                    # print("Setting E_flame to eval")
                    self.E_expression.eval()
                self.E_detail.train()
                # print("Setting E_detail to train")
                self.D_detail.train()
            else:
                raise ValueError(f"Invalid mode '{self.mode}'")
        else:
            self.E_expression.eval()
            self.E_detail.eval()
            self.D_detail.eval()
        return self


    
class EMICA(ExpDECA): 

    def __init__(self, config):
        self.use_mica_shape_dim = True
        # self.use_mica_shape_dim = False
        from .mica.config import get_cfg_defaults
        self.mica_cfg = get_cfg_defaults()
        super().__init__(config)
  
    def _create_model(self):
        # 1) Initialize DECA
        super()._create_model()
        from .mica.mica import MICA
        #TODO: MICA uses FLAME  
        # 1) This is redundant - get rid of it 
        # 2) Make sure it's the same FLAME as EMOCA
        if Path(self.config.mica_model_path).exists(): 
            mica_path = self.config.mica_model_path 
        else:
            from gdl.utils.other import get_path_to_assets
            mica_path = get_path_to_assets() / self.config.mica_model_path  
            assert mica_path.exists(), f"MICA model path does not exist: '{mica_path}'"

        self.mica_cfg.pretrained_model_path = str(mica_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.E_mica = MICA(self.mica_cfg, device, str(mica_path), instantiate_flame=False)
        # E_mica should be fixed 
        self.E_mica.requires_grad_(False)
        self.E_mica.testing = True

        # preprocessing for MICA
        if self.config.mica_preprocessing:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(224, 224))


    def _get_num_shape_params(self):
        if self.use_mica_shape_dim:
            return self.mica_cfg.model.n_shape
        return self.config.n_shape

    def _get_coarse_trainable_parameters(self):
        # MICA is not trainable so we don't wanna add it 
        return super()._get_coarse_trainable_parameters()


    def train(self, mode: bool = True):
        super().train(mode)
        self.E_mica.train(False) # MICA is pretrained and will be set to EVAL at all times 


    def _encode_flame(self, images):
        
        if self.config.mica_preprocessing:
            mica_image = self._dirty_image_preprocessing(images)
        else: 
            mica_image = F.interpolate(images, (112,112), mode='bilinear', align_corners=False)

        deca_code, exp_deca_code = super()._encode_flame(images)
        mica_code = self.E_mica.encode(images, mica_image) 
        mica_code = self.E_mica.decode(mica_code, predict_vertices=False)
        return deca_code, exp_deca_code, mica_code['pred_shape_code']
 
    def _dirty_image_preprocessing(self, input_image): 
        # breaks whatever gradient flow that may have gone into the image creation process
        from gdl.models.mica.detector import get_center, get_arcface_input
        from insightface.app.common import Face
        
        image = input_image.detach().clone().cpu().numpy() * 255. 
        # b,c,h,w to b,h,w,c
        image = image.transpose((0,2,3,1))
    
        min_det_score = 0.5
        image_list = list(image)
        aligned_image_list = []
        for i, img in enumerate(image_list):
            bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] == 0:
                aimg = resize(img, output_shape=(112,112), preserve_range=True)
                aligned_image_list.append(aimg)
                raise RuntimeError("No faces detected")
                continue
            i = get_center(bboxes, image)
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            # if det_score < min_det_score:
            #     continue
            kps = None
            if kpss is not None:
                kps = kpss[i]

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            blob, aimg = get_arcface_input(face, img)
            aligned_image_list.append(aimg)
        aligned_images = np.array(aligned_image_list)
        # b,h,w,c to b,c,h,w
        aligned_images = aligned_images.transpose((0,3,1,2))
        # to torch to correct device 
        aligned_images = torch.from_numpy(aligned_images).to(input_image.device)
        return aligned_images

    def decompose_code(self, code): 
        deca_code = code[0]
        expdeca_code = code[1]
        mica_code = code[2]

        code_list, deca_code_list_copy = super().decompose_code((deca_code, expdeca_code), )

        id_idx = 0 # identity is the first part of the vector
        # assert self.config.n_shape == mica_code.shape[-1]
        # assert code_list[id_idx].shape[-1] == mica_code.shape[-1]
        if self.use_mica_shape_dim:
            code_list[id_idx] = mica_code
        else: 
            code_list[id_idx] = mica_code[..., :self.config.n_shape]
        return code_list, deca_code_list_copy


def instantiate_deca(cfg, stage, prefix, checkpoint=None, checkpoint_kwargs=None):
    """
    Function that instantiates a DecaModule from checkpoint or config
    """

    if checkpoint is None:
        deca = DecaModule(cfg.model, cfg.learning, cfg.inout, prefix)
        if cfg.model.resume_training:
            # This load the DECA model weights from the original DECA release
            print("[WARNING] Loading EMOCA checkpoint pretrained by the old code")
            deca.deca._load_old_checkpoint()
    else:
        checkpoint_kwargs = checkpoint_kwargs or {}
        deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
        if stage == 'train':
            mode = True
        else:
            mode = False
        deca.reconfigure(cfg.model, cfg.inout, cfg.learning, prefix, downgrade_ok=True, train=mode)
    return deca
