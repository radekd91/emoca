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


import sys, os
from pathlib import Path

import numpy as np
from gdl.utils.other import get_path_to_externals

repo_dir = str(get_path_to_externals())
sys.path += [repo_dir]
tddfa_v2_dir = str(get_path_to_externals() / "TDDFA_V2")
sys.path += [tddfa_v2_dir]
from TDDFA_V2.TDDFA import TDDFA
from TDDFA_V2.FaceBoxes import FaceBoxes
from TDDFA_V2.utils.render import render
from TDDFA_V2.utils.functions import draw_landmarks
from utils.tddfa_util import (
    load_model, _parse_param, similar_transform,
    ToTensorGjz, NormalizeGjz
)

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import pytorch_lightning as pl
from gdl.utils.lightning_logging import _log_array_image, _log_wandb_image, _torch_image2np
from pytorch_lightning.loggers import WandbLogger


def _parse_param_batch(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[1]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    bs = param.shape[0]

    R_ = param[:, :trans_dim].reshape(bs, 3, -1)
    R = R_[:,: , :3]
    offset = R_[:,:, -1].reshape(bs, 3, 1)
    alpha_shp = param[:, trans_dim:trans_dim + shape_dim].reshape(bs, -1)
    alpha_exp = param[:, trans_dim + shape_dim:].reshape(bs, -1)

    return R, offset, alpha_shp, alpha_exp


class Face3DDFAModule(pl.LightningModule):

    def __init__(self, model_params, learning_params, inout_params, stage_name=""):
        super().__init__()
        self.learning_params = learning_params
        self.inout_params = inout_params
        self.model_params = model_params
        self.stage_name = stage_name
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        self.tddfa = Face3DDFAv2Wrapper(cfg=model_params.tddfa)

    def encode(self, batch, **kwargs):
        return self.tddfa.encode(batch)

    def decode(self, values, **kwargs):
        return self.tddfa.decode(values)

    def training_step(self):
        raise NotImplementedError()


    def validation_step(self):
        raise NotImplementedError()


    def test_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = "test_detail"
        losses_and_metrics_to_log = {}

        if dataloader_idx is not None:
            dataloader_str = str(dataloader_idx) + "_"
        else:
            dataloader_str = ''
            stage_str = dataloader_str + 'test_'

        with torch.no_grad():
            # training = False
            # testing = True
            values = self.encode(batch)
            values = self.decode( values)
            # if 'mask' in batch.keys():
            #     losses_and_metrics = self.compute_loss(values, batch, training=False, testing=testing)
            #     # losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach().cpu() for key, value in losses_and_metrics.items()}
            #     losses_and_metrics_to_log = {prefix + '_' + stage_str + key: value.detach().cpu().item() for key, value
            #                                  in losses_and_metrics.items()}
            # else:
            #     losses_and_metric = None

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
        # losses_and_metrics_to_log[prefix + '_' + stage_str + 'mem_usage'] = self.process.memory_info().rss
        losses_and_metrics_to_log[stage_str + 'epoch'] = self.current_epoch
        losses_and_metrics_to_log[stage_str + 'step'] = self.global_step
        losses_and_metrics_to_log[stage_str + 'batch_idx'] = batch_idx
        # losses_and_metrics_to_log[stage_str + 'mem_usage'] = self.process.memory_info().rss

        # We don't do logging for 3DDFA
        # if self.logger is not None:
            # self.logger.log_metrics(losses_and_metrics_to_log)
            # self.log_dict(losses_and_metrics_to_log, sync_dist=True, on_step=False, on_epoch=True)

        # # if self.global_step % 200 == 0:
        # uv_detail_normals = None
        # if 'uv_detail_normals' in values.keys():
        #     uv_detail_normals = values['uv_detail_normals']

        if self.model_params.test_vis_frequency > 0:
            if batch_idx % self.model_params.test_vis_frequency == 0:
                # if self.trainer.is_global_zero:
                visualizations = self._visualization_checkpoint(batch["image"].shape[0],
                                                                batch,
                                                                values,
                                                                self.global_step,
                                                                stage_str[:-1], prefix)

                visdict = self._create_visualizations_to_log(stage_str[:-1], batch, visualizations, values, batch_idx,
                                                             indices=0, dataloader_idx=dataloader_idx)
                if self.logger is not None:
                    self.logger.log_metrics(visdict)
        return None


    def visualize(self, visdict, savepath, catdim=1):
        import cv2

        grids = {}
        for key in visdict:
            # print(key)
            if visdict[key] is None:
                continue
            # grids[key] = torchvision.utils.make_grid(
            #     F.interpolate(visdict[key], [self.model_params.image_size, self.model_params.image_size])).detach().cpu()
            grids[key] = np.concatenate(list(visdict[key]), 0)
            if len(grids[key].shape) == 2:
                grids[key] = np.stack([grids[key], grids[key], grids[key]], 2)
            elif grids[key].shape[2] == 1:
                grids[key] = np.concatenate([grids[key], grids[key], grids[key]], 2)

        grid = np.concatenate(list(grids.values()), catdim-1)
        grid_image = (grid * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        if savepath is not None:
            cv2.imwrite(savepath, grid_image)
        return grid_image


    def _visualization_checkpoint(self, batch_size, batch, values, batch_idx, stage, prefix):
        visind = np.arange(batch_size)

        visdict = {}
        if 'image' in batch.keys():
            visdict['inputs'] = batch['image'].detach().cpu().numpy().transpose([0,2,3,1])[visind]
            visdict['landmarks_gt'] = batch['image'].detach().cpu().numpy().transpose([0,2,3,1])[visind]
            visdict['landmarks_predicted'] = batch['image'].detach().cpu().numpy().transpose([0,2,3,1])[visind]

        if 'overlays_img' in values.keys():
            visdict['output_images_coarse'] =  np.stack(values['overlays_img'])[visind]
            visdict['output_images_detail'] =  np.stack(values['overlays_img'])[visind]

        if 'geometry_coarse' in values.keys():
            visdict['geometry_coarse'] = np.stack(values['geometry_coarse'])[visind]
            visdict['geometry_detail'] = np.stack(values['geometry_coarse'])[visind]
            visdict['mask'] = (visdict['geometry_detail'].mean(axis=3) != 0).astype(np.float32)

        return visdict


    def _get_logging_prefix(self):
        prefix = self.stage_name + "detail"
        return prefix

    def _create_visualizations_to_log(self, stage, batch, visdict, values, step, indices=None,
                                      dataloader_idx=None, output_dir=None):
        mode_ = "detail"
        prefix = self._get_logging_prefix()

        output_dir = output_dir or self.inout_params.full_run_dir

        log_dict = {}
        for key in visdict.keys():
            # images = _torch_image2np(visdict[key])
            images = visdict[key]
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
                    # if self.emonet_loss is not None:
                    if key == 'inputs':
                        if mode_ + "_valence_input" in values.keys():
                            caption += self.vae_2_str(
                                values[mode_ + "_valence_input"][i].detach().cpu().item(),
                                values[mode_ + "_arousal_input"][i].detach().cpu().item(),
                                np.argmax(values[mode_ + "_expression_input"][i].detach().cpu().numpy()),
                                prefix="emonet") + "\n"
                        if 'va' in batch.keys() and mode_ + "valence_gt" in values.keys():
                            caption += self.vae_2_str(
                                values[mode_ + "_valence_gt"][i].detach().cpu().item(),
                                values[mode_ + "_arousal_gt"][i].detach().cpu().item(),
                            # caption += self.vae_2_str(
                            #     values[mode_ + "valence_gt"][i].detach().cpu().item(),
                            #     values[mode_ + "arousal_gt"][i].detach().cpu().item(),
                            #     prefix="gt") + "\n"
                        # if 'expr7' in values.keys() and mode_ + "_expression_gt" in values.keys():
                        #     caption += "\n" + self.vae_2_str(
                        #         expr7=values[mode_ + "_expression_gt"][i].detach().cpu().numpy(),
                                prefix="gt") + "\n"
                        if 'affectnetexp' in batch.keys() and mode_ + "_expression_gt" in values.keys():
                            caption += "\n" + self.vae_2_str(
                                affnet_expr=values[mode_ + "_expression_gt"][i].detach().cpu().numpy(),
                                prefix="gt") + "\n"
                    # elif 'geometry_detail' in key:
                    #     if "emo_mlp_valence" in values.keys():
                    #         caption += self.vae_2_str(
                    #             values["emo_mlp_valence"][i].detach().cpu().item(),
                    #             values["emo_mlp_arousal"][i].detach().cpu().item(),
                    #             prefix="mlp")
                    #     if 'emo_mlp_expr_classification' in values.keys():
                    #         caption += "\n" + self.vae_2_str(
                    #             affnet_expr=values["emo_mlp_expr_classification"][i].detach().cpu().argmax().numpy(),
                    #             prefix="mlp") + "\n"
                    # elif key == 'output_images_' + mode_:
                    #     if mode_ + "_valence_output" in values.keys():
                    #         caption += self.vae_2_str(values[mode_ + "_valence_output"][i].detach().cpu().item(),
                    #                                          values[mode_ + "_arousal_output"][i].detach().cpu().item(),
                    #                                          np.argmax(values[mode_ + "_expression_output"][i].detach().cpu().numpy())) + "\n"
                    #
                    # elif key == 'output_translated_images_' + mode_:
                    #     if mode_ + "_translated_valence_output" in values.keys():
                    #         caption += self.vae_2_str(values[mode_ + "_translated_valence_output"][i].detach().cpu().item(),
                    #                                          values[mode_ + "_translated_arousal_output"][i].detach().cpu().item(),
                    #                                          np.argmax(values[mode_ + "_translated_expression_output"][i].detach().cpu().numpy())) + "\n"

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


class Face3DDFAv2Wrapper(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        cfg_ = OmegaConf.to_container(cfg)
        if not Path(cfg_["bfm_fp"]).is_file():
            cfg_["bfm_fp"] = str(Path(tddfa_v2_dir) / cfg_["bfm_fp"])
            assert Path( cfg_["bfm_fp"]).is_file()

        if not Path(cfg_["param_mean_std_fp"]).is_file():
            cfg_["param_mean_std_fp"] = str(Path(tddfa_v2_dir) / cfg_["param_mean_std_fp"])
            assert Path(cfg_["param_mean_std_fp"]).is_file()

        if not Path(cfg_["checkpoint_fp"]).is_file():
            cfg_["checkpoint_fp"] = str(Path(tddfa_v2_dir) / cfg_["checkpoint_fp"])
            assert Path(cfg_["checkpoint_fp"]).is_file()

        self.tddfa = TDDFA(**cfg_)

        self.crop_images = False

        self.param_std = torch.from_numpy(self.tddfa.param_std).to(next(self.tddfa.model.parameters()).device)
        self.param_mean = torch.from_numpy(self.tddfa.param_mean).to(next(self.tddfa.model.parameters()).device)

        self.face_boxes = FaceBoxes()


    def forward(self, batch):
        values = self.encode(batch)
        values = self.decode(values)
        return values

    def encode(self, batch):
        img = batch["image"]

        if not self.crop_images:
            # the shortcut doesn't work
            resized_img = F.interpolate(img, (self.tddfa.size, self.tddfa.size), mode='bilinear')*255.
            resized_img = resized_img[:, [2, 1, 0], ...] #RGB to BGR
            transformed = self.tddfa.transform(resized_img)#.unsqueeze(0)+
            param = self.tddfa.model(transformed)
            # param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale

            # for param in param_lst:
            R, offset, alpha_shp, alpha_exp = _parse_param_batch(param)
            values = {
                "image": transformed,
                "posecode": R,
                "offset": offset,
                "shapecode": alpha_shp,
                "expcode": alpha_exp,
                "bboxes": torch.tensor([0,0, batch["image"].shape[2], batch["image"].shape[3]]).
                    unsqueeze(0).repeat((img.shape[0],1,)),
                "params": param,
                # "roi_box_lst": roi_box_lst
            }
        else:
        # # Detect faces, get 3DMM params and roi boxes
            bboxes = []
            params = []
            shapecodes = []
            expcodes = []
            posecodes = []
            offsets = []

            imgs = (img.detach().cpu().numpy()*255.).transpose([0,2,3,1]).astype(np.uint8)
            for i in range(imgs.shape[0]):
                img = imgs[i][:, :, ::-1]  #RGB to BGR
                boxes = self.face_boxes(img)
                n = len(boxes)
                if n == 0:
                    print(f'No face detected')
                    boxes = [[0, 0, batch["image"].shape[2], batch["image"].shape[3]]]
                    # print(f'No face detected, exit')
                    # sys.exit(-1)

                # bboxes += [boxes[0]]
                # print(f'Detect {n} faces')
            # param_lst, roi_box_lst = self.tddfa.model(img, boxes)
                param_lst, roi_box_lst = self.tddfa(img,  [boxes[0]])
                bboxes += [roi_box_lst[0]]
                params += [param_lst[0]]

                R, offset, alpha_shp, alpha_exp = _parse_param(param_lst[0])
                shapecodes += [alpha_shp]
                expcodes += [alpha_exp]
                posecodes += [R]
                offsets += [offset]

            # param = self.tddfa(bboxes)
            values = {
                "bboxes": bboxes,
                "params": params,
                "shapecode" : torch.from_numpy(np.hstack(shapecodes).T).to(self.param_mean.device),
                "expcode" : torch.from_numpy(np.hstack(expcodes).T).to(self.param_mean.device),
                "posecode" : torch.from_numpy(np.hstack(posecodes).T).to(self.param_mean.device),
                "offset": torch.from_numpy(np.hstack(offsets).T).to(self.param_mean.device),
                "image": img,
            }

        return values

    def decode(self, values):
        image = values["image"]
        # image = values["image"]
        # param_lst = values["param_lst"]
        # roi_box_lst = values["roi_box_lst"]

        param_lst = values["params"]
        roi_box_lst = values["bboxes"]
        wfp = None
        dense_flag = True
        if not self.crop_images:
            ver_lst = self.tddfa.recon_vers(param_lst.detach().cpu().numpy(), roi_box_lst.detach().cpu().numpy(),
                                        dense_flag=dense_flag)
        else:
            ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst,
                                            dense_flag=dense_flag)
        imgs = image.detach().cpu().numpy()

        geometry_imgs = []
        landmark_imgs = []
        overlays_imgs = []

        for i, img in enumerate(imgs):
            if self.crop_images:
                img = (img.transpose([1, 2, 0])[:, :, [2,1,0]]*255.).astype(np.uint8)
            else:
                img = (img.transpose([1, 2, 0])*255.).astype(np.uint8)
            show_flag = False
            dense_flag = False
            # landmark_image = draw_landmarks(img, [ver_lst[i].copy()], show_flag=False, dense_flag=dense_flag, wfp=wfp)
            img_geometry = render(img, [ver_lst[i].copy()], self.tddfa.tri, alpha=0.6, show_flag=show_flag, wfp=wfp, with_bg_flag=False)
            img_overlay = render(img, [ver_lst[i].copy()], self.tddfa.tri, alpha=0.6, show_flag=show_flag, wfp=wfp, with_bg_flag=True)
            geometry_imgs += [img_geometry]
            overlays_imgs += [img_overlay]
            # landmark_imgs += [landmark_image]

        values["geometry_coarse"] = geometry_imgs
        # values["landmark_img"] = overlays_imgs
        values["overlays_img"] = overlays_imgs

        return values




