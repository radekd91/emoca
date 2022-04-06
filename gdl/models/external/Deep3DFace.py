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
## BEGIN DANGER - PROTECTIVE GOGGLES ON!
# YES, IT IS BEYOND STUPID TO DO THIS, MAKES NO SENSE,  BUT NOW THE NVDIFFRAST dr.RasterizeGLContext DOES NOT CRASH
# WHEN CALLED BY Deep3DFaceRecon_pytorch. SO STOP COMPLAINING!!!
import torch
import nvdiffrast.torch as dr
glctx = dr.RasterizeGLContext(device=torch.device('cuda:0'))
del glctx
## END DANGER

from gdl.utils.other import get_path_to_externals

repo_dir = str(get_path_to_externals())
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
deep3d_face_dir = get_path_to_externals() / "Deep3DFaceRecon_pytorch")
if deep3d_face_dir not in sys.path:
    sys.path.insert(0, deep3d_face_dir)
from models import create_model
from util.visualizer import MyVisualizer


import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
from gdl.utils.lightning_logging import _log_array_image, _log_wandb_image, _torch_image2np
from pytorch_lightning.loggers import WandbLogger
from models import create_model
import gdl.utils.DecaUtils as dutil


class Deep3DFaceModule(pl.LightningModule):

    def __init__(self, model_params, learning_params, inout_params, stage_name=""):
        super().__init__()
        self.learning_params = learning_params
        self.inout_params = inout_params
        self.model_params = model_params
        self.stage_name = stage_name
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        self.deepface3d = Deep3DFaceWrapper(cfg=model_params.deep3dface)

    def encode(self, batch, **kwargs):
        return self.deepface3d.encode(batch)

    def decode(self, values, **kwargs):
        return self.deepface3d.decode(values)

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
            values = self.decode(batch, values)
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


    def _visualization_checkpoint(self, batch_size, batch, values, batch_idx, stage, prefix):
        visind = np.arange(batch_size)

        visuals = self.deepface3d.model.get_current_visuals()
        visual_n = visuals['output_vis'].shape[-1] // visuals['output_vis'].shape[-2]
        if visual_n == 2:
            output_im = visuals['output_vis'][:,:,:, visuals['output_vis'].shape[-2]:]
            predicted_lm = None
        else:
            predicted_lm = visuals['output_vis'][:, :, :, visuals['output_vis'].shape[-2]:visuals['output_vis'].shape[-2]*2]
            output_im = visuals['output_vis'][:, :, :, visuals['output_vis'].shape[-2]*2:]

        visdict = {}
        if 'image' in batch.keys():
            visdict['inputs'] = batch['image'].detach().cpu().numpy().transpose([0,2,3,1])[visind]
            # visdict['landmarks_gt'] = batch['image'].detach().cpu().numpy().transpose([0,2,3,1])[visind]
            if predicted_lm is None:
                visdict['landmarks_predicted'] = batch['image'].detach().cpu().numpy().transpose([0,2,3,1])[visind]
            else:
                visdict['landmarks_predicted'] = predicted_lm.detach().cpu().numpy().transpose([0,2,3,1])[visind]

            if "landmark" in batch.keys():
                visdict['landmarks_gt'] = dutil.tensor_vis_landmarks(batch['image'][visind],
                                                                     batch['landmark'][visind]).detach().cpu().numpy().transpose([0,2,3,1])


        im =  output_im.detach().cpu().numpy().transpose([0,2,3,1])[visind]
        visdict['output_images_coarse'] =  im
        visdict['output_images_detail'] =  im

        if 'geometry_coarse' in values.keys():
            visdict['geometry_coarse'] = values['geometry_coarse'].detach().cpu().numpy().transpose([0,2,3,1])[visind]
            visdict['geometry_detail'] = values['geometry_detail'].detach().cpu().numpy().transpose([0,2,3,1])[visind]
            visdict['mask'] = values['mask'].detach().cpu().numpy().transpose([0,2,3,1])[...,0][visind]

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


class Deep3DFaceWrapper(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.config = cfg
        cfg_ = OmegaConf.to_container(cfg)
        # if not Path(cfg_["img_folder"]).is_file():
        #     cfg_["img_folder"] = str(Path(deep3d_face_dir) / cfg_["img_folder"])
        #     assert Path( cfg_["img_folder"]).is_file()

        if not Path(cfg_["init_path"]).is_dir():
            cfg_["init_path"] = str(Path(deep3d_face_dir) / cfg_["init_path"])
            # assert Path(cfg_["init_path"]).is_file()

        if not Path(cfg_["checkpoints_dir"]).is_dir():
            cfg_["checkpoints_dir"] = str(Path(deep3d_face_dir) / cfg_["checkpoints_dir"])
            assert Path(cfg_["checkpoints_dir"]).is_dir()

        if not Path(cfg_["bfm_folder"]).is_dir():
            cfg_["bfm_folder"] = str(Path(deep3d_face_dir) / cfg_["bfm_folder"])
            assert Path(cfg_["bfm_folder"]).is_dir()

        cfg_ = DictConfig(cfg_)

        self.model = create_model(cfg_)
        self.model.setup(cfg_)
        self.model.device = torch.device(0)
        self.model.parallelize()
        self.model.eval()
        self.visualizer = MyVisualizer(cfg_)

    def forward(self, batch):
        values = self.encode(batch)
        values = self.decode(batch, values)
        return values

    def encode(self, batch):
        img = batch["image"]

        data = {
            'imgs': batch["image"], #TODO: resize, and otherwise process images here, however necessary
            # 'lms': batch["landmark"], # same for landmarks
        }
        self.model.set_input(data)  # unpack data from data loader
        self.model.forward()
        # self.model.test()

        #TODO: extract the 3DMM parameters somehow and add them to values
        # dict_keys(['id', 'exp', 'tex', 'angle', 'gamma', 'trans'])
        values = {
            "shapecode": self.model.pred_coeffs_dict["id"],
            "expcode": self.model.pred_coeffs_dict["exp"],
            "texcode": self.model.pred_coeffs_dict["tex"],
            "posecode": self.model.pred_coeffs_dict["angle"],
            "trans": self.model.pred_coeffs_dict["trans"],
            "gamma": self.model.pred_coeffs_dict["gamma"],
        }
        return values

    def decode(self, values):
        self.model.compute_visuals()
        visuals = self.model.get_current_visuals()  #

        # black_white = self.model.pred_color.mean(dim=2, keepdim=True).repeat((1, 1, 3))
        # pred_mask, depth, pred_face = self.model.renderer(
        #     self.model.pred_vertex, self.model.facemodel.face_buf, feat=black_white)
        geometry_im = self.compute_shape_render()
        values["geometry_coarse"] = geometry_im
        values["geometry_detail"] = geometry_im
        values["output_images_coarse"] = self.model.pred_face
        values["output_images_detail"] = self.model.pred_face
        values["mask"] = self.model.pred_mask

        return values

    def compute_shape_render(self):#, coef_dict):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        # coef_dict = self.split_coeff(coeffs)
        coef_dict = self.model.pred_coeffs_dict

        face_shape = self.model.facemodel.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.model.facemodel.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.model.facemodel.transform(face_shape, rotation, coef_dict['trans'])
        face_vertex = self.model.facemodel.to_camera(face_shape_transformed)

        face_proj = self.model.facemodel.to_image(face_vertex)
        # landmark = self.model.get_landmarks(face_proj)

        face_texture = self.model.facemodel.compute_texture(coef_dict['tex'])
        face_texture = torch.ones_like(face_texture)*0.8
        face_norm = self.model.facemodel.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        # face_color = self.model.facemodel.compute_color(face_texture, face_norm_roted, coef_dict['gamma'])
        # mask_im, depth_im, geometry_im = self.model.renderer(
        #     face_vertex, self.model.facemodel.face_buf, feat=face_color)

        batch_size = coef_dict['id'].shape[0]
        light_positions = torch.tensor(
            [
                [-1, 1, 1],
                [1, 1, 1],
                [-1, -1, 1],
                [1, -1, 1],
                [0, 0, 1]
            ]
        )[None, :, :].expand(batch_size, -1, -1).float()
        light_intensities = torch.ones_like(light_positions).float() * 1.3
        lights = torch.cat((light_positions, light_intensities), 2).to(face_texture.device)
        face_color = self.model.facemodel.compute_color_directional_light(face_norm_roted, lights)

        mask_im, depth_im, geometry_im = self.model.renderer(
            face_vertex, self.model.facemodel.face_buf, feat=face_color)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # im = geometry_im.detach().cpu()[0].numpy().transpose([1, 2, 0])
        # plt.imshow(im)
        # plt.show()

        return geometry_im

    def compute_unlit_render(self):#, coef_dict):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        # coef_dict = self.split_coeff(coeffs)
        coef_dict = self.model.pred_coeffs_dict

        face_shape = self.model.facemodel.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.model.facemodel.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.model.facemodel.transform(face_shape, rotation, coef_dict['trans'])
        face_vertex = self.model.facemodel.to_camera(face_shape_transformed)

        face_proj = self.model.facemodel.to_image(face_vertex)
        # landmark = self.model.get_landmarks(face_proj)

        face_texture = self.model.facemodel.compute_texture(coef_dict['tex'])



        mask_im, depth_im, unlit_im = self.model.renderer(
            face_vertex, self.model.facemodel.face_buf, feat=face_texture.contiguous())
        return unlit_im



def instantiate_other_face_models(cfg, stage, prefix, checkpoint=None, checkpoint_kwargs=None):
    module_class = Deep3DFaceModule

    if checkpoint is None:
        face_model = module_class(cfg.model, cfg.learning, cfg.inout, prefix)

    else:
        checkpoint_kwargs = checkpoint_kwargs or {}
        face_model = module_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
        # if stage == 'train':
        #     mode = True
        # else:
        #     mode = False
        # face_model.reconfigure(cfg.model, cfg.inout, cfg.learning, prefix, train=mode)
    return face_model
