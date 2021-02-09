import os, sys
from pathlib import Path
sys.path += [str(Path(__file__).parent.parent)]

import numpy as np
from datasets.FaceVideoDataset import FaceVideoDataModule, \
    AffectNetExpressions, Expression7, AU8, expr7_to_affect_net
from datasets.EmotionalDataModule import EmotionDataModule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
import datetime
# import hydra
import yaml
import torch
import torch.nn.functional as F
from typing import Dict, Any
import pickle as pkl
# from decalib.deca import DECA
# from decalib.datasets import datasets
# from decalib.utils import util

DECA_IMPORTED = False

def _import_deca():
    global DECA_IMPORTED

    DECA_IMPORTED = True

import face_alignment

from enum import Enum


class DecaMode(Enum):
    COARSE = 1
    DETAIL = 2


class DecaModule(LightningModule):

    def __init__(self, model_params, learning_params, inout_params):
        super().__init__()
        # run DECA
        # from decalib.utils.config import cfg as deca_cfg
        # deca_cfg.model.use_tex = False
        # deca = DECA(config=deca_cfg, device=args.device)
        # from decalib.utils.config import cfg as deca_cfg
        # deca_cfg.model.use_tex = args.useTex
        # deca_cfg.model.use_tex = False
        sys.path += [str(Path(__file__).parent.parent.parent.parent.absolute() / 'DECA-training')]
        from lib.model_deca import DECA
        from lib.utils.util import dict_tensor2npy
        import lib.utils.util as util
        import lib.utils.lossfunc as lossfunc
        self.util = util
        self.lossfunc = lossfunc
        self.learning_params = learning_params
        self.inout_params = inout_params
        self.deca = DECA(config=model_params, device=self.device)
        self.mode = DecaMode[str(model_params.mode).upper()]
        self.dict_tensor2npy = dict_tensor2npy
        self.landmark_detector = None

    def _check_device_for_extra_params(self):
        if self.deca.uv_face_eye_mask.device != self.device:
            self.deca.uv_face_eye_mask = self.deca.uv_face_eye_mask.to(self.device)
        if self.deca.fixed_uv_dis.device != self.device:
            self.deca.fixed_uv_dis = self.deca.fixed_uv_dis.to(self.device)

    # def to(self, device, *args, **kwargs):
    #     self.deca.uv_face_eye_mask = self.deca.uv_face_eye_mask.to(device)
    #     self.deca.fixed_uv_dis = self.deca.fixed_uv_dis.to(device)
    #     return super().to(*args, device=device, **kwargs)

    # def to(self, *args, **kwargs):
    #     self.deca.E_flame.to(*args, **kwargs)
    #     self.deca.E_detail.to(*args, **kwargs)
    #     self.deca.D_detail.to(*args, **kwargs)
    #     return self
    #
    # def cuda(self, device):

    # def _instantiate_landmark_detector(self):
    #     if self.landmark_detector is None:
    #         self.landmark_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
    #                                                   device=self.device,
    #                                                   flip_input=False,
    #                                                   face_detector=self.face_detector,
    #                                                   face_detector_kwargs=self.face_detector_kwargs)

    def forward(self, image):
        codedict = self.deca.encode(image)
        opdict, visdict = self.deca.decode(codedict)
        opdict = self.dict_tensor2npy(opdict)


    def on_train_epoch_start(self) -> None:
        self.deca.E_flame.eval()
        self.deca.E_detail.train()
        self.deca.D_detail.train()



    def training_step(self, batch, batch_idx, debug=True):
        self._check_device_for_extra_params()
        if debug:
            with open(Path(__file__).parent / f"batch_{batch_idx}.pkl", "wb") as f:
                pkl.dump(batch_idx, f)
                pkl.dump(batch, f)

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images = batch['image']
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        lmk = batch['landmark']
        lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        masks = batch['mask']
        masks = masks.view(-1, images.shape[-2], images.shape[-1])

        # coarse step
        # -- encoder
        if self.mode == DecaMode.DETAIL:
            with torch.no_grad():
                parameters = self.deca.E_flame(images)
        elif self.mode == DecaMode.COARSE:
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

        if self.mode == DecaMode.COARSE:
            ### shape constraints
            if self.deca.config.shape_constrain_type == 'same':
                # reshape shapecode => [B, K, n_shape]
                shapecode_idK = shapecode.view(self.batch_size, self.deca.K, -1)
                # get mean id
                shapecode_mean = torch.mean(shapecode_idK, dim=[1])
                shapecode_new = shapecode_mean[:, None, :].repeat(1, self.deca.K, 1)
                shapecode = shapecode_new.view(-1, self.deca.config.model.n_shape)
            elif self.deca.config.shape_constrain_type == 'exchange':
                '''
                make sure s0, s1 is something to make shape close
                the difference from ||so - s1|| is 
                the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
                '''
                new_order = np.array([np.random.permutation(self.deca.config.train_K) + i * self.deca.config.train_K for i in range(self.deca.config.batch_size_train)])
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
                # import ipdb; ipdb.set_trace()

        # -- detail
        if self.mode == DecaMode.DETAIL:
            detailcode = self.deca.E_detail(images)

            if self.deca.config.detail_constrain_type == 'exchange':
                '''
                make sure s0, s1 is something to make shape close
                the difference from ||so - s1|| is 
                the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
                '''
                # new_order = np.array(
                #     [np.random.permutation(self.deca.config.K) + i * self.deca.config.K for i in range(self.deca.config.batch_size)])
                new_order = np.array(
                    [np.random.permutation(self.deca.config.train_K) + i * self.deca.config.train_K for i in range(self.deca.config.batch_size_train)])
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


        batch_size = images.shape[0]
        # -- decoder
        # FLAME - world space
        verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                     pose_params=posecode)
        # world to camera
        trans_verts = self.util.batch_orth_proj(verts, cam)
        predicted_landmarks = self.util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]

        albedo = self.deca.flametex(texcode)

        # ------ rendering
        ops = self.deca.render(verts, trans_verts, albedo, lightcode)
        # mask
        mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(batch_size, -1, -1, -1), ops['grid'].detach(),
                                      align_corners=False)
        # images
        predicted_images = ops['images'] * mask_face_eye * ops['alpha_images']

        if self.deca.config.useSeg:
            masks = masks[:, None, :, :]
        else:
            masks = mask_face_eye * ops['alpha_images']

        if self.mode == DecaMode.DETAIL:
            uv_z = self.deca.D_detail(torch.cat([posecode[:, 3:], expcode, detailcode], dim=1))
            # render detail
            uv_detail_normals, uv_coarse_vertices = self.deca.displacement2normal(uv_z, verts, ops['normals'])
            uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
            uv_texture = albedo.detach() * uv_shading
            predicted_detailed_image = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)

            # --- extract texture
            uv_pverts = self.deca.render.world2uv(trans_verts).detach()
            uv_gt = F.grid_sample(torch.cat([images, masks], dim=1), uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2],
                                  mode='bilinear')
            uv_texture_gt = uv_gt[:, :3, :, :].detach()
            uv_mask_gt = uv_gt[:, 3:, :, :].detach()
            # self-occlusion
            normals = self.util.vertex_normals(trans_verts, self.deca.render.faces.expand(batch_size, -1, -1))
            uv_pnorm = self.deca.render.world2uv(normals)

            uv_mask = (uv_pnorm[:, -1, :, :] < -0.05).float().detach()
            uv_mask = uv_mask[:, None, :, :]
            ## combine masks
            uv_vis_mask = uv_mask_gt * uv_mask * self.deca.uv_face_eye_mask
        else:
            uv_detail_normals = None
            predicted_detailed_image = None

        #### ----------------------- Losses
        losses = {}

        ############################# base shape
        if self.mode == DecaMode.COARSE or (self.mode == DecaMode.DETAIL and self.deca.config.train_coarse):

            # landmark losses (only useful if coarse model is being trained
            if self.deca.config.useWlmk:
                losses['landmark'] = self.lossfunc.weighted_landmark_loss(predicted_landmarks, lmk) * self.deca.config.lmk_weight
            else:
                losses['landmark'] = self.lossfunc.landmark_loss(predicted_landmarks, lmk) * self.deca.config.lmk_weight
            # losses['eye_distance'] = self.lossfunc.eyed_loss(predicted_landmarks, lmk) * self.deca.config.lmk_weight * 2
            losses['eye_distance'] = self.lossfunc.eyed_loss(predicted_landmarks, lmk) * self.deca.config.eyed
            losses['lip_distance'] = self.lossfunc.eyed_loss(predicted_landmarks, lmk) * self.deca.config.lipd

            # photometric loss
            losses['photometric_texture'] = (masks * (predicted_images - images).abs()).mean() * self.deca.config.photow

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

        ############################### details
        if self.mode == DecaMode.DETAIL:
            for pi in range(3):  # self.deca.face_attr_mask.shape[0]):
                # if pi==0:
                new_size = 256
                # else:
                #     new_size = 128
                # if self.deca.config.uv_size != 256:
                #     new_size = 128
                uv_texture_patch = F.interpolate(uv_texture[:, :, self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                                                 self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]],
                                                 [new_size, new_size], mode='bilinear')
                uv_texture_gt_patch = F.interpolate(
                    uv_texture_gt[:, :, self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                    self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
                uv_vis_mask_patch = F.interpolate(uv_vis_mask[:, :, self.deca.face_attr_mask[pi][2]:self.deca.face_attr_mask[pi][3],
                                                  self.deca.face_attr_mask[pi][0]:self.deca.face_attr_mask[pi][1]],
                                                  [new_size, new_size], mode='bilinear')

                losses['detail_l1_{}'.format(pi)] = (
                                                                uv_texture_patch * uv_vis_mask_patch - uv_texture_gt_patch * uv_vis_mask_patch).abs().mean() * \
                                                    self.deca.config.sfsw[pi]
                losses['detail_mrf_{}'.format(pi)] = self.deca.perceptual_loss(uv_texture_patch * uv_vis_mask_patch,
                                                                          uv_texture_gt_patch * uv_vis_mask_patch) * \
                                                     self.deca.config.sfsw[pi] * self.deca.config.mrfwr
                losses['photometric_detailed_texture'] = (masks * (
                            predicted_detailed_image - images).abs()).mean() * self.deca.config.photow

                if pi == 2:
                    uv_texture_gt_patch_ = uv_texture_gt_patch
                    uv_texture_patch_ = uv_texture_patch
                    uv_vis_mask_patch_ = uv_vis_mask_patch

            losses['z_reg'] = torch.mean(uv_z.abs()) * self.deca.config.zregw
            losses['z_diff'] = self.lossfunc.shading_smooth_loss(uv_shading) * self.deca.config.zdiffw
            nonvis_mask = (1 - self.util.binary_erosion(uv_vis_mask))
            losses['z_sym'] = (nonvis_mask * (uv_z - torch.flip(uv_z, [-1]).detach()).abs()).sum() * self.deca.config.zsymw

        else:
            uv_texture_gt_patch_ = None
            uv_texture_patch_ = None
            uv_vis_mask_patch_ = None


        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        # losses['all_loss'] = all_loss
        losses['loss'] = all_loss

        if batch_idx % 200:
            self._visualization_checkpoint(
                verts, trans_verts, ops,
                images, lmk, predicted_images, predicted_landmarks, masks, albedo,
                predicted_detailed_image, uv_detail_normals, batch_idx,
                uv_texture_patch_, uv_texture_gt_patch_, uv_vis_mask_patch_)

        # done by lightning
        # self.opt.zero_grad()
        # all_loss.backward()
        # self.opt.step()

        return losses

    def _visualization_checkpoint(self,  verts, trans_verts, ops, images, lmk, predicted_images,
                                  predicted_landmarks, masks,
                                  albedo, predicted_detailed_image, uv_detail_normals, batch_idx,
                                  uv_texture_patch=None, uv_texture_gt=None, uv_vis_mask_patch=None):
        # visualize
        # if iter % 200 == 1:
        # visind = np.arange(8)  # self.config.batch_size )
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

        visdict = {
            'inputs': images[visind],
            'landmarks_gt': self.util.tensor_vis_landmarks(images[visind], lmk[visind]),# , isScale=False),
            'landmarks': self.util.tensor_vis_landmarks(images[visind], predicted_landmarks[visind]),
            'shape': shape_images[visind],
            'predicted_images': predicted_images[visind],
            'albedo_images': ops['albedo_images'][visind],
            'mask': masks.repeat(1, 3, 1, 1)[visind],
            'albedo': albedo[visind],
            # details

        }
        if predicted_detailed_image is not None:
            visdict['detailed_images'] = predicted_detailed_image[visind]
            visdict['shape_detail_images'] = shape_detail_images[visind]
            visdict['detailed_images'] = predicted_detailed_image[visind]
            visdict['uv_detail_normals'] = uv_detail_normals[visind] * 0.5 + 0.5
            visdict['uv_texture_patch'] = uv_texture_patch[visind]
            visdict['uv_texture_gt'] = uv_texture_gt[visind]
            visdict['uv_vis_mask_patch'] = uv_vis_mask_patch[visind]

        savepath = '{}/{}/{}_{}.png'.format(self.inout_params.full_run_dir, 'train_images',
                                            self.current_epoch, batch_idx)
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        self.deca.visualize(visdict, savepath)


    # def training_step_end(self, *args, **kwargs):
        # iteration update
        # if iter > all_iter - self.start_iter - 1:
        #     self.start_iter = 0
        #     continue

        # if iter % 500 == 0:
        #     if self.deca.config.multi_gpu:
        #         torch.save(
        #             {
        #                 'E_flame': self.E_flame.module.state_dict(),
        #                 'E_detail': self.E_detail.module.state_dict(),
        #                 'D_detail': self.D_detail.module.state_dict(),
        #                 'opt': self.opt.state_dict(),
        #                 'epoch': epoch,
        #                 'iter': iter,
        #                 'all_iter': all_iter,
        #                 'batch_size': self.config.batch_size
        #             },
        #             os.path.join(self.config.savefolder, 'model' + '.tar')
        #         )
        #     else:
        #         torch.save(
        #             {
        #                 'E_flame': self.deca.E_flame.state_dict(),
        #                 'E_detail': self.deca.E_detail.state_dict(),
        #                 'D_detail': self.deca.D_detail.state_dict(),
        #                 'opt': self.opt.state_dict(),
        #                 'epoch': self.current_epoch,
        #                 # 'iter': iter,
        #                 'all_iter': all_iter,
        #                 'batch_size': self.deca.config.batch_size
        #             },
        #             os.path.join(self.deca.config.savefolder, 'model' + '.tar')
        #         )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['epoch'] = self.current_epoch
        checkpoint['iter'] = -1 # to be deprecated
        checkpoint['all_iter'] = -1 # to be deprecated
        checkpoint['batch_size'] = self.deca.config.batch_size

    # def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
    #                     strict: bool = True):



    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        trainable_params = []
        if self.mode == DecaMode.COARSE:
            trainable_params += [self.deca.E_flame.parameters()]
            trainable_params += [self.deca.D_detail.parameters()]
        elif self.mode == DecaMode.DETAIL:
            trainable_params += [self.deca.E_detail.parameters()]
            trainable_params += [self.deca.D_detail.parameters()]
        else:
            raise ValueError(f"Invalid deca mode: {self.mode}")

        if self.learning_params.optimizer == 'Adam':
            self.deca.opt = torch.optim.Adam(
                trainable_params,
                lr=self.learning_params.learning_rate,
                amsgrad=False)

        elif self.learning_params.optimizer == 'SGD':
            self.deca.opt = torch.optim.SGD(
                list(self.deca.E_detail.parameters()) + list(self.deca.D_detail.parameters()),
                lr=self.learning_params.learning_rate)

        return self.deca.opt


    def validation_step(self, *args, **kwargs):
        pass


def finetune_deca(data_params, learning_params, model_params, inout_params):

    fvdm = FaceVideoDataModule(Path(data_params.data_root), Path(data_params.data_root) / "processed",
                               data_params.processed_subfolder)
    dm = EmotionDataModule(fvdm, image_size=model_params.image_size,
                           with_landmarks=True, with_segmentations=True)
    dm.prepare_data()


    # index = 220
    # index = 120
    index = data_params.sequence_index
    annotation_list = data_params.annotation_list
    if index == -1:
        sequence_name = annotation_list[0]
        if annotation_list[0] == 'va':
            filter_pattern = 'VA_Set'
        elif annotation_list[0] == 'expr7':
            filter_pattern = 'Expression_Set'
    else:
        sequence_name = str(fvdm.video_list[index])
        filter_pattern = sequence_name
        if annotation_list[0] == 'va' and 'VA_Set' not in sequence_name:
            print("No GT for valence and arousal. Skipping")
            # sys.exit(0)
        if annotation_list[0] == 'expr7' and 'Expression_Set' not in sequence_name:
            print("No GT for expressions. Skipping")
            # sys.exit(0)

    deca = DecaModule(model_params, learning_params, inout_params)

    project_name = 'EMOCA_finetune'
    name = inout_params.name + '_' + str(filter_pattern) + "_" + \
           datetime.datetime.now().strftime("%b_%d_%Y_%H-%M-%S")

    train_data_loader = dm.train_dataloader(annotation_list, filter_pattern,
                                    # TODO: find a better! way to incorporate the K and the batch size
                                    batch_size=model_params.batch_size_train * model_params.train_K,
                                    num_workers=data_params.num_workers,
                                    split_ratio=data_params.split_ratio,
                                    split_style=data_params.split_style)
    val_data_loader = dm.val_dataloader(annotation_list, filter_pattern,
                                        # TODO: find a better! way to incorporate the K and the batch size
                                        batch_size=model_params.batch_size_val * model_params.val_K,
                                        num_workers=data_params.num_workers)

    # out_folder = Path(inout_params.output_dir) / name
    # out_folder.mkdir(parents=True)

    # wandb.init(project_name)
    # wandb_logger = WandbLogger(name=name, project=project_name)
    wandb_logger = None
    trainer = Trainer(gpus=1)
    # trainer = Trainer(gpus=1, logger=wandb_logger)
    # trainer.fit(deca, datamodule=dm)
    print("The training begins")
    trainer.fit(deca, train_dataloader=train_data_loader, val_dataloaders=[val_data_loader,])




import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="deca_finetune")
def main(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    # root_path = root / "Aff-Wild2_ready"
    root_path = root
    processed_data_path = root / "processed"
    # subfolder = 'processed_2020_Dec_21_00-30-03'
    subfolder = 'processed_2021_Jan_19_20-25-10'

    run_dir = cfg.inout.output_dir + "_" + datetime.datetime.now().strftime("%Y_%b_%d_%H-%M-%S")

    full_run_dir = Path(cfg.inout.output_dir) / run_dir
    full_run_dir.mkdir(parents=True)

    cfg["inout"]['full_run_dir'] = str(full_run_dir)

    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    finetune_deca(cfg['data'], cfg['learning'], cfg['model'], cfg['inout'])


if __name__ == "__main__":
    main()
