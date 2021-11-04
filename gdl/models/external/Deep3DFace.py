import sys, os
from pathlib import Path

import numpy as np

repo_dir = str(Path(__file__).parents[4])
sys.path += [repo_dir]
deep3d_face = str(Path(__file__).parents[4] / "Deep3DFaceRecon_pytorch")
sys.path += [deep3d_face]
from models import create_model

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import pytorch_lightning as pl
from gdl.utils.lightning_logging import _log_array_image, _log_wandb_image, _torch_image2np
from pytorch_lightning.loggers import WandbLogger
from models import create_model


class Deep3DFaceModule(pl.LightningModule):

    def __init__(self, model_params, learning_params, inout_params, stage_name=""):
        super().__init__()
        self.learning_params = learning_params
        self.inout_params = inout_params
        self.model_params = model_params
        self.stage_name = stage_name
        if len(self.stage_name) > 0:
            self.stage_name += "_"
        self.tddfa = Face3DDFAv2Wrapper(cfg=model_params.tddfa)

    def encode(self, batch):
        return self.tddfa.encode(batch)

    def decode(self, batch, values):
        return self.tddfa.decode(batch, values)

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


class Deep3DFaceWrapper(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model = create_model(cfg)
        self.model.setup(cfg)
        self.model.device = device
        self.model.parallelize()
        self.model.eval()
        self.visualizer = MyVisualizer(cfg)

    def forward(self, batch):
        values = self.encode(batch)
        values = self.decode(batch, values)
        return values

    def encode(self, batch):
        img = batch["image"]

        data = {
            'imgs': batch["image"], #TODO: resize, and otherwise process images here, however necessary
            'lms': batch["landmark"], # same for landmarks
        }
        self.model.set_input(data)  # unpack data from data loader

        self.model.test()
        #TODO: extract the 3DMM parameters somehow and add them to values
        values = {}

        return values

    def decode(self, batch, values):
        visuals = self.model.get_current_visuals()  # ge

        return visuals




