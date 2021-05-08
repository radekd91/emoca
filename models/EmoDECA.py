from .DECA import DecaModule, instantiate_deca
from .EmotionRecognitionModuleBase import EmotionRecognitionBase
from .MLP import MLP
import torch
import pytorch_lightning as pl
import numpy as np
from utils.other import class_from_str
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger


class EmoDECA(EmotionRecognitionBase):

    def __init__(self, config):
        super().__init__(config)
        # model_params = config.model.deca_cfg.model
        # learning_params = config.model.deca_cfg.learning
        # inout_params = config.model.deca_cfg.inout
        deca_checkpoint = config.model.deca_checkpoint
        deca_stage = config.model.deca_stage
        deca_checkpoint_kwargs = {
            "model_params": config.model.deca_cfg.model,
            "learning_params": config.model.deca_cfg.learning,
            "inout_params": config.model.deca_cfg.inout,
            "stage_name": "testing",
        }
        self.deca = instantiate_deca(config.model.deca_cfg, deca_stage , "test", deca_checkpoint, deca_checkpoint_kwargs)
        self.deca.inout_params.full_run_dir = config.inout.full_run_dir
        self._setup_deca(False)

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

        hidden_layer_sizes = config.model.num_mlp_layers * [in_size]

        out_size = 0
        if self.config.model.predict_expression:
            self.num_classes = 9
            out_size += self.num_classes
        if self.config.model.predict_valence:
            out_size += 1
        if self.config.model.predict_arousal:
            out_size += 1

        self.mlp = MLP(in_size, out_size, hidden_layer_sizes)

        # TODO: delete, this was moved to base
        # if 'va_activation' in config.model.keys():
        #     self.va_activation = None
        # else:
        #     self.va_activation = class_from_str(self.config.model.va_loss, F)
        # if 'exp_activation' in config.model.keys():
        #     self.exp_activation = F.log_softmax
        # else:
        #     self.exp_activation = class_from_str(self.config.model.exp_activation, F)
        #
        # self.va_loss = class_from_str(self.config.model.va_loss, F)
        # self.exp_loss = class_from_str(self.config.model.exp_loss, F)

    def _get_trainable_parameters(self):
        trainable_params = []
        if self.config.model.finetune_deca:
            trainable_params += self.deca._get_trainable_parameters()
        trainable_params += list(self.mlp.parameters())
        return trainable_params

    # def configure_optimizers(self):
    #     trainable_params = []
    #     if self.config.model.finetune_deca:
    #         trainable_params += self.deca._get_trainable_parameters()
    #     trainable_params += list(self.mlp.parameters())
    #
    #     if self.config.learning.optimizer == 'Adam':
    #         opt = torch.optim.Adam(
    #             trainable_params,
    #             lr=self.config.learning.learning_rate,
    #             amsgrad=False)
    #
    #     elif self.config.learning.optimizer == 'SGD':
    #         opt = torch.optim.SGD(
    #             trainable_params,
    #             lr=self.config.learning.learning_rate)
    #     else:
    #         raise ValueError(f"Unsupported optimizer: '{self.config.learning.optimizer}'")
    #
    #     optimizers = [opt]
    #     schedulers = []
    #     if 'learning_rate_decay' in self.config.learning.keys():
    #         scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.config.learning.learning_rate_decay)
    #         schedulers += [scheduler]
    #     if len(schedulers) == 0:
    #         return opt
    #
    #     return optimizers, schedulers

    def _setup_deca(self, train : bool):
        if self.config.model.finetune_deca:
            self.deca.train(train)
            self.deca.requires_grad_(True)
        else:
            self.deca.train(False)
            self.deca.requires_grad_(False)

    def train(self, mode=True):
        self._setup_deca(mode)
        self.mlp.train(mode)


    def forward(self, batch):
        values = self.deca.encode(batch)
        shapecode = values['shapecode']
        # texcode = values['texcode']
        expcode = values['expcode']
        posecode = values['posecode']
        if self.config.model.use_detail_code:
            detailcode = values['detailcode']
        else:
            detailcode = None

        global_pose = posecode[:, :3]
        jaw_pose = posecode[:, 3:]

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

        input = torch.cat(input_list, dim=1)
        output = self.mlp(input)

        out_idx = 0
        if self.config.model.predict_expression:
            expr_classification = output[:, out_idx:(out_idx + self.num_classes)]
            if self.exp_activation is not None:
                expr_classification = self.exp_activation(output[:, out_idx:(out_idx + self.num_classes)], dim=1)
            out_idx += self.num_classes
        else:
            expr_classification = None

        if self.config.model.predict_valence:
            valence = output[:, out_idx:(out_idx+1)]
            if self.v_activation is not None:
                valence = self.v_activation(valence)
            out_idx += 1
        else:
            valence = None

        if self.config.model.predict_arousal:
            arousal = output[:, out_idx:(out_idx+1)]
            if self.a_activation is not None:
                arousal = self.a_activation(output[:, out_idx:(out_idx + 1)])
            out_idx += 1
        else:
            arousal = None

        values["valence"] = valence
        values["arousal"] = arousal
        values["expr_classification"] = expr_classification
        return values

    # TODO: remove this, moved to base
    # def compute_loss(self,
    #                  valence_pred, arousal_pred, expr_classification_pred,
    #                  valence_gt, arousal_gt, expr_classification_gt,
    #                  class_weight):
    #     losses = {}
    #     metrics = {}
    #     if valence_pred is not None and arousal_pred is not None:
    #         va_pred = torch.cat([valence_pred, arousal_pred], dim=1)
    #         va_gt = torch.cat([valence_gt, arousal_gt], dim=1)
    #         losses["va"] = self.va_loss(va_pred, va_gt)
    #         metrics["va_mae"] = F.l1_loss(va_pred, va_gt)
    #         metrics["va_mse"] = F.mse_loss(va_pred, va_gt)
    #         metrics["va_rmse"] = torch.sqrt(F.mse_loss(va_pred, va_gt))
    #     elif valence_pred is not None:
    #         losses["v"] = self.va_loss(valence_pred, valence_gt)
    #         metrics["v_mae"] = F.l1_loss(valence_pred, valence_gt)
    #         metrics["v_mse"] = F.mse_loss(valence_pred, valence_gt)
    #         metrics["v_rmse"] = torch.sqrt(F.mse_loss(valence_pred, valence_gt))
    #     elif arousal_pred is not None:
    #         losses["a"] = self.va_loss(arousal_pred, arousal_gt)
    #         metrics["a_mae"] = F.l1_loss(arousal_pred, arousal_gt)
    #         metrics["a_mse"] = F.mse_loss(arousal_pred, arousal_gt)
    #         metrics["a_rmse"] = torch.sqrt(F.mse_loss(arousal_pred, arousal_gt))
    #
    #     if expr_classification_pred is not None:
    #         if self.config.model.expression_balancing:
    #             weight = class_weight
    #         else:
    #             weight = torch.ones_like(class_weight)
    #
    #         losses["expr"] = self.exp_loss(expr_classification_pred, expr_classification_gt[:, 0], weight)
    #         # metrics["expr_cross_entropy"] = F.cross_entropy(expr_classification_pred, expr_classification_gt[:, 0], torch.ones_like(class_weight))
    #         # metrics["expr_weighted_cross_entropy"] = F.cross_entropy(expr_classification_pred, expr_classification_gt[:, 0], class_weight)
    #         metrics["expr_nll"] = F.nll_loss(expr_classification_pred, expr_classification_gt[:, 0], torch.ones_like(class_weight))
    #         metrics["expr_weighted_nll"] = F.nll_loss(expr_classification_pred, expr_classification_gt[:, 0], class_weight)
    #
    #     loss = 0.
    #     for key, value in losses.items():
    #         loss += value
    #
    #     losses["total"] = loss
    #
    #     return losses, metrics

    # def training_step(self, batch, batch_idx):
    #     values = self.forward(batch)
    #     valence_pred = values["valence"]
    #     arousal_pred = values["arousal"]
    #     expr_classification_pred = values["expr_classification"]
    #
    #     valence_gt = batch["va"][:, 0:1]
    #     arousal_gt = batch["va"][:, 1:2]
    #     expr_classification_gt = batch["affectnetexp"]
    #     if "expression_weight" in batch.keys():
    #         class_weight = batch["expression_weight"][0]
    #     else:
    #         class_weight = None
    #
    #     losses, metrics = self.compute_loss(valence_pred, arousal_pred, expr_classification_pred,
    #                       valence_gt, arousal_gt, expr_classification_gt, class_weight)
    #
    #     self._log_losses_and_metrics(losses, metrics, "train")
    #     total_loss = losses["total"]
    #     return total_loss
    #
    # def validation_step(self, batch, batch_idx, dataloader_idx=None):
    #     values = self.forward(batch)
    #     valence_pred = values["valence"]
    #     arousal_pred = values["arousal"]
    #     expr_classification_pred = values["expr_classification"]
    #
    #     valence_gt = batch["va"][:, 0:1]
    #     arousal_gt = batch["va"][:, 1:2]
    #     expr_classification_gt = batch["affectnetexp"]
    #     if "expression_weight" in batch.keys():
    #         class_weight = batch["expression_weight"][0]
    #     else:
    #         class_weight = None
    #
    #     losses, metrics = self.compute_loss(valence_pred, arousal_pred, expr_classification_pred,
    #                                         valence_gt, arousal_gt, expr_classification_gt, class_weight)
    #
    #     self._log_losses_and_metrics(losses, metrics, "val")
    #     total_loss = losses["total"]
    #     return total_loss

    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        valence_pred = output_values["valence"]
        arousal_pred = output_values["arousal"]
        expr_classification_pred = output_values["expr_classification"]

        valence_gt = input_batch["va"][:, 0:1]
        arousal_gt = input_batch["va"][:, 1:2]
        expr_classification_gt = input_batch["affectnetexp"]

        with torch.no_grad():
            values = self.deca.decode(output_values)

        self.deca.logger = self.logger
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
            self.logger.log_metrics(visdict)
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
