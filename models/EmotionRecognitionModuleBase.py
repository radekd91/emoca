import torch
import pytorch_lightning as pl
import numpy as np
from utils.other import class_from_str
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from layers.losses.EmoNetLoss import get_emonet
from utils.emotion_metrics import *
from torch.nn.functional import mse_loss, cross_entropy, nll_loss, l1_loss, log_softmax
import sys


def loss_from_cfg(config, loss_name):
    if loss_name in config.model.keys():
        if isinstance(config.model[loss_name], str):
            loss = class_from_str(config.model[loss_name], sys.modules[__name__])
        else:
            cont = OmegaConf.to_container(config.model[loss_name])
            if isinstance(cont, list):
                loss = {name: 1. for name in cont}
            elif isinstance(cont, dict):
                loss = cont
            else:
                raise ValueError(f"Unkown type of loss '{type(cont)}' for loss '{loss_name}'")
    else:
        loss = None
    return loss


class EmotionRecognitionBase(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if 'v_activation' in config.model.keys():
            self.v_activation = class_from_str(self.config.model.v_activation, sys.modules[__name__])
        else:
            self.v_activation = None

        if 'a_activation' in config.model.keys():
            self.a_activation = class_from_str(self.config.model.a_activation, sys.modules[__name__])
        else:
            self.a_activation = None

        if 'exp_activation' in config.model.keys():
            self.exp_activation = class_from_str(self.config.model.exp_activation, sys.modules[__name__])
        else:
            self.exp_activation = F.log_softmax

        self.va_loss = loss_from_cfg(config, 'va_loss')
        self.v_loss = loss_from_cfg(config, 'v_loss')
        self.a_loss = loss_from_cfg(config, 'a_loss')
        self.exp_loss = loss_from_cfg(config, 'exp_loss')
        # if 'va_loss' in config.model.keys():
        #     if isinstance(self.config.model.va_loss, str):
        #         self.va_loss = class_from_str(self.config.model.va_loss, sys.modules[__name__])
        #     else:
        #         cont = OmegaConf.to_container(self.config.model.va_loss)
        #         if isinstance(cont, list):
        #             self.va_loss = {name: 1. for name in cont}
        #         elif isinstance(cont, dict):
        #             self.va_loss = cont
        #         else:
        #             raise ValueError(f"Unkown type of loss {type(cont)}")
        #
        # else:
        #     self.va_loss = None

        # if 'v_loss' in config.model.keys():
        #     self.v_loss = class_from_str(self.config.model.v_loss, sys.modules[__name__])
        # else:
        #     self.v_loss = None
        #
        # if 'a_loss' in config.model.keys():
        #     self.a_loss = class_from_str(self.config.model.a_loss, sys.modules[__name__])
        # else:
        #     self.a_loss = None
        #
        # if 'exp_loss' in config.model.keys():
        #     self.exp_loss = class_from_str(self.config.model.exp_loss, sys.modules[__name__])
        # else:
        #     self.exp_loss = None


    def forward(self, image):
        raise NotImplementedError()

    def _get_trainable_parameters(self):
        raise NotImplementedError()

    def configure_optimizers(self):
        trainable_params = []
        trainable_params += list(self._get_trainable_parameters())

        if self.config.learning.optimizer == 'Adam':
            opt = torch.optim.Adam(
                trainable_params,
                lr=self.config.learning.learning_rate,
                amsgrad=False)

        elif self.config.learning.optimizer == 'SGD':
            opt = torch.optim.SGD(
                trainable_params,
                lr=self.config.learning.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: '{self.config.learning.optimizer}'")

        optimizers = [opt]
        schedulers = []
        if 'learning_rate_decay' in self.config.learning.keys():
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.config.learning.learning_rate_decay)
            schedulers += [scheduler]
        if len(schedulers) == 0:
            return opt

        return optimizers, schedulers

    def _get_step_loss_weights(self, training):
        va_loss_weights = {}
        for key in self.v_loss:
            va_loss_weights[key] = self.v_loss[key]

        for key in self.a_loss:
            va_loss_weights[key] = self.a_loss[key]

        for key in self.va_loss:
            va_loss_weights[key] = self.va_loss[key]

        # if training:
        #     return va_loss_weights

        n_terms = len(va_loss_weights)

        if 'va_loss_scheme' in self.config.model.keys():
            if not training and self.config.model.va_loss_scheme == 'shake':
                for key in va_loss_weights:
                    va_loss_weights[key] = np.random.rand(1)[0]
                total_w = 0.
                for key in va_loss_weights:
                    total_w += va_loss_weights[key]
                for key in va_loss_weights:
                    va_loss_weights[key] /= total_w
            elif self.config.model.va_loss_scheme == 'norm':
                total_w = 0.
                for key in va_loss_weights:
                    total_w += va_loss_weights[key]

                for key in va_loss_weights:
                    va_loss_weights[key] /= total_w
        return va_loss_weights


    def compute_loss(self,
                     valence_pred, arousal_pred, expr_classification_pred,
                     valence_gt, arousal_gt, expr_classification_gt,
                     class_weight,
                     training=True):
        losses = {}
        metrics = {}

        weights = self._get_step_loss_weights(training)

        if valence_pred is not None:
            metrics["v_mae"] = F.l1_loss(valence_pred, valence_gt)
            metrics["v_mse"] = F.mse_loss(valence_pred, valence_gt)
            metrics["v_rmse"] = torch.sqrt(metrics["v_mse"])
            metrics["v_pcc"] = PCC_torch(valence_pred, valence_gt, batch_first=False)
            metrics["v_ccc"] = CCC_torch(valence_pred, valence_gt, batch_first=False)
            metrics["v_sagr"] = SAGR_torch(valence_pred, valence_gt)
            # metrics["v_icc"] = ICC_torch(valence_pred, valence_gt)
            if self.v_loss is not None:
                if callable(self.v_loss):
                    losses["v"] = self.v_loss(valence_pred, valence_gt)
                elif isinstance(self.v_loss, dict):
                    for name, weight in self.v_loss.items():
                        # losses[name] = metrics[name]*weight
                        losses[name] = metrics[name]*weights[name]
                else:
                    raise RuntimeError(f"Uknown expression loss '{self.v_loss}'")

        if arousal_pred is not None:
            metrics["a_mae"] = F.l1_loss(arousal_pred, arousal_gt)
            metrics["a_mse"] = F.mse_loss(arousal_pred, arousal_gt)
            metrics["a_rmse"] = torch.sqrt( metrics["a_mse"])
            metrics["a_pcc"] = PCC_torch(arousal_pred, arousal_gt, batch_first=False)
            metrics["a_ccc"] = CCC_torch(arousal_pred, arousal_gt, batch_first=False)
            metrics["a_sagr"] = SAGR_torch(arousal_pred, arousal_gt)
            # metrics["a_icc"] = ICC_torch(arousal_pred, arousal_gt)
            if self.a_loss is not None:
                if callable(self.a_loss):
                    losses["a"] = self.a_loss(arousal_pred, arousal_gt)
                elif isinstance(self.a_loss, dict):
                    for name, weight in self.a_loss.items():
                        # losses[name] = metrics[name]*weight
                        losses[name] = metrics[name]*weights[name]
                else:
                    raise RuntimeError(f"Uknown expression loss '{self.a_loss}'")

        if valence_pred is not None and arousal_pred is not None:
            va_pred = torch.cat([valence_pred, arousal_pred], dim=1)
            va_gt = torch.cat([valence_gt, arousal_gt], dim=1)
            metrics["va_mae"] = F.l1_loss(va_pred, va_gt)
            metrics["va_mse"] = F.mse_loss(va_pred, va_gt)
            metrics["va_rmse"] = torch.sqrt(metrics["va_mse"])
            metrics["va_lpcc"] = (1. - 0.5*(metrics["a_pcc"] + metrics["v_pcc"]))[0][0]
            metrics["va_lccc"] = (1. - 0.5*(metrics["a_ccc"] + metrics["v_ccc"]))[0][0]
            if self.va_loss is not None:
                if callable(self.va_loss):
                    losses["va"] = self.va_loss(va_pred, va_gt)
                elif isinstance(self.va_loss, dict):
                    for name, weight in self.va_loss.items():
                        # losses[name] = metrics[name]*weight
                        losses[name] = metrics[name] * weights[name]
                else:
                    raise RuntimeError(f"Uknown expression loss '{self.va_loss}'")

        if expr_classification_pred is not None:
            if self.config.model.expression_balancing:
                weight = class_weight
            else:
                weight = torch.ones_like(class_weight)

            # metrics["expr_cross_entropy"] = F.cross_entropy(expr_classification_pred, expr_classification_gt[:, 0], torch.ones_like(class_weight))
            # metrics["expr_weighted_cross_entropy"] = F.cross_entropy(expr_classification_pred, expr_classification_gt[:, 0], class_weight)
            metrics["expr_nll"] = F.nll_loss(expr_classification_pred, expr_classification_gt[:, 0],
                                             torch.ones_like(class_weight))
            metrics["expr_weighted_nll"] = F.nll_loss(expr_classification_pred, expr_classification_gt[:, 0],
                                                      class_weight)
            metrics["expr_acc"] = ACC_torch( torch.argmax(expr_classification_pred, dim=1), expr_classification_gt[:, 0])


            if self.exp_loss is not None:
                if callable(self.exp_loss):
                    losses["expr"] = self.exp_loss(expr_classification_pred, expr_classification_gt[:, 0], weight)
                elif isinstance(self.exp_loss, dict):
                    for name, weight in self.exp_loss.items():
                        losses[name] = metrics[name]*weight
                else:
                    raise RuntimeError(f"Uknown expression loss '{self.exp_loss}'")

        loss = 0.
        for key, value in losses.items():
            loss += value

        losses["total"] = loss

        return losses, metrics

    def training_step(self, batch, batch_idx):
        values = self.forward(batch)
        valence_pred = values["valence"]
        arousal_pred = values["arousal"]
        expr_classification_pred = values["expr_classification"]

        valence_gt = batch["va"][:, 0:1]
        arousal_gt = batch["va"][:, 1:2]
        expr_classification_gt = batch["affectnetexp"]
        if "expression_weight" in batch.keys():
            class_weight = batch["expression_weight"][0]
        else:
            class_weight = None

        losses, metrics = self.compute_loss(valence_pred, arousal_pred, expr_classification_pred,
                                            valence_gt, arousal_gt, expr_classification_gt, class_weight, training=True)

        self._log_losses_and_metrics(losses, metrics, "train")
        total_loss = losses["total"]
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        values = self.forward(batch)
        valence_pred = values["valence"]
        arousal_pred = values["arousal"]
        expr_classification_pred = values["expr_classification"]

        valence_gt = batch["va"][:, 0:1]
        arousal_gt = batch["va"][:, 1:2]
        expr_classification_gt = batch["affectnetexp"]
        if "expression_weight" in batch.keys():
            class_weight = batch["expression_weight"][0]
        else:
            class_weight = None

        losses, metrics = self.compute_loss(valence_pred, arousal_pred, expr_classification_pred,
                                            valence_gt, arousal_gt, expr_classification_gt, class_weight, training=False)

        self._log_losses_and_metrics(losses, metrics, "val")
        total_loss = losses["total"]
        return total_loss

    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        values = self.forward(batch)
        valence_pred = values["valence"]
        arousal_pred = values["arousal"]
        expr_classification_pred = values["expr_classification"]
        if "expression_weight" in batch.keys():
            class_weight = batch["expression_weight"][0]
        else:
            class_weight = None

        if "va" in batch.keys():
            valence_gt = batch["va"][:, 0:1]
            arousal_gt = batch["va"][:, 1:2]
            expr_classification_gt = batch["affectnetexp"]
            losses, metrics = self.compute_loss(valence_pred, arousal_pred, expr_classification_pred,
                                                valence_gt, arousal_gt, expr_classification_gt, class_weight,
                                                training=False)

        if self.config.learning.test_vis_frequency > 0:
            if batch_idx % self.config.learning.test_vis_frequency == 0:
                self._test_visualization(values, batch, batch_idx, dataloader_idx=dataloader_idx)

            self._log_losses_and_metrics(losses, metrics, "test")
            self.logger.log_metrics({f"test_step": batch_idx})

    def _log_losses_and_metrics(self, losses, metrics, stage):
        if stage in ["train", "val"]:
            on_epoch = True
            on_step = False
            self.log_dict({f"{stage}_loss_" + key: value for key, value in losses.items()}, on_epoch=on_epoch,
                          on_step=on_step)
            self.log_dict({f"{stage}_metric_" + key: value for key, value in metrics.items()}, on_epoch=on_epoch,
                          on_step=on_step)
        else:
            on_epoch = False
            on_step = True
            self.logger.log_metrics({f"{stage}_loss_" + key: value.detach().cpu() for key, value in
                                     losses.items()})  # , on_epoch=on_epoch, on_step=on_step)
            #
            self.logger.log_metrics({f"{stage}_metric_" + key: value.detach().cpu() for key, value in
                                     metrics.items()})  # , on_epoch=on_epoch, on_step=on_step)

