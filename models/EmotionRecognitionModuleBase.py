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
import adabound


def loss_from_cfg(config, loss_name):
    if loss_name in config.keys():
        if isinstance(config[loss_name], str):
            loss = class_from_str(config[loss_name], sys.modules[__name__])
        else:
            cont = OmegaConf.to_container(config[loss_name])
            if isinstance(cont, list):
                loss = {name: 1. for name in cont}
            elif isinstance(cont, dict):
                loss = cont
            else:
                raise ValueError(f"Unkown type of loss '{type(cont)}' for loss '{loss_name}'")
    else:
        loss = None
    return loss


class EmotionRecognitionBaseModule(pl.LightningModule):

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

        self.va_loss = loss_from_cfg(config.model, 'va_loss')
        self.v_loss = loss_from_cfg(config.model, 'v_loss')
        self.a_loss = loss_from_cfg(config.model, 'a_loss')
        self.exp_loss = loss_from_cfg(config.model, 'exp_loss')

        # self.val_conf_mat = pl.metrics.ConfusionMatrix(self.num_classes, 'true')
        # self.val_conf_mat = pl.metrics.ConfusionMatrix(self.num_classes, 'true')


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
        elif self.config.learning.optimizer == 'AdaBound':
            opt = adabound.AdaBound(
                trainable_params,
                lr=self.config.learning.learning_rate,
                final_lr=self.config.learning.final_learning_rate
            )

        elif self.config.learning.optimizer == 'SGD':
            opt = torch.optim.SGD(
                trainable_params,
                lr=self.config.learning.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: '{self.config.learning.optimizer}'")

        optimizers = [opt]
        schedulers = []

        opt_dict = {}
        opt_dict['optimizer'] = opt
        if 'learning_rate_patience' in self.config.learning.keys():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                                   patience=self.config.learning.learning_rate_patience,
                                                                   factor=self.config.learning.learning_rate_decay,
                                                                   mode=self.config.learning.lr_sched_mode)
            schedulers += [scheduler]
            opt_dict['lr_scheduler'] = scheduler
            opt_dict['monitor'] = 'val_loss_total'
        elif 'learning_rate_decay' in self.config.learning.keys():
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.config.learning.learning_rate_decay)
            opt_dict['lr_scheduler'] = scheduler
            schedulers += [scheduler]
        return opt_dict
        #
        # if len(schedulers) == 0:
        #     return opt
        #
        # return optimizers, schedulers

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


    # def validation_epoch_end(self, outputs) -> None:
    #     if isinstance(self.logger, WandbLogger):
    #         import wandb
    #
    #         {"conf_mat": wandb.plot.confusion_matrix(probs=None,
    #                                 y_true = ground_truth, preds = predictions,
    #                                 class_names = class_names}
    #
    #         self.wandb_logger


    def _compute_loss(self,
                      pred,
                      gt,
                      class_weight,
                      training=True,
                      pred_prefix="",
                      valence_sample_weight=None,
                      arousal_sample_weight=None,
                      va_sample_weight=None,
                      expression_sample_weight=None
                      ):
        losses = {}
        metrics = {}

        scheme = None if 'va_loss_scheme' not in self.config.model.keys() else self.config.model.va_loss_scheme
        loss_term_weights = _get_step_loss_weights(self.v_loss, self.a_loss, self.va_loss, scheme, training)

        if 'continuous_va_balancing' in self.config.model.keys():
            if self.config.model.continuous_va_balancing == '1d':
                v_weight = valence_sample_weight
                a_weight = arousal_sample_weight
            elif self.config.model.continuous_va_balancing == '2d':
                v_weight = va_sample_weight
                a_weight = va_sample_weight
            elif self.config.model.continuous_va_balancing == 'expr':
                v_weight = expression_sample_weight
                a_weight = expression_sample_weight
            else:
                raise RuntimeError(f"Invalid continuous affect balancing"
                                   f" '{self.config.model.continuous_va_balancing}'")
        else:
            v_weight = None
            a_weight = None

        losses, metrics = v_or_a_loss(self.v_loss, pred, gt, loss_term_weights, metrics, losses, "valence",
                                      pred_prefix=pred_prefix, permit_dropping_corr=not training,
                                      sample_weights=v_weight)
        losses, metrics = v_or_a_loss(self.a_loss, pred, gt, loss_term_weights, metrics, losses, "arousal",
                                      pred_prefix=pred_prefix, permit_dropping_corr=not training,
                                      sample_weights=a_weight)
        losses, metrics = va_loss(self.va_loss, pred, gt, loss_term_weights, metrics, losses, pred_prefix=pred_prefix,
                                  permit_dropping_corr=not training)


        losses, metrics = exp_loss(self.exp_loss, pred, gt, class_weight, metrics, losses,
                                   self.config.model.expression_balancing, self.num_classes, pred_prefix=pred_prefix)

        # if not training:
        #     self.val_conf_mat(pred[pred_prefix + "expr_classification"], gt["expr_classification"][:, 0])

        # if pred[pred_prefix + "valence"] is not None:
        #     metrics[pred_prefix + "v_mae"] = F.l1_loss(pred[pred_prefix + "valence"], gt["valence"])
        #     metrics[pred_prefix + "v_mse"] = F.mse_loss(pred[pred_prefix + "valence"], gt["valence"])
        #     metrics[pred_prefix + "v_rmse"] = torch.sqrt(metrics[pred_prefix + "v_mse"])
        #     metrics[pred_prefix + "v_pcc"] = PCC_torch(pred[pred_prefix + "valence"], gt["valence"], batch_first=False)
        #     metrics[pred_prefix + "v_ccc"] = CCC_torch(pred[pred_prefix + "valence"], gt["valence"], batch_first=False)
        #     metrics[pred_prefix + "v_sagr"] = SAGR_torch(pred[pred_prefix + "valence"], gt["valence"])
        #     # metrics["v_icc"] = ICC_torch(pred["valence"], gt["valence"])
        #     if self.v_loss is not None:
        #         if callable(self.v_loss):
        #             losses["v"] = self.v_loss(pred[pred_prefix + "valence"], gt["valence"])
        #         elif isinstance(self.v_loss, dict):
        #             for name, weight in self.v_loss.items():
        #                 # losses[name] = metrics[name]*weight
        #                 losses[name] = metrics[pred_prefix + name]*weights[name]
        #         else:
        #             raise RuntimeError(f"Uknown expression loss '{self.v_loss}'")
        #
        # if pred[pred_prefix + "arousal"] is not None:
        #     metrics[pred_prefix + "a_mae"] = F.l1_loss(pred[pred_prefix + "arousal"], gt["arousal"])
        #     metrics[pred_prefix + "a_mse"] = F.mse_loss(pred[pred_prefix + "arousal"], gt["arousal"])
        #     metrics[pred_prefix + "a_rmse"] = torch.sqrt( metrics[pred_prefix + "a_mse"])
        #     metrics[pred_prefix + "a_pcc"] = PCC_torch(pred[pred_prefix + "arousal"], gt["arousal"], batch_first=False)
        #     metrics[pred_prefix + "a_ccc"] = CCC_torch(pred[pred_prefix + "arousal"], gt["arousal"], batch_first=False)
        #     metrics[pred_prefix + "a_sagr"] = SAGR_torch(pred[pred_prefix + "arousal"], gt["arousal"])
        #     # metrics["a_icc"] = ICC_torch(pred["arousal"], gt["arousal"])
        #     if self.a_loss is not None:
        #         if callable(self.a_loss):
        #             losses[pred_prefix + "a"] = self.a_loss(pred[pred_prefix + "arousal"], gt["arousal"])
        #         elif isinstance(self.a_loss, dict):
        #             for name, weight in self.a_loss.items():
        #                 # losses[name] = metrics[name]*weight
        #                 losses[pred_prefix + name] = metrics[pred_prefix + name]*weights[name]
        #         else:
        #             raise RuntimeError(f"Uknown expression loss '{self.a_loss}'")
        #
        # if pred[pred_prefix + "valence"] is not None and pred[pred_prefix + "arousal"] is not None:
        #     va_pred = torch.cat([pred[pred_prefix + "valence"], pred[pred_prefix + "arousal"]], dim=1)
        #     va_gt = torch.cat([gt["valence"], gt["arousal"]], dim=1)
        #     metrics[pred_prefix + "va_mae"] = F.l1_loss(va_pred, va_gt)
        #     metrics[pred_prefix + "va_mse"] = F.mse_loss(va_pred, va_gt)
        #     metrics[pred_prefix + "va_rmse"] = torch.sqrt(metrics[pred_prefix + "va_mse"])
        #     metrics[pred_prefix + "va_lpcc"] = (1. - 0.5*(metrics[pred_prefix + "a_pcc"] + metrics[pred_prefix + "v_pcc"]))[0][0]
        #     metrics[pred_prefix + "va_lccc"] = (1. - 0.5*(metrics[pred_prefix + "a_ccc"] + metrics[pred_prefix + "v_ccc"]))[0][0]
        #     if self.va_loss is not None:
        #         if callable(self.va_loss):
        #             losses[pred_prefix + "va"] = self.va_loss(va_pred, va_gt)
        #         elif isinstance(self.va_loss, dict):
        #             for name, weight in self.va_loss.items():
        #                 # losses[name] = metrics[name]*weight
        #                 losses[pred_prefix + name] = metrics[pred_prefix + name] * weights[name]
        #         else:
        #             raise RuntimeError(f"Uknown expression loss '{self.va_loss}'")
        #
        # if pred[pred_prefix + "expr_classification"] is not None:
        #     if self.config.model.expression_balancing:
        #         weight = class_weight
        #     else:
        #         weight = torch.ones_like(class_weight)
        #
        #     # metrics["expr_cross_entropy"] = F.cross_entropy(pred["expr_classification"], gt["expr_classification"][:, 0], torch.ones_like(class_weight))
        #     # metrics["expr_weighted_cross_entropy"] = F.cross_entropy(pred["expr_classification"], gt["expr_classification"][:, 0], class_weight)
        #     metrics[pred_prefix + "expr_nll"] = F.nll_loss(pred[pred_prefix + "expr_classification"],
        #                                                    gt["expr_classification"][:, 0],
        #                                      torch.ones_like(class_weight))
        #     metrics[pred_prefix + "expr_weighted_nll"] = F.nll_loss(pred[pred_prefix + "expr_classification"],
        #                                                             gt["expr_classification"][:, 0],
        #                                               class_weight)
        #     metrics[pred_prefix + "expr_acc"] = ACC_torch( torch.argmax(pred[pred_prefix + "expr_classification"], dim=1),
        #                                                    gt["expr_classification"][:, 0])
        #
        #
        #     if self.exp_loss is not None:
        #         if callable(self.exp_loss):
        #             losses[pred_prefix + "expr"] = self.exp_loss(pred[pred_prefix + "expr_classification"], gt["expr_classification"][:, 0], weight)
        #         elif isinstance(self.exp_loss, dict):
        #             for name, weight in self.exp_loss.items():
        #                 losses[pred_prefix + name] = metrics[pred_prefix + name]*weight
        #         else:
        #             raise RuntimeError(f"Uknown expression loss '{self.exp_loss}'")

        return losses, metrics

    def compute_loss(self,
                     pred,
                     gt,
                     class_weight,
                     valence_sample_weight=None,
                     arousal_sample_weight=None,
                     va_sample_weight=None,
                     expression_sample_weight=None,
                     training=True):
        losses, metrics = self._compute_loss(pred, gt, class_weight, training,
                                             valence_sample_weight=valence_sample_weight,
                                             arousal_sample_weight=arousal_sample_weight,
                                             va_sample_weight=va_sample_weight,
                                             expression_sample_weight=expression_sample_weight,
                                             )
        loss = 0.
        for key, value in losses.items():
            if value.ndim == 0:
                loss += value
            elif value.ndim == 1:
                loss += value[0]
            else:
                raise RuntimeError(f"Invalid loss shape for term '{key}': {value.shape}")
        losses["total"] = loss
        return losses, metrics

    def training_step(self, batch, batch_idx):
        values = self.forward(batch)
        # valence_pred = values["valence"]
        # arousal_pred = values["arousal"]
        # expr_classification_pred = values["expr_classification"]

        valence_gt = batch["va"][:, 0:1]
        arousal_gt = batch["va"][:, 1:2]
        expr_classification_gt = batch["affectnetexp"]
        if "expression_weight" in batch.keys():
            class_weight = batch["expression_weight"][0]
        else:
            class_weight = None

        valence_sample_weight = batch["valence_sample_weight"] if "valence_sample_weight" in batch.keys() else None
        arousal_sample_weight = batch["arousal_sample_weight"] if "arousal_sample_weight" in batch.keys() else None
        va_sample_weight = batch["va_sample_weight"] if "va_sample_weight" in batch.keys() else None
        expression_sample_weight = batch["expression_sample_weight"] if "expression_sample_weight" in batch.keys() else None

        gt = {}
        gt["valence"] = valence_gt
        gt["arousal"] = arousal_gt
        gt["expr_classification"] = expr_classification_gt

        pred = values
        losses, metrics = self.compute_loss(pred, gt, class_weight, training=True,
                                            valence_sample_weight=valence_sample_weight,
                                            arousal_sample_weight=arousal_sample_weight,
                                            va_sample_weight=va_sample_weight,
                                            expression_sample_weight=expression_sample_weight,
                                            )

        self._log_losses_and_metrics(losses, metrics, "train")
        total_loss = losses["total"]
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        values = self.forward(batch)
        # valence_pred = values["valence"]
        # arousal_pred = values["arousal"]
        # expr_classification_pred = values["expr_classification"]

        valence_gt = batch["va"][:, 0:1]
        arousal_gt = batch["va"][:, 1:2]
        expr_classification_gt = batch["affectnetexp"]
        if "expression_weight" in batch.keys():
            class_weight = batch["expression_weight"][0]
        else:
            class_weight = None

        gt = {}
        gt["valence"] = valence_gt
        gt["arousal"] = arousal_gt
        gt["expr_classification"] = expr_classification_gt

        pred = values
        # pred = {}
        # pred["valence"] = valence_pred
        # pred["arousal"] = arousal_pred
        # pred["expr_classification"] = expr_classification_pred

        valence_sample_weight = batch["valence_sample_weight"] if "valence_sample_weight" in batch.keys() else None
        arousal_sample_weight = batch["arousal_sample_weight"] if "arousal_sample_weight" in batch.keys() else None
        va_sample_weight = batch["va_sample_weight"] if "va_sample_weight" in batch.keys() else None
        expression_sample_weight = batch[
            "expression_sample_weight"] if "expression_sample_weight" in batch.keys() else None

        losses, metrics = self.compute_loss(pred, gt, class_weight, training=False,
                                            valence_sample_weight=valence_sample_weight,
                                            arousal_sample_weight=arousal_sample_weight,
                                            va_sample_weight=va_sample_weight,
                                            expression_sample_weight=expression_sample_weight,
                                            )

        self._log_losses_and_metrics(losses, metrics, "val")
        # visdict = self._test_visualization(values, batch, batch_idx, dataloader_idx=dataloader_idx)
        total_loss = losses["total"]
        return total_loss

    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        values = self.forward(batch)
        # valence_pred = values["valence"]
        # arousal_pred = values["arousal"]
        # expr_classification_pred = values["expr_classification"]
        if "expression_weight" in batch.keys():
            class_weight = batch["expression_weight"][0]
        else:
            class_weight = None

        if "va" in batch.keys():
            valence_gt = batch["va"][:, 0:1]
            arousal_gt = batch["va"][:, 1:2]
            expr_classification_gt = batch["affectnetexp"]
            gt = {}
            gt["valence"] = valence_gt
            gt["arousal"] = arousal_gt
            gt["expr_classification"] = expr_classification_gt

            pred = values
            losses, metrics = self.compute_loss(pred, gt, class_weight,
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



def v_or_a_loss(loss, pred, gt, term_weights,
                metrics, losses, measure, pred_prefix="", permit_dropping_corr=False, sample_weights=None):

    if measure not in ["valence", "arousal"]:
        raise ValueError(f"Invalid measure {measure}")

    measure_label = pred_prefix + measure

    if pred[pred_prefix + measure] is not None:
        metrics[pred_prefix + f"{measure[0]}_mae"] = F.l1_loss(pred[measure_label], gt[measure])
        metrics[pred_prefix + f"{measure[0]}_mse"] = F.mse_loss(pred[measure_label], gt[measure])
        metrics[pred_prefix + f"{measure[0]}_rmse"] = torch.sqrt(metrics[pred_prefix + f"{measure[0]}_mse"])
        
        if sample_weights is not None:
            metrics[pred_prefix + f"{measure[0]}_mse_weighted"] = (sample_weights * F.mse_loss(pred[measure_label], gt[measure], reduction='none')).mean()
            metrics[pred_prefix + f"{measure[0]}_rmse_weighted"] = torch.sqrt(metrics[pred_prefix + f"{measure[0]}_mse_weighted"])


        if gt[measure].numel() >= 2:
            metrics[pred_prefix + f"{measure[0]}_pcc"] = PCC_torch(pred[measure_label], gt[measure], batch_first=False)[0]
            metrics[pred_prefix + f"{measure[0]}_ccc"] = CCC_torch(pred[measure_label], gt[measure], batch_first=False)[0]

            if sample_weights is not None:
                metrics[pred_prefix + f"{measure[0]}_pcc_weighted"] = PCC_torch(pred[measure_label], gt[measure],
                                                                       batch_first=False, weights=sample_weights)
                metrics[pred_prefix + f"{measure[0]}_ccc_weighted"] = CCC_torch(pred[measure_label], gt[measure],
                                                                       batch_first=False, weights=sample_weights)
        elif permit_dropping_corr:
            pass
        else:
            raise RuntimeError("Cannot compute correlation for a single sample")
        metrics[pred_prefix + f"{measure[0]}_sagr"] = SAGR_torch(pred[measure_label], gt[measure])
        # metrics["v_icc"] = ICC_torch(pred[measure_label], gt[measure])
        if loss is not None:
            if callable(loss):
                losses[pred_prefix + measure[0]] = loss(pred[measure_label], gt[measure])
            elif isinstance(loss, dict):
                # print(metrics.keys() )
                for name, weight in loss.items():
                    # losses[name] = metrics[name]*weight
                    if permit_dropping_corr and pred_prefix + name not in metrics.keys():
                        continue
                    losses[pred_prefix + name] = metrics[pred_prefix + name] * term_weights[name]
            else:
                raise RuntimeError(f"Uknown {measure} loss '{loss}'")
    return losses, metrics


def va_loss(loss, pred, gt, weights, metrics, losses, pred_prefix="", permit_dropping_corr=False):
    if pred[pred_prefix + "valence"] is not None and pred[pred_prefix + "arousal"] is not None:
        va_pred = torch.cat([pred[pred_prefix + "valence"], pred[pred_prefix + "arousal"]], dim=1)
        va_gt = torch.cat([gt["valence"], gt["arousal"]], dim=1)
        metrics[pred_prefix + "va_mae"] = F.l1_loss(va_pred, va_gt)
        metrics[pred_prefix + "va_mse"] = F.mse_loss(va_pred, va_gt)
        metrics[pred_prefix + "va_rmse"] = torch.sqrt(metrics[pred_prefix + "va_mse"])
        if pred_prefix + "a_pcc" in metrics.keys():
            metrics[pred_prefix + "va_lpcc"] = \
                (1. - 0.5 * (metrics[pred_prefix + "a_pcc"] + metrics[pred_prefix + "v_pcc"]))[0]#[0]
            metrics[pred_prefix + "va_lccc"] = \
                (1. - 0.5 * (metrics[pred_prefix + "a_ccc"] + metrics[pred_prefix + "v_ccc"]))[0]#[0]
        elif permit_dropping_corr:
            pass
        else:
            raise RuntimeError(f"Missing computed correlation for the combined correlation loss: "
                               f"'{pred_prefix + 'a_pcc'}'")

        if pred_prefix + "a_pcc_weighted" in metrics.keys():
            metrics[pred_prefix + "va_lpcc_weighted"] = \
                (1. - 0.5 * (metrics[pred_prefix + "a_pcc_weighted"] + metrics[pred_prefix + "v_pcc_weighted"]))[0]#[0]
            metrics[pred_prefix + "va_lccc_weighted"] = \
                (1. - 0.5 * (metrics[pred_prefix + "a_ccc_weighted"] + metrics[pred_prefix + "v_ccc_weighted"]))[0]#[0]

        elif permit_dropping_corr:
            pass
        else:
            raise RuntimeError(f"Missing computed correlation for the combined weighted correlation loss: "
                               f"'{pred_prefix + 'a_pcc'}'")

        if loss is not None:
            if callable(loss):
                losses[pred_prefix + "va"] = loss(va_pred, va_gt)
            elif isinstance(loss, dict):
                for name, weight in loss.items():
                    # losses[name] = metrics[name]*weight
                    if permit_dropping_corr and pred_prefix + name not in metrics.keys():
                        continue
                    losses[pred_prefix + name] = metrics[pred_prefix + name] * weights[name]
            else:
                raise RuntimeError(f"Uknown expression loss '{loss}'")
    return losses, metrics


def exp_loss(loss, pred, gt, class_weight, metrics, losses, expression_balancing, num_classes, pred_prefix=""):
    if pred[pred_prefix + "expr_classification"] is not None:
        if class_weight.shape[0] != num_classes:
            weight = None
        elif expression_balancing:
            weight = class_weight
        else:
            weight = torch.ones_like(class_weight)

        # metrics["expr_cross_entropy"] = F.cross_entropy(pred["expr_classification"], gt["expr_classification"][:, 0], torch.ones_like(class_weight))
        # metrics["expr_weighted_cross_entropy"] = F.cross_entropy(pred["expr_classification"], gt["expr_classification"][:, 0], class_weight)
        metrics[pred_prefix + "expr_nll"] = F.nll_loss(pred[pred_prefix + "expr_classification"],
                                                       gt["expr_classification"][:, 0],
                                                       None)
                                                       # torch.ones_like(class_weight))
        if weight is not None:
            metrics[pred_prefix + "expr_weighted_nll"] = F.nll_loss(pred[pred_prefix + "expr_classification"],
                                                                    gt["expr_classification"][:, 0],
                                                                    class_weight)
            metrics[pred_prefix + "expr_acc"] = ACC_torch(
                torch.argmax(pred[pred_prefix + "expr_classification"], dim=1),
                                                          gt["expr_classification"][:, 0])
        else:
            metrics[pred_prefix + "expr_weighted_nll"] = F.nll_loss(pred[pred_prefix + "expr_classification"],
                                                       gt["expr_classification"][:, 0],
                                                       None)

        if loss is not None:
            if callable(loss):
                losses[pred_prefix + "expr"] = loss(pred[pred_prefix + "expr_classification"],
                                                             gt["expr_classification"][:, 0], weight)
            elif isinstance(loss, dict):
                for name, weight in loss.items():
                    losses[pred_prefix + name] = metrics[pred_prefix + name] * weight
            else:
                raise RuntimeError(f"Uknown expression loss '{loss}'")

    return losses, metrics


def _get_step_loss_weights(v_loss, a_loss, va_loss, scheme, training):
    va_loss_weights = {}
    for key in v_loss:
        va_loss_weights[key] = v_loss[key]

    for key in a_loss:
        va_loss_weights[key] = a_loss[key]

    for key in va_loss:
        va_loss_weights[key] = va_loss[key]

    # if training:
    #     return va_loss_weights
    # n_terms = len(va_loss_weights)

    if scheme is not None:
        if training and scheme == 'shake':
            for key in va_loss_weights:
                va_loss_weights[key] = np.random.rand(1)[0]
            total_w = 0.
            for key in va_loss_weights:
                total_w += va_loss_weights[key]
            for key in va_loss_weights:
                va_loss_weights[key] /= total_w
        elif scheme == 'norm':
            total_w = 0.
            for key in va_loss_weights:
                total_w += va_loss_weights[key]

            for key in va_loss_weights:
                va_loss_weights[key] /= total_w
    return va_loss_weights
