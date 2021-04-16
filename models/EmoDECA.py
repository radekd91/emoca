from .DECA import DecaModule, instantiate_deca
from .MLP import MLP
import torch
import pytorch_lightning as pl
import numpy as np
from utils.other import class_from_str
import torch.nn.functional as F



class EmoDECA(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        # model_params = config.model.deca_cfg.model
        # learning_params = config.model.deca_cfg.learning
        # inout_params = config.model.deca_cfg.inout
        deca_checkpoint = config.model.deca_checkpoint
        self.deca = instantiate_deca(config.mode.deca_cfg, "test", deca_checkpoint )

        # if deca_checkpoint is not None:
        #     self.deca = DecaModule.load_from_checkpoint(checkpoint_path=deca_checkpoint,
        #                                                 model_params=model_params,
        #                                                 learning_params=learning_params,
        #                                                 inout_params=inout_params
        #                                                 )
        # else:
        #     self.deca = DecaModule(model_params=model_params,
        #                            learning_params=learning_params,
        #                            inout_params=inout_params
        #                            )
        self.mlp = MLP(config)

        self.va_loss = class_from_str(self.config.model.va_loss, F)
        self.exp_loss = class_from_str(self.config.model.exp_loss, F)


    def forward(self, batch):
        values = self.deca.encode(batch)
        # shapecode = values['shapecode']
        # texcode = values['texcode']
        expcode = values['expcode']
        posecode = values['posecode']
        detailcode = values['detailcode']

        global_pose = posecode[:, :3]
        jaw_pose = posecode[:, 3:]

        input_list = []

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
            num_classes = 6
            expr_classification = output[:, out_idx:(out_idx+num_classes)]
            out_idx += num_classes
        else:
            expr_classification = None

        if self.config.model.predict_valence:
            valence = output[:, out_idx:(out_idx+1)]
            out_idx += 1
        else:
            valence = None

        if self.config.model.predict_arousal:
            arousal = output[:, out_idx:(out_idx+1)]
            out_idx += 1
        else:
            arousal = None

        return valence, arousal, expr_classification


    def compute_loss(self,
                     valence_pred, arousal_pred, expr_classification_pred,
                     valence_gt, arousal_gt, expr_classification_gt,
                     class_weight):
        losses = {}
        metrics = {}
        if valence_pred is not None and arousal_pred is not None:
            va_pred = torch.cat([valence_pred, arousal_pred], dim=1)
            va_gt = torch.cat([valence_gt, arousal_gt], dim=1)
            losses["va"] = self.va_loss(va_pred, va_gt)
            metrics["va_mae"] = F.l1_loss(va_pred, va_gt)
            metrics["va_mse"] = F.mse_loss(va_pred, va_gt)
            metrics["va_rmse"] = torch.sqrt(F.mse_loss(va_pred, va_gt))
        elif valence_pred is not None:
            losses["v"] = self.va_loss(valence_pred, valence_gt)
            metrics["v_mae"] = F.l1_loss(valence_pred, valence_gt)
            metrics["v_mse"] = F.mse_loss(valence_pred, valence_gt)
            metrics["v_rmse"] = torch.sqrt(F.mse_loss(valence_pred, valence_gt))
        elif arousal_pred is not None:
            losses["a"] = self.va_loss(arousal_pred, arousal_gt)
            metrics["a_mae"] = F.l1_loss(arousal_pred, arousal_gt)
            metrics["a_mse"] = F.mse_loss(arousal_pred, arousal_gt)
            metrics["a_rmse"] = torch.sqrt(F.mse_loss(arousal_pred, arousal_gt))

        if expr_classification_pred is not None:
            if self.config.model.expression_balancing:
                weight = class_weight
            else:
                weight = torch.ones_like(class_weight)

            losses["expr"] = self.expr_loss(arousal_pred, arousal_gt, weight)
            metrics["expr_cross_entropy"] = F.cross_entropy(expr_classification_pred, expr_classification_gt, torch.ones_like(class_weight))
            metrics["expr_weighted_cross_entropy"] = F.cross_entropy(expr_classification_pred, expr_classification_gt, class_weight)

        loss = 0.
        for key, value in losses.items():
            loss += value

        losses["total"] = loss

        return losses, metrics


    def training_step(self, batch, batch_idx):
        valence_pred, arousal_pred, expr_classification_pred = self.forward(batch)
        valence_gt = batch["valence"]
        arousal_gt = batch["arousal"]
        expr_classification_gt = batch["expression"]
        class_weight = batch["expression_weight"]

        losses, metrics = self.compute_loss(valence_pred, arousal_pred, expr_classification_pred,
                          valence_gt, arousal_gt, expr_classification_gt, class_weight)

        self._log_losses_and_metrics(losses, metrics, "train")
        total_loss = losses["total"]
        return total_loss


    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        valence_pred, arousal_pred, expr_classification_pred = self.forward(batch)
        valence_gt = batch["valence"]
        arousal_gt = batch["arousal"]
        expr_classification_gt = batch["expression"]
        class_weight = batch["expression_weight"]

        losses, metrics = self.compute_loss(valence_pred, arousal_pred, expr_classification_pred,
                                            valence_gt, arousal_gt, expr_classification_gt, class_weight)

        self._log_losses_and_metrics(losses, metrics, "val")
        total_loss = losses["total"]
        return total_loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        pass

    def _log_losses_and_metrics(self, losses, metrics, stage):
        if stage in ["train", "val"]:
            on_epoch = True
            on_step = False
        else:
            on_epoch = True
            on_step = True

        self.log_dict({f"{stage}_loss_" + key: value for key,value in losses.items()}, on_epoch=on_epoch, on_step=on_step)
        self.log_dict({f"{stage}_metric_" + key: value for key,value in metrics.items()}, on_epoch=on_epoch, on_step=on_step)

