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


import os, sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from gdl.datasets.AffectNetDataModule import AffectNetDataModule, AffectNetExpressions
from gdl_apps.EMOCA.train_expdeca import prepare_data, create_logger
from gdl_apps.EMOCA.train_deca_modular import get_checkpoint
from gdl.models.IO import locate_checkpoint, get_checkpoint_with_kwargs

from gdl.models.EmoDECA import EmoDECA
try:
    from gdl.models.EmoNetModule import EmoNetModule
except ImportError as e: 
    print("Skipping EmoNetModule because EmoNet it is not installed.  Make sure you pull the repository with submodules to enable EmoNet.")
from gdl.utils.other import class_from_str
import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from gdl_apps.EMOCA.interactive_deca_decoder import hack_paths
import torch
import wandb
from tqdm import auto

project_name = 'EmotionRecognition'


def validation_set_pass(cfg,
                        # stage, prefix,
                        dm=None, logger=None,
                        data_preparation_function=None,
                        checkpoint=None, checkpoint_kwargs=None):
    if dm is None:
        dm, sequence_name = data_preparation_function(cfg)

    if logger is None:
        N = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))

        if hasattr(cfg.inout, 'time') and hasattr(cfg.inout, 'random_id'):
            version = cfg.inout.time + "_" + cfg.inout.random_id
        elif hasattr(cfg.inout, 'time'):
            version = cfg.inout.time # + "_" + cfg.inout.name
        else:
            version = sequence_name[:N] # unfortunately time doesn't cut it if two jobs happen to start at the same time

        logger = create_logger(
                    cfg.learning.logger_type,
                    name=cfg.inout.name,
                    project_name=project_name,
                    version=version,
                    save_dir=cfg.inout.full_run_dir)

    # if deca is None:
    if 'emodeca_type' in cfg.model:
        deca_class = class_from_str(cfg.model.emodeca_type, sys.modules[__name__])
    else:
        deca_class = EmoDECA

    if logger is not None:
        logger.finalize("")
    # if checkpoint is None:
    #     deca = deca_class(cfg)
    # else:
    checkpoint_kwargs = checkpoint_kwargs or {}
    deca = deca_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)

    deca_class = type(deca)


    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp2'
    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp'
    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp_spawn' # ddp only seems to work for single .fit/test calls unfortunately,
    # accelerator = None if cfg.learning.num_gpus == 1 else 'dp'  # ddp only seems to work for single .fit/test calls unfortunately,

    # if accelerator is not None and 'LOCAL_RANK' not in os.environ.keys():
    #     print("SETTING LOCAL_RANK to 0 MANUALLY!!!!")
    #     os.environ['LOCAL_RANK'] = '0'

    # loss_to_monitor = 'val_loss_total'
    # dm.setup()
    # val_data = dm.val_dataloader()
    # if isinstance(val_data, list):
    #     loss_to_monitor = loss_to_monitor + "/dataloader_idx_0"
        # loss_to_monitor = '0_' + loss_to_monitor + "/dataloader_idx_0"
    # if len(prefix) > 0:
    #     loss_to_monitor = prefix + "_" + loss_to_monitor

    # callbacks = []
    # checkpoint_callback = ModelCheckpoint(
    #     monitor=loss_to_monitor,
    #     filename='deca-{epoch:02d}-{' + loss_to_monitor + ':.8f}',
    #     save_top_k=3,
    #     save_last=True,
    #     mode='min',
    #     dirpath=cfg.inout.checkpoint_dir
    # )
    # callbacks += [checkpoint_callback]
    # if hasattr(cfg.learning, 'early_stopping') and cfg.learning.early_stopping:
    #     patience = 3
    #     if hasattr(cfg.learning.early_stopping, 'patience') and cfg.learning.early_stopping.patience:
    #         patience = cfg.learning.early_stopping.patience
    #
    #     early_stopping_callback = EarlyStopping(monitor=loss_to_monitor,
    #                                             mode='min',
    #                                             patience=patience,
    #                                             strict=True)
    #     callbacks += [early_stopping_callback]

    #
    # val_check_interval = 1.0
    # if 'val_check_interval' in cfg.learning.keys():
    #     val_check_interval = cfg.learning.val_check_interval
    # print(f"Setting val_check_interval to {val_check_interval}")
    #
    # max_steps = None
    # if hasattr(cfg.learning, 'max_steps'):
    #     max_steps = cfg.learning.max_steps
    #     print(f"Setting max steps to {max_steps}")
    #
    # print(f"After training checkpoint strategy: {cfg.learning.checkpoint_after_training}")

    # trainer = Trainer(gpus=cfg.learning.num_gpus,
    #                   max_epochs=cfg.learning.max_epochs,
    #                   max_steps=max_steps,
    #                   default_root_dir=cfg.inout.checkpoint_dir,
    #                   logger=logger,
    #                   accelerator=accelerator,
    #                   callbacks=callbacks,
    #                   val_check_interval=val_check_interval,
    #                   # num_sanity_val_steps=0
    #                   )
    dm.prepare_data()
    dm.setup()
    dl = dm.val_dataloader()


    deca.cuda()
    deca.eval()

    prefixes = ['', 'emonet_coarse_', 'emonet_detail_']


    gt_list = []
    pred_dict = {}
    for p in prefixes:
        pred_dict[p] = []


    for bi, batch in  enumerate(auto.tqdm(dl)):
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        with torch.no_grad():
            values = deca.forward(batch)
        # valence_pred = values["valence"]
        # arousal_pred = values["arousal"]
        # valence_gt = batch["va"][:, 0:1]
        # arousal_gt = batch["va"][:, 1:2]


        for p in prefixes:
            if p + "expr_classification" in values.keys():
                expr_classification_pred = values[p + "expr_classification"]
                pred_dict[p] += [expr_classification_pred]

        expr_classification_gt = batch["affectnetexp"]
        # if "expression_weight" in batch.keys():
        #     class_weight = batch["expression_weight"][0]
        # else:
        #     class_weight = None

        # gt = {}
        # gt["valence"] = valence_gt
        # gt["arousal"] = arousal_gt
        # gt["expr_classification"] = expr_classification_gt
        #
        # pred = values
        # pred = {}
        # pred["valence"] = valence_pred
        # pred["arousal"] = arousal_pred
        # pred["expr_classification"] = expr_classification_pred


        gt_list += [expr_classification_gt ]
        # pred_list += [  expr_classification_pred]


    gt_labels = torch.cat(gt_list)

    names = [AffectNetExpressions(i).name for i in range(9)]

    for p in prefixes:
        if len(pred_dict[p]) > 0:
            probs = torch.cat(pred_dict[p])
            conf_mat = wandb.plot.confusion_matrix(y_true=gt_labels.detach().cpu().numpy()[:,0], probs=probs.detach().cpu().numpy(), class_names=names, title="Expression confusion matrix")
            wandb.log({p + "val_conf_mat" : conf_mat})

    print("DONE!")



def load_configs(run_path):
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    return conf



def main():
    if len(sys.argv) < 2:
        root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/emodeca/")
        relative_to_path = '/ps/scratch/'
        replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
        # resume_from = '/ps/scratch/rdanecek/emoca/finetune_deca/2021_04_02_18-46-51_va_DeSegFalse_DeNone_Aug_DwC_early'
        # run_path = "2021_05_12_22-53-50_EmoNet_shake_early"
        # run_path = '2021_05_12_14-54-24_EmoDECA_Affec_ExpDECA_EmoNetC_unpose_light_cam_shake_early'
        # run_path = '2021_05_12_14-51-36_EmoDECA_Affec_ExpDECA_EmoNetCD_unpose_light_cam_shake_early'
        # run_path = '2021_05_12_14-22-36_EmoDECA_Affec_ExpDECA_EmoNetD_unpose_light_cam_shake_early'
        # run_path = '2021_05_11_22-57-26_EmoDECA_Affec_ExpDECA_EmoNetCD_unpose_light_cam_shake_early'
        run_path = '2021_05_12_14-22-40_EmoDECA_Affec_ExpDECA_EmoNetD_shake_early'

        path = root / run_path
    else:
        root = '/ps/scratch/rdanecek/emoca/emodeca/'
        relative_to_path = None
        replace_root_path = None
        path = sys.argv[1]

    stages_prefixes = ""


    cfg = load_configs(path)

    checkpoint_mode = 'best' # resuming in the same stage, we want to pick up where we left of
    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, stages_prefixes, checkpoint_mode=checkpoint_mode,
                                                               relative_to=relative_to_path,
                                                               replace_root=replace_root_path)

    validation_set_pass(cfg,
                        # stage, stages_prefixe,
                        data_preparation_function=prepare_data,
                        checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs)


if __name__ == "__main__":
    main()
