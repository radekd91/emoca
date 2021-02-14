import os, sys
from pathlib import Path
sys.path += [str(Path(__file__).parent.parent)]

import numpy as np
from datasets.FaceVideoDataset import FaceVideoDataModule, \
    AffectNetExpressions, Expression7, AU8, expr7_to_affect_net
from datasets.EmotionalDataModule import EmotionDataModule
from pytorch_lightning import Trainer

from models.DECA import DecaModule
from pytorch_lightning.loggers import WandbLogger
import wandb
import datetime
# import hydra
from omegaconf import DictConfig, OmegaConf


def finetune_deca(cfg_coarse, cfg_detail):

    fvdm = FaceVideoDataModule(Path(cfg_coarse.data.data_root), Path(cfg_coarse.data.data_root) / "processed",
                               cfg_coarse.data.processed_subfolder)
    dm = EmotionDataModule(fvdm, image_size=cfg_coarse.model.image_size,
                           with_landmarks=True, with_segmentations=True)
    dm.prepare_data()

    # index = 220
    # index = 120
    index = cfg_coarse.data.sequence_index
    annotation_list = cfg_coarse.data.annotation_list

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

    deca = DecaModule(cfg_coarse.model, cfg_coarse.learning, cfg_coarse.inout)
    conf = DictConfig({})
    conf.coarse = cfg_coarse
    conf.detail = cfg_detail

    project_name = 'EmotionalDeca'
    time = datetime.datetime.now().strftime("%b_%d_%Y_%H-%M-%S")
    experiment_name = time+ "_" + sequence_name

    full_run_dir = Path(cfg_coarse.inout.output_dir) / experiment_name
    full_run_dir.mkdir(parents=True)

    coarse_checkpoint_dir = full_run_dir / "coarse"
    coarse_checkpoint_dir.mkdir(parents=True)

    detail_checkpoint_dir = full_run_dir / "detail"
    detail_checkpoint_dir.mkdir(parents=True)

    cfg_coarse.inout.full_run_dir = str(full_run_dir / "coarse")
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_detail.inout.full_run_dir = str(full_run_dir / "detail")
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)

    # name = cfg_coarse.inout.name + '_' + str(filter_pattern) + "_" + \
    #        datetime.datetime.now().strftime("%b_%d_%Y_%H-%M-%S")

    #
    # train_data_loader = dm.train_dataloader(annotation_list, filter_pattern,
    #                                         batch_size=cfg_coarse.model.batch_size_train,
    #                                         num_workers=cfg_coarse.data.num_workers,
    #                                         split_ratio=cfg_coarse.data.split_ratio,
    #                                         split_style=cfg_coarse.data.split_style,
    #                                         K=cfg_coarse.model.train_K,
    #                                         K_policy=cfg_coarse.model.K_policy)
    #
    # val_data_loader = dm.val_dataloader(annotation_list, filter_pattern,
    #                                     batch_size=cfg_coarse.model.batch_size_val,
    #                                     num_workers=cfg_coarse.data.num_workers)

    test_data_loader = dm.test_dataloader(annotation_list, filter_pattern,
                                         batch_size=cfg_coarse.model.batch_size_val,
                                         num_workers=cfg_coarse.data.num_workers,
                                         K=cfg_coarse.model.test_K,
                                         K_policy=cfg_coarse.model.K_policy)

    wandb_logger = WandbLogger(name=experiment_name,
                               project=project_name,
                               config=dict(conf),
                               version=time,
                               save_dir=full_run_dir)

    configs = [cfg_coarse, cfg_detail]
    # configs = [cfg_detail]
    for i, cfg in enumerate(configs):
        if i > 0:
            deca.reconfigure(cfg.model)

        accelerator = None if cfg.learning.num_gpus == 1 else 'ddp'
        trainer = Trainer(gpus=cfg.learning.num_gpus, max_epochs=cfg.learning.max_epochs,
                          default_root_dir=cfg.inout.checkpoint_dir,
                          logger=wandb_logger,
                          accelerator=accelerator)

        # trainer.fit(deca, train_dataloader=train_data_loader, val_dataloaders=[val_data_loader, ])

        wandb_logger.finalize("")
        trainer.test(deca,
                     test_dataloaders=[test_data_loader],
                     ckpt_path=None)
        # to make sure WANDB has the correct step
        wandb_logger.finalize("")


# @hydra.main(config_path="deca_conf", config_name="deca_finetune")
# def main(cfg : DictConfig):
def main():
    from hydra.experimental import compose, initialize
    # override = ['learning.num_gpus=2', 'model/paths=cluster']
    override = sys.argv[1:]
    initialize(config_path="deca_conf", job_name="finetune_deca")
    cfg_coarse = compose(config_name="deca_finetune_coarse", overrides=override)
    cfg_detail = compose(config_name="deca_finetune_detail", overrides=override)
    finetune_deca(cfg_coarse, cfg_detail)


if __name__ == "__main__":
    main()
