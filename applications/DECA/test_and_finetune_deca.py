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
import copy


# def get_sequence_name(cfg):
#
#     # print(f"Looking for video {index} in {len(fvdm.video_list)}")
#     return fvdm, sequence_name, filter_pattern

def prepare_data(cfg):
    fvdm = FaceVideoDataModule(Path(cfg.data.data_root), Path(cfg.data.data_root) / "processed",
                               cfg.data.processed_subfolder)
    fvdm.prepare_data()
    fvdm.setup()

    index = cfg.data.sequence_index
    annotation_list = copy.deepcopy(
        cfg.data.annotation_list)  # sth weird is modifying the list, that's why deep copy
    # annotation_list = cfg_coarse.data.annotation_list.copy()

    if index == -1:
        sequence_name = annotation_list[0]
        if annotation_list[0] == 'va':
            filter_pattern = 'VA_Set'
        elif annotation_list[0] == 'expr7':
            filter_pattern = 'Expression_Set'
        else:
            raise NotImplementedError()
    else:
        sequence_name = str(fvdm.video_list[index])
        filter_pattern = sequence_name
        if annotation_list[0] == 'va' and 'VA_Set' not in sequence_name:
            print("No GT for valence and arousal. Skipping")
            # sys.exit(0)
        if annotation_list[0] == 'expr7' and 'Expression_Set' not in sequence_name:
            print("No GT for expressions. Skipping")
            # sys.exit(0)


    # index = 220
    # index = 120
    index = cfg.data.sequence_index
    annotation_list = copy.deepcopy(
        cfg.data.annotation_list)  # sth weird is modifying the list, that's why deep copy
    # annotation_list = cfg_coarse.data.annotation_list.copy()

    print(f"Looking for video {index} in {len(fvdm.video_list)}")

    dm = EmotionDataModule(fvdm,
                           image_size=cfg.model.image_size,
                           with_landmarks=True,
                           with_segmentations=True,
                           split_ratio=cfg.data.split_ratio,
                           split_style=cfg.data.split_style,
                           train_K=cfg.learning.train_K,
                           train_K_policy=cfg.learning.train_K_policy,
                           val_K=cfg.learning.val_K,
                           val_K_policy=cfg.learning.val_K_policy,
                           test_K=cfg.learning.train_K,
                           test_K_policy=cfg.learning.test_K_policy,
                           annotation_list=annotation_list,
                           filter_pattern=filter_pattern,
                           num_workers=cfg.data.num_workers,
                           train_batch_size=cfg.learning.batch_size_train,
                           val_batch_size=cfg.learning.batch_size_val,
                           test_batch_size=cfg.learning.batch_size_test)
    return dm, sequence_name


def single_stage_deca_pass(deca, cfg, stage, prefix, dm=None, logger=None):
    if dm is None:
        dm, sequence_name = prepare_data(cfg)

    if deca is None:
        logger.finalize("")
        deca = DecaModule(cfg.model, cfg.learning, cfg.inout, prefix)

    else:
        deca.reconfigure(cfg.model, prefix, downgrade_ok=True)

    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp2'
    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp'
    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp_spawn' # ddp only seems to work for single .fit/test calls unfortunately,
    accelerator = None if cfg.learning.num_gpus == 1 else 'dp'  # ddp only seems to work for single .fit/test calls unfortunately,

    if accelerator is not None and 'LOCAL_RANK' not in os.environ.keys():
        print("SETTING LOCAL_RANK to 0 MANUALLY!!!!")
        os.environ['LOCAL_RANK'] = '0'

    trainer = Trainer(gpus=cfg.learning.num_gpus, max_epochs=cfg.learning.max_epochs,
                      default_root_dir=cfg.inout.checkpoint_dir,
                      logger=logger,
                      accelerator=accelerator)
    if stage == "train":
        # trainer.fit(deca, train_dataloader=train_data_loader, val_dataloaders=[val_data_loader, ])
        trainer.fit(deca, datamodule=dm)
    elif stage == "test":
        # trainer.test(deca,
        #              test_dataloaders=[test_data_loader],
        #              ckpt_path=None)
        trainer.test(deca,
                     datamodule=dm,
                     ckpt_path=None)
    else:
        raise ValueError(f"Invalid stage {stage}")
    logger.finalize("")
    return deca


def finetune_deca(cfg_coarse, cfg_detail, test_first=True):
    conf = DictConfig({})
    conf.coarse = cfg_coarse
    conf.detail = cfg_detail
    # configs = [cfg_coarse, cfg_detail]
    configs = [cfg_coarse, cfg_detail, cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    stages = ["test", "test", "train", "test", "train", "test"]
    stages_prefixes = ["start", "start", "", "", "", ""]

    if not test_first:
        num_test_stages = 2
        configs = configs[num_test_stages:]
        stages = stages[num_test_stages:]
        stages_prefixes = stages_prefixes[num_test_stages:]

    dm, sequence_name = prepare_data(configs[0])

    project_name = 'EmotionalDeca'
    time = datetime.datetime.now().strftime("%b_%d_%Y_%H-%M-%S")
    experiment_name = time + "_" + sequence_name

    full_run_dir = Path(configs[0].inout.output_dir) / experiment_name
    full_run_dir.mkdir(parents=True)

    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    coarse_checkpoint_dir = full_run_dir / "coarse"
    coarse_checkpoint_dir.mkdir(parents=True)

    detail_checkpoint_dir = full_run_dir / "detail"
    detail_checkpoint_dir.mkdir(parents=True)

    cfg_coarse.inout.full_run_dir = str(full_run_dir / "coarse")
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_detail.inout.full_run_dir = str(full_run_dir / "detail")
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)

    wandb_logger = WandbLogger(name=experiment_name,
                         project=project_name,
                         config=dict(conf),
                         version=time,
                         save_dir=full_run_dir)

    deca = None
    for i, cfg in enumerate(configs):
        dm.reconfigure(
            train_batch_size=cfg.learning.batch_size_train,
            val_batch_size=cfg.learning.batch_size_val,
            test_batch_size=cfg.learning.batch_size_test,
            train_K=cfg.learning.train_K,
            val_K=cfg.learning.val_K,
            test_K=cfg.learning.test_K,
            train_K_policy=cfg.learning.train_K_policy,
            val_K_policy=cfg.learning.val_K_policy,
            test_K_policy=cfg.learning.test_K_policy,
        )
        deca = single_stage_deca_pass(deca, cfg, stages[i], stages_prefixes[i], dm, wandb_logger)


# @hydra.main(config_path="deca_conf", config_name="deca_finetune")
# def main(cfg : DictConfig):
def main():
    from hydra.experimental import compose, initialize
    # override = ['learning.num_gpus=2', 'model/paths=cluster']
    override = sys.argv[1:]
    initialize(config_path="deca_conf", job_name="finetune_deca")
    cfg_coarse = compose(config_name="deca_finetune_coarse_emonet", overrides=override)
    cfg_detail = compose(config_name="deca_finetune_detail_emonet", overrides=override)
    # cfg_coarse = compose(config_name="deca_finetune_coarse", overrides=override)
    # cfg_detail = compose(config_name="deca_finetune_detail", overrides=override)
    finetune_deca(cfg_coarse, cfg_detail)


if __name__ == "__main__":
    main()
