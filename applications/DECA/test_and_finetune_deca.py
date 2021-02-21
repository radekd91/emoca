import os, sys
from pathlib import Path
sys.path += [str(Path(__file__).parent.parent)]

import numpy as np
from datasets.FaceVideoDataset import FaceVideoDataModule, \
    AffectNetExpressions, Expression7, AU8, expr7_to_affect_net
from datasets.EmotionalDataModule import EmotionDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models.DECA import DecaModule
from pytorch_lightning.loggers import WandbLogger
import wandb
import datetime
# import hydra
from omegaconf import DictConfig, OmegaConf
import copy

project_name = 'EmotionalDeca'


def prepare_data(cfg):
    print(f"The data will be loaded from: '{cfg.data.data_root}'")
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


def single_stage_deca_pass(deca, cfg, stage, prefix, dm=None, logger=None,
                           data_preparation_function=None,
                           # data_preparation_function=prepare_data,
                           ):
    if dm is None:
        dm, sequence_name = data_preparation_function(cfg)

    if logger is None:
        N = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        version = sequence_name[:N]
        logger = WandbLogger(name=cfg.inout.name,
                                   project=project_name,
                                   # config=dict(conf),
                                   version=version,
                                   save_dir=cfg.inout.full_run_dir)

    if deca is None:
        logger.finalize("")
        deca = DecaModule(cfg.model, cfg.learning, cfg.inout, prefix)

    else:
        if stage == 'train':
            mode = True
        else:
            mode = False
        deca.reconfigure(cfg.model, prefix, downgrade_ok=True, train=mode)

    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp2'
    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp'
    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp_spawn' # ddp only seems to work for single .fit/test calls unfortunately,
    accelerator = None if cfg.learning.num_gpus == 1 else 'dp'  # ddp only seems to work for single .fit/test calls unfortunately,

    if accelerator is not None and 'LOCAL_RANK' not in os.environ.keys():
        print("SETTING LOCAL_RANK to 0 MANUALLY!!!!")
        os.environ['LOCAL_RANK'] = '0'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='deca-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    trainer = Trainer(gpus=cfg.learning.num_gpus, max_epochs=cfg.learning.max_epochs,
                      default_root_dir=cfg.inout.checkpoint_dir,
                      logger=logger,
                      accelerator=accelerator,
                      callbacks=[checkpoint_callback])
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


def create_experiment_name(cfg_coarse, cfg_detail, sequence_name, version=0):
    if version == 0:
        experiment_name = sequence_name
        experiment_name = experiment_name.replace("/", "_")
        if cfg_coarse.model.use_emonet_loss and cfg_detail.use_emonet_loss:
            experiment_name += '_EmoNetLossB'
        elif cfg_coarse.model.use_emonet_loss:
            experiment_name += '_EmoNetLossC'
        elif cfg_detail.use_emonet_loss:
            experiment_name += '_EmoNetLossD'

        if cfg_coarse.model.use_gt_emotion_loss and cfg_detail.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossB'
        elif cfg_coarse.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossC'
        elif cfg_detail.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossD'

        if cfg_coarse.model.useSeg:
            experiment_name += '_CoSegmentGT'
        else:
            experiment_name += '_CoSegmentRend'

        if cfg_detail.model.useSeg:
            experiment_name += '_DeSegmentGT'
        else:
            experiment_name += '_DeSegmentRend'

    else:
        raise NotImplementedError("Unsupported naming versino")

    return experiment_name


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

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    experiment_name = time + "_" + create_experiment_name(cfg_coarse, cfg_detail, sequence_name)

    if cfg_coarse.inout.full_run_dir == 'todo':
        full_run_dir = Path(configs[0].inout.output_dir) / experiment_name
    else:
        full_run_dir = cfg_coarse.inout.full_run_dir

    full_run_dir.mkdir(parents=True)
    print(f"The run will be saved  to: '{str(full_run_dir)}'")
    with open("out_folder.txt", "w") as f:
        f.write(str(full_run_dir))

    coarse_checkpoint_dir = full_run_dir / "coarse"
    coarse_checkpoint_dir.mkdir(parents=True)

    cfg_coarse.inout.full_run_dir = str(full_run_dir / "coarse")
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_coarse.inout.name = experiment_name

    # if cfg_detail.inout.full_run_dir == 'todo':
    detail_checkpoint_dir = full_run_dir / "detail"
    detail_checkpoint_dir.mkdir(parents=True)

    cfg_detail.inout.full_run_dir = str(full_run_dir / "detail")
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)
    cfg_detail.inout.name = experiment_name

    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

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


def configure_and_finetune(coarse_cfg_default, coarse_overrides, detail_cfg_default, detail_overrides):
    cfg_coarse, cfg_detail = configure(coarse_cfg_default, coarse_overrides, detail_cfg_default, detail_overrides)
    finetune_deca(cfg_coarse, cfg_detail)


def configure(coarse_cfg_default, coarse_overrides, detail_cfg_default, detail_overrides):
    from hydra.experimental import compose, initialize
    initialize(config_path="deca_conf", job_name="finetune_deca")
    cfg_coarse = compose(config_name=coarse_cfg_default, overrides=coarse_overrides)
    cfg_detail = compose(config_name=detail_cfg_default, overrides=detail_overrides)
    return cfg_coarse, cfg_detail


# @hydra.main(config_path="deca_conf", config_name="deca_finetune")
# def main(cfg : DictConfig):
def main():
    configured = False
    if len(sys.argv) >= 3:
        if Path(sys.argv[1]).is_file():
            configured = True
            with open(sys.argv[1], 'r') as f:
                coarse_conf = OmegaConf.load(f)
            with open(sys.argv[2], 'r') as f:
                detail_conf = OmegaConf.load(f)
        else:
            coarse_conf = sys.argv[1]
            detail_conf = sys.argv[2]
    else:
        coarse_conf = "deca_finetune_coarse_emonet"
        detail_conf = "deca_finetune_detail_emonet"

    if len(sys.argv) >= 5:
        coarse_override = sys.argv[3]
        detail_override = sys.argv[4]
    else:
        coarse_override = []
        detail_override = []
    if configured:
        finetune_deca(coarse_conf, detail_conf)
    else:
        configure_and_finetune(coarse_conf, coarse_override, detail_conf, detail_override)

if __name__ == "__main__":
    main()
