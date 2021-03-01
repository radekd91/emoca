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
    # index = -1 # TODO: delete
    # cfg.data.split_style = 'manual' # TODO: delete
    if index == -1:
        sequence_name = annotation_list[0]
        filters = []
        if 'va' in annotation_list:
            filters += ['VA_Set']
        if 'expr7' in annotation_list:
            filters += ['Expression_Set']
        if 'au' in annotation_list:
            filters += ['AU_Set']
        filter_pattern = f"({'|'.join(filters)})"
        if len(filters) == 0:
            raise NotImplementedError()

        # if 'va' in annotation_list and 'expr7' in annotation_list:
        #     filter_pattern = "(VA_Set|Expression_Set)"
        # elif annotation_list[0] == 'va':
        #     filter_pattern = 'VA_Set'
        # elif annotation_list[0] == 'expr7':
        #     filter_pattern = 'Expression_Set'
        # else:
        #     raise NotImplementedError()
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
                           checkpoint=None, checkpoint_kwargs=None):
    if dm is None:
        dm, sequence_name = data_preparation_function(cfg)

    if logger is None:
        N = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        version = sequence_name[:N] # unfortunately time doesn't cut it if two jobs happen to start at the same time
        # version = project_name #TODO
        logger = WandbLogger(name=cfg.inout.name,
                                   project=project_name,
                                   # config=dict(conf),
                                   version=version,
                                   save_dir=cfg.inout.full_run_dir)

    if deca is None:
        logger.finalize("")
        if checkpoint is None:
            deca = DecaModule(cfg.model, cfg.learning, cfg.inout, prefix)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, **checkpoint_kwargs)
            if stage == 'train':
                mode = True
            else:
                mode = False
            deca.reconfigure(cfg.model, cfg.inout, prefix, downgrade_ok=True, train=mode)
    else:
        if stage == 'train':
            mode = True
        else:
            mode = False
        # if checkpoint is not None:
        #     deca.load_from_checkpoint(checkpoint_path=checkpoint)
        deca.reconfigure(cfg.model, cfg.inout, prefix, downgrade_ok=True, train=mode)



    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp2'
    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp'
    # accelerator = None if cfg.learning.num_gpus == 1 else 'ddp_spawn' # ddp only seems to work for single .fit/test calls unfortunately,
    accelerator = None if cfg.learning.num_gpus == 1 else 'dp'  # ddp only seems to work for single .fit/test calls unfortunately,

    if accelerator is not None and 'LOCAL_RANK' not in os.environ.keys():
        print("SETTING LOCAL_RANK to 0 MANUALLY!!!!")
        os.environ['LOCAL_RANK'] = '0'

    loss_to_monitor = 'val_loss'
    dm.setup()
    val_data = dm.val_dataloader()
    if isinstance(val_data, list):
        loss_to_monitor = loss_to_monitor + "/dataloader_idx_0"
        # loss_to_monitor = '0_' + loss_to_monitor + "/dataloader_idx_0"
    # if len(prefix) > 0:
    #     loss_to_monitor = prefix + "_" + loss_to_monitor

    checkpoint_callback = ModelCheckpoint(
        monitor=loss_to_monitor,
        # filename='deca-{epoch:02d}-{val_loss:.2f}',
        filename='deca-{epoch:02d}-{' + loss_to_monitor + ':.2f}',
        save_top_k=3,
        mode='min',
        dirpath=cfg.inout.checkpoint_dir
    )

    val_check_interval = 1.0
    if 'val_check_interval' in cfg.model.keys():
        val_check_interval = cfg.model.val_check_interval

    trainer = Trainer(gpus=cfg.learning.num_gpus, max_epochs=cfg.model.max_epochs,
                      default_root_dir=cfg.inout.checkpoint_dir,
                      logger=logger,
                      accelerator=accelerator,
                      callbacks=[checkpoint_callback],
                      val_check_interval=val_check_interval,
                      # num_sanity_val_steps=0
                      )
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
        if cfg_coarse.model.use_emonet_loss and cfg_detail.model.use_emonet_loss:
            experiment_name += '_EmoNetLossB'
        elif cfg_coarse.model.use_emonet_loss:
            experiment_name += '_EmoNetLossC'
        elif cfg_detail.model.use_emonet_loss:
            experiment_name += '_EmoNetLossD'
        if cfg_coarse.model.use_emonet_loss or cfg_detail.model.use_emonet_loss:
            experiment_name += '_'
            if cfg_coarse.model.use_emonet_feat_1:
                experiment_name += 'F1'
            if cfg_coarse.model.use_emonet_feat_2:
                experiment_name += 'F2'
            if cfg_coarse.model.use_emonet_valence:
                experiment_name += 'V'
            if cfg_coarse.model.use_emonet_arousal:
                experiment_name += 'A'
            if cfg_coarse.model.use_emonet_expression:
                experiment_name += 'E'
            if cfg_coarse.model.use_emonet_combined:
                experiment_name += 'C'

        if cfg_coarse.model.use_emonet_loss or cfg_detail.model.use_emonet_loss:
            experiment_name += 'w-%.05f' % cfg_coarse.model.emonet_weight



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

        if not cfg_detail.model.use_detail_l1:
            experiment_name += '_NoDetL1'
        if not cfg_detail.model.use_detail_mrf:
            experiment_name += '_NoMRF'

        if not cfg_coarse.model.background_from_input and not cfg_detail.model.background_from_input:
            experiment_name += '_BackBlackB'
        elif not cfg_coarse.model.background_from_input:
            experiment_name += '_BackBlackC'
        elif not cfg_detail.model.background_from_input:
            experiment_name += '_BackBlackD'

        if cfg_coarse.learning.learning_rate != 0.0001:
            experiment_name += f'CoLR-{cfg_coarse.learning.learning_rate}'
        if cfg_detail.learning.learning_rate != 0.0001:
            experiment_name += f'DeLR-{cfg_detail.learning.learning_rate}'

        if cfg_coarse.model.config.use_photometric:
            experiment_name += 'CoPhoto'
        if cfg_coarse.model.config.use_landmarks:
            experiment_name += 'CoLMK'

    else:
        raise NotImplementedError("Unsupported naming versino")

    return experiment_name


def finetune_deca(cfg_coarse, cfg_detail, test_first=True, start_i=0):
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
    dm.setup()
    # sys.exit(0) ## TODO: DELETE
    if cfg_coarse.inout.full_run_dir == 'todo':
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        experiment_name = time + "_" + create_experiment_name(cfg_coarse, cfg_detail, sequence_name)
        full_run_dir = Path(configs[0].inout.output_dir) / experiment_name
        exist_ok = False # a path for a new experiment should not yet exist
    else:
        experiment_name = cfg_coarse.inout.name
        len_time_str = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        time = experiment_name[:len_time_str]
        full_run_dir = Path(cfg_coarse.inout.full_run_dir).parent
        exist_ok = True # a path for an old experiment should exist

    full_run_dir.mkdir(parents=True, exist_ok=exist_ok)
    print(f"The run will be saved  to: '{str(full_run_dir)}'")
    with open("out_folder.txt", "w") as f:
        f.write(str(full_run_dir))

    coarse_checkpoint_dir = full_run_dir / "coarse" / "checkpoints"
    coarse_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_coarse.inout.full_run_dir = str(coarse_checkpoint_dir.parent)
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_coarse.inout.name = experiment_name

    # if cfg_detail.inout.full_run_dir == 'todo':
    detail_checkpoint_dir = full_run_dir / "detail" / "checkpoints"
    detail_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_detail.inout.full_run_dir = str(detail_checkpoint_dir.parent)
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)
    cfg_detail.inout.name = experiment_name

    conf = DictConfig({})
    conf.coarse = cfg_coarse
    conf.detail = cfg_detail
    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    wandb_logger = WandbLogger(name=experiment_name,
                         project=project_name,
                         config=dict(conf),
                         # version=time,
                         version=experiment_name,
                         save_dir=full_run_dir)

    deca = None
    checkpoint = None
    checkpoint_kwargs = None
    if start_i > 0:
        print(f"Looking for checkpoint in '{configs[start_i-1].inout.checkpoint_dir}'")
        checkpoints = sorted(list(Path(configs[start_i-1].inout.checkpoint_dir).glob("*.ckpt")))
        print(f"Found {len(checkpoints)} checkpoints")
        for ckpt in checkpoints:
            print(f" - {str(ckpt)}")
        checkpoint = str(checkpoints[-1])
        print(f"Loading a checkpoint: {checkpoint} and starting from stage {start_i}")
        checkpoint_kwargs = {
            "model_params": configs[start_i-1].model,
            "learning_params": configs[start_i-1].learning,
            "inout_params": configs[start_i-1].inout,
            "stage_name":  stages_prefixes[start_i-1],
        }

    for i in range(start_i, len(configs)):
        cfg = configs[i]
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
        deca = single_stage_deca_pass(deca, cfg, stages[i], stages_prefixes[i], dm, wandb_logger,
                                      data_preparation_function=prepare_data,
                                      checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs)
        checkpoint = None


def configure_and_finetune(coarse_cfg_default, coarse_overrides, detail_cfg_default, detail_overrides):
    cfg_coarse, cfg_detail = configure(coarse_cfg_default, coarse_overrides, detail_cfg_default, detail_overrides)
    finetune_deca(cfg_coarse, cfg_detail)


def resume_training(run_path, start_at_stage):
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    cfg_coarse = conf.coarse
    cfg_detail = conf.detail
    finetune_deca(cfg_coarse, cfg_detail, start_i=start_at_stage)


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
