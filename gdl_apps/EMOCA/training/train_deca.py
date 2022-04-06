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


from gdl_apps.EMOCA.training.test_and_finetune_deca import single_stage_deca_pass, get_checkpoint_with_kwargs, create_logger
from gdl.datasets.DecaDataModule import DecaDataModule
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import datetime


project_name = 'EmotionalDeca'


def prepare_data(cfg):
    dm = DecaDataModule(cfg)
    sequence_name = "ClassicDECA"
    return dm, sequence_name


def create_experiment_name(cfg_coarse_pre, cfg_coarse, cfg_detail, version=1):
    experiment_name = "DECA_"
    if version <= 1:
        experiment_name = experiment_name.replace("/", "_")
        if cfg_coarse.model.use_emonet_loss and cfg_detail.model.use_emonet_loss:
            experiment_name += '_EmoLossB'
        elif cfg_coarse.model.use_emonet_loss:
            experiment_name += '_EmoLossC'
        elif cfg_detail.model.use_emonet_loss:
            experiment_name += '_EmoLossD'
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


        e_flame_type = 'ResnetEncoder'
        if 'e_flame_type' in cfg_coarse.model.keys():
            e_flame_type = cfg_coarse.model.e_flame_type
        if e_flame_type != 'ResnetEncoder':
            experiment_name += "_EF" + e_flame_type[:6]

        e_detail_type = 'ResnetEncoder'
        if 'e_detail_type' in cfg_detail.model.keys():
            e_detail_type = cfg_detail.model.e_detail_type
        if e_detail_type != 'ResnetEncoder':
            experiment_name += "_ED" + e_detail_type[:6]

        if cfg_coarse.model.use_gt_emotion_loss and cfg_detail.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossB'
        elif cfg_coarse.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossC'
        elif cfg_detail.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossD'

        if version == 0:
            if cfg_coarse.model.useSeg:
                experiment_name += '_CoSegGT'
            else:
                experiment_name += '_CoSegRend'

            if cfg_detail.model.useSeg:
                experiment_name += '_DeSegGT'
            else:
                experiment_name += '_DeSegRend'

        if cfg_detail.model.useSeg:
            experiment_name += f'_DeSeg{cfg_detail.model.useSeg}'
        else:
            experiment_name += f'_DeSeg{cfg_detail.model.useSeg}'

        if not cfg_detail.model.use_detail_l1:
            experiment_name += '_NoDetL1'
        if not cfg_detail.model.use_detail_mrf:
            experiment_name += '_NoMRF'

        # if not cfg_coarse.model.background_from_input and not cfg_detail.model.background_from_input:
        #     experiment_name += '_BackBlackB'
        # elif not cfg_coarse.model.background_from_input:
        #     experiment_name += '_BackBlackC'
        # elif not cfg_detail.model.background_from_input:
        #     experiment_name += '_BackBlackD'
        if not cfg_coarse.model.background_from_input and not cfg_detail.model.background_from_input:
            experiment_name += '_BlackB'
        elif not cfg_coarse.model.background_from_input:
            experiment_name += '_BlackC'
        elif not cfg_detail.model.background_from_input:
            experiment_name += '_BlackD'

        if version == 0:
            if cfg_coarse.learning.learning_rate != 0.0001:
                experiment_name += f'CoLR-{cfg_coarse.learning.learning_rate}'
            if cfg_detail.learning.learning_rate != 0.0001:
                experiment_name += f'DeLR-{cfg_detail.learning.learning_rate}'

        if version == 0:
            if cfg_coarse.model.use_photometric:
                experiment_name += 'CoPhoto'
            if cfg_coarse.model.use_landmarks:
                experiment_name += 'CoLMK'
            if cfg_coarse.model.idw:
                experiment_name += f'_IDW-{cfg_coarse.model.idw}'

        if cfg_coarse.model.shape_constrain_type != 'exchange':
            experiment_name += f'_Co{cfg_coarse.model.shape_constrain_type}'
        if cfg_detail.model.detail_constrain_type != 'exchange':
            experiment_name += f'_De{cfg_coarse.model.detail_constrain_type}'

        if 'augmentation' in cfg_coarse.data.keys() and len(cfg_coarse.data.augmentation) > 0:
            experiment_name += "_Aug"

        if cfg_detail.model.train_coarse:
            experiment_name += "_DwC"

        if hasattr(cfg_coarse.learning, 'early_stopping') and cfg_coarse.learning.early_stopping \
            and hasattr(cfg_detail.learning, 'early_stopping') and cfg_detail.learning.early_stopping:
            experiment_name += "_early"

    return experiment_name


def train_deca(cfg_coarse_pretraining, cfg_coarse, cfg_detail, start_i=-1, resume_from_previous = True,
               force_new_location=False):
    configs = [cfg_coarse_pretraining, cfg_coarse_pretraining, cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    stages = ["train", "test", "train", "test", "train", "test"]
    stages_prefixes = ["pretrain", "pretrain", "", "", "", ""]
    # configs = [cfg_coarse_pretraining, cfg_coarse, cfg_detail]
    # stages = ["train", "train", "train",]
    # stages_prefixes = ["pretrain", "", ""]

    if start_i >= 0 or force_new_location:
        if resume_from_previous:
            resume_i = start_i - 1
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the next stage {start_i})")
        else:
            resume_i = start_i
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the same stage {start_i})")
        checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(configs[resume_i], stages_prefixes[resume_i])
    else:
        checkpoint, checkpoint_kwargs = None, None

    if cfg_coarse.inout.full_run_dir == 'todo' or force_new_location:
        if force_new_location:
            print("The run will be resumed in a new foler (forked)")
            cfg_coarse.inout.previous_run_dir = cfg_coarse.inout.full_run_dir
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        experiment_name = create_experiment_name(cfg_coarse_pretraining, cfg_coarse, cfg_detail)
        full_run_dir = Path(configs[0].inout.output_dir) / (time + "_" + experiment_name)
        # exist_ok = False # a path for a new experiment should not yet exist
        exist_ok = True # actually, for multi-gpu training it might be initialized by one of the processes earlier.
    else:
        experiment_name = cfg_coarse.inout.name
        len_time_str = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        if hasattr(cfg_coarse.inout, 'time') and cfg_coarse.inout.time is not None:
            time = cfg_coarse.inout.time
        else:
            time = experiment_name[:len_time_str]
        full_run_dir = Path(cfg_coarse.inout.full_run_dir).parent
        exist_ok = True # a path for an old experiment should exist

    full_run_dir.mkdir(parents=True, exist_ok=exist_ok)
    print(f"The run will be saved  to: '{str(full_run_dir)}'")
    with open("out_folder.txt", "w") as f:
        f.write(str(full_run_dir))

    coarse_pretrain_checkpoint_dir = full_run_dir / "coarse_pretrain" / "checkpoints"
    coarse_pretrain_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_coarse_pretraining.inout.full_run_dir = str(coarse_pretrain_checkpoint_dir.parent)
    cfg_coarse_pretraining.inout.checkpoint_dir = str(coarse_pretrain_checkpoint_dir)
    cfg_coarse_pretraining.inout.name = experiment_name
    cfg_coarse_pretraining.inout.time = time

    coarse_checkpoint_dir = full_run_dir / "coarse" / "checkpoints"
    coarse_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_coarse.inout.full_run_dir = str(coarse_checkpoint_dir.parent)
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_coarse.inout.name = experiment_name
    cfg_coarse.inout.time = time

    # if cfg_detail.inout.full_run_dir == 'todo':
    detail_checkpoint_dir = full_run_dir / "detail" / "checkpoints"
    detail_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_detail.inout.full_run_dir = str(detail_checkpoint_dir.parent)
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)
    cfg_detail.inout.name = experiment_name
    cfg_detail.inout.time = time

    # save config to target folder
    conf = DictConfig({})
    conf.coarse_pretraining = cfg_coarse_pretraining
    conf.coarse = cfg_coarse
    conf.detail = cfg_detail
    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    # wandb_logger = WandbLogger(name=experiment_name,
    #                      project=project_name,
    #                      config=OmegaConf.to_container(conf),
    #                      version=time,
    #                      save_dir=full_run_dir)
    wandb_logger = create_logger(
                         cfg_coarse_pretraining.learning.logger_type,
                         name=experiment_name,
                         project_name=project_name,
                         config=OmegaConf.to_container(conf),
                         version=time,
                         save_dir=full_run_dir)

    deca = None
    if start_i >= 0 or force_new_location:
        print(f"Loading a checkpoint: {checkpoint} and starting from stage {start_i}")
    if start_i == -1:
        start_i = 0

    for i in range(start_i, len(configs)):
        cfg = configs[i]
        deca = single_stage_deca_pass(deca, cfg, stages[i], stages_prefixes[i], dm=None, logger=wandb_logger,
                                      data_preparation_function=prepare_data,
                                      checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs
                                      )
        checkpoint = None


def configure(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
              coarse_cfg_default, coarse_overrides,
              detail_cfg_default, detail_overrides):
    from hydra.experimental import compose, initialize
    initialize(config_path="../emoca_conf", job_name="train_deca")
    cfg_coarse_pretrain = compose(config_name=coarse_pretrain_cfg_default, overrides=coarse_pretrain_overrides)
    cfg_coarse = compose(config_name=coarse_cfg_default, overrides=coarse_overrides)
    cfg_detail = compose(config_name=detail_cfg_default, overrides=detail_overrides)
    return cfg_coarse_pretrain, cfg_coarse, cfg_detail



def configure_and_train(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                        coarse_cfg_default, coarse_overrides,
                        detail_cfg_default, detail_overrides):
    cfg_coarse_pretrain, cfg_coarse, cfg_detail = configure(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                                       coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)
    train_deca(cfg_coarse_pretrain, cfg_coarse, cfg_detail,
               # start_i=4,
               # force_new_location=True
               )


def configure_and_resume(run_path,
                         coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                         coarse_cfg_default, coarse_overrides,
                         detail_cfg_default, detail_overrides,
                         start_at_stage):
    cfg_coarse_pretrain, cfg_coarse, cfg_detail = configure(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                                       coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)

    cfg_coarse_pretrain_, cfg_coarse_, cfg_detail_ = load_configs(run_path)

    if start_at_stage < 2:
        raise RuntimeError("Resuming before stage 2 makes no sense, that would be training from scratch")
    if start_at_stage == 2:
        cfg_coarse_pretrain = cfg_coarse_pretrain_
    elif start_at_stage == 3:
        raise RuntimeError("Resuming for stage 3 makes no sense, that is a testing stage")
    elif start_at_stage == 4:
        cfg_coarse_pretrain = cfg_coarse_pretrain_
        cfg_coarse = cfg_coarse_
    elif start_at_stage == 5:
        raise RuntimeError("Resuming for stage 5 makes no sense, that is a testing stage")
    else:
        raise RuntimeError(f"Cannot resume at stage {start_at_stage}")

    train_deca(cfg_coarse_pretrain, cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=True, #important, resume from previous stage's checkpoint
               force_new_location=True)


def load_configs(run_path):
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    cfg_coarse_pretraining = conf.coarse_pretraining
    cfg_coarse = conf.coarse
    cfg_detail = conf.detail
    return cfg_coarse_pretraining, cfg_coarse, cfg_detail


def resume_training(run_path, start_at_stage, resume_from_previous, force_new_location):
    cfg_coarse_pretraining, cfg_coarse, cfg_detail = load_configs(run_path)
    train_deca(cfg_coarse_pretraining, cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=resume_from_previous,
               force_new_location=force_new_location)

# @hydra.main(config_path="../emoca_conf", config_name="deca_finetune")
# def main(cfg : DictConfig):
def main():
    configured = False
    if len(sys.argv) >= 4:
        if Path(sys.argv[1]).is_file():
            configured = True
            with open(sys.argv[1], 'r') as f:
                coarse_pretrain_conf = OmegaConf.load(f)
            with open(sys.argv[2], 'r') as f:
                coarse_conf = OmegaConf.load(f)
            with open(sys.argv[3], 'r') as f:
                detail_conf = OmegaConf.load(f)
        else:
            coarse_pretrain_conf = sys.argv[1]
            coarse_conf = sys.argv[2]
            detail_conf = sys.argv[3]
    else:
        coarse_pretrain_conf = "deca_train_coarse_pretrain"
        coarse_conf = "deca_train_coarse"
        detail_conf = "deca_train_detail"

        # flame_encoder = 'swin_tiny_patch4_window7_224'
        # detail_encoder = 'swin_tiny_patch4_window7_224'
        # nr = "stargan"
        nr = "none"
        # logger = "none"
        logger = "wandb"

        ## DECA COARSE PRETRAINING STAGE CONFIGS (WITHOUT RENDERING)
        coarse_pretrain_override = [
                                    'learning/batching=single_gpu_coarse_pretrain_32gb',
                                    # 'learning/batching=single_gpu_coarse_pretrain',
                                    # f'model/neural_rendering={nr}',
                                    f'learning/logging={logger}',
                                    # f'+model.e_flame_type={flame_encoder}',
                                    # f'+model.e_detail_type={detail_encoder}'
                                    ]
        
        ## DECA COARSE STAGE CONFIGS
        coarse_override = [
                            # 'learning/batching=single_gpu_coarse_32gb',
                            'learning/batching=single_gpu_coarse',
                           f'model/neural_rendering={nr}',
                            f'learning/logging={logger}',
                           # f'+model.e_flame_type={flame_encoder}',
                           # f'+model.e_detail_type={detail_encoder}'
                           ]

        ## DECA DETAIL STAGE CONFIGS
        detail_override = [
                            # 'learning/batching=single_gpu_detail_32gb',
                            'learning/batching=single_gpu_detail',
                           f'model/neural_rendering={nr}',
                            f'learning/logging={logger}',
                            f'model.background_from_input=False',
                           # f'+model.e_flame_type={flame_encoder}',
                           # f'+model.e_detail_type={detail_encoder}'
                           ]


    if len(sys.argv) >= 7:
        coarse_pretrain_override = sys.argv[4]
        coarse_override = sys.argv[5]
        detail_override = sys.argv[6]

    if configured:
        train_deca(coarse_pretrain_conf, coarse_conf, detail_conf)
    else:
        configure_and_train(coarse_pretrain_conf, coarse_pretrain_override,
                            coarse_conf, coarse_override, detail_conf, detail_override)


if __name__ == "__main__":
    main()

