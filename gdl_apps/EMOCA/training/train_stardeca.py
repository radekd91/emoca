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


from gdl_apps.EMOCA.training.test_and_finetune_deca import single_stage_deca_pass, create_logger #, get_checkpoint_with_kwargs
from gdl.models.IO import get_checkpoint_with_kwargs
from gdl.datasets.DecaDataModule import DecaDataModule
from gdl.datasets.AffectNetDataModule import AffectNetDataModule
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import datetime
from gdl.utils.other import class_from_str

project_name = 'EmotionalDeca'


def prepare_data(cfg):
    if 'data_class' in cfg.data.keys():
        data_class = cfg.data.data_class
    else:
        data_class = 'DecaDataModule'
    if data_class == 'DecaDataModule':

        if 'expression_constrain_type' in cfg.model.keys() and \
                (cfg.model.expression_constrain_type is not None and str(cfg.model.expression_constrain_type).lower() != 'none'):
            raise ValueError("DecaDataModule does not support expression exchange!")

        dm = DecaDataModule(cfg)
        sequence_name = "DecaData"
    elif 'AffectNet' in data_class:
        if 'augmentation' in cfg.data.keys() and len(cfg.data.augmentation) > 0:
            augmentation = OmegaConf.to_container(cfg.data.augmentation)
        else:
            augmentation = None

        ring_type = cfg.data.ring_type if 'ring_type' in cfg.data.keys() and str(cfg.data.ring_type).lower() != "none" else None
        ring_size = cfg.data.ring_size if 'ring_size' in cfg.data.keys() and str(cfg.data.ring_size).lower() != "none" else None

        if ring_size is not None and 'shape_constrain_type' in cfg.model.keys() and (cfg.model.shape_constrain_type is not None and str(cfg.model.shape_constrain_type).lower() != 'none'):
            raise ValueError("AffectNet does not support shape exchange!")

        drop_last = cfg.data.drop_last if 'drop_last' in cfg.data.keys() and str(cfg.data.drop_last).lower() != "none" else False

        data_module = class_from_str(data_class, sys.modules[__name__])
        dm = data_module(
        # dm = AffectNetDataModule(
            input_dir=cfg.data.input_dir,
            output_dir=cfg.data.output_dir,
            processed_subfolder=cfg.data.processed_subfolder,
            ignore_invalid=False if "ignore_invalid" not in cfg.data.keys() else cfg.data.ignore_invalid,
            mode=cfg.data.mode,
            face_detector=cfg.data.face_detector,
            face_detector_threshold=cfg.data.face_detector_threshold,
            image_size=cfg.data.image_size,
            scale=cfg.data.scale,
            train_batch_size=cfg.learning.batch_size_train,
            val_batch_size=cfg.learning.batch_size_val,
            test_batch_size=cfg.learning.batch_size_test,
            num_workers=cfg.data.num_workers,
            augmentation=augmentation,
            ring_type=ring_type,
            ring_size=ring_size,
            drop_last=drop_last,
            sampler="uniform" if "sampler" not in cfg.data.keys() else cfg.data.sampler,
            processed_ext=".png" if "processed_ext" not in cfg.data.keys() else cfg.data.processed_ext,
        )
        sequence_name = "AffNet"
    else:
        raise ValueError(f"Invalid data_class '{data_class}'")
    return dm, sequence_name


def create_experiment_name(cfg_coarse, cfg_detail, version=2):
    # experiment_name = "ExpDECA"
    experiment_name = cfg_coarse.model.deca_class
    if 'neural_renderer' in cfg_coarse.model.keys() and bool(cfg_coarse.model.neural_renderer):
        experiment_name += "Star"
    if version <= 2:
        if cfg_coarse.data.data_class:
            experiment_name += '_' + cfg_coarse.data.data_class[:5]

        if 'sampler' in cfg_coarse.data.keys() and cfg_coarse.data.sampler != 'uniform':
            experiment_name += '_' + cfg_coarse.data.sampler


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

        d_detail_conditioning = 'concat'
        if 'detail_conditioning_type' in cfg_detail.model.keys():
            d_detail_conditioning = cfg_detail.model.detail_conditioning_type
        if d_detail_conditioning != 'concat':
            experiment_name += "_DD" + d_detail_conditioning

        if cfg_coarse.model.deca_class == "ExpDECA":
            if cfg_coarse.model.expression_backbone == 'deca_parallel':
                experiment_name += '_para'
            elif cfg_coarse.model.expression_backbone == 'deca_clone':
                experiment_name += '_clone'
            elif cfg_coarse.model.expression_backbone == 'emonet_trainable':
                experiment_name += '_EmoTrain'
            elif cfg_coarse.model.expression_backbone == 'emonet_static':
                experiment_name += '_EmoStat'

            if cfg_coarse.model.exp_deca_global_pose:
                experiment_name += '_Glob'
            if cfg_coarse.model.exp_deca_jaw_pose:
                experiment_name += '_Jaw'

        if cfg_coarse.learning.train_K == 1:
            experiment_name += '_NoRing'

        if 'use_vgg' in cfg_coarse.model.keys() and  cfg_coarse.model.use_vgg:
            experiment_name += '_VGGl'

        if not cfg_coarse.model.use_photometric:
            experiment_name += "_noPho"

        experiment_name = experiment_name.replace("/", "_")
        if cfg_coarse.model.use_emonet_loss and cfg_detail.model.use_emonet_loss:
            # experiment_name += '_EmoLossB'
            experiment_name += '_EmoB'
        elif cfg_coarse.model.use_emonet_loss:
            # experiment_name += '_EmoLossC'
            experiment_name += '_EmoC'
        elif cfg_detail.model.use_emonet_loss:
            # experiment_name += '_EmoLossD'
            experiment_name += '_EmoD'
        if cfg_coarse.model.use_emonet_loss or cfg_detail.model.use_emonet_loss:
            experiment_name += '_'
            if 'emonet_model_path' in cfg_coarse.model.keys():
                experiment_name += Path(cfg_coarse.model.emonet_model_path).name[20:30]
                experiment_name += '_'
                if 'emoloss_dual' in cfg_coarse.model.keys() and cfg_coarse.model.emoloss_dual:
                    experiment_name += "du_"
                elif 'emoloss_trainable' in cfg_coarse.model.keys() and cfg_coarse.model.emoloss_trainable:
                    experiment_name += "ft_"
            if cfg_coarse.model.use_emonet_feat_1:
                experiment_name += 'F1'
            if cfg_coarse.model.use_emonet_feat_2:
                experiment_name += 'F2'
                if 'normalize_features' in cfg_coarse.model.keys() and cfg_coarse.model.normalize_features:
                    experiment_name += 'n'
                if 'emo_feat_loss' in cfg_coarse.model.keys() and cfg_coarse.model.emo_feat_loss != 'l1_loss':
                    experiment_name += cfg_coarse.model.emo_feat_loss[:3]
            if cfg_coarse.model.use_emonet_valence:
                experiment_name += 'V'
            if cfg_coarse.model.use_emonet_arousal:
                experiment_name += 'A'
            if cfg_coarse.model.use_emonet_expression:
                experiment_name += 'E'
            if cfg_coarse.model.use_emonet_combined:
                experiment_name += 'C'

        if 'au_loss' in cfg_coarse.model.keys():
            experiment_name += '_AU'
            if cfg_coarse.model.au_loss.feat_loss != 'l1_loss':
                experiment_name += 'f-' + cfg_coarse.model.au_loss.feat_loss[:3]
            if cfg_coarse.model.au_loss.au_loss != 'l1_loss':
                experiment_name += '_c-' + cfg_coarse.model.au_loss.au_loss[:3]

        # if expression exchange and geometric errors are to be computed even for the exchanged
        if 'use_geometric_losses_expression_exchange' in cfg_coarse.model.keys() and \
                cfg_coarse.model.use_geometric_losses_expression_exchange and \
                'expression_constrain_type' in cfg_coarse.model.keys() \
                and cfg_coarse.model.expression_constrain_type == 'exchange':
            experiment_name += '_GeEx'

        if version == 0:
            if cfg_coarse.model.use_emonet_loss or cfg_detail.model.use_emonet_loss:
                experiment_name += 'w-%.05f' % cfg_coarse.model.emonet_weight

        if cfg_coarse.model.use_gt_emotion_loss and cfg_detail.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossB'
        elif cfg_coarse.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossC'
        elif cfg_detail.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossD'

        if cfg_detail.model.useSeg:
            experiment_name += f'_DeSeg{cfg_detail.model.useSeg}'
        else:
            experiment_name += f'_DeSeg{cfg_detail.model.useSeg}'

        if not cfg_detail.model.use_detail_l1:
            experiment_name += '_NoDetL1'
        if not cfg_detail.model.use_detail_mrf:
            experiment_name += '_NoMRF'

        if not cfg_coarse.model.background_from_input and not cfg_detail.model.background_from_input:
            experiment_name += '_BlackB'
        elif not cfg_coarse.model.background_from_input:
            experiment_name += '_BlackC'
        elif not cfg_detail.model.background_from_input:
            experiment_name += '_BlackD'

        if hasattr(cfg_coarse.model, 'expression_constrain_type') and str(cfg_coarse.model.expression_constrain_type).lower() != "none":
            experiment_name += "_Ex" + cfg_coarse.data.ring_type


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

        if 'id_metric' in cfg_coarse.model.keys():
            if cfg_coarse.model.id_metric == 'barlow_twins_headless':
                experiment_name += "_idBTH"
            elif cfg_coarse.model.id_metric == 'barlow_twins':
                experiment_name += "_idBT"
            elif isinstance(cfg_coarse.model.id_metric, str):
                experiment_name += "_" +  cfg_coarse.model.id_metric

            if 'id_trainable' in cfg_coarse.model.keys() and cfg_coarse.model.id_trainable:
                experiment_name += "-ft"
            if 'id_loss_start_step' in cfg_coarse.model.keys() and cfg_coarse.model.id_loss_start_step > 0:
                experiment_name += f"-s{cfg_coarse.model.id_loss_start_step}"

            if 'id_contrastive' in cfg_coarse.model.keys() and cfg_coarse.model.id_contrastive:
                experiment_name += f"-cont"
                if isinstance(cfg_coarse.model.id_contrastive, str):
                    experiment_name += f"{cfg_coarse.model.id_contrastive}"


        if not cfg_detail.model.use_landmarks and cfg_detail.model.train_coarse:
            experiment_name += "NoLmk"

        if cfg_coarse.learning.train_K > 1:
            if version <= 1:
                # if cfg_coarse.model.shape_constrain_type != 'exchange':
                #     experiment_name += f'_Co{cfg_coarse.model.shape_constrain_type}'
                if cfg_detail.model.detail_constrain_type != 'exchange':
                    experiment_name += f'_De{cfg_detail.model.detail_constrain_type}'
            else:
                # if cfg_coarse.model.shape_constrain_type != 'none':
                #     experiment_name += f'_Co{cfg_coarse.model.shape_constrain_type[:2]}'
                if cfg_detail.model.detail_constrain_type != 'none':
                    experiment_name += f'_De{cfg_detail.model.detail_constrain_type[:2]}'

        if 'mlp_emotion_predictor' in cfg_coarse.model.keys() and cfg_coarse.model.mlp_emotion_predictor:
            experiment_name += f"_MLP_{cfg_coarse.model.mlp_emotion_predictor_weight}"

            detach_name = ""
            if 'detach_shape' in cfg_coarse.model.mlp_emotion_predictor.keys() and cfg_coarse.model.mlp_emotion_predictor.detach_shape:
                detach_name += 'S'
            if 'detach_expression' in cfg_coarse.model.mlp_emotion_predictor.keys() and cfg_coarse.model.mlp_emotion_predictor.detach_expression:
                detach_name += 'E'
            if 'detach_detailcode' in cfg_coarse.model.mlp_emotion_predictor.keys() and cfg_coarse.model.mlp_emotion_predictor.detach_detailcode:
                detach_name += 'D'
            if 'detach_jaw' in cfg_coarse.model.mlp_emotion_predictor.keys() and cfg_coarse.model.mlp_emotion_predictor.detach_jaw:
                detach_name += 'J'
            if 'detach_global_pose' in cfg_coarse.model.mlp_emotion_predictor.keys() and cfg_coarse.model.mlp_emotion_predictor.detach_global_pose:
                detach_name += 'G'
            if len(detach_name) > 0:
                experiment_name += "_det" + detach_name

        if 'augmentation' in cfg_coarse.data.keys() and len(cfg_coarse.data.augmentation) > 0:
            experiment_name += "_Aug"

        if cfg_detail.model.train_coarse:
            experiment_name += "_DwC"

        if hasattr(cfg_coarse.learning, 'early_stopping') and cfg_coarse.learning.early_stopping \
                and hasattr(cfg_detail.learning, 'early_stopping') and cfg_detail.learning.early_stopping:
            experiment_name += "_early"

    return experiment_name


def train_stardeca(cfg_coarse, cfg_detail, start_i=-1, resume_from_previous = True,
               force_new_location=False, deca=None):
    configs = [cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    stages = ["train", "test", "train", "test"]
    stages_prefixes = ["", "", "", ""]

    # CAREFUL: debug hacks that have no business being commited
    # configs = [cfg_detail, cfg_detail]
    # stages = ["train", "test"]
    # stages_prefixes = ["", ""]
    # configs = [cfg_coarse, cfg_detail]
    # stages = ["train", "train",]
    # stages_prefixes = ["", ""]

    if start_i >= 0 or force_new_location:
        if resume_from_previous:
            resume_i = start_i - 1
            checkpoint_mode = None # loads latest or best based on cfg
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the next stage {start_i})")
        else:
            resume_i = start_i
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the same stage {start_i})")
            checkpoint_mode = 'latest' # resuminng in the same stage, we want to pick up where we left of
        checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(configs[resume_i], stages_prefixes[resume_i], checkpoint_mode)
        # checkpoint_kwargs = {
        #     "model_params": checkpoint_kwargs["config"]["model"],
        #     "learning_params": checkpoint_kwargs["config"]["learning"],
        #     "inout_params": checkpoint_kwargs["config"]["inout"],
        #     "stage_name": stages_prefixes[resume_i]
        # }
    else:
        checkpoint, checkpoint_kwargs = None, None

    if cfg_coarse.inout.full_run_dir == 'todo' or force_new_location:
        if force_new_location:
            print("The run will be resumed in a new foler (forked)")
            cfg_coarse.inout.previous_run_dir = cfg_coarse.inout.full_run_dir
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        random_id = str(hash(time))
        experiment_name = create_experiment_name(cfg_coarse, cfg_detail)
        full_run_dir = Path(configs[0].inout.output_dir) / (time + "_" + random_id+ "_" + experiment_name)
        exist_ok = False # a path for a new experiment should not yet exist
    else:
        experiment_name = cfg_coarse.inout.name
        len_time_str = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        if hasattr(cfg_coarse.inout, 'time') and cfg_coarse.inout.time is not None:
            time = cfg_coarse.inout.time
        else:
            time = experiment_name[:len_time_str]
        if hasattr(cfg_coarse.inout, 'random_id') and cfg_coarse.inout.random_id is not None:
            random_id = cfg_coarse.inout.random_id
        else:
            random_id = ""
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
    cfg_coarse.inout.time = time
    cfg_coarse.inout.random_id = random_id

    # if cfg_detail.inout.full_run_dir == 'todo':
    detail_checkpoint_dir = full_run_dir / "detail" / "checkpoints"
    detail_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_detail.inout.full_run_dir = str(detail_checkpoint_dir.parent)
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)
    cfg_detail.inout.name = experiment_name
    cfg_detail.inout.time = time
    cfg_detail.inout.random_id = random_id

    # save config to target folder
    conf = DictConfig({})
    conf.coarse = cfg_coarse
    conf.detail = cfg_detail
    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    version = time
    if random_id is not None and len(random_id) > 0:
        version += "_" + cfg_detail.inout.random_id

    wandb_logger = create_logger(
                         cfg_coarse.learning.logger_type,
                         name=experiment_name,
                         project_name=project_name,
                         config=OmegaConf.to_container(conf),
                         version=version,
                         save_dir=full_run_dir)

    # deca = None
    if start_i >= 0 or force_new_location:
        print(f"Loading a checkpoint: {checkpoint} and starting from stage {start_i}")
    if start_i == -1:
        start_i = 0
    for i in range(start_i, len(configs)):
        cfg = configs[i]
        deca = single_stage_deca_pass(deca, cfg, stages[i], stages_prefixes[i], dm=None, logger=wandb_logger,
                                      data_preparation_function=prepare_data,
                                      checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs)
        checkpoint = None


def configure(coarse_cfg_default, coarse_overrides,
              detail_cfg_default, detail_overrides):
    from hydra.experimental import compose, initialize
    initialize(config_path="../emoca_conf", job_name="train_deca")
    cfg_coarse = compose(config_name=coarse_cfg_default, overrides=coarse_overrides)
    cfg_detail = compose(config_name=detail_cfg_default, overrides=detail_overrides)
    return cfg_coarse, cfg_detail


def configure_and_train(coarse_cfg_default, coarse_overrides,
                        detail_cfg_default, detail_overrides):
    cfg_coarse, cfg_detail = configure(coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)
    train_stardeca(cfg_coarse, cfg_detail)


def configure_and_resume(run_path,
                         coarse_cfg_default, coarse_overrides,
                         detail_cfg_default, detail_overrides,
                         start_at_stage):
    cfg_coarse, cfg_detail = configure(
                                       coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)

    cfg_coarse_pretrain_, cfg_coarse_, cfg_detail_ = load_configs(run_path)
    deca = None
    if cfg_coarse_pretrain_ is not None:
        checkpoint_mode = 'best'
        checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg_coarse_pretrain_, "train", checkpoint_mode)
        deca = instantiate_deca(cfg_coarse_pretrain_, "train",  checkpoint, checkpoint_kwargs )

        if cfg_coarse.model.resume_training:
            raise ValueError("We just loaded a pretrained model and the config is set to reload the old originam model. "
                             "Ths is probably not what you want.")

    if start_at_stage < 2:
        raise RuntimeError("Resuming before stage 2 makes no sense, that would be training from scratch")
    elif start_at_stage == 2:
        cfg_coarse = cfg_coarse_
    elif start_at_stage == 3:
        raise RuntimeError("Resuming for stage 3 makes no sense, that is a testing stage")
    else:
        raise RuntimeError(f"Cannot resume at stage {start_at_stage}")

    train_stardeca(cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=True, #important, resume from previous stage's checkpoint
               force_new_location=True,
               deca=deca)


def load_configs(run_path):
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    if 'coarse_pretraining' in conf.keys():
        cfg_pretrain = conf.coarse_pretraining
    else:
        cfg_pretrain = None
    cfg_coarse = conf.coarse
    cfg_detail = conf.detail
    return cfg_pretrain, cfg_coarse, cfg_detail


def resume_training(run_path, start_at_stage, resume_from_previous, force_new_location):
    cfg_pretrain, cfg_coarse, cfg_detail = load_configs(run_path)

    train_stardeca(cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=resume_from_previous,
               force_new_location=force_new_location)


# def resume_training_(run_path, start_at_stage, resume_from_previous, force_new_location):
#     cfg_coarse, cfg_detail = load_configs(run_path)
#
#     ## LOAD the best checkpoint from the pretrained path.
#     checkpoint_mode = 'best'
#     mode = "train"
#     checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg_pretrain_, mode, checkpoint_mode=checkpoint_mode,
#                                                                replace_root=replace_root, relative_to=relative_to)
#     # make sure you use the deca class of the target (for instance, if target is ExpDECA but we're starting from
#     # pretrained EMOCA)
#     # cfg_pretrain_.model.deca_class = cfg_coarse.model.deca_class
#     # checkpoint_kwargs["config"]["model"]["deca_class"] = cfg_coarse.model.deca_class
#     # load from configs
#     from gdl.models.EMOCA import instantiate_deca
#
#     deca_checkpoint_kwargs = {
#         "model_params": checkpoint_kwargs["config"]["model"],
#         "learning_params": checkpoint_kwargs["config"]["learning"],
#         "inout_params": checkpoint_kwargs["config"]["inout"],
#         "stage_name": "train",
#     }
#
#     deca = instantiate_deca(cfg_pretrain_, mode, "", checkpoint, deca_checkpoint_kwargs)
#
#     train_stardeca(cfg_coarse, cfg_detail,
#                    start_i=start_at_stage,
#                    resume_from_previous=resume_from_previous,
#                    force_new_location=force_new_location)

def configure_and_finetune_from_pretrained(coarse_conf, coarse_override, detail_conf, detail_override, path_to_resume_from,
                                           replace_root = None, relative_to = None,):


    cfg_coarse, cfg_detail = configure(coarse_conf, coarse_override,
                                       detail_conf, detail_override)

    cfg_pretrain_, cfg_coarse_, cfg_detail_ = load_configs(path_to_resume_from)
    if cfg_pretrain_ is None:
        raise ValueError("Pretrained regressor seems to be None")
    else:
        ## LOAD the best checkpoint from the pretrained path.
        checkpoint_mode = 'best'
        mode = "train"
        checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg_pretrain_, mode, checkpoint_mode=checkpoint_mode,
                                                                   replace_root = replace_root, relative_to = relative_to)
        # make sure you use the deca class of the target (for instance, if target is ExpDECA but we're starting from
        # pretrained EMOCA)
        # cfg_pretrain_.model.deca_class = cfg_coarse.model.deca_class
        # checkpoint_kwargs["config"]["model"]["deca_class"] = cfg_coarse.model.deca_class
        # load from configs
        from gdl.models.DECA import instantiate_deca

        deca_checkpoint_kwargs = {
            "model_params": checkpoint_kwargs["config"]["model"],
            "learning_params": checkpoint_kwargs["config"]["learning"],
            "inout_params": checkpoint_kwargs["config"]["inout"],
            "stage_name": "train",
        }

        deca = instantiate_deca(cfg_pretrain_, mode, "",  checkpoint, deca_checkpoint_kwargs )

    train_stardeca(cfg_coarse, cfg_detail,
               # start_i=start_at_stage,
               # resume_from_previous=resume_from_previous,
               # force_new_location=True,
               deca=deca)

def finetune_from_pretrained(coarse_conf, detail_conf, resume_from_run_path):

    cfg_pretrain, cfg_coarse, cfg_detail = load_configs(resume_from_run_path)
    if cfg_pretrain is None:
        raise ValueError("Pretrained regressor seems to be None")
    else:
        ## LOAD the best checkpoint from the pretrained path.
        checkpoint_mode = 'best'
        mode = "train"
        checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg_pretrain, mode, checkpoint_mode=checkpoint_mode,
                                                                   replace_root = None, relative_to = None)
        # make sure you use the deca class of the target (for instance, if target is ExpDECA but we're starting from
        # pretrained EMOCA)

        # cfg_pretrain_.model.deca_class = cfg_coarse.model.deca_class
        # checkpoint_kwargs["config"]["model"]["deca_class"] = cfg_coarse.model.deca_class
        # load from configs
        from gdl.models.DECA import instantiate_deca
        deca_checkpoint_kwargs = {
            "model_params": checkpoint_kwargs["config"]["model"],
            "learning_params": checkpoint_kwargs["config"]["learning"],
            "inout_params": checkpoint_kwargs["config"]["inout"],
            "stage_name": "train",
        }
        deca = instantiate_deca(cfg_pretrain, mode, "",  checkpoint, deca_checkpoint_kwargs )

    train_stardeca(coarse_conf, detail_conf,
               # start_i=start_at_stage,
               # resume_from_previous=resume_from_previous,
               # force_new_location=True,
               deca=deca)


def main():
    configured = False

    if len(sys.argv) <= 2:
        # coarse_conf = "deca_train_coarse_stargan"
        coarse_conf = "deca_train_coarse"
        # coarse_conf = "deca_train_detail"
        # coarse_conf = "deca_train_detail_stargan"
        # detail_conf = "deca_train_detail_stargan"
        detail_conf = "deca_train_detail"

        path_to_resume_from = None
        # path_to_resume_from = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_21-50-45_DECA__DeSegFalse_early/"  # My EMOCA, ResNet backbones
        # path_to_resume_from = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_23-19-03_DECA__EFswin_s_EDswin_s_DeSegFalse_early/" # My EMOCA, SWIN small
        # path_to_resume_from = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_23-19-04_DECA__EFswin_t_EDswin_t_DeSegFalse_early/" # My EMOCA, SWIN tiny

        # flame_encoder = 'swin_tiny_patch4_window7_224'
        # detail_encoder = 'swin_tiny_patch4_window7_224'
        flame_encoder = 'ResnetEncoder'
        detail_encoder = 'ResnetEncoder'

        detail_conditioning_list = ['identity', 'detail']  # default
        emo_detail_conditioning_list = ['jawpose', 'expression', 'detailemo']
        n_detail_emo = 50

        coarse_override = [
            # 'model/settings=coarse_train',
            # 'model/settings=coarse_train_emonet',
            'model/settings=coarse_train_expdeca',
            # 'model/settings=coarse_train_expdeca_emonet',
            # 'model/settings=coarse_train_expdeca_emomlp',
            # 'model/settings=coarse_train_expdeca_emomlp',
            # 'model.expression_constrain_type=exchange',
            # 'model.expression_constrain_use_jaw_pose=True',
            # 'model.expression_constrain_use_global_pose=False',
            # 'model.use_geometric_losses_expression_exchange=True',

            # '+model.mlp_emotion_predictor.detach_shape=True',
            # '+model.mlp_emotion_predictor.detach_expression=True',
            # '+model.mlp_emotion_predictor.detach_detailcode=True',
            # '+model.mlp_emotion_predictor.detach_jaw=True',
            # '+model.mlp_emotion_predictor.detach_global_pose=True',

            # 'data/datasets=affectnet_desktop', # affectnet vs deca dataset
            # # f'data.ring_type=gt_va',
            # #  'data.ring_size=4',
            # #  'learning/batching=single_gpu_expdeca_coarse_ring',
            'data.num_workers=0',
            f'model.resume_training={path_to_resume_from is None}', # load the original EMOCA model
            'model.useSeg=False', # do not segment out the background from the coarse image
            'model.shape_constrain_type=shuffle_expression',
            'model.background_from_input=input',
            # '+model.detail_conditioning_type=adain',
            # 'learning.early_stopping.patience=5',
            'learning/logging=none',
            # 'learning.num_gpus=2',
            'learning.batch_size_train=4',
            'learning.batch_size_val=4',
            'learning.train_K=2',
            'learning.val_K=2',
            'learning.test_K=2',
            # '+model.emonet_model_path=/ps/scratch/rdanecek/emoca/emodeca/2021_08_20_09-43-26_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000',
            # '+model.emonet_model_path=/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_23-50-06_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early',
            '+model.emonet_model_path=/ps/scratch/rdanecek/emoca/emodeca/2021_08_23_22-52-24_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early',
            # '+model.emoloss_dual=true',
            '+model/additional=vgg_loss',
            f'+model.e_flame_type={flame_encoder}',
            f'+model.e_detail_type={detail_encoder}',
            f'+model.detail_conditioning={detail_conditioning_list}',
            f'+model.detailemo_conditioning={emo_detail_conditioning_list}',
            '+model.emoloss_trainable=true',
            # '+model.normalize_features=true', # normalize emonet features before applying loss
            # '+model.emo_feat_loss=l1_loss', # emonet feature loss
            # '+model.emo_feat_loss=barlow_twins_headless', # emonet feature loss
            # '+model.id_metric=barlow_twins_headless',
            '+model.emo_feat_loss=barlow_twins',  # emonet feature loss
            '+model.emo_contrastive=True',  # emonet feature contrastive
            '+model.id_metric=barlow_twins',
            '+model.id_trainable=True',
            '+model.id_contrastive=True',
            # '+model.id_loss_start_step=3',
            # '+model.emo_feat_loss=cosine_similarity', # emonet feature loss
            # '+model/additional=au_loss_dual', # emonet feature loss
            # 'model.au_loss.feat_loss=cosine_similarity',  # emonet feature loss
            # 'model.au_loss.feat_loss=kl_div',  # emonet feature loss
                              ]
        detail_override = [
            # 'model/settings=detail_train',
            # 'model/settings=detail_train_emonet',
            'model/settings=detail_train_expdeca',
            # 'model/settings=detail_train_expdeca_emonet',
            # 'model/settings=detail_train_expdeca_emomlp',
            # 'model.expression_constrain_type=exchange',
            # 'model.expression_constrain_use_jaw_pose=True',
            # 'model.expression_constrain_use_global_pose=False',
            # 'model.use_geometric_losses_expression_exchange=True',
            # '+model.mlp_emotion_predictor.detach_shape=True',
            # '+model.mlp_emotion_predictor.detach_expression=True',
            # '+model.mlp_emotion_predictor.detach_detailcode=True',
            # '+model.mlp_emotion_predictor.detach_jaw=True',
            # '+model.mlp_emotion_predictor.detach_global_pose=True',
            # 'data/datasets=affectnet_desktop', # affectnet vs deca dataset
            # f'data.ring_type=gt_va',
            #  'learning/batching=single_gpu_expdeca_detail_ring',
            #  'data.ring_size=4',
            # 'learning.early_stopping.patience=5',
            'learning/logging=none',
            'data.num_workers=0',
            'model.useSeg=False',
            'model.background_from_input=input',
            '+model.detail_conditioning_type=adain',
            # 'learning.batch_size_train=4',
            # 'learning.train_K=1',
            # '+model.emonet_model_path=/ps/scratch/rdanecek/emoca/emodeca/2021_08_20_09-43-26_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000',
            # '+model.emonet_model_path=/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_23-50-06_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early',
            '+model.emonet_model_path=/ps/scratch/rdanecek/emoca/emodeca/2021_08_23_22-52-24_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early',
            # '+model.emoloss_trainable=true',
            # '+model.emoloss_dual=true',
            '+model/additional=vgg_loss',
            f'+model.e_flame_type={flame_encoder}',
            f'+model.e_detail_type={detail_encoder}',
            # '+model.normalize_features=true',  # normalize emonet features before applying loss
            # '+model.emo_feat_loss=l1_loss',  # emonet feature loss
            # '+model.emo_feat_loss=barlow_twins_headless',  # emonet feature loss
            '+model.emo_feat_loss=cosine_similarity',  # emonet feature loss
            # '+model/additional=au_loss_dual',  # emonet feature loss
            # 'model.au_loss.feat_loss=cosine_similarity',  # emonet feature loss
            # 'model.au_loss.feat_loss=kl_div',  # emonet feature loss
        ]


        # coarse_conf = detail_conf
        # coarse_override = detail_override

    elif len(sys.argv) > 2:
        if Path(sys.argv[1]).is_file():
            configured = True
            with open(sys.argv[1], 'r') as f:
                coarse_conf = OmegaConf.load(f)
            with open(sys.argv[2], 'r') as f:
                detail_conf = OmegaConf.load(f)
        else:
            coarse_conf = sys.argv[1]
            detail_conf = sys.argv[2]

        if len(sys.argv) > 3:
            path_to_resume_from = sys.argv[3]
        else:
            path_to_resume_from = None

    else:
        coarse_conf = "deca_finetune_coarse_cluster"
        detail_conf = "deca_finetune_detail_cluster"
        coarse_override = []
        detail_override = []

    if len(sys.argv) > 4:
        coarse_override = sys.argv[3]
        detail_override = sys.argv[4]
    # else:
    #     coarse_override = []
    #     detail_override = []
    if configured:
        if path_to_resume_from is None:
            train_stardeca(coarse_conf, detail_conf)
        else:
            finetune_from_pretrained(coarse_conf, detail_conf, path_to_resume_from)
    else:
        if path_to_resume_from is None:
            configure_and_train(coarse_conf, coarse_override, detail_conf, detail_override)
        else:
            relative_to_path = '/ps/scratch/' #local run hack
            replace_root_path = '/home/rdanecek/Workspace/mount/scratch/' #local run hack
            configure_and_finetune_from_pretrained(coarse_conf, coarse_override, detail_conf, detail_override,
                                                   path_to_resume_from,
                                                   relative_to=relative_to_path,
                                                   replace_root=replace_root_path)


if __name__ == "__main__":
    main()

