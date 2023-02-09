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
from gdl.datasets.AffectNetDataModule import AffectNetDataModule, AffectNetDataModuleValTest, \
    AffectNetEmoNetSplitModuleValTest, AffectNetEmoNetSplitModule, AffectNetEmoNetSplitModuleTest
from gdl.datasets.AfewVaDataModule import AfewVaDataModule
from gdl.datasets.CombinedDataModule import CombinedDataModule
from gdl.datasets.EmotioNetDataModule import EmotioNetDataModule
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import datetime
from gdl.utils.other import class_from_str


project_name = 'EmotionalDeca'

def create_single_dm(cfg, data_class):
    if data_class in 'DecaDataModule':
        if 'expression_constrain_type' in cfg.model.keys() and \
                (cfg.model.expression_constrain_type is not None and str(cfg.model.expression_constrain_type).lower() != 'none'):
            raise ValueError("DecaDataModule does not support expression exchange!")

        dm = DecaDataModule(cfg)
        sequence_name = "DecaData"
        return dm, sequence_name

    if 'augmentation' in cfg.data.keys() and len(cfg.data.augmentation) > 0:
        augmentation = OmegaConf.to_container(cfg.data.augmentation)
    else:
        augmentation = None

    if 'AffectNet' in data_class:
        if 'augmentation' in cfg.data.keys() and len(cfg.data.augmentation) > 0:
            augmentation = OmegaConf.to_container(cfg.data.augmentation)
        else:
            augmentation = None

        ring_type = cfg.data.ring_type if 'ring_type' in cfg.data.keys() and str(cfg.data.ring_type).lower() != "none" else None
        ring_size = cfg.data.ring_size if 'ring_size' in cfg.data.keys() and str(cfg.data.ring_size).lower() != "none" else None

        if ring_size is not None and 'shape_constrain_type' in cfg.model.keys() and (cfg.model.shape_constrain_type is not None and str(cfg.model.shape_constrain_type).lower() != 'none'):
            raise ValueError("AffectNet does not support shape exchange!")

        drop_last = cfg.data.drop_last if 'drop_last' in cfg.data.keys() and str(cfg.data.drop_last).lower() != "none" else False

        if 'AffectNet' in data_class:
            data_module = class_from_str(data_class, sys.modules[__name__])
            # dm = AffectNetDataModule(
            dm = data_module(
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
                dataset_type=cfg.data.dataset_type if "dataset_type" in cfg.data.keys() else None,
                use_gt=cfg.data.use_gt if "use_gt" in cfg.data.keys() else True,
            )
        else:
            raise ValueError(f"Uknown data module class '{data_class}'")
        sequence_name = "AffNet"
    elif data_class == 'EmotioNetDataModule':

        dm = EmotioNetDataModule(
            input_dir=cfg.data.input_dir,
            output_dir=cfg.data.output_dir,
            processed_subfolder=cfg.data.processed_subfolder,
            scale=cfg.data.scale,
            image_size=cfg.data.image_size,
            bb_center_shift_x=cfg.data.bb_center_shift_x,
            bb_center_shift_y=cfg.data.bb_center_shift_y,
            augmentation=augmentation
        )

        sequence_name = "EmotioNet"
    elif data_class == 'AfewVaDataModule':
        dm = AfewVaDataModule(
            input_dir=cfg.data.input_dir,
            output_dir=cfg.data.output_dir,
            processed_subfolder=cfg.data.processed_subfolder,
            # ignore_invalid=False if "ignore_invalid" not in cfg.data.keys() else cfg.data.ignore_invalid,
            face_detector=cfg.data.face_detector,
            face_detector_threshold=cfg.data.face_detector_threshold,
            image_size=cfg.data.image_size,
            scale=cfg.data.scale,
            train_batch_size=cfg.learning.batch_size_train,
            val_batch_size=cfg.learning.batch_size_val,
            test_batch_size=cfg.learning.batch_size_test,
            num_workers=cfg.data.num_workers,
            augmentation=augmentation,
            ring_type=None,
            ring_size=None,
            drop_last=cfg.data.drop_last if 'drop_last' in cfg.data.keys() and str(cfg.data.drop_last).lower() != "none" else False,
            sampler="uniform" if "sampler" not in cfg.data.keys() else cfg.data.sampler,
            processed_ext=".png" if "processed_ext" not in cfg.data.keys() else cfg.data.processed_ext,
            dataset_type=cfg.data.dataset_type if "dataset_type" in cfg.data.keys() else None,
            # use_gt=cfg.data.use_gt if "use_gt" in cfg.data.keys() else True,
            k_fold_crossvalidation=cfg.data.k_fold_crossvalidation if "k_fold_crossvalidation" in cfg.data.keys() else None,
            k_index=cfg.data.k_index if "k_index" in cfg.data.keys() else None,
        )
        sequence_name = "AFEW-VA"
    else:
        raise ValueError(f"Invalid data_class '{data_class}'")
    return dm, sequence_name

def prepare_data(cfg):
    if 'data_class' in cfg.data.keys():
        data_class = cfg.data.data_class
    else:
        data_class = 'DecaDataModule'

    if data_class == 'CombinedDataModule':
        data_classes =  cfg.data.data_classes
        dms = {}
        for data_class in data_classes:
            dm, sequence_name = create_single_dm(cfg, data_class)
            dms[data_class] = dm
        data_class_weights = OmegaConf.to_container(cfg.data.data_class_weights) if 'data_class_weights' in cfg.data.keys() else None
        if data_class_weights is not None:
            data_class_weights = dict(zip(data_classes, data_class_weights))
        dm = CombinedDataModule(dms, data_class_weights)
        sequence_name = "ComboNet"
    else:
        dm, sequence_name = create_single_dm(cfg, data_class)

    return dm, sequence_name


def create_experiment_name(cfg_coarse, cfg_detail, version=2):
    # experiment_name = "ExpDECA"
    experiment_name = cfg_coarse.model.deca_class
    if version <= 2:
        if cfg_coarse.data.data_class:
            experiment_name += '_' + cfg_coarse.data.data_class[:5]

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

        zero_out = cfg_coarse.model.get('zero_out_last_enc_layer', False)
        if zero_out:
            experiment_name += '_Z'

        if cfg_coarse.learning.train_K == 1:
            experiment_name += '_NoRing'

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

        if 'au_loss' in cfg_coarse.model.keys():
            experiment_name += '_AU'
            if cfg_coarse.model.au_loss.feat_loss != 'l1_loss':
                experiment_name += 'f-' + cfg_coarse.model.au_loss.feat_loss[:3]
            if cfg_coarse.model.au_loss.au_loss != 'l1_loss':
                experiment_name += '_c-' + cfg_coarse.model.au_loss.au_loss[:3]
        
        if 'lipread_loss' in cfg_coarse.model.keys():
            experiment_name += '_LR'
            
        if cfg_coarse.model.get('use_mediapipe_landmarks', False) or \
            cfg_coarse.model.get('use_mouth_corner_distance_mediapipe', False)\
                or cfg_coarse.model.get('use_lip_distance_mediapipe', False)\
                    or cfg_coarse.model.get('use_eye_distance_mediapipe', False):
            experiment_name += '_MP'

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

        if not cfg_detail.model.use_landmarks and cfg_detail.model.train_coarse:
            experiment_name += "NoLmk"

        if cfg_coarse.learning.train_K > 1:
            if version <= 1:
                if cfg_coarse.model.shape_constrain_type != 'exchange':
                    experiment_name += f'_Co{cfg_coarse.model.shape_constrain_type}'
                if cfg_detail.model.detail_constrain_type != 'exchange':
                    experiment_name += f'_De{cfg_detail.model.detail_constrain_type}'
            else:
                if cfg_coarse.model.shape_constrain_type != 'none':
                    experiment_name += f'_Co{cfg_coarse.model.shape_constrain_type[:2]}'
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


def train_expdeca(cfg_coarse, cfg_detail, start_i=-1, resume_from_previous = True,
               force_new_location=False):
    configs = [cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    stages = ["train", "test", "train", "test"]
    stages_prefixes = ["", "", "", ""]

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
    else:
        checkpoint, checkpoint_kwargs = None, None

    if cfg_coarse.inout.full_run_dir == 'todo' or force_new_location:
        if force_new_location:
            print("The run will be resumed in a new foler (forked)")
            cfg_coarse.inout.previous_run_dir = cfg_coarse.inout.full_run_dir
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        random_id = str(hash(time))
        experiment_name = create_experiment_name(cfg_coarse, cfg_detail)
        full_run_dir = Path(configs[0].inout.output_dir) / (time + "_" + random_id + "_" + experiment_name)
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
    # with open("out_folder.txt", "w") as f:
        # f.write(str(full_run_dir))

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

    deca = None
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


def configure_detail(detail_cfg_default, detail_overrides):
    from hydra.experimental import compose, initialize
    initialize(config_path="../emoca_conf", job_name="train_deca")
    cfg_detail = compose(config_name=detail_cfg_default, overrides=detail_overrides)
    return cfg_detail


def configure_and_train(coarse_cfg_default, coarse_overrides,
                        detail_cfg_default, detail_overrides):
    cfg_coarse, cfg_detail = configure(coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)
    train_expdeca(cfg_coarse, cfg_detail)


def configure_and_resume(run_path,
                         coarse_cfg_default, coarse_overrides,
                         detail_cfg_default, detail_overrides,
                         start_at_stage):
    cfg_coarse, cfg_detail = configure(
                                       coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)

    cfg_coarse_, cfg_detail_ = load_configs(run_path)

    if start_at_stage < 2:
        raise RuntimeError("Resuming before stage 2 makes no sense, that would be training from scratch")
    elif start_at_stage == 2:
        cfg_coarse = cfg_coarse_
    elif start_at_stage == 3:
        raise RuntimeError("Resuming for stage 3 makes no sense, that is a testing stage")
    else:
        raise RuntimeError(f"Cannot resume at stage {start_at_stage}")

    train_expdeca(cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=True, #important, resume from previous stage's checkpoint
               force_new_location=True)


def load_configs(run_path):
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    cfg_coarse = conf.coarse
    cfg_detail = conf.detail
    return cfg_coarse, cfg_detail


def resume_training(run_path, start_at_stage, resume_from_previous, force_new_location):
    cfg_coarse, cfg_detail = load_configs(run_path)
    train_expdeca(cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=resume_from_previous,
               force_new_location=force_new_location)


def main():
    configured = False
    num_workers = 0

    if len(sys.argv) <= 2:
        coarse_conf = "deca_train_coarse"
        detail_conf = "deca_train_detail"

        emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_05-15-38_-8198495972451127810_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"
        photometric_normalization = 'mean'
        use_mouth_corner_distance = True 
        use_eye_distance = True
        use_lip_distance = True
        emo_feature_loss_type = 'mse'

        coarse_override = [
            # 'model/settings=coarse_train',
            # 'model/settings=coarse_train_emonet',
            # 'model/settings=coarse_train_expdeca',
            # 'model/settings=coarse_train_expdeca_emonet',
            # 'model/settings=coarse_train_emica',
            'model/settings=coarse_train_emica_emonet',
            # 'model/settings=coarse_train_expdeca_emomlp',
            # 'model.expression_constrain_type=exchange',
            # 'model.expression_constrain_use_jaw_pose=True',
            'model.expression_constrain_use_global_pose=False',
            # 'model.use_geometric_losses_expression_exchange=True',

            'model.use_emonet_feat_1=False',
            'model.use_emonet_feat_2=True',
            'model.use_emonet_valence=False',
            'model.use_emonet_arousal=False',
            'model.use_emonet_expression=False',
            'model.use_emonet_combined=False',
            f'+model.photometric_normalization={photometric_normalization}',
            f'+model.use_mouth_corner_distance={use_mouth_corner_distance}',
            f'+model.use_eye_distance={use_eye_distance}',
            f'+model.use_lip_distance={use_lip_distance}',
            f'+model.emo_feat_loss={emo_feature_loss_type}',  # emonet feature loss
            # '+model.mlp_emotion_predictor.detach_shape=True',
            # '+model.mlp_emotion_predictor.detach_expression=True',
            # '+model.mlp_emotion_predictor.detach_detailcode=True',
            # '+model.mlp_emotion_predictor.detach_jaw=True',
            # '+model.mlp_emotion_predictor.detach_global_pose=True',
            f'+model.emonet_model_path={emonet}',
            'data/datasets=affectnet_desktop', # affectnet vs deca dataset
            # f'data.ring_type=gt_va',
            #  'data.ring_size=4',
            #  'learning/batching=single_gpu_expdeca_coarse_ring',
            f'data.num_workers={num_workers}',
            'model.resume_training=True', # load the original EMOCA model
            'learning.early_stopping.patience=5',
            'learning/logging=none',
            'learning.batch_size_train=4',
                              ]
        detail_override = [
            # 'model/settings=detail_train',
            # 'model/settings=detail_train_emonet',
            # 'model/settings=detail_train_expdeca_emonet',
            'model/settings=detail_train_emica_emonet',
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
            f'+model.emonet_model_path={emonet}',
            'data/datasets=affectnet_desktop', # affectnet vs deca dataset
            
            # f'data.ring_type=gt_va',
            #  'learning/batching=single_gpu_expdeca_detail_ring',
            #  'data.ring_size=4',
            'learning.early_stopping.patience=5',
            'learning/logging=none',
            f'data.num_workers={num_workers}',
            'learning.batch_size_train=4',

            'model.use_emonet_feat_1=False',
            'model.use_emonet_feat_2=True',
            'model.use_emonet_valence=False',
            'model.use_emonet_arousal=False',
            'model.use_emonet_expression=False',
            'model.use_emonet_combined=False',
            f'+model.photometric_normalization={photometric_normalization}',
            f'+model.use_mouth_corner_distance={use_mouth_corner_distance}',
            f'+model.use_eye_distance={use_eye_distance}',
            f'+model.use_lip_distance={use_lip_distance}',
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
            start_from = int(sys.argv[3])
            if len(sys.argv) > 4:
                resume_from_previous = bool(int(sys.argv[4]))
                if len(sys.argv) > 5:
                    force_new_location = bool(int(sys.argv[5]))
                else:
                    force_new_location = True
            else:
                resume_from_previous = True
        else:
            resume_from_previous = True
            force_new_location = False
            start_from = -1
    else:
        coarse_conf = "deca_finetune_coarse_cluster"
        detail_conf = "deca_finetune_detail_cluster"
        coarse_override = []
        detail_override = []

    # if len(sys.argv) > 4:
    #     coarse_override = sys.argv[3]
    #     detail_override = sys.argv[4]
    # else:
    #     coarse_override = []
    #     detail_override = []
    if configured:
        train_expdeca(coarse_conf, detail_conf,
                      start_from, resume_from_previous, force_new_location)
    else:
        configure_and_train(coarse_conf, coarse_override, detail_conf, detail_override)


if __name__ == "__main__":
    main()

