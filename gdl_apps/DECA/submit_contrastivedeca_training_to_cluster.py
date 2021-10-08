from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_stardeca
import datetime
from omegaconf import OmegaConf
import time as t


def submit(cfg_coarse, cfg_detail, pretrained_run_location=None, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/expdeca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/expdeca/submission"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(train_stardeca.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    coarse_file = submission_folder_local / "submission_coarse_config.yaml"
    detail_file = submission_folder_local / "submission_detail_config.yaml"

    with open(coarse_file, 'w') as outfile:
        OmegaConf.save(config=cfg_coarse, f=outfile)
    with open(detail_file, 'w') as outfile:
        OmegaConf.save(config=cfg_detail, f=outfile)


    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg_coarse.learning.gpu_memory_min_gb * 1024
    # gpu_mem_requirement_mb = None
    cpus = cfg_coarse.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg_coarse.learning.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_deca"
    cuda_capability_requirement = 6
    mem_gb = 40
    args = f"{coarse_file.name} {detail_file.name}"
    if pretrained_run_location is not None:
        args += f" {pretrained_run_location}"

    execute_on_cluster(str(cluster_script_path),
                       args,
                       str(submission_folder_local),
                       str(submission_folder_cluster),
                       str(cluster_repo_path),
                       python_bin=python_bin,
                       username=username,
                       gpu_mem_requirement_mb=gpu_mem_requirement_mb,
                       cpus=cpus,
                       mem_gb=mem_gb,
                       gpus=gpus,
                       num_jobs=num_jobs,
                       bid=bid,
                       max_time_h=max_time_h,
                       max_price=max_price,
                       job_name=job_name,
                       cuda_capability_requirement=cuda_capability_requirement
                       )
    t.sleep(1)


def train_on_selected_sequences():
    from hydra.core.global_hydra import GlobalHydra



    # ring_type = "gt_expression"
    # ring_type = "gt_va"
    # ring_type = "emonet_feature"

    finetune_modes = [
        # # DEFAULT without jaw
        # [
        #     ['model.useSeg=gt', 'model.exp_deca_jaw_pose=False'],
        #     ['model.useSeg=rend']
        # ],

        #
        # # DEFAULT
        # [
        #     ['model.useSeg=gt'],
        #     ['model.useSeg=rend']
        # ],

        # DEFAULT, with VGG perceptual loss
        [
            ['model.useSeg=gt', '+model/additional=vgg_loss',],
            ['model.useSeg=rend', '+model/additional=vgg_loss',]
        ],
        #
        # # DEFAULT, with VGG perceptual loss, without photometric
        # [
        #     ['model.useSeg=gt', '+model/additional=vgg_loss', 'model.use_photometric=False'],
        #     ['model.useSeg=rend', '+model/additional=vgg_loss', 'model.use_photometric=False']
        # ],

        # # DEFAULT, rendered mask
        # [
        #     ['model.useSeg=rend'],
        #     ['model.useSeg=rend']
        # ],

        # # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING
        # [
        #     ['model.useSeg=gt', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', ]
        # ],

        # #EmonetStatic backbone, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None',  'learning.batch_size_test=1',
        #      'model.expression_backbone=emonet_static'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None',  'learning.batch_size_test=1',
        #      'model.expression_backbone=emonet_static']
        # ],

        # #Emonet trainable backbone, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None',  'learning.batch_size_test=1',
        #      'model.expression_backbone=emonet_trainable'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None',  'learning.batch_size_test=1',
        #      'model.expression_backbone=emonet_trainable']
        # ],
        #
        # # DECA cloned trainable backbone, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1',
        #      'model.expression_backbone=deca_clone'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'learning.batch_size_test=1',
        #      'model.expression_backbone=deca_clone']
        # ],

        # # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse_32gb',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail_32gb',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'learning.batch_size_test=1']
        # ],
        #
        # # # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK, DETAIL WITH COARSE
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse_32gb',
        #      'model.shape_constrain_type=None',  'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail_32gb',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'model.train_coarse=true',  'learning.batch_size_test=1']
        # ],

        # # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse_32gb',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1', '+model/additional=vgg_loss'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail_32gb',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'learning.batch_size_test=1', '+model/additional=vgg_loss']
        # ],
        #
        # # # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK, DETAIL WITH COARSE
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse_32gb',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1', '+model/additional=vgg_loss'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail_32gb',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'model.train_coarse=true', 'learning.batch_size_test=1', '+model/additional=vgg_loss']
        # ],

        # # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK, EXPRESSION RING EXCHANGE
        # [
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.shape_constrain_type=None',
        #      'model.expression_constrain_type=exchange',
        #      'model.expression_constrain_use_jaw_pose=True',
        #      'model.expression_constrain_use_global_pose=False',
        #      'model.use_geometric_losses_expression_exchange=True',
        #      'model.background_from_input=False',
        #      f'data.ring_type={ring_type}',
        #      # 'data.ring_type=gt_va',
        #      # 'data.ring_type=emonet_feature',
        #      'data.ring_size=4',
        #      'learning/batching=single_gpu_expdeca_coarse_ring',
        #      'learning.gpu_memory_min_gb=24',
        #      'learning.batch_size_test=1'
        #      ],
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.expression_constrain_type=exchange',
        #      'model.expression_constrain_use_jaw_pose=True',
        #      'model.expression_constrain_use_global_pose=False',
        #      'model.use_geometric_losses_expression_exchange=True',
        #      'model.background_from_input=False',
        #      f'data.ring_type={ring_type}',
        #      # 'data.ring_type=gt_va',
        #      # 'data.ring_type=emonet_feature',
        #      'data.ring_size=4',
        #      'learning/batching=single_gpu_expdeca_detail_ring',
        #      'learning.gpu_memory_min_gb=24',
        #      # 'model.shape_constrain_type=None',
        #      'learning.batch_size_test=1']
        # ],
        #
        #DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK, DETAIL WITH COARSE, EXPRESSION RING EXCHANGE
        # [
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.expression_constrain_type=exchange',
        #      'model.expression_constrain_use_jaw_pose=True',
        #      'model.expression_constrain_use_global_pose=False',
        #      'model.use_geometric_losses_expression_exchange=False',
        #      'model.background_from_input=False',
        #      f'data.ring_type={ring_type}',
        #      # 'data.ring_type=gt_va',
        #      # 'data.ring_type=emonet_feature',
        #      'data.ring_size=4',
        #      'learning/batching=single_gpu_expdeca_coarse_ring',
        #      'learning.gpu_memory_min_gb=24',
        #      'model.shape_constrain_type=None',  'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.expression_constrain_type=exchange',
        #      'model.expression_constrain_use_jaw_pose=True',
        #      'model.expression_constrain_use_global_pose=False',
        #      'model.use_geometric_losses_expression_exchange=False',
        #      'model.background_from_input=False',
        #      f'data.ring_type={ring_type}',
        #      # 'data.ring_type=gt_va',
        #      # 'data.ring_type=emonet_feature',
        #      'data.ring_size=4',
        #      'learning/batching=single_gpu_expdeca_detail_ring',
        #      'learning.gpu_memory_min_gb=24',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=none', 'model.train_coarse=true',  'learning.batch_size_test=1']
        # ],

        # # DEFAULT DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK,
        # # DETAIL WITH COARSE, no landmarks for detail stage
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None',
        #      'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None',
        #      'learning.batch_size_test=1', 'model.train_coarse=true',
        #      'model.use_landmarks=False']
        # ],


        # # # AffectNet with augmentation, DEFAULT DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING
        # [
        #     ['model.useSeg=gt', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None', 'data/datasets=affectnet_cluster',
        #      'learning.batch_size_test=1', 'data/augmentations=default'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'data/datasets=affectnet_cluster',
        #      'learning.batch_size_test=1', 'data/augmentations=default']
        # ],

        # # DEFAULT but train coarse with detail
        # [
        #     ['model.useSeg=rend'],
        #     ['model.useSeg=rend']
        # ],

        # # DEFAULT but train coarse with detail
        # [
        #     ['model.useSeg=gt'],
        #     ['model.useSeg=rend', 'model.train_coarse=true']
        # ],

        # DEFAULT but train coarse with detail
        # [
        #     ['model.useSeg=rend'],
        #     ['model.useSeg=rend', 'model.train_coarse=true']
        # ],

        # # DEFAULT but cloned ResNet backbone
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=deca_clone'],
        #     ['model.useSeg=rend', 'model.expression_backbone=deca_clone']
        # ],

        # # DEFAULT but cloned ResNet backbone, rendered mask
        # [
        #     ['model.useSeg=rend', 'model.expression_backbone=deca_clone'],
        #     ['model.useSeg=rend', 'model.expression_backbone=deca_clone']
        # ],


        # # DEFAULT but static EmoNet backbone
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=emonet_static'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_static']
        # ],

        # # DEFAULT but static EmoNet backbone, rendered mask
        # [
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_static'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_static']
        # ],

        # # DEFAULT but static EmoNet backbone, no jaw
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=emonet_static', 'model.exp_deca_jaw_pose=False'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_static']
        # ],

        # # DEFAULT but trainable EmoNet backbone
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=emonet_trainable'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_trainable']
        # ]

        # DEFAULT but trainable EmoNet backbone, rendered mask
        # [
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_trainable'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_trainable']
        # ]

        # # REGULAR DECA, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.max_epochs=8',
        #      'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.max_epochs=8',
        #      'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'learning.batch_size_test=1']
        # ],
        #
        # # REGULAR DECA, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK, DETAIL WITH COARSE
        # [
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.max_epochs=8',
        #      'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.max_epochs=8',
        #      'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'model.train_coarse=true', 'learning.batch_size_test=1']
        # ],

    ]


    # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_20_09-43-26_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000'
    # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_23_22-52-24_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early'
    emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_30_11-12-32_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early'
    # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_23-50-06_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early'
    # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_13-06-58_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early'
    # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_13-06-04_EmoSwin_swin_tiny_patch4_window7_224_shake_samp-balanced_expr_Aug_early'


    # resume_from = None # resume from Original DECA
    resume_from = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_21-50-45_DECA__DeSegFalse_early/" # My DECA, ResNet backbones
    # resume_from = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_23-19-03_DECA__EFswin_s_EDswin_s_DeSegFalse_early/" # My DECA, SWIN small
    # resume_from = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_23-19-04_DECA__EFswin_t_EDswin_t_DeSegFalse_early/" # My DECA, SWIN tiny

    flame_encoder = 'ResnetEncoder'
    detail_encoder = 'ResnetEncoder'
    # flame_encoder = 'swin_small_patch4_window7_224'
    # detail_encoder = 'swin_small_patch4_window7_224'
    # flame_encoder = 'swin_tiny_patch4_window7_224'
    # detail_encoder = 'swin_tiny_patch4_window7_224'

    dataset_coarse =  "data/datasets=coarse_data_cluster"
    dataset_detail =  "data/datasets=detail_data_cluster"
    # dataset_coarse = "data/datasets=affectnet_cluster"
    # dataset_detail = "data/datasets=affectnet_cluster"

    augmentation = "data/augmentations=default"
    # augmentation = "data/augmentations=none"

    # coarse_conf = "deca_train_coarse_stargan_cluster" # uses neural rendering
    # detail_conf = "deca_train_detail_stargan_cluster" # uses neural rendering

    coarse_conf = "deca_train_coarse_cluster"
    detail_conf = "deca_train_detail_cluster"


    # sampler="data.sampler=uniform"
    sampler="data.sampler=balanced_expr"

    use_emo_loss = 'False'
    # use_emo_loss = 'True'
    emo_feature_loss_type = 'cosine_similarity'
    # emo_feature_loss_type = 'l1_loss'
    # emo_feature_loss_type = 'barlow_twins_headless'
    # emo_feature_loss_type = 'barlow_twins'

    # id_feature_loss_type = 'cosine_similarity'
    # id_feature_loss_type = 'l1_loss'
    # id_feature_loss_type = 'barlow_twins_headless'
    id_feature_loss_type = 'barlow_twins'

    fixed_overrides_coarse = [
        'model/settings=coarse_train',
        # 'model/settings=coarse_train_emonet',
        # 'model/settings=coarse_train_expdeca',
        # 'model/settings=coarse_train_expdeca_emonet',
        # 'model/settings=coarse_train_expdeca_emomlp',
        # '+model.mlp_emotion_predictor.detach_shape=True',
        # '+model.mlp_emotion_predictor.detach_expression=False',
        # '+model.mlp_emotion_predictor.detach_detailcode=False',
        # '+model.mlp_emotion_predictor.detach_jaw=True',
        # '+model.mlp_emotion_predictor.detach_global_pose=False',
        # 'data/datasets=affectnet_cluster', # affectnet vs deca dataset
        f'model.resume_training={resume_from == None}', # load the original DECA model
        'learning.early_stopping.patience=5',
        'learning.train_K=1',
        'learning.batch_size_train=16',
        # 'model.useSeg=False',
        'model.background_from_input=input',
        f'+model.emonet_model_path={emonet}',
        # '+model.emoloss_trainable=true',
        '+model.emoloss_dual=true',
        f'+model.e_flame_type={flame_encoder}',
        f'+model.e_detail_type={detail_encoder}',
        dataset_coarse,
        augmentation,
        # sampler,
        # '+model.normalize_features=true',  # normalize emonet features before applying loss
        # '+model.emo_feat_loss=l1_loss', # emonet feature loss
        # '+model.emo_feat_loss=cosine_similarity',  # emonet feature loss
        f'+model.emo_feat_loss={emo_feature_loss_type}',  # emonet feature loss
        # '+model.emo_feat_loss=barlow_twins',  # emonet feature loss
        f'model.use_emonet_loss={use_emo_loss}',
        'model.use_emonet_feat_1=False',
        'model.use_emonet_feat_2=True',
        'model.use_emonet_valence=False',
        'model.use_emonet_arousal=False',
        'model.use_emonet_expression=False',
        'model.use_emonet_combined=False',
        f'+model.id_metric={id_feature_loss_type}',
        # '+model.id_metric=barlow_twins',
        # '+model/additional=au_loss_dual', # emonet feature loss
        # 'model.au_loss.au_loss=cosine_similarity', # emonet feature loss
        # 'model.au_loss.feat_loss=cosine_similarity',
        # 'model.au_loss.normalize_features=True', # emonet feature loss
    ]

    fixed_overrides_detail = [
        'model/settings=detail_train',
        # 'model/settings=detail_train_emonet',
        # 'model/settings=detail_train_expdeca',
        # 'model/settings=detail_train_expdeca_emonet',
        # 'model/settings=detail_train_expdeca_emomlp',
        # '+model.mlp_emotion_predictor.detach_shape=True',
        # '+model.mlp_emotion_predictor.detach_expression=False',
        # '+model.mlp_emotion_predictor.detach_detailcode=False',
        # '+model.mlp_emotion_predictor.detach_jaw=True',
        # '+model.mlp_emotion_predictor.detach_global_pose=False',
        # 'data/datasets=affectnet_cluster', # affectnet vs deca dataset
        'learning.early_stopping.patience=5',
        'learning.train_K=1',
        'learning.batch_size_train=8',
        # 'model.useSeg=False',
        'model.background_from_input=input',
        f'+model.emonet_model_path={emonet}',
        # '+model.emoloss_trainable=true',
        # '+model.emoloss_dual=true',
        # f'+model.e_flame_type={flame_encoder}',
        # f'+model.e_detail_type={detail_encoder}',
        dataset_detail,
        augmentation,
        # sampler,
        # '+model.normalize_features=true',  # normalize emonet features before applying loss
        # '+model.emo_feat_loss=l1_loss', # emonet feature loss
        f'+model.emo_feat_loss={emo_feature_loss_type}', # emonet feature loss
        # '+model.emo_feat_loss=barlow_twins', # emonet feature loss
        f'model.use_emonet_loss={use_emo_loss}',
        'model.use_emonet_feat_1=False',
        'model.use_emonet_feat_2=True',
        'model.use_emonet_valence=False',
        'model.use_emonet_arousal=False',
        'model.use_emonet_expression=False',
        'model.use_emonet_combined=False',
        # '+model.id_metric={id_feature_loss_type}',
        # '+model.emo_feat_loss=cosine_similarity',  # emonet feature loss
        # '+model/additional=au_loss_dual',  # emonet feature loss
        # 'model.au_loss.au_loss=cosine_similarity',  # emonet feature loss
        # 'model.au_loss.feat_loss=cosine_similarity',
        # 'model.au_loss.normalize_features=True', # emonet feature loss
    ]

    # emonet_weights = [5.0, 1.0, 0.5, 0.5/5, 0.5/10, 0.5/50, 0.5/100]
    emonet_weights = [0.0015]
    # emomlp_weights = [0.5, 0.1, 0.05, 0.005]
    # emomlp_weights = [1.0] # with detached jaw pose
    # emomlp_weights = [10.0, 100.0, 1000.0] # stress test
    # emomlp_weights = [0.05] # this one seems to be close to the sweet spot

    config_pairs = []
    for emonet_weight in emonet_weights:
        for fmode in finetune_modes:
            coarse_overrides = fixed_overrides_coarse.copy()
            detail_overrides = fixed_overrides_detail.copy()
            # if len(fmode[0]) != "":
            coarse_overrides += fmode[0]
            detail_overrides += fmode[1]

            # data_override = f'data.sequence_index={video_index}'
            # pretrain_coarse_overrides += [data_override]
            # coarse_overrides += [data_override]
            # detail_overrides += [data_override]

            # au_weight_override = f'model.emonet_weight={emonet_weight}'
            # coarse_overrides += [au_weight_override]
            # detail_overrides += [au_weight_override]

            # au_weight_override = f'model.au_loss.au_weight={emonet_weight}'
            # coarse_overrides += [au_weight_override]
            # detail_overrides += [au_weight_override]

            cfgs = train_stardeca.configure(
                coarse_conf, coarse_overrides,
                detail_conf, detail_overrides
            )

            GlobalHydra.instance().clear()
            config_pairs += [cfgs]

            submit(cfgs[0], cfgs[1], resume_from)
                # break
            # break

    # for cfg_pair in config_pairs:
    #     submit(cfg_pair[0], cfg_pair[1])


def default_main():
    coarse_conf = "deca_train_coarse_cluster"
    coarse_overrides = []

    detail_conf = "deca_train_detail_cluster"
    detail_overrides = []

    cfg_coarse, cfg_detail = train_stardeca.configure(
        coarse_conf, coarse_overrides,
        detail_conf, detail_overrides)

    submit(cfg_coarse, cfg_detail)


if __name__ == "__main__":
    # default_main()
    train_on_selected_sequences()

