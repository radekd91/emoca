from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_emodeca
import datetime
from omegaconf import OmegaConf
import time as t
import random
import pandas as pd
import wandb



def submit(cfg, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    # submission_dir_local_mount = "/ps/scratch/rdanecek/emoca/submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/submission"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(random.randint(0, 100000))) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(train_emodeca.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    config_file = submission_folder_local / "config.yaml"

    with open(config_file, 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg.learning.gpu_memory_min_gb * 1024
    # gpu_mem_requirement_mb = None
    cpus = cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.learning.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_deca"
    cuda_capability_requirement = 7
    # mem_gb = 16
    mem_gb = 30
    args = f"{config_file.name}"

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
                       cuda_capability_requirement=cuda_capability_requirement,
                       chmod=False,
                       max_concurrent_jobs=30,
                       concurrency_tag="emodeca_train",
                       modules_to_load=['cuda/11.4'],
                       )
    # t.sleep(2)


def train_emodeca_on_cluster():
    from hydra.core.global_hydra import GlobalHydra


    training_modes = [
        # # DEFAULT
        # [
        #     ['model.num_mlp_layers=1'],
        #     []
        # ],
        # [
        #     ['model.num_mlp_layers=2'],
        #     []
        # ],
        # [
        #     ['model.num_mlp_layers=3'],
        #     []
        # ],
        # [
        #     ['model.use_detail_code=true',],
        #     []
        # ],
        # [
        #     [],
        #     []
        # ],
        [
            ['data.sampler=balanced_expr'],
            []
        ],
        # [
        #     ['model.use_detail_code=true',
        #      'data.sampler=balanced_expr'],
        #     []
        # ],
        # [
        #     ['data.sampler=balanced_va'],
        #     []
        # ],
        # [
        #     ['data.sampler=balanced_expr',
        #      'model.use_detail_code=true',
        #      ],
        #     []
        # ],
        # [
        #     ['data.sampler=balanced_va',
        #      'model.use_detail_code=true',],
        #     []
        # ],
        # [
        #     ['model/settings=coarse_emodeca_emonet'],
        #     []
        # ],
        #
        # [
        #     ['model/settings=coarse_emodeca_emonet',
        #      'model.use_coarse_image_emonet=true',
        #      'model.use_detail_image_emonet=false',
        #      ],
        #     []
        # ],
        #
        # [
        #     [
        #         'model/settings=coarse_emodeca_emonet',
        #         'model.unpose_global_emonet=false',
        #         'model.use_coarse_image_emonet=false',
        #         'model.use_detail_image_emonet=true',
        #         'model.static_cam_emonet=false',
        #         'model.static_light=false',
        #     ],
        #     []
        # ],
        #
        # [
        #     ['model.use_detail_code=true'],
        #     []
        # ],
        # [
        #     [
        #         'model.use_detail_code=true',
        #         '+model.mlp_norm_layer=BatchNorm1d',
        #      ],
        #     []
        # ],

        # [
        #     ['model.expression_balancing=true'],
        #     []
        # ],

        # [
        #     ['model.use_detail_code=true',
        #      'model.expression_balancing=true'],
        #     []
        # ],
        #
        # [
        #     [
        #         'model.use_detail_code=true',
        #         '+model.mlp_norm_layer=BatchNorm1d',
        #         'model.expression_balancing=true',
        #         'model.mlp_dimension_factor=4',
        #      ],
        #     []
        # ],

        # [
        #     [   'model.expression_balancing=true',
        #         'model.continuous_va_balancing=1d',
        #      ],
        #     []
        # ],
        #
        # [
        #     ['model.expression_balancing=true',
        #      'model.continuous_va_balancing=2d',
        #      ],
        #     []
        # ],

        # [
        #     ['model.expression_balancing=true',
        #      'model.continuous_va_balancing=expr',
        #      ],
        #     []
        # ],

    ]

    # #1 EMONET
    # conf = "emonet_cluster"
    # fixed_overrides_cfg = [
    #     'model/settings=emonet_trainable',
    #     # 'model/settings=emonet_trainable_weighted_va',
    #     # 'model/settings=emonet_trainable_weighted_va_mse',
    #     # '+learning/lr_scheduler=reduce_on_plateau',
    #     '+learning/lr_scheduler=exponential',
    #     # 'learning.max_steps=0',
    #     # 'learning.max_epochs=0',
    #     # 'learning/optimizer=adabound',
    #     'data/augmentations=default',
    # ]
    # deca_conf = None
    # deca_conf_path = None
    # fixed_overrides_deca = None
    # stage = None

    # # # #2 EMOSWIN
    # conf = "emoswin"
    # fixed_overrides_cfg = [
    #     'model/backbone=swin',
    #     # 'model/backbone=resnet50_cluster',
    #     # 'model/backbone=vgg13',
    #     # 'model/backbone=vgg16',
    #     # 'model/backbone=vgg16_bn',
    #     # 'model/backbone=vgg19_bn',
    #     # 'model/settings=AU_emotionet',
    #     'model/settings=AU_emotionet_bce',
    #     # 'model/settings=AU_emotionet_bce_weighted',
    #     # '+learning/lr_scheduler=reduce_on_plateau',
    #     # '+learning/lr_scheduler=exponential',
    #     # 'learning.batch_size_train=32',
    #     # swin_type: swin_base_patch4_window7_224
    #     # swin_type: swin_small_patch4_window7_224
    #     # swin_type: swin_tiny_patch4_window7_224
    #     'learning.batch_size_train=16',
    #     # 'model.swin_type=swin_large_patch4_window7_224_22k',
    #     # 'model.swin_type=swin_base_patch4_window7_224',
    #     'model.swin_type=swin_small_patch4_window7_224',
    #     # 'model.swin_type=swin_tiny_patch4_window7_224',
    #     # 'data/datasets=affectnet_cluster',
    #     # 'data/datasets=affectnet_v1_cluster',
    #     # 'data/datasets=emotionet_0_cluster',
    #     'data/datasets=emotionet_cluster',
    #     # 'learning.max_steps=0',
    #     # 'learning.max_epochs=0',
    #     'learning/training=emotionet',
    #     # 'learning/optimizer=adabound',
    #     # 'data/augmentations=default',
    #     'data/augmentations=default_with_resize',
    # ]
    # deca_conf = None
    # deca_conf_path = None
    # fixed_overrides_deca = None
    # stage = None

    # EMODECA
    conf = "emodeca_coarse_cluster"
    fixed_overrides_cfg = [
        # 'model/settings=AU_emotionet',
        # 'model/settings=AU_emotionet_bce',
        # 'model/settings=AU_emotionet_bce_weighted',
        # '+model.mlp_norm_layer=BatchNorm1d',
        # 'model.use_identity=True', #
        # 'data/augmentations=default',
        # 'learning/optimizer=adabound',
        # 'data/datasets=affectnet_cluster',
        # 'data.data_class=AffectNetDataModuleValTest',
        'data/datasets=affectnet_cluster_emonet_cleaned',
        'data.num_workers=16',
        # 'data/datasets=affectnet_v1_cluster',
        # 'data/datasets=emotionet_0_cluster',
        # 'data/datasets=emotionet_cluster',
        # 'learning/training=emotionet',
    ]

    # deca_conf_path = None
    # deca_conf = "deca_train_detail_cluster"
    # stage = None
    # fixed_overrides_deca = [
    #     # 'model/settings=coarse_train',
    #     'model/settings=detail_train',
    #     'model.resume_training=True',  # load the original DECA model
    #     'model.useSeg=rend', 'model.idw=0',
    #     'learning/batching=single_gpu_coarse',
    #     # 'learning/batching=single_gpu_detail',
    #     #  'model.shape_constrain_type=None',
    #      'model.detail_constrain_type=None',
    #     # 'data/datasets=affectnet_cluster',
    #     'data/datasets=emotionet_cluster',
    #     'learning.batch_size_test=1',
    #     # 'data/augmentations=default',
    #     # 'data/datasets=emotionet_cluster',
    # ]

    # # # EMOEXPDECA
    # deca_conf_path = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca/2021_04_19_18-59-19_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"
    run_names = []

    # # ExpDECA with stronger emonet loss withouth jaw
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-59-10_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-58-28_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # #run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-58-11_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-55-47_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-55-25_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # #run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-55-14_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-55-12_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-54-31_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-53-48_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-53-44_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_10-50-22_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # #run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_15-02-19_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-59-51_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-59-38_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # #run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-59-25_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-57-14_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-57-09_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-52-01_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-52-00_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-09-55_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-09-45_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-09-16_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_27_14-09-02_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # #run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_26_11-58-32_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_26_11-57-47_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # #run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_26_11-56-07_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_25_18-54-54_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_25_18-54-45_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_25_18-53-53_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_25_18-30-31_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_25_18-30-14_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_25_10-18-37_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_25_10-18-18_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_25_10-18-05_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_22_12-23-05_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_22_12-23-04_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_22_12-23-03_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_22_10-38-21_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    #
    # ## Barlow Twins
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_22-41-34_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_22-39-54_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_22-39-46_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_22-39-42_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_11-22-58_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_11-19-45_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_11-19-20_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_11-19-17_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_28_11-19-10_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # # other emotion networks
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_29_22-23-28_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_29_22-23-25_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # #
    # # # para architecture
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_29_21-30-28_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_29_21-34-01_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_29_22-01-32_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_29_22-01-40_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_30_20-50-53_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_30_20-50-54_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_30_20-48-16_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_30_20-47-54_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_30_20-47-47_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_30_20-33-39_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # final ResNet sweep
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-41-24_7057622275122671174_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-41-06_-6506673705064889607_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-41-06_-1007531484471246016_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-41-03_480128111237298530_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-41-03_-3847743713390055217_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-41-01_7226661150207254923_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-41-01_7193545667483921831_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-41-01_-7746686909198123775_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-59_-6970716391423648964_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-59_-4293993865315558856_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-54_-8956728687580574108_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-52_3535801695749609832_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-52_-776769225150723181_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-48_-3557093149321491446_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-48_-1130509047528431540_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-45_6758302146806216456_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-33_8529519700345615983_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-33_7640188424869169886_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-27_4344460465829536839_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-27_6877124675180108840_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-03_-8436446076366773310_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-03_-4636033309105620245_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-02_833576158064688874_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-40-02_8285199837830669798_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # final SWIN sweep
    ## run_names += [
    ##     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_03_02-20-54_-1968661455773213379_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    ## run_names += [
    ##     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_03_02-42-36_2407313258403191383_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_13-12-49_-2741982989276466203_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_13-10-49_1731595146375171932_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_13-10-49_-7667543226652993592_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_13-10-40_6872394600987091012_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_13-10-40_-8572124953572605249_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_13-10-40_-7473769445402844399_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_13-10-40_-5241005287738579855_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-54-01_2637759665938415282_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-47-35_5168561227047398084_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-47-35_2319744141436125537_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-47-35_2073066276032009236_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    ## run_names += [
    ##     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-47-35_-6306367650010438382_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-47-35_-3218674826605317504_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-47-35_-1846739961689335557_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-47-35_-1022988189955888024_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-42-55_5806971874713117653_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-42-55_-5348873875193364241_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-42-54_-7598613731487617091_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-42-45_5490409369290264125_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-42-45_1092553962037855966_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-42-43_2271671740894586800_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-42-43_-5593491350755409121_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-42-41_4316282956709408142_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_02_12-42-41_-1213070571142271333_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]


    ## ONE Sanity check - reproducing ResNet clode best candidate, run the rest if necessary
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_06_22-18-51_-8514519696387528299_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]


    # EMONET SPLIT RUN:
    tags = None
    api = wandb.Api()

    # emonet ablation
    # # tags = ["EMONET_SPLIT_ABLATION_EMONET"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-15-52_8154275745776863855_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-15-52_349713347846449814_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-15-52_8341774161001263236_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-15-52_-1548615666948242852_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-15-52_2916708914926921364_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-16-26_2689968017949274893_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-16-26_2689968017949274893_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    #
    # # tags = ["EMONET_SPLIT_ABLATION_EMONET_JAW"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-21-17_3117709423447065408_ExpDECA_Affec_clone_Jaw_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-21-17_-3658324653371799778_ExpDECA_Affec_clone_Jaw_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    #
    # # tags = ['EMONET_SPLIT_ABLATION_PHOTOMETRIC_REL']
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-05-06_244631536517617441_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-05-01_5101174495546322475_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-04-15_9157589239172551865_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-04-08_-6517858133142386828_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # tags = ['EMONET_SPLIT_ABLATION_LANDMARKS']
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-00-11_2212703344027741137_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-00-11_1092543351772726789_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-00-11_1056504990304470492_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-00-11_-3505404531826926943_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    #
    # # tags = ['EMONET_SPLIT_ABLATION_EMO_METRIC']
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-57-41_6160996897661237206_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-57-41_1218762018464274311_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-57-41_-5511487677556972267_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # # tags = ['EMONET_SPLIT_ABLATION_NO_EMO_CLONE']
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-49-32_-5959946206105776497_ExpDECA_Affec_clone_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-49-11_-7854117761220635898_ExpDECA_Affec_clone_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-48-54_3114387149519252327_ExpDECA_Affec_clone_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-48-36_-2088077727545369691_ExpDECA_Affec_clone_NoRing_DeSegrend_BlackB_Aug_early"]
    # # tags = ['EMONET_SPLIT_ABLATION_NO_EMO']
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-47-40_-6121237435910246400_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-47-38_-7658985706608461505_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-46-17_-3537904820935564917_ExpDECA_Affec_para_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-45-58_6136426326225856038_ExpDECA_Affec_para_NoRing_DeSegrend_BlackB_Aug_early"]
    ## RESNET WEIGHT ABLATION
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-35_4654975036132116438_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-35_-2012595522172194483_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-08_4772041050212257497_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-07_-3997268493304040250_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    ## SWIN WEIGHT ABLATION - to run
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-08_3278107752429068516_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # #run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-07_-3310835230647295291_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-07_-2154597728523907962_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-07_-2106219737797182304_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # DecaD ablation
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-56-46_5920957646486902084_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-56-39_-8971851772753744759_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-38_1354461056444555550_ExpDECA_DecaD_para_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-41_7798762876288315974_ExpDECA_DecaD_clone_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-32_-428770426719310834_ExpDECA_DecaD_para_NoRing_DeSegrend_BlackB_Aug_early"]
    #
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-28_6450489661335316335_ExpDECA_DecaD_clone_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-19_-698052302382081628_ExpDECA_DecaD_clone_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-17_-6566800429279817771_ExpDECA_DecaD_clone_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    #
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-08-55_-7847515130004126177_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-07-31_-2183917122794074619_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-22_-3360331398526735766_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-22_4582523459040385488_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-27_8115149509825457198_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-30_-5150018129787658113_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    #
    # # unbalanced ExpDECA on Affecntet
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-27_7449334996109808959_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-07_-753452132482044016_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-07_-6499863499965279138_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-27-09_3536700504397748218_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # # lr ablation
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-32-15_7264067905024760402_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-30-18_3842660621685827882_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-25-15_-7606645522376246067_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-25-05_5658338137145609621_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # stage = 'detail'
    stage = 'coarse'

    for deca_conf_path in  run_names:
        name = str(Path(deca_conf_path).name)
        idx = name.find("ExpDECA")
        run_id = name[:idx-1]
        run = api.run("rdanecek/EmotionalDeca/" + run_id)
        tags = run.tags
        tags += ["NEW_SPLIT"]
        fixed_overrides_cfg += [f"+learning.tags={ '['+', '.join(tags)+ ']'}"]

        deca_conf = None
        fixed_overrides_deca = None


        for mode in training_modes:
            conf_overrides = fixed_overrides_cfg.copy()
            conf_overrides += mode[0]
            if deca_conf_path is None and fixed_overrides_deca is not None:
                deca_overrides = fixed_overrides_deca.copy()
                deca_overrides += mode[1]
            else:
                deca_overrides=None

            cfg = train_emodeca.configure(
                conf, conf_overrides,
                deca_default=deca_conf, deca_overrides=deca_overrides,
                deca_conf_path=deca_conf_path ,
                deca_stage=stage
            )
            GlobalHydra.instance().clear()

            submit(cfg)


if __name__ == "__main__":
    train_emodeca_on_cluster()

