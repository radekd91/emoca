from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_emodeca
import datetime
from omegaconf import OmegaConf
import time as t

def submit(cfg, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    # submission_dir_local_mount = "/ps/scratch/rdanecek/emoca/submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/submission"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
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
    mem_gb = 16
    # mem_gb = 30
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
                       max_concurrent_jobs=30,
                       max_time_h=max_time_h,
                       max_price=max_price,
                       job_name=job_name,
                       cuda_capability_requirement=cuda_capability_requirement,
                       chmod=False
                       )
    t.sleep(2)


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
        # [
        #     ['data.sampler=balanced_expr'],
        #     []
        # ],
        [
            ['model.use_detail_code=true',
             'data.sampler=balanced_expr'],
            []
        ],
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
        # 'model.use_jaw_pose=False',
        # 'data/augmentations=default',
        # 'learning/optimizer=adabound',
        'data/datasets=affectnet_cluster',
        'data.data_class=AffectNetDataModuleValTest',
        # 'data/datasets=affectnet_v1_cluster',
        # 'data/datasets=emotionet_0_cluster',
        # 'data/datasets=emotionet_cluster',
        # 'learning/training=emotionet',
    ]

    deca_conf_path = None
    deca_conf = "deca_train_detail_cluster"
    stage = None
    fixed_overrides_deca = [
        # 'model/settings=coarse_train',
        'model/settings=detail_train',
        'model.resume_training=True',  # load the original DECA model
        'model.useSeg=rend', 'model.idw=0',
        'learning/batching=single_gpu_coarse',
        # 'learning/batching=single_gpu_detail',
        #  'model.shape_constrain_type=None',
         'model.detail_constrain_type=None',
        # 'data/datasets=affectnet_cluster',
        'data/datasets=emotionet_cluster',
        'learning.batch_size_test=1',
        # 'data/augmentations=default',
        # 'data/datasets=emotionet_cluster',
    ]

    # # # EMOEXPDECA
    # deca_conf_path = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca/2021_04_19_18-59-19_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"
    # deca_conf = None
    # fixed_overrides_deca = None
    # stage = 'detail'

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

