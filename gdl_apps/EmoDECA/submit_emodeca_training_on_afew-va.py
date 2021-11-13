from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_emodeca
import datetime
from omegaconf import OmegaConf
import time as t
import random
import pandas as pd
import wandb

project_name = "EmoDECA_Afew-VA"

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
    args = f"{config_file.name} {-1} {0} {1} {project_name}"

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
        #     ['model.use_detail_code=true',
        #     'model.predict_expression=false'],
        #     []
        # ],
        [
            ['model.predict_expression=false'],
            []
        ],
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
        'data/datasets=afew_va',
        'data.num_workers=16',
        'learning.max_epochs=100',
        'learning.val_check_interval=1.0',
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
    # run_names = []
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_12_19-56-13_704003715291275370_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-42-30_8680779076656978317_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_12-13-16_-8024089022190881636_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_01-59-07_-9007648997833454518_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_01-58-56_1043302978911105834_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-58-31_-7948033884851958030_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-58-27_-5553059236244394333_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-57-28_-4957717700349337532_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-34-49_8015192522733347822_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-33-02_-5975857231436227431_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-33-00_-1889770853677981780_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-32-49_-6879167987895418873_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]

    # run_names = []
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
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-07-31_-2183917122794074619_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-22_-3360331398526735766_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-22_4582523459040385488_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-27_8115149509825457198_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-30_-5150018129787658113_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    #
    # # # unbalanced ExpDECA on Affecntet
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-27_7449334996109808959_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-07_-753452132482044016_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-07_-6499863499965279138_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-27-09_3536700504397748218_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]


    # EMONET SPLIT RUN:
    tags = None
    api = wandb.Api()

    # stage = 'detail'
    stage = 'coarse'

    for deca_conf_path in  run_names:
        name = str(Path(deca_conf_path).name)
        idx = name.find("ExpDECA")
        run_id = name[:idx-1]
        run = api.run("rdanecek/EmotionalDeca/" + run_id)
        tags = set(run.tags)

        allowed_tags = set(["COMPARISON", "INTERESTING", "FINAL_CANDIDATE", "BEST_CANDIDATE", "BEST_IMAGE_BASED"])

        # if len(allowed_tags.intersection(tags)) == 0:
        #     print(f"Run '{name}' is not tagged to be tested and will be skipped.")
        #     continue
        deca_conf = None
        fixed_overrides_deca = None


        for mode in training_modes:
            conf_overrides = fixed_overrides_cfg.copy()

            conf_overrides += [f"+learning.tags={'[' + ', '.join(tags) + ']'}"]

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

            sub = True
            # sub = False
            if sub:
                submit(cfg)
            else:
                train_emodeca.train_emodeca(cfg, -1, True, False, project_name)


if __name__ == "__main__":
    train_emodeca_on_cluster()

