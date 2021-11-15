from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_emodeca
import datetime
from omegaconf import OmegaConf
import time as t
import random
import pandas as pd
import wandb
import yaml
project_name = "EmoDECA_Afew-VA"
from omegaconf import open_dict, DictConfig

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
    # gpu_mem_requirement_mb = cfg.learning.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb = 20 * 1024
    # gpu_mem_requirement_mb_max = 40 * 1024
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
    args = f"{config_file.name} {0} {1} {1} {project_name}"

    execute_on_cluster(str(cluster_script_path),
                       args,
                       str(submission_folder_local),
                       str(submission_folder_cluster),
                       str(cluster_repo_path),
                       python_bin=python_bin,
                       username=username,
                       gpu_mem_requirement_mb=gpu_mem_requirement_mb,
                       # gpu_mem_requirement_mb_max=gpu_mem_requirement_mb_max,
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
    # from hydra.core.global_hydra import GlobalHydra
    #
    #
    # # #1 EMONET
    # # conf = "emonet_cluster"
    # # fixed_overrides_cfg = [
    # #     'model/settings=emonet_trainable',
    # #     # 'model/settings=emonet_trainable_weighted_va',
    # #     # 'model/settings=emonet_trainable_weighted_va_mse',
    # #     # '+learning/lr_scheduler=reduce_on_plateau',
    # #     '+learning/lr_scheduler=exponential',
    # #     # 'learning.max_steps=0',
    # #     # 'learning.max_epochs=0',
    # #     # 'learning/optimizer=adabound',
    # #     'data/augmentations=default',
    # # ]
    # # deca_conf = None
    # # deca_conf_path = None
    # # fixed_overrides_deca = None
    # # stage = None
    #
    # # # # #2 EMOSWIN
    # # conf = "emoswin"
    # # fixed_overrides_cfg = [
    # #     'model/backbone=swin',
    # #     # 'model/backbone=resnet50_cluster',
    # #     # 'model/backbone=vgg13',
    # #     # 'model/backbone=vgg16',
    # #     # 'model/backbone=vgg16_bn',
    # #     # 'model/backbone=vgg19_bn',
    # #     # 'model/settings=AU_emotionet',
    # #     'model/settings=AU_emotionet_bce',
    # #     # 'model/settings=AU_emotionet_bce_weighted',
    # #     # '+learning/lr_scheduler=reduce_on_plateau',
    # #     # '+learning/lr_scheduler=exponential',
    # #     # 'learning.batch_size_train=32',
    # #     # swin_type: swin_base_patch4_window7_224
    # #     # swin_type: swin_small_patch4_window7_224
    # #     # swin_type: swin_tiny_patch4_window7_224
    # #     'learning.batch_size_train=16',
    # #     # 'model.swin_type=swin_large_patch4_window7_224_22k',
    # #     # 'model.swin_type=swin_base_patch4_window7_224',
    # #     'model.swin_type=swin_small_patch4_window7_224',
    # #     # 'model.swin_type=swin_tiny_patch4_window7_224',
    # #     # 'data/datasets=affectnet_cluster',
    # #     # 'data/datasets=affectnet_v1_cluster',
    # #     # 'data/datasets=emotionet_0_cluster',
    # #     'data/datasets=emotionet_cluster',
    # #     # 'learning.max_steps=0',
    # #     # 'learning.max_epochs=0',
    # #     'learning/training=emotionet',
    # #     # 'learning/optimizer=adabound',
    # #     # 'data/augmentations=default',
    # #     'data/augmentations=default_with_resize',
    # # ]
    # # deca_conf = None
    # # deca_conf_path = None
    # # fixed_overrides_deca = None
    # # stage = None
    #
    # # EMODECA
    # conf = "emodeca_coarse_cluster"
    # fixed_overrides_cfg = [
    #     # 'model/settings=AU_emotionet',
    #     # 'model/settings=AU_emotionet_bce',
    #     # 'model/settings=AU_emotionet_bce_weighted',
    #     # '+model.mlp_norm_layer=BatchNorm1d',
    #     # 'model.use_identity=True', #
    #     # 'data/augmentations=default',
    #     # 'learning/optimizer=adabound',
    #     # 'data/datasets=affectnet_cluster',
    #     # 'data.data_class=AffectNetDataModuleValTest',
    #     'data/datasets=afew_va',
    #     'data.num_workers=16',
    #     'learning.max_epochs=100',
    #     'learning.val_check_interval=1.0',
    #     # 'data/datasets=affectnet_v1_cluster',
    #     # 'data/datasets=emotionet_0_cluster',
    #     # 'data/datasets=emotionet_cluster',
    #     # 'learning/training=emotionet',
    # ]

    run_names = []
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-33-08_2929045501486288941_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_23-27-01_6878473137680141500_EmoExpNet_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_22-43-49_3394193947032742938_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_22-44-04_8688670044855202446_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-14-55_3882362656686027659_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-14-55_-6806554933314077525_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-33-14_-168201693333588918_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-33-14_4337140264808142166_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-33-58_6855768109315710901_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-33-29_-1963146520759335803_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-39-09_-5731619448091644006_EmoMGCNet_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-32-49_-6879167987895418873_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-57-28_-4957717700349337532_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-58-27_-5553059236244394333_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-33-02_-5975857231436227431_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_14-07-53_4456762721214245215_Emo3DDFA_shake_samp-balanced_expr_early']
    run_names += [
        '/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-39-32_6108539290469616315_EmoDeep3DFace_shake_samp-balanced_expr_early']
    # EMONET SPLIT RUN:
    tags = None
    api = wandb.Api()

    # stage = 'detail'
    stage = 'coarse'

    for model_path in  run_names:
        name = str(Path(model_path).name)
        idx = name.find("Emo")
        run_id = name[:idx-1]
        run = api.run("rdanecek/EmoDECA/" + run_id)
        tags = set(run.tags)

        # allowed_tags = set(["COMPARISON", "INTERESTING", "FINAL_CANDIDATE", "BEST_CANDIDATE", "BEST_IMAGE_BASED"])
        #
        # if len(allowed_tags.intersection(tags)) == 0:
        #     print(f"Run '{name}' is not tagged to be tested and will be skipped.")
        #     continue


        deca_conf = None
        fixed_overrides_deca = None

        # augmenter = yaml.load(open(Path(__file__).parents[2] / "gdl_apps" / "EmoDECA" / "emodeca_conf" /
        #                            "data" / "augmentations" / "default.yaml"))#["augmentation"]
        # dataset =  yaml.load(open(Path(__file__).parents[2] / "gdl_apps" / "EmoDECA" / "emodeca_conf" /
        #                        "data" / "datasets" / "afew_va.yaml"))
        #
        # augmenter = OmegaConf.load(Path(__file__).parents[2] / "gdl_apps" / "EmoDECA" / "emodeca_conf" /
        #                            "data" / "augmentations" / "default.yaml")#["augmentation"]
        dataset = OmegaConf.load(Path(__file__).parents[2] / "gdl_apps" / "EmoDECA" / "emodeca_conf" /
                               "data" / "datasets" / "afew_va.yaml")


        # for mode in training_modes:
        # conf_overrides = fixed_overrides_cfg.copy()

        # # conf_overrides += [f"+learning.tags={'[' + ', '.join(tags) + ']'}"]
        #
        # conf_overrides += mode[0]
        # if model_path is None and fixed_overrides_deca is not None:
        #     deca_overrides = fixed_overrides_deca.copy()
        #     deca_overrides += mode[1]
        # else:
        #     deca_overrides=None

        # cfg = train_emodeca.configure(
        #     conf, conf_overrides,
        #     deca_default=deca_conf, deca_overrides=deca_overrides,
        #     deca_conf_path=model_path ,
        #     deca_stage=stage
        # )
        # GlobalHydra.instance().clear()

        cfg = OmegaConf.load(Path(model_path) / "cfg.yaml")

        # cfg.data.augmentation = augmenter

        keys_to_remove = []
        for key in cfg.data.keys():
            if key == "augmentation":
                continue
            keys_to_remove.append(key)
        for key in keys_to_remove:
            cfg.data.pop(key)

        for key in dataset.keys():
            cfg.data[key] = dataset[key]

        cfg.learning.val_check_interval = 1.
        cfg.learning.max_epochs = 100
        cfg.learning.checkpoint_after_training = "best"
        cfg.model.exp_loss = "none"
        cfg.data.augmentation = "none"
        try:
            cfg = OmegaConf.to_container(cfg)
            del cfg["data"]["augmentation"]
            cfg = DictConfig(cfg)
        #     with open_dict(cfg.data):
        #         del cfg.data.augmentation
        # #     if list(cfg.data.augmentation[1]["OneOf"][0].keys())[0] == 'JpegCompression':
        # #         del cfg.data.augmentation[1]["OneOf"][0]
        except:
            pass

        sub = True
        # sub = False
        if sub:
            submit(cfg, bid=30)
        else:
            cfg.data.num_workers = 2
            train_emodeca.train_emodeca(cfg, 0, True, True, project_name)


if __name__ == "__main__":
    train_emodeca_on_cluster()

