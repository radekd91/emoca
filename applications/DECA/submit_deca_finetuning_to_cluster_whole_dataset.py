from utils.condor import execute_on_cluster
from pathlib import Path
import test_and_finetune_deca
import datetime
from omegaconf import OmegaConf


def submit(cfg_coarse, cfg_detail, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(test_and_finetune_deca.__file__).absolute()
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
    gpus = cfg_coarse.learning.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 8000
    job_name = "finetune_deca"
    cuda_capability_requirement = 6
    mem_gb = 12
    args = f"{coarse_file.name} {detail_file.name}"

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


def finetune_on_all_sequences():
    from hydra.core.global_hydra import GlobalHydra

    coarse_conf = "deca_finetune_coarse_cluster_all"
    detail_conf = "deca_finetune_detail_cluster_all"

    finetune_modes = [
        # [['model/settings=default_coarse_emonet'], ['model/settings=default_detail_emonet']], # with emonet loss
        # [['model/settings=default_coarse_emonet', 'model.useSeg=true'], ['model/settings=default_detail_emonet']], # with emonet loss, segmentation coarse
        # [['model/settings=default_coarse_emonet', 'model.useSeg=true', 'learning/optimizer=finetune_adam_coarse_lower_lr'],
        #  ['model/settings=default_detail_emonet', 'learning/optimizer=finetune_adam_coarse_lower_lr']], # with emonet loss, segmentation coarse, lower lr
        # [['model/settings=default_coarse_emonet', 'model.useSeg=true'], ['model/settings=default_detail_emonet', 'model.useSeg=true']], # with emonet loss, segmentation both
        # [['model/settings=default_coarse_emonet'], ['model/settings=default_detail_emonet']], # with emonet loss
        # [['model.useSeg=true'], []], # segmentation coarse

        [['model.useSeg=true', 'data/augmentations=default'], ['data/augmentations=default']], # segmentation coarse, DATA AUGMENTATION

        [['model.useSeg=true', 'model/settings=default_coarse_emonet', 'data/augmentations=default'],
            ['data/augmentations=default', 'model/settings=default_detail_emonet']], # segmentation coarse, DATA AUGMENTATION , with EmoNet

        # [['model.useSeg=true', 'data/augmentations=default'],
        #  ['data/augmentations=default', 'model.detail_constrain_type=none']], # segmentation coarse, DATA AUGMENTATION , no detail constraint
        #
        # [['model.useSeg=true', 'model/settings=default_coarse_emonet', 'data/augmentations=default'],
        #  ['data/augmentations=default', 'model/settings=default_detail_emonet', 'model.detail_constrain_type=none']], # segmentation coarse, DATA AUGMENTATION , with EmoNet, no detail constraint
        #
        # [['model.useSeg=true', 'data/augmentations=default'],
        #  ['data/augmentations=default', 'model.detail_constrain_type=none', 'model.train_coarse=true']], # segmentation coarse, DATA AUGMENTATION , train detail and coarse together
        #
        # [['model/settings=default_coarse_emonet', 'model.useSeg=true'],
        #     ['model/settings=default_detail_emonet', 'model.use_detail_l1=false', 'model.use_detail_mrf=false']], # without other detail losses, emo only
        # [['model/settings=default_coarse_emonet', 'model.useSeg=true'],
        #     ['model/settings=default_detail_emonet', 'model.use_detail_mrf=false']], # without mrf losses
        # [['model/settings=default_coarse_emonet', 'model.useSeg=true'],
        #     ['model/settings=default_detail_emonet', 'model.use_detail_l1=false']] # without mrf losses
        # [['model/settings=default_coarse_emonet', 'model.background_from_input=false'],
        #     ['model/settings=default_detail_emonet', 'model.background_from_input=false']], # with emonet loss, background black
        # [['model/settings=default_coarse_emonet', 'model.background_from_input=false', 'model.useSeg=true'],
        #     ['model/settings=default_detail_emonet', 'model.background_from_input=false']],
        # with emonet loss, background black
        # [[], []],# without emonet loss
        # [['model.useSeg=true', 'learning/optimizer=finetune_adam_coarse_lower_lr'],
        #     ['learning/optimizer=finetune_adam_coarse_lower_lr']], #segmentation coarse, lower lr
        # [['model/settings=default_coarse_emonet', 'model.useSeg=true'], ['model/settings=default_detail_emonet']],
        # with emonet loss, segmentation coarse
    ]

    # test_vis_frequency: 30
    # val_vis_frequency: 200
    # train_vis_frequency: 100
    fixed_overrides_coarse = ["model.val_vis_frequency=3000", "model.train_vis_frequency=500", "model.test_vis_frequency=1000"]
    fixed_overrides_detail = ["model.val_vis_frequency=3000", "model.train_vis_frequency=500", "model.test_vis_frequency=1000"]

    emonet_weights = [0.15/100,] # new default
    # emonet_weights = [0.15, 0.15/5, 0.15/10, 0.15/50, 0.15/100]

    config_pairs = []
    for emeonet_reg in emonet_weights:
        for fmode in finetune_modes:
            coarse_overrides = fixed_overrides_coarse.copy()
            detail_overrides = fixed_overrides_detail.copy()
            # if len(fmode[0]) != "":
            coarse_overrides += fmode[0]
            detail_overrides += fmode[1]

            emonet_weight_override = f'model.emonet_weight={emeonet_reg}'
            # data_override = f'data.sequence_index={video_index}'
            # coarse_overrides += [data_override]
            # detail_overrides += [data_override]
            coarse_overrides += [emonet_weight_override]
            detail_overrides += [emonet_weight_override]

            cfgs = test_and_finetune_deca.configure(
                coarse_conf, coarse_overrides, detail_conf, detail_overrides)

            GlobalHydra.instance().clear()
            config_pairs += [cfgs]

            submit(cfgs[0], cfgs[1])
                # break
            # break

    # for cfg_pair in config_pairs:
    #     submit(cfg_pair[0], cfg_pair[1])


def default_main():
    coarse_conf = "deca_finetune_coarse_cluster_all"
    coarse_overrides = []

    detail_conf = "deca_finetune_detail_cluster_all"
    detail_overrides = []

    cfg_coarse, cfg_detail = test_and_finetune_deca.configure(
        coarse_conf, coarse_overrides, detail_conf, detail_overrides)

    submit(cfg_coarse, cfg_detail)


if __name__ == "__main__":
    # default_main()
    finetune_on_all_sequences()

