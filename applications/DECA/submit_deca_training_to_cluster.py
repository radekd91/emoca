from utils.condor import execute_on_cluster
from pathlib import Path
import train_deca
import datetime
from omegaconf import OmegaConf


def submit(cfg_coarse_pretrain, cfg_coarse, cfg_detail, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(train_deca.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    pretrain_coarse_file = submission_folder_local / "submission_coarse_pretrain_config.yaml"
    coarse_file = submission_folder_local / "submission_coarse_config.yaml"
    detail_file = submission_folder_local / "submission_detail_config.yaml"

    with open(pretrain_coarse_file, 'w') as outfile:
        OmegaConf.save(config=cfg_coarse_pretrain, f=outfile)
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
    max_price = 10000
    job_name = "train_deca"
    cuda_capability_requirement = 6
    mem_gb = 12
    args = f"{pretrain_coarse_file.name} {coarse_file.name} {detail_file.name}"

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


def train_on_selected_sequences():
    from hydra.core.global_hydra import GlobalHydra

    pretrain_coarse_conf = "deca_train_coarse_pretrain_cluster"
    coarse_conf = "deca_train_coarse_cluster"
    detail_conf = "deca_train_detail_cluster"

    finetune_modes = [
        # [['model/settings=default_coarse_emonet'], ['model/settings=default_detail_emonet']], # with emonet loss
        # [['model/settings=default_coarse_emonet', 'model.useSeg=true'], ['model/settings=default_detail_emonet']], # with emonet loss, segmentation coarse
        # [['model/settings=default_coarse_emonet', 'model.useSeg=true'], ['model/settings=default_detail_emonet', 'model.useSeg=true']], # with emonet loss, segmentation both
        # [['model/settings=default_coarse_emonet'], ['model/settings=default_detail_emonet']], # with emonet loss
        # [['model.useSeg=true'], []], # segmentation coarse
        [[], [], []],# without emonet loss
    ]
    fixed_overrides_coarse_pretrain = []
    fixed_overrides_coarse = []
    fixed_overrides_detail = []

    # emonet_weights = [0.15,] #default
    emonet_weights = [0.15/100,] #new default
    # emonet_weights = [0.15, 0.15/5, 0.15/10, 0.15/50, 0.15/100]

    config_pairs = []
    for emeonet_reg in emonet_weights:
        for fmode in finetune_modes:
            pretrain_coarse_overrides = fixed_overrides_coarse_pretrain.copy()
            coarse_overrides = fixed_overrides_coarse.copy()
            detail_overrides = fixed_overrides_detail.copy()
            # if len(fmode[0]) != "":
            pretrain_coarse_overrides += fmode[0]
            coarse_overrides += fmode[1]
            detail_overrides += fmode[2]

            # data_override = f'data.sequence_index={video_index}'
            # pretrain_coarse_overrides += [data_override]
            # coarse_overrides += [data_override]
            # detail_overrides += [data_override]
            emonet_weight_override = f'model.emonet_weight={emeonet_reg}'
            pretrain_coarse_overrides += [emonet_weight_override]
            coarse_overrides += [emonet_weight_override]
            detail_overrides += [emonet_weight_override]

            cfgs = train_deca.configure(
                pretrain_coarse_conf, pretrain_coarse_overrides,
                coarse_conf, coarse_overrides,
                detail_conf, detail_overrides
            )

            GlobalHydra.instance().clear()
            config_pairs += [cfgs]

            submit(cfgs[0], cfgs[1], cfgs[2])
            # break
        # break

    # for cfg_pair in config_pairs:
    #     submit(cfg_pair[0], cfg_pair[1])


def default_main():
    pretrain_coarse_conf = "deca_train_coarse_pretrain_cluster"
    pretrain_coarse_overrides = []

    coarse_conf = "deca_train_coarse_cluster"
    coarse_overrides = []

    detail_conf = "deca_train_detail_cluster"
    detail_overrides = []

    cfg_coarse_pretrain, cfg_coarse, cfg_detail = train_deca.configure(
        pretrain_coarse_conf, pretrain_coarse_overrides,
        coarse_conf, coarse_overrides,
        detail_conf, detail_overrides)

    submit(cfg_coarse_pretrain, cfg_coarse, cfg_detail)


if __name__ == "__main__":
    # default_main()
    train_on_selected_sequences()

