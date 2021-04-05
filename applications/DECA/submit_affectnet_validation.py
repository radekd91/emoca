from utils.condor import execute_on_cluster
from pathlib import Path
import affectnet_validation
import datetime
from omegaconf import OmegaConf


def submit(cfg, model_folder_name, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(affectnet_validation.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg.detail.learning.gpu_memory_min_gb * 1024
    # gpu_mem_requirement_mb = None
    cpus = cfg.detail.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.detail.learning.num_gpus
    num_jobs = 1
    max_time_h = 10
    max_price = 8000
    job_name = "finetune_deca"
    cuda_capability_requirement = 6
    mem_gb = 12
    args = f"{model_folder_name}"

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


def main():
    path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'

    run_names = []
    run_names += ['2021_03_25_19-42-13_DECA_training'] # DECA EmoNet
    # run_names += ['2021_03_18_21-10-25_DECA_training'] # Basic DECA
    # run_names += ['2021_03_26_15-05-56_DECA__DeSegFalse_DwC_early'] # Detail with coarse
    # run_names += ['2021_03_26_14-36-03_DECA__DeSegFalse_DeNone_early'] # No detail exchange

    for run_name in run_names:
        run_path = Path(path_to_models) / run_name
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)
        submit(conf, run_name)


if __name__ == "__main__":
    main()
