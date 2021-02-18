from utils.condor import execute_on_cluster
from pathlib import Path
import test_and_finetune_deca
import datetime
from omegaconf import OmegaConf


def submit(cfg_coarse, cfg_detail):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"
    time = datetime.datetime.now().strftime("%b_%d_%Y_%H-%M-%S")
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


    python_bin = 'python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = 14000
    # gpu_mem_requirement_mb = None
    cpus = cfg_coarse.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg_coarse.learning.num_gpus
    num_jobs = 1
    bid = 10
    max_time_h = 24
    max_price = 5000
    job_name = "finetune_deca"
    cuda_capability_requirement = 6
    mem_gb = 20
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

def main():
    coarse_conf = "deca_finetune_coarse_cluster"
    coarse_overrides = []

    detail_conf = "deca_finetune_detail_cluster"
    detail_overrides = []

    cfg_coarse, cfg_detail = test_and_finetune_deca.configure(
        coarse_conf, coarse_overrides, detail_conf, detail_overrides)

    submit(cfg_coarse, cfg_detail)


if __name__ == "__main__":
    main()


