from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_emodeca
import datetime
from omegaconf import OmegaConf
import time as t
import random
from wandb import Api

def submit(resume_folder,
           stage = None,
           bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    # submission_dir_local_mount = "/ps/scratch/rdanecek/emoca/submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/submission"


    result_dir_local_mount = "/is/cluster/work/rdanecek/emoca/emodeca"
    result_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/emodeca"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(random.randint(0,10000000))) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(train_emodeca.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name


    submission_folder_local.mkdir(parents=True)

    config_file = Path(result_dir_local_mount) / resume_folder / "cfg.yaml"

    with open(config_file, 'r') as f:
        cfg = OmegaConf.load(f)

    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg.learning.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb = 15 * 1024
    gpu_mem_requirement_mb_max = 35 * 1024
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
    args = f"{str(Path(result_dir_cluster_side) / resume_folder / 'cfg.yaml')}"

    start_from_previous = 1
    force_new_location = 1
    stage = 1
    project_name = "AffectNetEmoDECATest"
    args += f" {stage} {start_from_previous} {force_new_location} {project_name} "

    execute_on_cluster(str(cluster_script_path),
                       args,
                       str(submission_folder_local),
                       str(submission_folder_cluster),
                       str(cluster_repo_path),
                       python_bin=python_bin,
                       username=username,
                       gpu_mem_requirement_mb=gpu_mem_requirement_mb,
                       gpu_mem_requirement_mb_max=gpu_mem_requirement_mb_max,
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
                       modules_to_load=['cuda/11.4'],
                       # chmod=False
                       )
    # t.sleep(2)


def train_emodeca_on_cluster():

    # # # EMOEXPDECA
    resume_folders = []
    bid = 10
    stage = 1 # test stage
    api = Api()

    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-39-36_-1044078422889696991_EmoDeep3DFace_shake_samp-balanced_expr_early"]
    resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-39-32_6108539290469616315_EmoDeep3DFace_shake_samp-balanced_expr_early"]
    # resume_folders += [""]
    # resume_folders += [""]
    # resume_folders += [""]

    for resume_folder in resume_folders:
        name = str(Path(resume_folder).name)
        idx = name.find("Emo")
        run_id = name[:idx-1]
        run = api.run("rdanecek/EmoDECA/" + run_id)
        tags = set(run.tags)

        allowed_tags = set(["COMPARISON", "INTERESTING", "FINAL_CANDIDATE", "BEST_CANDIDATE"])

        if len(allowed_tags.intersection(tags)) == 0:
            print(f"Run '{name}' is not tagged to be tested and will be skipped.")
            continue

        cfg = OmegaConf.load(Path(resume_folder) / "cfg.yaml")

        cfg.data.data_class = "AffectNetEmoNetSplitModuleTest"

        submit_ = False
        # submit_ = True
        if submit_:
            submit(cfg, stage, bid=bid)
        else:
            project_name = "E"
            stage = 1
            start_from_previous = 1
            force_new_location = 1
            train_emodeca.train_emodeca(cfg, stage, start_from_previous, force_new_location, "AffectNetEmoDECATest")


if __name__ == "__main__":
    train_emodeca_on_cluster()

