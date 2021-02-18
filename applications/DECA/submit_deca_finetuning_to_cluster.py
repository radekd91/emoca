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


    python_bin = 'python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = 14000
    # gpu_mem_requirement_mb = None
    cpus = cfg_coarse.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg_coarse.learning.num_gpus
    num_jobs = 1
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



test_video_indices = [
 148,
 399,
 169,
 167,
 195,
 207,
 241,
 269,
 294,
 374,
 380,
 381,
 382,
 385,
 390,
 394,
 397,
 404,
 151,
 161,
 393,
 145,
 150
]

test_videos = [
    '119-30-848x480.mp4', # black lady with at Oscars
    '9-15-1920x1080.mp4', # smiles, sadness, tears, girl with glasses
    '19-24-1920x1080.mp4', # angry young black guy on stage
    '17-24-1920x1080.mp4', # black guy on stage, difficult light
    '23-24-1920x1080.mp4', # white woman, over-articulated expressions
    '24-30-1920x1080-2.mp4', # white woman, over-articulated expressions
    '28-30-1280x720-1.mp4', # angry black guy
    '31-30-1920x1080.mp4', # crazy white guy, beard, view from the side
    '34-25-1920x1080.mp4', # white guy, mostly neutral
    '50-30-1920x1080.mp4', # baby
    '60-30-1920x1080.mp4', # smiling asian woman
    '61-24-1920x1080.mp4', # very lively white woman
    '63-30-1920x1080.mp4', # smiling asian woman
    '66-25-1080x1920.mp4', # white girl acting out an emotional performance
    '71-30-1920x1080.mp4', # old white woman, camera shaking
    '83-24-1920x1080.mp4', # excited black guy (but expressions mostly neutral)
    '87-25-1920x1080.mp4', # white guy explaining stuff, mostly neutral
    '95-24-1920x1080.mp4', # white guy explaining stuff, mostly neutral
    '122-60-1920x1080-1.mp4', # crazy white youtuber, lots of overexaggerated expressiosn
    '135-24-1920x1080.mp4', # a couple watching a video, smiles, sadness, tears
    '82-25-854x480.mp4', # Rachel McAdams, sadness, anger
    '111-25-1920x1080.mp4', # disgusted white guy
    '121-24-1920x1080.mp4', # white guy scared and happy faces
]


def finetune_on_selected_sequences():
    from hydra.core.global_hydra import GlobalHydra

    coarse_conf = "deca_finetune_coarse_cluster"
    detail_conf = "deca_finetune_detail_cluster"

    finetune_modes = [
        ["", ""], # without emonet loss
        ['model/settings=default_coarse_emonet', 'model/settings=default_coarse_emonet'] # with emonet loss
    ]
    fixed_overrides_coarse = []
    fixed_overrides_detail = []

    config_pairs = []
    for i, video_index in enumerate(test_video_indices):
        for fmode in finetune_modes:
            coarse_overrides = fixed_overrides_coarse.copy()
            detail_overrides = fixed_overrides_detail.copy()
            if fmode[0] != "":
                coarse_overrides += [fmode[0]]
                detail_overrides += [fmode[0]]
            data_override = f'data.sequence_index={video_index}'
            coarse_overrides += [data_override]
            detail_overrides += [data_override]

            cfgs = test_and_finetune_deca.configure(
                coarse_conf, coarse_overrides, detail_conf, detail_overrides)

            GlobalHydra.instance().clear()
            config_pairs += [cfgs]
            # break
        # break

    for cfg_pair in config_pairs:
        submit(cfg_pair[0], cfg_pair[1])


def default_main():
    coarse_conf = "deca_finetune_coarse_cluster"
    coarse_overrides = []

    detail_conf = "deca_finetune_detail_cluster"
    detail_overrides = []

    cfg_coarse, cfg_detail = test_and_finetune_deca.configure(
        coarse_conf, coarse_overrides, detail_conf, detail_overrides)

    submit(cfg_coarse, cfg_detail)


if __name__ == "__main__":
    # default_main()
    finetune_on_selected_sequences()

