from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import create_attribution_maps_emotionet as script
import datetime
from omegaconf import OmegaConf
import time as t

def submit(resume_folder, deca_path, deca_image, trainable_deca_emonet, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    # submission_dir_local_mount = "/ps/scratch/rdanecek/InterpretableEmotion/submission"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/InterpretableEmotion/submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/InterpretableEmotion/submission"
    submission_dir_local_mount = "/is/cluster/work/rdanecek/InterpretableEmotion/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/InterpretableEmotion/submission"

    # submission_dir_local_mount = "/is/cluster/work/rdanecek/nterpretableEmotion/submission"
    # submission_dir_cluster_side = "/is/cluster/work/rdanecek/nterpretableEmotion/submission"

    result_dir_local_mount = "/is/cluster/work/rdanecek/InterpretableEmotion/results"
    result_dir_cluster_side = "/is/cluster/work/rdanecek/InterpretableEmotion/results"
    # result_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/InterpretableEmotion/results"
    # result_dir_local_mount = "/ps/scratch/rdanecek/InterpretableEmotion/results"
    # result_dir_cluster_side = "/ps/scratch/rdanecek/InterpretableEmotion/results"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    coarse_file = Path(result_dir_local_mount) / resume_folder / "cfg.yaml"
    with open(coarse_file, 'r') as f:
        cfg = OmegaConf.load(f)

    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg.learning.gpu_memory_min_gb * 1024
    # gpu_mem_requirement_mb = None
    # cpus = cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.learning.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_deca"
    cuda_capability_requirement = 6
    mem_gb = 16
    args = f"{str(Path(result_dir_cluster_side) / resume_folder)}"
    if deca_path is not None:
        args += f" {deca_path} {deca_image} {trainable_deca_emonet}"

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
                       chmod=True
                       )
    t.sleep(1)


def compute_confusion_matrix_on_cluster():
    deca_path = None
    ##deca_path = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_09_07_21-13-42_ExpDECA_Affec_balanced_expr_para_Jaw_NoRing_EmoB_EmoCnn_vgg_du_F2nVAE_DeSegrend_Aug_DwC_early"

    # Emo-AU VGG19BN
    deca_path = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_09_09_15-25-08_ExpDECA_Affec_balanced_expr_para_Jaw_NoRing_AU_DeSegrend_Aug_DwC_early/"

    # deca_image = None
    # deca_image = "predicted_images"
    deca_image = "predicted_detailed_image"
    # deca_image = "predicted_translated_image"
    # deca_image = "predicted_detailed_translated_image",

    trainable_deca_emonet = 1
    # trainable_deca_emonet = 0

    resume_folders = []
    resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_09_02_20-39-05_EmoCnn_vgg19_bn_none_AU_Aug_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_09_02_20-39-02_EmoCnn_vgg19_bn_none_AU_Aug_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_09_02_20-39-24_EmoCnn_vgg19_bn_none_AU_Aug_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_09_02_20-41-13_EmoSwin_swin_small_patch4_window7_224_none_AU_Aug_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_09_02_20-41-28_EmoSwin_swin_small_patch4_window7_224_none_AU_Aug_early']

    for resume_folder in resume_folders:
        submit(resume_folder, deca_path, deca_image, trainable_deca_emonet)


if __name__ == "__main__":
    compute_confusion_matrix_on_cluster()

