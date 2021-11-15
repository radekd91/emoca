from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_emodeca
import datetime
from omegaconf import OmegaConf
import time as t
import random

project_name = "EmoDECA_Afew-VA"

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
    if stage is not None:
        args += f" {stage} {1} {0} {project_name}"

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
                       modules_to_load=['cuda/11.4'],
                       # chmod=False
                       )
    # t.sleep(2)


def train_emodeca_on_cluster():

    # # # EMOEXPDECA
    resume_folders = []

    # AFEW-VA
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_05-44-22_1355786939759696886_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_05-43-38_3263741745524855578_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_05-43-38_-2173935624180351936_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_05-43-37_6419514851386673597_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_05-43-37_5804902462619208346_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_05-43-27_4690126832397182704_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_05-43-27_2517176429429832703_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_05-43-04_4390452123163373707_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-52_5063894700880742144_EmoSwin_swin_base_patch4_window7_224_shake_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-08_-6557913516496940171_EmoNet_shake_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-06_5824575168153882277_EmoSwin_swin_tiny_patch4_window7_224_shake_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-03_1650355822631362210_EmoCnn_vgg19_bn_shake_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-50-47_2165311197696423212_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-50-45_-1078162224083866132_EmoCnn_resnet50_shake_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-50-34_-4147098627249931340_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-50-19_-2753470226601469411_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-57_-8073653537661498145_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-57_8253184588845804991_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-56_-769750103465679934_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-53_-1178328002648582111_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-51_3416645760118062473_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-49_5394202179722405086_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-48_-6639782345534213702_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-47_9100739897692345548_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-46_1477440115901307368_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-46_9029272801247232242_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-44_-830365848295859353_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-40_815930939297900178_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-30_-4866079105883540143_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-30_8644805620872834117_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-49-30_6952350694460824219_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_early"]

    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-39-00_4363551435501292966_EmoCnn_vgg19_bn_shake_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-39-00_5784706256688858056_EmoCnn_resnet50_shake_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-03_1650355822631362210_EmoCnn_vgg19_bn_shake_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-50-45_-1078162224083866132_EmoCnn_resnet50_shake_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-39-11_8967464683573230563_EmoSwin_swin_base_patch4_window7_224_shake_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-38-43_-2897877075877919183_EmoSwin_swin_tiny_patch4_window7_224_shake_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-52_5063894700880742144_EmoSwin_swin_base_patch4_window7_224_shake_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-06_5824575168153882277_EmoSwin_swin_tiny_patch4_window7_224_shake_Aug_early"]


    bid = 10

    stage = 1 # test stage

    for resume_folder in resume_folders:
        submit(resume_folder, stage, bid=bid)


if __name__ == "__main__":
    train_emodeca_on_cluster()

