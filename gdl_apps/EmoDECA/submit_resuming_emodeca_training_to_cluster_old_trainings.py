from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_emodeca
import datetime
from omegaconf import OmegaConf
import time as t

def submit(resume_folder,
           stage = None,
           force_new_location = False,
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
    submission_folder_name = time + "_" + "submission"
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
        args += f" {stage} {int(force_new_location)}"

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
                       # chmod=False
                       )
    t.sleep(2)


def train_emodeca_on_cluster():

    # # # EMOEXPDECA
    resume_folders = []
    #
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-40-15_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-40-14_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-40-08_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-40-07_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-39-28_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-39-17_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-39-15_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-39-09_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-39-08_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-39-02_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-38-57_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-38-43_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-38-34_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-38-22_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-38-19_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-38-17_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-38-16_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-38-09_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-37-41_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-37-33_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-37-29_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-36-57_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-36-49_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-32-50_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-19-06_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-18-37_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-18-15_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-18-15_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-17-20_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]

    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_08_20_09-43-26_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_08_23_22-52-24_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_08_30_11-12-32_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_23-50-06_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_13-06-58_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_13-06-04_EmoSwin_swin_tiny_patch4_window7_224_shake_samp-balanced_expr_Aug_early']

    # resume_folders += [
    #     '/is/cluster/work/rdanecek/emoca/emodeca/2021_08_20_09-43-26_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000']
    # resume_folders += [
    #     '/is/cluster/work/rdanecek/emoca/emodeca/2021_08_23_22-52-24_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early']
    # resume_folders += [
    #     '/is/cluster/work/rdanecek/emoca/emodeca/2021_08_30_11-12-32_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early']
    # resume_folders += [
    #     '/is/cluster/work/rdanecek/emoca/emodeca/2021_08_22_23-50-06_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early']
    # resume_folders += [
    #     '/is/cluster/work/rdanecek/emoca/emodeca/2021_08_22_13-06-58_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early']
    # resume_folders += [
    #     '/is/cluster/work/rdanecek/emoca/emodeca/2021_08_22_13-06-04_EmoSwin_swin_tiny_patch4_window7_224_shake_samp-balanced_expr_Aug_early']

    stage = 1 # test stage
    force_new_location = True

    for resume_folder in resume_folders:
        submit(resume_folder, stage, force_new_location)


if __name__ == "__main__":
    train_emodeca_on_cluster()

