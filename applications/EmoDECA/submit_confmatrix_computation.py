from utils.condor import execute_on_cluster
from pathlib import Path
import compute_confusion_matrix as script
import datetime
from omegaconf import OmegaConf
import time as t

def submit(resume_folder, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    # submission_dir_local_mount = "/ps/scratch/rdanecek/emoca/submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/submission"

    # result_dir_local_mount = "/is/cluster/work/rdanecek/emoca/finetune_deca"
    # result_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/finetune_deca"
    result_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/emodeca"
    result_dir_cluster_side = "/ps/scratch/rdanecek/emoca/emodeca"

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
    cpus = cfg.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.learning.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_deca"
    cuda_capability_requirement = 6
    mem_gb = 16
    args = f"{str(Path(result_dir_cluster_side) / resume_folder)}"

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
                       chmod=False
                       )
    t.sleep(1)


def compute_confusion_matrix_on_cluster():
    resume_folders = []
    # resume_folders += ['2021_05_11_16-48-17_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_detail_shake_Aug_early_AdaBound']
    # resume_folders += ['2021_05_11_16-48-20_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_detail_shake_early']
    # resume_folders += ['2021_05_07_21-23-23_EmoNet_shake_early_AdaBound']
    # resume_folders += ['2021_05_12_22-25-56_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_detail_shake_early']
    # resume_folders += ['2021_05_12_22-30-22_EmoNet_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_11_17-00-58_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_detail_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_11_17-00-16_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_detail_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_11_16-48-20_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_detail_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_11_16-48-17_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_detail_shake_Aug_early_AdaBound']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_11_16-39-58_EmoNet_shake_Aug_early_AdaBound']
    resume_folders += [
        '/ps/scratch/rdanecek/emoca/emodeca/2021_05_08_13-58-37_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_detail_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_08_13-58-15_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_08_13-47-53_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_shake_early_AdaBound']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_08_13-47-52_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_detail_shake_early_AdaBound']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_08_13-46-30_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_detail_shake_early_AdaBound']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_08_13-46-12_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_shake_early_AdaBound']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_07_21-23-23_EmoNet_shake_early_AdaBound']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_07_21-19-57_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_detail_shake_early_AdaBound']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_07_20-55-36_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_detail_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_07_20-55-22_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_shake_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_06_17-03-45_EmoNet_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_06_13-33-46_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_detail_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_06_13-27-18_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_detail_shake_balanced_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_06_13-26-45_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_shake_early']

    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_05_10-56-37_EmoDECA_Affec_Orig_nl-4_exp_jaw_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_05_00-48-00_EmoDECA_Affec_Orig_nl-4_exp_jaw_detail_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_05_00-45-45_EmoDECA_Affec_Orig_nl-4_exp_jaw_detail_shake_balanced_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_05_00-45-23_EmoDECA_Affec_Orig_nl-4_exp_jaw_shake_balanced_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_05_00-44-44_EmoDECA_Affec_Orig_nl-2_exp_jaw_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_05_00-44-39_EmoDECA_Affec_Orig_nl-3_exp_jaw_shake_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_05_00-44-25_EmoDECA_Affec_Orig_nl-1_exp_jaw_shake_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-24-39_EmoNet_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-06-33_EmoDECA_Affec_Orig_nl-4_exp_jaw_detail_balanced_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-06-27_EmoDECA_Affec_Orig_nl-4_exp_jaw_balanced_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-05-53_EmoDECA_Affec_Orig_nl-4_exp_jaw_detail_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-05-51_EmoDECA_Affec_Orig_nl-4_exp_jaw_early']
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-05-33_EmoDECA_Affec_Orig_nl-3_exp_jaw_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-04-13_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_detail_balanced_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-04-07_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_detail_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-03-54_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_balanced_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-03-35_EmoDECA_Affec_ExpDECA_nl-3_exp_jaw_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_04_03-03-34_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_04_23_16-51-54_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_detail_early']
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_04_23_16-51-53_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_early']
    #     # resume_folders += ['']

    # resume_folders += ['2021_05_12_14-54-24_EmoDECA_Affec_ExpDECA_EmoNetC_unpose_light_cam_shake_early']
    # resume_folders += ['2021_05_12_14-51-36_EmoDECA_Affec_ExpDECA_EmoNetCD_unpose_light_cam_shake_early']
    # resume_folders += ['2021_05_12_14-22-36_EmoDECA_Affec_ExpDECA_EmoNetD_unpose_light_cam_shake_early']
    # resume_folders += ['2021_05_11_22-57-26_EmoDECA_Affec_ExpDECA_EmoNetCD_unpose_light_cam_shake_early']
    # resume_folders += ['2021_05_12_14-22-40_EmoDECA_Affec_ExpDECA_EmoNetD_shake_early']

    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-19-53_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-19-32_EmoDECA_Affec_Orig_nl-4_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-19-07_EmoDECA_Affec_Orig_nl-4_exp_jaw_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-12-42_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-12-14_EmoDECA_Affec_Orig_nl-4_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-12-03_EmoDECA_Affec_Orig_nl-4_exp_jaw_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-09-33_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-08-51_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_00-27-01_EmoNet_shake_balanced_early_AdaBound']
    #
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_00-26-42_EmoNet_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_00-22-32_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_00-21-29_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_00-15-43_EmoDECA_Affec_ExpDECA_nl-4_exp_jaw_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-19-53_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-19-32_EmoDECA_Affec_Orig_nl-4_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-19-07_EmoDECA_Affec_Orig_nl-4_exp_jaw_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-12-42_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-12-14_EmoDECA_Affec_Orig_nl-4_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-12-03_EmoDECA_Affec_Orig_nl-4_exp_jaw_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-09-33_EmoDECA_Affec_ExpDECA_nl-4_id_exp_jaw_shake_balanced_early']
    #
    # resume_folders += [
    #     '/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_01-08-51_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_balanced_early']
    #
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_00-27-01_EmoNet_shake_balanced_early_AdaBound']
    #
    # resume_folders += ['/ps/scratch/rdanecek/emoca/emodeca/2021_05_18_00-26-42_EmoNet_shake_balanced_early']


    # resume_folders = sorted(list(Path('/ps/scratch/rdanecek/emoca/emodeca').glob("*")))
    # resume_folders = [str(s) for s in resume_folders]
    for resume_folder in resume_folders:
        submit(resume_folder)


if __name__ == "__main__":
    compute_confusion_matrix_on_cluster()

