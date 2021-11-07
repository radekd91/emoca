from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_emodeca
import datetime
from omegaconf import OmegaConf
import time as t

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
    submission_folder_name = time + "_" + str(hash(time)) + "_" + "submission"
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
        args += f" {stage}"

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

    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-19-06_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-18-37_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_detail_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-18-15_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-18-15_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_00-17-20_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # # # resume_folders += [""]
    #
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-49-52_3616560481833876621_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-49-28_-925203403207733775_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-48-16_-299945756512845344_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-48-05_-5664379014652292077_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-47-50_-3207519268063100363_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-46-43_8658869785020395462_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-46-23_8146525650800657465_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-42-26_5548877040424169402_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-40-18_4381403620612548112_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-39-55_6668992776665662435_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-39-22_-1200922317249422165_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-38-22_4892029920419801355_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-37-01_4694466027278472924_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-33-56_-7014844697435846956_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_01-26-25_3048606992670197583_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-13-44_7594812148218371447_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-12-45_8686444472974725079_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-56_-7151845829000593112_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-10-01_8849525666457080895_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-12-22_8905147559991240403_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-12-00_8170997529930865189_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-11-48_6943917563097387424_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-11-48_-4313232006188099698_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-11-04_-616848941231868407_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-10-36_-6204014578632901572_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    ## resume_folders += [ broken
    ##     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-30_-7504387131299355288_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-10-34_6655774557877447571_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-56_8205822725187552677_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-10-01_7830062538309299471_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-54_517161774264792209_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-54_878519420551507667_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-52_1472303342094344016_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-40_-6423494592759815171_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-40_-517800139346669070_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-20_-9048713322939927482_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-20_-5878500815728530877_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-17_-6217749203068621462_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-17_-2484383088484391993_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-16_-6990945706460088493_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-16_-4099604869929879823_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-11_7904005134964528615_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-11_369811129168178053_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-09_-7501041025296508430_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-09-09_-1100436455311063297_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-08-52_8739672784865170383_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_05_00-08-52_2240919865846595319_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # # resume_folders += [
    #     # "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_04_21-02-56_7522256879763590267_Emo3DDFA_shake_samp-balanced_expr_early"]
    # # resume_folders += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_04_21-02-30_5426539069370145078_Emo3DDFA_shake_samp-balanced_expr_early"]
    # # resume_folders += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_04_21-02-04_-7464743312740992253_Emo3DDFA_shake_samp-balanced_expr_early"]
    # # resume_folders += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_04_21-01-46_8158034355824336420_Emo3DDFA_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_03_21-40-03_5989905820300931279_EmoNet_shake_samp-balanced_va_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_03_21-33-57_-1075576883523196937_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_shake_samp-balanced_va_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_03_21-33-47_3812680534035783971_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_03_21-33-45_6761071402828749734_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_va_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_03_21-32-48_7047090560845025255_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_03_21-32-48_3354256371503729220_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_va_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_03_21-32-32_3314831948588698128_EmoDECA_Affec_Orig_nl-4BatchNorm1d_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-36-55_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-36-32_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-36-21_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-26-02_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-15-31_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-15-09_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-13-48_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-12-49_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-12-42_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_21-12-41_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # ## resume_folders += [
    # ##     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_20-49-39_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_20-45-38_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_20-34-20_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_20-34-17_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_20-34-13_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_20-33-55_EmoSwin_swin_tiny_patch4_window7_224_shake_samp-balanced_expr_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_14-38-11_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_14-37-39_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_14-37-32_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_14-36-54_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_14-36-51_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_14-36-49_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_14-36-46_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_01_14-36-37_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
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
    #
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

    # # sanity checker - can this repeat the clone performance?
    # resume_folders += [
    #     "2021_11_07_12-09-07_-6901351237565527680_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"
    # ]

    # # comparison methods
    resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_06_00-52-04_4368104297342516790_EmoDeep3DFace_shake_samp-balanced_expr_early"]
    resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_06_00-50-56_3141812600970784200_EmoDeep3DFace_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_04_21-02-56_7522256879763590267_Emo3DDFA_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_04_21-02-30_5426539069370145078_Emo3DDFA_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_04_21-02-04_-7464743312740992253_Emo3DDFA_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_04_21-01-46_8158034355824336420_Emo3DDFA_shake_samp-balanced_expr_early"]

    stage = 1 # test stage

    for resume_folder in resume_folders:
        submit(resume_folder, stage)


if __name__ == "__main__":
    train_emodeca_on_cluster()

