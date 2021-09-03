from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import resume_expdeca_training
import datetime
from omegaconf import OmegaConf
import time as t

def submit(
        resume_folder,
        stage,
        resume_from_previous,
        force_new_location,
        bid=10):

    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/expdeca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/expdeca/submission"

    # result_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca"
    # result_dir_cluster_side = "/ps/scratch/rdanecek/emoca/finetune_deca"

    result_dir_local_mount = "/is/cluster/work/rdanecek/emoca/finetune_deca"
    result_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/finetune_deca"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(resume_expdeca_training.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name


    coarse_file = Path(result_dir_local_mount) / resume_folder / "cfg.yaml"
    with open(coarse_file, 'r') as f:
        cfg = OmegaConf.load(f)
    cfg_coarse = cfg.coarse

    submission_folder_local.mkdir(parents=True)
    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg_coarse.learning.gpu_memory_min_gb * 1024
    # gpu_mem_requirement_mb = None
    # cpus = cfg_coarse.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg_coarse.learning.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_deca"
    cuda_capability_requirement = 6
    mem_gb = 40
    args = f"{str(Path(result_dir_cluster_side) / resume_folder)} {stage} {int(resume_from_previous)} {int(force_new_location)}"

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
    t.sleep(1)


def main():
    # stage = 0
    # resume_from_previous = False
    # force_new_location = False
    #
    # resume_folders = []
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_19-47-28_DECAStar_DecaD_EFswin_s_EDswin_s_VGGl_DeSegrend_Deex_early"] # later
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_19-47-21_DECA_DecaD_EFswin_s_EDswin_s_VGGl_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_10-31-15_DECAStar_DecaD_VGGl_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_10-28-11_DECA_DecaD_VGGl_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_01-09-40_DECAStar_DecaD_EFswin_s_EDswin_s_VGGl_noPho_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_01-06-23_DECAStar_DecaD_EFswin_s_EDswin_s_VGGl_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_01-06-22_DECAStar_DecaD_EFswin_s_EDswin_s_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-53-14_DECA_DecaD_EFswin_s_EDswin_s_VGGl_noPho_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-53-14_DECA_DecaD_EFswin_s_EDswin_s_VGGl_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-53-04_DECA_DecaD_EFswin_s_EDswin_s_DeSegrend_Deex_early"]
    # ## resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-49-21_DECAStar_DecaD_EFswin_t_EDswin_t_VGGl_noPho_DeSegrend_Deex_early"] # have not yet crashed
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-49-03_DECA_DecaD_EFswin_t_EDswin_t_VGGl_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-49-03_DECA_DecaD_EFswin_t_EDswin_t_DeSegrend_Deex_early"]
    # ## resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-49-02_DECAStar_DecaD_EFswin_t_EDswin_t_VGGl_DeSegrend_Deex_early"] # have not yet crashed
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-49-01_DECA_DecaD_EFswin_t_EDswin_t_VGGl_noPho_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-48-58_DECAStar_DecaD_EFswin_t_EDswin_t_DeSegrend_Deex_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-38-20_DECA_DecaD_DeSegrend_Deex_early"]
    ## resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_21-50-45_DECA__DeSegFalse_early/"]


    # for resume_folder in resume_folders:
    #     submit(resume_folder, stage, resume_from_previous, force_new_location)

    # stage = 2
    # resume_from_previous = False
    # force_new_location = False
    #
    # resume_folders = []


    # for resume_folder in resume_folders:
    #     submit(resume_folder, stage, resume_from_previous, force_new_location)

    # stage = 2
    # resume_from_previous = True
    # force_new_location = False

    resume_folders = []


    # for resume_folder in resume_folders:
    #     submit(resume_folder, stage, resume_from_previous, force_new_location)

    stage = 2
    resume_from_previous = False
    force_new_location = False

    ## ExpDECASTAR (neural rendering, different emonetion networks)

    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_10-52-43_ExpDECAStar_Affec_para_Jaw_NoRing_EmoB_EmoSwin_sw_F2VAE_DeSegrend_Exnone_DwC_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_10-52-30_ExpDECAStar_Affec_para_Jaw_NoRing_EmoB_EmoSwin_sw_F2VAE_DeSegrend_Exnone_DwC_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_10-52-29_ExpDECAStar_Affec_para_Jaw_EmoB_EmoSwin_sw_F2VAE_DeSegrend_Exnone_DeNo_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_10-52-18_ExpDECAStar_Affec_para_Jaw_NoRing_EmoB_EmoCnn_res_F2VAE_DeSegrend_Exnone_DwC_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_10-52-13_ExpDECAStar_Affec_para_Jaw_EmoB_EmoSwin_sw_F2VAE_DeSegrend_Exnone_DeNo_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_10-52-13_ExpDECAStar_Affec_para_Jaw_EmoB_EmoCnn_res_F2VAE_DeSegrend_Exnone_DeNo_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_10-51-59_ExpDECAStar_Affec_para_Jaw_NoRing_EmoB_EmoCnn_vgg_F2VAE_DeSegrend_Exnone_DwC_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_10-51-58_ExpDECAStar_Affec_para_Jaw_NoRing_EmoB_EmoNet_sha_F2VAE_DeSegrend_Exnone_DwC_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_10-51-44_ExpDECAStar_Affec_para_Jaw_EmoB_EmoCnn_vgg_F2VAE_DeSegrend_Exnone_DeNo_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_24_00-13-33_ExpDECAStar_Affec_para_Jaw_EmoB_F2VAE_DeSegrend_Exnone_DeNo_early/"]


    for resume_folder in resume_folders:
        submit(resume_folder, stage, resume_from_previous, force_new_location)
#


if __name__ == "__main__":
    main()
