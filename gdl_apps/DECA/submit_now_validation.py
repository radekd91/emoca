from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import now_benchmark as script
import datetime
from omegaconf import OmegaConf
import time as t

def submit(cfg, model_folder_name, mode, stage, bid=10):
# def submit(cfg, model_folder_name, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    submission_dir_local_mount = "/is/cluster/work/rdanecek/emoca/now_validation_submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/now_validation_submission"
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg.detail.learning.gpu_memory_min_gb * 1024
    # gpu_mem_requirement_mb = None
    # cpus = cfg.detail.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.detail.learning.num_gpus
    num_jobs = 1
    max_time_h = 10
    max_price = 8000
    job_name = "deca_now"
    cuda_capability_requirement = 6
    mem_gb = 18
    args = f"{model_folder_name} {mode} {stage}"
    # args = f"{model_folder_name}"

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
    # path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'

    run_names = []
    # # run_names += ['/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_10-28-11_DECA_DecaD_VGGl_DeSegrend_Deex_early']
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_19-47-28_DECAStar_DecaD_EFswin_s_EDswin_s_VGGl_DeSegrend_Deex_early"] # later
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_19-47-21_DECA_DecaD_EFswin_s_EDswin_s_VGGl_DeSegrend_Deex_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-49-03_DECA_DecaD_EFswin_t_EDswin_t_DeSegrend_Deex_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-48-58_DECAStar_DecaD_EFswin_t_EDswin_t_DeSegrend_Deex_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_10-31-15_DECAStar_DecaD_VGGl_DeSegrend_Deex_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-38-20_DECA_DecaD_DeSegrend_Deex_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-53-04_DECA_DecaD_EFswin_s_EDswin_s_DeSegrend_Deex_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_00-42-34_DECAStar_DecaD_DeSegrend_Deex_early"]
    
    # barlow twin experiment round
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_18-59-03_DECA_DecaD_NoRing_VGGl_DeSegrend_idBTH-s10000_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_18-25-12_DECA_DecaD_NoRing_VGGl_DeSegrend_idBTH_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_16-40-04_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT-ft_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_16-40-04_DECA_DecaD_NoRing_VGGl_DeSegrend_idBTH-ft_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-41-54_DECA_DecaD_NoRing_VGGl_EmoB_EmoCnn_vgg_du_F2bar_DeSegrend_idBTH_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-41-51_DECA_DecaD_NoRing_VGGl_EmoB_EmoCnn_vgg_du_F2_DeSegrend_l1_loss_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-41-45_DECA_DecaD_NoRing_VGGl_EmoB_EmoCnn_vgg_du_F2bar_DeSegrend_idBT_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-41-23_DECA_DecaD_NoRing_VGGl_EmoB_EmoCnn_vgg_du_F2cos_DeSegrend_cosine_similarity_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-39-18_DECA_DecaD_NoRing_VGGl_DeSegrend_cosine_similarity_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-39-06_DECA_DecaD_NoRing_VGGl_DeSegrend_l1_loss_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-39-04_DECA_DecaD_NoRing_VGGl_DeSegrend_idBTH_Aug_early"]
    run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-38-50_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT_Aug_early"]
    
    
    mode = 'latest'
    stage = 'coarse'
    # stage = 'detail'


    for run_name in run_names:
        run_path = Path(path_to_models) / run_name
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)
        submit(conf, run_name, mode, stage)


if __name__ == "__main__":
    main()

