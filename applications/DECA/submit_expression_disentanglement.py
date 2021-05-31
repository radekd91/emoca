from utils.condor import execute_on_cluster
from pathlib import Path
import expression_disentanglement_experiment
import datetime
from omegaconf import OmegaConf
import time as t


def submit(cfg, model_folder_name, codes_to_exchange, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    submission_dir_local_mount = "/is/cluster/work/rdanecek/emoca/affectnet_test_submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/affectnet_test_submission"
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(expression_disentanglement_experiment.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg.detail.learning.gpu_memory_min_gb * 1024
    # gpu_mem_requirement_mb = None
    cpus = cfg.detail.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.detail.learning.num_gpus
    num_jobs = 1
    max_time_h = 10
    max_price = 8000
    job_name = "finetune_deca"
    cuda_capability_requirement = 6
    mem_gb = 12
    args = f"{model_folder_name} {','.join(codes_to_exchange)}"

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
    path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'

    run_names = []
    # run_names += ['2021_03_25_19-42-13_DECA_training'] # DECA EmoNet
    # run_names += ['2021_03_29_23-14-42_DECA__EmoLossB_F2VAEw-0.00150_DeSegFalse_early'] # DECA EmoNet
    # run_names += ['2021_03_18_21-10-25_DECA_training'] # Basic DECA
    # run_names += ['2021_03_26_15-05-56_DECA__DeSegFalse_DwC_early'] # Detail with coarse
    # run_names += ['2021_03_26_14-36-03_DECA__DeSegFalse_DeNone_early'] # No detail exchange

    # aff-wild 2 fintuned models
    # run_names += ['2021_04_02_18-46-31_va_DeSegFalse_Aug_early'] # DECA
    # run_names += ['2021_04_02_18-46-47_va_EmoLossB_F2VAEw-0.00150_DeSegFalse_Aug_early'] # DECA with EmoNet
    # run_names += ['2021_04_02_18-46-34_va_DeSegFalse_Aug_DwC_early'] # DECA detail with coarse
    # run_names += ['2021_04_02_18-46-51_va_DeSegFalse_DeNone_Aug_DwC_early'] # DECA detail with coarse , no exchange

    ## ExpDeca ablations
    ## Deca set
    # run_names += ['2021_04_19_18-57-31_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrend_early'] # ran
    # run_names += ['2021_04_19_18-57-50_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrendNoLmk_DwC_early'] # ran
    # run_names += ['2021_04_19_18-57-53_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrend_DwC_early'] # ran
    # run_names += ['2021_04_19_18-58-27_ExpDECA_DecaD_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrendNoLmk_DwC_early'] # ran
    # run_names += ['2021_04_20_19-09-37_ExpDECA_DecaD_EmoTrain_Jaw_NoRing_DeSegrend_early'] # ran
    # run_names += ['2021_04_20_19-09-44_ExpDECA_DecaD_clone_Jaw_NoRing_DeSegrend_early'] # ran
    # run_names += ['2021_04_20_19-10-57_ExpDECA_DecaD_para_Jaw_DeSegrend_early'] ## ran
    # run_names += ['2021_04_19_18-55-46_ExpDECA_DecaD_EmoStat_Jaw_NoRing_DeSegrend_early'] # ran
    ## run_names += ['2021_04_23_10-35-48_ExpDECA_DecaD_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_early'] ## stil running

    # # AffectNet
    # run_names += ['2021_04_19_18-58-39_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_early'] # ran
    # run_names += ['2021_04_19_18-58-40_ExpDECA_Affec_para_Jaw_NoRing_DeSegrendNoLmk_DwC_early'] # ran
    # run_names += ['2021_04_19_18-59-02_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_early'] ### ran
    # run_names += ['2021_04_19_18-59-19_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early'] # ran
    # run_names += ['2021_04_19_19-04-35_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_DwC_early'] ## ran
    # run_names += ['2021_04_19_19-05-43_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrendNoLmk_DwC_early'] # ran
    # run_names += ['2021_04_20_21-39-59_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_early']  ### ran
    # run_names += ['2021_04_20_18-36-33_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early']  ### ran
    # run_names += ['2021_04_19_18-56-08_ExpDECA_Affec_EmoStat_Jaw_NoRing_DeSegrend_early']  ## ran

    ### DECA no ring ablations
    # # Deca set
    # run_names += ['2021_04_23_17-06-29_ExpDECA_DecaD_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early']
    # run_names += ['2021_04_23_17-05-49_ExpDECA_DecaD_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_early']
    # run_names += ['2021_04_23_17-00-40_ExpDECA_DecaD_NoRing_DeSegrend_early']
    # run_names += ['']

    # AffectNet
    # run_names += ['2021_04_23_17-12-20_DECA_Affec_NoRing_DeSegrend_DwC_early']  # ran
    # run_names += ['2021_04_23_17-12-05_DECA_Affec_NoRing_DeSegrend_early']  # ran
    # run_names += ['2021_04_23_17-11-08_DECA_Affec_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early']  # ran
    # run_names += ['2021_04_23_17-10-53_DECA_Affec_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_early'] # ran



    path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'

    ### ExpDECA expression rings with geometric losses
    # run_names += ['2021_05_07_20-48-30_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_DwC_early']
    # run_names += ['2021_05_07_20-46-09_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_early']
    # run_names += ['2021_05_07_20-45-33_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_DwC_early']
    # run_names += ['2021_05_07_20-36-43_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_early']

    ### ExpDECA expression rings without geometric losses
    # run_names += ['2021_05_02_12-43-06_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_early'] # ran
    # run_names += ['2021_05_02_12-42-01_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_DwC_early'] # ran
    # run_names += ['2021_05_02_12-37-20_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_DwC_early'] # ran
    # run_names += ['2021_05_02_12-36-00_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_DwC_early'] # ran
    # run_names += ['2021_05_02_12-35-44_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_early'] # ran
    # run_names += ['2021_05_02_12-34-47_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_early'] # ran

    ### ExpDECA with EmoMLP
    # run_names += ['2021_05_21_15-44-46_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.1_early']
    # run_names += ['2021_05_21_15-44-48_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.5_early']
    # run_names += ['2021_05_21_15-44-49_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.05_early']
    # run_names += ['2021_05_21_15-44-45_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.005_early']

    ### ExpDECA DwC with EmoMLP
    # run_names += ['2021_05_24_12-22-17_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.1_DwC_early']
    run_names += ['2021_05_24_12-22-17_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.05_DwC_early']
    # run_names += ['2021_05_24_12-22-21_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.005_DwC_early']
    # run_names += ['2021_05_24_12-21-45_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.5_DwC_early']

    codes_to_exchange = ['detailcode', 'expcode', 'jawpose']
    # codes_to_exchange = ['expcode', 'jawpose']

    for run_name in run_names:
        run_path = Path(path_to_models) / run_name
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)
        submit(conf, run_path, codes_to_exchange)


if __name__ == "__main__":
    main()

