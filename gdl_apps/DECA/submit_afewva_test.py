from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import afew_va_test_new_split  as script
import datetime
from omegaconf import OmegaConf
import time as t
import random
import wandb

def submit(cfg, model_folder_name, mode, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    submission_dir_local_mount = "/is/cluster/work/rdanecek/emoca/affectnet_test_submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/affectnet_test_submission"
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(random.randint(0,10000000))) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(script.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    # gpu_mem_requirement_mb = cfg.detail.learning.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb = 14 * 1024
    # gpu_mem_requirement_mb = None
    # cpus = cfg.detail.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg.detail.learning.num_gpus
    num_jobs = 1
    max_time_h = 10
    max_price = 8000
    job_name = "finetune_deca"
    cuda_capability_requirement = 7
    mem_gb = 12
    args = f"{model_folder_name} {mode}"

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
                       # max_concurrent_jobs = 5,
                       max_concurrent_jobs = 20,
                       concurrency_tag = "afewva_test",
                       modules_to_load=['cuda/11.4'],
                       )
    # t.sleep(1)

def main():
    # path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'

    run_names = []

    # # DecaD ablation
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-56-46_5920957646486902084_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-56-39_-8971851772753744759_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-38_1354461056444555550_ExpDECA_DecaD_para_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-41_7798762876288315974_ExpDECA_DecaD_clone_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-32_-428770426719310834_ExpDECA_DecaD_para_NoRing_DeSegrend_BlackB_Aug_early"]
    #
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-28_6450489661335316335_ExpDECA_DecaD_clone_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-19_-698052302382081628_ExpDECA_DecaD_clone_NoRing_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-17_-6566800429279817771_ExpDECA_DecaD_clone_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    #
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-08-55_-7847515130004126177_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-07-31_-2183917122794074619_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-22_-3360331398526735766_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-22_4582523459040385488_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-27_8115149509825457198_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-30_-5150018129787658113_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # # unbalanced ExpDECA on Affecntet
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-27_7449334996109808959_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-07_-753452132482044016_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-07_-6499863499965279138_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-27-09_3536700504397748218_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]


    # # # important
    # # run_names = []
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-57-41_-5511487677556972267_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-57-41_1218762018464274311_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-57-41_6160996897661237206_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-07_-3310835230647295291_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-08_3278107752429068516_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-15-52_349713347846449814_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-15-52_8341774161001263236_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-16-26_2689968017949274893_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-35_4654975036132116438_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    #

    # final models - ran, check if worked
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-07_-2106219737797182304_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-08-55_-7847515130004126177_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_11_01-28-27_7449334996109808959_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-07-31_-2183917122794074619_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-08_3278107752429068516_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_21-34-35_4654975036132116438_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-15-52_349713347846449814_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-57-41_6160996897661237206_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-57-41_1218762018464274311_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_18-16-26_2689968017949274893_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # mode = 'coarse'

    # # TRYING EMODECA WITH detail
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-48-09_-5406564322738469847_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSegrend_BlackC_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-48-09_-4816207216138971302_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-47-58_403387054595674003_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSegrend_BlackC_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-47-58_1356413750339666012_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_-8338540563465925024_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSegrend_BlackC_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_-8022299573244041531_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSegrend_BlackC_Aug_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-36_187831196779018813_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"]
    #

    run_names += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-48-09_-5406564322738469847_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSegrend_BlackC_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-48-09_-4816207216138971302_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-47-58_403387054595674003_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSegrend_BlackC_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-47-58_1356413750339666012_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_-8338540563465925024_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSegrend_BlackC_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_-8022299573244041531_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSegrend_BlackC_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-36_187831196779018813_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"]
    mode = 'detail'


    tags = None

    bid = 100

    for run_name in run_names:

        api = wandb.Api()
        name = str(Path(run_name).name)
        idx = name.find("ExpDECA")
        run_id = name[:idx-1]
        run = api.run("rdanecek/EmotionalDeca/" + run_id)
        tags = run.tags
        # tags += ["NEW_SPLIT"]
        # fixed_overrides_cfg += [f"+learning.tags={ '['+'_'.join(tags)+ ']'}"]

        run_path = Path(path_to_models) / run_name
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)
        submit(conf, run_name, mode, bid=bid)


if __name__ == "__main__":
    main()

