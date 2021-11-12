from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import resume_expdeca_training
import datetime
from omegaconf import OmegaConf
import time as t
import random

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
    submission_folder_name = time + "_" + str(hash(random.randint(0,1000000))) + "_" + "submission"
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
    cuda_capability_requirement = 7
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
    # t.sleep(1)


def main():
    stage = 0
    resume_from_previous = False
    force_new_location = False

    resume_folders = []
    # resume_folders += ['2021_04_13_19-02-10_ExpDECA__EmoTrain_Jaw_DeSegrend_early'] # has been resumed, will crash on detail, will have to be resumed again
    # resume_folders += ['2021_04_13_19-02-11_ExpDECA__EmoStat_Jaw_DeSegrend_early'] # has been resumed, will crash on detail, will have to be resumed again
    # resume_folders += ['2021_04_13_19-02-31_ExpDECA__clone_Jaw_DeSegrend_early'] # has been resumed, will crash on detail, will have to be resumed again
    # resume_folders += ['2021_04_13_18-54-34_ExpDECA__para_Jaw_DeSegrend_early'] # has been resumed, will crash on detail, will have to be resumed again
    # resume_folders += ['2021_04_13_18-47-00_ExpDECA__para_DeSegrend_early'] # has been resumed, will crash on detail, will have to be resumed again, was resumed again
    # resume_folders += ['2021_04_13_18-39-24_ExpDECA__para_Jaw_DeSegrend_early'] # has been resumed, will crash on detail, will have to be resumed again
    # resume_folders += ['2021_04_14_14-00-16_ExpDECA__para_Jaw_NoRing_DeSegrend_CoNone_DeNone_early']
    # resume_folders += ['2021_04_14_17-39-46_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_early']
    # resume_folders += ['2021_04_14_18-10-36_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_Aug_early']
    # resume_folders += ['2021_04_14_22-49-50_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_early']
    # resume_folders += ['2021_04_14_23-02-10_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_early']

    # resume_folders += ['2021_04_30_21-06-06_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_DwC_early']
    # resume_folders += ['2021_04_30_21-08-49_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_early']
    # resume_folders += ['2021_04_30_20-57-40_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_early']
    # resume_folders += ['2021_04_30_20-59-38_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_DwC_early']
    # resume_folders += ['2021_04_30_21-05-23_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_early']
    # resume_folders += ['2021_04_30_21-01-13_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_DwC_early']
    # # resume_folders += ['']

    # EmoResNet ExpDECA sweep
    # resume_folders += [
    #     "2021_11_02_12-41-24_7057622275122671174_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-41-06_-6506673705064889607_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-41-06_-1007531484471246016_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-41-03_480128111237298530_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-41-03_-3847743713390055217_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-41-01_7226661150207254923_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-41-01_7193545667483921831_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-41-01_-7746686909198123775_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-59_-6970716391423648964_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-59_-4293993865315558856_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-54_-8956728687580574108_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-52_3535801695749609832_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-52_-776769225150723181_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-48_-3557093149321491446_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-48_-1130509047528431540_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-45_6758302146806216456_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-33_8529519700345615983_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-33_7640188424869169886_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-27_4344460465829536839_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-27_6877124675180108840_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-03_-8436446076366773310_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-03_-4636033309105620245_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-02_833576158064688874_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-40-02_8285199837830669798_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # # EmSwin ExpDECA sweep
    # resume_folders += [
    #     "2021_11_03_02-20-54_-1968661455773213379_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_03_02-42-36_2407313258403191383_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_13-12-49_-2741982989276466203_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_13-10-49_1731595146375171932_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_13-10-49_-7667543226652993592_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_13-10-40_6872394600987091012_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_13-10-40_-8572124953572605249_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_13-10-40_-7473769445402844399_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_13-10-40_-5241005287738579855_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-54-01_2637759665938415282_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-47-35_5168561227047398084_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-47-35_2319744141436125537_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-47-35_2073066276032009236_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-47-35_-6306367650010438382_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-47-35_-3218674826605317504_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-47-35_-1846739961689335557_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-47-35_-1022988189955888024_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-42-55_5806971874713117653_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-42-55_-5348873875193364241_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-42-54_-7598613731487617091_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-42-45_5490409369290264125_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-42-45_1092553962037855966_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-42-43_2271671740894586800_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-42-43_-5593491350755409121_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-42-41_4316282956709408142_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "2021_11_02_12-42-41_-1213070571142271333_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    # final ExpDECA ablations on DECA dataset

    resume_folders += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-56-46_5920957646486902084_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    resume_folders += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-56-39_-8971851772753744759_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-38_1354461056444555550_ExpDECA_DecaD_para_NoRing_DeSegrend_BlackB_Aug_early"]
    resume_folders += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-41_7798762876288315974_ExpDECA_DecaD_clone_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]
    # ##resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-32_-428770426719310834_ExpDECA_DecaD_para_NoRing_DeSegrend_BlackB_Aug_early"]
    resume_folders += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-28_6450489661335316335_ExpDECA_DecaD_clone_NoRing_DeSegrend_BlackB_Aug_early"]
    resume_folders += [
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-19_-698052302382081628_ExpDECA_DecaD_clone_NoRing_DeSegrend_BlackB_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-55-17_-6566800429279817771_ExpDECA_DecaD_clone_Jaw_NoRing_DeSegrend_BlackB_Aug_early"]

    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-08-55_-7847515130004126177_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-07-31_-2183917122794074619_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-22_-3360331398526735766_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-22_4582523459040385488_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-27_8115149509825457198_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_10_23-57-30_-5150018129787658113_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    #
    for resume_folder in resume_folders:
        submit(resume_folder, stage, resume_from_previous, force_new_location)

    # stage = 2
    # resume_from_previous = False
    # force_new_location = False
    #
    # resume_folders = []
    # # resume_folders += ['2021_04_18_13-43-32_ExpDECA_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_CoNone_DeNone_DwC_early'] # resumed
    # # resume_folders += ['2021_04_18_13-43-46_ExpDECA_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_CoNone_DeNone_DwC_early'] # resumed
    # # resume_folders += ['2021_04_18_13-43-55_ExpDECA_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_CoNone_DeNone_Aug_early'] # resumed
    # # resume_folders += ['2021_04_18_12-47-40_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_DwC_early'] # resumed


    # for resume_folder in resume_folders:
    #     submit(resume_folder, stage, resume_from_previous, force_new_location)

    # stage = 2
    # resume_from_previous = True
    # force_new_location = False

    resume_folders = []
    # resume_folders += ['2021_04_19_19-04-35_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_DwC_early'] # resumed
    # resume_folders += ['2021_04_20_19-10-57_ExpDECA_DecaD_para_Jaw_DeSegrend_early'] # resumed
    # resume_folders += ['2021_04_23_17-11-08_DECA_Affec_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early']  # resumed

    # em-MLP models
    # resume_folders += ['2021_05_17_01-24-26_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.0015_DwC_early']
    # resume_folders += ['2021_05_17_01-22-52_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.0015_early']
    # resume_folders += ["2021_10_26_15-08-44_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += ["2021_10_26_13-15-16_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += ["2021_10_26_12-35-21_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]
    # resume_folders += ["2021_10_26_12-29-38_ExpDECA_DecaD_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]


    # for resume_folder in resume_folders:
    #     submit(resume_folder, stage, resume_from_previous, force_new_location)
    #
    # stage = 2
    # resume_from_previous = False
    # force_new_location = False
    # resume_folders = []
    # resume_folders += ['2021_05_02_12-43-06_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_early']
    # resume_folders += ['2021_05_02_12-42-01_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_DwC_early']
    # resume_folders += ['2021_05_02_12-37-20_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_DwC_early']
    # resume_folders += ['2021_05_02_12-36-00_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_DwC_early']
    # resume_folders += ['2021_05_02_12-35-44_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_early']
    # resume_folders += ['2021_05_02_12-34-47_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_early']

    ## expression ring,
    # resume_folders += ['2021_05_07_20-48-30_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_DwC_early']
    # resume_folders += ['2021_05_07_20-46-09_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_early']
    # resume_folders += ['2021_05_07_20-45-33_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_DwC_early']
    # resume_folders += ['2021_05_07_20-36-43_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_early']
    #

    ## emo MLP
    # resume_folders += ['2021_05_21_15-44-52_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_1.0_early']
    # resume_folders += ['2021_05_21_15-44-49_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.05_early']
    # resume_folders += ['2021_05_21_15-44-48_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.5_early']
    # resume_folders += ['2021_05_21_15-44-46_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.1_early']
    # resume_folders += ['2021_05_21_15-44-45_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.01_early']
    # resume_folders += ['2021_05_21_15-44-45_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.005_early']

    ## emo MLP DwC
    # resume_folders += ["2021_05_24_12-22-21_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.005_DwC_early"]
    # resume_folders += ["2021_05_24_12-22-17_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.05_DwC_early"]
    # resume_folders += ["2021_05_24_12-22-17_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.1_DwC_early"]
    # resume_folders += ["2021_05_24_12-21-45_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.5_DwC_early"]

    ## Emo MLP emotion ring, ge ex
    # resume_folders += ['2021_05_28_16-35-17_ExpDECA_Affec_para_Jaw_DeSegrend_BlackB_Exemonet_feature_CoNo_MLP_0.05_DwC_early']
    # resume_folders += ['2021_05_28_16-35-32_ExpDECA_Affec_para_Jaw_DeSegrend_BlackB_Exgt_va_CoNo_MLP_0.05_DwC_early'] # weird
    # resume_folders += ['2021_05_28_16-35-33_ExpDECA_Affec_para_Jaw_DeSegrend_BlackB_Exgt_expression_CoNo_MLP_0.05_DwC_early']

    ## Emo MLP emotion ring
    # resume_folders += ['2021_05_28_16-45-22_ExpDECA_Affec_para_Jaw_DeSegrend_BlackB_Exgt_expression_CoNo_MLP_0.05_DwC_early']
    # resume_folders += ['2021_05_28_16-45-34_ExpDECA_Affec_para_Jaw_DeSegrend_BlackB_Exgt_va_CoNo_MLP_0.05_DwC_early']
    # resume_folders += ['2021_05_28_16-46-05_ExpDECA_Affec_para_Jaw_DeSegrend_BlackB_Exemonet_feature_CoNo_MLP_0.05_DwC_early']

    ## ExpDECA emo ring emonet featue ge ex
    # resume_folders += ['2021_05_28_17-59-11_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_DeSegrend_BlackB_Exgt_va_CoNo_DwC_early']
    # resume_folders += ['2021_05_28_18-12-17_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exgt_va_CoNo_DwC_early']


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



    # for resume_folder in resume_folders:
    #     submit(resume_folder, stage, resume_from_previous, force_new_location)



if __name__ == "__main__":
    main()
