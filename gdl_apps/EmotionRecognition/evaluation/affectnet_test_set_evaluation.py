"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""


from gdl.utils.condor import execute_on_cluster
from pathlib import Path
from gdl_apps.EmotionRecognition.training import train_emodeca
import datetime
from omegaconf import OmegaConf
import time as t
import random
from wandb import Api

def submit(cfg,
           resume_folder,
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
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[2].name / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name


    submission_folder_local.mkdir(parents=True)

    config_file  = submission_folder_local / "cfg.yaml"
    with open(config_file, 'w') as f:
         OmegaConf.save(cfg, f)

    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg.learning.gpu_memory_min_gb * 1024
    gpu_mem_requirement_mb = 15 * 1024
    gpu_mem_requirement_mb_max = 35 * 1024
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

    start_from_previous = 1
    force_new_location = 1
    stage = 1
    project_name = "AffectNetEmoDECATest"
    args = f" {config_file.name} {stage} {start_from_previous} {force_new_location} {project_name} "

    execute_on_cluster(str(cluster_script_path),
                       args,
                       str(submission_folder_local),
                       str(submission_folder_cluster),
                       str(cluster_repo_path),
                       env='work36_cu11',
                       python_bin=python_bin,
                       username=username,
                       gpu_mem_requirement_mb=gpu_mem_requirement_mb,
                       gpu_mem_requirement_mb_max=gpu_mem_requirement_mb_max,
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
    bid = 100
    stage = 1 # test stage
    api = Api()

    # comparison methods
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-39-36_-1044078422889696991_EmoDeep3DFace_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-39-32_6108539290469616315_EmoDeep3DFace_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_14-18-44_-5607778736970990207_Emo3DDFA_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_14-08-18_15578703095531241_Emo3DDFA_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_14-08-15_-7029531744226117801_Emo3DDFA_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_14-07-53_4456762721214245215_Emo3DDFA_shake_samp-balanced_expr_early"]

    ###resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-39-09_-5731619448091644006_EmoMGCNet_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-39-23_-1563842771012871107_EmoMGCNet_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-40-47_788901720705055085_EmoExpNet_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-39-22_7185746630127973131_EmoExpNet_shake_samp-balanced_expr_early"]

    # image based
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_20-48-55_-7323345455363258885_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_05-15-38_-8198495972451127810_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-12-56_7559763461347220097_EmoNet_shake_samp-balanced_expr_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-05-57_1011354483695245068_EmoSwin_swin_tiny_patch4_window7_224_shake_samp-balanced_expr_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-04-01_-3592833751800073730_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-02-49_-1360894345964690046_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early"]

    # # some of the best candidates
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_12_19-56-13_704003715291275370_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_15-42-30_8680779076656978317_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_12-13-16_-8024089022190881636_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_01-59-07_-9007648997833454518_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_11_01-58-56_1043302978911105834_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-58-31_-7948033884851958030_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-58-27_-5553059236244394333_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-57-28_-4957717700349337532_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-34-49_8015192522733347822_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-33-02_-5975857231436227431_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-33-00_-1889770853677981780_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-32-49_-6879167987895418873_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]

    # TODO: some of the EMOCA dataset trained ExpDECAs are missing ablations are missing
    # resume_folders += [""]
    # resume_folders += [""]
    # resume_folders += [""]
    # resume_folders += [""]

    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_20-16-17_-5054515021187918905_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-34-37_-2191187378900156758_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-33-58_6855768109315710901_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-33-58_4842056703157202150_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-33-29_-1963146520759335803_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-33-14_4337140264808142166_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-33-14_-168201693333588918_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-32-57_5584855956880651498_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]

    # original EMOCA

    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-14-55_3882362656686027659_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-14-55_-6806554933314077525_EmoDECA_Affec_Orig_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]

    # run these when jobs finish
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_22-44-04_8688670044855202446_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_22-43-49_3394193947032742938_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_22-43-48_-2000805732683960480_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_22-43-49_-4830182881615106196_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_22-41-49_-4075039212061619119_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_22-41-42_7906195635813312189_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_22-41-42_-1060774189444035688_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]

    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_23-27-01_6878473137680141500_EmoExpNet_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_23-27-01_1047316684664543951_EmoExpNet_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_23-26-36_7200280701837784842_EmoExpNet_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_23-26-22_-1905009454407858005_EmoExpNet_shake_samp-balanced_expr_early"]

    # para ExpDECA wit Emonet sweep

    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_00-46-21_-3135770645448281419_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_00-45-51_2411712055369009825_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_00-45-48_4945489383177306533_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_00-45-46_6073303847436754510_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_00-45-47_-6635553845380560579_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_00-45-43_5755972502667511983_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_00-45-43_1188039325412337929_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # run_names += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_00-45-43_-6861660935874010707_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]

    # # detailed EmoDECA (with finetuned detail) TODO: run when ready - careful, most of them actually turned NaN
    # ran
    #
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-40_8347105671606189869_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-37_9207638064621644556_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-36_4236836109803982115_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-31_1777376976408189062_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-31_-8190793573362684320_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-30_6980015522888660644_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-26_7466909277650738450_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-26_-7775568986167485599_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-24_283650042734872595_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-00-25_-8235832594847732343_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_01-04-42_6108428734734499933_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_01-04-42_-3074281716798214244_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_01-04-42_-8668091520165208464_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_01-04-42_6310542865318504681_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_01-04-42_-758267559414284479_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # # run_names += [
    # #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_01-04-42_-2347747697855330620_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-10-16_-6893288210826685729_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]

    # forgotten one:
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-33-08_2929045501486288941_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]

    # no emo ablations
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-56-06_7354625454882033560_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-51-51_7269374409381363675_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]

    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_16_16-22-28_-517043186899879900_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_16_16-22-18_2828871051475683460_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_16_16-22-13_1648771539220881707_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_16_16-22-11_-2838029234690005150_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_16_16-22-01_-8730011397142988359_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]
    # resume_folders += [
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_16_16-21-53_3503787604916481104_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"]

    # resume_folders += [ # lmk
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-56-58_-4794736824102217101_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"
    # ]

    # resume_folders += [ # no mouth corner
    #     "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_20-57-10_-5464385799477569099_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"
    # ]
    #

    # resume_folders += [  # final model with detail
    #         "/is/cluster/work/rdanecek/emoca/emodeca/2022_01_29_22-13-02_1880637297624421180_EmotionRecognition_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early" ]
    # resume_folders += [  # final model without detail (to sanity-check the one above)
    #         "/is/cluster/work/rdanecek/emoca/emodeca/2022_01_29_21-59-07_3822537833951552856_EmotionRecognition_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early" ]

    # final model with detail for cam ready
    resume_folders += ["/is/cluster/work/rdanecek/emoca/emodeca/2022_03_09_01-38-30_-3310155981496657988_EmotionRecognition_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_detail_shake_samp-balanced_expr_early"]

    # submit_ = False
    submit_ = True

    for resume_folder in resume_folders:
        name = str(Path(resume_folder).name)
        idx = name.find("Emo")
        run_id = name[:idx-1]
        run = api.run("rdanecek/EmoDECA/" + run_id)
        tags = set(run.tags)

        allowed_tags = set(["COMPARISON", "INTERESTING", "FINAL_CANDIDATE", "BEST_CANDIDATE", "BEST_IMAGE_BASED", "FINAL_CANDIDATE_REBUTTAL"])

        if len(allowed_tags.intersection(tags)) == 0:
            print(f"Run '{name}' is not tagged to be tested and will be skipped.")
            continue

        cfg = OmegaConf.load(Path(resume_folder) / "cfg.yaml")

        cfg.data.data_class = "AffectNetEmoNetSplitModuleTest"


        if submit_:
            submit(cfg, stage, bid=bid)
        else:
            stage = 1
            start_from_previous = 1
            force_new_location = 1
            train_emodeca.train_emodeca(cfg, stage, start_from_previous, force_new_location, "AffectNetEmoDECATest")


if __name__ == "__main__":
    train_emodeca_on_cluster()

