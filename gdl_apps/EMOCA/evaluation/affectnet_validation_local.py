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


from affectnet_validation import *
from gdl_apps.EMOCA.utils.load import load_model


def main():
    # path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    #
    path_to_affectnet = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"
    # path_to_processed_affectnet = "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/"

    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    # path_to_affectnet = "/ps/project/EmotionalFacialAnimation/data/affectnet/"
    # path_to_processed_affectnet = "/ps/scratch/rdanecek/data/affectnet/"
    path_to_processed_affectnet = "/is/cluster/work/rdanecek/data/affectnet/"

    run_names = []
    # run_names += ['2021_03_25_19-42-13_DECA_training'] # EMOCA EmoNet
    # run_names += ['2021_03_29_23-14-42_DECA__EmoLossB_F2VAEw-0.00150_DeSegFalse_early'] # EMOCA EmoNet 2
    # run_names += ['2021_03_18_21-10-25_DECA_training'] # Basic EMOCA
    run_names += ['2021_03_26_15-05-56_DECA__DeSegFalse_DwC_early'] # Detail with coarse
    # run_names += ['2021_03_26_14-36-03_DECA__DeSegFalse_DeNone_early'] # No detail exchange
    # run_names += ['2021_05_21_15-44-45_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.005_early'] # EMOCA MLP
    # run_names += ['2021_06_01_15-02-35_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_1.0_detJ_DwC_early/'] # EMOCA MLP

    mode = 'detail'
    # mode = 'coarse'

    for run_name in run_names:
        print(f"Beginning testing for '{run_name}' in mode '{mode}'")
        # relative_to_path = '/ps/scratch/'
        # replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'

        relative_to_path = None
        replace_root_path = None

        deca, conf = load_model(path_to_models, run_name, mode, relative_to_path, replace_root_path)
        # deca, conf = load_model(path_to_models, run_name, mode)

        run_name = conf[mode].inout.name

        deca.deca.config.resume_training = True
        deca.deca.config.pretrained_modelpath = "/home/rdanecek/Workspace/Repos/DECA/data/deca_model.tar"
        deca.deca._load_old_checkpoint()
        run_name = "Original_DECA"

        deca.eval()

        import datetime
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        conf[mode].inout.random_id = str(hash(time))
        conf[mode].learning.logger_type = None
        conf['detail'].learning.logger_type = None
        conf['coarse'].learning.logger_type = None
        # conf['detail'].data.root_dir = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"
        # conf['coarse'].data.root_dir = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"
        # conf[mode].data.root_dir = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"

        dm = data_preparation_function(conf[mode], path_to_affectnet, path_to_processed_affectnet)
        conf[mode].model.test_vis_frequency = 1
        # conf[mode].inout.name = "affectnet_test_" + conf[mode].inout.name
        conf[mode].inout.name = "afft_" + run_name
        # conf[mode].inout.name = "Original_DECA"
        single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_net", dm=dm, project_name_="AffectNetTests")
        print("We're done y'all")


if __name__ == '__main__':
    main()
