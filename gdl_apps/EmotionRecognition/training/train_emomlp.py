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



from train_emodeca import *

project_name = 'EmotionRecognition'



def main():
    if len(sys.argv) < 2:

        deca_default = None
        deca_overrides = None
        # deca_conf_path = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca/2021_04_19_18-59-19_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"
        # # deca_conf_path = "/run/user/1001/gvfs/smb-share:server=ps-access.is.localnet,share=scratch/rdanecek/emoca/finetune_deca/2021_04_19_18-59-19_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"
        deca_conf_path = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_29_21-30-28_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"
        # # emoca_conf = None
        stage = 'detail'
        #
        # relative_to_path = '/ps/scratch/'
        # replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
        relative_to_path = None
        replace_root_path = None


        ## EmoMGCNET
        emodeca_default = "emomgcnet"
        emodeca_overrides = [
            # 'model.mlp_dim=2048',
            # 'data/datasets=emotionet_desktop',
            # 'data.data_class=AffectNetEmoNetSplitModuleValTest',
            '+data.dataset_type=AffectNetWithMGCNetPredictions',
            'data.num_workers=0',
            # 'data.num_workers=16',
            'learning/logging=none',
        ]

        ## EmoExpNET
        # emodeca_default = "emoexpnet"
        # emodeca_overrides = [
        #     # 'model.mlp_dim=2048',
        #     # 'data/datasets=emotionet_desktop',
        #     # 'data.data_class=AffectNetEmoNetSplitModuleValTest',
        #     # '+data.dataset_type=AffectNetWithExpNetPredictions',
        #     '+data.dataset_type=AffectNetWithExpNetPredictionsMyCrop',
        #     'data.num_workers=0',
        #     # 'data.num_workers=16',
        #     'learning/logging=none',
        # ]
        deca_conf = None
        deca_conf_path = None
        fixed_overrides_deca = None
        stage = None
        deca_default = None
        deca_overrides = None


        cfg = configure(emodeca_default,
                        emodeca_overrides,
                        deca_default,
                        deca_overrides,
                        deca_conf_path=deca_conf_path,
                        deca_stage=stage,
                        relative_to_path=relative_to_path,
                        replace_root_path=replace_root_path)
        start_stage = -1
        start_from_previous = False
        force_new_location = False
    else:
        cfg_path = sys.argv[1]
        print(f"Training from config '{cfg_path}'")
        with open(cfg_path, 'r') as f:
            cfg = OmegaConf.load(f)
        start_stage = -1
        start_from_previous = False
        force_new_location = False
        if len(sys.argv) > 2:
            start_stage = int(sys.argv[2])
        if len(sys.argv) > 3:
            force_new_location = bool(int(sys.argv[3]))
        if start_stage == 1:
            start_from_previous = True

    project_name_ = "EmoDECATest"

    train_emodeca(cfg, start_stage, project_name_=project_name_, resume_from_previous=start_from_previous,
                  force_new_location=force_new_location)


if __name__ == "__main__":
    main()
