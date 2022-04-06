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


import os, sys
from pathlib import Path

# import NoW related stuff
# sys.path += [str(Path(__file__).absolute().parents[3] / "now_evaluation")]
sys.path.insert(0, str(Path(__file__).absolute().parents[3] / "now_evaluation"))
from main import generating_cumulative_error_plots


def main():
    # stage = 'detail'
    stage = 'coarse'


    run_names = {}
    run_names['Original_DECA'] = ["Paper EMOCA", "detail"] # Original EMOCA from Yao
    # run_names['2021_03_18_21-10-25_DECA_training'] = "Basic EMOCA (trained by me)" # Basic EMOCA
    # run_names['2021_03_25_19-42-13_DECA_training'] = "My retrained with EmoNet loss" # EMOCA EmoNet
    # run_names['2021_03_29_23-14-42_DECA__EmoLossB_F2VAEw-0.00150_DeSegFalse_early'] = "Deca with EmoNetLoss" # EMOCA EmoNet
    # run_names['2021_03_26_15-05-56_DECA__DeSegFalse_DwC_early'] = "Detail with coarse jointly" # Detail with coarse
    # run_names['2021_03_26_14-36-03_DECA__DeSegFalse_DeNone_early'] = "EMOCA no detail exchange" # No detail exchange

    # aff-wild 2 models
    # run_names['2021_04_02_18-46-31_va_DeSegFalse_Aug_early'] = "AffWild EMOCA" # EMOCA
    # run_names['2021_04_02_18-46-47_va_EmoLossB_F2VAEw-0.00150_DeSegFalse_Aug_early'] = "AffWild with EmoNetLoss" # EMOCA with EmoNet
    # run_names['2021_04_02_18-46-34_va_DeSegFalse_Aug_DwC_early'] # EMOCA detail with coarse
    # run_names['2021_04_02_18-46-51_va_DeSegFalse_DeNone_Aug_DwC_early'] # EMOCA detail with coarse , no exchange


    # ### no-RING DECAs
    # # EMOCA dataset
    # run_names['2021_04_23_17-06-29_ExpDECA_DecaD_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early'] =\
    #     'DECA_DecaD_NoRing_EmoLossB_DwC'
    # run_names['2021_04_23_17-05-49_ExpDECA_DecaD_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_early'] = \
    #     'DECA_DecaD_NoRing_EmoLossB'
    # run_names['2021_04_23_17-00-40_ExpDECA_DecaD_NoRing_DeSegrend_early'] = \
    #     'DECA_DecaD_NoRing'
    # # # run_names += ['']
    # #
    # # # # AffectNet
    # run_names['2021_04_23_17-12-20_DECA_Affec_NoRing_DeSegrend_DwC_early'] = \
    #     'DECA_Affec_NoRing_DwC'
    # run_names['2021_04_23_17-12-05_DECA_Affec_NoRing_DeSegrend_early'] = \
    #     'DECA_Affec_NoRing'
    # run_names['2021_04_23_17-11-08_DECA_Affec_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early'] = \
    #     'DECA_Affec_NoRing_EmoLossB_DwC'
    # run_names['2021_04_23_17-10-53_DECA_Affec_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_early'] = \
    #     'DECA_Affec_NoRing_EmoLossB'

    run_names_new = {}
    # run_names_news['2021_06_23_21-03-02_DECA__EFswin_s_EFswin_s_DeSegFalse_early'] =[ "SWIN S", "detail"]# EMOCA EmoNet
    # run_names_new['2021_06_23_21-03-46_DECA__EFswin_t_EFswin_t_DeSegFalse_early'] = ["SWIN T", "detail"]
    run_names_new['2021_06_24_10-44-02_DECA__DeSegFalse_early'] = ["EMOCA v1" , "detail"]# EMOCA EmoNets
    # run_names_new['2021_08_29_00-38-20_DECA_DecaD_DeSegrend_Deex_early'] = "EMOCA v2" # EMOCA EmoNet
    # run_names_new['2021_08_29_10-28-11_DECA_DecaD_VGGl_DeSegrend_Deex_early'] = "EMOCA with VGG loss"
    # run_names_new['2021_08_29_10-31-15_DECAStar_DecaD_VGGl_DeSegrend_Deex_early'] = "EMOCA SWIN with VGG loss"
    # run_names_new['2021_08_29_00-42-34_DECAStar_DecaD_DeSegrend_Deex_early'] = "DECAStar"
    run_names_new["2021_08_29_00-49-03_DECA_DecaD_EFswin_t_EDswin_t_DeSegrend_Deex_early"] = ["EMOCA SWINT T", "detail"]
    # run_names_new["2021_08_29_00-48-58_DECAStar_DecaD_EFswin_t_EDswin_t_DeSegrend_Deex_early"] = "DECAStar SWINT T"
    # run_names_new["2021_08_29_19-47-21_DECA_DecaD_EFswin_s_EDswin_s_VGGl_DeSegrend_Deex_early"] = "EMOCA SWIN-S + VGG loss"
    # run_names_new["2021_08_29_19-47-28_DECAStar_DecaD_EFswin_s_EDswin_s_VGGl_DeSegrend_Deex_early"] = ["DECAStar SWIN-S + VGG loss", "detail"]

    # run_names_new["2021_10_08_18-59-03_DECA_DecaD_NoRing_VGGl_DeSegrend_idBTH-s10000_Aug_early"] = ["BTH late", "coarse"]
    run_names_new["2021_10_08_18-25-12_DECA_DecaD_NoRing_VGGl_DeSegrend_idBTH_Aug_early"]  = ["BTH", "coarse"]
    run_names_new["2021_10_08_16-40-04_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT-ft_Aug_early"]  = ["BT finetune id", "coarse"]
    # run_names_new["2021_10_08_16-40-04_DECA_DecaD_NoRing_VGGl_DeSegrend_idBTH-ft_Aug_early"]  = ["BTH finetune id", "coarse"]
    # run_names_new[
        # "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-41-54_DECA_DecaD_NoRing_VGGl_EmoB_EmoCnn_vgg_du_F2bar_DeSegrend_idBTH_Aug_early"]
    # run_names_new[
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-41-51_DECA_DecaD_NoRing_VGGl_EmoB_EmoCnn_vgg_du_F2_DeSegrend_l1_loss_Aug_early"]
    # run_names_new[
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-41-45_DECA_DecaD_NoRing_VGGl_EmoB_EmoCnn_vgg_du_F2bar_DeSegrend_idBT_Aug_early"]
    # run_names_new[
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-41-23_DECA_DecaD_NoRing_VGGl_EmoB_EmoCnn_vgg_du_F2cos_DeSegrend_cosine_similarity_Aug_early"]
    run_names_new["2021_10_08_12-39-18_DECA_DecaD_NoRing_VGGl_DeSegrend_cosine_similarity_Aug_early"] = ["cos id", "coarse"]
    # run_names_new[
    #     "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_08_12-39-06_DECA_DecaD_NoRing_VGGl_DeSegrend_l1_loss_Aug_early"]
    # run_names_new["2021_10_08_12-39-04_DECA_DecaD_NoRing_VGGl_DeSegrend_idBTH_Aug_early"] =  ["BTH", "coarse"]
    # run_names_new["2021_10_08_12-38-50_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT_Aug_early"]  =  ["BT", "coarse"]
    #
    # run_names_new["2021_10_11_10-48-59_DECA_DecaD_NoRing_VGGl_DeSegrend_cosine_similarity_Aug_early"] =  ["cos 0.5", "coarse"]
    # run_names_new["2021_10_11_10-48-48_DECA_DecaD_NoRing_VGGl_DeSegrend_cosine_similarity_Aug_early"] =  ["cos 1", "coarse"]
    # run_names_new["2021_10_11_10-48-46_DECA_DecaD_NoRing_VGGl_DeSegrend_cosine_similarity_Aug_early"] =  ["cos 0.3", "coarse"]
    # run_names_new["2021_10_11_10-48-37_DECA_DecaD_NoRing_VGGl_DeSegrend_cosine_similarity_Aug_early"] =  ["cos 0.1", "coarse"]
    # run_names_new["2021_10_11_10-48-33_DECA_DecaD_NoRing_VGGl_DeSegrend_cosine_similarity_Aug_early"] =  ["cos 0.05", "coarse"]
    # run_names_new["2021_10_11_10-48-32_DECA_DecaD_NoRing_VGGl_DeSegrend_cosine_similarity_Aug_early"] =  ["cos 0", "coarse"]
    # run_names_new["2021_10_10_21-01-48_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT_Aug_early"] =  ["BT 0.3", "coarse"]
    # run_names_new["2021_10_10_21-01-38_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT_Aug_early"] =  ["BT 0.1", "coarse"]
    # # run_names_new["2021_10_10_21-01-32_DECA_DecaD_NoRing_VGGl_DeSegrend_l1_loss_Aug_early"]=  ["l1 0.3", "coarse"]
    # # run_names_new["2021_10_10_21-01-19_DECA_DecaD_NoRing_VGGl_DeSegrend_l1_loss_Aug_early"]=  ["l1 0.1", "coarse"]
    # # run_names_new["2021_10_10_20-58-06_DECA_DecaD_NoRing_VGGl_DeSegrend_l1_loss_Aug_early"]=  ["l1 1", "coarse"]
    # # run_names_new["2021_10_10_20-57-57_DECA_DecaD_NoRing_VGGl_DeSegrend_l1_loss_Aug_early"]=  ["l1 0.5", "coarse"]
    # # run_names_new["2021_10_10_20-57-36_DECA_DecaD_NoRing_VGGl_DeSegrend_l1_loss_Aug_early"]=  ["l1 0.05", "coarse"]
    # run_names_new["2021_10_10_20-57-27_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT_Aug_early"] =  ["BT 1", "coarse"]
    # run_names_new["2021_10_10_20-57-18_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT_Aug_early"] =  ["BT 0.5", "coarse"]
    # run_names_new["2021_10_10_20-57-17_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT_Aug_early"] =  ["BT 0.05", "coarse"]

    # run_names_new[2021_10_13_10-49-17_DECA_DecaD_VGGl_DeSegrend_idBT-ft-cont_Deex_early"] =  ["BT 1", "coarse"]
    # run_names_new["2021_10_10_20-57-18_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT_Aug_early"] =  ["BT 0.5", "coarse"]
    # run_names_new["2021_10_10_20-57-17_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT_Aug_early"] =  ["BT 0.05", "coarse"]

    run_names_new["2021_10_13_10-49-29_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT-ft_early"] =  ["BT finetune id v2 - on diag norm", "coarse"]
    run_names_new["2021_10_12_22-52-16_DECA_DecaD_VGGl_DeSegrend_idBT-ft-cont_Deex_early"] =  ["BT contrastive, ring 2, 0.2", "coarse"]
    run_names_new["2021_10_12_22-05-00_DECA_DecaD_VGGl_DeSegrend_idBT-ft-cont_Deex_early"] =  ["BT contrastive, ring 2, 0.3", "coarse"]
    run_names_new["2021_10_15_13-32-33_DECA__DeSegFalse_early"] =  ["EMOCA, large batch", "coarse"]


    use_dense_topology = False
    # use_dense_topology = True

    path_to_old_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    path_to_new_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'


    run_files = []
    nicks = []
    path_to_models = path_to_old_models
    for run_name, nick in run_names.items():
        nick, stage = nick
        if use_dense_topology:
            savefolder = Path(path_to_models) / run_name / stage / "NoW_dense"
        else:
            savefolder = Path(path_to_models) / run_name / stage / "NoW_flame"

        run_files += [str(savefolder / "results" / "_computed_distances.npy")]
        nicks += [nick]

    path_to_models = path_to_new_models
    for run_name, nick in run_names_new.items():
        nick, stage = nick
        try:
            if use_dense_topology:
                savefolder = Path(path_to_models) / run_name / stage / "NoW_dense"
            else:
                savefolder = Path(path_to_models) / run_name / stage / "NoW_flame"
        except:
            if use_dense_topology:
                savefolder = Path(path_to_models) / run_name / 'coarse' / "NoW_dense"
            else:
                savefolder = Path(path_to_models) / run_name / 'coarse' / "NoW_flame"
        run_files += [str(savefolder / "results" / "_computed_distances.npy")]
        nicks += [nick]

    generating_cumulative_error_plots(run_files, nicks, "out.png")


if __name__ == "__main__":
    main()
