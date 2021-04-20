from applications.DECA.train_expdeca import resume_training, load_configs
from applications.DECA.train_deca_modular import get_checkpoint_with_kwargs
from models.DECA import instantiate_deca
from applications.DECA.interactive_deca_decoder import load_deca_and_data, load_deca
import sys, os


def main():
    model_folder = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca/"

    # resume_from_1 = os.path.join(model_folder, "2021_04_18_12-39-46_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_DwC_early")
    # resume_from_2 = os.path.join(model_folder, "2021_04_18_12-39-46_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_DwC_early")
    # # resume_from_2 = sys.argv[1]
    # stage = 0
    # resume_from_previous = False
    # force_new_location = False
    #
    # # resume_from = '/ps/scratch/rdanecek/emoca/finetune_deca/2021_03_09_10-04-28_vaCoPhotoCoLMK_IDW-0.15_Aug_early'
    # # resume_from = '/ps/scratch/rdanecek/emoca/finetune_deca/2021_04_02_18-46-51_va_DeSegFalse_DeNone_Aug_DwC_early'
    # cfg_coarse_1, cfg_detail_1 = load_configs(resume_from_1)
    # checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg_detail_1, "")
    #
    # cfg_coarse_2, cfg_detail_2 = load_configs(resume_from_2)
    # deca_1 = instantiate_deca(cfg_detail_1, "test", "", checkpoint, checkpoint_kwargs)

    path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # run_name = '2021_03_01_11-31-57_VA_Set_videos_Train_Set_119-30-848x480.mp4_EmoNetLossB_F1F2VAECw-0.00150_CoSegmentGT_DeSegmentRend'
    # run_name_1 =  "2021_04_18_12-39-46_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_DwC_early"
    # run_name_1 =  "2021_04_19_13-01-34_ExpDECA_EmoTrain_Jaw_DeSegrend_early"
    # run_name_1 =  "2021_04_19_14-01-29_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_DwC_early"
    run_name_1 =  "2021_04_19_18-57-50_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrendNoLmk_DwC_early"
    # run_name_2 =  "2021_04_14_18-10-36_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_Aug_early"
    # run_name_2 =  "2021_04_16_10-09-30_ExpDECA_para_Jaw_NoRing_DeSegrend_CoNone_DeNone_DwC_early"
    # run_name_2 =  "2021_04_19_13-01-48_ExpDECA_clone_Jaw_DeSegrend_early"
    # run_name_2 =  "2021_04_19_14-01-13_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_early"
    run_name_2 =  "2021_04_19_18-57-31_ExpDECA_DecaD_para_Jaw_NoRing_DeSegrend_early"
    # stage = 'detail'
    stage = 'coarse'
    relative_to_path = '/ps/scratch/'
    replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
    deca = load_deca_and_data(path_to_models, run_name_1, stage, relative_to_path, replace_root_path, load_data=False)
    deca2 = load_deca_and_data(path_to_models, run_name_2, stage, relative_to_path, replace_root_path, load_data=False)
    # deca.deca.config.resume_training = True
    # deca.deca._load_old_checkpoint()
    # deca2.deca.config.resume_training = True
    # deca2.deca._load_old_checkpoint()

    params_deca =  deca.deca.E_flame.named_parameters()
    params_deca2 =  deca2.deca.E_flame.named_parameters()

    num_inconsistencies = 0
    for param in params_deca:
        param2 = next(params_deca2)
        assert param[0] == param2[0]
        if not ((param2[1] == param[1]).cpu().numpy().all()):
            # print(f"{param[0]} has been changed")
            num_inconsistencies += 1
    print(f"Num num_inconsistencies in E_FLAME {num_inconsistencies}")

    params_deca =  deca.deca.E_expression.named_parameters()
    params_deca2 =  deca2.deca.E_expression.named_parameters()

    num_inconsistencies = 0
    for param in params_deca:
        param2 = next(params_deca2)
        assert param[0] == param2[0]
        if not ((param2[1] == param[1]).cpu().numpy().all()):
            # print(f"{param[0]} has been changed")
            num_inconsistencies += 1
    print(f"Num num_inconsistencies in E_EXPRESSION {num_inconsistencies}")

if __name__ == "__main__":
    main()

