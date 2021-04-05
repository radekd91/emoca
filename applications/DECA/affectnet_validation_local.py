from affectnet_validation import *



def main():
    path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'

    path_to_affectnet = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"
    path_to_processed_affectnet = "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/"

    run_names = []
    # run_names += ['2021_03_25_19-42-13_DECA_training'] # DECA EmoNet
    run_names += ['2021_03_29_23-14-42_DECA__EmoLossB_F2VAEw-0.00150_DeSegFalse_early'] # DECA EmoNet 2
    # run_names += ['2021_03_18_21-10-25_DECA_training'] # Basic DECA
    # run_names += ['2021_03_26_15-05-56_DECA__DeSegFalse_DwC_early'] # Detail with coarse
    # run_names += ['2021_03_26_14-36-03_DECA__DeSegFalse_DeNone_early'] # No detail exchange


    for run_name in run_names:

        mode = 'detail'
        relative_to_path = '/ps/scratch/'
        replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
        deca, conf = load_model(path_to_models, run_name, mode, relative_to_path, replace_root_path)
        # deca, conf = load_model(path_to_models, run_name, mode)

        # deca.deca.config.resume_training = True
        # deca.deca._load_old_checkpoint()
        # run_name = "Original_DECA"

        deca.eval()

        dm = data_preparation_function(conf[mode], path_to_affectnet, path_to_processed_affectnet)
        conf[mode].model.test_vis_frequency = 1
        # # conf[mode].inout.name = "affectnet_test"
        # conf[mode].inout.name = "Original_DECA"
        single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_net", dm=dm)
        print("We're done y'all")


if __name__ == '__main__':
    main()
