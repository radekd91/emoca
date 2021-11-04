from train_emodeca import *

project_name = 'EmoDECA'



def main():
    if len(sys.argv) < 2:


        # #2 EMODECA
        # # emodeca_default = "emodeca_emonet_coarse"
        # emodeca_default = "emodeca_coarse"
        # emodeca_overrides = [#'learning/logging=none',
        #                      'model/backbone=coarse_emodeca',
        #                      # 'model/backbone=coarse_emodeca_emonet',
        #                      # 'model/settings=AU_emotionet',
        #                      # '+model.mlp_norm_layer=BatchNorm1d', # this one is now default
        #                      # 'model.unpose_global_emonet=false',
        #                      # 'model.use_coarse_image_emonet=false',
        #                      # 'model.use_detail_image_emonet=true',
        #                      # 'model.static_cam_emonet=false',
        #                      # 'model.static_light=false',
        #                     # 'model.mlp_dimension_factor=4',
        #                     'model.mlp_dim=2048',
        #                     # 'data/datasets=emotionet_desktop',
        #                     'data.data_class=AffectNetDataModuleValTest',
        #                     'data/augmentations=default_with_resize',
        #                     'data.num_workers=0'
        #                      ]

        # deca_default = "deca_train_coarse_cluster"
        # deca_overrides = [
        #     # 'model/settings=coarse_train',
        #     'model/settings=detail_train',
        #     'model/paths=desktop',
        #     'model/flame_tex=bfm_desktop',
        #     'model.resume_training=True',  # load the original DECA model
        #     'model.useSeg=rend', 'model.idw=0',
        #     'learning/batching=single_gpu_coarse',
        #     'learning/logging=none',
        #     # 'learning/batching=single_gpu_detail',
        #     #  'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None',
        #     # 'data/datasets=affectnet_cluster',
        #     # 'data/datasets=emotionet_desktop',
        #      'learning.batch_size_test=1'
        # ]
        # deca_conf_path = None
        # stage = None

        deca_default = None
        deca_overrides = None
        # deca_conf_path = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca/2021_04_19_18-59-19_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"
        # # deca_conf_path = "/run/user/1001/gvfs/smb-share:server=ps-access.is.localnet,share=scratch/rdanecek/emoca/finetune_deca/2021_04_19_18-59-19_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"
        deca_conf_path = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_29_21-30-28_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"
        # # deca_conf = None
        stage = 'detail'
        #
        relative_to_path = '/ps/scratch/'
        # # # replace_root_path = '/run/user/1001/gvfs/smb-share:server=ps-access.is.localnet,share=scratch/'
        replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
        #
        # # replace_root_path = None
        # # relative_to_path = None
        #
        # # emodeca_default = "emonet"
        # # emodeca_overrides = ['model/settings=emonet_trainable']
        # # deca_default = None
        # # deca_overrides = None
        # # deca_conf_path = None
        # # stage = None
        # # relative_to_path = None
        # # replace_root_path = None
        #
        # #3) EmoSWIN or EmoCNN
        # emodeca_default = "emoswin"
        # emodeca_overrides = [
        #     # 'model/backbone=swin',
        #     # 'model/backbone=resnet50',
        #     # 'model/backbone=vgg19_bn',
        #     # 'model/backbone=vgg16_bn',
        #     'model/backbone=vgg13_bn',
        #     # 'model/settings=AU_emotionet',
        #     'model/settings=AU_emotionet_bce_weighted',
        #     'learning/logging=none',
        #     # 'learning.max_steps=1',
        #     'learning.max_epochs=1',
        #     'learning.checkpoint_after_training=latest',
        #     # 'learning.batch_size_train=32',
        #     # 'learning.batch_size_val=1',
        #     # '+learning/lr_scheduler=reduce_on_plateau',
        #     # 'model.continuous_va_balancing=1d',
        #     # 'model.continuous_va_balancing=2d',
        #     # 'model.continuous_va_balancing=expr',
        #     # 'learning.val_check_interval=1',
        #     # 'learning.learning_rate=0',
        #     # 'learning/optimizer=adabound',
        #     # 'data/datasets=affectnet_desktop',
        #     # 'data/datasets=affectnet_v1_desktop',
        #     'data/datasets=emotionet_desktop',
        #     # 'data/augmentations=default',
        #     'data/augmentations=default_with_resize',
        #     'data.num_workers=0'
        # ]
        # deca_conf = None
        # deca_conf_path = None
        # fixed_overrides_deca = None
        # stage = None
        # deca_default = None
        # deca_overrides = None

        ## 4) Emo 3DDFA_V2
        emodeca_default = "emo3ddfa_v2"
        emodeca_overrides = [
            'model/backbone=3ddfa_v2',
            # 'model/backbone=3ddfa_v2_resnet',
            'model.mlp_dim=2048',
            # 'data/datasets=emotionet_desktop',
            'data.data_class=AffectNetDataModuleValTest',
            'data/augmentations=default_with_resize',
            'data.num_workers=0',
            'learning/logging=none',
        ]
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
