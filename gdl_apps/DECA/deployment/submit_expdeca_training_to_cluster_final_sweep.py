from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import train_expdeca
import datetime
from omegaconf import OmegaConf
import time as t


def submit(cfg_coarse, cfg_detail, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"

    submission_dir_local_mount = "/is/cluster/work/rdanecek/expdeca/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/expdeca/submission"

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + str(hash(time)) + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(train_expdeca.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    coarse_file = submission_folder_local / "submission_coarse_config.yaml"
    detail_file = submission_folder_local / "submission_detail_config.yaml"

    with open(coarse_file, 'w') as outfile:
        OmegaConf.save(config=cfg_coarse, f=outfile)
    with open(detail_file, 'w') as outfile:
        OmegaConf.save(config=cfg_detail, f=outfile)


    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg_coarse.learning.gpu_memory_min_gb * 1024
    # gpu_mem_requirement_mb = None
    cpus = cfg_coarse.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    # cpus = 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg_coarse.learning.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_deca"
    cuda_capability_requirement = 7
    mem_gb = 40
    args = f"{coarse_file.name} {detail_file.name}"

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
    # t.sleep(2)

def train_on_selected_sequences():
    from hydra.core.global_hydra import GlobalHydra

    coarse_conf = "deca_train_coarse_cluster"
    detail_conf = "deca_train_detail_cluster"

    # ring_type = "gt_expression"
    ring_type = "gt_va"
    # ring_type = "emonet_feature"

    finetune_modes = [
        # # DEFAULT without jaw
        # [
        #     ['model.useSeg=gt', 'model.exp_deca_jaw_pose=False'],
        #     ['model.useSeg=rend']
        # ],
        # # DEFAULT
        # [
        #     ['model.useSeg=gt'],
        #     ['model.useSeg=rend']
        # ],

        # # DEFAULT, rendered mask
        # [
        #     ['model.useSeg=rend'],
        #     ['model.useSeg=rend']
        # ],

        # # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING
        # [
        #     ['model.useSeg=gt', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', ]
        # ],

        # #EmonetStatic backbone, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None',  'learning.batch_size_test=1',
        #      'model.expression_backbone=emonet_static'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None',  'learning.batch_size_test=1',
        #      'model.expression_backbone=emonet_static']
        # ],

        # #Emonet trainable backbone, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None',  'learning.batch_size_test=1',
        #      'model.expression_backbone=emonet_trainable'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None',  'learning.batch_size_test=1',
        #      'model.expression_backbone=emonet_trainable']
        # ],
        #
        # # DECA cloned trainable backbone, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1',
        #      'model.expression_backbone=deca_clone'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'learning.batch_size_test=1',
        #      'model.expression_backbone=deca_clone']
        # ],

        # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'learning.batch_size_test=1']
        # ],

        # # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK, DETAIL WITH COARSE
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None',  'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'model.train_coarse=true',  'learning.batch_size_test=1']
        # ],

        # # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK, EXPRESSION RING EXCHANGE
        # [
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.shape_constrain_type=None',
        #      'model.expression_constrain_type=exchange',
        #      'model.expression_constrain_use_jaw_pose=True',
        #      'model.expression_constrain_use_global_pose=False',
        #      'model.use_geometric_losses_expression_exchange=True',
        #      'model.background_from_input=False',
        #      f'data.ring_type={ring_type}',
        #      # 'data.ring_type=gt_va',
        #      # 'data.ring_type=emonet_feature',
        #      'data.ring_size=4',
        #      'learning/batching=single_gpu_expdeca_coarse_ring',
        #      'learning.gpu_memory_min_gb=24',
        #      'learning.batch_size_test=1'
        #      ],
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.expression_constrain_type=exchange',
        #      'model.expression_constrain_use_jaw_pose=True',
        #      'model.expression_constrain_use_global_pose=False',
        #      'model.use_geometric_losses_expression_exchange=True',
        #      'model.background_from_input=False',
        #      f'data.ring_type={ring_type}',
        #      # 'data.ring_type=gt_va',
        #      # 'data.ring_type=emonet_feature',
        #      'data.ring_size=4',
        #      'learning/batching=single_gpu_expdeca_detail_ring',
        #      'learning.gpu_memory_min_gb=24',
        #      # 'model.shape_constrain_type=None',
        #      'learning.batch_size_test=1']
        # ],
        #
        #DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK, DETAIL WITH COARSE, EXPRESSION RING EXCHANGE
        # [
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.expression_constrain_type=exchange',
        #      'model.expression_constrain_use_jaw_pose=True',
        #      'model.expression_constrain_use_global_pose=False',
        #      'model.use_geometric_losses_expression_exchange=False',
        #      'model.background_from_input=False',
        #      f'data.ring_type={ring_type}',
        #      # 'data.ring_type=gt_va',
        #      # 'data.ring_type=emonet_feature',
        #      'data.ring_size=4',
        #      'learning/batching=single_gpu_expdeca_coarse_ring',
        #      'learning.gpu_memory_min_gb=24',
        #      'model.shape_constrain_type=None',  'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.expression_constrain_type=exchange',
        #      'model.expression_constrain_use_jaw_pose=True',
        #      'model.expression_constrain_use_global_pose=False',
        #      'model.use_geometric_losses_expression_exchange=False',
        #      'model.background_from_input=False',
        #      f'data.ring_type={ring_type}',
        #      # 'data.ring_type=gt_va',
        #      # 'data.ring_type=emonet_feature',
        #      'data.ring_size=4',
        #      'learning/batching=single_gpu_expdeca_detail_ring',
        #      'learning.gpu_memory_min_gb=24',
        #         #'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=none', 'model.train_coarse=true',  'learning.batch_size_test=1']
        # ],

        # # DEFAULT DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK,
        # # DETAIL WITH COARSE, no landmarks for detail stage
        # [
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None',
        #      'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None',
        #      'learning.batch_size_test=1', 'model.train_coarse=true',
        #      'model.use_landmarks=False']
        # ],


        # # # # AffectNet with augmentation, DEFAULT DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING
        # [
        #     ['model.useSeg=gt', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None', 'data/datasets=affectnet_cluster',
        #      'learning.batch_size_test=1', 'data/augmentations=default'],
        #     ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'data/datasets=affectnet_cluster',
        #      'learning.batch_size_test=1', 'data/augmentations=default']
        # ],

        # # cloned expression encoder AffectNet with augmentation, DEFAULT DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING
        [
            [
             # 'model.useSeg=gt',
             'model.useSeg=rend',
             'model.idw=0',
             'model.expression_backbone=deca_clone',
             'learning/batching=single_gpu_expdeca_coarse_32gb',
             'model.shape_constrain_type=None',
             # 'data/datasets=affectnet_cluster',
             'learning.batch_size_test=1',
             'data/augmentations=default'],

            ['model.useSeg=rend', 'model.idw=0',
             'model.expression_backbone=deca_clone',
             'learning/batching=single_gpu_expdeca_detail_32gb',
             # 'model.shape_constrain_type=None',
             'model.detail_constrain_type=None',
             # 'data/datasets=affectnet_cluster',
             'learning.batch_size_test=1',
             'data/augmentations=default']
        ],

        # # DEFAULT but train coarse with detail
        # [
        #     ['model.useSeg=rend'],
        #     ['model.useSeg=rend']
        # ],

        # # DEFAULT but train coarse with detail
        # [
        #     ['model.useSeg=gt'],
        #     ['model.useSeg=rend', 'model.train_coarse=true']
        # ],

        # DEFAULT but train coarse with detail
        # [
        #     ['model.useSeg=rend'],
        #     ['model.useSeg=rend', 'model.train_coarse=true']
        # ],

        # # DEFAULT but cloned ResNet backbone
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=deca_clone'],
        #     ['model.useSeg=rend', 'model.expression_backbone=deca_clone']
        # ],

        # # DEFAULT but cloned ResNet backbone, rendered mask
        # [
        #     ['model.useSeg=rend', 'model.expression_backbone=deca_clone'],
        #     ['model.useSeg=rend', 'model.expression_backbone=deca_clone']
        # ],


        # # DEFAULT but static EmoNet backbone
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=emonet_static'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_static']
        # ],

        # # DEFAULT but static EmoNet backbone, rendered mask
        # [
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_static'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_static']
        # ],

        # # DEFAULT but static EmoNet backbone, no jaw
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=emonet_static', 'model.exp_deca_jaw_pose=False'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_static']
        # ],

        # # DEFAULT but trainable EmoNet backbone
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=emonet_trainable'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_trainable']
        # ]

        # DEFAULT but trainable EmoNet backbone, rendered mask
        # [
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_trainable'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_trainable']
        # ]

        # # REGULAR DECA, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK
        # [
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.max_epochs=8',
        #      'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.max_epochs=8',
        #      'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'learning.batch_size_test=1']
        # ],
        #
        # # REGULAR DECA, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING, RENDERED MASK, DETAIL WITH COARSE
        # [
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.max_epochs=8',
        #      'learning/batching=single_gpu_expdeca_coarse',
        #      'model.shape_constrain_type=None', 'learning.batch_size_test=1'],
        #     ['model.useSeg=rend', 'model.idw=0',
        #      'model.max_epochs=8',
        #      'learning/batching=single_gpu_expdeca_detail',
        #      # 'model.shape_constrain_type=None',
        #      'model.detail_constrain_type=None', 'model.train_coarse=true', 'learning.batch_size_test=1']
        # ],

    ]
    #
    sampler = "data.sampler=False"
    # sampler = "data.sampler=balanced_expr"
    dataset_coarse = 'data/datasets=affectnet_cluster_emonet_cleaned'
    dataset_detail = 'data/datasets=affectnet_cluster_emonet_cleaned'
    # dataset_coarse = "data/datasets=affectnet_cluster"
    # dataset_detail = 'data/datasets=affectnet_cluster'
    # #
    # sampler = "+data.sampler=False"
    # # dataset_coarse = "data/datasets=coarse_data_cluster"
    # # dataset_detail = 'data/datasets=detail_data_cluster'
    # dataset_coarse = "data/datasets=coarse_data_cluster_different_scaling"
    # dataset_detail = 'data/datasets=detail_data_cluster_different_scaling'

    # learning_rates = [0.0001]
    # learning_rates = [0.0001, 0.00005, 0.00001]
    learning_rates = [0.00005]
    # learning_rates = [0.0001, 0.00005, 0.00001, 0.000005]
    # learning_rates = [ 0.00005, 0.000005]
    # learning_rates = [0.000001]

    for lr in learning_rates:

        emonets = []
        # OLD Emotion networks on old AffectNet split
        # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_20_09-43-26_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000'
        # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_23_22-52-24_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early'
        # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_30_11-12-32_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early'
        # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_23-50-06_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early'
        # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_13-06-58_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early'
        # emonet = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_13-06-04_EmoSwin_swin_tiny_patch4_window7_224_shake_samp-balanced_expr_Aug_early'

        # NEW Emotion networks on EmoNet split
        # Emonet
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-13-04_-3054817521854059132_EmoNet_shake_samp-balanced_expr_Aug_early"
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-12-37_-8481804677643293586_EmoNet_shake_samp-balanced_expr_Aug_early"
        # emonets += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-12-56_7559763461347220097_EmoNet_shake_samp-balanced_expr_Aug_early"]

        # SWIN - other
        # emonets += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-05-57_1011354483695245068_EmoSwin_swin_tiny_patch4_window7_224_shake_samp-balanced_expr_Aug_early"]
        # emonets += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-04-52_-2546023918050637211_EmoSwin_swin_small_patch4_window7_224_shake_samp-balanced_expr_Aug_early"]

        # SWIN - base
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-04-52_5038147139113833903_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early"
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-04-16_3124535509353356305_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early"
        emonets += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-04-01_-3592833751800073730_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early"]

        # VGG 19BN
        # emonets += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-02-49_-1360894345964690046_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early"]
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-01-54_-540506871079560164_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early"
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-01-36_-4781895632192757268_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early"

        # ResNet
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-00-41_-6700805205318505957_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_04-00-06_-8559276509623672361_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_05-16-15_-4954470508546110068_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"
        ##emonet = "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_05-15-06_2594210365109986369_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"
        # emonets += ["/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_05-15-38_-8198495972451127810_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"]

        for emonet in emonets:

            # emo_feature_losses = ['mse_loss', 'l1_loss', 'cosine_similarity']
            emo_feature_losses = ['mse_loss']

            for emo_loss in emo_feature_losses:
                emo_feature_loss_type = emo_loss

                # emo_feature_loss_type = 'cosine_similarity'
                # emo_feature_loss_type = 'l1_loss'
                # emo_feature_loss_type = 'mse_loss'
                # emo_feature_loss_type = 'barlow_twins_headless'
                # emo_feature_loss_type = 'barlow_twins'

                resume_from = None # resume from Original DECA
                # resume_from = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_21-50-45_DECA__DeSegFalse_early/" # My DECA, ResNet backbones
                # resume_from = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_23-19-03_DECA__EFswin_s_EDswin_s_DeSegFalse_early/" # My DECA, SWIN small
                # resume_from = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_26_23-19-04_DECA__EFswin_t_EDswin_t_DeSegFalse_early/" # My DECA, SWIN tiny

                use_emo_loss = True
                # use_emo_loss = False

                use_au_loss = None
                # use_au_loss = '+model/additional=au_feature_loss' # au feature loss

                # photometric_uses = [True, False]
                photometric_uses = [True,]

                for photo_use  in photometric_uses:
                    use_photometric = photo_use
                    # use_photometric = True
                    # # use_photometric = False
                    photometric_normalization='mean'
                    # photometric_normalization='rel_mask_value'
                    # photometric_normalization='inv_rel_mask_value'
                    # photometric_normalization='neg_rel_mask_value'
                    # photometric_normalization='abs_mask_value'

                    # landmark_uses = [True, False]
                    landmark_uses = [False]
                    for lmk_use in landmark_uses:
                        use_landmarks = lmk_use
                        # # use_landmarks = True
                        # use_landmarks = False

                        # relative_distance_uses = [True, False]
                        relative_distance_uses = [True, ]
                        # relative_distance_uses = [False, ]
                        #
                        for rel_dist in relative_distance_uses:
                            use_eye_distance = rel_dist
                            use_lip_distance = rel_dist
                            use_mouth_corner_distance = rel_dist

                            # use_eye_distance = True
                            # use_eye_distance = False
                            # # use_lip_distance = True
                            # use_lip_distance = False
                            # # use_mouth_corner_distance = True
                            # use_mouth_corner_distance = False


                            # exp_deca_jaw_pose = True
                            exp_deca_jaw_pose = False

                            fixed_overrides_coarse = [
                                # 'model/settings=coarse_train',
                                # 'model/settings=coarse_train_emonet',
                                # 'model/settings=coarse_train_expdeca',
                                'model/settings=coarse_train_expdeca_emonet',
                                # 'model/settings=coarse_train_expdeca_emomlp',
                                # '+model.mlp_emotion_predictor.detach_shape=True',
                                # '+model.mlp_emotion_predictor.detach_expression=False',
                                # '+model.mlp_emotion_predictor.detach_detailcode=False',
                                # '+model.mlp_emotion_predictor.detach_jaw=True',
                                # '+model.mlp_emotion_predictor.detach_global_pose=False',
                                f'+model.emonet_model_path={emonet}',
                                f'model.resume_training={resume_from == None}', # load the original DECA model
                                'learning.early_stopping.patience=15',
                                f'learning.learning_rate={lr}',
                                'model.max_epochs=30',
                                f'+model.emo_feat_loss={emo_feature_loss_type}',  # emonet feature loss
                                f'model.use_emonet_loss={use_emo_loss}',
                                'model.use_emonet_feat_1=False',
                                'model.use_emonet_feat_2=True',
                                'model.use_emonet_valence=False',
                                'model.use_emonet_arousal=False',
                                'model.use_emonet_expression=False',
                                'model.use_emonet_combined=False',
                                f'model.exp_deca_jaw_pose={exp_deca_jaw_pose}',
                                f'model.use_landmarks={use_landmarks}',
                                f'model.use_photometric={use_photometric}',
                                f'+model.photometric_normalization={photometric_normalization}',
                                f'+model.use_mouth_corner_distance={use_mouth_corner_distance}',
                                f'+model.use_eye_distance={use_eye_distance}',
                                f'+model.use_lip_distance={use_lip_distance}',
                                'model.background_from_input=False',
                                dataset_coarse, # affectnet vs deca dataset
                                sampler,
                            ]
                            if use_au_loss is not None:
                                fixed_overrides_coarse += [use_au_loss]

                            fixed_overrides_detail = [
                                # 'model/settings=detail_train',
                                # 'model/settings=detail_train_emonet',
                                'model/settings=detail_train_expdeca_emonet',
                                # 'model/settings=detail_train_expdeca_emomlp',
                                # '+model.mlp_emotion_predictor.detach_shape=True',
                                # '+model.mlp_emotion_predictor.detach_expression=False',
                                # '+model.mlp_emotion_predictor.detach_detailcode=False',
                                # '+model.mlp_emotion_predictor.detach_jaw=True',
                                # '+model.mlp_emotion_predictor.detach_global_pose=False',
                                f'+model.emonet_model_path={emonet}',
                                'learning.early_stopping.patience=5',
                                f'learning.learning_rate={lr}',
                                f'+model.emo_feat_loss={emo_feature_loss_type}',  # emonet feature loss
                                f'model.use_emonet_loss={use_emo_loss}',
                                'model.use_emonet_feat_1=False',
                                'model.use_emonet_feat_2=True',
                                'model.use_emonet_valence=False',
                                'model.use_emonet_arousal=False',
                                'model.use_emonet_expression=False',
                                'model.use_emonet_combined=False',
                                f'model.exp_deca_jaw_pose={exp_deca_jaw_pose}',
                                f'model.use_landmarks={use_landmarks}',
                                f'model.use_photometric={use_photometric}',
                                f'+model.photometric_normalization={photometric_normalization}',
                                f'+model.use_mouth_corner_distance={use_mouth_corner_distance}',
                                f'+model.use_eye_distance={use_eye_distance}',
                                f'+model.use_lip_distance={use_lip_distance}',
                                'model.background_from_input=False',
                                dataset_detail,
                                sampler,
                            ]
                            if use_au_loss is not None:
                                fixed_overrides_detail += [use_au_loss]

                            emonet_weights = [1.0]
                            # emonet_weights = [0.1, 0.5,  5., 10.]

                            config_pairs = []
                            for emonet_weight in emonet_weights:
                                for fmode in finetune_modes:
                                    coarse_overrides = fixed_overrides_coarse.copy()
                                    detail_overrides = fixed_overrides_detail.copy()
                                    # if len(fmode[0]) != "":
                                    coarse_overrides += fmode[0]
                                    detail_overrides += fmode[1]

                                    # data_override = f'data.sequence_index={video_index}'
                                    # pretrain_coarse_overrides += [data_override]
                                    # coarse_overrides += [data_override]
                                    # detail_overrides += [data_override]
                                    # emonet_weight_override = f'model.mlp_emotion_predictor_weight={emomlp_weight}'
                                    # coarse_overrides += [emonet_weight_override]
                                    # detail_overrides += [emonet_weight_override]

                                    emonet_weight_override = f'model.emonet_weight={emonet_weight}'
                                    coarse_overrides += [emonet_weight_override]
                                    detail_overrides += [emonet_weight_override]

                                    if use_au_loss is not None:
                                        auloss_weight_override = f'model.au_loss.au_weight={emonet_weight}'
                                        coarse_overrides += [auloss_weight_override]
                                        detail_overrides += [auloss_weight_override]

                                    cfgs = train_expdeca.configure(
                                        coarse_conf, coarse_overrides,
                                        detail_conf, detail_overrides
                                    )

                                    GlobalHydra.instance().clear()
                                    config_pairs += [cfgs]

                                    submit(cfgs[0], cfgs[1])
                                    # break
                                # break

                            # for cfg_pair in config_pairs:
                            #     submit(cfg_pair[0], cfg_pair[1])


def default_main():
    coarse_conf = "deca_train_coarse_cluster"
    coarse_overrides = []

    detail_conf = "deca_train_detail_cluster"
    detail_overrides = []

    cfg_coarse, cfg_detail = train_expdeca.configure(
        coarse_conf, coarse_overrides,
        detail_conf, detail_overrides)

    submit(cfg_coarse, cfg_detail)


if __name__ == "__main__":
    # default_main()
    train_on_selected_sequences()

