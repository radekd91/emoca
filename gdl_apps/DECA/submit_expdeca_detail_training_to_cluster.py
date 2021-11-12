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

    start_at_stage = 2
    resume_from_previous = 1
    force_new_location = 1

    args = f"{coarse_file.name} {detail_file.name} {start_at_stage} {resume_from_previous} {force_new_location}"

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


    resume_folders = []
    # resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_04-49-32_-5959946206105776497_ExpDECA_Affec_clone_Jaw_NoRing_DeSegrend_BlackB_Aug_early"] # TODO: add path here
    resume_folders += ["/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_09_19-00-11_2212703344027741137_ExpDECA_Affec_clone_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]



    for resume_folder in resume_folders:
        resume_folder = Path(resume_folder)
        cfg_coarse_to_fork = OmegaConf.load(resume_folder / "cfg.yaml").coarse

        finetune_modes = [
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

            # AffectNet with augmentation, DEFAULT DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING
            # [
            #     ['model.useSeg=rend',
            #      'model.idw=0',
            #      'learning/batching=single_gpu_expdeca_detail_32gb',
            #      'model.detail_constrain_type=None',
            #      'learning.batch_size_test=1',
            #      'data/augmentations=default'
            #      ]
            # ],
            # [
            #     ['model.useSeg=gt',
            #      'model.idw=0',
            #      'learning/batching=single_gpu_expdeca_detail_32gb',
            #      'model.detail_constrain_type=None',
            #      'learning.batch_size_test=1',
            #      'data/augmentations=default'
            #      ]
            # ],
            [
                ['model.useSeg=gt',
                 'model.idw=0',
                 'learning/batching=single_gpu_expdeca_detail_32gb',
                 'model.detail_constrain_type=None',
                 'learning.batch_size_test=1',
                 'data/augmentations=default',
                 'model.zsymw=0'
                ],
                ['model.useSeg=rend',
                 'model.idw=0',
                 'learning/batching=single_gpu_expdeca_detail_32gb',
                 'model.detail_constrain_type=None',
                 'learning.batch_size_test=1',
                 'data/augmentations=default',
                 'model.zsymw=0'
                 ]
            ],

        ]
        #
        # # sampler = "data.sampler=False"
        sampler = "data.sampler=balanced_expr"
        dataset_detail = 'data/datasets=affectnet_cluster_emonet_cleaned'
        dataset_detail_ring_type = "augment"
        # # # dataset_detail = 'data/datasets=affectnet_cluster'
        # #
        # sampler = "+data.sampler=False"
        # # dataset_detail = 'data/datasets=detail_data_cluster'
        # dataset_detail = 'data/datasets=detail_data_cluster_different_scaling'
        # dataset_detail_ring_type = None

        # sampler = "data.sampler=False"
        # dataset_detail = 'data/datasets=combo_decadetail_affectnet_cluster_emonet_cleaned'
        # dataset_detail_ring_type = "augment"



        learning_rates = [0.0001]
        # learning_rates = [0.0001, 0.00005, 0.00001]
        # learning_rates = [0.00005]
        # learning_rates = [0.0001, 0.00005, 0.00001, 0.000005]
        # learning_rates = [ 0.00005, 0.000005]
        # learning_rates = [0.000001]

        for lr in learning_rates:

            train_K = 4
            batch_size_train = 4
            val_K = 1
            batch_size_val = 4
            # train_K = 2
            # batch_size_train = 1
            # val_K = 2
            # batch_size_val = 1
            background_from_input = "True"
            train_coarse = "False"
            detail_constraint = "exchange"

            fixed_overrides_detail = [
                # 'model/settings=detail_train',
                # 'model/settings=detail_train_emonet',
                'model/settings=detail_train_expdeca',
                # 'model/settings=detail_train_expdeca_emonet',
                f'model.expression_backbone={cfg_coarse_to_fork.model.expression_backbone}',
                # 'model/settings=detail_train_expdeca_emomlp',
                # '+model.mlp_emotion_predictor.detach_shape=True',
                # '+model.mlp_emotion_predictor.detach_expression=False',
                # '+model.mlp_emotion_predictor.detach_detailcode=False',
                # '+model.mlp_emotion_predictor.detach_jaw=True',
                # '+model.mlp_emotion_predictor.detach_global_pose=False',
                f'+model.emonet_model_path={cfg_coarse_to_fork.model.emonet_model_path}',
                'learning.early_stopping.patience=5',
                f'learning.learning_rate={lr}',
                f'model.exp_deca_jaw_pose={cfg_coarse_to_fork.model.exp_deca_jaw_pose}',
                f'model.exp_deca_global_pose={cfg_coarse_to_fork.model.exp_deca_global_pose}',
                f'model.use_emonet_loss=False',
                'model.use_emonet_feat_1=False',
                'model.use_emonet_feat_2=False',
                'model.use_emonet_valence=False',
                'model.use_emonet_arousal=False',
                'model.use_emonet_expression=False',
                'model.use_emonet_combined=False',
                f'model.train_coarse={train_coarse}',
                f'model.detail_constrain_type={detail_constraint}',
                f'model.background_from_input={background_from_input}',
                f'learning.train_K={train_K}',
                f'learning.batch_size_train={batch_size_train}',
                f'learning.val_K={val_K}',
                f'learning.batch_size_val={batch_size_val}',
                dataset_detail,
                sampler,
            ]
            if dataset_detail_ring_type is not None:
                fixed_overrides_detail.append(f'data.ring_type={dataset_detail_ring_type}')
                fixed_overrides_detail.append(f'data.ring_size={batch_size_train}')


            for fmode in finetune_modes:
                detail_overrides = fixed_overrides_detail.copy()

                detail_overrides += fmode[0]

                cfg_detail = train_expdeca.configure_detail(
                    detail_conf, detail_overrides
                )
                GlobalHydra.instance().clear()

                # submit_ = False
                submit_ = True
                if submit_:
                    submit(cfg_coarse_to_fork, cfg_detail, bid=20)
                else:
                    train_expdeca.train_expdeca(cfg_coarse_to_fork, cfg_detail, start_i=2,
                                                resume_from_previous=True, force_new_location=True)


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

