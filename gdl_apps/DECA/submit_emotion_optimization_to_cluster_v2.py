import optimize_latent_space
from optimize_latent_space import optimization_with_specified_loss, optimization_with_specified_loss_v2, loss_function_config
from pathlib import Path
import copy
import datetime
from omegaconf import DictConfig, OmegaConf
from gdl.utils.condor import execute_on_cluster
import time as t
import random
import pytorch3d.transforms as trans
import torch

def submit_single_optimization(path_to_models, relative_to_path, replace_root_path, out_folder, model_name,
                               model_folder, stage, image_index, target_image,
                               # keyword,
                               num_repeats, optim_kwargs):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    # submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/optimize_emotion/submission"
    # submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/optimize_emotion/submission"
    submission_dir_local_mount = "/is/cluster/work/rdanecek/emoca/optimize_emotion_v2/submission"
    submission_dir_cluster_side = "/is/cluster/work/rdanecek/emoca/optimize_emotion_v2/submission"
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission_%.3d" % random.randint(0,100)
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(optimize_latent_space.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)
    t.sleep(3)

    cgf_name = "optim_kwargs.yaml"
    with open(submission_folder_local / cgf_name, 'w') as outfile:
        OmegaConf.save(config=optim_kwargs, f=outfile)

    # python_bin = 'python'
    bid = 10
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    # gpu_mem_requirement_mb = 14 * 1024
    gpu_mem_requirement_mb = 19 * 1024
    # gpu_mem_requirement_mb = None
    cpus = 1
    gpus = 1
    num_jobs = 1
    max_time_h = 24
    max_price = 8000
    job_name = "optimize_emotion"
    cuda_capability_requirement = 7
    # mem_gb = 20
    mem_gb = 31
    args = f"{path_to_models} {relative_to_path} {replace_root_path} {out_folder} {model_name} " \
                    f"{model_folder} {stage} {image_index} {target_image} {num_repeats} {cgf_name}"

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
                       cuda_capability_requirement=cuda_capability_requirement,
                       concurrency_tag="emo_optim",
                       max_concurrent_jobs=50,
                       )


def optimization_for_different_targets(path_to_models, relative_to_path, replace_root_path, out_folder, model_name,
                                       model_folder, stage, starting_image_index, target_images,
                                       loss_keywords,
                                       num_repeats,
                                       optim_kwargs,
                                       submit=True):
    for target_image in target_images:
        optimization_with_different_losses(
            path_to_models, relative_to_path, replace_root_path,
            Path(out_folder) / Path(model_name) / target_image.parent.stem / target_image.stem,
            model_name, model_folder, stage, starting_image_index, target_image, loss_keywords,
            num_repeats,
            optim_kwargs, submit)


def optimization_for_different_targets_v2(path_to_models, relative_to_path, replace_root_path,
                                          out_folder,
                                          model_name,
                                       model_folder, stage, starting_image, target_images,
                                       num_repeats,
                                       optim_kwargs,
                                       submit=True):
    for target_image in target_images:
        if not submit:
            optimization_with_specified_loss_v2(path_to_models,
                                             relative_to_path,
                                             replace_root_path,
                                             str(Path(out_folder) / model_name / target_image.parent.stem / target_image.stem),
                                             model_name,
                                             model_folder,
                                             stage,
                                             starting_image,
                                             target_image,
                                             num_repeats,
                                             optim_kwargs)
        else:
            submit_single_optimization(path_to_models,
                                       relative_to_path,
                                       replace_root_path,
                                       str(Path(out_folder) / model_name / target_image.parent.stem / target_image.stem),
                                       model_name,
                                       model_folder,
                                       stage,
                                       starting_image,
                                       target_image,
                                       num_repeats,
                                       optim_kwargs)


def optimization_with_different_losses(path_to_models,
                                       relative_to_path,
                                       replace_root_path,
                                       out_folder,
                                       model_name,
                                       model_folder,
                                       stage,
                                       starting_image_index,
                                       target_image,
                                       loss_keywords,
                                        num_repeats,
                                       optim_kwargs,
                                       submit = True):

    for keyword in loss_keywords:
        if not submit:
            optimization_with_specified_loss(path_to_models,
                            relative_to_path,
                            replace_root_path,
                            out_folder / keyword,
                            model_name,
                            model_folder,
                            stage,
                            starting_image_index,
                            target_image,
                            keyword,
                            num_repeats,
                            optim_kwargs)
        else:
            submit_single_optimization(path_to_models,
                            relative_to_path,
                            replace_root_path,
                            out_folder / keyword,
                            model_name,
                            model_folder,
                            stage,
                            starting_image_index,
                            target_image,
                            keyword,
                            num_repeats,
                            optim_kwargs)


def main():
    # cluster
    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    relative_to_path = None
    replace_root_path = None
    out_folder = '/is/cluster/work/rdanecek/emoca/optimize_emotion_v2'
    # target_image_path = Path("/ps/scratch/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10")
    target_image_path = Path("/is/cluster/work/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10")
    submit = True
    # submit = False

    # # not on cluster
    # path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # relative_to_path = '/ps/scratch/'
    # replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
    # out_folder = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/optimize_emotion'
    # target_image_path = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10")

    start_image = target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/000480_000.png" # Rachel 1
    # start_image = target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/002805_000.png" # Rachel 2

    deca_models = {}
    deca_models["2021_09_07_19-19-36_ExpDECA_Affec_balanced_expr_para_Jaw_NoRing_EmoB_EmoCnn_vgg_du_F2VAE_DeSegrend_Aug_DwC_early"] = \
        ['2021_09_07_19-19-36_ExpDECA_Affec_balanced_expr_para_Jaw_NoRing_EmoB_EmoCnn_vgg_du_F2VAE_DeSegrend_Aug_DwC_early',
         'detail', start_image]
    # deca_models["Original_DECA"] = \
    #     ['2021_08_29_10-28-11_DECA_DecaD_VGGl_DeSegrend_Deex_early',
    #      'detail', start_image]
    # deca_models["DECA_DecaD_VGGl_DeSegrend_Deex_early"] = \
    #     ['2021_08_29_10-28-11_DECA_DecaD_VGGl_DeSegrend_Deex_early',
    #      'detail', start_image]


    target_images = [
        # target_image_path / "VA_Set/detections/Train_Set/119-30-848x480/000640_000.png", # Octavia
        ## target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/000480_000.png", # Rachel 1
        target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/002805_000.png", # Rachel 1
        # target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/003899_000.png", # Rachel 2
        # target_image_path / "VA_Set/detections/Train_Set/111-25-1920x1080/000685_000.png", # disgusted guy
        # target_image_path / "VA_Set/detections/Train_Set/122-60-1920x1080-1/001739_000.png", # crazy youtuber
        # target_image_path / "VA_Set/detections/Train_Set/122-60-1920x1080-1/001644_000.png", # crazy youtuber
        # target_image_path / "VA_Set/detections/Train_Set/122-60-1920x1080-1/000048_000.png", # crazy youtuber
        # target_image_path / "VA_Set/detections/Train_Set/135-24-1920x1080/000001_000.png", # couple
        # target_image_path / "VA_Set/detections/Train_Set/135-24-1920x1080/000080_001.png", # couple
    ]

    for t in target_images:
        if not t.exists():
            print(t)
        # print(t.exists())

    # num_repeats = 5
    num_repeats = 1

    emonet = {}
    # emonet["path"] = None
    emonet["path"] = "None"
    # kw["emonet"]["path"] = "/ps/scratch/rdanecek/emoca/emodeca/2021_08_23_22-52-24_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early"
    ## kw["emonet"]["path"] = "/ps/scratch/rdanecek/emoca/emodeca/2021_09_02_19-54-43_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early"
    emonet["path"] = "/ps/scratch/rdanecek/emoca/emodeca/2021_08_30_11-12-32_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early"
    # kw["emonet"]["path"] = "/ps/scratch/rdanecek/emoca/emodeca/2021_08_24_00-17-40_EmoCnn_vgg19_shake_samp-balanced_expr_Aug_early"
    # emonet["path"] = "/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_23-50-06_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"
    # emonet["path"] = "/ps/scratch/rdanecek/emoca/emodeca/2021_08_20_09-43-26_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000"
    # emonet["path"] = '/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_13-06-58_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early'
    # kw["emonet"]["path"] = "Synth"
    # emonet["feature_metric"] = "l1_loss"
    # emonet["feature_metric"] = "mse_loss"
    emonet["feature_metric"] = "cosine_similarity"

    optim_kwargs = {
        "output_image_key": "predicted_detailed_image",
        # "output_image_key": "predicted_images",
        "optimize_detail": False,
        "optimize_identity": False,
        "optimize_expression": False,
        "optimize_neck_pose": False,
        "optimize_jaw_pose": False,
        "optimize_texture": False,
        "optimize_cam": False,
        "optimize_light": False,
        # "lr": 1.0,
        # "lr": 0.1,
        "lr": 0.01,
        # "lr": 0.001,
        # "optimizer_type" : "LBFGS",
        "optimizer_type" : "SGD",
        "max_iters": 1000,
        # "max_iters": 100,
        "patience": 100,
        "visualize_progress" : False,
        "visualize_result" : False,
    }
    optim_kwargs["jaw_lr"] = optim_kwargs["lr"]
    # optim_kwargs["jaw_lr"] = optim_kwargs["lr"] / 10.
    # optim_kwargs["jaw_lr"] = optim_kwargs["lr"] / 100.
    # optim_kwargs["jaw_lr"] = optim_kwargs["lr"] / 100.
    # # detail
    # kw = copy.deepcopy(optim_kwargs)
    # kw["optimize_detail"] = True
    #
    # # identity
    # kw = copy.deepcopy(optim_kwargs)
    # kw["optimize_identity"] = True
    #
    # # expression
    # kw = copy.deepcopy(optim_kwargs)
    # kw["optimize_expression"] = True
    #
    # # pose
    # kw = copy.deepcopy(optim_kwargs)
    # kw["optimize_neck_pose"] = True
    # kw["optimize_jaw_pose"] = True

    # expression, detail
    # kw = copy.deepcopy(optim_kwargs)
    # kw["optimize_detail"] = True
    # kw["optimize_expression"] = True
    # # kw["optimize_neck_pose"] = True
    # kw["optimize_jaw_pose"] = True
    # kw["losses_to_use"] = {
    #     # "emotion_f1": 1.,
    #     "emotion_f2": 1.,
    #     # "emotion_va": 1.,
    #     # "emotion_vae": 1.,
    #     # "emotion_e": 1.,
    #     # "emotion_f12vae": 1.,
    #     # "loss_shape_reg": 100.,
    #     # "loss_expression_reg" : 100.,
    #     "loss_expression_reg" : 10.,
    #     # "loss_z_reg" : 10.,
    # }


    # kw = copy.deepcopy(optim_kwargs)
    # kw["output_image_key"] = "predicted_detailed_image"
    # kw["emonet"] = emonet
    # kw["optimize_detail"] = True
    # # kw["optimize_detail"] = False
    # kw["optimize_expression"] = True
    # kw["optimize_neck_pose"] = False
    # # kw["optimize_jaw_pose"] = True
    # kw["optimize_jaw_pose"] = False
    # kw["losses_to_use"] = {
    #     # "emotion_f1": 1.,
    #     "emotion_f2": 1.,
    #     # "emotion_va": 1.,
    #     # "emotion_vae": 1.,
    #     # "emotion_e": 1.,
    #     # "emotion_f12vae": 1.,
    #     # "loss_shape_reg": 100.,
    #     # "loss_expression_reg" : 100.,
    #     "loss_expression_reg": 10.,
    #     # "loss_z_reg" : 10.,
    #     # "jaw_reg": {
    #     #     "loss_type": "l1",
    #     #     # "loss_type": "l2",
    #     #     "reference_type": "euler",
    #     #     "reference_pose": torch.deg2rad(torch.tensor([15., 0., 0.])).numpy().tolist(),
    #     #     # "reference_type": "quat",
    #     #     # "reference_pose": trans.euler_angles_to_quaternion(torch.deg2rad([0., 0., 0.])).numpy().tolist(),
    #     #     "weight" : 0.1,
    #     # }
    # }

    # kw = copy.deepcopy(optim_kwargs)
    # kw["output_image_key"] = "predicted_detailed_image"
    # kw["emonet"] = emonet
    # kw["optimize_detail"] = False
    # # kw["optimize_detail"] = False
    # kw["optimize_expression"] = True
    # kw["optimize_neck_pose"] = False
    # # kw["optimize_jaw_pose"] = True
    # kw["optimize_jaw_pose"] = False
    # kw["losses_to_use"] = {
    #     # "emotion_f1": 1.,
    #     "emotion_f2": 1.,
    #     # "emotion_va": 1.,
    #     # "emotion_vae": 1.,
    #     # "emotion_e": 1.,
    #     # "emotion_f12vae": 1.,
    #     # "loss_shape_reg": 100.,
    #     # "loss_expression_reg" : 100.,
    #     "loss_expression_reg": 10.,
    #     # "loss_z_reg" : 10.,
    #     # "jaw_reg": {
    #     #     "loss_type": "l1",
    #     #     # "loss_type": "l2",
    #     #     "reference_type": "euler",
    #     #     "reference_pose": torch.deg2rad(torch.tensor([15., 0., 0.])).numpy().tolist(),
    #     #     # "reference_type": "quat",
    #     #     # "reference_pose": trans.euler_angles_to_quaternion(torch.deg2rad([0., 0., 0.])).numpy().tolist(),
    #     #     "weight" : 0.1,
    #     # }
    # }


    kw = copy.deepcopy(optim_kwargs)
    kw["output_image_key"] = "predicted_images"
    kw["emonet"] = emonet
    kw["optimize_detail"] = False
    # kw["optimize_detail"] = False
    kw["optimize_expression"] = True
    # kw["optimize_expression"] = False
    kw["optimize_neck_pose"] = False
    # kw["optimize_neck_pose"] = True
    # kw["optimize_jaw_pose"] = False
    kw["optimize_jaw_pose"] = True
    # kw["optimize_cam"] = True
    kw["optimize_cam"] = False
    kw["losses_to_use"] = {
        # "emotion_f1": 1.,
        "emotion_f2": 10.,
        # "emotion_va": 1.,
        # "emotion_vae": 1.,
        # "emotion_e": 1.,
        # "emotion_f12vae": 1.,
        # "loss_shape_reg": 100.,
        # "loss_expression_reg" : 100.,
        "loss_expression_reg": 10.,
        "loss_photometric_texture": 1.,
        # "loss_landmark": 1.,
        # "loss_lip_distance": 1.,
        # "metric_mouth_corner_distance": 1.,

        # "loss_z_reg" : 10.,
        # "jaw_reg": {
        #     # "loss_type": "l1",
        #     "loss_type": "l2",
        #     "reference_type": "euler",
        #     # "reference_pose": torch.deg2rad(torch.tensor([0., 0., 0.])).numpy().tolist(),
        #     "reference_pose": torch.deg2rad(torch.tensor([5., 0., 0.])).numpy().tolist(),
        #     # "weight" : 0.1,
        #     # "weight" : 1.0,
        #     # "weight" : 10.0,
        #     "weight" : 100., # mouth opens, loss does minimize, but the mouth stays open a little too much
        #     # "weight" : 50.,
        # },
        # "jaw_reg": {
        #     # "loss_type": "l1",
        #     "loss_type": "l2",
        #     "reference_type": "quat",
        #     "reference_pose": "from_target",
        #     # "reference_pose": trans.matrix_to_quaternion(trans.euler_angles_to_matrix(
        #     #     torch.deg2rad(torch.tensor([0., 0., 0.])), "XYZ")).numpy().tolist(),
        #     # "weight" : 0.1,
        #     # "weight" : 0.5,
        #     # "weight" : 1.,
        #     # "weight" : 5.,
        #     "weight" : 10.,
        #     # "weight": 50.,
        # }
    }


    experiment_name = ""
    for key in kw["losses_to_use"].keys():
        if key == "jaw_reg":
            experiment_name += f"{key}_"
        else:
            experiment_name += f"{key}_{kw['losses_to_use'][key]:.2f}_"
    experiment_name += kw["optimizer_type"] + "_" + str(kw["lr"])

    # experiment_name = f"f12vae_exp_{kw['losses_to_use']['loss_expression_reg']:.2f}_det"
    # experiment_name = "vae_exp_det"
    # experiment_name = "vae_exp_det"



    # loss_keywords = ["emotion",
    #                   "emotion_f1_reg_exp",
    #                 "emotion_f2_reg_exp",
    #                 "emotion_f12_reg_exp",
    #                  "emotion_va_reg_exp",
    #                  "emotion_e_reg_exp",
    #                  "emotion_vae_reg_exp",
    #                  "emotion_f12vae_reg_exp"]

    # jaw_lrs = [1., 0.1, 0.01, 0.001, 0.0001, 0.]
    jaw_lrs = [0.01]
    for jaw_lr in jaw_lrs:
        kw["jaw_lr"] = kw["lr"] * jaw_lr
        for name, cfg in deca_models.items():
            time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

            experiment_name_ = time + "_" + experiment_name
            print("Running experiment: " + experiment_name_)
            out_folder = str(Path(out_folder) / experiment_name_)

            model_folder = cfg[0]
            stage = cfg[1]
            starting_image_index = cfg[2]
            optimization_for_different_targets_v2(path_to_models, relative_to_path, replace_root_path,
                                                  out_folder, name,
                                               model_folder, stage, starting_image_index,
                                               target_images,
                                               # loss_keywords,
                                               num_repeats, kw, submit)

if __name__ == "__main__":
    main()
