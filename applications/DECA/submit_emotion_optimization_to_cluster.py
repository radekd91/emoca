import optimize_latent_space
from optimize_latent_space import optimization_with_specified_loss, loss_function_config
from pathlib import Path
import copy
import datetime
from omegaconf import DictConfig, OmegaConf
from utils.condor import execute_on_cluster
import time as t

def submit_single_optimization(path_to_models, relative_to_path, replace_root_path, out_folder, model_name,
                               model_folder, stage, image_index, target_image, keyword, optim_kwargs):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(optimize_latent_space.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)
    t.sleep(1)

    cgf_name = "optim_kwargs.yaml"
    with open(submission_folder_local / cgf_name, 'w') as outfile:
        OmegaConf.save(config=optim_kwargs, f=outfile)

    # python_bin = 'python'
    bid = 10
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = 12 * 1024
    # gpu_mem_requirement_mb = None
    cpus = 1
    gpus = 1
    num_jobs = 1
    max_time_h = 24
    max_price = 8000
    job_name = "optimize_emotion"
    cuda_capability_requirement = 6
    mem_gb = 10
    args = f"{path_to_models} {relative_to_path} {replace_root_path} {out_folder} {model_name} " \
                    f"{model_folder} {stage} {image_index} {target_image} {keyword} {cgf_name}"

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


def optimization_for_different_targets(path_to_models, relative_to_path, replace_root_path, out_folder, model_name,
                                       model_folder, stage, starting_image_index, target_images, loss_keywords, optim_kwargs,
                                       submit=True):
    for target_image in target_images:
        optimization_with_different_losses(
            path_to_models, relative_to_path, replace_root_path,
            Path(out_folder) / Path(model_name) / target_image.parent.stem / target_image.stem,
            model_name, model_folder, stage, starting_image_index, target_image, loss_keywords, optim_kwargs, submit)


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
                            optim_kwargs)


def main():
    # cluster
    path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    relative_to_path = None
    replace_root_path = None
    out_folder = '/ps/scratch/rdanecek/emoca/optimize_emotion'
    target_image_path = Path("/ps/scratch/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10")
    submit = True

    # ## not on cluster
    # path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # relative_to_path = '/ps/scratch/'
    # replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
    # out_folder = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/optimize_emotion'
    # target_image_path = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10")
    # submit = False

    deca_models = {}
    deca_models["Octavia"] = \
        ['2021_03_08_22-30-55_VA_Set_videos_Train_Set_119-30-848x480.mp4CoPhotoCoLMK_IDW-0.15_Aug_early', 'detail', 390 * 4 + 1]
    deca_models["Rachel"] = \
        ['2021_03_05_16-31-05_VA_Set_videos_Train_Set_82-25-854x480.mp4CoPhotoCoLMK_IDW-0.15_Aug_early', 'detail', 90*4]
    deca_models["General1"] = \
        ['2021_03_08_22-30-55_VA_Set_videos_Train_Set_119-30-848x480.mp4CoPhotoCoLMK_IDW-0.15_Aug_early', None, 390*4]
    deca_models["General2"] = \
        ['2021_03_05_16-31-05_VA_Set_videos_Train_Set_82-25-854x480.mp4CoPhotoCoLMK_IDW-0.15_Aug_early', None, 90*4]


    target_images = [
        target_image_path / "VA_Set/detections/Train_Set/119-30-848x480/000640_000.png", # Octavia
        target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/000480_000.png", # Rachel 1
        target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/002805_000.png", # Rachel 1
        target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/003899_000.png", # Rachel 2
        target_image_path / "VA_Set/detections/Train_Set/111-25-1920x1080/000685_000.png", # disgusted guy
        target_image_path / "VA_Set/detections/Train_Set/122-60-1920x1080-1/001739_000.png", # crazy youtuber
        target_image_path / "VA_Set/detections/Train_Set/122-60-1920x1080-1/001644_000.png", # crazy youtuber
        target_image_path / "VA_Set/detections/Train_Set/122-60-1920x1080-1/000048_000.png", # crazy youtuber
        target_image_path / "VA_Set/detections/Train_Set/135-24-1920x1080/000001_000.png", # couple
        target_image_path / "VA_Set/detections/Train_Set/135-24-1920x1080/000080_001.png", # couple
    ]

    for t in target_images:
        if not t.exists():
            print(t)
        # print(t.exists())

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    experiment_name = "det_exp"
    # experiment_name = "det_exp"

    experiment_name = time + "_" + experiment_name
    out_folder = str(Path(out_folder) / experiment_name)

    optim_kwargs = {
        "optimize_detail": False,
        "optimize_identity": False,
        "optimize_expression": False,
        "optimize_neck_pose": False,
        "optimize_jaw_pose": False,
        "optimize_texture": False,
        "optimize_cam": False,
        "optimize_light": False,
        "lr": 0.01,
        "optimizer_type" : "LBFGS",
        "max_iters": 1000,
        "patience": 20,
        "visualize_progress" : False,
        "visualize_result" : False,
    }

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
    kw = copy.deepcopy(optim_kwargs)
    kw["optimize_detail"] = True
    kw["optimize_expression"] = True

    # # expression, detail, pose
    # kw = copy.deepcopy(optim_kwargs)
    # kw["optimize_detail"] = True
    # kw["optimize_expression"] = True
    # kw["optimize_neck_pose"] = True
    # kw["optimize_jaw_pose"] = True

    loss_keywords = ["emotion",
                      "emotion_f1_reg_exp",
                    "emotion_f2_reg_exp",
                    "emotion_f12_reg_exp",
                     "emotion_va_reg_exp",
                     "emotion_e_reg_exp",
                     "emotion_vae_reg_exp",
                     "emotion_f12vae_reg_exp"]

    for name, cfg in deca_models.items():
        model_folder = cfg[0]
        stage = cfg[1]
        starting_image_index = cfg[2]
        optimization_for_different_targets(path_to_models, relative_to_path, replace_root_path, out_folder, name,
                                           model_folder, stage, starting_image_index,
                                           target_images, loss_keywords, kw, submit)

if __name__ == "__main__":
    main()
