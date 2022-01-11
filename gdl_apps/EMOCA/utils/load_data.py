from gdl.models.DECA import DecaModule
from gdl.models.IO import locate_checkpoint
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from gdl_apps.EMOCA.training.test_and_finetune_deca import prepare_data
import torch
import matplotlib.pyplot as plt
from tqdm import auto


def hack_paths(cfg, replace_root_path=None, relative_to_path=None):
    if relative_to_path is not None and replace_root_path is not None:
        cfg.model.flame_model_path = str(Path(replace_root_path) / Path(cfg.model.flame_model_path).relative_to(relative_to_path))
        cfg.model.flame_lmk_embedding_path = str(Path(replace_root_path) / Path(cfg.model.flame_lmk_embedding_path).relative_to(relative_to_path))
        cfg.model.tex_path = str(Path(replace_root_path) / Path(cfg.model.tex_path).relative_to(relative_to_path))
        cfg.model.topology_path = str(Path(replace_root_path) / Path(cfg.model.topology_path).relative_to(relative_to_path))
        cfg.model.face_mask_path = str(Path(replace_root_path) / Path(cfg.model.face_mask_path).relative_to(relative_to_path))
        cfg.model.face_eye_mask_path = str(Path(replace_root_path) / Path(cfg.model.face_eye_mask_path).relative_to(relative_to_path))
        cfg.model.fixed_displacement_path = str(Path(replace_root_path) / Path(cfg.model.fixed_displacement_path).relative_to(relative_to_path))
        cfg.model.pretrained_vgg_face_path = str(Path(replace_root_path) / Path(cfg.model.pretrained_vgg_face_path).relative_to(relative_to_path))

        cfg.data.data_root = str(Path(replace_root_path) / Path(cfg.data.data_root).relative_to(relative_to_path))

    return cfg



def load_data(path_to_models=None,
                       run_name=None,
                       stage=None,
                       relative_to_path = None,
                       replace_root_path = None,
                       ckpt_index=-1):

    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)

    cfg = conf[stage]

    if relative_to_path is not None and replace_root_path is not None:
        checkpoint = locate_checkpoint(cfg, replace_root_path, relative_to_path, mode=ckpt_index)
        print(f"Loading checkpoint '{checkpoint}'")
        cfg = hack_paths(cfg, replace_root_path=replace_root_path, relative_to_path=relative_to_path)

    cfg.model.resume_training = False
    checkpoint_kwargs = {
        "model_params": cfg.model,
        "learning_params": cfg.learning,
        "inout_params": cfg.inout,
        "stage_name": "testing",
    }


    annotation_list = ['va', 'expr7', 'au8']
    # # annotation_list = ['va', 'expr7',]
    # # annotation_list = ['va']
    # # annotation_list = ['expr7']
    # annotation_list = ['au8']
    index = -1
    # cfg.data.split_style = 'manual'
    # cfg.data.split_style = 'random_by_label'
    cfg.data.split_style = 'sequential_by_label'
    cfg.data.annotation_list = annotation_list
    cfg.data.sequence_index = index
    cfg.learning.train_K_policy = 'sequential'
    dm, name = prepare_data(cfg)
    dm.setup()
    return dm


def test(dm, image_index = None, values = None):
    if image_index is None and values is None:
        raise ValueError("Specify either an image to encode-decode or values to decode.")
    print(f"Training set size: {len(dm.training_set)}")
    print(f"Validation set size: {len(dm.training_set)}")
    print(f"Test set size: {len(dm.test_set)}")

    import numpy as np
    idxs = np.arange(len(dm.training_set), dtype=np.int32)
    np.random.shuffle(idxs)
    for i in auto.tqdm(range(1000)):
        sample = dm.training_set[idxs[i]]
        dm.training_set.visualize_sample(sample)
    print("Done")


def plot_results(vis_dict, title, detail=True):
    # plt.figure()

    # plt.subplot(pos, **kwargs)
    # plt.subplot(**kwargs)
    # plt.subplot(ax)

    if detail:
        fig, axs = plt.subplots(1, 8)
        axs[0].imshow(vis_dict['detail_detail__inputs'])
        axs[1].imshow(vis_dict['detail_detail__landmarks_gt'])
        axs[2].imshow(vis_dict['detail_detail__landmarks_predicted'])
        axs[3].imshow(vis_dict['detail_detail__mask'])
        axs[4].imshow(vis_dict['detail_detail__geometry_coarse'])
        axs[5].imshow(vis_dict['detail_detail__geometry_detail'])
        axs[6].imshow(vis_dict['detail_detail__output_images_coarse'])
        axs[7].imshow(vis_dict['detail_detail__output_images_detail'])
    else:
        fig, axs = plt.subplots(1, 6)
        axs[0].imshow(vis_dict['coarse_coarse__inputs'])
        axs[1].imshow(vis_dict['coarse_coarse__landmarks_gt'])
        axs[2].imshow(vis_dict['coarse_coarse__landmarks_predicted'])
        axs[3].imshow(vis_dict['coarse_coarse__mask'])
        axs[4].imshow(vis_dict['coarse_coarse__geometry_coarse'])
        axs[5].imshow(vis_dict['coarse_coarse__output_images_coarse'])

    # axs[0,0].imshow(vis_dict['detail_detail__inputs'])
    # axs[0,1].imshow(vis_dict['detail_detail__landmarks_gt'])
    # axs[0,2].imshow(vis_dict['detail_detail__landmarks_predicted'])
    # axs[0,3].imshow(vis_dict['detail_detail__mask'])
    # axs[0,4].imshow(vis_dict['detail_detail__geometry_coarse'])
    # axs[0,5].imshow(vis_dict['detail_detail__geometry_detail'])
    # axs[0,6].imshow(vis_dict['detail_detail__output_images_coarse'])
    # axs[0,7].imshow(vis_dict['detail_detail__output_images_detail'])
    # axs[6].imshow(vis_dict['detail_detail__landmarks_predicted'])
    # axs[7].imshow(vis_dict['detail_detail__landmarks_predicted'])
    # axs[8].imshow(vis_dict['detail_detail__landmarks_predicted'])
    # axs[9].imshow(vis_dict['detail_detail__landmarks_predicted'])
    # axs[10].imshow(vis_dict['detail_detail__landmarks_predicted'])

    # plt.title(title)
    fig.suptitle(title)

    plt.show()
    #


import os
import psutil

def main():
    path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # run_name = '2021_03_01_11-31-57_VA_Set_videos_Train_Set_119-30-848x480.mp4_EmoNetLossB_F1F2VAECw-0.00150_CoSegmentGT_DeSegmentRend'
    run_name = '2021_03_25_18-32-46_VA_Set_videos_Train_Set_82-25-854x480.mp4_EmoLossB_F2VAEw-0.00150_DeSeggt_DeNone_Aug_DwC_early'
    stage = 'detail'
    relative_to_path = '/ps/scratch/'
    replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
    dm = load_data(path_to_models, run_name, stage, relative_to_path, replace_root_path)
    image_index = 390*4

    dl = dm.train_dataloader()

    process = psutil.Process(os.getpid())

    for batch_idx, batch in enumerate(auto.tqdm(dl)):
        if batch_idx < 10:
            print(batch.keys())
            for key in batch.keys():
                print(type(batch[key]))
        if batch_idx % 100 == 0:
            print(process.memory_info().rss)  # in bytes

    values = test( dm, image_index)

    # plot_results(visdict, "title")



if __name__ == "__main__":
    main()
