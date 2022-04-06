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


import torch
import matplotlib.pyplot as plt

from gdl_apps.EMOCA.utils.load import load_deca_and_data


def test(deca, dm=None, image_index = None, values = None, batch=None):
    if image_index is None and values is None and batch is None:
        raise ValueError("Specify either an image to encode-decode or values to decode.")

    if image_index is not None:
        test_set = dm.test_set
        # image_index = 0
        image_index = image_index
        batch = test_set[image_index]
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.cuda()

    if batch is not None:
        with torch.no_grad():
            values = deca.encode(batch, training=False)

    with torch.no_grad():
        values = deca.decode(values, training=False)
        losses_and_metrics = deca.compute_loss(values, batch, training=False)
        values = {**values, **losses_and_metrics}

    uv_detail_normals = None
    if 'uv_detail_normals' in values.keys():
        uv_detail_normals = values['uv_detail_normals']
    visualizations, grid_image = deca._visualization_checkpoint(
        values['verts'],
        values['trans_verts'],
        values['ops'],
        uv_detail_normals,
        values,
        0,
        "",
        "",
        save=False
    )

    vis_dict = deca._create_visualizations_to_log("", visualizations, values, 0, indices=0)

    return values, vis_dict

    # predicted_images = values['predicted_images']
    # predicted_detailed_image = values['predicted_detailed_image']


def plot_results(vis_dict, title, detail=True, show=True, save_path=None):
    # plt.figure()

    # plt.subplot(pos, **kwargs)
    # plt.subplot(**kwargs)
    # plt.subplot(ax)

    if detail:
        fig, axs = plt.subplots(1, 8)
        prefix = None
        for key in list(vis_dict.keys()):
            if 'detail__inputs' in key:
                start_idx = key.rfind('detail__inputs')
                prefix = key[:start_idx]
                # print(f"Prefix was found to be: '{prefix}'")
                break
        if prefix is None:
            print(vis_dict.keys())
            raise RuntimeError(f"Uknown disctionary content. Available keys {vis_dict.keys()}")
        axs[0].imshow(vis_dict[f'{prefix}detail__inputs'])
        if f'{prefix}detail__landmarks_gt'in vis_dict.keys():
            axs[1].imshow(vis_dict[f'{prefix}detail__landmarks_gt'])
        axs[2].imshow(vis_dict[f'{prefix}detail__landmarks_predicted'])
        if f'{prefix}detail__mask'in vis_dict.keys():
            axs[3].imshow(vis_dict[f'{prefix}detail__mask'])
        axs[4].imshow(vis_dict[f'{prefix}detail__geometry_coarse'])
        axs[5].imshow(vis_dict[f'{prefix}detail__geometry_detail'])
        axs[6].imshow(vis_dict[f'{prefix}detail__output_images_coarse'])
        axs[7].imshow(vis_dict[f'{prefix}detail__output_images_detail'])
    else:
        fig, axs = plt.subplots(1, 6)
        prefix = None
        for key in list(vis_dict.keys()):
            if 'coarse__inputs' in key:
                start_idx = key.rfind('coarse__inputs')
                prefix = key[:start_idx]
                break
        if prefix is None:
            print(vis_dict.keys())
            raise RuntimeError(f"Uknown disctionary content. Avaliable keys {vis_dict.keys()}")
        axs[0].imshow(vis_dict[f'{prefix}coarse__inputs'])
        if f'{prefix}coarse__landmarks_gt' in vis_dict.keys():
            axs[1].imshow(vis_dict[f'{prefix}coarse__landmarks_gt'])
        axs[2].imshow(vis_dict[f'{prefix}coarse__landmarks_predicted'])
        if f'{prefix}coarse__mask'in vis_dict.keys():
            axs[3].imshow(vis_dict[f'{prefix}coarse__mask'])
        axs[4].imshow(vis_dict[f'{prefix}coarse__geometry_coarse'])
        axs[5].imshow(vis_dict[f'{prefix}coarse__output_images_coarse'])

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

    fig.set_size_inches(18.5, 5.5)
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
    if show:
        plt.show()
    else:
        plt.close()



def main():
    path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # run_name = '2021_03_01_11-31-57_VA_Set_videos_Train_Set_119-30-848x480.mp4_EmoNetLossB_F1F2VAECw-0.00150_CoSegmentGT_DeSegmentRend'
    run_name =  '2021_03_01_11-31-57_VA_Set_videos_Train_Set_119-30-848x480.mp4_EmoNetLossB_F1F2VAECw-0.00150_CoSegmentGT_DeSegmentRend'
    stage = 'detail'
    relative_to_path = '/ps/scratch/'
    replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
    deca, dm = load_deca_and_data(path_to_models, run_name, stage, relative_to_path, replace_root_path)
    image_index = 390*4

    values, visdict = test(deca, dm, image_index)

    plot_results(visdict, "title")



if __name__ == "__main__":
    main()
