from gdl_apps.EMOCA.utils.load import load_model
from gdl.utils.FaceDetector import FAN
from gdl.datasets.ImageTestDataset import TestData
import gdl
import matplotlib.pyplot as plt
import gdl.utils.DecaUtils as util
import numpy as np
import cv2
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse


def save_obj(deca, filename, opdict):
    '''
    vertices: [nv, 3], tensor
    texture: [3, h, w], tensor
    '''
    i = 0
    # dense_template_path = '/home/rdanecek/Workspace/Repos/DECA/data/texture_data_256.npy'
    dense_template_path = '/is/cluster/rdanecek/workspace/repos/DECA/data/texture_data_256.npy'
    dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1').item()
    vertices = opdict['verts'][i].detach().cpu().numpy()
    faces = deca.deca.render.faces[0].detach().cpu().numpy()
    texture = util.tensor2image(opdict['uv_texture_gt'][i])
    uvcoords = deca.deca.render.raw_uvcoords[0].detach().cpu().numpy()
    uvfaces = deca.deca.render.uvfaces[0].detach().cpu().numpy()
    # save coarse mesh, with texture and normal map
    normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
    util.write_obj(filename, vertices, faces,
                   texture=texture,
                   uvcoords=uvcoords,
                   uvfaces=uvfaces,
                   normal_map=normal_map)
    # upsample mesh, save detailed mesh
    texture = texture[:, :, [2, 1, 0]]
    normals = opdict['normals'][i].detach().cpu().numpy()
    displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
    dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map, texture,
                                                                   dense_template)
    util.write_obj(filename.replace('.obj', '_detail.obj'),
                   dense_vertices,
                   dense_faces,
                   colors=dense_colors,
                   inverse_face_order=True)


def save_images(outfolder, name, vis_dict):
    prefix = None
    for key in list(vis_dict.keys()):
        if 'detail__inputs' in key:
            start_idx = key.rfind('detail__inputs')
            prefix = key[:start_idx]
            # print(f"Prefix was found to be: '{prefix}'")
            break
    if prefix is None:
        print(vis_dict.keys())
        raise RuntimeError(f"Uknown dictionary content. Available keys {vis_dict.keys()}")

    final_out_folder = Path(outfolder) / name
    final_out_folder.mkdir(parents=True, exist_ok=True)

    imsave(final_out_folder / f"inputs.png", vis_dict[f'{prefix}detail__inputs'])
    # if f'{prefix}detail__landmarks_gt' in vis_dict.keys():
    #     cv2.imwrite(savefolder + f"/{name}_inputs.jpg", vis_dict[f'{prefix}detail__landmarks_gt'])

    imsave(final_out_folder / f"landmarks.png", vis_dict[f'{prefix}detail__landmarks_predicted'])
    # if f'{prefix}detail__mask' in vis_dict.keys():
    #     imsave(savefolder / f"inputs.jpg", vis_dict[f'{prefix}detail__mask'])
    imsave(final_out_folder / f"geometry_coarse.png", vis_dict[f'{prefix}detail__geometry_coarse'])
    imsave(final_out_folder / f"geometry_detail.png", vis_dict[f'{prefix}detail__geometry_detail'])
    imsave(final_out_folder / f"out_im_coarse.png", vis_dict[f'{prefix}detail__output_images_coarse'])
    imsave(final_out_folder / f"out_im_detail.png", vis_dict[f'{prefix}detail__output_images_detail'])


def save_codes(output_folder, name, vals):
    np.save(output_folder / name / f"shape.npy", vals["shapecode"].detach().cpu().numpy())
    np.save(output_folder / name / f"exp.npy", vals["expcode"].detach().cpu().numpy())
    np.save(output_folder / name / f"tex.npy", vals["texcode"].detach().cpu().numpy())
    np.save(output_folder / name / f"pose.npy", vals["posecode"].detach().cpu().numpy())
    np.save(output_folder / name / f"detail.npy", vals["detailcode"].detach().cpu().numpy())


def test(deca, img):
    img["image"] = img["image"].cuda()
    img["image"] = img["image"].view(1,3,224,224)
    vals = deca.encode(img, training=False)
    # vals = deca.decode(vals)
    vals, visdict = decode(deca, vals, training=False)
    return vals, visdict


def decode(deca, values, training=False):
    with torch.no_grad():
        values = deca.decode(values, training=training)
        # losses = deca.compute_loss(values, training=False)

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
    # return values, losses, vis_dict



def main():
    parser = argparse.ArgumentParser()
    # add the input folder arg 
    parser.add_argument('--input_folder', type=str, default="/ps/data/SignLanguage/SignLanguage_210805_03586_GH/IOI/2021-08-05_ASL_PNG_MH/SignLanguage_210805_03586_GH_LiebBitte_2/Cam_0_35mm_90CW")
    # add the output folder arg 
    parser.add_argument('--output_folder', type=str, default="/ps/scratch/rdanecek/For_Nima/SignLanguage_210805_03586_GH/IOI/2021-08-05_ASL_PNG_MH/SignLanguage_210805_03586_GH_LiebBitte_2/Cam_0_35mm_90CW")
    # add the model name arg
    parser.add_argument('--model_name', type=str, default='EMOCA')

    parser.add_argument('--path_to_models', type=str, default=Path(gdl.__file__).parents[1] / "assets/EMOCA/models")

    args = parser.parse_args()


    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = args.path_to_models
    input_folder = args.input_folder
    output_folder = args.output_folder
    model_name = args.model_name

    mode = 'detail'
    # mode = 'coarse'
    # model_name = '2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early'

    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

  
    dataset = TestData(input_folder, face_detector="fan", scaling_factor=0.25)


    for i in auto.tqdm( range(len(dataset))):
        img = dataset[i]
        vals, visdict = test(emoca, img)
        name = f"{i:02d}"

        sample_output_folder = Path(output_folder) / name
        sample_output_folder.mkdir(parents=True, exist_ok=True)

        save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals)
        save_images(output_folder, name, visdict)
        save_codes(Path(output_folder), name, vals)

    print("Done")


if __name__ == '__main__':
    main()
