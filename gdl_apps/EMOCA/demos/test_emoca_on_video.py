from gdl_apps.EMOCA.utils.load import load_model
from gdl.utils.FaceDetector import FAN
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
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
from gdl.utils.lightning_logging import _fix_image


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)


def save_obj(deca, filename, opdict, i=0):
    # dense_template_path = '/home/rdanecek/Workspace/Repos/DECA/data/texture_data_256.npy'
    # dense_template_path = '/is/cluster/rdanecek/workspace/repos/DECA/data/texture_data_256.npy'
    dense_template_path = Path(gdl.__file__).parents[1] / 'assets' / "DECA" / "data" / 'texture_data_256.npy'
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


def save_images(outfolder, name, vis_dict, i = 0):
    prefix = None
    final_out_folder = Path(outfolder) / name
    final_out_folder.mkdir(parents=True, exist_ok=True)

    # imsave(final_out_folder / f"inputs.png",  _fix_image(torch_img_to_np(vis_dict['inputs'][i])))
    imsave(final_out_folder / f"geometry_coarse.png",  _fix_image(torch_img_to_np(vis_dict['geometry_coarse'][i])))
    imsave(final_out_folder / f"geometry_detail.png", _fix_image(torch_img_to_np(vis_dict['geometry_detail'][i])))
    imsave(final_out_folder / f"out_im_coarse.png", _fix_image(torch_img_to_np(vis_dict['output_images_coarse'][i])))
    imsave(final_out_folder / f"out_im_detail.png", _fix_image(torch_img_to_np(vis_dict['output_images_detail'][i])))


def save_codes(output_folder, name, vals, i = None):
    if i is None:
        np.save(output_folder / name / f"shape.npy", vals["shapecode"].detach().cpu().numpy())
        np.save(output_folder / name / f"exp.npy", vals["expcode"].detach().cpu().numpy())
        np.save(output_folder / name / f"tex.npy", vals["texcode"].detach().cpu().numpy())
        np.save(output_folder / name / f"pose.npy", vals["posecode"].detach().cpu().numpy())
        np.save(output_folder / name / f"detail.npy", vals["detailcode"].detach().cpu().numpy())
    else: 
        np.save(output_folder / name / f"shape.npy", vals["shapecode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"exp.npy", vals["expcode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"tex.npy", vals["texcode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"pose.npy", vals["posecode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"detail.npy", vals["detailcode"][i].detach().cpu().numpy())


def test(deca, img):
    img["image"] = img["image"].cuda()
    # img["image"] = img["image"].view(1,3,224,224)
    vals = deca.encode(img, training=False)
    vals, visdict = decode(deca, vals, training=False)
    return vals, visdict


def decode(deca, values, training=False):
    with torch.no_grad():
        values = deca.decode(values, training=training)
        # losses = deca.compute_loss(values, training=False)
        # batch_size = values["expcode"].shape[0]
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

    return values, visualizations


def main():
    parser = argparse.ArgumentParser()
    # add the input folder arg 
    # parser.add_argument('--input_video', type=str, default="/ps/project/EmotionalFacialAnimation/data/aff-wild2/Aff-Wild2_ready/AU_Set/videos/Test_Set/82-25-854x480.mp4")
    parser.add_argument('--input_video', type=str, default="/ps/project/EmotionalFacialAnimation/data/aff-wild2/Aff-Wild2_ready/AU_Set/videos/Test_Set/30-30-1920x1080.mp4")
    # add the output folder arg 
    parser.add_argument('--output_folder', type=str, default="/ps/scratch/rdanecek/EMOCA/Test")
    # add the model name arg
    parser.add_argument('--model_name', type=str, default='EMOCA')

    parser.add_argument('--path_to_models', type=str, default=Path(gdl.__file__).parents[1] / "assets/EMOCA/models")

    ## add save_images as boolean arg
    parser.add_argument('--save_images', type=bool, default=True)
    ## add save_codes as boolean arg
    parser.add_argument('--save_codes', type=bool, default=False)
    ## add save_mesh as boolean arg
    parser.add_argument('--save_mesh', type=bool, default=False)

    args = parser.parse_args()


    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = args.path_to_models
    input_video = args.input_video
    output_folder = args.output_folder
    model_name = args.model_name

    mode = 'detail'
    # mode = 'coarse'
    # model_name = '2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early'

    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    processed_subfolder="processed_2022_Jan_14_17-45-25"
    dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=processed_subfolder, batch_size=2)
    dm.prepare_data()
    dm.setup()
    dl = dm.test_dataloader()


    outfolder = str(Path(output_folder) / processed_subfolder / Path(input_video).stem / "results" / model_name)

    for j, batch in enumerate (auto.tqdm( dl)):

        current_bs = batch["image"].shape[0]
        img = batch
        vals, visdict = test(emoca, img)
        for i in range(current_bs):
            # name = f"{(j*batch_size + i):05d}"
            name =  batch["image_name"][i]

            sample_output_folder = Path(outfolder) /name
            sample_output_folder.mkdir(parents=True, exist_ok=True)

            if args.save_mesh:
                save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, i)
            if args.save_images:
                save_images(outfolder, name, visdict, i)
            if args.save_codes:
                save_codes(Path(outfolder), name, vals, i)

    dm.create_reconstruction_video(0,  rec_method='EMOCA', image_type="geometry_detail", overwrite=True)


    print("Done")


if __name__ == '__main__':
    main()
