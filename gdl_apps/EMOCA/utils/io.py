from gdl_apps.EMOCA.utils.load import load_model
from gdl.utils.FaceDetector import FAN
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
import gdl
import matplotlib.pyplot as plt
import gdl.utils.DecaUtils as util
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from gdl.utils.lightning_logging import _fix_image


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)


def save_obj(emoca, filename, opdict, i=0):
    # dense_template_path = '/home/rdanecek/Workspace/Repos/DECA/data/texture_data_256.npy'
    # dense_template_path = '/is/cluster/rdanecek/workspace/repos/DECA/data/texture_data_256.npy'
    dense_template_path = Path(gdl.__file__).parents[1] / 'assets' / "DECA" / "data" / 'texture_data_256.npy'
    dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1').item()
    vertices = opdict['verts'][i].detach().cpu().numpy()
    faces = emoca.deca.render.faces[0].detach().cpu().numpy()
    texture = util.tensor2image(opdict['uv_texture_gt'][i])
    uvcoords = emoca.deca.render.raw_uvcoords[0].detach().cpu().numpy()
    uvfaces = emoca.deca.render.uvfaces[0].detach().cpu().numpy()
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


def save_images(outfolder, name, vis_dict, i = 0, with_detection=False):
    prefix = None
    final_out_folder = Path(outfolder) / name
    final_out_folder.mkdir(parents=True, exist_ok=True)

    if with_detection:
        imsave(final_out_folder / f"inputs.png",  _fix_image(torch_img_to_np(vis_dict['inputs'][i])))
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
    if len(img["image"].shape) == 3:
        img["image"] = img["image"].view(1,3,224,224)
    vals = deca.encode(img, training=False)
    vals, visdict = decode(deca, vals, training=False)
    return vals, visdict


def decode(emoca, values, training=False):
    with torch.no_grad():
        values = emoca.decode(values, training=training)
        # losses = deca.compute_loss(values, training=False)
        # batch_size = values["expcode"].shape[0]
        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']
        visualizations, grid_image = emoca._visualization_checkpoint(
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
