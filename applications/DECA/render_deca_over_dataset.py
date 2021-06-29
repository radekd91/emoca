from affectnet_validation import load_model
from utils.FaceDetector import FAN
from datasets.FaceVideoDataset import TestData
import matplotlib.pyplot as plt
import utils.DecaUtils as util
import numpy as np
import cv2
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto

def save_obj(deca, filename, opdict):
    '''
    vertices: [nv, 3], tensor
    texture: [3, h, w], tensor
    '''
    i = 0
    dense_template_path = '/home/rdanecek/Workspace/Repos/DECA/data/texture_data_256.npy'
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


def save_images(deca, savefolder, name, vis_dict, vals):
    # for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
    #     if vis_name not in vis_dict.keys():
    #         continue
    #     image = util.tensor2image(vis_dict[vis_name][0])
    #     cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name + '.jpg'),
    #                 util.tensor2image(vis_dict[vis_name][0]))

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
    # cv2.imwrite(savefolder + f"/{name}_inputs.jpg", vis_dict[f'{prefix}detail__inputs'])
    # if f'{prefix}detail__landmarks_gt' in vis_dict.keys():
    #     cv2.imwrite(savefolder + f"/{name}_inputs.jpg", vis_dict[f'{prefix}detail__landmarks_gt'])
    # imsave(savefolder + f"/{name}_landmarks.jpg", vis_dict[f'{prefix}detail__landmarks_predicted'])
    # if f'{prefix}detail__mask' in vis_dict.keys():
    #     imsave(savefolder + f"/{name}_inputs.jpg", vis_dict[f'{prefix}detail__mask'])

    depth_image = deca.deca.render.render_depth(vals['trans_verts']).repeat(1, 3, 1, 1)

    parents = list(name.parents)[-2]
    geo_coarse_path = Path(savefolder) / parents / "geometry_coarse" / name.relative_to(parents)
    geo_detail_path = Path(savefolder) / parents / "geometry_detail" / name.relative_to(parents)
    im_coarse_path = Path(savefolder) / parents / "out_im_coarse" / name.relative_to(parents)
    im_detail_path = Path(savefolder) / parents / "out_im_detail" / name.relative_to(parents)
    depth_path = Path(savefolder) / parents / "depth" / name.relative_to(parents)

    geo_coarse_path.parent.mkdir(exist_ok=True, parents=True)
    geo_detail_path.parent.mkdir(exist_ok=True, parents=True)
    im_coarse_path.parent.mkdir(exist_ok=True, parents=True)
    im_detail_path.parent.mkdir(exist_ok=True, parents=True)
    depth_path.parent.mkdir(exist_ok=True, parents=True)

    imsave(geo_coarse_path,
           vis_dict[f'{prefix}detail__geometry_coarse'])
    imsave(geo_detail_path,
           vis_dict[f'{prefix}detail__geometry_detail'])
    imsave(im_coarse_path,
           vis_dict[f'{prefix}detail__output_images_coarse'])
    imsave(im_detail_path,
           vis_dict[f'{prefix}detail__output_images_detail'])
    imsave(depth_path,
           np.transpose(depth_image.detach().cpu().numpy()[0], [1,2,0]))


def test(deca, img):
    img["image"] = img["image"].cuda()
    img["image"] = img["image"].view(1,3,224,224)
    vals = deca.encode(img)
    # vals = deca.decode(vals)
    vals, visdict = decode(deca, vals)
    return vals, visdict


def decode(deca, values):
    with torch.no_grad():
        values = deca.decode(values)
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
    path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    relative_to_path = '/ps/scratch/'
    replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'

    from pathlib import Path
    dataset_path = Path("/home/rdanecek/Workspace/Repos/stargan-v2/data/celeba_hq")
    output_path = dataset_path.parent / "celeba_hq_deca"
    ims = []
    for ext in ["jpg","png","bmp","jpeg"]:
        ims += list(dataset_path.rglob("*." + ext))
    ims = sorted([str(p) for p in ims])
    dataset = TestData(ims,
                       face_detector="fan")
    output_image_size = 1024


    mode = 'detail'
    # mode = 'coarse'
    run_name = '2021_04_19_18-59-19_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early'
    output_path = output_path / run_name

    deca, conf = load_model(path_to_models, run_name, mode,
                            relative_to_path=relative_to_path,
                            replace_root_path=replace_root_path)
    deca.cuda()
    deca.eval()

    from models.Renderer import Pytorch3dRasterizer
    deca.deca.render.image_size = output_image_size
    deca.deca.render.rasterizer = Pytorch3dRasterizer(output_image_size)

    # img_path = "/home/rdanecek/Downloads/lea.jpeg"
    # img_path = "/home/rdanecek/Downloads/lea.jpeg"

    savefolder = str(output_path)
    # for i in auto.tqdm(range(3542, len(dataset))):
    for i in auto.tqdm(range(len(dataset))):
        try:
            img = dataset[i]
            with torch.no_grad():
                vals, visdict = test(deca, img)
            img_path = Path(img['image_path'])
            name = img_path.relative_to(img_path.parents[2])
            # save_obj(deca, f"{savefolder}/{name}.obj", vals)
            save_images(deca, savefolder, name, visdict, vals)
        except:
            print(f"Skipping sample {i}")
        # if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
        #     os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        # if args.saveDepth:
        #     depth_image = deca.render.render_depth(opdict['transformed_vertices']).repeat(1, 3, 1, 1)
        #     visdict['depth_images'] = depth_image
        #     cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        # if args.saveKpt:
        #     np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
        #     np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        # if args.saveObj:
        #     deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        # if args.saveMat:
        #     opdict = util.dict_tensor2npy(opdict)
        #     savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        # if args.saveVis:
        #     cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
        # if args.saveImages:
        #     for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
        #         if vis_name not in visdict.keys():
        #             continue
        #         image = util.tensor2image(visdict[vis_name][0])
        #         cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name + '.jpg'),
        #                     util.tensor2image(visdict[vis_name][0]))

    print("Done")


if __name__ == '__main__':
    main()
