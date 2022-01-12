from pathlib import Path
from gdl.datasets.FaceVideoDataModule import FaceVideoDataModule
import sys
import argparse
import os
import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..' ,'..', '..', 'EMOCA')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import util
from tqdm import tqdm
import cv2


def main(args):
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    subfolder = 'processed_2020_Dec_21_00-30-03'
    dm = FaceVideoDataModule(str(root_path), str(output_path), processed_subfolder=subfolder)
    # dm.prepare_data()
    dm.setup()

    sequence_id = args.seq
    testdata = dm.test_dataloader(sequence_id)

    # run EMOCA
    deca_cfg.model.use_tex = args.useTex

    deca = DECA(config = deca_cfg, device=args.device)

    video_writer = None

    for i, batch in enumerate(tqdm(testdata)):
        name = batch['image_name'][0]
        path = Path(batch['image_path'][0])
        save_folder = path.parents[3] / 'predictions' / path.parents[1].stem / path.parents[0].stem
        images = batch['image'].to(args.device)
        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict) #tensor
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            # os.makedirs(os.path.join(save_folder, name), exist_ok=True)
            save_folder.mkdir(exist_ok=True, parents=True)
        # -- save results
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['transformed_vertices']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            depth_folder = save_folder / 'depth'
            depth_folder.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(depth_folder / (name + 'depth.jpg')), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            kpt2d_folder = save_folder / 'kpt2d'
            kpt3d_folder = save_folder / 'kpt3d'
            kpt2d_folder.mkdir(exist_ok=True, parents=True)
            kpt3d_folder.mkdir(exist_ok=True, parents=True)
            np.savetxt(str(kpt2d_folder / (name + '.txt')), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(str(kpt3d_folder / (name + '.txt')), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            mesh_folder = save_folder / 'meshes'
            mesh_folder.mkdir(exist_ok=True, parents=True)
            deca.save_obj(str(mesh_folder/ (name + '.obj')), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            mat_folder = save_folder / 'mat'
            mat_folder.mkdir(exist_ok=True, parents=True)
            savemat(str(mat_folder / (name + '.mat')), opdict)
        if args.saveVis:
            vis_folder = save_folder / 'vis'
            vis_folder.mkdir(exist_ok=True, parents=True)
            vis_im = deca.visualize(visdict)
            cv2.imwrite(str(vis_folder / (name + '.jpg')), vis_im)
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # video_writer = cv2.VideoWriter(filename=str(vis_folder / "video.mp4"), apiPreference=cv2.CAP_FFMPEG,
                #                                fourcc=fourcc, fps=dm.video_metas[sequence_id]['fps'], frameSize=(vis_im.shape[1], vis_im.shape[0]))
                video_writer = cv2.VideoWriter(str(vis_folder / "video.mp4"), cv2.CAP_FFMPEG,
                                               fourcc, int(dm.video_metas[sequence_id]['fps'].split('/')[0]), (vis_im.shape[1], vis_im.shape[0]), True)
            video_writer.write(vis_im)
        if args.saveImages:
            ims_folder = save_folder / 'ims'
            ims_folder.mkdir(exist_ok=True, parents=True)
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                Path(ims_folder / vis_name).mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(ims_folder / vis_name / (name +'.jpg')), image)
    print(f'-- please check the results in {str(save_folder)}')
    if video_writer is not None:
        video_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EMOCA: Detailed Expression Capture and Animation')
    parser.add_argument('--seq', default=0, type=int,
                        help='set device, cpu for using cpu')
    # parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
    #                     help='path to the test data, can be image folder, image path, image list, video')
    # parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
    #                     help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME flame_tex model to generate uv flame_tex map, \
                            set it to True only if you downloaded flame_tex model')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    main(parser.parse_args())
