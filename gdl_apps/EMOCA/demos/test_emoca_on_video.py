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

from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
import gdl
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default=str(Path(gdl.__file__).parents[1] / "data/EMOCA_test_example_data/videos/82-25-854x480_affwild2.mp4"), 
        help="Filename of the video for reconstruction.")
    parser.add_argument('--output_folder', type=str, default="video_output", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA', help='Name of the model to use. Currently EMOCA or DECA are available.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    # add a string argument with several options for image type
    parser.add_argument('--image_type', type=str, default='geometry_detail', 
        choices=["geometry_detail", "geometry_coarse", "output_images_detail", "output_images_coarse"], 
        help="Which image to use for the reconstruction video.")
    parser.add_argument('--processed_subfolder', type=str, default=None, 
        help="If you want to resume previously interrupted computation over a video, make sure you specify" \
            "the subfolder where the got unpacked. It will be in format 'processed_%Y_%b_%d_%H-%M-%S'")
    parser.add_argument('--cat_dim', type=int, default=0, 
        help="The result video will be concatenated vertically if 0 and horizontally if 1")
    parser.add_argument('--include_transparent', type=bool, default=False, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    args = parser.parse_args()
    print("Path to models " + args.path_to_models)
    path_to_models = args.path_to_models
    input_video = args.input_video
    output_folder = args.output_folder
    model_name = args.model_name
    image_type = args.image_type
    cat_dim = args.cat_dim
    include_transparent = bool(args.include_transparent)
    print("Include transparent:", include_transparent)
    processed_subfolder = args.processed_subfolder

    mode = 'detail'
    # mode = 'coarse'
   
    ## 1) Process the video - extract the frames from video and detected faces
    # processed_subfolder="processed_2022_Jan_15_02-43-06"
    # processed_subfolder=None
    dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=processed_subfolder, 
        batch_size=4, num_workers=4)
    dm.prepare_data()
    dm.setup()
    processed_subfolder = Path(dm.output_dir).name

    ## 2) Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    outfolder = str(Path(output_folder) / processed_subfolder / Path(input_video).stem / "results" / model_name)

    ## 3) Get the data loadeer with the detected faces
    dl = dm.test_dataloader()

    ## 4) Run the model on the data
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

    ## 5) Create the reconstruction video (reconstructions overlayed on the original video)
    dm.create_reconstruction_video(0,  rec_method=model_name, image_type=image_type, overwrite=True, 
            cat_dim=cat_dim, include_transparent=include_transparent)
    print("Done")


if __name__ == '__main__':
    main()
