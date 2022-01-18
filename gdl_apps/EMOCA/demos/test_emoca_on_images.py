from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import TestData
import gdl
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test


def main():
    parser = argparse.ArgumentParser()
    # add the input folder arg 
    parser.add_argument('--input_folder', type=str, default="/ps/data/SignLanguage/SignLanguage_210805_03586_GH/IOI/2021-08-05_ASL_PNG_MH/SignLanguage_210805_03586_GH_LiebBitte_2/Cam_0_35mm_90CW")
    parser.add_argument('--output_folder', type=str, default="/ps/scratch/rdanecek/EMOCA/TestImages", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default=Path(gdl.__file__).parents[1] / "assets/EMOCA/models")
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    
    args = parser.parse_args()


    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = args.path_to_models
    input_folder = args.input_folder
    output_folder = args.output_folder
    model_name = args.model_name

    mode = 'detail'
    # mode = 'coarse'

    # 1) Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    # 2) Create a dataset
    dataset = TestData(input_folder, face_detector="fan", scaling_factor=0.25, max_detection=20)

    ## 4) Run the model on the data
    for i in auto.tqdm( range(len(dataset))):
        batch = dataset[i]
        vals, visdict = test(emoca, batch)
        # name = f"{i:02d}"
        current_bs = batch["image"].shape[0]

        for j in range(current_bs):
            name =  batch["image_name"][j]

            sample_output_folder = Path(output_folder) / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)

            if args.save_mesh:
                save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
            if args.save_images:
                save_images(output_folder, name, visdict, with_detection=True, i=j)
            if args.save_codes:
                save_codes(Path(output_folder), name, vals, i=j)

    print("Done")


if __name__ == '__main__':
    main()
