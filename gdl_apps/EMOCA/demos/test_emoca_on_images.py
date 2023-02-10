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
    parser.add_argument('--input_folder', type=str, default= str(Path(gdl.__file__).parents[1] / "assets/data/EMOCA_test_example_data/images/affectnet_test_examples"))
    parser.add_argument('--output_folder', type=str, default="image_output", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    parser.add_argument('--mode', type=str, default='detail', help="coarse or detail")
    
    args = parser.parse_args()

    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = args.path_to_models
    input_folder = args.input_folder
    model_name = args.model_name
    output_folder = args.output_folder + "/" + model_name

    mode = args.mode
    # mode = 'detail'
    # mode = 'coarse'

    # 1) Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    # 2) Create a dataset
    dataset = TestData(input_folder, face_detector="fan", max_detection=20)

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
