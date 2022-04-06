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

import os, sys
from pathlib import Path
import numpy as np
import scipy as sp
import torch
import pytorch_lightning as pl
import pandas as pd
from skimage.io import imread
from PIL import Image
import util
from util.preprocess import align_img # from Deep3DFace repo
from util.load_mats import load_lm3d # from Deep3DFace repo



class HavenSet(torch.utils.data.Dataset):

    def __init__(self, folder):
        self.folder = Path(folder)
        self.bfm_folder = Path(util.__path__[0]).parent / "BFM"
        self.image_folder = self.folder / "crops"
        self.lmk_folder = self.folder / "crop-lmks"

        self.image_list = sorted(list(self.image_folder.rglob("*.png")))
        self.lmk_list = sorted(list(self.lmk_folder.rglob("*.npy")))
        self.to_tensor = True

        assert len(self.image_list) == len(self.lmk_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_file = self.image_list[index] 
        lmk_file = self.lmk_list[index] 
        
        assert str(  (image_file.parent / image_file.stem).relative_to(self.image_folder)) == str((lmk_file.parent / image_file.stem).relative_to(self.lmk_folder))
        
        im = Image.open(image_file).convert('RGB')
        lm = np.load(lmk_file)

        W, H = im.size
        # lm = np.loadtxt(lmk_file).astype(np.float32)
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]

        lm3d_std = load_lm3d(self.bfm_folder)

        _, im, lm, _ = align_img(im, lm, lm3d_std)

        if self.to_tensor:
            im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1)
            lm = torch.tensor(lm)

        return {
            "image": im,
            # "landmark": lm,
            "image_path": str(image_file),
        }




def main():
    folder = "/is/cluster/scratch/hfeng/light_albedo/albedo-benchmark/full_benchmark/test_hard/"
    dataset = HavenSet(folder)

    for i in range(len(dataset)):
        sample = dataset[i]



if __name__ == "__main__":
    main()