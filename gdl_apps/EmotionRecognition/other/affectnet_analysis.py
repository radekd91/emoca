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


from omegaconf import DictConfig, OmegaConf
from gdl.datasets.AffectNetDataModule import AffectNetDataModule
from tqdm.auto import tqdm
from gdl.layers.losses.EmonetLoader import get_emonet
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import os, sys
import torch.nn.functional as F


def main():
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "train"

    if len(sys.argv) > 2:
        scratch = sys.argv[2]
    else:
        # scratch = "/ps/scratch/"
        scratch = "/home/rdanecek/Workspace/mount/scratch/"

    if len(sys.argv) > 3:
        project = sys.argv[3]
    else:
        # project = "/ps/project/"
        project = "/home/rdanecek/Workspace/mount/project/"

    print(f"Analyzing dataset AffectNet {dataset} data")
    #
    # config = DictConfig({})
    # config.data = DictConfig({})
    # config.data.split_ratio = 1.
    # config.data.split_style = 'sequential'
    # # config.data.datasets = ['vggface2', 'vox2', 'ethnicity']
    # # config.data.datasets = ['vggface2hq', 'vox2']
    # config.data.datasets = [dataset]
    # config.data.scale_min = 1.3
    # config.data.scale_max = 1.3
    # config.data.trans_scale = 0.0
    # config.model = DictConfig({})
    # config.model.image_size = 256
    # # config.model.train_K = 4
    # config.learning = DictConfig({})
    # config.learning.batch_size_train = 64
    # config.learning.train_K = 'max'
    # config.data.num_workers = 6
    # # config.data.output_path = "/home/rdanecek/Workspace/mount/scratch/face2d3d"
    # config.data.output_path = scratch + "face2d3d"

    # out_file_path = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/") / dataset
    out_file_path = Path(scratch) / "rdanecek" / "data" / "affectnet"
    out_file_path.mkdir(exist_ok=True, parents=True)

    dm = AffectNetDataModule(
        str(Path(project) / "EmotionalFacialAnimation/data/affectnet/"),
        str(Path(scratch) / "rdanecek/data/affectnet"),
        # processed_subfolder="processed_2021_Apr_02_03-13-33",
        processed_subfolder="processed_2021_Apr_05_15-22-18",
        mode="manual",
        scale=1.25)
    print(dm.num_subsets)
    dm.prepare_data()
    dm.setup()
    # dl = dm.val_dataloader()
    print(f"len training set: {len(dm.training_set)}")
    print(f"len validation set: {len(dm.validation_set)}")

    if dataset == 'train':
        dset = dm.training_set
    else:
        dset = dm.validation_set

    dl = DataLoader(dset,
                  batch_size=32, shuffle=False,
                  num_workers=4)
    # # dl = dm.train_dataloader()

    image_size = 224

    emonet = get_emonet()

    d = {}
    d['path'] = []
    d['valence'] = []
    d['arousal'] = []
    d['expression'] = []

    # for idx, batch in enumerate(tqdm(dl)):
    for idx in tqdm(range(len(dset))):
    # for idx in tqdm(range(10)):
        batch = dset[idx]
        images = batch['image'].view(-1, 3, image_size, image_size)
        images = images.cuda()
        with torch.no_grad():
            images = F.interpolate(images, 256, mode='bilinear')
            result = emonet(images, intermediate_features=True)
        # self.emonet(images, intermediate_features=True)

        # feat2 = result['emo_feat_2']
        # result['emo_feat']
        v = result['valence']
        a = result['arousal']
        e = result['expression']


        if isinstance(batch['path'][0], list):
            paths = []
            for plist in batch['path']:
                paths += plist
        else:
            paths = batch['path']

        if isinstance(paths, str):
            paths = [paths]
        paths = [str(Path(p).relative_to(Path(scratch) / "rdanecek/data/affectnet")) for p in paths]

        d['path'] += paths
        d['valence'] += v.cpu().numpy().tolist()
        d['arousal'] += a.cpu().numpy().tolist()
        d['expression'] += np.argmax(e.cpu().numpy(), axis=1).tolist()


        # for bi in range(images.shape[0]):
        #     img = np.transpose(images[bi].cpu().numpy(), [1,2,0])
        #     plt.figure()
        #     plt.imshow(img)
        #     plt.show()

        # if idx > 10:
        #     break

        if idx % 2000 == 0:
            print(f"Processing batch {idx}")
            df = pd.DataFrame(data=d)
            df.to_csv(out_file_path / f"{dataset}_vae.csv")

    print("Done processing. Saving ...")
    df = pd.DataFrame(data=d)
    df.to_csv(out_file_path / f"{dataset}_vae.csv")
    print("Data saved.")


if __name__ == "__main__":
    main()
