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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gdl.datasets.AffectNetDataModule import AffectNetExpressions


def main():
    scratch = "/home/rdanecek/Workspace/mount/scratch/"
    # dataset_name = "vggface2"
    dataset_name = "vggface2hq"
    # dataset_name = "ethnicity"
    prefix = ""
    # dataset_name = "affectnet"
    # prefix = "train_"
    # prefix = "validation_"
    dataset_path = Path(scratch) / "rdanecek" / "data" / dataset_name

    va_file = dataset_path / f"{prefix}vae.csv"

    df = pd.read_csv(va_file)

    v = df["valence"].to_numpy()
    a = df["arousal"].to_numpy()
    e = df["expression"].to_numpy()

    # va = np.stack([v, a]).T
    cmap = plt.cm.OrRd
    cmap.set_bad(color='black')
    cmap.set_under(color='black')

    # plt.figure()
    # plt.hist2d(v, a, bins=20, range=((-1,1), (-1,1)), cmap=cmap, density=True,
    #            cmin=0.0000001)
    # plt.xlabel("valence")
    # plt.ylabel("arousal")
    # cb = plt.colorbar()
    # cb.set_label('density')
    # plt.title(dataset_name + f" {prefix[:-1]}")
    # plt.savefig(dataset_path / f"{prefix}hist.png")
    # plt.show()

    plt.figure(figsize=(12, 6), dpi=80)
    plt.hist(e, bins=list(range(len(AffectNetExpressions))), density=True, rwidth=0.9)
    plt.xlabel("emotion")
    plt.ylabel("density")
    x_pos = np.arange(len(AffectNetExpressions), dtype=np.float32)
    bars = [AffectNetExpressions(int(i)).name for i in x_pos]
    x_pos += 0.5
    plt.xticks(x_pos, bars)
    # cb = plt.colorbar()
    # cb.set_label('density')
    plt.title(dataset_name + f" {prefix[:-1]}")
    plt.savefig(dataset_path / f"{prefix}hist_expression.png")
    plt.show()


if __name__ == "__main__":
    main()
