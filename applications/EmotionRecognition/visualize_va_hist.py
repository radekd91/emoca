import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    scratch = "/home/rdanecek/Workspace/mount/scratch/"
    dataset_name = "vggface2"
    # dataset_name = "vggface2hq"
    # dataset_name = "ethnicity"
    # prefix = ""
    dataset_name = "affectnet"
    prefix = "train_"
    # prefix = "validation_"
    dataset_path = Path(scratch) / "rdanecek" / "data" / dataset_name

    va_file = dataset_path / f"{prefix}vae.csv"

    df = pd.read_csv(va_file)

    v = df["valence"].to_numpy()
    a = df["arousal"].to_numpy()

    # va = np.stack([v, a]).T
    cmap = plt.cm.OrRd
    cmap.set_bad(color='black')
    cmap.set_under(color='black')

    plt.hist2d(v, a, bins=20, range=((-1,1), (-1,1)), cmap=cmap, density=True,
               cmin=0.0000001)
    plt.xlabel("valence")
    plt.ylabel("arousal")
    cb = plt.colorbar()
    cb.set_label('density')
    plt.title(dataset_name + f" {prefix[:-1]}")
    plt.savefig(dataset_path / f"{prefix}hist.png")
    plt.show()




if __name__ == "__main__":
    main()
