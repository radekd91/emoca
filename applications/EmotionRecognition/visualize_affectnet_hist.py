import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    scratch = "/home/rdanecek/Workspace/mount/scratch/"
    project = "/home/rdanecek/Workspace/mount/project/"
    # dataset_name = "vggface2"
    # dataset_name = "vggface2hq"
    # dataset_name = "ethnicity"
    dataset_name = "affectnet"
    # prefix = "training"
    prefix = "validation"
    dataset_path = Path(project) / "EmotionalFacialAnimation/data/affectnet/Manually_Annotated"
    processed_dataset_path = Path(scratch) / "rdanecek" / "data" / dataset_name

    va_file = dataset_path / f"{prefix}.csv"

    df = pd.read_csv(va_file)

    v = df["valence"].to_numpy()
    a = df["arousal"].to_numpy()



    # va = np.stack([v, a]).T
    cmap = plt.cm.OrRd
    cmap.set_bad(color='black')
    cmap.set_under(color='black')

    # np.histogram2d(v,a, bins=10)

    plt.hist2d(v, a, bins=20, range=((-1,1), (-1,1)), cmap=cmap, density=True, cmin=0.001)
    plt.xlabel("valence")
    plt.ylabel("arousal")
    cb = plt.colorbar()
    cb.set_label('density')
    plt.title(dataset_name + f" {prefix}")
    plt.savefig(processed_dataset_path / f"{prefix}_gt_hist.png")
    plt.show()




if __name__ == "__main__":
    main()
