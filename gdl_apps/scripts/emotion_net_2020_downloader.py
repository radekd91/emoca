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


import pandas
import pandas as pd
from pathlib import Path
from tqdm import auto
import urllib.request

path_to_files = "/ps/project_cifs/EmotionalFacialAnimation/data/emotionnet/"
output_path = "/ps/project_cifs/EmotionalFacialAnimation/data/emotionnet/EmotioNet_FACS_aws_2020_24600_downloaded"
path_to_full_table = Path(path_to_files) / "EmotioNet_FACS_aws_2020_24600.csv"

def read_original_file_list(index):
    image_lists = sorted(list(Path(path_to_files).glob("*.txt")))
    print(f"Found {len(image_lists)} lists")
    print(f"Opening file: {index}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    data_frames = []

    columns = ["url", "orig_url", ]
    columns += [f"AU{i}" for i in range(1,61)]

    df = pd.read_csv(image_lists[index], delimiter="\t", names = columns)
    data_frames += [df]

    return df


def process_original_file_lists():
    image_lists = sorted(list(Path(path_to_files).glob("*.txt")))
    Path(output_path).mkdir(parents=True, exist_ok=True)
    data_frames = []

    columns = ["url", "orig_url", ]
    columns += [f"AU{i}" for i in range(1,61)]

    for l in image_lists:
        df = pd.read_csv(l, delimiter="\t", names = columns)
        data_frames += [df]

    full_df = pandas.concat(data_frames)
    full_df.to_csv(path_to_full_table, index=False)

    return full_df


def download_images(df=None, index=None):
    N = len(df)

    local_full_df = df.copy(deep=True)
    local_full_df = local_full_df.assign(path=pd.Series(dtype=str))
    my_column = local_full_df.pop('path')
    local_full_df.insert(0, my_column.name, my_column)
    local_full_df['path'] = local_full_df['path'].astype(str)
    local_full_df.drop("URL",1, inplace=True)
    local_full_df.drop("URL orig", 1, inplace=True)

    columns = ['path']
    for i in range(1, len(local_full_df.columns)):
        columns += [local_full_df.columns[i].replace("'", "").replace(" ", "")]
    local_full_df.columns = columns

    indices_to_remove = []

    for i in auto.tqdm(range(N)):
        # row = full_df.iloc[[i]]
        url = df.iloc[i]["URL"]
        if url[-1] in ["'", '"']:
            url = url[:-1]
        if url[0] in ["'", '"']:
            url = url[1:]
        old_url = df.iloc[i]["URL orig"]

        rel_path = Path(url).relative_to(Path(url).parents[1])
        dl_path = Path("images") / rel_path

        abs_dl_path = Path(output_path) / dl_path
        abs_dl_path.parent.mkdir(exist_ok=True, parents=True)

        success = False
        if abs_dl_path.exists():
            # print(f"File already exists. Skipping ... {abs_dl_path}")
            success = True
        else:
            try:
                urllib.request.urlretrieve(url, abs_dl_path)
                success = True
            except Exception:
                try:
                    urllib.request.urlretrieve(old_url, abs_dl_path)
                    success = True
                except Exception:
                    success = False

        if not success:
            print(f"Could not download file from '{url}' or '{old_url}")
            indices_to_remove += [i]
            continue

        local_full_df["path"][i] = str(rel_path)
        # print(local_full_df.iloc[i]["path"])

    if len(indices_to_remove) > 0:
        local_full_df.drop(index=indices_to_remove, inplace=True)

    print("Downloading images completed. Saving the data frame")
    if index is None:
        local_full_df.to_csv(Path(output_path) / "image_list.csv", index=False)
    else:
        local_full_df.to_csv(Path(output_path) / f"image_list_{index:02d}.csv", index=False)
    print(f"The dataset has a total of {len(df)} images")
    print(f"Out of that {len(local_full_df)} were found and downloaded successfully.")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
        df = read_original_file_list(index)
        download_images(df, index)
    else:
        index = None
        full_df = pd.read_csv(path_to_full_table)
        download_images(full_df)
