import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import auto
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import shutil

# Neutral = []
Happy = [334, 480,  481, 493, 496, 497]
Sad = [6, 9, 70, 77, 97, ] #146, ]
Surprise = [232, 166, 169, 228, 242, 265, 177]
Fear = [5, 239, 231, 49, 51, 108, 293]
Disgust = [28, 110, 146, 67, 140]
Anger = [109, 142, 210, 238, 300, 57]



selected_indices = [0, 1, 5, 6 , 8, 9, 10, 12, 19,  20, 23, 26, 27, 28, 30, 31, 35, 37, 49, 51, 54, 55, 57, 59, 63, 67,
70, 75, 77, 88, 96, 97, 101, 105, 107 , 110, 111, 125, 127, 138, 143, 142, 166, 169, 168, 176, 210, 207, 235, 248, 191, 244,286,
164, 275, 295, 277, 300, 323, 345, 349, 338, 357, 355, 361, 377, 378, 384, 383, 407, 429,  452, 433, 464, 467, 450, 489, 482, 490, 472, 405
]


# new_mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study_v2/")
new_mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study_v3")



def copy_mturk_images(method,
                      # input_image_list,
                      method_image_list,
                      image_selection):
    # Path(new_mturk_root) / "method" / method / ;



    path_to_images = []

    # methods_images = Path("/is/cluster/work/rdanecek/emoca/finetune_deca/") / method / "detail" \
    #                  / "affect_net_mturk_detail_test" / folder
    # image_list = [p for p in list(mturk_root.glob("*")) if p.is_dir()]
    #
    # image_list = "/is/cluster/work/rdanecek/emoca/finetune_deca/" \
    #              "2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_detail")



    out_path = Path(new_mturk_root) / method
    (out_path ).mkdir(parents=True, exist_ok=True)
    for i in auto.tqdm( image_selection):
        # path_to_input_image = input_image_list[i]
        path_to_method_image = method_image_list[i]


        # copy the images
        # shutil.copy(path_to_input_image, out_path / "input" / path_to_method_image.name)
        # shutil.copy(path_to_method_image, out_path  / path_to_method_image.name)

        shutil.copy(path_to_method_image, out_path  / f"0000_{i:04d}_00{path_to_method_image.suffix}")
    print(len(image_selection), "copied")


def get_image_selection():
    image_selection = []

    for i in range(5):
        image_selection += [Happy[i]]
        image_selection += [Sad[i]]
        image_selection += [Surprise[i]]
        image_selection += [Fear[i]]
        image_selection += [Disgust[i]]
        image_selection += [Anger[i]]
    return image_selection


def copy_for_all_methods():
    method = "EmocaDetail"

    image_selection = get_image_selection()
    methods = {}
    methods[
        "real"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/inputs"
    methods[
        "MGCNet"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_MGCNet/detail/inputs"
    methods[
        "3DDFA_v2"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-40-45_-5868754668879675020_Face3DDFAModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "Deep3DFace"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-09-34_6754141025581837735_Deep3DFaceModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "DecaCoarse"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "DecaDetail"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/geometry_detail"

    methods[
        "EmocaCoarse"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "EmocaDetail"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_detail"

    for method, path in methods.items():
        path = Path(path)
        if "MGC" in str(path):
            method_images = sorted(list(path.glob("**/GeoOrigin.jpg")))
        else:
            method_images = sorted(list(path.glob("*.png")))

        copy_mturk_images(method, method_images, image_selection)
        print("Done for method %s" % method)


import random

cloud_prefix = "https://emoca.s3.eu-central-1.amazonaws.com"

catch_indices = []

def create_csv(method, image_selection, num_repeats=10):
    fname = Path(new_mturk_root) / f"{method}.csv"

    # image_selection = np.array(image_selection)
    np.random.shuffle(image_selection)
    # image_selection = image_selection.tolist()

    image_list = []
    for i in image_selection:
        if "MGCNet" in method:
            image_list += [f"{cloud_prefix}/{method}/0000_{i:04d}_00.jpg"]
        else:
            image_list += [f"{cloud_prefix}/{method}/0000_{i:04d}_00.png"]

        image_list += [f"{cloud_prefix}/real/0000_{i:04d}_00.png"]
    random.shuffle(image_list)

    image_list += [f"{cloud_prefix}/0000_0077_00_sad.png"]
    image_list += [f"{cloud_prefix}/0000_0238_00_anger.png"]
    image_list += [f"{cloud_prefix}/0000_0239_00_fear.png"]
    image_list += [f"{cloud_prefix}/0000_0300_00_angry.png"]
    image_list += [f"{cloud_prefix}/0000_0480_00_happy.png"]
    random.shuffle(image_list)
    image_list += image_list[:num_repeats]

    image_list1 = ";".join( image_list)
    return image_list1






def create_csv_for_all_methods():
    methods = {}
    methods[
        "MGCNet"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_MGCNet/detail/inputs"
    methods[
        "3DDFA_v2"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-40-45_-5868754668879675020_Face3DDFAModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "Deep3DFace"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-09-34_6754141025581837735_Deep3DFaceModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "DecaCoarse"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "DecaDetail"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/geometry_detail"

    methods[
        "EmocaCoarse"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "EmocaDetail"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_detail"


    new_mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study_v3")

    image_selection = get_image_selection()

    image_lists = []

    for method in methods.keys():
        image_lists += [create_csv(method, image_selection)]

    # create a pandas table with column "Images"
    df = pd.DataFrame(columns=["images"])
    # for each image in image_list, add a row to the df
    for i, image_list in enumerate(image_lists):
        df.loc[i] = image_list


    # save the dataframe to a csv file
    df.to_csv(new_mturk_root / "mturk_v2.csv", sep=" ", index=False)


# def sanity_check():
#     table = pd.read_csv(new_mturk_root / "mturk_v3.csv")
#     for i in range(7):
#         image_list = table["images"].iloc[i]
#         paths = image_list.split(";")
#
#         for pi, path in enumerate( paths):
#             p = Path(path).parent.name
#             f = Path(path).parent.name
#             fn = Path(path).name
#
#             final = Path(new_mturk_root) / f / fn
#
#             if not final.is_file():
#                 if not (Path(new_mturk_root) / Path(final.name)).is_file():
#                     print(path)
#

def main():
    # df = pd.read_csv(
    #     "/ps/project/EmotionalFacialAnimation/data/affectnet/Automatically_Annotated/Automatically_annotated_file_list/representative.csv")
    # # print counts of unique elemnts in column "expression"
    # print(df["expression"].value_counts())
    # copy_for_all_methods()
    # create_csv_for_all_methods()
    # get_image_selection()
    sanity_check()



#
if __name__ == "__main__":
    main()
