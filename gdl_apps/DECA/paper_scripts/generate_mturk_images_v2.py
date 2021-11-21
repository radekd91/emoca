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

mapping = {
    1: "neutral",
    2: "happy",
    3: "sad",
    4: "surprise",
    5: "fear",
    6: "disgust",
    7: "anger",
}

# catch_labels = {
#     "0000_0480_00_happy": 2,
#     "0000_0077_00_sad" : 3,
#     "0000_0239_00_fear": 5,
#     "0000_0300_00_angry": 7,
#     "0000_0238_00_anger": 7,
# }
#
catch_labels = {
    "0000_0480_00": 2,
    "0000_0077_00" : 3,
    "0000_0239_00": 5,
    "0000_0300_00": 7,
    "0000_0238_00": 7,
}




def analyze_response(method, answers, image_list, num_repeats=10):

    assert len(answers) == len(image_list)

    answers = answers[num_repeats:]
    image_list = image_list[num_repeats:]

    image_labels = {}
    rec_lables = {}
    catch_responses = {}


    N = len(answers)
    for i in range(N):
        image_name = image_list[i]
        if "real" in image_name:
            image_labels[Path(image_name).stem] = answers[i]
        elif method in image_name:
            rec_lables[Path(image_name).stem] = answers[i]
        else:
            key = Path(image_name).stem
            key = key[:12]
            catch_responses[key] = answers[i]
    if len(rec_lables) == 0:
        print(method)
        # sys.exit()


    num_correct_catches = 0
    for key in catch_responses.keys():
        if int(catch_responses[key]) == int(catch_labels[key]):
            num_correct_catches += 1

    num_consistent_catches = 0
    for key in catch_responses.keys():
        if int(catch_responses[key]) == int(image_labels[key]):
            num_consistent_catches += 1

    if len(catch_responses) < 5:
        print("Num catch responses", len(catch_responses))

    discard_by_mislabel = False
    if num_correct_catches < 3:
        discard_by_mislabel = True

    discard_by_incosistency = False
    if num_consistent_catches < 3:
        discard_by_incosistency = True


    total_score = 0
    correct_scores = {}
    incorrect_scores = {}

    for i in range(1,8):
        correct_scores[i] = 0
        incorrect_scores[i] = 0

    for key, value in image_labels.items():
        if key in rec_lables.keys():
            if value != rec_lables[key]:
                # print("Mismatch", key, value, rec_lables[key])
                incorrect_scores[int(value)] += 1
            else:
                correct_scores[int(value)] += 1
                total_score += 1


    return discard_by_mislabel, discard_by_incosistency, total_score, correct_scores, incorrect_scores


def analyze_results():
    # file = "PartialResultsEmocaSecondRunBatch_4607964_batch_results.csv"
    file = "Emoca_Results_CSV.csv"
    fullf = Path(new_mturk_root) / file

    df =  pd.read_csv(fullf)

    print(len(df))

    method_total_scores = {}
    method_scores = {}
    method_correct_scores = {}
    method_incorrect_scores = {}
    method_num_discarded = {}
    method_num_valid = {}

    for i in range(len(df)):
        ei = df.loc[i]

        answers = ei["Answer.submitValues"].split(",")
        if answers[-1] == "":
            answers = answers[:-1]
        image_list = ei["Input.images"].split(";")
        method = ""
        for im in image_list:
            # impath = Path(im)
            if "MGC" in im:
                method = "MGCNet"
            elif "Deep3DFace" in im:
                method = "Deep3DFace"
            elif "3DDFA_v2" in im:
                method = "3DDFA"
            elif "DecaCoarse" in im:
                method = "DecaCoarse"
            elif "DecaDetail" in im:
                method = "DecaDetail"
            elif "EmocaCoarse" in im:
                method = "EmocaCoarse"
            elif "EmocaDetail" in im:
                method = "EmocaDetail"
            else:
                continue
            break
        if method == "":
            raise Exception("Method not found")

        discard_by_mislabel, discard_by_incosistency, total_score, correct_scores, incorrect_scores = \
            analyze_response(method, answers, image_list)

        if method not in method_scores.keys():
            method_scores[method] = []
            method_correct_scores[method] = []
            method_incorrect_scores[method] = []
            method_num_discarded[method] = 0
            method_total_scores[method] = 0
            method_num_valid[method] = 0

        if not (discard_by_mislabel or discard_by_incosistency):
            method_scores[method] += [total_score]
            method_total_scores[method] += total_score
            method_correct_scores[method] += [correct_scores]
            method_scores[method] += [incorrect_scores]
            method_num_valid[method] += 1
        else:
            method_num_discarded[method] += 1

    for method in method_scores.keys():
        method_total_scores[method] /= method_num_valid[method]
        method_total_scores[method] /= 30

    print("Method scores:")
    print(method_total_scores)
    print("Discarded:")
    print(method_num_discarded)
    print("_____________________________")
    print("Valid:")
    print(method_num_valid)
    print("_____________________________")



def main():
    # df = pd.read_csv(
    #     "/ps/project/EmotionalFacialAnimation/data/affectnet/Automatically_Annotated/Automatically_annotated_file_list/representative.csv")
    # # print counts of unique elemnts in column "expression"
    # print(df["expression"].value_counts())
    # copy_for_all_methods()
    # create_csv_for_all_methods()
    # get_image_selection()
    # sanity_check()
    analyze_results()



#
if __name__ == "__main__":
    main()
