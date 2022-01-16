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

inv_mapping = dict(zip(mapping.values(), mapping.keys()))


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


def fname2gt(fname):
    fname_idx = int(fname.split('_')[1])
    if fname_idx in Happy:
        return inv_mapping["happy"]
    if fname_idx in Sad:
        return inv_mapping["sad"]
    if fname_idx in Fear:
        return inv_mapping["fear"]
    if fname_idx in Disgust:
        return inv_mapping["disgust"]
    if fname_idx in Anger:
        return inv_mapping["anger"]
    if fname_idx in Surprise:
        return inv_mapping["surprise"]
    raise Exception(f"{fname} not found in any class")



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

    gt_im_confusion_matrix = np.zeros((7, 6))
    gt_rec_confusion_matrix = np.zeros((7, 6))
    gt_class_counts = np.zeros(6)

    confusion_matrix = np.zeros((7, 7))
    class_counts = np.zeros(7)

    for i in range(1,8):
        correct_scores[i] = 0
        incorrect_scores[i] = 0

    used_fnames_im = set()
    used_fnames_rec = set()

    for key, value in image_labels.items():
        if key in rec_lables.keys():
            if value != rec_lables[key]:
                # print("Mismatch", key, value, rec_lables[key])
                incorrect_scores[int(value)] += 1
            else:
                correct_scores[int(value)] += 1
                total_score += 1
            confusion_matrix[int(value)-1, int(rec_lables[key])-1] += 1
            class_counts[int(value)-1] += 1

            gt_label = fname2gt(key)

            if key not in used_fnames_im:
                gt_im_confusion_matrix[int(value)-1, gt_label-2] += 1
                used_fnames_im = used_fnames_rec.union(set([key]))
                gt_class_counts[gt_label - 2] += 1
            if key not in used_fnames_rec:
                gt_rec_confusion_matrix[int(rec_lables[key])-1, gt_label-2] += 1
                used_fnames_rec = used_fnames_rec.union(set([key]))


    return discard_by_mislabel, discard_by_incosistency, total_score, correct_scores, incorrect_scores, \
           confusion_matrix, class_counts, gt_im_confusion_matrix, gt_rec_confusion_matrix, gt_class_counts


def plot_confusion_matrix(method_list, confusion_matrix, num_classes, class_labels,
                          abs_conf_matrix=None, abs_class_counts=None):
    if isinstance(num_classes, int):
        num_classes_1 = num_classes
        num_classes_2 = num_classes
    else:
        num_classes_1 = num_classes[0]
        num_classes_2 = num_classes[1]

    if not isinstance(class_labels, list):
        class_labels_1 = class_labels
        class_labels_2 = class_labels
    else:
        class_labels_1 = class_labels[0]
        class_labels_2 = class_labels[1]


    fig, axes = plt.subplots(num_classes_1, num_classes_2, figsize=(20, 20))
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    # for each row and column
    for i in range(num_classes_1):
        # for each column
        for j in range(num_classes_2):
            # for each method, get the confusion matrix score
            scores = []
            names = []
            for method in method_list:
                # get the score for the current method
                # score = total_rel_confusion_matrices[method][i, j]
                score = confusion_matrix[method][i, j]
                # append the score to the list
                scores.append(score)
                # append the method name to the list
                # if j != 0:
                #     names.append("")
                # else:
                names.append(method)

            if i == 0:
                # set the subtitle to the emotion label
                if class_labels_1 is class_labels_2:
                    axes[i, j].set_title(class_labels_2[j+1], fontsize=20)
                else:
                    # [WARNING] UGLY LABEL HACK
                    # axes[i, j].set_title(class_labels_2[j+2], fontsize=20)
                    # axes[i, j].set_title(class_labels_1[j+1], fontsize=20)
                    axes[i, j].set_title(class_labels_2[j+1], fontsize=20)

            # create a bar plot, each bar has a unique color
            # and the width is the score
            # the y-axis is the method name
            # the x-axis is the score
            # create the bar plot
            ax = axes[i, j]
            barlist = ax.barh(names, scores)

            # if j > 0, disable y-tick labels
            if j > 0:
                ax.set_yticklabels([])
            else:
                # set y label to the emotion label with a large font
                # ax.set_ylabel(class_labels_1[i+1], fontsize=20)
                if class_labels_1 is class_labels_2:
                    ax.set_ylabel(class_labels_1[i+1], fontsize=20)
                else:
                    # [WARNING] UGLY LABEL HACK
                    ax.set_ylabel(class_labels_1[i+2], fontsize=20)
                # ax.set_ylabel(mapping[i+1])

            # set x axis range to [0, 1]
            ax.set_xlim([0, 1.25])

            # set a unique color for each bar
            for bi, bar in enumerate( barlist):
                bar.set_color(colors[bi % len(colors)])
                # add a text label to the right end of each bar
                # with the score as its value
                # ax.text(bar.get_width() + 0.01, bar.get_y() + 0.5,

                # height = bar.get_height()
                width = bar.get_width()
                # plt.text(bar.get_y() + bar.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
                # plt.text(bar.get_y() + bar.get_width() / 2.0, height, f'ha', ha='center', va='bottom')
                # plt.text(width, bar.get_x() + bar.get_height() / 2.0, f'ha', ha='center', va='bottom')
                # ax.text(bar.get_x() + bar.get_height() / 2.0, width, f'ha', ha='center', va='bottom')

                if abs_conf_matrix is not None:
                    if not (class_labels_1 is class_labels_2):
                        bar_label = f"{int(abs_conf_matrix[method_list[bi]][i, j])}/{int(abs_class_counts[method_list[bi]][i+1])}"
                    else:
                        bar_label = f"{int(abs_conf_matrix[method_list[bi]][i,j])}/{int(abs_class_counts[method_list[bi]][i])}"
                    ax.text(bar.get_width() + 0.15, bar.get_y(), bar_label, ha='center', va='bottom')
                # ax.text(bar.get_width() + 0.05, bar.get_y(), f'ha', ha='center', va='bottom')

            if class_labels_1 is class_labels_2:
                if i == j:
                    #make the backround beige
                    ax.set_facecolor('#f5f5dc')
            else:
                if i == j-1:
                    #make the backround beige
                    ax.set_facecolor('#f5f5dc')

    # set the title of the figure
    # fig.suptitle("Relative Confusion Matrix Scores")
    return fig


def analyze_results():
    # file = "PartialResultsEmocaSecondRunBatch_4607964_batch_results.csv"
    file = "Emoca_Results_CSV.csv"
    fullf = Path(new_mturk_root) / file

    df =  pd.read_csv(fullf)

    print(len(df))

    method_total_scores = {}
    method_scores = {}
    method_confusion_matrices = {}
    method_confusion_matrices_rec_gt = {}
    method_confusion_matrices_im_gt = {}
    method_class_counts = {}
    method_gt_class_counts = {}
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

        discard_by_mislabel, discard_by_incosistency, total_score, correct_scores, incorrect_scores, \
        confusion_matrix, class_counts, gt_im_confusion_matrix, gt_rec_confusion_matrix, gt_class_counts = \
            analyze_response(method, answers, image_list)

        if method not in method_scores.keys():
            method_scores[method] = []
            method_correct_scores[method] = []
            method_incorrect_scores[method] = []
            method_num_discarded[method] = 0
            method_total_scores[method] = 0
            method_num_valid[method] = 0
            method_confusion_matrices[method] = []
            method_class_counts[method] = []
            method_gt_class_counts[method] = []
            method_confusion_matrices_rec_gt[method] = []
            method_confusion_matrices_im_gt[method] = []


        if not (discard_by_mislabel or discard_by_incosistency):
            method_scores[method] += [total_score]
            method_total_scores[method] += total_score
            method_correct_scores[method] += [correct_scores]
            method_scores[method] += [incorrect_scores]
            method_num_valid[method] += 1
            method_confusion_matrices[method] += [confusion_matrix]
            method_class_counts[method] += [class_counts]
            method_gt_class_counts[method] += [gt_class_counts]
            method_confusion_matrices_rec_gt[method] += [gt_rec_confusion_matrix]
            method_confusion_matrices_im_gt[method] += [gt_im_confusion_matrix]
        else:
            method_num_discarded[method] += 1


    total_confusion_matrices = {}
    total_rel_confusion_matrices = {}
    total_rel_gt_rec_confusion_matrices = {}
    total_gt_rec_confusion_matrices = {}

    total_rel_gt_im_confusion_matrices = {}
    total_gt_im_confusion_matrices = {}

    # total_rel2_confusion_matrices = {}
    total_class_counts = {}
    total_gt_class_counts = {}
    for method in method_scores.keys():
        method_total_scores[method] /= method_num_valid[method]
        method_total_scores[method] /= 30
        total_confusion_matrices[method] = np.stack(method_confusion_matrices[method]).sum(axis=0)
        total_class_counts[method] = np.stack(method_class_counts[method]).sum(axis=0)
        # total_rel2_confusion_matrices[method] = np.stack(method_class_counts[method]).sum(axis=0)
        total_rel_confusion_matrices[method] = total_confusion_matrices[method] / total_class_counts[method].reshape(-1, 1)

        total_rel_gt_rec_confusion_matrices[method] = np.stack(method_confusion_matrices_rec_gt[method]).sum(axis=0)
        total_gt_rec_confusion_matrices[method] = np.stack(method_confusion_matrices_rec_gt[method]).sum(axis=0)
        total_gt_class_counts[method] = np.stack( len(method_gt_class_counts[method]) *
                                                  np.array( [0, *(method_gt_class_counts[method][0].tolist())]))#.sum(axis=0)

        # total_gt_class_counts[method] = np.stack( len(method_gt_class_counts[method]) *
        #                                           method_gt_class_counts[method][0]
        #                                           )#.sum(axis=0)

        total_rel_gt_rec_confusion_matrices[method] /= total_gt_class_counts[method][1:].reshape(1, -1)

        total_rel_gt_im_confusion_matrices[method] = np.stack(method_confusion_matrices_im_gt[method]).sum(axis=0)
        total_gt_im_confusion_matrices[method] = np.stack(method_confusion_matrices_im_gt[method]).sum(axis=0)
        total_rel_gt_im_confusion_matrices[method] /= total_gt_class_counts[method][1:].reshape(1, -1)

    # labels
    #
    # colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    #
    # #create a figure with 7x7 subfigures
    # fig, axes = plt.subplots(7, 7, figsize=(20, 20))
    # # for each row and column
    # for i in range(7):
    #     # for each column
    #     for j in range(7):
    #         # get the index of the subplot
    #         index = i * 7 + j
    #         # for each method, get the confusion matrix score
    #         scores = []
    #         names = []
    #         for method in method_scores.keys():
    #             # get the score for the current method
    #             # score = total_rel_confusion_matrices[method][i, j]
    #             score = total_confusion_matrices[method][i, j]
    #             # append the score to the list
    #             scores.append(score)
    #             # append the method name to the list
    #             # if j != 0:
    #             #     names.append("")
    #             # else:
    #             names.append(method)
    #
    #         if i == 0:
    #             # set the subtitle to the emotion label
    #             axes[i, j].set_title(mapping[j+1], fontsize=20)
    #
    #         # create a bar plot, each bar has a unique color
    #         # and the width is the score
    #         # the y-axis is the method name
    #         # the x-axis is the score
    #         # create the bar plot
    #         ax = axes[i, j]
    #         barlist = ax.barh(names, scores)
    #
    #         # if j > 0, disable y-tick labels
    #         if j > 0:
    #             ax.set_yticklabels([])
    #         else:
    #             # set y label to the emotion label with a large font
    #             ax.set_ylabel(mapping[i+1], fontsize=20)
    #             # ax.set_ylabel(mapping[i+1])
    #
    #         # set the x-axis label
    #         # ax.set_xlabel("Relative Confusion Matrix Score")
    #         # set a unique color for each bar
    #         print(len(barlist))
    #         for bi, bar in enumerate( barlist):
    #             bar.set_color(colors[bi])
    #
    #
    # # set the title of the figure
    # fig.suptitle("Relative Confusion Matrix Scores")
    # # show the figure
    # plt.show()

    fig1 = plot_confusion_matrix(list(method_scores.keys()), total_rel_confusion_matrices, 7, mapping,
                                 abs_conf_matrix=total_confusion_matrices, abs_class_counts=total_class_counts,
                                 )
    # fig2 = plot_confusion_matrix(list(method_scores.keys()), total_confusion_matrices, 7, mapping)

    mapping_copy = mapping.copy()
    del mapping_copy[1]
    mapping_list = [mapping, mapping_copy]

    # fig2 = plot_confusion_matrix(list(method_scores.keys()), total_rel_gt_rec_confusion_matrices, (7, 6), mapping_list,
    #                              abs_conf_matrix=total_gt_rec_confusion_matrices,
    #                              abs_class_counts=total_gt_class_counts,
    #                              )


    mapping_list = [mapping_copy, mapping, ]
    for key in total_rel_gt_rec_confusion_matrices.keys():
        total_rel_gt_rec_confusion_matrices[key] = total_rel_gt_rec_confusion_matrices[key].T
        total_gt_rec_confusion_matrices[key] = total_gt_rec_confusion_matrices[key].T
        total_rel_gt_im_confusion_matrices[key] = total_rel_gt_im_confusion_matrices[key].T
        total_gt_im_confusion_matrices[key] = total_gt_im_confusion_matrices[key].T
        # total_rel_gt_rec_confusion_matrices[key] = total_rel_gt_rec_confusion_matrices[key].T

    fig2 = plot_confusion_matrix(list(method_scores.keys()), total_rel_gt_rec_confusion_matrices, (6, 7), mapping_list,
                                 abs_conf_matrix=total_gt_rec_confusion_matrices,
                                 abs_class_counts=total_gt_class_counts,
                                 )

    fig3 = plot_confusion_matrix(list(method_scores.keys()), total_rel_gt_im_confusion_matrices, (6, 7), mapping_list,
                                 abs_conf_matrix=total_gt_im_confusion_matrices,
                                 abs_class_counts=total_gt_class_counts,
                                 )
    # show the figure
    plt.show()

    fig1.savefig("user_label_confusion_matrix.pdf")
    fig2.savefig("gt_label_confusion_matrix_rec.pdf")
    fig3.savefig("gt_label_confusion_matrix_im.pdf")


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
