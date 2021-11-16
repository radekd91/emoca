import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import auto
from skimage.io import imread, imsave


def create_comparison_image(image_input, image_output_left, image_output_right):
    # Concatenate the images horizontally and return them as a numpy array
    image_output = np.hstack((image_output_left, image_input, image_output_right))
    return image_output

def create_mturk_experiment(input_image_path, output_image_path_1,
                            output_images_path_2,
                            output_path,
                            mask_image_path_1=None,
                            mask_image_path_2=None,
                            ):
    # find all png files in the folders using pathlib and sort them
    input_image_path = Path(input_image_path)
    output_image_path_1 = Path(output_image_path_1)
    output_images_path_2 = Path(output_images_path_2)

    input_image_list = sorted(list(input_image_path.glob("*.png")))
    output_image_list_1 = sorted(list(output_image_path_1.glob("*.png")))
    if mask_image_path_1 is not None:
        mask_image_path_1 = Path(mask_image_path_1)
        mask_image_list_1 = sorted(list(mask_image_path_1.glob("*.png")))
    else:
        mask_image_list_1 = None

    output_image_list_2 = sorted(list(output_images_path_2.glob("*.png")))
    if mask_image_path_2 is not None:
        mask_image_path_2 = Path(mask_image_path_2)
        mask_image_list_2 = sorted(list(mask_image_path_2.glob("*.png")))
    else:
        mask_image_list_2 = None

    output_path.mkdir(parents=True, exist_ok=True)

    assert len(input_image_list) == len(output_image_list_1) == len(output_image_list_2)
    if mask_image_list_1 is not None:
        assert len(input_image_list) == len(mask_image_list_1)
    if mask_image_path_2 is not None:
        assert len(input_image_list) == len(mask_image_list_2)

    N = len(input_image_list)

    # create a pandas table with the following columns: filename, was_swapped
    df = pd.DataFrame(columns=["filename", "was_swapped"])

    for i in auto.tqdm(range(N)):
        # read the images
        input_image = imread(str(input_image_list[i]))
        output_image_1 = imread(str(output_image_list_1[i]))
        output_image_2 = imread(str(output_image_list_2[i]))

        # check that the filenames match
        assert input_image_list[i].stem == output_image_list_1[i].stem == output_image_list_2[i].stem

        if mask_image_list_1 is not None:
            # check the filenames match
            assert input_image_list[i].stem == mask_image_list_1[i].stem
            mask_image_1 = imread(str(mask_image_list_1[i]))
            # apply the mask to the output image_output
            output_image_1[mask_image_1 == 0] = 0
        if mask_image_list_2 is not None:
            # check the filenames match
            assert input_image_list[i].stem == mask_image_list_2[i].stem
            mask_image_2 = imread(str(mask_image_list_2[i]))
            # apply the mask to the output image_output
            output_image_2[mask_image_2 == 0] = 0

        # swap the two output images with 50% chance
        if np.random.rand() > 0.5:
            output_image_1, output_image_2 = output_image_2, output_image_1
            swapped = True
        else:
            swapped = False

        # create the comparison image
        comparison_image = create_comparison_image(input_image, output_image_1, output_image_2)

        # save the comparison image to the output folder
        outfname =  f"{i:04d}.png"
        imsave(str(output_path / outfname), comparison_image)
        # add the filename and whether the images were swapped to the dataframe
        df.loc[i] = [str(outfname), swapped]

    # save the table to the output_path
    df.to_csv(str(output_path / "data.csv"), index=False)

    # create an info.txt file
    with open(str(output_path / "info.txt"), "w") as f:
        # write the used folders to the info.txt file
        f.write(f"input_image_path: {input_image_path}\n")
        f.write(f"output_image_path_1: {output_image_path_1}\n")
        f.write(f"output_images_path_2: {output_images_path_2}\n")
        if mask_image_path_1 is not None:
            f.write(f"mask_image_path_1: {mask_image_path_1}\n")
        if mask_image_path_2 is not None:
            f.write(f"mask_image_path_2: {mask_image_path_2}\n")

    # images created
    print(f"{N} images created")


def emoca_detail_vs_deca_detail():
    input_images = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/inputs")
    path_emoca_detail = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_detail")
    path_emoca_mask = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/mask")
    path_deca_detail = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/geometry_detail")
    path_deca_mask = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/mask")

    mturk_root = Path("/is/cluster/work/rdanecek/emoca/mturk_study")
    output_folder = "EmocaDetail-DecaDetail"

    output_path = mturk_root / output_folder
    create_mturk_experiment(input_images, path_emoca_detail, path_deca_detail, output_path, path_emoca_mask,
                            path_deca_mask)


def emoca_coarse_vs_deca_coarse():
    input_images = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/inputs")
    path_emoca_coarse = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_coarse")
    path_emoca_mask = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/mask")
    path_deca_coarse = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/geometry_coarse")
    path_deca_mask = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2/detail/affect_net_mturk_detail_test/mask")

    mturk_root = Path("/is/cluster/work/rdanecek/emoca/mturk_study")
    output_folder = "EmocaCoarse-DecaCoarse"

    output_path = mturk_root / output_folder
    create_mturk_experiment(input_images, path_emoca_coarse, path_deca_coarse, output_path, path_emoca_mask,
                            path_deca_mask)


def emoca_detail_vs_method(method_path, method_name):
    input_images = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/inputs")
    path_emoca_coarse = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_coarse")
    path_emoca_mask = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/mask")

    mturk_root = Path("/is/cluster/work/rdanecek/emoca/mturk_study")
    output_folder = f"EmocaDetail-{method_name}"

    output_path = mturk_root / output_folder
    create_mturk_experiment(input_images, path_emoca_coarse, method_path, output_path, path_emoca_mask)


def emoce_coarse_vs_method(method_path, method_name):
    input_images = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/inputs")
    path_emoca_coarse = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_coarse")
    path_emoca_mask = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/mask")

    mturk_root = Path("/is/cluster/work/rdanecek/emoca/mturk_study")
    output_folder = f"EmocaCoarse-{method_name}"

    output_path = mturk_root / output_folder
    create_mturk_experiment(input_images, path_emoca_coarse, method_path, output_path, path_emoca_mask)


def main():
    emoca_detail_vs_deca_detail()
    # emoca_coarse_vs_deca_coarse()
    #
    # # dictionary of methods and method their image paths:
    # methods = {}
    # methods["3DDFA_v2"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-40-45_-5868754668879675020_Face3DDFAModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    # methods["Deep3DFace"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-09-34_6754141025581837735_Deep3DFaceModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    # methods["MGCNet"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_MGCNet/detail"




if __name__ == "__main__":
    main()
