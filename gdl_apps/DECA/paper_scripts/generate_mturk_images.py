import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import auto
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

def create_comparison_image(image_input, image_output_left, image_output_right):
    # Concatenate the images horizontally and return them as a numpy array
    image_output = np.hstack((image_output_left, image_input, image_output_right))
    return image_output

def create_mturk_experiment(input_image_path, output_image_path_1,
                            output_images_path_2,
                            output_path,
                            mask_image_path_1=None,
                            mask_image_path_2=None,
                            pattern_1=None,
                            pattern_2=None,
                            mask_pattern_1=None,
                            mask_pattern_2=None,
                            ):
    pattern_1 = "*.png" if pattern_1 is None else pattern_1
    pattern_2 = "*.png" if pattern_2 is None else pattern_2
    mask_pattern_1 = "*.png" if mask_pattern_1 is None else mask_pattern_1
    mask_pattern_2 = "*.png" if mask_pattern_2 is None else mask_pattern_2

    # find all png files in the folders using pathlib and sort them
    input_image_path = Path(input_image_path)
    output_image_path_1 = Path(output_image_path_1)
    output_images_path_2 = Path(output_images_path_2)

    input_image_list = sorted(list(input_image_path.glob(pattern_1)))
    output_image_list_1 = sorted(list(output_image_path_1.glob(pattern_1)))
    if mask_image_path_1 is not None:
        mask_image_path_1 = Path(mask_image_path_1)
        mask_image_list_1 = sorted(list(mask_image_path_1.glob(mask_pattern_1)))
    else:
        mask_image_list_1 = None

    output_image_list_2 = sorted(list(output_images_path_2.glob(pattern_2)))
    if mask_image_path_2 is not None:
        mask_image_path_2 = Path(mask_image_path_2)
        mask_image_list_2 = sorted(list(mask_image_path_2.glob(mask_pattern_2)))
    else:
        mask_image_list_2 = None

    output_path.mkdir(parents=True, exist_ok=True)

    N = min(len(input_image_list), len(output_image_list_1), len(output_image_list_2))
    input_image_list = input_image_list[:N]
    output_image_list_1 = output_image_list_1[:N]
    output_image_list_2 = output_image_list_2[:N]
    if mask_image_list_1 is not None:
        mask_image_list_1 = mask_image_list_1[:N]
    if mask_image_list_2 is not None:
        mask_image_list_2 = mask_image_list_2[:N]

    assert len(input_image_list) == len(output_image_list_1) == len(output_image_list_2)
    if mask_image_list_1 is not None:
        assert len(input_image_list) == len(mask_image_list_1)
    if mask_image_path_2 is not None:
        assert len(input_image_list) == len(mask_image_list_2)

    # create a pandas table with the following columns: filename, was_swapped
    df = pd.DataFrame(columns=["filename", "was_swapped"])

    for i in auto.tqdm(range(N)):
        # read the images
        input_image = imread(str(input_image_list[i]))
        output_image_1 = imread(str(output_image_list_1[i]))
        output_image_2 = imread(str(output_image_list_2[i]))

        # check that the filenames match
        assert input_image_list[i].stem == output_image_list_1[i].stem #== output_image_list_2[i].stem

        if mask_image_list_1 is not None:
            # check the filenames match
            assert input_image_list[i].stem == mask_image_list_1[i].stem
            mask_image_1 = imread(str(mask_image_list_1[i]))
            # if mask has only one channel, duplicate it to match the input image
            if len(mask_image_1.shape) == 2 or mask_image_1.shape[2] == 1:
                mask_image_1 = np.stack((mask_image_1, mask_image_1, mask_image_1), axis=2)

            # convert the mask and the image to float32
            mask_image_1 = mask_image_1.astype(np.float32)
            output_image_1 = output_image_1.astype(np.float32) / 255.0
            output_image_1 = output_image_1 * mask_image_1

            # apply the mask to the output image_output
            # output_image_1[mask_image_1 == 0] = 0
        if mask_image_list_2 is not None:
            # check the filenames match
            assert input_image_list[i].stem == mask_image_list_2[i].stem
            mask_image_2 = imread(str(mask_image_list_2[i]))
            # if mask has only one channel, duplicate it to match the input image
            if len(mask_image_2.shape) == 2 or mask_image_2.shape[2] == 1:
                mask_image_2 = np.stack((mask_image_2, mask_image_2, mask_image_2), axis=2)
            # multiply the mask with the input image
            mask_image_2 = mask_image_2.astype(np.float32)
            output_image_2 = output_image_2.astype(np.float32) / 255.0
            output_image_2 = output_image_2 * mask_image_2

            # apply the mask to the output image_output
            # output_image_2[mask_image_2 == 0] = 0

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

    mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study")
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

    mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study")
    output_folder = "EmocaCoarse-DecaCoarse"

    output_path = mturk_root / output_folder
    create_mturk_experiment(input_images, path_emoca_coarse, path_deca_coarse, output_path, path_emoca_mask,
                            path_deca_mask)


def emoca_detail_vs_method(method_path, method_name, method_image_pattern, method_mask_pattern):
    input_images = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/inputs")
    path_emoca_coarse = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_detail")
    path_emoca_mask = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/mask")

    mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study")
    output_folder = f"EmocaDetail-{method_name}"

    output_path = mturk_root / output_folder
    create_mturk_experiment(input_images, path_emoca_coarse, method_path, output_path, path_emoca_mask,
                            pattern_2=method_image_pattern,
                            mask_pattern_2=method_mask_pattern,
                            )


def emoca_coarse_vs_method(method_path, method_name, method_image_pattern, method_mask_pattern):
    input_images = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/inputs")
    path_emoca_coarse = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/geometry_coarse")
    path_emoca_mask = Path(
        "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/affect_net_mturk_detail_test/mask")

    mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study")
    output_folder = f"EmocaCoarse-{method_name}"

    output_path = mturk_root / output_folder
    create_mturk_experiment(input_images, path_emoca_coarse, method_path, output_path, path_emoca_mask,
                            pattern_2=method_image_pattern,
                            mask_pattern_2=method_mask_pattern,
                            )


def emoca_vs_method(method_path, method_name, method_image_pattern=None, method_mask_pattern=None):
    emoca_coarse_vs_method(method_path, method_name, method_image_pattern, method_mask_pattern)
    emoca_detail_vs_method(method_path, method_name, method_image_pattern, method_mask_pattern)


def generate_mturk_images():
    emoca_detail_vs_deca_detail()
    emoca_coarse_vs_deca_coarse()
    #
    # # dictionary of methods and method their image paths:
    methods = {}
    methods[
        "3DDFA_v2"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-40-45_-5868754668879675020_Face3DDFAModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    methods[
        "Deep3DFace"] = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-09-34_6754141025581837735_Deep3DFaceModule/detail/affect_net_mturk_detail_test/geometry_coarse"
    for name, path in methods.items():
        emoca_vs_method(path, name)

    emoca_vs_method("/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_MGCNet/detail", "MGCNet",
                    method_image_pattern="**/GeoOrigin.jpg")

def filter_mturk_images(max_samples, threshold=0.5, seed=0):
    df = pd.read_csv("/ps/project/EmotionalFacialAnimation/data/affectnet/Automatically_Annotated/Automatically_annotated_file_list/representative.csv")
    # print the number of images in the dataset
    print(df.shape)
    # tale only first 499 rows # 499 because some methods somehow failed for the last two (didn't save the image)
    df = df.iloc[:499]

    # get a numpy array of indices to df, where "arousal" is either higher than 0.5 or lower than -0.5
    arousal_indices1 = np.where(df["arousal"] > threshold)[0]
    arousal_indices2 = np.where(df["arousal"] < -threshold)[0]
    # take the union of the two arrays
    arousal_indices = np.union1d(arousal_indices1, arousal_indices2)
    # do the same for valence
    valence_indices1 = np.where(df["valence"] > threshold)[0]
    valence_indices2 = np.where(df["valence"] < -threshold)[0]
    valence_indices = np.union1d(valence_indices1, valence_indices2)
    # get the union of the two
    indices = np.union1d(arousal_indices, valence_indices)
    # get the number of elements that fullfill this condition
    print("Filtered out: ", indices.shape)

    np.random.seed(seed)
    np.random.shuffle(indices)

    mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study")
    np.save(mturk_root / f"mturk_indices_{threshold:0.2f}_{max_samples:04d}_{seed}.npy", indices)

    return indices[:max_samples]


def create_mturk_csv(indices, output_path, n_repeat_at_the_end=20, catch_sample_indices=None):

    to_repeat = indices[:n_repeat_at_the_end]
    # append the indices to the end of the list
    indices = np.concatenate([indices, to_repeat])

    if catch_sample_indices is not None:
        indices = indices.tolist()
        catch_sample_indices.sort(reverse=True)
        for catch_sample_idx in catch_sample_indices:
            indices[catch_sample_idx:catch_sample_idx] = [-1]



    # create a new table with a column named "images"
    df = pd.DataFrame(columns=["images"])
    df_rel = pd.DataFrame(columns=["images"])
    for i in range(len(indices)):
        idx = indices[i]
        if indices[i] == -1:
            df.loc[i] = Path(output_path).parent / "catch_sample.png"
            df_rel.loc[i] = "../catch_sample.png"
        else:
            df.loc[i] = [f"{str(output_path)}/{idx:04d}.png"]
            df_rel.loc[i] = f"{idx:04d}.png"
    return df, df_rel


def filter_and_generate_csv(num_samples, indices=None, seed=0, threshold=0.5):
    if indices is None:
        indices = filter_mturk_images(max_samples=num_samples, threshold=threshold, seed=seed)
    else: 
        np.random.seed(0)
        np.random.shuffle(indices)

    mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study")
    experiments = [p for p in list(mturk_root.glob("*")) if p.is_dir()]

    n_repeat_at_the_end = 15
    catch_sample_indices = [20, 50]

    for experiment in experiments:
        df, df_rel = create_mturk_csv(indices, str(experiment), n_repeat_at_the_end, catch_sample_indices)
        if indices is None:
            outfile = str(experiment) + f"/mturk_images_{num_samples}_{seed}_{threshold:0.2f}"
        else: 
            outfile = str(experiment) + f"/mturk_images_selected"
            np.save( str(experiment) + f"/selection_indices_{seed}", indices)
        df.to_csv(outfile + ".csv", index=False)
        df.to_csv(outfile + "_rel.csv", index=False)
        print("Wrote dataframe to: ", outfile)


def filter_and_generate_final_csv():
    # threshold = 0.5
    # seed = 0
    # filter_and_generate_csv(50, seed=seed, threshold= threshold)
    # filter_and_generate_csv(75, seed=seed, threshold= threshold)
    # filter_and_generate_csv(100, seed=seed, threshold= threshold)
    indices = np.array(selected_indices, dtype=np.int32)
    indices.sort()
    print (indices.shape)
    filter_and_generate_csv(100, indices=indices)


def sanity_check():
    #path  = "/ps/scratch/ps_shared/rdanecek/mturk_study/EmocaDetail-Deep3DFace/mturk_images_50_0_0.75.csv"
    #path = "/ps/scratch/ps_shared/rdanecek/mturk_study/EmocaDetail-Deep3DFace/mturk_images_100_0_0.75.csv"
    # path = "/ps/scratch/ps_shared/rdanecek/mturk_study/EmocaCoarse-Deep3DFace/mturk_images_100_0_0.85.csv"
    path = "/ps/scratch/ps_shared/rdanecek/mturk_study/EmocaDetail-Deep3DFace/mturk_images_selected.csv"
    df = pd.read_csv(path)
    # for reach row
    for i in range(df.shape[0]):
        # get the image path
        image_path = df.iloc[i]["images"]
        # load the image with skimage
        image = imread(image_path)
        # show the image with matplotlib
        plt.imshow(image)
        plt.show()


def create_catch_sample():
    mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study")
    catch_sample_path1 = mturk_root / "catch_1.png"
    catch_sample_path2 = mturk_root / "catch_2.png"
    catch_sample_mask1 = mturk_root / "catch_1_mask.png"
    catch_sample_mask2 = mturk_root / "catch_2_mask.png"
    input = mturk_root / "catch_1_input.png"

    # load the images
    image1 = imread(catch_sample_path1)
    image2 = imread(catch_sample_path2)
    mask1 = imread(catch_sample_mask1)
    mask2 = imread(catch_sample_mask2)
    input_image = imread(input)

    mask1 = mask1.astype(np.float32)
    image1 = image1.astype(np.float32) / 255.0
    image1 = image1 * mask1

    mask2 = mask2.astype(np.float32)
    image2 = image2.astype(np.float32) / 255.0
    image2 = image2 * mask2

    # concatenate image1, input and image2 together
    image = np.concatenate([image1, input_image, image2], axis=1)
    #save the image
    imsave(str(mturk_root / "catch_sample.png"), image)


selected_indices = [0, 1, 5, 6 , 8, 9, 10, 12, 19,  20, 23, 26, 27, 28, 30, 31, 35, 37, 49, 51, 54, 55, 57, 59, 63, 67, 
70, 75, 77, 88, 96, 97, 101, 105, 107 , 110, 111, 125, 127, 138, 143, 142, 166, 169, 168, 176, 210, 207, 235, 248, 191, 244,286, 
164, 275, 295, 277, 300, 323, 345, 349, 338, 357, 355, 361, 377, 378, 384, 383, 407, 429,  452, 433, 464, 467, 450, 489, 482, 490, 472, 405
]


def compile_final_files():
    mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study")
    cloud_root = Path("https://emoca.s3.eu-central-1.amazonaws.com/")
    experiments = [p for p in list(mturk_root.glob("*")) if p.is_dir()]

    # create a new dataframe with images
    df = pd.DataFrame(columns=["images"])
    df_rel = pd.DataFrame(columns=["images"])
    df_cloud = pd.DataFrame(columns=["images"])

    for exp in experiments:
        table = pd.read_csv(exp / "mturk_images_selected.csv")
        table_rel = pd.read_csv(exp / "mturk_images_selected_rel.csv")

        # get the image paths
        image_paths = table["images"].values
        image_paths_cloud = []
        image_paths_rel = []
        for i, path in enumerate( image_paths):
            image_path_rel = Path(path).relative_to(mturk_root)
            image_path_cloud = str(cloud_root / image_path_rel)
            # table_rel.loc[table_rel["images"] == str(image_paths_rel), "images"] = str(path)
            image_paths_rel += [str(image_path_rel)]
            image_paths_cloud += [str(image_path_cloud)]

        print(len(image_paths_cloud))
        paths = [";".join(image_paths.tolist())]
        paths_rel = [";".join(image_paths_rel)]
        paths_cloud = [";".join(image_paths_cloud)]

        # add the paths to the dataframe
        df = df.append(pd.DataFrame(paths, columns=["Images"]))
        df_rel = df_rel.append(pd.DataFrame(paths_rel, columns=["Images"]))
        df_cloud = df_cloud.append(pd.DataFrame(paths_cloud, columns=["Images"]))


    df.to_csv(mturk_root / "mturk_images_final.csv", index=False)
    df_rel.to_csv(mturk_root / "mturk_images_final_rel.csv", index=False)
    df_cloud.to_csv(mturk_root / "mturk_images_final_cloud.csv", index=False)



def main():
    #generate_mturk_images()
    #filter_and_generate_final_csv()
    #create_catch_sample()
    # sanity_check()
    # compile_final_files()



if __name__ == "__main__":
    main()
