from skimage.io import imread, imsave
from pathlib import Path
import numpy as np
from tqdm import auto

def main():

    # image_names = [
    #     "0000_0003_00.png",
    #     "0000_0006_00.png",
    #     "0000_0013_00.png",
    #     "0000_0026_00.png",
    #     "0000_0029_00.png",
    #     "0000_0042_00.png",
    #     "0000_0049_00.png",
    #     "0000_0064_00.png",
    #     "0000_0066_00.png",
    #     "0000_0076_00.png",
    #     "0000_0105_00.png",
    #     "0000_0112_00.png",
    #     "0000_0203_00.png",
    #     "0000_0293_00.png",
    #     "0000_0439_00.png",
    #     "0000_0468_00.png",
    #     "0000_0498_00.png"
    # ]

    # model_folder = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_10_29_22-01-32_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early/"

    # MTurk model
    model_folder = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"


    stage = "detail"

    # result_subfolder = "affect_net_detail_test"
    result_subfolder = "affectnet_validation_new_split_detail_test"
    # result_subfolder = "afew_test_new_split_detail_test"
    image_types = ["inputs",
                   "geometry_coarse",
                   "geometry_detail"]

    axis = 0
    # axis = 1
    composed_ims = []
    im_names = Path(model_folder) / stage / result_subfolder / "inputs"
    image_names = [p.name for p in sorted(list(im_names.glob("*.png")))]

    for im_name in auto.tqdm(image_names):
        images = []
        for image_type in image_types:
            im_path = Path(model_folder) / stage / result_subfolder / image_type / im_name
            images += [imread(im_path)]
        composed_im = np.concatenate(images, axis=axis)
        composed_ims += [composed_im]
        outfolder = Path(model_folder) / stage / result_subfolder / "compositions" / "_".join( image_types)
        outfolder.mkdir(exist_ok=True, parents=True)
        imsave( outfolder / im_name, composed_im)

    final_composed_im = np.concatenate(composed_ims, axis=1-axis)
    imsave(outfolder / "composition.png", final_composed_im)

    print(f"Results in {Path(model_folder) / stage / result_subfolder / 'compositions'  }")

if __name__ == '__main__':
    main()