from skimage.io import imread, imsave
from pathlib import Path
import numpy as np
from tqdm import auto
import datetime

def main():

    image_names = [
        "0000_0003_00.png",
        "0000_0006_00.png",
        "0000_0013_00.png",
        "0000_0026_00.png",
        "0000_0029_00.png",
        "0000_0042_00.png",
        "0000_0049_00.png",
        "0000_0064_00.png",
        "0000_0066_00.png",
        "0000_0076_00.png",
        "0000_0105_00.png",
        "0000_0112_00.png",
        "0000_0203_00.png",
        "0000_0293_00.png",
        "0000_0439_00.png",
        "0000_0468_00.png",
        "0000_0498_00.png"
    ]
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

    out_folder = Path("/is/cluster/work/rdanecek/emoca/figures/") / time
    out_folder.mkdir(exist_ok=True, parents=True)

    model_folder = "/is/cluster/work/rdanecek/emoca/finetune_deca/"
    model_names = []
    # model_names += ["2021_11_05_00-26-38_-7399861690108686758_afft_Face3DDFAModule"] #mobilenet
    model_names += ["2021_11_05_00-23-14_-1727155792220193876_afft_Face3DDFAModule"] # resnet
    model_names += ["2021_11_06_01-01-38_8742010531224499117_afft_Deep3DFaceModule"]
    model_names += ["2021_03_26_15-05-56_DECA_original"]
    model_names += ["2021_10_29_22-01-32_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]

    stage = "detail"

    result_subfolder = "affect_net_detail_test"
    # image_types = ["inputs",
    #                "geometry_coarse",
    #                "geometry_detail"]
    image_type = "geometry_coarse"
    image_inputs = "inputs"

    out_folder = out_folder / image_type
    out_folder.mkdir(exist_ok=True, parents=True)

    axis = 0
    # axis = 1
    composed_ims = []

    for im_i, im_name in enumerate(auto.tqdm(image_names)):
        images = []
        # for image_type in image_types:
        for mi, model in enumerate(model_names):
            if mi == 0:
                im_path = Path(model_folder) / model / stage / result_subfolder / image_inputs / im_name
                images += [imread(im_path)]
            im_path = Path(model_folder) / model / stage / result_subfolder / image_type / im_name
            images += [imread(im_path)]
        composed_im = np.concatenate(images, axis=axis)
        composed_ims += [composed_im]
        imsave(out_folder / im_name, composed_im)

    final_composed_im = np.concatenate(composed_ims, axis=1-axis)
    imsave(out_folder / "composition.png", final_composed_im)

    with open(out_folder / "info.txt", "w") as f:
        f.write("models:\n")
        f.write("\n".join(model_names))

    print(f"Results in {out_folder}")

if __name__ == '__main__':
    main()