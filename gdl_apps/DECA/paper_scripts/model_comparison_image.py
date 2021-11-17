from skimage.io import imread, imsave
from pathlib import Path
import numpy as np
from tqdm import auto
import datetime

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
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

    out_folder = Path("/is/cluster/work/rdanecek/emoca/figures/") / time
    out_folder.mkdir(exist_ok=True, parents=True)

    model_folder = "/is/cluster/work/rdanecek/emoca/finetune_deca/"
    model_names = []
    # # model_names += ["2021_11_05_00-26-38_-7399861690108686758_afft_Face3DDFAModule"] #mobilenet
    # model_names += ["2021_11_05_00-23-14_-1727155792220193876_afft_Face3DDFAModule"] # resnet
    # model_names += ["2021_11_06_01-01-38_8742010531224499117_afft_Deep3DFaceModule"]
    # model_names += ["2021_03_26_15-05-56_DECA_original"]
    # model_names += ["2021_10_29_22-01-32_ExpDECA_Affec_para_NoRing_EmoB_F2_DeSegrend_BlackB_Aug_early"]


    # model_names += ["2021_11_15_17-40-45_-5868754668879675020_Face3DDFAModule"]
    # model_names += ["2021_11_13_03-43-40_MGCNet"]
    # model_names += ["2021_11_15_17-09-34_6754141025581837735_Deep3DFaceModule"]

    model_names += ["2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2"]
    model_names += ["2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"]
# "/is/cluster/work/rdanecek/emoca/finetune_deca//detail"
    stage = "detail"

    # result_subfolder = "affect_net_detail_test"
    result_subfolder = "affect_net_mturk_detail_test"
    # image_types = ["inputs",
    #                "geometry_coarse",
    #                "geometry_detail"]
    image_type = "geometry_detail"
    # image_type_deca = "geometry_coarse"
    image_type_deca = "geometry_detail"
    image_inputs = "inputs"

    out_folder = out_folder / image_type
    out_folder.mkdir(exist_ok=True, parents=True)

    axis = 0
    # axis = 1
    composed_ims = []

    input_imlist = (Path(model_folder) / model_names[-1]  / stage / result_subfolder / "inputs").glob("*.png")

    input_im_names = sorted(list(input_imlist))
    rec_image_names = {}
    N = 499
    for model_name in model_names:
        if "MGCNet" in model_name:
            rec_image_names[model_name] = [p for p in sorted(list((Path(model_folder) / model_name / "detail" / "inputs"  ).glob("**/GeoOrigin.jpg")))]
            print(len(rec_image_names[model_name]))
        elif "DECA" in model_name:
            rec_image_names[model_name] = [p for p in sorted(
                list((Path(model_folder) / model_name / "detail" / result_subfolder / image_type_deca).glob("*.png")))]
            # N = len(rec_image_names[model_name])

        else:
            rec_image_names[model_name] = [p for p in sorted(list((Path(model_folder) / model_name  / "detail" / result_subfolder /image_type ).glob("*.png")))]
            # N = len(rec_image_names[model_name])


    for im_i in auto.tqdm(range(N)):
        images = []
        # for image_type in image_types:

        im_path = input_im_names[im_i]
        images += [imread(im_path)]
        for mi in range(len(model_names)):
            model = model_names[mi]
            im_path = rec_image_names[model][im_i]
            images += [imread(im_path)]

        composed_im = np.concatenate(images, axis=axis)
        composed_ims += [composed_im]
        imsave(out_folder / im_path.name, composed_im)

    # final_composed_im = np.concatenate(composed_ims, axis=1-axis)
    # imsave(out_folder / "composition.png", final_composed_im)

    with open(out_folder / "info.txt", "w") as f:
        f.write("models:\n")
        f.write("\n".join(model_names))

    print(f"Results in {out_folder}")

if __name__ == '__main__':
    main()