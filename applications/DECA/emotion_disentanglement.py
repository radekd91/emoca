from interactive_deca_decoder import load_deca
from pathlib import Path
from omegaconf import OmegaConf
from skimage.io import imread
from utils.image import numpy_image_to_torch
import torch
import matplotlib.pyplot as plt
import numpy as np
from train_expdeca import prepare_data
from torch.utils.data.dataloader import DataLoader


def exchange_codes(vals1, vals2, codes_to_exchange):
    values_21 = {**vals2}
    values_12 = {**vals1}

    if 'jawpose' in codes_to_exchange and 'globalpose' in codes_to_exchange:
        codes_to_exchange.remove('jawpose')
        codes_to_exchange.remove('globalpose')
        if 'posecode' not in codes_to_exchange:
            codes_to_exchange += ['posecode']

    for code in codes_to_exchange:
        if code in ['jawpose', 'globalpose']:
            jaw1 = vals1['posecode'][:, 3:]
            jaw2 = vals2['posecode'][:, 3:]
            glob1 = vals1['posecode'][:, :3]
            glob2 = vals2['posecode'][:, :3]
            pose12 = torch.clone(vals1['posecode'])
            pose21 = torch.clone(vals2['posecode'])

            if code == 'jawpose':
                pose21[:, 3:] = jaw1
                pose12[:, 3:] = jaw2

            if code == 'globalpose':
                pose21[:, 3:] = glob1
                pose12[:, 3:] = glob2

            values_21['posecode'], values_12['posecode'] = pose21, pose12

        else:
            values_21[code], values_12[code] = vals1[code], vals2[code]
    return values_12, values_21


def decode(deca, values, training=True):
    with torch.no_grad():
        values = deca.decode(values, training=training)
        # losses = deca.compute_loss(values, training=False)

        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']
        visualizations, grid_image = deca._visualization_checkpoint(
            values['verts'],
            values['trans_verts'],
            values['ops'],
            uv_detail_normals,
            values,
            0,
            "",
            "",
            save=False
        )
        vis_dict = deca._create_visualizations_to_log("", visualizations, values, 0, indices=0)
    return values, vis_dict
    # return values, losses, vis_dict


def exchange_and_decode(deca, vals1, vals2, codes_to_exchange):
    values_12, values_21 = exchange_codes( vals1, vals2, codes_to_exchange)

    values_12, vis_dict_12 = decode(deca, values_12)
    values_21, vis_dict_21 = decode(deca, values_21)

    # values_12, losses_12, vis_dict_12 = decode(deca, values_12)
    # values_21, losses_21, vis_dict_21 = decode(deca, values_21)

    return [values_21, vis_dict_21], [values_12, vis_dict_12]
    # return [values_21, losses_21, vis_dict_21], [values_12, losses_12, vis_dict_12]


def visualize(vis_dict, title, axs=None, fig=None, ri=None):
    if axs is None:
        fig, axs = plt.subplots(1, 7)
        fig.suptitle(title, fontsize=16)
    # else:
        # axs.set_title(title + "\n", fontsize=16)
        # n_cols = 7
        # ax = []
        # for i in range(0,n_cols):
        #     ax += [fig.add_subplot(1, 7, i+1)]
            # ax += [fig.add_subplot(1, 7, ri+i+1)]
            # ax += [fig.add_subplot(1, 7, ri*n_cols+1 + i)]
        # plt.subplots(figsize=(15.0, 15.0), nrows=3, ncols=1, sharey=True)

    prefix = None
    for key in list(vis_dict.keys()):
        if 'detail__inputs' in key:
            start_idx = key.rfind('detail__inputs')
            prefix = key[:start_idx]
            # print(f"Prefix was found to be: '{prefix}'")
            break
    if prefix is None:
        print(vis_dict.keys())
        raise RuntimeError(f"Unknown disctionary content. Available keys {vis_dict.keys()}")


    axs[0].annotate(title, xy=(0, 0.5), xytext=(-axs[0].yaxis.labelpad, 0),
                    xycoords=axs[0].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    i = 0
    axs[i].imshow(vis_dict[f'{prefix}detail__inputs'])
    axs[i].set_title("input")
    i += 1
    # if f'{prefix}detail__landmarks_gt' in vis_dict.keys():
    #     ax[1].imshow(vis_dict[f'{prefix}detail__landmarks_gt'])
    axs[i].imshow(vis_dict[f'{prefix}detail__landmarks_predicted'])
    axs[i].set_title("predicted landmarks")
    i += 1
    if f'{prefix}detail__mask' in vis_dict.keys():
        axs[i].imshow(vis_dict[f'{prefix}detail__mask'])
        i += 1
    axs[i].imshow(vis_dict[f'{prefix}detail__geometry_coarse'])
    axs[i].set_title("coarse geometry")
    i += 1
    axs[i].imshow(vis_dict[f'{prefix}detail__geometry_detail'])
    axs[i].set_title("detail geometry")
    i += 1
    axs[i].imshow(vis_dict[f'{prefix}detail__output_images_coarse'])
    axs[i].set_title("coarse image")
    i += 1
    axs[i].imshow(vis_dict[f'{prefix}detail__output_images_detail'])
    axs[i].set_title("detail image")
    i += 1


def test(deca, img):
    img["image"] = img["image"].cuda()
    img["image"] = img["image"].view(1,3,224,224)



    vals = deca.encode(img, training=False)
    # vals = deca.decode(vals)
    vals, visdict = decode(deca, vals, training=False)
    return vals, visdict


def plot_comparison(names, visdicts):
    fig = plt.figure()
    # # fig, axs = plt.subplots(14, 7)
    # fig, big_ax = plt.subplots(figsize=(15.0, 15.0), nrows=len(names), ncols=1, sharey=True)
    # # fig.suptitle("Exchange results", fontsize=16)
    #
    # for row, big_ax in enumerate(big_ax, start=1):
    # #     big_ax.set_title(f"{names[row-1]} \n", fontsize=16)
    # #
    #     # Turn off axis lines and ticks of the big subplot
    #     # obs alpha is 0 in RGBA string!
    #     big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    #     # removes the white frame
    #     # big_ax._frameon = False

    n_cols = 7
    n_rows = len(names)
    axs = []
    gs = fig.add_gridspec(n_rows, n_cols)
    for i in range(0, n_rows):
        ax_row = []
        for j in range(0, n_cols):
            ax = fig.add_subplot(gs[i, j])
            # ax.set_title('Plot title ' + str(i))
            ax_row += [ax]
        axs += [ax_row]

    for i in range(len(names)):
        # visualize(visdicts[i], names[i], axs[i*n_cols:(i+1)*n_cols], fig, i)
        visualize(visdicts[i], names[i], axs[i], fig, i)
    fig.set_facecolor('w')
    plt.tight_layout()



def exchange_and_visualize(deca, img1_path_or_batch, img2_path_or_batch):
    if isinstance(img1_path_or_batch, (str, Path)):
        img1 = imread(img1_path_or_batch).astype(np.float32) / 255.
        img1_torch = numpy_image_to_torch(img1).view(1, 3, img1.shape[0], img1.shape[1]).cuda()
        img1_path_or_batch = {"image": img1_torch}
    elif not isinstance(img1_path_or_batch, dict):
        raise ValueError("img1_path_or_batch must be a path to an image or a dataset sample dict")

    if isinstance(img2_path_or_batch, (str, Path)):
        img2 = imread(img2_path_or_batch).astype(np.float32) / 255.
        img2_torch = numpy_image_to_torch(img2).view(1, 3, img2.shape[0], img2.shape[1]).cuda()
        img2_path_or_batch = {"image": img2_torch}
    elif not isinstance(img2_path_or_batch, dict):
        raise ValueError("img2_path_or_batch must be a path to an image or a dataset sample dict")

    values_img1, visdict1 = test(deca, img1_path_or_batch)
    values_img2, visdict2 = test(deca, img2_path_or_batch)

    # exchange codes
    vals21_s, vals12_s = exchange_and_decode(deca, values_img1, values_img2, ['shapecode'])
    vals21_d, vals12_d = exchange_and_decode(deca, values_img1, values_img2, ['detailcode'])
    vals21_e, vals12_e = exchange_and_decode(deca, values_img1, values_img2, ['expcode', 'jawpose'])

    vals21_sd, vals12_sd = exchange_and_decode(deca, values_img1, values_img2, ['shapecode', 'detailcode'])
    vals21_de, vals12_de = exchange_and_decode(deca, values_img1, values_img2, ['detailcode', 'expcode', 'jawpose'])
    vals21_se, vals12_se = exchange_and_decode(deca, values_img1, values_img2, ['shapecode', 'expcode', 'jawpose'])
    # visualize and analyze

    names1 = []
    visdicts1 = []
    names2 = []
    visdicts2 = []

    names1 += ["Input"]
    visdicts1 += [visdict1]
    names1 += ["Target"]
    visdicts1 += [visdict2]
    names2 += ["Input"]
    visdicts2 += [visdict2]
    names2 += ["Target"]
    visdicts2 += [visdict1]


    names1 += ["Shape exchange 1-2"]
    visdicts1 += [vals12_s[1]]

    names2 += ["Shape exchange 2-1"]
    visdicts2 += [vals21_s[1]]

    names1 += ["Detail exchange 1-2"]
    visdicts1 += [vals12_d[1]]

    names2 += ["Detail exchange 2-1"]
    visdicts2 += [vals21_d[1]]
    #
    names1 += [ "Expression exchange 1-2"]
    visdicts1 += [vals12_e[1]]

    names2 += ["Expression exchange 2-1"]
    visdicts2 += [vals21_e[1]]
    #
    names1 += ["Shape+Detail exchange 1-2"]
    visdicts1 += [vals12_sd[1]]

    names2 += ["Shape+Detail exchange 2-1"]
    visdicts2 += [vals21_sd[1]]

    names1 += ["Detail+expression exchange 1-2"]
    visdicts1 += [vals12_de[1]]

    names2 += ["Detail+expression exchange 2-1"]
    visdicts2 += [vals21_de[1]]
    #
    names1 += ["Shape+expression exchange 1-2"]
    visdicts1 += [vals12_se[1]]

    names2 += ["Shape+expression exchange 2-1"]
    visdicts2 += [vals21_se[1]]

    plot_comparison(names1, visdicts1)
    plot_comparison(names2, visdicts2)
    plt.show()

    #
    # i = 0
    # visualize(visdict1, "Img 1", axs[i], fig, i)
    # i += 1
    # visualize(visdict2, "Img 2", axs[i], fig, i)
    # i += 1
    #
    # visualize(vals12_s[1], "Shape exchange 1-2", axs[i], fig, i)
    # i += 1
    # visualize(vals21_s[1], "Shape exchange 2-1", axs[i], fig, i)
    # i += 1
    #
    # visualize(vals12_d[1], "Detail exchange 1-2", axs[i], fig, i)
    # i += 1
    # visualize(vals21_d[1], "Detail exchange 2-1", axs[i], fig, i)
    # i += 1
    #
    # visualize(vals12_e[1], "Expression exchange 1-2", axs[i], fig, i)
    # i += 1
    # visualize(vals21_e[1], "Expression exchange 2-1", axs[i], fig, i)
    # i += 1
    #
    # visualize(vals12_sd[1], "Shape+Detail exchange 1-2", axs[i], fig, i)
    # i += 1
    # visualize(vals21_sd[1], "Shape+Detail exchange 2-1", axs[i], fig, i)
    # i += 1
    #
    # visualize(vals12_de[1], "Detail+expression exchange 1-2", axs[i], fig, i)
    # i += 1
    # visualize(vals21_de[1], "Detail+expression exchange 2-1", axs[i], fig, i)
    # i += 1
    #
    # visualize(vals12_se[1], "Shape+expression exchange 1-2", axs[i], fig, i)
    # i += 1
    # visualize(vals21_se[1], "Shape+expression exchange 2-1", axs[i], fig, i)
    # i += 1
    #
    # fig.set_facecolor('w')
    # plt.tight_layout()
    # plt.show()
    # print("Done")



def main_aff_wild_selection(deca):
    target_image_path = Path("/ps/scratch/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10")
    target_images = [
        target_image_path / "VA_Set/detections/Train_Set/119-30-848x480/000640_000.png", # Octavia
        target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/000480_000.png", # Rachel 1
        target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/002805_000.png", # Rachel 1
        target_image_path / "VA_Set/detections/Train_Set/82-25-854x480/003899_000.png", # Rachel 2
        target_image_path / "VA_Set/detections/Train_Set/111-25-1920x1080/000685_000.png", # disgusted guy
        target_image_path / "VA_Set/detections/Train_Set/122-60-1920x1080-1/001739_000.png", # crazy youtuber
        target_image_path / "VA_Set/detections/Train_Set/122-60-1920x1080-1/001644_000.png", # crazy youtuber
        target_image_path / "VA_Set/detections/Train_Set/122-60-1920x1080-1/000048_000.png", # crazy youtuber
        target_image_path / "VA_Set/detections/Train_Set/135-24-1920x1080/000001_000.png", # couple
        target_image_path / "VA_Set/detections/Train_Set/135-24-1920x1080/000080_001.png", # couple
    ]
    exchange_and_visualize(deca, target_images[0], target_images[2])



def main_affectnet(deca, conf, stage ):

    conf[stage].data.input_dir = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet"
    conf[stage].data.output_dir = "/is/cluster/work/rdanecek/data/affectnet"

    dm, name = prepare_data(conf[stage])

    dm.val_batch_size = 1
    dm.ring_size = None
    dm.ring_type = None
    # dm.root_dir = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet"

    dm.prepare_data()
    dm.setup()
    dm.val_dataloader()

    import pytorch_lightning as pl
    # pl.utilities.seed.seed_everything(0, workers=True)
    pl.utilities.seed.seed_everything(0)

    dl = DataLoader(dm.validation_set, shuffle=True, num_workers=dm.num_workers,
                          batch_size=dm.val_batch_size, drop_last=dm.drop_last)

    from tqdm import auto
    prev_batch = 0
    for bi, batch in auto.tqdm(enumerate(dl)):
        if bi == 0:
            prev_batch = batch
            continue
        exchange_and_visualize(deca, batch, prev_batch)
        prev_batch = batch


def main():
    # path_to_models = "/is/cluster/work/rdanecek/emoca/finetune_deca/"
    ## relative_to_path = None
    ## replace_root_path = None

    path_to_models = "/ps/scratch/rdanecek/emoca/finetune_deca/"
    relative_to_path = '/ps/scratch/'
    replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
    stage = 'detail'

    ## ExpDECA on AffectNet, emotion loss
    run_name = "2021_04_20_18-36-33_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"

    # DECA on DECA dataset, no ring, emotion loss
    # run_name = "2021_04_23_17-06-29_ExpDECA_DecaD_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"

    # ExpDECA on AffectNet, Expression ring without geometric losses (exchange punished for emotion mismatched only)
    # run_name += '2021_05_02_12-43-06_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_early'
    # run_name += '2021_05_02_12-42-01_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_DwC_early'
    # run_name += '2021_05_02_12-37-20_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_DwC_early'
    # run_name += '2021_05_02_12-36-00_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_DwC_early'
    # run_name += '2021_05_02_12-35-44_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_early'
    # run_name += '2021_05_02_12-34-47_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_early'

    # ExpDECA on AffectNet, Expression ring with geometric losses
    # run_name = '2021_05_07_20-48-30_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_DwC_early'
    # run_name = '2021_05_07_20-46-09_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_early'
    # run_name = '2021_05_07_20-45-33_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_DwC_early'
    # run_name = '2021_05_07_20-36-43_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_early'


    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)

    deca = load_deca(conf, stage, 'best', relative_to_path, replace_root_path)
    deca.cuda()
    deca.eval()

    # main_aff_wild_selection(deca)
    main_affectnet(deca, conf, stage)


if __name__ == '__main__':
    main()
