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
from datasets.AffectNetDataModule import AffectNetDataModule, AffectNetExpressions


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


def decode(deca, values, batch=None, training=False):
    with torch.no_grad():
        values = deca.decode(values, training=training)
        if batch is not None and 'landmark' in batch.keys(): # a full batch came, including supervision
            losses = deca.compute_loss(values, batch, training=training)
        else:
            losses = None

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
    # return values, vis_dict
    return values, vis_dict, losses


def exchange_and_decode(deca, vals1, vals2, codes_to_exchange, batch1, batch2):
    values_12, values_21 = exchange_codes( vals1, vals2, codes_to_exchange)

    values_12, vis_dict_12, losses_12 = decode(deca, values_12, batch1)
    values_21, vis_dict_21, losses_21 = decode(deca, values_21, batch2)

    # return [values_21, vis_dict_21], [values_12, vis_dict_12]
    return [values_21, vis_dict_21, losses_21], [values_12, vis_dict_12, losses_12]


def visualize(vis_dict, title, values, losses, batch, axs=None, fig=None, ri=None):
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
        raise RuntimeError(f"Unknown dictionoary content. Available keys {vis_dict.keys()}")


    axs[0].annotate(title, xy=(0, 0.5), xytext=(-axs[0].yaxis.labelpad, 0),
                    xycoords=axs[0].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    i = 0
    axs[i].imshow(vis_dict[f'{prefix}detail__inputs'])

    stage = "detail"

    axs[i].set_title("input")

    if ri < 2:
        title = ""
        if f"{stage}_valence_gt" in values.keys():
            title += f'\nV_GT={ values[stage + "_valence_gt"][0].detach().cpu().item():+3.2f}'
            title += f'\nA_GT={ values[stage + "_arousal_gt"][0].detach().cpu().item():+3.2f}'
        if "affectnetexp" in batch.keys():
            title += f'\nE_GT={AffectNetExpressions(batch["affectnetexp"][0].detach().cpu().item()).name}'
        if f'{stage}_valence_input' in values.keys():
            title += f'\nV_EN={ values[stage + "_valence_input"][0].detach().cpu().item():+3.2f}'
            title += f'\nA_EN={ values[stage + "_arousal_input"][0].detach().cpu().item():+3.2f}'
        if f'{stage}_expression_input' in values.keys():
            title += f'\nE_EN={AffectNetExpressions(np.argmax(values[stage + "_expression_input"][i].detach().cpu().numpy())).name}'
        axs[i].text(256.0, 112.0, title, size=12, verticalalignment='center', horizontalalignment='left')

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
    # title = "coarse image"
    title = ""
    if f'coarse_valence_output' in values.keys():
        title += f'\nV_EN={ values["coarse_valence_output"][0].detach().cpu().item():+3.2f}'
        title += f'\nA_EN={ values["coarse_arousal_output"][0].detach().cpu().item():+3.2f}'
    if f'coarse_expression_output' in values.keys():
        title += f'\nA_EN={AffectNetExpressions(np.argmax(values["coarse_expression_output"][0].detach().cpu().numpy())).name}'
    axs[i].set_title("coarse image")
    axs[i].text(256.0, 112.0, title, size=12, verticalalignment='center', horizontalalignment='left')
    i += 1
    axs[i].imshow(vis_dict[f'{prefix}detail__output_images_detail'])
    # title = "detail image"
    title = ""
    if f'{stage}_valence_output' in values.keys():
        title += f'\nV_EN={ values[stage + "_valence_output"][0].detach().cpu().item():+3.2f}'
        title += f'\nA_EN={ values[stage + "_arousal_output"][0].detach().cpu().item():+3.2f}'
    if f'{stage}_expression_output' in values.keys():
        title += f'\nE_EN={AffectNetExpressions(np.argmax(values[stage + "_expression_output"][0].detach().cpu().numpy())).name}'
    axs[i].set_title("detail image")
    axs[i].text(256.0, 112.0, title, size=12, verticalalignment='center', horizontalalignment='left')
    i += 1


def test(deca, batch):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()
    # batch["image"] = batch["image"].cuda()
    # batch["image"] = batch["image"].view(1, 3, 224, 224)

    vals = deca.encode(batch, training=False)
    # vals = deca.decode(vals)
    vals, visdict, losses = decode(deca, vals, batch, training=False)
    return vals, visdict, losses


def plot_comparison(names, outputs, batch1, batch2, deca_name):
    fig = plt.figure()
    plt.title(deca_name)
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
        values, visdict, losses = outputs[i]
        # visualize(visdicts[i], names[i], axs[i*n_cols:(i+1)*n_cols], fig, i)
        # visualize(visdicts[i], names[i], values[i], losses[i], axs[i], fig, i)
        if i == 1:
            batch = batch2
        else:
            batch = batch1
        visualize(visdict, names[i], values, losses, batch, axs[i], fig, i)
    fig.set_facecolor('w')
    # mng = plt.get_current_fig_manager()
    # mng.frame.Maximize(True)
    plot_backend = plt.get_backend()
    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend[:2] == 'Qt': #'Qt4Agg':
        mng.window.showMaximized()
    else:
        raise RuntimeError(f"Invalid backend: {plot_backend}")

    fig.set_size_inches(32, 18)  # set figure's size manually to your full screen (32x18)
    # plt.savefig('filename.png', bbox_inches='tight')
    # plt.tight_layout()
    return fig


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

    values_img1, visdict1, losses1 = test(deca, img1_path_or_batch)
    values_img2, visdict2, losses2 = test(deca, img2_path_or_batch)

    # exchange codes
    vals21_s, vals12_s = exchange_and_decode(deca, values_img1, values_img2, ['shapecode'], img1_path_or_batch, img2_path_or_batch)
    vals21_d, vals12_d = exchange_and_decode(deca, values_img1, values_img2, ['detailcode'], img1_path_or_batch, img2_path_or_batch)
    vals21_e, vals12_e = exchange_and_decode(deca, values_img1, values_img2, ['expcode', 'jawpose'], img1_path_or_batch, img2_path_or_batch)

    vals21_sd, vals12_sd = exchange_and_decode(deca, values_img1, values_img2, ['shapecode', 'detailcode'], img1_path_or_batch, img2_path_or_batch)
    vals21_de, vals12_de = exchange_and_decode(deca, values_img1, values_img2, ['detailcode', 'expcode', 'jawpose'], img1_path_or_batch, img2_path_or_batch)
    vals21_se, vals12_se = exchange_and_decode(deca, values_img1, values_img2, ['shapecode', 'expcode', 'jawpose'], img1_path_or_batch, img2_path_or_batch)
    # visualize and analyze

    names1 = []
    visdicts1 = []
    values1 = []
    results1 = []
    names2 = []
    visdicts2 = []
    values2 = []
    results2 = []

    names1 += ["Input"]
    visdicts1 += [visdict1]
    values1 += [values_img1]
    results1 += [[values_img1, visdict1, losses1],]

    names1 += ["Target"]
    visdicts1 += [visdict2]
    values1 += [values_img2]
    results1 += [[values_img2, visdict2, losses2],]

    names2 += ["Input"]
    visdicts2 += [visdict2]
    values2 += [values_img2]
    results2 += [[values_img2, visdict2, losses2], ]

    names2 += ["Target"]
    visdicts2 += [visdict1]
    values2 += [values_img1]
    results2 += [[values_img1, visdict1, losses1], ]

    names1 += ["Shape exchange"]
    visdicts1 += [vals12_s[1]]
    values1 += [vals12_s[0]]
    results1 += [vals12_s,]

    names2 += ["Shape exchange"]
    visdicts2 += [vals21_s[1]]
    values2 += [vals21_s[0]]
    results2 += [vals21_s, ]

    names1 += ["Detail exchange"]
    visdicts1 += [vals12_d[1]]
    values1 += [vals12_d[0]]
    results1 += [vals12_d,]

    names2 += ["Detail exchange"]
    visdicts2 += [vals21_d[1]]
    values2 += [vals21_d[0]]
    results2 += [vals21_d, ]
    #
    names1 += [ "Expression exchange"]
    visdicts1 += [vals12_e[1]]
    values1 += [vals12_e[0]]
    results1 += [vals12_e, ]

    names2 += ["Expression exchange"]
    visdicts2 += [vals21_e[1]]
    values2 += [vals21_e[0]]
    results2 += [vals21_e, ]
    #
    names1 += ["Detail+expression exchange"]
    visdicts1 += [vals12_de[1]]
    values1 += [vals12_de[0]]
    results1 += [vals12_de, ]

    names2 += ["Detail+expression exchange"]
    visdicts2 += [vals21_de[1]]
    values2 += [vals21_de[0]]
    results2 += [vals21_de, ]

    names1 += ["Shape+Detail exchange"]
    visdicts1 += [vals12_sd[1]]
    values1 += [vals12_sd[0]]
    results1 += [vals12_sd, ]

    names2 += ["Shape+Detail exchange"]
    visdicts2 += [vals21_sd[1]]
    values2 += [vals21_sd[0]]
    results2 += [vals21_sd, ]

    #
    names1 += ["Shape+expression exchange"]
    visdicts1 += [vals12_se[1]]
    values1 += [vals12_se[0]]
    results1 += [vals12_se, ]

    names2 += ["Shape+expression exchange"]
    visdicts2 += [vals21_se[1]]
    values2 += [vals21_se[0]]
    results2 += [vals21_se, ]

    deca_name = deca.inout_params.name

    fig12 = plot_comparison(names1, results1, img1_path_or_batch, img2_path_or_batch, deca_name)
    fig21 = plot_comparison(names2, results2, img2_path_or_batch, img1_path_or_batch, deca_name)
    # plot_comparison(names1, visdicts1, values1)
    # plot_comparison(names2, visdicts2, values2)
    # mng = plt.get_current_fig_manager()
    # mng.frame.Maximize(True)
    # mng.window.showMaximized()

    return fig12, fig21


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



def load_affectnet():
    # scratch = "/home/rdanecek/Workspace/mount/scratch/"
    project = "/home/rdanecek/Workspace/mount/project/"
    work = "/is/cluster/work"


    dm = AffectNetDataModule(
        str(Path(project) / "EmotionalFacialAnimation/data/affectnet/"),
        # str(Path(scratch) / "rdanecek/data/affectnet"),
        str(Path(work) / "rdanecek/data/affectnet"),
        # processed_subfolder="processed_2021_Apr_02_03-13-33",
        processed_subfolder="processed_2021_Apr_05_15-22-18",
        mode="manual",
        scale=1.25)

    return dm


def main_affectnet(deca, conf, stage ):

    conf[stage].data.input_dir = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet"
    conf[stage].data.output_dir = "/is/cluster/work/rdanecek/data/affectnet"

    # dm, name = prepare_data(conf[stage])
    dm = load_affectnet()

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

    result_dir = Path(deca.inout_params.full_run_dir).parent / "tests" / "AffectNetExchange"
    result_dir.mkdir(exist_ok=True, parents=True)

    from tqdm import auto
    prev_batch = 0
    for bi, batch in auto.tqdm(enumerate(dl)):
        if bi == 0:
            prev_batch = batch
            continue
        fig12, fig21 = exchange_and_visualize(deca, batch, prev_batch)

        # plt.show(block=False)
        fig12.savefig( result_dir / f"{bi:04d}_{bi-1:04d}.png", bbox_inches='tight')#, dpi = 300)
        fig21.savefig( result_dir / f"{bi-1:04d}_{bi:04d}.png", bbox_inches='tight')#, dpi = 300)
        plt.close(fig12)
        plt.close(fig21)

        prev_batch = batch
        if bi > 100:
            break


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
