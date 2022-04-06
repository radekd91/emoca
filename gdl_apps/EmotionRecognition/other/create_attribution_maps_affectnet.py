"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""


import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# from captum.insights import AttributionVisualizer, Batch
# from captum.insights.attr_vis.features import ImageFeature

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import Saliency
from captum.attr import visualization as viz

from matplotlib.colors import LinearSegmentedColormap

from gdl.datasets.AffectNetDataModule import AffectNetExpressions, AffectNetDataModule, AffectNetTestModule
from gdl.models.EmotionRecognitionModuleBase import EmotionRecognitionBaseModule
from gdl.layers.losses.emotion_loss_loader import emo_network_from_path
from gdl.models.DECA import DECA, DecaModule
from gdl.models.IO import locate_checkpoint
from gdl_apps.EMOCA.load_data import hack_paths

from tqdm.auto import tqdm
from gdl_apps.EMOCA.test_and_finetune_deca import create_logger

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from wandb import Image


project_name = "InterpretableEmotion"


def get_classes():
    classes = [e.name for e in AffectNetExpressions]
    return classes


def get_pretrained_model_from_path(path):
    return get_pretrained_model(emo_network_from_path(path))
    

def get_pretrained_model(emo_net):
    # emo_net = emo_network_from_path(path)
    emo_net.cuda()

    class Net(nn.Module):

        def __init__(self, net: EmotionRecognitionBaseModule):
            super().__init__()
            self.net = net

        def forward(self, image):
            # print("whatever")
            # print(type(image))
            # print(type(full_batch))
            # print(full_batch.keys())
            # if image.shape[0] == 1:
            #     plt.figure()
            #     plt.imshow( np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)))
            #     plt.show()
            # else:
            #     for i in range(image.shape[0]):
            #         plt.figure()
            #         plt.imshow( np.transpose(image[i].cpu().detach().numpy(), (1, 2, 0)))
            #         plt.show()

            full_batch = {}
            full_batch["image"] = image
            image = image.cuda()
            img_size = self.net.config.model.image_size
            if not img_size:
                img_size = 224
            image = F.interpolate(image, (img_size,img_size), mode='bilinear')
            batch = full_batch.copy()
            batch["image"] = image
            output = self.net(batch)
            if "expression" in output.keys():
                out = output["expression"]
            elif "expr_classification" in output.keys():
                out = output["expr_classification"]
            else:
                raise ValueError("Missing expression prediction")
            # print(out.shape)
            return out


    net = Net(emo_net)
    net.cuda()
    net.eval()
    return net


def baseline_func(input):
    return input * 0


def get_dataseat():
    # dm = AffectNetDataModule(
    dm = AffectNetTestModule(
             # "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
             # "/ps/project_cifs/EmotionalFacialAnimation/data/affectnet/",
             "/ps/project/EmotionalFacialAnimation/data/affectnet/",
             # "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/",
             # "/home/rdanecek/Workspace/mount/work/rdanecek/data/affectnet/",
             "/is/cluster/work/rdanecek/data/affectnet/",
             # processed_subfolder="processed_2021_Aug_27_19-58-02",
             # processed_ext=".jpg",
             processed_subfolder="processed_2021_Apr_05_15-22-18",
             processed_ext=".png",
             mode="manual",
             scale=1.7,
             # image_size=512,
             image_size=256,
             # image_size=224,
             bb_center_shift_x=0,
             bb_center_shift_y=-0.3,
             ignore_invalid=True,
             # ring_type="gt_expression",
             # ring_type="gt_va",
             # ring_type="emonet_feature",
             # ring_size=4,
            augmentation=None,
            )
    dm.prepare_data()
    dm.setup()
    # dataset = torchvision.datasets.CIFAR10(
    #     root="data/test", train=False, download=True, transform=transforms.ToTensor()
    # )
    # dataset = dm.val_dataloader()
    # dataset = dm.validation_set
    dataset = dm.test_set
    return dataset
    # while True:
    #     batch = next(dataloader)
    #     for k, v in batch.items():
    #         if isinstance(v, torch.Tensor):
    #             batch[k] = v.cuda()
    #     images = batch["image"]
    #     print(images.shape)
    #     labels = batch["affectnetexp"]
    #     yield Batch(inputs=images, labels=labels, additional_args=batch)


def create_attribution_maps(root_folder, model, input_batch, sample_index, prefix_name="", logger=None):

    if len(prefix_name) > 0:
        prefix_name += "_"

    img = input_batch["image"]
    with torch.no_grad():
        output = model(input_batch["image"])
        output = F.softmax(output, dim=1)

    valence = input_batch["va"][0][0].item()
    arousal = input_batch["va"][0][1].item()
    expression_idx = AffectNetExpressions(input_batch["affectnetexp"][0].item()).value
    expression_label = AffectNetExpressions(input_batch["affectnetexp"][0].item()).name

    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    # predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    predicted_label = AffectNetExpressions(pred_label_idx.item()).name
    print('GT:', expression_label)
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')


    targets = [pred_label_idx, expression_idx, ]
    target_labels = [predicted_label, expression_label, ]
    target_prefix = ["pred", "gt", ]
    # targets = [pred_label_idx, expression_idx, ]

    # subtitles = ["Original Image", 'Heat Map', "Blended Heat Map"]
    # methods = ["original_image", 'heat_map', "blended_heat_map"]
    subtitles = ['Heat Map', "Blended Heat Map"]
    methods = ['heat_map', "blended_heat_map"]
    vis_dict = {}



    for ti, target in enumerate(targets):

        caption = f"GT: {expression_label}\n"
        caption += f"Predicted: {predicted_label}, {prediction_score.squeeze().item():.04f}\n"
        caption += f"Target: {target_labels[ti]}"

        prefix = prefix_name + target_prefix[ti] + "_"
        if target >= 9:
            continue

        saliency = Saliency(model)
        attributions_sal = saliency.attribute(img, target=target, abs=False)
        fig, ax = viz.visualize_image_attr_multiple(
            np.transpose(attributions_sal.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            methods=methods,
            show_colorbar=True,
            signs=["all"]*len(methods),
            outlier_perc=1,
            use_pyplot=False)

        fig.savefig(Path(root_folder) / f"{sample_index:05d}_{prefix}sal.png")
        for ai, a in enumerate(ax):
            a.set_title("Saliency " + subtitles[ai])
            extent = a.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            impath = Path(root_folder) / f'{sample_index:05d}_{prefix}sal-{ai:02d}.png'
            fig.savefig(impath, bbox_inches=extent.expanded(1.25, 1.45))
            vis_dict[impath.stem[6:]] = Image(str(impath), caption=caption)

        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(img, target=target, n_steps=50)
        fig, ax = viz.visualize_image_attr_multiple(
            np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            methods=methods,
            show_colorbar=True,
            signs=["all"]*len(methods),
            outlier_perc=1,
            use_pyplot=False)

        fig.savefig(Path(root_folder) / f"{sample_index:05d}_{prefix}ig.png")
        for ai, a in enumerate(ax):
            a.set_title("IntGrad " + subtitles[ai])
            extent = a.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            impath = Path(root_folder) / f'{sample_index:05d}_{prefix}ig-{ai:02d}.png'
            fig.savefig(impath, bbox_inches=extent.expanded(1.25, 1.45))
            vis_dict[impath.stem[6:]] = Image(str(impath), caption=caption)

        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions_ig_nt = noise_tunnel.attribute(img, nt_samples=50, nt_samples_batch_size=1, nt_type='smoothgrad_sq',
                                                    target=target,
                                                    n_steps=50,
                                                    # stdevs=0.1,
                                                    stdevs=0.05,
                                                    # stdevs=1./255.
                                                    )

        fig, ax = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                         np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                         methods=methods,
                                         show_colorbar=True,
                                         signs=["all"]*len(methods),
                                         outlier_perc=1,
                                     use_pyplot=False)
        fig.savefig(Path(root_folder) / f"{sample_index:05d}_{prefix}noise.png")
        for ai, a in enumerate(ax):
            a.set_title("Smooth IntGrad " + subtitles[ai])
            extent = a.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            impath = Path(root_folder) / f'{sample_index:05d}_{prefix}noise-{ai:02d}.png'
            fig.savefig(impath, bbox_inches=extent.expanded(1.25, 1.45))
            vis_dict[impath.stem[6:]] = Image(str(impath), caption=caption)

        # plt.show()

        torch.manual_seed(0)
        np.random.seed(0)

        gradient_shap = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([img * 0, img * 1])

        attributions_gs = gradient_shap.attribute(img,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=target)
        fig, ax = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                                    np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                                    methods=methods,
                                                    signs=["all"] * len(methods),
                                                    show_colorbar=True,
                                                    use_pyplot=False)
        fig.savefig(Path(root_folder) / f"{sample_index:05d}_{prefix}gradient_shap.png")
        for ai, a in enumerate(ax):
            a.set_title("Gradient SHAP " + subtitles[ai])
            extent = a.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            impath = Path(root_folder) /  f'{sample_index:05d}_{prefix}gradient_shap-{ai:02d}.png'
            fig.savefig(impath, bbox_inches=extent.expanded(1.25, 1.45))
            vis_dict[impath.stem[6:]] = Image(str(impath), caption=caption)

        occlusion = Occlusion(model)

        attributions_occ = occlusion.attribute(img,
                                               strides=(3, 8, 8),
                                               target=target,
                                               sliding_window_shapes=(3, 16, 16),
                                               baselines=0)

        fig, ax = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                                    np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                                    methods=methods,
                                                    signs=["all"] * len(methods),
                                                    show_colorbar=True,
                                                    outlier_perc=2,
                                                    use_pyplot = False
                                              )


        fig.savefig(Path(root_folder) / f"{sample_index:05d}_{prefix}occlusion.png")
        for ai, a in enumerate(ax):
            a.set_title("Occlusion 16x16 " + subtitles[ai])
            extent = a.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            impath = Path(root_folder) / f'{sample_index:05d}_{prefix}occlusion-{ai:02d}.png'
            fig.savefig(impath, bbox_inches=extent.expanded(1.25, 1.45))
            vis_dict[impath.stem[6:]] = Image(str(impath), caption=caption)

        attributions_occ2 = occlusion.attribute(img,
                                               strides=(3, 16, 16),
                                               target=target,
                                               sliding_window_shapes=(3, 32, 32),
                                               baselines=0)

        fig, ax = viz.visualize_image_attr_multiple(
            np.transpose(attributions_occ2.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                            methods=methods,
                                            signs=["all"] * len(methods),
                                            show_colorbar=True,
                                            outlier_perc=2,
                                            use_pyplot = False
                                            )
        fig.savefig(Path(root_folder) / f"{sample_index:05d}_{prefix}occlusion2.png")
        for ai, a in enumerate(ax):
            a.set_title("Occlusion 32x32 " + subtitles[ai])
            extent = a.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            impath = Path(root_folder) / f"{sample_index:05d}_{prefix}occlusion2-{ai:02d}.png"
            fig.savefig(impath, bbox_inches=extent.expanded(1.25, 1.45))
            vis_dict[impath.stem[6:]] = Image(str(impath), caption=caption)

    caption = f"GT: {expression_label}\n"
    caption += f"Predicted: {predicted_label}, {prediction_score.squeeze().item():.04f}\n"
    vis_dict["input"] = Image(np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                              caption=caption)

    if logger is not None:
        logger.log_metrics(vis_dict, sample_index)


    # sys.exit(0)

def load_deca(conf,
              stage,
              mode,
              relative_to_path=None,
              replace_root_path=None,
              ):
    print(f"Taking config of stage '{stage}'")
    print(conf.keys())
    if stage is not None:
        cfg = conf[stage]
    else:
        cfg = conf
    if relative_to_path is not None and replace_root_path is not None:
        cfg = hack_paths(cfg, replace_root_path=replace_root_path, relative_to_path=relative_to_path)
    cfg.model.resume_training = False

    checkpoint = locate_checkpoint(cfg, replace_root_path, relative_to_path, mode=mode)
    print(f"Loading checkpoint '{checkpoint}'")
    # if relative_to_path is not None and replace_root_path is not None:
    #     cfg = hack_paths(cfg, replace_root_path=replace_root_path, relative_to_path=relative_to_path)

    checkpoint_kwargs = {
        "model_params": cfg.model,
        "learning_params": cfg.learning,
        "inout_params": cfg.inout,
        "stage_name": "testing",
    }
    deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, **checkpoint_kwargs)
    return deca

def load_deca_model(deca_path,
              stage="detail",
              relative_to_path=None,
              replace_root_path=None,
              mode='best'
              ):
    run_path = Path(deca_path)
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    deca = load_deca(conf,
              stage,
              mode,
              relative_to_path,
              replace_root_path,
              )
    return deca, conf


import datetime
def create_attribution_maps_for_models(emonet_model_path, deca_model_path=None, deca_image_name=None, load_emonet_from_deca=False):
    # models = [get_pretrained_model(path) for path in model_paths]
    if deca_model_path is not None:
        stage="detail"
        deca_model, deca_conf = load_deca_model(deca_model_path, stage)
        deca_model.eval()
        assert deca_image_name is not None
        if load_emonet_from_deca:
            assert deca_conf[stage].model.emoloss_dual is True
            # state_dict = deca_model.emonet_loss.trainable_backbone.state_dict()
            model = get_pretrained_model(deca_model.emonet_loss.trainable_backbone)

    else:
        deca_model = None
        deca_conf = None

    if not load_emonet_from_deca:
        model = get_pretrained_model_from_path(emonet_model_path)
    # else:
    #     model = None
    assert model is not None

    dataset = get_dataseat()
    data_loader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    )
    bi = 0
    max_samples = 1000

    version = model.net.config.inout.time

    root = Path("/ps/scratch/rdanecek/InterpretableEmotion")

    if deca_model is None:
        folder_name = (version + "_" + model.net.config.inout.name)
        conf = OmegaConf.to_container(model.net.config)
    else:
        deca_name = deca_conf[stage].inout.name
        deca_name = deca_name.split("_")[0]
        suf = ""
        if load_emonet_from_deca:
            suf = "synth_"
        folder_name = (version + "_" + deca_name + "_" + deca_image_name + "_" + suf + model.net.config.inout.name)
        conf = DictConfig({})
        conf.emonet = model.net.config
        conf.deca = deca_conf
        conf = OmegaConf.to_container(conf)

    full_run_dir = root / folder_name
    full_run_dir.mkdir(parents=True, exist_ok=True)

    name = full_run_dir.name[len(version)+1:]

    logger = create_logger(
                         "WandbLogger",
                         # name=model.net.config.model.experiment_name,
                         name=name,
                         project_name=project_name,
                         config=conf,
                         version=version + str(hash(datetime.datetime.now())),
                         save_dir=str(full_run_dir))

    for bi in tqdm(range(min(len(dataset), max_samples))):
        batch = next(data_loader, None)

        if deca_model is not None:
            # deca_batch = batch.clone()
            deca_batch = {}
            deca_batch["image"] = F.interpolate(batch["image"],
                                       (deca_conf[stage].data.image_size, deca_conf[stage].data.image_size), mode='bilinear')
            deca_out = deca_model(deca_batch)
            batch["image"] = deca_out[deca_image_name]

        if batch is None:
            break
        # for model in models:
        create_attribution_maps(full_run_dir, model, batch, bi, "", logger)
        # bi += 1





if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "/ps/scratch/rdanecek/emoca/emodeca/2021_08_23_22-52-24_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early"
        # model_paths += ["/ps/scratch/rdanecek/emoca/emodeca/2021_08_24_00-17-40_EmoCnn_vgg19_shake_samp-balanced_expr_Aug_early"]
        # # model_paths += ["/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_23-50-06_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"]
        # model_paths += ["/ps/scratch/rdanecek/emoca/emodeca/2021_08_20_09-43-26_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000"]
        # model_paths += ['/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_13-06-58_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early']

    if len(sys.argv) > 2:
        deca_path = sys.argv[2]
        deca_image = sys.argv[3]
    else:
        # deca_path = None
        # deca_image = None
        # deca_path = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_09_07_21-13-42_ExpDECA_Affec_balanced_expr_para_Jaw_NoRing_EmoB_EmoCnn_vgg_du_F2nVAE_DeSegrend_Aug_DwC_early"
        deca_path = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_09_07_19-19-36_ExpDECA_Affec_balanced_expr_para_Jaw_NoRing_EmoB_EmoCnn_vgg_du_F2VAE_DeSegrend_Aug_DwC_early"
        # deca_image = "predicted_images"
        deca_image = "predicted_detailed_image"
        # deca_image = "predicted_translated_image"
        # deca_image = "predicted_detailed_translated_image"

    if len(sys.argv) > 4:
        trainable_deca_emonet = bool(int(sys.argv[4]))
    else:
        trainable_deca_emonet = True
        # trainable_deca_emonet = False

    assert deca_image in [None, "predicted_images", "predicted_detailed_image", "predicted_translated_image", "predicted_detailed_translated_image"]
    create_attribution_maps_for_models(model_path, deca_path, deca_image, trainable_deca_emonet)
    sys.exit(0)
