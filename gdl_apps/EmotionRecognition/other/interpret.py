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

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

from gdl.datasets.AffectNetDataModule import AffectNetExpressions, AffectNetDataModule
from gdl.models.EmotionRecognitionModuleBase import EmotionRecognitionBaseModule
from gdl.layers.losses.emotion_loss_loader import emo_network_from_path


def get_classes():
    classes = [e.name for e in AffectNetExpressions]
    return classes


def get_pretrained_model(path):
    emo_net = emo_network_from_path(path)
    emo_net.cuda()

    class Net(nn.Module):

        def __init__(self, net: EmotionRecognitionBaseModule):
            super().__init__()
            self.net = net

        def forward(self, image, full_batch):
            # print("whatever")
            # print(type(image))
            # print(type(full_batch))
            # print(full_batch.keys())
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
            print(out.shape)
            return out


    net = Net(emo_net)
    net.cuda()
    net.eval()
    return net


def baseline_func(input):
    return input * 0


def formatted_data_iter():
    dm = AffectNetDataModule(
             # "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
             # "/ps/project_cifs/EmotionalFacialAnimation/data/affectnet/",
             "/ps/project/EmotionalFacialAnimation/data/affectnet/",
             # "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/",
             # "/home/rdanecek/Workspace/mount/work/rdanecek/data/affectnet/",
             "/is/cluster/work/rdanecek/data/affectnet/",
             processed_subfolder="processed_2021_Aug_27_19-58-02",
             processed_ext=".jpg",
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
    dataset = dm.validation_set

    dataloader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    )
    while True:
        batch = next(dataloader)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        images = batch["image"]
        print(images.shape)
        labels = batch["affectnetexp"]
        yield Batch(inputs=images, labels=labels, additional_args=batch)


def setup_visualizer(path_to_net):
    if not isinstance(path_to_net, list):
        path_to_net = [path_to_net]

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    models = [get_pretrained_model(path) for path in path_to_net]
    visualizer = AttributionVisualizer(
        models=models,
        score_func=lambda o: torch.nn.functional.softmax(o, 1),
        classes=get_classes(),
        features=[
            ImageFeature(
                "Photo",
                baseline_transforms=[baseline_func],
                # input_transforms=[normalize],
                input_transforms=None,
            )
        ],
        dataset=formatted_data_iter(),
    )
    return visualizer


def visualize(visualizer):
    visualizer.render()
    from IPython.display import Image
    Image(filename='img/captum_insights.png')


if __name__ == "__main__":
    path_to_net = []
    path_to_net += ["/ps/scratch/rdanecek/emoca/emodeca/2021_08_23_22-52-24_EmoCnn_vgg13_shake_samp-balanced_expr_Aug_early"]
    path_to_net += ["/ps/scratch/rdanecek/emoca/emodeca/2021_08_24_00-17-40_EmoCnn_vgg19_shake_samp-balanced_expr_Aug_early"]
    path_to_net += ["/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_23-50-06_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early"]
    path_to_net += ["/ps/scratch/rdanecek/emoca/emodeca/2021_08_20_09-43-26_EmoNet_shake_samp-balanced_expr_Aug_early_d0.9000"]
    path_to_net += ['/ps/scratch/rdanecek/emoca/emodeca/2021_08_22_13-06-58_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early']
    visualizer = setup_visualizer(path_to_net)
    visualizer.serve(debug=True, bind_all=True)

