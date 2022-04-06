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


from venv import create

import torch
import torch.functional as F
from gdl_apps.EMOCA.interactive_deca_decoder import load_deca_and_data, test #, plot_results
from affectnet_validation import load_model
import copy
from gdl.layers.losses.EmoNetLoss import EmoNetLoss, EmoLossBase, EmoBackboneLoss, EmoNetModule
from gdl.layers.losses.emotion_loss_loader import emo_network_from_path
from gdl.models.DECA import DecaModule, DECA, DecaMode
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys, os
from tqdm import auto
import pytorch3d.transforms as trans
from pytorch_lightning.loggers import WandbLogger
import datetime
import wandb

from gdl.utils.image import numpy_image_to_torch
from gdl.datasets.IO import load_and_process_segmentation
from gdl.utils.FaceDetector import load_landmark
import pickle as pkl

def load_image_to_batch(image):
    batch = {}
    if isinstance(image, str) or isinstance(image, Path):
        image_path = image
        image = imread(image_path)[:, :, :3]

        if 'detections' in str(image_path):
            lmk_path = str(image_path).replace('detections', "landmarks")
            lmk_path = str(lmk_path).replace('.png', ".pkl")
            if Path(lmk_path).is_file():
                landmark_type, landmark = load_landmark(lmk_path)
                landmark = landmark[np.newaxis, ...]
                # normalize to <-1:1>
                landmark /= image.shape[0]
                landmark -= 0.5
                landmark *= 2
            else:
                landmark = None

            seg_path = str(image_path).replace('detections', "segmentations")
            seg_path = str(seg_path).replace('.png', ".pkl")
            if Path(seg_path).is_file():
                # seg_im = load_and_process_segmentation(seg_path)[...,0]
                seg_im = load_and_process_segmentation(seg_path)[0, ...]
                # seg_im = seg_im[None, ...]
            else:
                seg_im = None
        else:
            landmark = None
            seg_im = None

    if isinstance(image, np.ndarray):
        image = np.transpose(image, [2, 0, 1])[None, ...]
        if image.dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
            image = image.astype(np.float32)
            image /= 255.

    image = torch.from_numpy(image).cuda()
    batch["image"] = image
    if landmark is not None:
        batch["landmark"] = torch.from_numpy(landmark).cuda()
    if seg_im is not None:
        batch["mask"] = numpy_image_to_torch(seg_im)[None, ...].cuda()
    return batch


class TargetEmotionCriterion(torch.nn.Module):

    def __init__(self,
                 target_image,
                 use_feat_1 = False,
                 use_feat_2 = True,
                 use_valence = False,
                 use_arousal = False,
                 use_expression = False,
                 emonet_loss_instance = None,
                 ):
        super().__init__()
        if emonet_loss_instance is None:
            print("No emotion network passed into TargetEmotionCriterion. Defaulting to original EmoNet")
        self.emonet_loss = emonet_loss_instance or EmoNetLoss('cuda')
        self.emonet_loss.eval()

        # if isinstance(target_image, str) or isinstance(target_image, Path):
        #     target_image = imread(target_image)[:,:,:3]
        #
        # if isinstance(target_image, np.ndarray):
        #     target_image = np.transpose(target_image, [2,0,1])[None, ...]
        #     if target_image.dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
        #         target_image = target_image.astype(np.float32)
        #         target_image /= 255.

        # if target_image.shape[2] != self.emonet_loss.size[0] or target_image.shape[3] != self.emonet_loss.size[1]:
        #     target_image = F.interpolate(target_image)

        # target_image = torch.from_numpy(target_image).cuda()

        target_image = load_image_to_batch(target_image)["image"]

        # self.target_image = target_image
        self.register_buffer('target_image', target_image)
        self.target_emotion = self.emonet_loss(target_image)

        self.use_feat_1 = use_feat_1
        self.use_feat_2 = use_feat_2
        self.use_valence = use_valence
        self.use_arousal = use_arousal
        self.use_expression = use_expression

    def __call__(self, image):
        return self.forward(image)

    def forward(self, image):
        return self.compute(image)

    def compute(self, image):
        input_emotion = self.emonet_loss(image)

        emo_feat_loss_2 = self.emonet_loss.emo_feat_loss(input_emotion['emo_feat_2'], self.target_emotion['emo_feat_2'])
        valence_loss = self.emonet_loss.valence_loss(input_emotion['valence'], self.target_emotion['valence'])
        arousal_loss = self.emonet_loss.arousal_loss(input_emotion['arousal'], self.target_emotion['arousal'])
        if 'expression' in input_emotion.keys():
            expression_loss = self.emonet_loss.expression_loss(input_emotion['expression'], self.target_emotion['expression'])
        else:
            expression_loss = self.emonet_loss.expression_loss(input_emotion['expr_classification'], self.target_emotion['expr_classification'])


        total_loss = torch.zeros_like(emo_feat_loss_2)
        if self.use_feat_1:
            emo_feat_loss_1 = self.emonet_loss.emo_feat_loss(input_emotion['emo_feat'], self.target_emotion['emo_feat'])
            total_loss = total_loss + emo_feat_loss_1

        if self.use_feat_2:
            total_loss = total_loss + emo_feat_loss_2

        if self.use_valence:
            total_loss = total_loss + valence_loss

        if self.use_arousal:
            total_loss = total_loss + arousal_loss

        if self.use_expression:
            total_loss = total_loss + expression_loss

        return total_loss

    @property
    def name(self):
        return "EmotionLoss"

    def get_target_image(self):
        im = np.transpose(self.target_image.detach().cpu().numpy()[0, ...], [1,2,0])
        return im

    def save_target_image(self, path):
        im = self.get_target_image()
        print(im.shape)
        imsave(path, im)


class DecaTermCriterion(torch.nn.Module):

    def __init__(self,
                 keyword
                 ):
        super().__init__()
        self.keyword = keyword

    def forward(self, loss_dict):
        return loss_dict[self.keyword]


def convert_rotation(input, rot_type):
    if rot_type == "aa":
        pass  # already in axis angle
    elif rot_type == "quat":
        jaw_pose = trans.axis_angle_to_quaternion(input)
    elif rot_type == "euler":
        jaw_pose = trans.matrix_to_euler_angles(trans.axis_angle_to_matrix(input), "XYZ")
    else:
        raise ValueError(f"Invalid rotaion reference type: '{rot_type}'")
    return jaw_pose


class TargetJawCriterion(torch.nn.Module):

    def __init__(self,
                 reference_pose,
                 reference_type,
                 loss_type="l1"
                 ):
        super().__init__()
        self.reference_pose = torch.tensor(reference_pose).cuda()
        self.reference_type = reference_type
        self.loss_type = loss_type

    def __call__(self, posecode):
        return self.forward(posecode)

    def forward(self, posecode):
        return self.compute(posecode)

    @property
    def name(self):
        return f"JawReg_{self.reference_type}_{self.loss_type}"

    def compute(self, posecode):
        jaw_pose = posecode[:, 3:]
        #
        # if self.reference_type == "aa":
        #     pass # already in axis angle
        # elif self.reference_type == "quat":
        #     jaw_pose = trans.axis_angle_to_quaternion(jaw_pose)
        # elif self.reference_type == "euler":
        #     jaw_pose = trans.matrix_to_euler_angles(trans.axis_angle_to_matrix(jaw_pose), "XYZ")
        # else:
        #     raise ValueError(f"Invalid rotaion reference type: '{self.reference_type}'")

        jaw_pose = convert_rotation(jaw_pose, self.reference_type)

        if self.loss_type == "l1":
            reg = torch.abs(jaw_pose - self.reference_pose).sum()
        elif self.loss_type == "l2":
            reg = torch.square(jaw_pose - self.reference_pose).sum()
        else:
            raise NotImplementedError(f"Invalid loss: '{self.loss_type}'")
        return reg


    def save_target_image(self, path):
        pass


class CriterionWrapper(torch.nn.Module):

    def __init__(self, criterion, key):
        super().__init__()
        self.criterion = criterion
        self.key = key

    def forward(self, d):
        return self.criterion(d[self.key])

    @property
    def name(self):
        return self.criterion.name


import matplotlib.pyplot as plt


def plot_single_status(input_image, coarse_prediction, detail_prediction, coarse_image, detail_image, title,
                       show=False, save_path=None, save_images=False):
    fig, axs = plt.subplots(1,5, figsize=(5, 1.25))
    fig.suptitle(title, fontsize=8)

    images = [input_image, coarse_prediction, detail_prediction, coarse_image, detail_image]
    titles = ["Input", "Coarse shape", "Detail shape", "Coarse output", "Detail output"]

    for i in range(len(images)):
        if images[i] is not None:
            axs[i].imshow(images[i])
            axs[i].set_title(titles[i], fontsize=4)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)



    if save_path is not None:
        if save_images:
            for i in range(len(images)):
                if images[i] is not None:
                    path = save_path.parent / save_path.stem
                    name = titles[i]
                    name.replace(" ","_")
                    path.mkdir(exist_ok=True, parents=True)
                    imsave(path/ (name + ".png"), images[i])

        plt.savefig(save_path, dpi=300)

    if show:
        fig.show()
    plt.close()



def plot_results(vis_dict, title, detail=True, show=False, save_path=None):
    pass
    #1) Input image
    #2) Input coarse prediction
    #3) Input detail prediction
    #4) Input coarse image
    #5) Input detail image

    #6) Target image
    #7) Target coarse prediction
    #8) Target detail prediction
    #9) Target coarse image
    #10) Target detail image

    #11) Starting iteration coarse
    #12) Starting iteration detail
    #13) Starting iteration coarse image
    #14) Starting iteration detail image

    # 15) Current iteration coarse
    # 17) Current iteration detail
    # 18) Current iteration coarse image
    # 19) Current iteration detail image

    # 11) Best iteration coarse
    # 12) Best iteration detail
    # 13) Best iteration coarse image
    # 14) Best iteration detail image

    # 11) Loss total
    # 12) Loss components
    # 13) Best iteration coarse image
    # 14) Best iteration detail image


def extract_images(vis_dict, detail=True):
    prefix = None

    for key in list(vis_dict.keys()):
        if 'detail__inputs' in key:
            start_idx = key.rfind('detail__inputs')
            prefix = key[:start_idx]
            # print(f"Prefix was found to be: '{prefix}'")
            break

    if prefix is None:
        print(vis_dict.keys())
        raise RuntimeError(f"Uknown disctionary content. Available keys {vis_dict.keys()}")

    if detail:
        input = vis_dict[f'{prefix}detail__inputs']
        geom_coarse = vis_dict[f'{prefix}detail__geometry_coarse']
        geom_detail = vis_dict[f'{prefix}detail__geometry_detail']
        image_coarse = vis_dict[f'{prefix}detail__output_images_coarse']
        image_detail = vis_dict[f'{prefix}detail__output_images_detail']
    else:
        input = vis_dict[f'{prefix}coarsel__inputs']
        geom_coarse = vis_dict[f'{prefix}coarse__geometry_coarse']
        geom_detail = None
        image_coarse = vis_dict[f'{prefix}coarse__output_images_coarse']
        image_detail = None

    return input, geom_coarse, geom_detail, image_coarse, image_detail


def save_visualization(deca, values, title, save_path=None, show=False, with_input=True, detail=True, save_images=False):
    uv_detail_normals = None
    if 'uv_detail_normals' in values.keys():
        uv_detail_normals = values['uv_detail_normals']
    visualizations, grid_image = deca._visualization_checkpoint(values['verts'],
                                                                values['trans_verts'],
                                                                values['ops'],
                                                                uv_detail_normals,
                                                                values,
                                                                0,
                                                                "",
                                                                "",
                                                                save=False)
    vis_dict = deca._create_visualizations_to_log("", visualizations, values, 0, indices=0)
    # input, geom_coarse, geom_detail, image_coarse, image_detail = extract_images(vis_dict, detail=detail)
    # if not with_input:
    #     input = None
    # plot_single_status(input, geom_coarse, geom_detail, image_coarse, image_detail, title,
    #                    show=show, save_path=save_path)
    save_visualization_step(vis_dict, title, save_path, show, with_input, detail, save_images=save_images)
    return vis_dict


def save_visualization_step(visdict, title, save_path=None, show=False, with_input=True, detail=True, save_images=False):
    input, geom_coarse, geom_detail, image_coarse, image_detail = extract_images(visdict, detail=detail)
    if not with_input:
        input = None
    plot_single_status(input, geom_coarse, geom_detail, image_coarse, image_detail, title,
                       show=show, save_path=save_path, save_images=save_images)
    return visdict


def copy_values(values):
    copied_values = {}
    for key, val in values.items():
        try:
            copied_values[key] = val.detach().clone()
        except Exception as e:
            try:
                copied_values[key] = copy.deepcopy(val)
            except Exception as e2:
                # print(val)
                # try:
                copied_values[key] = {k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                                    for k, v in values[key].items()}
    return copied_values


def plot_optimization(losses_by_step, logs, iter=None, save_path=None, prefix=None):
    suffix = ''
    if iter is not None:
        suffix = f"_{iter:04d}"
    prefix = prefix or ''
    if len(prefix) > 0:
        prefix += "_"

    plt.figure()
    plt.title("Optimization")
    plt.plot(losses_by_step)
    if save_path is not None:
        plt.savefig(save_path / f"{prefix}optimization{suffix}.png", dpi=200)
    plt.close()

    # print(logs)
    for term in logs.keys():
        plt.figure()
        plt.title(f"{term}")
        plt.plot(logs[term])
        if save_path is not None:
            plt.savefig(save_path / f"{prefix}term_{term}{suffix}.png", dpi=200)
    plt.close()

def create_video(term_names, save_path, clean_up=True):
    save_path = Path(save_path)
    optimization_files = sorted(list(save_path.glob("optimization*.png")))

    iteration_files = sorted(list(save_path.glob("step_*.png")))

    optimization_term_files = {}
    for key in term_names:
        optimization_term_files[key] = sorted(list(save_path.glob(f"term_{key}*.png")))

    # optimization_images = []
    # print("Loading images")
    # for f in auto.tqdm(optimization_files):
    #     optimization_images += [imread(f)]
    #
    # term_images = {}
    # for key in term_names:
    #     term_images[key] = []
    #     for f in auto.tqdm(optimization_term_files[key]):
    #         term_images[key] += [imread(f)]
    #
    # iteration_images = []
    # for f in auto.tqdm(iteration_files):
    #     iteration_images += [imread(f)]

    start_image = imread(save_path / "source.png")
    best_image = imread(save_path / "best.png")
    target_image = imread(save_path / "target.png")

    num_iters = len(optimization_files)

    import cv2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(filename=str(vis_folder / "video.mp4"), apiPreference=cv2.CAP_FFMPEG,
    #                                fourcc=fourcc, fps=dm.video_metas[sequence_id]['fps'], frameSize=(vis_im.shape[1], vis_im.shape[0]))
    fps = 10
    video_writer = None

    target_width = start_image.shape[1]
    target_height = None

    print("Creating video")
    for i in auto.tqdm(range(num_iters)):
        # iteration_im = iteration_images[i]
        iteration_im = imread(iteration_files[i])
        width = iteration_im.shape[1]
        scale = target_width / width
        iteration_im = rescale(iteration_im, (scale, scale, 1))
        iteration_im = (iteration_im * 255).astype(np.uint8)

        frame = np.concatenate([start_image, target_image, iteration_im, best_image], axis=0)
        if i == 0:
            target_height = frame.shape[0]


        # opt_im = optimization_images[i]
        opt_im = imread(optimization_files[i])
        # resize opt_im to frame width
        # scale = target_width / opt_im.shape[1]
        scale = target_height / opt_im.shape[0]
        opt_im = rescale(opt_im, (scale, scale, 1))
        opt_im = (opt_im * 255).astype(np.uint8)

        term_im = []
        for key in term_names:
            # term_im += [term_images[key][i]]
            term_im += [imread(optimization_term_files[key][i])]
        term_im = np.concatenate(term_im, axis=1)

        # scale = target_width / term_im.shape[1]
        scale = opt_im.shape[1] / term_im.shape[1]
        # scale = target_height / term_im.shape[0]
        term_im = rescale(term_im, (scale, scale, 1))
        term_im = (term_im * 255).astype(np.uint8)
        # resize term_im to frame width
        frame_optim = np.concatenate([opt_im, term_im], axis=0)

        scale = target_height / frame_optim.shape[0]
        frame_optim = rescale(frame_optim, (scale, scale, 1))
        frame_optim = (frame_optim * 255).astype(np.uint8)

        frame_final = np.concatenate([frame, frame_optim], axis=1)


        if i == 0:
            video_writer = cv2.VideoWriter(str(save_path / "video.mp4"), cv2.CAP_FFMPEG,
                                           fourcc, fps,
                                           (frame_final.shape[1], frame_final.shape[0]), True)

        frame_bgr = frame_final[:,:, [2, 1, 0]]
        video_writer.write(frame_bgr)


    # the last frame for two more seconds
    for i in range(fps*2):
        video_writer.write(frame_bgr)

    video_writer.release()
    if clean_up:
        print("Cleaning up")
        optimization_files = sorted(list(save_path.glob("optimization_*.png")))
        for f in auto.tqdm(optimization_files):
            os.remove(str(f))

        iteration_files = sorted(list(save_path.glob("step_*.png")))
        for f in auto.tqdm(iteration_files):
            os.remove(str(f))

        optimization_term_files = {}
        for key in term_names:
            optimization_term_files[key] = sorted(list(save_path.glob(f"term_{key}_*.png")))
            for f in auto.tqdm(optimization_term_files[key]):
                os.remove(str(f))

def optimize(deca,
             values,
             # optimize_detail=True,
             optimize_detail=False,
             optimize_identity=False,
             optimize_expression=True,
             # optimize_expression=False,
             optimize_neck_pose=False,
             # optimize_neck_pose=True,
             optimize_jaw_pose=False,
             # optimize_jaw_pose=True,
             optimize_texture=False,
             optimize_cam=False,
             optimize_light=False,
             lr = 0.01,
             losses_to_use=None,
             loss_weights=None,
             visualize_progress=False,
             visualize_result=False,
             max_iters = 1000,
             patience = 20,
             verbose=True,
             save_path=None,
             optimizer_type= "SGD",
             logger= None,
             jaw_lr = None,
             ):
    if sum([optimize_detail,
             optimize_identity,
             optimize_expression,
             optimize_neck_pose,
             optimize_jaw_pose,
             optimize_texture,
             optimize_cam,
             optimize_light]) == 0:
        raise ValueError("Nothing to optimizze for. Everything is set to false")

    # log_prefix = f"{save_path.parents[1]}/{save_path.parents[0] / save_path.name}"
    log_prefix = f"{save_path.parents[0].name}/{save_path.name}"

    if save_path is not None:
        print(f"Results will be saved to '{save_path}'")

    losses_to_use = losses_to_use or "loss"
    if not isinstance(losses_to_use, list):
        losses_to_use = [losses_to_use,]
    loss_weights = loss_weights or [1.] * len(losses_to_use)

    # deca.deca.config.train_coarse = True
    # deca.deca.config.mode = DecaMode.DETAIL
    # # deca.deca.config.mode = DecaMode.COARSE

    parameters = []
    if optimize_detail:
        values['detailcode'] = torch.autograd.Variable(values['detailcode'].detach().clone(), requires_grad=True)
        parameters += [{'params': values['detailcode'], "lr": lr}]

    if optimize_identity:
        values['shapecode'] = torch.autograd.Variable(values['shapecode'].detach().clone(), requires_grad=True)
        parameters += [{'params': values['shapecode'], "lr": lr}]

    if optimize_expression:
        values['expcode'] = torch.autograd.Variable(values['expcode'].detach().clone(), requires_grad=True)
        parameters += [{'params': values['expcode'], "lr": lr}]

    if optimize_neck_pose:
        neck_pose = torch.autograd.Variable(values['posecode'][:, :3].detach().clone(), requires_grad=True)
        parameters += [{'params': neck_pose, "lr": lr}]
    else:
        neck_pose = values['posecode'][:, :3].detach().clone()

    if optimize_jaw_pose:
        jaw_pose = torch.autograd.Variable(values['posecode'][:, 3:].detach().clone(), requires_grad=True)
        parameters += [{'params': jaw_pose, "lr": jaw_lr if jaw_lr is not None else lr}]
    else:
        jaw_pose = values['posecode'][:, 3:].detach().clone()

    # if optimize_neck_pose or optimize_jaw_pose:
        # values['posecode'] = torch.autograd.Variable(values['posecode'].detach().clone(), requires_grad=True)
        # parameters += [ values['posecode']]
    values['posecode'] = torch.cat([neck_pose, jaw_pose], dim=1)

    if optimize_texture:
        values['texcode'] = torch.autograd.Variable(values['texcode'].detach().clone(), requires_grad=True)
        parameters += [{'params': values['texcode'], "lr": lr}]

    if optimize_cam:
        values['cam'] = torch.autograd.Variable(values['cam'].detach().clone(), requires_grad=True)
        parameters += [{'params': values['cam'], "lr": lr}]

    if optimize_light:
        values['lightcode'] = torch.autograd.Variable(values['lightcode'].detach().clone(), requires_grad=True)
        parameters += [{'params': values['lightcode'], "lr": lr}]

    if len(parameters) == 0:
        raise RuntimeError("No parameters are being optimized")

    def criterion(vals, losses_and_metrics, logs=None):
        total_loss = 0
        for i, loss in enumerate(losses_to_use):
            if isinstance(loss, str):
                term = losses_and_metrics[loss]
                if logs is not None:
                    if loss not in logs.keys():
                        logs[loss] = []
                    logs[loss] += [term.item()]
                total_loss = total_loss + (term*loss_weights[i])
            else:
                term = loss(vals)
                if logs is not None:
                    if loss.name not in logs.keys():
                        logs[loss.name] = []
                    logs[loss.name] += [term.item()]
                total_loss = total_loss + (term * loss_weights[i])

        return total_loss

    if save_path is not None:
        print(f"Creating savepath: {save_path}")
        save_path.mkdir(exist_ok=True, parents=True)
    else:
        print("No visuals will be saved")

    logs = {}
    losses_by_step = []


    losses_and_metrics = deca.compute_loss(values, {}, training=True)
    loss = criterion(values, losses_and_metrics, logs)
    current_loss = loss.item()
    # losses_by_step += [current_loss]

    save_visualization(deca, values,
                       save_path=save_path / f"step_{0:02d}.png" if save_path is not None else None,
                       title=f"Start, loss={current_loss:.10f}",
                       show=visualize_result,
                       detail=deca.mode == DecaMode.DETAIL,
                       with_input=False,
                       save_images=True)

    # optimizer = torch.optim.Adam(parameters, lr=0.01)
    # # optimizer = torch.optim.SGD(parameters, lr=0.001)
    # # optimizer = torch.optim.LBFGS(parameters, lr=lr)

    optimizer_class = getattr(torch.optim, optimizer_type)

    if optimizer_class is torch.optim.LBFGS:
        # LBFGS does not support separate learning rate
        params = []
        for p in parameters:
            # del p["lr"]
            params += p["params"]
        parameters = params

    optimizer = optimizer_class(parameters, lr=lr)

    best_loss = 99999999999999.
    best_values = copy_values(values)
    value_history = []
    history_values_to_keep = ['shapecode', 'expcode', 'posecode', 'texcode', 'detailcode', 'lightcode', 'detailemocode', 'cam']
    eps = 1e-6

    history_entry = {}
    for v in history_values_to_keep:
        history_entry[v] = copy.deepcopy(values[v].detach().cpu())
    value_history += [history_entry]

    stopping_condition_hit = False

    since_last_improvement = 0
    for i in range(1,max_iters+1):

        def closure():
            optimizer.zero_grad()
            values_ = deca.decode(values, training=False)
            # losses_and_metrics = deca.compute_loss(values_, training=False)
            losses_and_metrics = deca.compute_loss(values_, {}, training=True)
            loss = criterion(values_, losses_and_metrics)
            loss.backward(retain_graph=True)
            return loss

        optimizer.zero_grad()

        if optimize_neck_pose or optimize_jaw_pose:
            values['posecode'] = torch.cat([neck_pose, jaw_pose], dim=1)

        values_ = deca.decode(values, training=False)
        # losses_and_metrics = deca.compute_loss(values, training=False)
        losses_and_metrics = deca.compute_loss(values_, {}, training=True)

        loss = criterion(values, losses_and_metrics, logs)
        loss.backward(retain_graph=True)
        # closure()
        current_loss = loss.item()

        optimizer.step(closure=closure)
        # make sure pose vector is updated


        if visualize_progress or save_path is not None:
            save_visualization(deca, values,
                               save_path=save_path / f"step_{i:04d}.png" if save_path is not None else None,
                               title=f"Iter {i:04d}, loss={current_loss:.10f}",
                               show=visualize_progress,
                               detail=deca.mode == DecaMode.DETAIL,
                               with_input=False)
            if i == 1:
                save_visualization(deca, values,
                                   save_path=save_path / f"init.png" if save_path is not None else None,
                                   title=f"Iter {i:04d}, loss={current_loss:.10f}",
                                   show=visualize_progress,
                                   detail=deca.mode == DecaMode.DETAIL,
                                   with_input=False,
                                   save_images=True)
                if logger is not None:
                    logger.log_metrics({f"{log_prefix}/init":
                                   wandb.Image(str(save_path / f"step_{i:04d}.png"))})
            if logger is not None:
                logger.log_metrics({f"{log_prefix}/optim_step":
                               wandb.Image(str(save_path / f"step_{i:04d}.png"))})
            plot_optimization(losses_by_step, logs, iter=i, save_path=save_path)


        losses_by_step += [current_loss]

        history_entry = {}
        for v in history_values_to_keep:
            history_entry[v] = copy.deepcopy(values[v].detach().cpu())
        value_history += [history_entry]

        if verbose:
            print(f"Iter {i:04d}, loss={current_loss:.10f}")

        if loss < eps:
            stopping_condition_hit = True
            break

        since_last_improvement += 1
        if best_loss > current_loss:
            since_last_improvement = 0
            best_iter = i
            best_loss = current_loss
            best_values = copy_values(values)

        if since_last_improvement > patience:
            stopping_condition_hit = True
            print(f"Breaking at iteration {i} becase there was no improvement for the last {since_last_improvement} steps.")
            break


        if logger is not None:
            log_dict = {f"{log_prefix}/opt_step" : i}
            for key, vals in logs.items():
                log_dict[f"{log_prefix}/{key}"] = vals[-1]
            log_dict[f"{log_prefix}/optim"] = losses_by_step[-1]

            logger.log_metrics(log_dict)

    if not stopping_condition_hit:
        print(f"[WARNING] Optimization terminated after max number of iterations, not becaused it reached the desired tolerance")


    values = deca.decode(best_values, training=False)
    losses_and_metrics = deca.compute_loss(values, {}, training=False)

    save_visualization(deca, values,
                       save_path=save_path / f"best.png" if save_path is not None else None,
                       title=f"Best iter {best_iter:04d}, loss={best_loss:.10f}",
                       show=visualize_progress,
                       detail=deca.mode == DecaMode.DETAIL,
                       with_input=False,
                       save_images=True
    )

    with open(save_path / "best_values.pkl", "wb") as f:
        pkl.dump(best_values, f)


    plot_optimization(losses_by_step, logs, save_path=save_path, prefix="final")

    if save_path or visualize_result:
        for i, loss in enumerate(losses_to_use):
            if isinstance(loss, CriterionWrapper):
                if save_path:
                    loss.criterion.save_target_image(save_path / f"target_{i:02d}.png")
                if visualize_result:
                    plt.figure()
                    plt.imshow(loss.criterion.get_target_image())
                    plt.title(f"Target {i:02d} {loss.name}")

    # plt.figure()
    # plt.plot(losses_by_step)
    # plt.legend("Optimization")
    # if save_path is not None:
    #     plt.savefig(save_path / "optimization.png", dpi=100)
    if visualize_result:
        plt.show()

    if save_path:
        logger.log_metrics({f"{log_prefix}/best":
                       wandb.Image(str(save_path / f"best.png"))})

        # create_video(list(logs.keys()), save_path)

    return values


def parameter_configurations():
    kwargs = {
        "optimize_detail": False,
        "optimize_identity": False,
        "optimize_expression": False,
        "optimize_neck_pose": False,
        "optimize_jaw_pose": False,
        "optimize_texture": False,
        "optimize_cam": False,
        "optimize_light": False,
    }
    kwarg_dict = {}

    # detail
    kw = copy.deepcopy(kwargs)
    kw["optimize_detail"] = True
    kwarg_dict["detail"] = kw

    # identity
    kw = copy.deepcopy(kwargs)
    kw["optimize_identity"] = True
    kwarg_dict["identity"] = kw

    # expression
    kw = copy.deepcopy(kwargs)
    kw["optimize_expression"] = True
    kwarg_dict["expression"] = kw

    # pose
    kw = copy.deepcopy(kwargs)
    kw["optimize_neck_pose"] = True
    kw["optimize_jaw_pose"] = True
    kwarg_dict["pose"] = kw

    # expression, detail
    kw = copy.deepcopy(kwargs)
    kw["optimize_detail"] = True
    kw["optimize_expression"] = True
    kwarg_dict["expression_detail"] = kw

    # expression, detail, pose
    kw = copy.deepcopy(kwargs)
    kw["optimize_detail"] = True
    kw["optimize_expression"] = True
    kw["optimize_neck_pose"] = True
    kw["optimize_jaw_pose"] = True
    kwarg_dict["expression_detail_pose"] = kw

    return kwarg_dict





def loss_function_config(target_image, keyword, emonet=None):

    losses = []
    if keyword == "emotion":
        losses += [CriterionWrapper(TargetEmotionCriterion(target_image, emonet_loss_instance=emonet),
                                    "predicted_detailed_image")]
        return losses

    if keyword == "emotion_f1_reg_exp":
        losses += [CriterionWrapper(TargetEmotionCriterion(
            target_image, use_feat_1=True, use_feat_2=False, emonet_loss_instance=emonet),
            "predicted_detailed_image")]
        losses += ["loss_expression_reg"]
        return losses

    if keyword == "emotion_f2_reg_exp":
        losses += [CriterionWrapper(TargetEmotionCriterion(
            target_image, use_feat_2=True, emonet_loss_instance=emonet),
            "predicted_detailed_image")]
        losses += ["loss_expression_reg"]
        return losses

    if keyword == "emotion_f12_reg_exp":
        losses += [CriterionWrapper(TargetEmotionCriterion(
            target_image, use_feat_1=True, use_feat_2=True, emonet_loss_instance=emonet),
            "predicted_detailed_image")]
        losses += ["loss_expression_reg"]
        return losses

    if keyword == "emotion_va_reg_exp":
        losses += [CriterionWrapper(TargetEmotionCriterion(
            target_image, use_valence=True, use_arousal=True, use_feat_2=False, emonet_loss_instance=emonet),
            "predicted_detailed_image")]
        losses += ["loss_expression_reg"]
        return losses

    if keyword == "emotion_e_reg_exp":
        losses += [CriterionWrapper(TargetEmotionCriterion(
            target_image, use_expression=True, use_feat_2=False, emonet_loss_instance=emonet),
            "predicted_detailed_image")]
        losses += ["loss_expression_reg"]
        return losses

    if keyword == "emotion_vae_reg_exp":
        losses = []
        losses += [CriterionWrapper(TargetEmotionCriterion(
            target_image, use_valence=True, use_arousal=True, use_expression=True, use_feat_2=False, emonet_loss_instance=emonet),
            "predicted_detailed_image")]
        losses += ["loss_expression_reg"]
        return losses

    if keyword == "emotion_f12vae_reg_exp":
        losses = []
        losses += [CriterionWrapper(TargetEmotionCriterion(
            target_image, use_feat_1=True, use_feat_2=True, use_valence=True, use_arousal=True, use_expression=True, emonet_loss_instance=emonet),
            "predicted_detailed_image")]
        losses += ["loss_expression_reg"]
        return losses

    # if keyword == "emotion_reg_exp_detail":
    #     losses = []
    #     losses += [CriterionWrapper(TargetEmotionCriterion(target_image, emonet_loss_instance=emonet), "predicted_detailed_image")]
    #     losses += ["loss_expression_reg"]
    #     losses += ["loss_z_reg"]
    #     return losses

    raise ValueError(f"Unknown keyword '{keyword}'")


def create_emotion_loss(emonet_loss, deca=None):
    feature_metric = None
    if isinstance(emonet_loss, str):
        emonet = emo_network_from_path(emonet_loss)
    elif isinstance(emonet_loss, dict):
        if emonet_loss["path"] == "Synth":
            emonet = deca.emonet_loss.trainable_backbone
        elif emonet_loss["path"] is None or emonet_loss["path"] == "None":
            return None
        else:
            emonet = emo_network_from_path(emonet_loss["path"])
        if "feature_metric" in emonet_loss.keys():
            feature_metric = emonet_loss["feature_metric"]
    else:
        raise ValueError(f"Invalid input argument type '{type(emonet_loss)}'")

    if isinstance(emonet, EmoNetModule):
        emonet = EmoNetLoss( torch.device('cuda:0'), emonet.emonet, emo_feat_loss=feature_metric)
    else:
        emonet = EmoBackboneLoss( torch.device('cuda:0'), emonet, emo_feat_loss=feature_metric)
    emonet.cuda()
    return emonet


def loss_function_config_v2(target_image, loss_dict, emonet=None, deca=None,
                            output_image_key="predicted_detailed_image",
                            values_input=None,
                            values_target=None,
                            ):
    #
    # if emonet is not None and isinstance(emonet, str):
    #     emonet = emo_network_from_path(emonet)
    #     if isinstance(emonet, EmoNetModule):
    #         emonet = EmoNetLoss( torch.device('cuda:0'), emonet.emonet)
    #     else:
    #         emonet = EmoBackboneLoss( torch.device('cuda:0'), emonet)
    #         emonet.cuda()
    emonet = create_emotion_loss(emonet, deca=deca)

    # output_image_key = "predicted_detailed_image"

    losses = []
    loss_weights = []
    for keyword, weight in loss_dict.items():
        if isinstance(weight, dict):
            loss_weights += [weight["weight"]]
        else:
            loss_weights += [weight]
        if keyword == "emotion_f2":
            losses += [CriterionWrapper(TargetEmotionCriterion(target_image, emonet_loss_instance=emonet), output_image_key)]
        elif keyword == "emotion_f1":
            losses += [CriterionWrapper(TargetEmotionCriterion(
                target_image, use_feat_1=True, use_feat_2=False, emonet_loss_instance=emonet),
                output_image_key)]

        elif keyword == "emotion_f12":
            losses += [CriterionWrapper(TargetEmotionCriterion(
                target_image, use_feat_1=True, use_feat_2=True, emonet_loss_instance=emonet),
                output_image_key)]

        elif keyword == "emotion_va":
            losses += [CriterionWrapper(TargetEmotionCriterion(
                target_image, use_valence=True, use_arousal=True, use_feat_2=False, emonet_loss_instance=emonet),
                output_image_key)]

        elif keyword == "emotion_e":
            losses += [CriterionWrapper(TargetEmotionCriterion(
                target_image, use_expression=True, use_feat_2=False, emonet_loss_instance=emonet),
                output_image_key)]

        elif keyword == "emotion_vae":
            losses += [CriterionWrapper(TargetEmotionCriterion(
                target_image, use_valence=True, use_arousal=True, use_expression=True, use_feat_2=False, emonet_loss_instance=emonet),
                output_image_key)]
        elif keyword == "emotion_f12vae":
            losses += [CriterionWrapper(TargetEmotionCriterion(
                target_image, use_feat_1=True, use_feat_2=True, use_valence=True, use_arousal=True, use_expression=True, emonet_loss_instance=emonet),
                output_image_key)]
        elif keyword == "jaw_reg":
            if weight["reference_pose"] == "from_target":
                ref_pose = values_target["posecode"][:, 3:]
                ref_pose = convert_rotation(ref_pose, weight["reference_type"])
            elif weight["reference_type"] == "from_input":
                ref_pose = values_input["posecode"][:, 3:]
                ref_pose = convert_rotation(ref_pose, weight["reference_type"])
            else:
                ref_pose = weight["reference_pose"] # already in the right representation

            losses += [CriterionWrapper(TargetJawCriterion(
                ref_pose,
                weight["reference_type"],
                weight["loss_type"],
            ), "posecode")]
        #
        # elif keyword == "loss_photometric_texture":
        #     losses += [CriterionWrapper(DecaTermCriterion(
        #         "loss_photometric_texture"
        #     ), "posecode")]

        else:
            losses += [keyword]

    return losses, loss_weights


def loss_function_configs(target_image, emonet=None):
    loss_configs = {}

    target_emo = TargetEmotionCriterion(target_image, ememonet_loss_instance=emonet)
    emonet = target_emo.emonet_loss

    loss_configs["emotion"] = loss_function_config(target_image, "emotion", emonet)
    loss_configs["emotion_f1_reg_exp"] = loss_function_config(target_image, "emotion_f1_reg_exp", emonet)
    loss_configs["emotion_f2_reg_exp"] = loss_function_config(target_image, "emotion_f2_reg_exp", emonet)
    loss_configs["emotion_f12_reg_exp"] = loss_function_config(target_image, "emotion_f12_reg_exp", emonet)
    loss_configs["emotion_va_reg_exp"] = loss_function_config(target_image, "emotion_va_reg_exp", emonet)
    loss_configs["emotion_e_reg_exp"] = loss_function_config(target_image, "emotion_e_reg_exp", emonet)
    loss_configs["emotion_vae_reg_exp"] = loss_function_config(target_image, "emotion_vae_reg_exp", emonet)
    loss_configs["emotion_f12vae_reg_exp"] = loss_function_config(target_image, "emotion_f12vae_reg_exp", emonet)
    loss_configs["emotion_reg_exp_detail"] = loss_function_config(target_image, "emotion_reg_exp_detail", emonet)
    return loss_configs


def loss_function_configs_v2(target_image, emonet=None):
    loss_configs = {}

    target_emo = TargetEmotionCriterion(target_image, ememonet_loss_instance=emonet)
    emonet = target_emo.emonet_loss

    loss_configs["emotion"] = loss_function_config(target_image, "emotion", emonet)
    # loss_configs["emotion_f1_reg_exp"] = loss_function_config(target_image, "emotion_f1_reg_exp", emonet)
    loss_configs["emotion_f2_reg_exp"] = loss_function_config(target_image, "emotion_f2_reg_exp", emonet)
    loss_configs["emotion_f12_reg_exp"] = loss_function_config(target_image, "emotion_f12_reg_exp", emonet)
    loss_configs["emotion_va_reg_exp"] = loss_function_config(target_image, "emotion_va_reg_exp", emonet)
    loss_configs["emotion_e_reg_exp"] = loss_function_config(target_image, "emotion_e_reg_exp", emonet)
    loss_configs["emotion_vae_reg_exp"] = loss_function_config(target_image, "emotion_vae_reg_exp", emonet)
    loss_configs["emotion_f12vae_reg_exp"] = loss_function_config(target_image, "emotion_f12vae_reg_exp", emonet)
    loss_configs["emotion_reg_exp_detail"] = loss_function_config(target_image, "emotion_reg_exp_detail", emonet)
    return loss_configs



def replace_codes(values_input, values_target,
                  optimize_detail=False,
                  optimize_identity=False,
                  optimize_expression=False,
                  optimize_neck_pose=False,
                  optimize_jaw_pose=False,
                  optimize_texture=False,
                  optimize_cam=False,
                  optimize_light=False,
                  replace_detail=True, replace_exp=True,
                  replace_jaw=True,
                  replace_pose=True,
                  replace_cam=True,
                  **kwargs):
    # if optimize_detail:

    # always copy identity
    # if optimize_identity:
    values_target['shapecode'] = values_input['shapecode'].detach().clone()

    # if optimize_expression:
    if not replace_exp:
        values_target['expcode'] = values_input['expcode'].detach().clone()

    if not replace_detail:
        values_target['detailcode'] = values_input['detailcode'].detach().clone()

    if not replace_pose and not replace_jaw:
        values_target['posecode'] = values_input['posecode'].detach().clone()
    elif not replace_pose:
        posecode = values_input['posecode'].detach().clone()
        global_pose = posecode[:,:3]
        values_target['posecode'] = torch.cat([global_pose, values_target['posecode'][:, 3:]], dim=1)
    elif not replace_jaw:
        posecode = values_input['posecode'].detach().clone()
        jaw_pose = posecode[:, 3:]
        values_target['posecode'] = torch.cat([values_target['posecode'][:, :3], jaw_pose], dim=1)

    # if optimize_neck_pose and optimize_jaw_pose:
    #     values_target['posecode'] = values_input['posecode'].detach().clone()
    # elif optimize_neck_pose:
    #     posecode = values_input['posecode'].detach().clone()
    #     neck_pose = posecode[:,:3]
    #     values_target['posecode'] = torch.cat([neck_pose, posecode[:, 3:]], dim=1)
    # elif optimize_jaw_pose:
    #     posecode = values_input['posecode'].detach().clone()
    #     jaw_pose = posecode[:, 3:]
    #     values_target['posecode'] = torch.cat([posecode[:, :3], jaw_pose], dim=1)

    # if optimize_texture:
    values_target['texcode'] = values_input['texcode'].detach().clone()

    # if optimize_cam:
    if not replace_cam:
        values_target['cam'] = values_input['cam'].detach().clone()

    # if optimize_light: # always copy light
    values_target['lightcode'] = values_input['lightcode'].detach().clone()
    return values_target


def single_optimization(path_to_models, relative_to_path, replace_root_path, out_folder, model_name,
                        model_folder, stage, image_index, target_image,
                        num_repeats=1,
                        losses_to_use: dict = None , **kwargs):
    if losses_to_use is None:
        raise RuntimeError("No losses specified. ")
    losses_to_use_dict = losses_to_use
    losses_to_use, loss_weights = loss_function_config_v2(target_image, losses_to_use)

    # if loss_weights is None:
    #     loss_weights = [1.] * len(losses_to_use)

    # if losses_to_use is not None:
    #     target_image = "~/Workspace/mount/scratch/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/VA_Set/" \
    #                    "detections/Train_Set/82-25-854x480/002400_000.png"
    #     losses_to_use = []
    #     losses_to_use += [CriterionWrapper(TargetEmotionCriterion(target_image), "predicted_detailed_image")]
    #     losses_to_use += ["loss_shape_reg"]
        # losses_to_use += ["loss_expression_reg"]


    # path_to_models = in_folder[0]

    # stage = 'detail'
    # relative_to_path = in_folder[1]
    # relative_to_path = '/ps/scratch/'
    # replace_root_path = in_folder[2]
    # replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'

    # image_index = model_cfg[2]
    # stage = model_cfg[1]
    # model_folder = model_cfg[0]

    deca, dm = load_deca_and_data(path_to_models, model_folder, stage, relative_to_path, replace_root_path)
    deca.deca.config.train_coarse = True
    deca.deca.config.mode = DecaMode.DETAIL
    deca.eval()
    # deca.deca.config.mode = DecaMode.COARSE
    # image_index = 390 * 4 + 1
    # image_index = 90*4

    initializations = {}
    values_input, visdict_input = test(deca, dm, image_index=image_index)
    # print(values_input.keys())
    # if not initialize_from_target:
    initializations["from_input"] = [values_input, visdict_input]
        # plot_results(visdict, "title")
        # max_iters = 1000 # part of kwargs now

    # if initialize_from_target:
    batch = {}
    batch["image"] = load_image_to_batch(target_image)
    values_target, visdict_target = test(deca, dm, batch=batch)
    values_target = replace_codes(values_input, values_target, **kwargs)
    values_target["images"] = values_input["images"] # we don't want the target image but the input image (for inpainting by mask)
    initializations["from_target"] = [values_target, visdict_target]
    # TODO: possibly add an option for randomized

    # Path(out_folder / model_name).mkdir(exist_ok=True, parents=True)
    Path(out_folder ).mkdir(exist_ok=True, parents=True)
    with open("out_folder.txt", "w") as f:
        f.write(str(out_folder))
    with open(Path(out_folder) / "submission_folder.txt", "w") as f:
        f.write(os.getcwd())

    for key, vals in initializations.items():
        values, visdict = vals[0], vals[1]
        # num_repeats = 5
        # num_repeats = 1
        for i in range(num_repeats):
            # save_path = Path(out_folder) / model_name / key / f"{i:02d}"
            save_path = Path(out_folder) / key / f"{i:02d}"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving to '{save_path}'")
            save_visualization_step(visdict_input, "Source", save_path / "source.png")
            save_visualization_step(visdict_target, "Target", save_path / "target.png")
            optimize(deca,
                     copy_values(values),  #important to copy
                     losses_to_use=losses_to_use,
                     loss_weights=loss_weights,
                     # max_iters=max_iters,
                     verbose=True,
                     # visualize_progress=False,
                     save_path=save_path,
                     **kwargs)


def single_optimization_v2(path_to_models, relative_to_path, replace_root_path, out_folder, model_name,
                           model_folder, stage, start_image, target_image,
                           num_repeats=1,
                           losses_to_use: dict = None,
                           output_image_key="predicted_detailed_image",
                           **kwargs):

    if losses_to_use is None:
        raise RuntimeError("No losses specified. ")
    losses_to_use_dict = losses_to_use
    deca, _ = load_model(path_to_models, model_folder, stage)
    deca.deca.config.train_coarse = True
    deca.deca.config.mode = DecaMode.DETAIL
    deca.deca.config.background_from_input = False

    if model_name == "Original_DECA":
        # remember, this is the hacky way to load old Yao's model
        deca.deca.config.resume_training = True
        try:
            deca.deca._load_old_checkpoint()
        except FileNotFoundError as e:
            deca.deca.config.pretrained_modelpath = '/home/rdanecek/Workspace/Repos/DECA/data/deca_model.tar'
            deca.deca._load_old_checkpoint()
        run_name = "Original_DECA"

    deca.eval()
    deca.cuda()

    emonet_path = kwargs.pop("emonet") if "emonet" in kwargs.keys() else None
    if emonet_path == "None":
        emonet_path = None

    # start_batch = {}
    # start_batch["image"] = load_image_to_batch(start_image)
    start_batch = load_image_to_batch(start_image)

    values_input, visdict_input = test(deca, batch=start_batch)

    initializations = {}
    # initializations["all_from_input"] = [values_input, visdict_input]

    # if initialize_from_target:
    # batch = {}
    # batch["image"] = load_image_to_batch(target_image)
    batch = load_image_to_batch(target_image)



    values_target_, visdict_target_ = test(deca, batch=batch)

    # values_target = replace_codes(values_input, copy.deepcopy(values_target_),
    #                               replace_detail=True,
    #                               replace_exp=True,
    #                               replace_jaw=False,
    #                               replace_pose=True,
    #                               replace_cam =True,
    #                               **kwargs)
    # # values_target["images"] = values_input["images"] # we don't want the target image but the input image (for inpainting by mask)
    # initializations["all_from_target_but_jaw"] = [values_target, copy.deepcopy(visdict_target_)]

    # values_target = replace_codes(values_input, copy.deepcopy(values_target_),
    #                               replace_detail=True,
    #                               replace_exp=False,
    #                               replace_jaw=False,
    #                               replace_pose=True,
    #                               replace_cam =True,
    #                               **kwargs)
    # # values_target["images"] = values_input["images"] # we don't want the target image but the input image (for inpainting by mask)
    # initializations["all_from_target_but_jaw_and_exp"] = [values_target, copy.deepcopy(visdict_target_)]
    #
    # values_target = replace_codes(values_input, copy.deepcopy(values_target_),
    #                               replace_detail=True,
    #                               replace_jaw=True,
    #                               replace_exp=True,
    #                               replace_pose=True,
    #                               replace_cam = True,
    #                               **kwargs)

    ## values_target["images"] = values_input["images"] # we don't want the target image but the input image (for inpainting by mask)
    # initializations["all_from_target"] = [values_target, copy.deepcopy(visdict_target_)]
    # initializations["all_from_input"] = [values_input, visdict_input]

    losses_to_use, loss_weights = loss_function_config_v2(target_image, losses_to_use, emonet=emonet_path, deca=deca,
                                                          output_image_key=output_image_key,
                                                          values_input=values_input,
                                                          values_target=values_target_,
                                                          )



    # values_target = replace_codes(values_input, copy.deepcopy(values_target_),
    #                               replace_detail=True,
    #                               replace_exp=True,
    #                               replace_jaw=True,
    #                               replace_pose=False,
    #                               replace_cam =False,
    #                               **kwargs)
    # # values_target["images"] = values_input["images"] # we don't want the target image but the input image (for inpainting by mask)
    # initializations["all_from_target_but_pose"] = [values_target, copy.deepcopy(visdict_target_)]
    #
    # values_target = replace_codes(values_input, copy.deepcopy(values_target_),
    #                               replace_detail=False,
    #                               replace_exp=True,
    #                               replace_jaw=True,
    #                               replace_pose=True,
    #                               replace_cam = True,
    #                               **kwargs)
    # # values_target["images"] = values_input["images"] # we don't want the target image but the input image (for inpainting by mask)
    # initializations["all_from_target_but_detail"] = [values_target, copy.deepcopy(visdict_target_)]
    #
    # values_target = replace_codes(values_input, copy.deepcopy(values_target_),
    #                               replace_detail=False,
    #                               replace_exp=True,
    #                               replace_jaw=True,
    #                               replace_pose=False,
    #                               replace_cam =False,
    #                               **kwargs)
    # # values_target["images"] = values_input["images"] # we don't want the target image but the input image (for inpainting by mask)
    # initializations["all_from_target_but_pose_and_detail"] = [values_target, copy.deepcopy(visdict_target_)]

    # values_target = replace_codes(values_input, copy.deepcopy(values_target_), replace_exp=False, replace_pose=False, **kwargs)
    # values_target["images"] = values_input[
    #     "images"]  # we don't want the target image but the input image (for inpainting by mask)

    # values_target_detail = replace_codes(values_input, copy.deepcopy(values_target_),
    #                                      replace_detail=True,
    #                                      replace_exp=False,
    #                                      replace_jaw=False,
    #                                      replace_pose=False,
    #                                      replace_cam = False,
    #                                      **kwargs)
    # # values_target_detail["images"] = values_input["images"]  # we don't want the target image but the input image (for inpainting by mask)
    # initializations["detail_from_target"] = [values_target_detail, copy.deepcopy(visdict_target_)]




    values_target_pose = replace_codes(values_input, copy.deepcopy(values_target_),
                                       replace_detail=False,
                                       replace_exp=True,
                                       replace_jaw=True,
                                       replace_pose=True,
                                       replace_cam = True,
                                       **kwargs)
    # values_target_pose["images"] = values_input[
    #     "images"]  # we don't want the target image but the input image (for inpainting by mask)
    initializations["exp_pose_jaw_from_target"] = [values_target_pose, copy.deepcopy(visdict_target_)]


    values_target_pose = replace_codes(values_input, copy.deepcopy(values_target_),
                                       replace_detail=False,
                                       replace_exp=True,
                                       replace_jaw=False,
                                       replace_pose=True,
                                       replace_cam = True,
                                       **kwargs)
    # values_target_pose["images"] = values_input["images"]  # we don't want the target image but the input image (for inpainting by mask)
    initializations["exp_pose_from_target"] = [values_target_pose, copy.deepcopy(visdict_target_)]

    values_target_pose = replace_codes(values_input, copy.deepcopy(values_target_),
                                       replace_detail=False,
                                       replace_exp=False,
                                       replace_jaw=True,
                                       replace_pose=True,
                                       replace_cam = True,
                                       **kwargs)

    # values_target_pose["images"] = values_input["images"]  # we don't want the target image but the input image (for inpainting by mask)
    initializations["pose_jaw_from_target"] = [values_target_pose, copy.deepcopy(visdict_target_)]

    values_target_pose = replace_codes(values_input, copy.deepcopy(values_target_),
                                       replace_detail=False,
                                       replace_exp=False,
                                       replace_jaw=False,
                                       replace_pose=True,
                                       replace_cam = True,
                                       **kwargs)


    # values_target_pose["images"] = values_input["images"]  # we don't want the target image but the input image (for inpainting by mask)
    initializations["pose_from_target"] = [values_target_pose, copy.deepcopy(visdict_target_)]

    values_target_pose = replace_codes(values_input, copy.deepcopy(values_target_),
                                       replace_detail=False,
                                       replace_exp=True,
                                       replace_jaw=True,
                                       replace_pose=False,
                                       replace_cam = False,
                                       **kwargs)

    # values_target_pose["images"] = values_input["images"]  # we don't want the target image but the input image (for inpainting by mask)
    initializations["exp_jaw_from_target"] = [values_target_pose, copy.deepcopy(visdict_target_)]

    values_target_pose = replace_codes(values_input, copy.deepcopy(values_target_),
                                       replace_detail=False,
                                       replace_exp=True,
                                       replace_jaw=False,
                                       replace_pose=False,
                                       replace_cam = False,
                                       **kwargs)

    # values_target_pose["images"] = values_input["images"]  # we don't want the target image but the input image (for inpainting by mask)
    initializations["exp_from_target"] = [values_target_pose, copy.deepcopy(visdict_target_)]

    values_target_pose = replace_codes(values_input, copy.deepcopy(values_target_),
                                       replace_detail=False,
                                       replace_exp=False,
                                       replace_jaw=True,
                                       replace_pose=False,
                                       replace_cam = False,
                                       **kwargs)

    # values_target_pose["images"] = values_input["images"]  # we don't want the target image but the input image (for inpainting by mask)
    initializations["jaw_from_target"] = [values_target_pose, copy.deepcopy(visdict_target_)]

    values_target_pose = replace_codes(values_input, copy.deepcopy(values_target_),
                                       replace_detail=False,
                                       replace_exp=False,
                                       replace_jaw=False,
                                       replace_pose=False,
                                       replace_cam = False,
                                       **kwargs)
    # values_target_pose["images"] = values_input["images"]  # we don't want the target image but the input image (for inpainting by mask)
    initializations["nothing_from_target"] = [values_target_pose, copy.deepcopy(visdict_target_)]




    # values_target_detail_exp = replace_codes(values_input, copy.deepcopy(values_target_),
    #                                          replace_detail=True,
    #                                          replace_exp=True,
    #                                          replace_jaw=False,
    #                                          replace_pose=False,
    #                                          replace_cam = False,
    #                                          **kwargs)
    # # values_target_detail_exp["images"] = values_input[
    # #     "images"]  # we don't want the target image but the input image (for inpainting by mask)
    # initializations["detail_exp_from_target"] = [values_target_detail_exp, copy.deepcopy(visdict_target_)]

    # values_target_exp_jaw_detail = replace_codes(values_input, copy.deepcopy(values_target_),
    #                                              replace_detail=True,
    #                                              replace_exp=True,
    #                                              replace_jaw=True,
    #                                              replace_pose=False,
    #                                              replace_cam = False,
    #                                   **kwargs)
    # # values_target_exp_jaw_detail["images"] = values_input[
    # #     "images"]  # we don't want the target image but the input image (for inpainting by mask)
    # initializations["detail_exp_jaw_from_target"] = [values_target_exp_jaw_detail, copy.deepcopy(visdict_target_)]

    # values_target_exp_jaw_pose = replace_codes(values_input, copy.deepcopy(values_target_),
    #                                        replace_detail=False,
    #                                        replace_exp=True,
    #                                        replace_jaw=True,
    #                                        replace_pose=True,
    #                                        replace_cam = True,
    #                                        **kwargs)
    # # values_target_exp_jaw_pose["images"] = values_input[
    # #     "images"]  # we don't want the target image but the input image (for inpainting by mask)
    # initializations["detail_exp_jaw_pose_from_target"] = [values_target_exp_jaw_pose, copy.deepcopy(visdict_target_)]


    # values_target_pose = replace_codes(values_input, copy.deepcopy(values_target_),
    #                                    replace_detail=True,
    #                                    replace_exp=False,
    #                                    replace_jaw=False,
    #                                    replace_pose=True,
    #                                    replace_cam = True,
    #                                    **kwargs)
    # # values_target_pose["images"] = values_input[
    # #     "images"]  # we don't want the target image but the input image (for inpainting by mask)
    # initializations["detail_pose_from_target"] = [values_target_pose, copy.deepcopy(visdict_target_)]


    # TODO: possibly add an option for randomized

    # Path(out_folder / model_name).mkdir(exist_ok=True, parents=True)
    Path(out_folder ).mkdir(exist_ok=True, parents=True)
    with open("out_folder.txt", "w") as f:
        f.write(str(out_folder))
    with open(Path(out_folder) / "submission_folder.txt", "w") as f:
        f.write(os.getcwd())


    cfg = kwargs.copy()
    cfg["deca_model"] = model_name
    cfg["deca_model_path"] = str(Path(path_to_models) / model_name)
    cfg["out_folder"] = str(out_folder)
    if isinstance( emonet_path, str):
        cfg["emonet"] = str(Path(emonet_path).name)
        cfg["emonet_name"] = str(Path(emonet_path).name)
        # cfg["emonet"] = str(emonet_path)
    elif isinstance( emonet_path, dict):
        cfg["emonet"] = str(Path(emonet_path["path"]).name)
        cfg["emonet_name"] = str(Path(emonet_path["path"]).name)
        cfg["emonet_cfg"] = emonet_path
    else:
        cfg["emonet_name"] = "Original Emonet"
        cfg["emonet"] = "Original Emonet"

    cfg["source_image"] = str(start_image)
    cfg["target_image"] = str(target_image)
    cfg["deca_stage"] = str(stage)
    cfg["output_image_key"] = str(output_image_key)
    cfg["losses"] = losses_to_use_dict
    tags = cfg["tags"] if "tags" in cfg.keys() else None
    if "tags" in cfg.keys():
        del kwargs["tags"]

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    logger = WandbLogger(name=Path(out_folder).name,
                     project="EmotionOptimization",
                     config=cfg,
                     version=time + "_" + str(hash(time)) + "_" + Path(out_folder).name,
                     save_dir=out_folder,
                     tags=tags,
                         )

    for key, vals in initializations.items():
        values, visdict = vals[0], vals[1]
        # num_repeats = 5
        # num_repeats = 1
        for i in range(num_repeats):
            # save_path = Path(out_folder) / model_name / key / f"{i:02d}"
            save_path = Path(out_folder) / key / f"{i:02d}"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving to '{save_path}'")
            save_visualization_step(visdict_input, "Source", save_path / "source.png", save_images=True)
            if logger is not None:
                logger.log_metrics({f"{key}/{i:02d}/source": wandb.Image(str(Path(save_path / "source.png")))})

            save_visualization_step(visdict_target_, "Target", save_path / "target.png", save_images=True)

            if logger is not None:
                logger.log_metrics({f"{key}/{i:02d}/target": wandb.Image(str(Path(save_path / "target.png")))})

            optimize(deca,
                     copy_values(values),  #important to copy
                     losses_to_use=losses_to_use,
                     loss_weights=loss_weights,
                     # max_iters=max_iters,
                     verbose=True,
                     # visualize_progress=False,
                     save_path=save_path,
                     logger=logger,
                     **kwargs)

            # logger.log_metrics({f"{key}/{i:02d}/optimization_vid": wandb.Video(str(Path(save_path / "video.mp4")))})


def optimization_with_different_losses(path_to_models,
                                       relative_to_path,
                                       replace_root_path,
                                       out_folder,
                                       model_name,
                                       model_folder,
                                       stage,
                                       starting_image_index,
                                       target_image,
                                        num_repeats,
                                       optim_kwargs):
    if 'emonet' in optim_kwargs.keys():
        emonet = optim_kwargs.pop("emonet")
    else:
        emonet = None

    # loss_configs = loss_function_configs(target_image, emonet=emonet)
    loss_configs = loss_function_configs_v2(target_image, emonet=emonet)
    for key, loss_list in loss_configs.items():
        single_optimization(path_to_models,
                            relative_to_path,
                            replace_root_path,
                            out_folder / key,
                            model_name,
                            model_folder,
                            stage,
                            starting_image_index,
                            target_image,
                            loss_list,
                            num_repeats,
                            **optim_kwargs)


def optimization_with_specified_loss(path_to_models,
                                       relative_to_path,
                                       replace_root_path,
                                       out_folder,
                                       model_name,
                                       model_folder,
                                       stage,
                                       starting_image_index,
                                       target_image,
                                       # loss_keyword,
                                        num_repeats,
                                       optim_kwargs):
    # loss_list = loss_function_config(target_image, loss_keyword)

    single_optimization(path_to_models,
                        relative_to_path,
                        replace_root_path,
                        # out_folder / loss_keyword,
                        out_folder ,
                        model_name,
                        model_folder,
                        stage,
                        starting_image_index,
                        target_image,
                        # loss_list,
                        num_repeats,
                        **optim_kwargs)


def optimization_with_specified_loss_v2(path_to_models,
                                     relative_to_path,
                                     replace_root_path,
                                     out_folder,
                                     model_name,
                                     model_folder,
                                     stage,
                                     starting_image,
                                     target_image,
                                     # loss_keyword,
                                     num_repeats,
                                     optim_kwargs):
    # loss_list = loss_function_config(target_image, loss_keyword)

    single_optimization_v2(path_to_models,
                        relative_to_path,
                        replace_root_path,
                        # out_folder / loss_keyword,
                        out_folder,
                        model_name,
                        model_folder,
                        stage,
                        starting_image,
                        target_image,
                        # loss_list,
                        num_repeats,
                        **optim_kwargs)



def optimization_for_different_targets(path_to_models, relative_to_path, replace_root_path, out_folder, model_name,
                                       model_folder, stage, starting_image_index, target_images, optim_kwargs):
    for target_image in target_images:
        optimization_with_different_losses(
            path_to_models, relative_to_path, replace_root_path,
            Path(out_folder) / Path(model_name) / target_image.parent.stem / target_image.stem,
            model_name, model_folder, stage, starting_image_index, target_image, optim_kwargs)


def main():
    deca_models = {}
    # deca_models["Octavia"] = \
    #     ['2021_03_08_22-30-55_VA_Set_videos_Train_Set_119-30-848x480.mp4CoPhotoCoLMK_IDW-0.15_Aug_early', 'detail', 390 * 4 + 1]
    # deca_models["Rachel"] = \
    #     ['2021_03_05_16-31-05_VA_Set_videos_Train_Set_82-25-854x480.mp4CoPhotoCoLMK_IDW-0.15_Aug_early', 'detail', 90*4]
    # deca_models["General1"] = \
    #     ['2021_03_08_22-30-55_VA_Set_videos_Train_Set_119-30-848x480.mp4CoPhotoCoLMK_IDW-0.15_Aug_early', None, 390*4]
    # deca_models["General2"] = \
    #     ['2021_03_05_16-31-05_VA_Set_videos_Train_Set_82-25-854x480.mp4CoPhotoCoLMK_IDW-0.15_Aug_early', None, 90*4]


    # ExpDECA with VGG net for emotions, trainable
    # deca_models["2021_09_07_19-19-36_ExpDECA_Affec_balanced_expr_para_Jaw_NoRing_EmoB_EmoCnn_vgg_du_F2VAE_DeSegrend_Aug_DwC_early"] \
    #     = "/is/cluster/work/rdanecek/emoca/finetune_deca/" \
    #       "2021_09_07_19-19-36_ExpDECA_Affec_balanced_expr_para_Jaw_NoRing_EmoB_EmoCnn_vgg_du_F2VAE_DeSegrend_Aug_DwC_early/"
    deca_models[""] \
        = "/is/cluster/work/rdanecek/emoca/finetune_deca/" \
          "2021_09_07_19-19-36_ExpDECA_Affec_balanced_expr_para_Jaw_NoRing_EmoB_EmoCnn_vgg_du_F2VAE_DeSegrend_Aug_DwC_early/"
    # deca_models["ExpDECA_emonet"] = ""

    # target_image_path = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10")
    target_image_path = Path("/is/cluster/work/rdanecek/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10")

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

    for t in target_images:
        if not t.exists():
            print(t)
        # print(t.exists())


    # cluster
    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # relative_to_path = None
    # replace_root_path = None
    # out_folder = '/ps/scratch/rdanecek/emoca/finetune_deca/optimize_emotion'

    # not on cluster
    # path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    # relative_to_path = '/ps/scratch/'
    # replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
    relative_to_path = None
    replace_root_path = None
    # out_folder = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/optimize_emotion'
    out_folder = '/is/cluster/work/rdanecek/emoca/optimize_emotion_v2'

    for name, cfg in deca_models.items():
        model_folder = cfg[0]
        stage = cfg[1]
        starting_image_index = cfg[2]
        optimization_for_different_targets(path_to_models, relative_to_path, replace_root_path, out_folder, name,
                                           model_folder, stage, starting_image_index, target_images)


from omegaconf import OmegaConf, DictConfig


def probe_video():
    save_path = Path('/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/optimize_emotion/2021_03_19_16-11-58_emotion_vae_1.00_loss_expression_reg_10.00_SGD_0.01/General1/119-30-848x480/000640_000/from_input/00')
    terms = ['EmotionLoss', 'loss_expression_reg']
    create_video(terms, save_path)


if __name__ == "__main__":
    # probe_video()

    print("Running:" + __file__)
    for i, arg in enumerate(sys.argv):
        print(f"arg[{i}] = {arg}")

    if len(sys.argv) > 1:
        path_to_models = Path(sys.argv[1])
        relative_to_path = None if sys.argv[2] == "None" else sys.argv[2]
        replace_root_path = None if sys.argv[3] == "None" else sys.argv[3]
        out_folder = Path(sys.argv[4])
        model_name = sys.argv[5]
        model_folder = sys.argv[6]
        stage = None if sys.argv[7] == "None" else sys.argv[7]
        # starting_image_index = int(sys.argv[8])
        starting_image_path = sys.argv[8]
        target_image = sys.argv[9]
        # loss_keyword = sys.argv[10]
        num_repeats = int(sys.argv[10])
        optim_kwargs = OmegaConf.to_container(OmegaConf.load(sys.argv[11]))

    else:
        path_to_models = Path(sys.argv[1])
        relative_to_path = None if sys.argv[2] == "None" else sys.argv[2]
        replace_root_path = None if sys.argv[3] == "None" else sys.argv[3]

    optimization_with_specified_loss_v2(path_to_models,
                                        relative_to_path,
                                        replace_root_path,
                                        out_folder,
                                        model_name,
                                        model_folder,
                                        stage,
                                        starting_image_path,
                                        target_image,
                                        # loss_keyword,
                                        num_repeats,
                                        optim_kwargs)
