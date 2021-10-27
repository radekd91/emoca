## THIS FILE HAS BEEN COPIED FROM TDECA REPOSITORY AND NEEDS TO BE CLEANED UP

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
# import torch.autograd as autograd
from functools import reduce
import torchvision.models as models
# import cv2
import torchfile
# from torch.autograd import Variable

# from . import util


# def l2_error(verts1, verts2):
#     return np.sqrt(((verts1 - verts2)**2).sum(1)).mean()

def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2) ** 2).sum(2)).mean(1).mean()


### VAE
def kl_loss(texcode):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mu, logvar = texcode[:, :128], texcode[:, 128:]
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return KLD


### ------------------------------------- Losses/Regularizations for shading
# white shading
# uv_mask_tf = tf.expand_dims(tf.expand_dims(tf.constant( self.uv_mask, dtype = tf.float32 ), 0), -1)
# mean_shade = tf.reduce_mean( tf.multiply(shade_300W, uv_mask_tf) , axis=[0,1,2]) * 16384 / 10379
# G_loss_white_shading = 10*norm_loss(mean_shade,  0.99*tf.ones([1, 3], dtype=tf.float32), loss_type = "l2")
def shading_white_loss(shading):
    '''
    regularize lighting: assume lights close to white
    '''
    # rgb_diff = (shading[:,0] - shading[:,1])**2 + (shading[:,0] - shading[:,2])**2 + (shading[:,1] - shading[:,2])**2
    # rgb_diff = (shading[:,0].mean([1,2]) - shading[:,1].mean([1,2]))**2 + (shading[:,0].mean([1,2]) - shading[:,2].mean([1,2]))**2 + (shading[:,1].mean([1,2]) - shading[:,2].mean([1,2]))**2
    # rgb_diff = (shading.mean([2, 3]) - torch.ones((shading.shape[0], 3)).float().cuda())**2
    rgb_diff = (shading.mean([0, 2, 3]) - 0.99) ** 2
    return rgb_diff.mean()


def shading_smooth_loss(shading):
    '''
    assume: shading should be smooth
    ref: Lifting AutoEncoders: Unsupervised Learning of a Fully-Disentangled 3D Morphable Model using Deep Non-Rigid Structure from Motion
    '''
    dx = shading[:, :, 1:-1, 1:] - shading[:, :, 1:-1, :-1]
    dy = shading[:, :, 1:, 1:-1] - shading[:, :, :-1, 1:-1]
    gradient_image = (dx ** 2).mean() + (dy ** 2).mean()
    return gradient_image.mean()


### ------------------------------------- Losses/Regularizations for albedo
# texture_300W_labels_chromaticity = (texture_300W_labels + 1.0)/2.0
# texture_300W_labels_chromaticity = tf.divide(texture_300W_labels_chromaticity, tf.reduce_sum(texture_300W_labels_chromaticity, axis=[-1], keep_dims=True) + 1e-6)


# w_u = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :-1, :, :] - texture_300W_labels_chromaticity[:, 1:, :, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :-1, :, :] )
# G_loss_local_albedo_const_u = tf.reduce_mean(norm_loss( albedo_300W[:, :-1, :, :], albedo_300W[:, 1:, :, :], loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_u) / tf.reduce_sum(w_u+1e-6)


# w_v = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :, :-1, :] - texture_300W_labels_chromaticity[:, :, 1:, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :, :-1, :] )
# G_loss_local_albedo_const_v = tf.reduce_mean(norm_loss( albedo_300W[:, :, :-1, :], albedo_300W[:, :, 1:, :],  loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_v) / tf.reduce_sum(w_v+1e-6)

# G_loss_local_albedo_const = (G_loss_local_albedo_const_u + G_loss_local_albedo_const_v)*10

def albedo_constancy_loss(albedo, alpha=15, weight=1.):
    '''
    for similarity of neighbors
    ref: Self-supervised Multi-level Face Model Learning for Monocular Reconstruction at over 250 Hz
        Towards High-fidelity Nonlinear 3D Face Morphable Model
    '''
    albedo_chromaticity = albedo / (torch.sum(albedo, dim=1, keepdim=True) + 1e-6)
    weight_x = torch.exp(-alpha * (albedo_chromaticity[:, :, 1:, :] - albedo_chromaticity[:, :, :-1, :]) ** 2).detach()
    weight_y = torch.exp(-alpha * (albedo_chromaticity[:, :, :, 1:] - albedo_chromaticity[:, :, :, :-1]) ** 2).detach()
    albedo_const_loss_x = ((albedo[:, :, 1:, :] - albedo[:, :, :-1, :]) ** 2) * weight_x
    albedo_const_loss_y = ((albedo[:, :, :, 1:] - albedo[:, :, :, :-1]) ** 2) * weight_y

    albedo_constancy_loss = albedo_const_loss_x.mean() + albedo_const_loss_y.mean()
    return albedo_constancy_loss * weight


def albedo_ring_loss(texcode, ring_elements, margin, weight=1.):
    """
        computes ring loss for ring_outputs before FLAME decoder
        Inputs:
          ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
          Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
          Aim is to force each row (same subject) of each stream to produce same shape
          Each row of first N-1 strams are of the same subject and
          the Nth stream is the different subject
    """
    tot_ring_loss = (texcode[0] - texcode[0]).sum()
    diff_stream = texcode[-1]
    count = 0.0
    for i in range(ring_elements - 1):
        for j in range(ring_elements - 1):
            pd = (texcode[i] - texcode[j]).pow(2).sum(1)
            nd = (texcode[i] - diff_stream).pow(2).sum(1)
            tot_ring_loss = torch.add(tot_ring_loss,
                                      (torch.nn.functional.relu(margin + pd - nd).mean()))
            count += 1.0

    tot_ring_loss = (1.0 / count) * tot_ring_loss
    return tot_ring_loss * weight


def albedo_same_loss(albedo, ring_elements, weight=1.):
    """
        computes ring loss for ring_outputs before FLAME decoder
        Inputs:
          ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
          Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
          Aim is to force each row (same subject) of each stream to produce same shape
          Each row of first N-1 strams are of the same subject and
          the Nth stream is the different subject
    """
    loss = 0
    for i in range(ring_elements - 1):
        for j in range(ring_elements - 1):
            pd = (albedo[i] - albedo[j]).pow(2).mean()
            loss += pd
    loss = loss / ring_elements
    return loss * weight


### ------------------------------------- Losses/Regularizations for vertices
def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None:
        real_2d_kp[:, :, 2] = weights[None, :] * real_2d_kp[:, :, 2]
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k


def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt) #.cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1))#.cuda()
                             ], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    return loss_lmk_2d * weight


def eye_dis(landmarks):
    # left eye:  [38,42], [39,41] - 1
    # right eye: [44,48], [45,47] -1
    eye_up = landmarks[:, [37, 38, 43, 44], :]
    eye_bottom = landmarks[:, [41, 40, 47, 46], :]
    dis = torch.sqrt(((eye_up - eye_bottom) ** 2).sum(2))  # [bz, 4]
    return dis


def eyed_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt) #.cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=landmarks_gt.device) #.cuda()
                             ], dim=-1)
    pred_eyed = eye_dis(predicted_landmarks[:, :, :2])
    gt_eyed = eye_dis(real_2d[:, :, :2])

    loss = (pred_eyed - gt_eyed).abs().mean()
    return loss


# def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
#     # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
#     # import ipdb; ipdb.set_trace()
#     if torch.is_tensor(landmarks_gt) is not True:
#         real_2d = torch.cat(landmarks_gt).cuda()
#     else:
#         real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)

#     weights = torch.ones((68,)).cuda()
#     # nose points
#     weights[27:36] = 2
#     weights[31] = weights[35] = 6
#     # inner mouth
#     weights[60:68] = 4
#     weights[48:60] = 4
#     weights[48] = weights[54] = 8

#     loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
#     return loss_lmk_2d * weight

def lip_dis(landmarks):
    # up inner lip:  [62, 63, 64] - 1
    # down innder lip: [68, 67, 66] -1
    lip_up = landmarks[:, [61, 62, 63], :]
    lip_down = landmarks[:, [67, 66, 65], :]
    dis = torch.sqrt(((lip_up - lip_down) ** 2).sum(2))  # [bz, 4]
    return dis

def mouth_corner_dis(landmarks):
    # up inner lip:  [62, 63, 64] - 1
    # down innder lip: [68, 67, 66] -1
    lip_right = landmarks[:, [48, 60], :]
    lip_left = landmarks[:, [54, 64], :]
    dis = torch.sqrt(((lip_right - lip_left) ** 2).sum(2))  # [bz, 4]
    return dis


def lipd_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt)#.cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=predicted_landmarks.device) #.cuda()
                             ], dim=-1)
    pred_lipd = lip_dis(predicted_landmarks[:, :, :2])
    gt_lipd = lip_dis(real_2d[:, :, :2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss

def mouth_corner_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt)#.cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=predicted_landmarks.device) #.cuda()
                             ], dim=-1)
    pred_lipd = mouth_corner_dis(predicted_landmarks[:, :, :2])
    gt_lipd = mouth_corner_dis(real_2d[:, :, :2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss


def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # smaller inner landmark weights
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # import ipdb; ipdb.set_trace()
    real_2d = landmarks_gt
    weights = torch.ones((68,)).to(device=predicted_landmarks.device) #.cuda()
    weights[5:7] = 2
    weights[10:12] = 2
    # nose points
    weights[27:36] = 1.5
    weights[30] = 3
    weights[31] = 3
    weights[35] = 3
    # inner mouth
    weights[60:68] = 1.5
    weights[48:60] = 1.5
    weights[48] = 3
    weights[54] = 3

    if real_2d.shape[2] == 2:
        real_2d = torch.cat([real_2d, torch.ones((real_2d.shape[0], real_2d.shape[1], 1), device=real_2d.device)],
                            dim=2)

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight


def landmark_loss_tensor(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    loss_lmk_2d = batch_kp_2d_l1_loss(landmarks_gt, predicted_landmarks)
    return loss_lmk_2d * weight


def ring_loss(ring_outputs, ring_type, margin, weight=1.):
    """
        computes ring loss for ring_outputs before FLAME decoder
        Inputs:
            ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
            Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
            Aim is to force each row (same subject) of each stream to produce same shape
            Each row of first N-1 strams are of the same subject and
            the Nth stream is the different subject
        """
    tot_ring_loss = (ring_outputs[0] - ring_outputs[0]).sum()
    if ring_type == '51':
        diff_stream = ring_outputs[-1]
        count = 0.0
        for i in range(6):
            for j in range(6):
                pd = (ring_outputs[i] - ring_outputs[j]).pow(2).sum(1)
                nd = (ring_outputs[i] - diff_stream).pow(2).sum(1)
                tot_ring_loss = torch.add(tot_ring_loss,
                                          (torch.nn.functional.relu(margin + pd - nd).mean()))
                count += 1.0

    # elif ring_type == '33':
    #     count = 0.0
    #     for i in range(3):
    #         for j in range(3):
    #             if i==j:
    #                 continue
    #             pd = (ring_outputs[i] - ring_outputs[j]).pow(2).sum(1)
    #             nd = (ring_outputs[i] - ring_outputs[j+3]).pow(2).sum(1)
    #             tot_ring_loss = torch.add(tot_ring_loss,
    #                             (torch.nn.functional.relu(margin + pd - nd).mean()))
    #             count += 1.0

    elif ring_type == '33':
        perm_code = [(0, 1, 3),
                     (0, 1, 4),
                     (0, 1, 5),
                     (0, 2, 3),
                     (0, 2, 4),
                     (0, 2, 5),
                     (1, 0, 3),
                     (1, 0, 4),
                     (1, 0, 5),
                     (1, 2, 3),
                     (1, 2, 4),
                     (1, 2, 5),
                     (2, 0, 3),
                     (2, 0, 4),
                     (2, 0, 5),
                     (2, 1, 3),
                     (2, 1, 4),
                     (2, 1, 5)]
        count = 0.0
        for i in perm_code:
            pd = (ring_outputs[i[0]] - ring_outputs[i[1]]).pow(2).sum(1)
            nd = (ring_outputs[i[1]] - ring_outputs[i[2]]).pow(2).sum(1)
            tot_ring_loss = torch.add(tot_ring_loss,
                                      (torch.nn.functional.relu(margin + pd - nd).mean()))
            count += 1.0

    tot_ring_loss = (1.0 / count) * tot_ring_loss

    return tot_ring_loss * weight


######################################## images/features/perceptual
def gradient_dif_loss(prediction, gt):
    prediction_diff_x = prediction[:, :, 1:-1, 1:] - prediction[:, :, 1:-1, :-1]
    prediction_diff_y = prediction[:, :, 1:, 1:-1] - prediction[:, :, 1:, 1:-1]
    gt_x = gt[:, :, 1:-1, 1:] - gt[:, :, 1:-1, :-1]
    gt_y = gt[:, :, 1:, 1:-1] - gt[:, :, :-1, 1:-1]
    diff = torch.mean((prediction_diff_x - gt_x) ** 2) + torch.mean((prediction_diff_y - gt_y) ** 2)
    return diff.mean()


def get_laplacian_kernel2d(kernel_size: int):
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


def laplacian_hq_loss(prediction, gt):
    # https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters/laplacian.html
    b, c, h, w = prediction.shape
    kernel_size = 3
    kernel = get_laplacian_kernel2d(kernel_size).to(device=prediction.device).to(prediction.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = (kernel_size - 1) // 2
    lap_pre = F.conv2d(prediction, kernel, padding=padding, stride=1, groups=c)
    lap_gt = F.conv2d(gt, kernel, padding=padding, stride=1, groups=c)

    # cv2.imwrite('image.jpg', util.tensor2image(gt[0]))
    # cv2.imwrite('image_lap.jpg', util.tensor2image(lap_gt[0]))
    # cv2.imwrite('image_pre.jpg', util.tensor2image(prediction[0]))
    # cv2.imwrite('image_pre_lap.jpg', util.tensor2image(lap_pre[0]))
    # import ipdb; ipdb.set_trace()
    return ((lap_pre - lap_gt) ** 2).mean()


##
class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval() #.cuda()
        ## WHY ARE THE CONSTANTS SET THIS WAY
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) #.cuda()
        # self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) #.cuda()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) )
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x / self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        # print([x for x in out])
        return out


class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        ## gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for
                           layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
                             for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss

        # loss = 0
        # for key in self.feat_style_layers.keys():
        #     loss += torch.mean((gen_vgg_feats[key] - tar_vgg_feats[key])**2)
        # return loss

    def train(self, b = True):
        # there is nothing trainable about this loss
        return super().train(False)


######################################################## vgg16 face

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        # self.mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940]) / 255.).float().view(1, 3, 1, 1) #.cuda()
        self.register_buffer('mean', torch.Tensor(np.array([129.1863, 104.7624, 93.5940]) / 255.).float().view(1, 3, 1, 1))
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()

    def load_weights(self, path="pretrained/VGG_FACE.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        out = {}
        x = x - self.mean
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        out['relu3_2'] = x
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        out['relu4_2'] = x
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = self.fc8(x)
        out['last'] = x
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.featlayer = VGG_16().float()
        self.featlayer.load_weights(path="data/face_recognition_model/vgg_face_torch/VGG_FACE.t7")
        # self.featlayer = self.featlayer.cuda().eval()
        self.featlayer = self.featlayer.eval()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        ## gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for
                           layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
                             for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss
        # loss = 0
        # for key in self.feat_style_layers.keys():
        #     loss += torch.mean((gen_vgg_feats[key] - tar_vgg_feats[key])**2)
        # return loss


############################################################## from facenet
from facenet_pytorch import InceptionResnetV1


class IdentityLoss(nn.Module):
    def __init__(self, pretrained_data='vggface2'):
        super(IdentityLoss, self).__init__()
        self.reg_model = InceptionResnetV1(pretrained=pretrained_data).eval()#.cuda()
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def _cos_metric(self, x, y, dim=1):
        return F.cosine_similarity(x, y)

    def _l2_metric(self, x, y):
        return ((x - y) ** 2).mean()

    def reg_features(self, x):
        out = []
        x = F.interpolate(x * 2. - 1., [160, 160])
        # import ipdb; ipdb.set_trace()
        x = self.reg_model.conv2d_1a(x)
        x = self.reg_model.conv2d_2a(x)
        x = self.reg_model.conv2d_2b(x)
        x = self.reg_model.maxpool_3a(x)
        # out.append(x)
        x = self.reg_model.conv2d_3b(x)
        x = self.reg_model.conv2d_4a(x)
        x = self.reg_model.conv2d_4b(x)
        x = self.reg_model.repeat_1(x)
        x = self.reg_model.mixed_6a(x)
        x = self.reg_model.repeat_2(x)
        out.append(x)
        x = self.reg_model.mixed_7a(x)
        x = self.reg_model.repeat_3(x)
        x = self.reg_model.block8(x)
        out.append(x)
        x = self.reg_model.avgpool_1a(x)
        x = self.reg_model.dropout(x)
        x = self.reg_model.last_linear(x.view(x.shape[0], -1))
        x = self.reg_model.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        out.append(x)
        return out

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar, content_loss=False, identity_loss=True, content_type='mrf', identity_type='l2'):
        ## genL [bz, 3, 160, 160] rgb [-1,1]
        # gen_embedding = self.reg_model(gen)
        # tar_embedding = self.reg_model(tar)
        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)

        if identity_loss:
            if identity_type == 'l2':
                loss = ((gen_out[-1] - tar_out[-1]) ** 2).mean()
            else:
                loss = 1 - F.cosine_similarity(gen_out[-1], tar_out[-1]).mean()
        else:
            loss = 0.
        if content_loss:
            weight = [1, 1, 1, 1]
            for i in range(len(gen_out) - 1):
                if content_type == 'mrf':
                    loss_curr = self.mrf_loss(gen_out[i], tar_out[i]) * 0.0001
                elif content_type == 'l2':
                    loss_curr = self._l2_metric(gen_out[i], tar_out[i]) * 0.02
                loss = loss + loss_curr * weight[i]
                # import ipdb; ipdb.set_trace()
        # for i in range(gen_embedding.shape[0]):
        #     for j in range(gen_embedding.shape[0]):
        #         # print('{},{}- cos:{} , l2:{}'.format(i, j, self._cos_metric(tar_embedding[i], tar_embedding[j], dim=0), self._l2_metric(tar_embedding[i], tar_embedding[j])))
        #         print('{},{}- l2:{}'.format(i, j, self._l2_metric(tar_embedding[i], tar_embedding[j])))
        # import ipdb; ipdb.set_trace()
        return loss


####################################### face recognition

# # VGGFace
from .FRNet import resnet50, load_state_dict

from .BarlowTwins import BarlowTwinsLossHeadless, BarlowTwinsLoss


class VGGFace2Loss(nn.Module):
    def __init__(self, pretrained_checkpoint_path=None, metric='cosine_similarity', trainable=False):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval()
        checkpoint = pretrained_checkpoint_path or \
                     '/ps/scratch/rdanecek/FaceRecognition/resnet50_ft_weight.pkl'
                     # '/ps/scratch/face2d3d/ringnetpp/eccv/data/resnet50_ft_weight.pkl'
        load_state_dict(self.reg_model, checkpoint)
        # this mean needs to be subtracted from the input images if using the model above
        self.register_buffer('mean_bgr', torch.tensor([91.4953, 103.8827, 131.0912]))

        self.trainable = trainable

        if metric is None:
            metric = 'cosine_similarity'

        if metric not in ["l1", "l1_loss", "l2", "mse", "mse_loss", "cosine_similarity",
                          "barlow_twins", "barlow_twins_headless"]:
            raise ValueError(f"Invalid metric for face recognition feature loss: {metric}")

        if metric == "barlow_twins_headless":
            feature_size = self.reg_model.fc.in_features
            self.bt_loss = BarlowTwinsLossHeadless(feature_size)
        elif metric == "barlow_twins":
            feature_size = self.reg_model.fc.in_features
            self.bt_loss = BarlowTwinsLoss(feature_size)
        else:
            self.bt_loss = None

        self.metric = metric

    def _get_trainable_params(self):
        params = []
        if self.trainable:
            params += list(self.reg_model.parameters())
        if self.bt_loss is not None:
            params += list(self.bt_loss.parameters())
        return params

    def train(self, b = True):
        if not self.trainable:
            ret = super().train(False)
        else:
            ret = super().train(b)
        if self.bt_loss is not None:
            self.bt_loss.train(b)
        return ret

    def requires_grad_(self, b):
        super().requires_grad_(False) # face recognition net always frozen
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(b)

    def freeze_nontrainable_layers(self):
        if not self.trainable:
            super().requires_grad_(False)
        else:
            super().requires_grad_(True)
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(True)

    def reg_features(self, x):
        # TODO: is this hard-coded margin necessary?
        margin = 10
        x = x[:, :, margin:224 - margin, margin:224 - margin]
        x = F.interpolate(x * 2. - 1., [224, 224], mode='bilinear')
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # input images in RGB in range [0-1] but the network expects them in BGR  [0-255] with subtracted mean_bgr
        img = img[:, [2, 1, 0], :, :].permute(0, 2, 3, 1) * 255 - self.mean_bgr
        img = img.permute(0, 3, 1, 2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True, batch_size=None, ring_size=None):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)

        if self.metric == "cosine_similarity":
            loss = self._cos_metric(gen_out, tar_out).mean()
        elif self.metric in ["l1", "l1_loss", "mae"]:
            loss = torch.nn.functional.l1_loss(gen_out, tar_out)
        elif self.metric in ["mse", "mse_loss", "l2", "l2_loss"]:
            loss = torch.nn.functional.mse_loss(gen_out, tar_out)
        elif self.metric in ["barlow_twins_headless", "barlow_twins"]:
            loss = self.bt_loss(gen_out, tar_out, batch_size=batch_size, ring_size=ring_size)
        else:
            raise ValueError(f"Invalid metric for face recognition feature loss: {self.metric}")

        return loss

#
# if __name__ == "__main__":
#     loss = IDMRFLoss()
#     dummy = torch.zeros(size=(1,3, 256, 256))
#     loss(dummy, dummy)
#     print("ha")