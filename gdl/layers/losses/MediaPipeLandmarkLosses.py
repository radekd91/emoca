import numpy as np
import torch

from gdl.utils.MediaPipeLandmarkLists import left_eye_landmark_indices, right_eye_landmark_indices, mouth_landmark_indices

## MEDIAPIPE LANDMARK DESCRIPTIONS 

# LEFT EYE
# perspective of the landmarked person
LEFT_EYE_LEFT_CORNER = 263
LEFT_EYE_RIGHT_CORNER = 362 
# the upper and lower eyelid points are in correspondences, ordered from right to left (perspective of the landmarked person)
LEFT_UPPER_EYELID_INDICES = [398, 384, 385, 386, 387, 388, 466]
LEFT_LOWER_EYELID_INDICES = [382, 381, 380, 374, 373, 390, 249]

LEFT_UPPER_EYEBROW_INDICES = [336, 296, 334, 293, 300]
LEFT_LOWER_EYEBROW_INDICES = [285, 295, 282, 283, 276]

# RIGHT EYE
# perspective of the landmarked person
RIGHT_EYE_LEFT_CORNER = 133
RIGHT_EYE_RIGHT_CORNER = 33 
# the upper and lower eyelid points are in correspondences, ordered from right to left (perspective of the landmarked person)
RIGHT_UPPER_EYELID_INDICES = [246, 161, 160, 159, 158, 157, 173]
RIGHT_LOWER_EYELID_INDICES = [7  , 163, 144, 145, 153, 154, 155]

RIGHT_UPPER_EYEBROW_INDICES = [ 70,  63, 105,  66, 107]
RIGHT_LOWER_EYEBROW_INDICES = [ 46,  53,  52,  65,  55]

# MOUTH
LEFT_INNER_LIP_CORNER = 308 
LEFT_OUTTER_LIP_CORNER = 291 
RIGHT_INNER_LIP_CORNER = 78
RIGHT_OUTTER_LIP_CORNER = 61 
# from right to left, the upper and lower are in correspondence
UPPER_INNER_LIP_LINE = [191,  80, 81 , 82 , 13 , 312, 311, 310, 415]
LOWER_INNER_LIP_LINE = [ 95,  88, 178, 87 , 14 , 317, 402, 318, 324]
# from right to left, the upper and lower are in correspondence
UPPER_OUTTER_LIP_LINE = [185,  40,  39,  37,   0, 267, 269, 270, 409]
LOWER_OUTTER_LIP_LINE = [146,  91, 181,  84,  17, 314, 405, 321, 375]

# NOSE
# from up (between the eyes) downards (nose tip)
VERTICAL_NOSE_LINE = [168, 6, 197, 195, 5, 4]
# from right (next to the right nostril, just under the right nostril , under the nose) to left (landmarked person perspective)
HORIZONTAL_NOSE_LINE = [129,  98, 97,  2, 326, 327, 358]


# COMBINED LISTS 
UPPER_EYELIDS = np.array(sorted(LEFT_UPPER_EYELID_INDICES + RIGHT_UPPER_EYELID_INDICES), dtype=np.int64)
LOWER_EYELIDS = np.array(sorted(LEFT_LOWER_EYELID_INDICES + RIGHT_LOWER_EYELID_INDICES), dtype=np.int64) 
UPPER_EYELIDS_TORCH = torch.from_numpy(UPPER_EYELIDS).long()
LOWER_EYELIDS_TORCH = torch.from_numpy(LOWER_EYELIDS).long()

EMBEDDING_INDICES = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
        55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
        381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
        144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
        168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
          0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
        87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
        308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
        415]

EMBEDDING_INDICES_NP = np.array(EMBEDDING_INDICES, dtype=np.int64)

MEDIAPIPE_LANDMARK_NUMBER = 478
NON_EMBEDDING_INDICES = [i for i in range(MEDIAPIPE_LANDMARK_NUMBER ) if i not in EMBEDDING_INDICES]
NON_EMBEDDING_INDICES_NP = np.array(NON_EMBEDDING_INDICES, dtype=np.int64)

sorter = np.argsort(EMBEDDING_INDICES)
UPPER_EYELIDS_EM = sorter[np.searchsorted(EMBEDDING_INDICES, UPPER_EYELIDS, sorter=sorter)]
LOWER_EYELIDS_EM = sorter[np.searchsorted(EMBEDDING_INDICES, LOWER_EYELIDS, sorter=sorter)]



UPPER_OUTTER_LIP_LINE_EM = sorter[np.searchsorted(EMBEDDING_INDICES, UPPER_OUTTER_LIP_LINE, sorter=sorter)]
LOWER_OUTTER_LIP_LINE_EM = sorter[np.searchsorted(EMBEDDING_INDICES, LOWER_OUTTER_LIP_LINE, sorter=sorter)]
LOWER_INNER_LIP_LINE_EM = sorter[np.searchsorted(EMBEDDING_INDICES, LOWER_INNER_LIP_LINE, sorter=sorter)]
UPPER_INNER_LIP_LINE_EM = sorter[np.searchsorted(EMBEDDING_INDICES, UPPER_INNER_LIP_LINE, sorter=sorter)]

RIGHT_INNER_LIP_CORNER_EM =  sorter[np.searchsorted(EMBEDDING_INDICES, np.array([RIGHT_INNER_LIP_CORNER]), sorter=sorter)]
LEFT_INNER_LIP_CORNER_EM =  sorter[np.searchsorted(EMBEDDING_INDICES, np.array([LEFT_INNER_LIP_CORNER]), sorter=sorter)]
RIGHT_OUTTER_LIP_CORNER_EM = sorter[np.searchsorted(EMBEDDING_INDICES, np.array([RIGHT_OUTTER_LIP_CORNER]), sorter=sorter)]
LEFT_OUTTER_LIP_CORNER_EM = sorter[np.searchsorted(EMBEDDING_INDICES, np.array([LEFT_OUTTER_LIP_CORNER]), sorter=sorter)]


def get_mediapipe_indices():
    # This index array contains indices of mediapipe landmarks that are selected by Timo. 
    # These include the eyes, eyebrows, nose, and mouth. Not the face contour and others. 
    # Loaded from mediapipe_landmark_embedding.npz by Timo.
    indices = np.array([276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
        55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
        381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
        144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
        168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
          0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
        87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
        308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
        415])
    return indices


def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None:
        real_2d_kp[..., 2] = weights[None, :] * real_2d_kp[..., 2]
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[..., 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[..., :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k


def landmark_loss(predicted_landmarks, landmarks_gt, weights=None):
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt)
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1))
    #                          ], dim=-1)

    # loss_lmk_2d = batch_kp_2d_l1_loss(
    #     landmarks_gt[..., EMBEDDING_INDICES, :], 
    #     # real_2d[..., get_mediapipe_indices(), :], 
    #     predicted_landmarks[..., :, :])
    assert predicted_landmarks[..., :2].isnan().sum() == 0
    assert landmarks_gt[..., :2].isnan().sum() == 0
    loss_lmk_2d = (predicted_landmarks[..., :2] - landmarks_gt[..., EMBEDDING_INDICES, :2]).abs()
    if loss_lmk_2d.ndim == 3:
        loss_lmk_2d= loss_lmk_2d.mean(dim=2)
    elif loss_lmk_2d.ndim == 4: 
        loss_lmk_2d = loss_lmk_2d.mean(dim=(2,3))
    else: 
        raise ValueError(f"Wrong dimension of loss_lmk_2d: { loss_lmk_2d.ndim}")
    if weights is None: 
        return loss_lmk_2d.mean()
    if weights.sum().abs() < 1e-8:
        return torch.tensor(0)
    if weights is not None:
        w = weights / torch.sum(weights)
        loss_lmk_2d = w * loss_lmk_2d
        return loss_lmk_2d.sum()
    return loss_lmk_2d 



def lip_dis(lip_up, lip_down):
    # lip_up = landmarks[:, UPPER_OUTTER_LIP_LINE + UPPER_INNER_LIP_LINE, :]
    # lip_down = landmarks[:, LOWER_OUTTER_LIP_LINE + LOWER_INNER_LIP_LINE, :]
    dis = torch.sqrt(((lip_up - lip_down) ** 2).sum(2))  # [bz, 4]
    return dis


def mouth_corner_dis(lip_right, lip_left):
    # lip_right = landmarks[:, [LEFT_INNER_LIP_CORNER, LEFT_OUTTER_LIP_CORNER], :]
    # lip_left = landmarks[:,  [RIGHT_INNER_LIP_CORNER, RIGHT_OUTTER_LIP_CORNER], :]
    dis = torch.sqrt(((lip_right - lip_left) ** 2).sum(2))  # [bz, 4]
    return dis


def lipd_loss(predicted_landmarks, landmarks_gt, weights=None):
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt)
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=predicted_landmarks.device) #.cuda()
    #                          ], dim=-1)
    pred_lipd = lip_dis(predicted_landmarks[...,  np.concatenate([UPPER_OUTTER_LIP_LINE_EM, UPPER_INNER_LIP_LINE_EM]), :2] , 
                        predicted_landmarks[...,  np.concatenate([LOWER_OUTTER_LIP_LINE_EM, LOWER_INNER_LIP_LINE_EM]), :2])
    gt_lipd = lip_dis(landmarks_gt[...,  UPPER_OUTTER_LIP_LINE + UPPER_INNER_LIP_LINE, :2] , 
                      landmarks_gt[...,  LOWER_OUTTER_LIP_LINE + LOWER_INNER_LIP_LINE, :2])

    # gt_lipd = lip_dis(real_2d[... :2])

    loss = (pred_lipd - gt_lipd).abs()
    if weights is None: 
        return loss.mean()
    if weights.sum().abs() < 1e-8:
        return torch.tensor(0)
    if loss.ndim == 3:
        loss = loss.mean(dim=2)
    elif loss.ndim == 4: 
        loss = loss.mean(dim=(2,3))
    w = weights / torch.sum(weights)
    loss = w * loss
    return loss.sum()


def mouth_corner_loss(predicted_landmarks, landmarks_gt, weights=None):
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt)
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=predicted_landmarks.device) #.cuda()
    #                          ], dim=-1)

    pred_corner_d = mouth_corner_dis(
            predicted_landmarks[...,  np.concatenate([RIGHT_INNER_LIP_CORNER_EM, RIGHT_OUTTER_LIP_CORNER_EM]) , :2],
            predicted_landmarks[...,  np.concatenate([LEFT_INNER_LIP_CORNER_EM, LEFT_OUTTER_LIP_CORNER_EM]) , :2]
            )
    gt_corner_d = mouth_corner_dis(
            landmarks_gt[...,  [RIGHT_INNER_LIP_CORNER, RIGHT_OUTTER_LIP_CORNER] , :2],
            landmarks_gt[...,  [LEFT_INNER_LIP_CORNER, LEFT_OUTTER_LIP_CORNER] , :2])
    # gt_corner_d = mouth_corner_dis(real_2d[:, :, :2])

    loss = (pred_corner_d - gt_corner_d).abs()
    if weights is None: 
        return loss.mean()
    if weights.sum().abs() < 1e-8:
        return torch.tensor(0)
    if loss.ndim == 3:
        loss = loss.mean(dim=2)
    elif loss.ndim == 4: 
        loss = loss.mean(dim=(2,3))
    w = weights / torch.sum(weights)
    loss = w * loss
    return loss.sum()


def eye_dis(eye_upper, eye_lower):
    # eye_upper = landmarks[:, UPPER_EYELIDS_TORCH, :][..., :2]
    # eye_lower = landmarks[:, LOWER_EYELIDS_TORCH, :][..., :2]
    dis = torch.sqrt(((eye_upper - eye_lower) ** 2).sum(2))  # [bz, 4]
    return dis


def eyed_loss(predicted_landmarks, landmarks_gt, weights=None):
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt)
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=landmarks_gt.device) #.cuda()
    #                          ], dim=-1)
    pred_eyed = eye_dis(predicted_landmarks[..., UPPER_EYELIDS_EM , :2], 
                        predicted_landmarks[..., LOWER_EYELIDS_EM , :2])
    gt_eyed = eye_dis(landmarks_gt[..., UPPER_EYELIDS, :2], 
                        landmarks_gt[..., LOWER_EYELIDS, :2])
    # gt_eyed = eye_dis(real_2d[:, :, :2])

    loss = (pred_eyed - gt_eyed).abs().mean()
    if weights is None: 
        return loss.mean()
    if weights.sum().abs() < 1e-8:
        return torch.tensor(0)
    if loss.ndim == 3:
        loss = loss.mean(dim=2)
    elif loss.ndim == 4: 
        loss = loss.mean(dim=(2,3))
    w = weights / torch.sum(weights)
    loss = w * loss
    return loss.sum()
