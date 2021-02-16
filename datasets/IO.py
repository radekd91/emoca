import pickle as pkl
import compress_pickle as cpkl
from pathlib import Path
import numpy as np


def save_segmentation(filename, seg_image, seg_type):
    with open(filename, "wb") as f:
        # for some reason compressed pickle can only load one object (EOF bug)
        # so put it in the list
        cpkl.dump([seg_type, seg_image], f, compression='gzip')
        # pkl.dump(seg_type, f)
        # pkl.dump(seg_image, f)


def load_segmentation(filename):
    with open(filename, "rb") as f:
        seg = cpkl.load(f, compression='gzip')
        seg_type = seg[0]
        seg_image = seg[1]
        # seg_type = pkl.load(f)
        # seg_image = pkl.load(f)
    return seg_image, seg_type


def process_segmentation(segmentation, seg_type, discarded_labels):
    if seg_type == "face_parsing":
        labels = {
            0: 'background', # no
            1: 'skin',
            2: 'nose',
            3: 'eye_g',
            4: 'l_eye',
            5: 'r_eye',
            6: 'l_brow',
            7: 'r_brow',
            8: 'l_ear', #no?
            9: 'r_ear', # no?
            10: 'mouth',
            11: 'u_lip',
            12: 'l_lip',
            13: 'hair', # no
            14: 'hat', # no
            15: 'ear_r',
            16: 'neck_l', # no?
            17: 'neck', # no?
            18: 'cloth' # no
        }
        inv_labels = {v: k for k, v in labels.items()}
        discarded_labels = discarded_labels or [
            inv_labels['background'],
            inv_labels['l_ear'],
            inv_labels['r_ear'],
            inv_labels['hair'],
            inv_labels['hat'],
            inv_labels['neck'],
            inv_labels['neck_l']
        ]
        segmentation_proc = np.logical_not(np.isin(segmentation, discarded_labels))
        segmentation_proc = segmentation_proc.astype(np.float32)
        return segmentation_proc
    else:
        raise ValueError(f"Invalid segmentation type '{seg_type}'")
