import pickle as pkl
import compress_pickle as cpkl
from pathlib import Path
import numpy as np
from timeit import default_timer as timer


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


face_parsing_labels = {
    0: 'background',  # no
    1: 'skin',
    2: 'nose',
    3: 'eye_g',
    4: 'l_eye',
    5: 'r_eye',
    6: 'l_brow',
    7: 'r_brow',
    8: 'l_ear',  # no?
    9: 'r_ear',  # no?
    10: 'mouth',
    11: 'u_lip',
    12: 'l_lip',
    13: 'hair',  # no
    14: 'hat',  # no
    15: 'ear_r',
    16: 'neck_l',  # no?
    17: 'neck',  # no?
    18: 'cloth'  # no
}

face_parsin_inv_labels = {v: k for k, v in face_parsing_labels.items()}

default_discarded_labels = [
    face_parsin_inv_labels['background'],
    face_parsin_inv_labels['l_ear'],
    face_parsin_inv_labels['r_ear'],
    face_parsin_inv_labels['hair'],
    face_parsin_inv_labels['hat'],
    face_parsin_inv_labels['neck'],
    face_parsin_inv_labels['neck_l']
]


def process_segmentation(segmentation, seg_type, discarded_labels=None):
    if seg_type == "face_parsing":
        discarded_labels = discarded_labels or default_discarded_labels
        # start = timer()
        # segmentation_proc = np.ones_like(segmentation, dtype=np.float32)
        # for label in discarded_labels:
        #     segmentation_proc[segmentation == label] = 0.
        segmentation_proc = np.isin(segmentation, discarded_labels)
        segmentation_proc = np.logical_not(segmentation_proc)
        segmentation_proc = segmentation_proc.astype(np.float32)
        # end = timer()
        # print(f"Segmentation label discarding took {end - start}s")
        return segmentation_proc
    else:
        raise ValueError(f"Invalid segmentation type '{seg_type}'")
