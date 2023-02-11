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


import pickle as pkl
import compress_pickle as cpkl
import hickle as hkl
from pathlib import Path
import numpy as np
from timeit import default_timer as timer


def load_reconstruction_list(filename):
    reconstructions = hkl.load(filename)
    return reconstructions


def save_reconstruction_list(filename, reconstructions):
    hkl.dump(reconstructions, filename)


def load_emotion_list(filename):
    emotions = hkl.load(filename)
    return emotions


def save_emotion_list(filename, emotions):
    hkl.dump(emotions, filename)


def save_segmentation_list(filename, seg_images, seg_types, seg_names):
    with open(filename, "wb") as f:
        # for some reason compressed pickle can only load one object (EOF bug)
        # so put it in the list
        cpkl.dump([seg_types, seg_images, seg_names], f, compression='gzip')
        # pkl.dump(seg_type, f)
        # pkl.dump(seg_image, f)


def load_segmentation_list(filename):
    try:
        with open(filename, "rb") as f:
            seg = cpkl.load(f, compression='gzip')
            seg_types = seg[0]
            seg_images = seg[1]
            seg_names = seg[2]
    except EOFError as e: 
        print(f"Error loading segmentation list: {filename}")
        raise e
    return seg_images, seg_types, seg_names


def load_segmentation(filename):
    with open(filename, "rb") as f:
        seg = cpkl.load(f, compression='gzip')
        seg_type = seg[0]
        seg_image = seg[1]
        # seg_type = pkl.load(f)
        # seg_image = pkl.load(f)
    return seg_image, seg_type



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


def save_emotion(filename, emotion_features, emotion_type, version=0):
    with open(filename, "wb") as f:
        # for some reason compressed pickle can only load one object (EOF bug)
        # so put it in the list
        cpkl.dump([version, emotion_type, emotion_features], f, compression='gzip')


def load_emotion(filename):
    with open(filename, "rb") as f:
        emo = cpkl.load(f, compression='gzip')
        version = emo[0]
        emotion_type = emo[1]
        emotion_features = emo[2]
    return emotion_features, emotion_type



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


def load_and_process_segmentation(path):
    seg_image, seg_type = load_segmentation(path)
    seg_image = seg_image[np.newaxis, :, :, np.newaxis]
    # end = timer()
    # print(f"Segmentation reading took {end - start} s.")

    # start = timer()
    seg_image = process_segmentation(
        seg_image, seg_type).astype(np.uint8)
    return seg_image
