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


import imgaug
import imgaug.augmenters.meta as meta
import imgaug.augmenters as aug


def augmenter_from_key_value(name, kwargs):
    if hasattr(meta, name):
        sub_augmenters = []
        kwargs_ = {}
        for item in kwargs:
            key = list(item.keys())[0]
            if hasattr(aug, key): # is augmenter? if so, create and add to list
                sub_augmenters += [augmenter_from_key_value(key, item[key])]
            else: # otherwise it's not an augmenter and it will be a kwarg for this augmenter
                kwargs_[key] = item[key]
        cl = getattr(imgaug.augmenters, name)
        args_ = []
        if len(sub_augmenters) > 0:
            args_ += [sub_augmenters]
        return cl(*args_, **kwargs_)

    if hasattr(imgaug.augmenters, name):
        cl = getattr(imgaug.augmenters, name)
        kwargs_ = {k: v for d in kwargs for k, v in d.items()}
        for key in kwargs_.keys():
            if isinstance(kwargs_[key], list):
               kwargs_[key] = tuple(kwargs_[key])
        return cl(**kwargs_)

    raise RuntimeError(f"Augmenter with name '{name}' is either not supported or it does not exist")


def augmenter_from_dict(augmentation):
    augmenter_list = []
    for aug in augmentation:
        if len(aug) > 1:
            raise RuntimeError("This should be just a single element")
        key = list(aug.keys())[0]
        augmenter_list += [augmenter_from_key_value(key, kwargs=aug[key])]
    return imgaug.augmenters.Sequential(augmenter_list)


def create_image_augmenter(im_size, augmentation=None) -> imgaug.augmenters.Augmenter:
    # augmenter_list = [imgaug.augmenters.Resize(im_size)]
    augmenter_list = []
    if augmentation is not None:
        augmenter_list += [augmenter_from_dict(augmentation)]
    augmenter_list += [imgaug.augmenters.Resize(im_size)]
    augmenter = imgaug.augmenters.Sequential(augmenter_list)
    return augmenter
