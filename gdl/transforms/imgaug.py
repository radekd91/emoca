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
