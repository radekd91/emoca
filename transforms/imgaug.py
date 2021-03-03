import imgaug
import imgaug.augmenters.meta as meta


def augmenter_from_key_value(name, kwargs):
    if hasattr(meta, name):
        sub_augmenters = []
        for item in kwargs:
            key = list(item.keys())[0]
            # kwargs_ = {k: v for d in item for k, v in d.items()}
            sub_augmenters += [augmenter_from_key_value(key, item[key])]
            # sub_augmenters += [augmenter_from_key_value(key, kwargs[key])]
        cl = getattr(imgaug.augmenters, name)
        return cl(sub_augmenters)

    if hasattr(imgaug.augmenters, name):
        cl = getattr(imgaug.augmenters, name)
        kwargs_ = {k: v for d in kwargs for k, v in d.items()}
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
    augmenter_list = [imgaug.augmenters.Resize(im_size)]
    if augmentation is not None:
        augmenter_list += [augmenter_from_dict(augmentation)]
    augmenter = imgaug.augmenters.Sequential(augmenter_list)
    return augmenter
