from skimage.io import imsave
from pathlib import Path
from wandb import Image
import numpy as np


def _fix_image( image):
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _log_wandb_image(path, image, caption=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _fix_image(image)
    imsave(path, image)
    if caption is not None:
        caption_file = Path(path).parent / (Path(path).stem + ".txt")
        with open(caption_file, "w") as f:
            f.write(caption)
    wandb_image = Image(str(path), caption=caption)
    return wandb_image


def _log_array_image(path, image, caption=None):
    image = _fix_image(image)
    if path is not None:
        imsave(path, image)
    return image


def _torch_image2np(torch_image):
    image = torch_image.detach().cpu().numpy()
    if len(image.shape) == 4:
        image = image.transpose([0, 2, 3, 1])
    elif len(image.shape) == 3:
        image = image.transpose([1, 2, 0])
    return image
