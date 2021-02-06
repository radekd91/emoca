from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

import glob, os, sys
from pathlib import Path
# import pyvista as pv
# from utils.mesh import load_mesh
# from scipy.io import wavfile
# import resampy
import numpy as np
import torch
# import torchaudio
# from enum import Enum
from typing import Optional, Union, List, Any, overload
import pickle as pkl
# from collections import OrderedDict
from tqdm import tqdm
# import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DECA')))
# from decalib.deca import DECA
# from decalib.datasets import datasets
from utils.FaceDetector import FAN, MTCNN
from facenet_pytorch import InceptionResnetV1
from collections import OrderedDict
from PIL import Image
import gc
# from memory_profiler import profile

from enum import Enum


class EmotionDataModule(pl.LightningDataModule):

    def __init__(self, dm, image_size=256, with_landmarks=False, with_segmentations=False):
        super().__init__()
        self.dm = dm
        self.image_size = image_size
        self.training_set = None
        self.validation_set = None
        self.testing_set = None
        self.with_landmarks = with_landmarks
        self.with_segmentations = with_segmentations

    def prepare_data(self, *args, **kwargs):
        self.dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self,
                         annotation_list = None,
                         filter_pattern=None,
                         split_ratio=None,
                         split_style=None,
                         with_landmarks=False,
                         **dl_kwargs) -> DataLoader:
        from torchvision.transforms import Resize
        from transforms.keypoints import KeypointScale
        im_transforms = Resize((self.image_size, self.image_size), Image.BICUBIC)
        lmk_transforms = KeypointScale()
        seg_transforms = Resize((self.image_size, self.image_size), Image.NEAREST)
        dataset = self.dm.get_annotated_emotion_dataset(
            annotation_list, filter_pattern, image_transforms=im_transforms,
            split_style=split_style, split_ratio=split_ratio,
            with_landmarks=self.with_landmarks,
            landmark_transform=lmk_transforms,
            with_segmentations=self.with_segmentations,
            segmentation_transform=seg_transforms)
        if not (isinstance(dataset, list) or isinstance(dataset, tuple)):
            dataset = [dataset,]
        self.training_set = dataset[0]

        if len(dataset) > 1:
            self.validation_set = dataset[1]
            self.indices_train = dataset[2]
            self.indices_val = dataset[3]

        dl = DataLoader(self.training_set, shuffle=True, **dl_kwargs)
        return dl

    def val_dataloader(self, annotation_list = None, filter_pattern=None, **dl_kwargs) \
            -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.validation_set, shuffle=False, **dl_kwargs)

    def test_dataloader(self, annotation_list = None, filter_pattern=None, **dl_kwargs) \
            -> Union[DataLoader, List[DataLoader]]:
        from torchvision.transforms import Resize
        transforms = Resize((self.image_size, self.image_size))
        dataset = self.dm.get_annotated_emotion_dataset(annotation_list, filter_pattern, transforms, with_landmarks = self.with_landmarks)
        dl = DataLoader(dataset,  shuffle=False, **dl_kwargs)
        return dl
