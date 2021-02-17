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
import copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DECA')))
# from decalib.deca import DECA
# from decalib.datasets import datasets
# from utils.FaceDetector import FAN, MTCNN
# from facenet_pytorch import InceptionResnetV1
# from collections import OrderedDict
from PIL import Image
# import gc
# from memory_profiler import profile

from torchvision.transforms import Resize
from transforms.keypoints import KeypointScale, KeypointNormalization

class EmotionDataModule(pl.LightningDataModule):

    def __init__(self, dm, image_size=256, with_landmarks=False, with_segmentations=False,
                 train_K=None,
                 val_K=None,
                 test_K=None,
                 train_K_policy = None,
                 val_K_policy = None,
                 test_K_policy = None,
                 annotation_list = None,
                 filter_pattern = None,
                 split_ratio = None,
                 split_style = None,
                 num_workers=0,
                 train_batch_size=1,
                 val_batch_size=1,
                 test_batch_size=1
                 ):
        super().__init__()
        self.dm = dm
        self.image_size = image_size
        self.training_set = None
        self.validation_set = None
        self.testing_set = None
        self.with_landmarks = with_landmarks
        self.with_segmentations = with_segmentations
        self.train_K = train_K
        self.val_K = val_K
        self.test_K = test_K
        self.train_K_policy = train_K_policy
        self.val_K_policy = val_K_policy
        self.test_K_policy = test_K_policy
        self.annotation_list = annotation_list
        self.filter_pattern = filter_pattern
        self.split_ratio = split_ratio
        self.split_style = split_style
        self.num_workers = num_workers
        self.training_set = None
        self.test_set = None
        self.validation_set = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        self.dm.prepare_data()
        self.dm.setup()
        if self.training_set is not None:
            return
        from torchvision.transforms import Resize
        im_transforms = Resize((self.image_size, self.image_size), Image.BICUBIC)
        # lmk_transforms = KeypointScale()
        lmk_transforms = KeypointNormalization()
        seg_transforms = Resize((self.image_size, self.image_size), Image.NEAREST)
        dataset = self.dm.get_annotated_emotion_dataset(
            copy.deepcopy(self.annotation_list), self.filter_pattern, image_transforms=im_transforms,
            split_style=self.split_style, split_ratio=self.split_ratio,
            with_landmarks=self.with_landmarks,
            landmark_transform=lmk_transforms,
            with_segmentations=self.with_segmentations,
            segmentation_transform=seg_transforms,
            K=self.train_K,
            K_policy=self.train_K_policy)
        if not (isinstance(dataset, list) or isinstance(dataset, tuple)):
            dataset = [dataset,]
        self.training_set = dataset[0]

        if len(dataset) > 1:
            self.validation_set = dataset[1]
            self.validation_set.K = self.val_K
            self.validation_set.K_policy = self.val_K_policy
            self.indices_train = dataset[2]
            self.indices_val = dataset[3]

        im_transforms = Resize((self.image_size, self.image_size))
        lmk_transforms = KeypointNormalization()
        seg_transforms = Resize((self.image_size, self.image_size), Image.NEAREST)
        self.test_set = self.dm.get_annotated_emotion_dataset(copy.deepcopy(self.annotation_list), self.filter_pattern,
                                                        image_transforms=im_transforms,
                                                        with_landmarks = self.with_landmarks,
                                                        landmark_transform=lmk_transforms,
                                                        with_segmentations=self.with_segmentations,
                                                        segmentation_transform=seg_transforms,
                                                        K=self.test_K,
                                                        K_policy=self.test_K_policy)

    def train_dataloader(self,*args, **kwargs) -> DataLoader:
        dl = DataLoader(self.training_set, shuffle=True, num_workers=self.num_workers, batch_size=self.train_batch_size)
        return dl

    def val_dataloader(self,*args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.validation_set, shuffle=False, num_workers=self.num_workers, batch_size=self.val_batch_size)

    def test_dataloader(self, *args, **dl_kwargs)  -> Union[DataLoader, List[DataLoader]]:
        dl = DataLoader(self.test_set,  shuffle=False, num_workers=self.num_workers, batch_size=self.test_batch_size)
        return dl
