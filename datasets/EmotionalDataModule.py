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

import gc
# from memory_profiler import profile

from enum import Enum


class EmotionDataModule(pl.LightningDataModule):

    def __init__(self, dm, image_size = 256):
        super().__init__()
        self.dm = dm
        self.image_size=image_size

    def prepare_data(self, *args, **kwargs):
        self.dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        pass

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self, annotation_list = None, filter_pattern=None, **dl_kwargs) \
            -> Union[DataLoader, List[DataLoader]]:
        from torchvision.transforms import Resize
        transforms = Resize((self.image_size, self.image_size))
        dataset = self.dm.get_annotated_emotion_dataset(annotation_list, filter_pattern, transforms)
        dl = DataLoader(dataset,  shuffle=False, **dl_kwargs)
        return dl

    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     pass



