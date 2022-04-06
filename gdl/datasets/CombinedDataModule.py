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


from pytorch_lightning import LightningDataModule
# from .AffectNetDataModule import AffectNetDataModule, AffectNetEmoNetSplitModuleValTest, AffectNetEmoNetSplitModule
# from .DecaDataModule import DecaDataModule
import torch
from torch.utils.data._utils.collate import *
import numpy as np


def combined_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        max_size = 0
        max_i = 0
        min_size = 10000000
        # need to check if all elements in the batch have the same size on dim 0
        # for i, elem_ in enumerate(batch):
        #     if elem_.size(0) > max_size:
        #         max_size = elem_.size(0)
        #         max_i = i
        #     if elem_.size(0) < min_size:
        #         min_size = elem_.size(0)
        #         min_i = i
        #
        # # if not, fix this by duplication
        # new_batch = []
        # for i, elem_ in enumerate(batch):
        #     if batch[i].shape[0] != max_size:
        #         # batch[i] = batch[i].repeat(max_size, 1, 1, 1)
        #         # batch[i] = batch[i][:max_size, ...].clone()
        #         new_batch += [batch[i].repeat(max_size, 1, 1, 1)[:max_size, ...].clone()]
        #     else:
        #         new_batch += [batch[i].clone()]
        # final = torch.stack(new_batch, 0, out=out)
        final = torch.stack(batch, 0, out=out)
        return final
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return combined_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: combined_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        combined =elem_type(*(combined_collate(samples) for samples in zip(*batch)))
        return combined
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [combined_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class CombinedDataModule(LightningDataModule):

    def __init__(self, dms, weights=None):
        super().__init__()
        assert len(dms) > 0
        self.dms = dms
        if weights is None:
            self.weights = {key: 1.0 for key in dms.keys()}
        else:
            self.weights = weights

        # for key, value in config.items():
        #     if data_preparation_function is None:
        #         dm_class = class_from_str(key, sys.modules[__name__])
        #         self.dms[key] = dm_class(**value)
        #     else:
        #         self.dms[key] = data_preparation_function(value)

    def train_sampler(self):
        sample_weights_list = []
        for key, value in self.dms.items():
            sampler = value.train_sampler()
            if sampler is not None:
                sample_weights = sampler.weights
            else:
                sample_weights = torch.tensor([1.0] * len(value.training_set))
            sample_weights /= sample_weights.sum()
            sample_weights *= self.weights[key]
            sample_weights_list += [sample_weights]
        final_sample_weights = torch.cat(sample_weights_list)
        final_sample_weights /= final_sample_weights.sum()
        return torch.utils.data.WeightedRandomSampler(final_sample_weights, len(final_sample_weights))


    @property
    def train_batch_size(self):
        for key, value in self.dms.items():
            return value.train_batch_size
        raise RuntimeError("Something went wrong")

    @property
    def val_batch_size(self):
        for key, value in self.dms.items():
            return value.val_batch_size
        raise RuntimeError("Something went wrong")

    @property
    def test_batch_size(self):
        for key, value in self.dms.items():
            return value.test_batch_size
        raise RuntimeError("Something went wrong")

    @property
    def num_workers(self):
        for key, value in self.dms.items():
            return value.num_workers
        raise RuntimeError("Something went wrong")

    def prepare_data(self):
        for key, dm in self.dms.items():
            dm.prepare_data()

    def setup(self, stage=None):
        for key, dm in self.dms.items():
            dm.setup(stage)

    def train_dataloader(self):
        datasets = []
        for key, dm in self.dms.items():
            datasets += [dm.training_set]
            datasets[-1].drop_last = self.train_batch_size
        concat_dataset = torch.utils.data.ConcatDataset(datasets)
        # concat_dataset = torch.utils.data.ChainDataset(datasets)
        # sampler = self.train_sampler()
        sampler = None
        dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=self.train_batch_size,
                                                 num_workers=self.num_workers, pin_memory=True,
                                                 drop_last=True,  shuffle=sampler is None, sampler=sampler,
                                                 collate_fn=combined_collate)
        return dataloader

    def val_dataloader(self):
        first_datasets = []
        other_datasets = []
        for key, dm in self.dms.items():
            if isinstance(dm.validation_set, list):
                first_datasets += [dm.validation_set[0]]
                other_datasets += dm.validation_set[1:]
            else:
                first_datasets += [dm.validation_set]
            first_datasets[-1].drop_last = self.val_batch_size
        concat_dataset = torch.utils.data.ConcatDataset(first_datasets)
        dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=self.val_batch_size,
                                                 num_workers=self.num_workers,
                                                 drop_last=False,  shuffle=False, pin_memory=True,
                                                 collate_fn=combined_collate)

        other_dataloaders = []
        for datasets in other_datasets:
            other_dataloaders += [torch.utils.data.DataLoader(datasets, batch_size=self.val_batch_size,
                                                              num_workers=self.num_workers,
                                                              drop_last=False,  shuffle=False, pin_memory=True,
                                                              collate_fn=combined_collate)]
        return [dataloader] + other_dataloaders

    def test_dataloader(self):
        datasets = []
        for key, dm in self.dms.items():
            if isinstance(dm.validation_set, list):
                datasets += dm.test_set
            else:
                datasets += [dm.test_set]
        dataloaders = []
        for dataset in datasets:
            dataloaders += [torch.utils.data.DataLoader(concat_dataset,
                                                        batch_size=self.test_batch_size,
                                                        num_workers=self.num_workers,
                                                        drop_last=False,  shuffle=False, pin_memory=True,
                                                        collate_fn=combined_collate)]
        return dataloaders


if __name__ == '__main__':
    from gdl.datasets.AffectNetDataModule import AffectNetEmoNetSplitModuleValTest
    from gdl.datasets.DecaDataModule import DecaDataModule
    from omegaconf import OmegaConf, DictConfig
    import yaml
    from pathlib import Path

    cfg = {"data": {
        "path": "/ps/scratch/face2d3d/",
        "n_train": 10000000,
        "sampler": False,
        # "scale_max": 2.8,
        # "scale_min": 2,
        "scale_max": 1.6,
        "scale_min": 1.2,
        "data_class": "DecaDataModule",
        "num_workers": 0,
        "split_ratio": 0.9,
        "split_style": "random",
        # "trans_scale": 0.2,
        "trans_scale": 0.1,
        "testing_datasets": [
            "now-test",
            "now-val",
            "celeb-val"
        ],
        "training_datasets": [
            "vggface2hq",
            "vox2"
        ],
        "validation_datasets": [
            "now-val",
            "celeb-val"
        ]
    },
    "learning": {
                "val_K": 1,
                "test_K": 1,
                "train_K": 4,
                "num_gpus": 1,
                "optimizer": "Adam",
                "logger_type": "WandbLogger",
                "val_K_policy": "sequential",
                "learning_rate": 0.0001,
                "test_K_policy": "sequential",
                "batch_size_val": 32,
                "early_stopping": {
                    "patience": 15
                },
                "train_K_policy": "random",
                "batch_size_test": 1,
                "batch_size_train": 32,
                "gpu_memory_min_gb": 30,
                "checkpoint_after_training": "best"
            },
        "model": {
            "image_size": 224
        }
    }
    cfg = DictConfig(cfg)
    deca_dm = DecaDataModule(cfg)

    augmenter = yaml.load(open(Path(__file__).parents[
                                   2] / "gdl_apps" / "EmotionRecognition" / "emodeca_conf" / "data" / "augmentations" / "default_with_resize.yaml"))[
        "augmentation"]
    affn_dm = AffectNetEmoNetSplitModuleValTest(
    # dm = AffectNetDataModule(
             # "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
             "/ps/project_cifs/EmotionalFacialAnimation/data/affectnet/",
             # "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/",
             # "/home/rdanecek/Workspace/mount/work/rdanecek/data/affectnet/",
             "/is/cluster/work/rdanecek/data/affectnet/",
             # processed_subfolder="processed_2021_Aug_27_19-58-02",
             processed_subfolder="processed_2021_Apr_05_15-22-18",
             processed_ext=".png",
             mode="manual",
             scale=1.7,
             image_size=224,
             bb_center_shift_x=0,
             bb_center_shift_y=-0.3,
             ignore_invalid=True,
             # ignore_invalid="like_emonet",
             # ring_type="gt_expression",
             # ring_type="gt_va",
             ring_type="augment",
             ring_size=4,
            augmentation=augmenter,
            num_workers=0,
            use_gt=False,
            # use_clean_labels=True
            # dataset_type="AffectNetWithMGCNetPredictions",
            # dataset_type="AffectNetWithExpNetPredictions",
             sampler="balanced_expr"
            )

    dm = CombinedDataModule({"AffectNet" : affn_dm,
                             "Deca" : deca_dm
                             })
    dm.prepare_data()
    dm.setup()

    from tqdm import auto
    dl = dm.train_dataloader()
    for i, data in enumerate(auto.tqdm(dl)):
        print(i)
