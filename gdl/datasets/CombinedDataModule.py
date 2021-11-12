from pytorch_lightning import LightningDataModule
# from .AffectNetDataModule import AffectNetDataModule, AffectNetEmoNetSplitModuleValTest, AffectNetEmoNetSplitModule
# from .DecaDataModule import DecaDataModule
import torch
from torch.utils.data._utils.collate import default_collate


class CombinedDataModule(LightningDataModule):

    def __init__(self, dms):
        super().__init__()
        assert len(dms) > 0
        self.dms = dms
        # for key, value in config.items():
        #     if data_preparation_function is None:
        #         dm_class = class_from_str(key, sys.modules[__name__])
        #         self.dms[key] = dm_class(**value)
        #     else:
        #         self.dms[key] = data_preparation_function(value)

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

        concat_dataset = torch.utils.data.ConcatDataset(datasets)
        # concat_dataset = torch.utils.data.ChainDataset(datasets)
        dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=self.train_batch_size,
                                                 num_workers=self.num_workers,
                                                 drop_last=True,  shuffle=True)
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
        concat_dataset = torch.utils.data.ConcatDataset(first_datasets)
        dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=self.val_batch_size,
                                                 num_workers=self.num_workers,
                                                 drop_last=False,  shuffle=False)

        other_dataloaders = []
        for datasets in other_datasets:
            other_dataloaders += [torch.utils.data.DataLoader(datasets, batch_size=self.val_batch_size,
                                                              num_workers=self.num_workers,
                                                              drop_last=False,  shuffle=False)]
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
                                                        drop_last=False,  shuffle=False)]
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
                                   2] / "gdl_apps" / "EmoDECA" / "emodeca_conf" / "data" / "augmentations" / "default_with_resize.yaml"))[
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
