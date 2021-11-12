from pytorch_lightning import LightningDataModule
from .AffectNetDataModule import AffectNetDataModule, AffectNetEmoNetSplitModuleValTest, AffectNetEmoNetSplitModule
from .DecaDataModule import DecaDataModule
import torch


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