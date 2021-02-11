import os, sys
from pathlib import Path
sys.path += [str(Path(__file__).parent.parent)]

import numpy as np
from datasets.FaceVideoDataset import FaceVideoDataModule, \
    AffectNetExpressions, Expression7, AU8, expr7_to_affect_net
from datasets.EmotionalDataModule import EmotionDataModule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
import datetime
# import hydra
import yaml
import torch
import torch.nn.functional as F
from typing import Dict, Any
from torch.utils.data.dataloader import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle as pkl
from transforms.keypoints import KeypointScale, KeypointNormalization


def finetune_deca(data_params, learning_params, model_params, inout_params):

    from test_and_finetune_deca import DecaModule
    deca = DecaModule(model_params, learning_params, inout_params)
    deca.cuda()
    deca._check_device_for_extra_params()
    batch_idx = 1
    with open(Path(__file__).parent / "dummy_train.pkl", "rb") as f:
        training_set = pkl.load(f)
    with open(Path(__file__).parent / "dummy_val.pkl", "rb") as f:
        val_set = pkl.load(f)

    training_set_dl = DataLoader(training_set, shuffle=False,
                                 # batch_size=model_params.batch_size_train,
                                 batch_size=1,
                                 num_workers=data_params.num_workers)
    val_set_dl = DataLoader(val_set, shuffle=False,
                                 batch_size=model_params.batch_size_train,
                                 num_workers=data_params.num_workers)

    from tqdm import tqdm
    # for i, b in enumerate(tqdm(training_set_dl)):
    # for i, b in enumerate(tqdm(val_set_dl)):
    #     print(f"batch {i}")
    #     print(f" image batch \t\t {b['image'].shape}")
    #     print(f" mask batch \t\t {b['mask'].shape}")
    #     print(f" landmark batch \t {b['landmark'].shape}")
    #     # if b['image'].shape[0] != 2 or b['mask'].shape[0] != 2 or b['landmark'].shape[0] != 2:
    #     #     print("ha!")
    #     b['image'] = b['image'][...].cuda()
    #     b['mask'] = b['mask'][...].cuda()
    #     b['landmark'] = b['landmark'][...].cuda()
    #     deca.training_step(b, i)
    #     deca.validation_step(b, i)
    #     # b["mask"]
    #     # b["landmark"]

    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.fit(deca, train_dataloader=training_set_dl, val_dataloaders=[val_set_dl, ])
    # deca.training_step(batch, batch_idx, False)


@hydra.main(config_path="conf", config_name="deca_finetune")
def main(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    # root_path = root / "Aff-Wild2_ready"
    root_path = root
    processed_data_path = root / "processed"
    # subfolder = 'processed_2020_Dec_21_00-30-03'
    subfolder = 'processed_2021_Jan_19_20-25-10'

    run_dir = cfg.inout.output_dir + "_" + datetime.datetime.now().strftime("%Y_%b_%d_%H-%M-%S")

    full_run_dir = Path(cfg.inout.output_dir) / run_dir
    full_run_dir.mkdir(parents=True)

    # cfg["inout"]['run_dir'] = str(full_run_dir)

    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    finetune_deca(cfg['data'], cfg['learning'], cfg['model'], cfg['inout'])


if __name__ == "__main__":
    main()
