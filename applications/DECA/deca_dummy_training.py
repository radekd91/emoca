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
from models.DECA import DecaModule


def finetune_deca(cfg_coarse, cfg_detail):
    deca = DecaModule(cfg_coarse.model, cfg_coarse.learning, cfg_coarse.inout)
    # deca.cuda()
    # deca._move_extra_params_to_correct_device()
    batch_idx = 1
    with open(Path(__file__).parent / "dummy_train.pkl", "rb") as f:
        training_set = pkl.load(f)
    with open(Path(__file__).parent / "dummy_val.pkl", "rb") as f:
        val_set = pkl.load(f)

    # training_set.image_list = training_set.image_list[:50]
    # val_set.image_list = val_set.image_list[:50]

    conf = DictConfig({})
    conf.coarse = cfg_coarse
    conf.detail = cfg_detail

    wandb_logger = WandbLogger(name="test_" + datetime.datetime.now().strftime("%b_%d_%Y_%H-%M-%S"),
                               project="EmotionalDeca",
                               config=dict(conf))

    configs = [cfg_coarse, cfg_detail]
    for i, cfg in enumerate(configs):
        if i > 0:
            deca.reconfigure(cfg.model)

        training_set_dl = DataLoader(training_set, shuffle=True,
                                     batch_size=cfg.model.batch_size_train,
                                     num_workers=cfg.data.num_workers)
        val_set_dl = DataLoader(val_set, shuffle=False,
                                     batch_size=cfg.model.batch_size_train,
                                     num_workers=cfg.data.num_workers)

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

        trainer = Trainer(gpus=cfg.learning.num_gpus, max_epochs=cfg.learning.max_epochs,
                          default_root_dir=cfg.inout.full_run_dir, logger=wandb_logger)
        trainer.fit(deca, train_dataloader=training_set_dl, val_dataloaders=[val_set_dl, ])
        trainer.test(deca, test_dataloaders=[val_set_dl], ckpt_path='best')

    # deca.training_step(batch, batch_idx, False)

from hydra.experimental import compose, initialize


# @hydra.main(config_path="deca_conf", config_name="deca_finetune_coarse")
# @hydra.main(config_path="deca_conf", config_name="deca_finetune_all")
# def main(cfg : DictConfig):
def main():

    initialize(config_path="deca_conf", job_name="finetune_deca")
    cfg_coarse = compose(config_name="deca_finetune_coarse")
    cfg_detail = compose(config_name="deca_finetune_detail")



    # print(OmegaConf.to_yaml(cfg))
    # root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    # root_path = root / "Aff-Wild2_ready"
    # root_path = root
    # processed_data_path = root / "processed"
    subfolder = 'processed_2021_Jan_19_20-25-10'

    # run_dir = cfg.inout.output_dir + "_" + datetime.datetime.now().strftime("%Y_%b_%d_%H-%M-%S")
    #
    # full_run_dir = Path(cfg.inout.output_dir) / run_dir
    # full_run_dir.mkdir(parents=True)

    # cfg["inout"]['full_run_dir'] = str(full_run_dir)

    # with open(full_run_dir / "cfg.yaml", 'w') as outfile:
    #     OmegaConf.save(config=cfg, f=outfile)

    # finetune_deca(cfg['data'], cfg['learning'], cfg['model'], cfg['inout'])
    finetune_deca(cfg_coarse, cfg_detail)


if __name__ == "__main__":
    main()

