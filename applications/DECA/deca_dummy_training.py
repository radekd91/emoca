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

    time = datetime.datetime.now().strftime("%b_%d_%Y_%H-%M-%S")
    experiment_name = "test_" + time

    full_run_dir = Path(cfg_coarse.inout.output_dir) / experiment_name
    full_run_dir.mkdir(parents=True)

    coarse_checkpoint_dir = full_run_dir / "coarse"
    coarse_checkpoint_dir.mkdir(parents=True)

    detail_checkpoint_dir = full_run_dir / "detail"
    detail_checkpoint_dir.mkdir(parents=True)

    cfg_coarse.inout.full_run_dir = str(full_run_dir / "coarse")
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_detail.inout.full_run_dir = str(full_run_dir / "detail")
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)

    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    wandb_logger = WandbLogger(name=experiment_name,
                               project="EmotionalDeca",
                               config=dict(conf),
                               version=time,
                               save_dir=full_run_dir)

    training_set_dl = DataLoader(training_set, shuffle=True,
                                 # batch_size=cfg_coarse.model.batch_size_train,
                                 batch_size=4,
                                 num_workers=cfg_coarse.data.num_workers)

    val_set_dl = DataLoader(val_set, shuffle=False,
                            batch_size=cfg_coarse.model.batch_size_train,
                            num_workers=cfg_coarse.data.num_workers)
    configs = [cfg_coarse, cfg_detail]
    # configs = [cfg_detail]
    for i, cfg in enumerate(configs):
        if i > 0:
            deca.reconfigure(cfg.model)

        from tqdm import tqdm
        for i, b in enumerate(tqdm(training_set_dl)):
        # for i, b in enumerate(tqdm(val_set_dl)):
            print(f"batch {i}")
            print(f" image batch \t\t {b['image'].shape}")
            print(f" mask batch \t\t {b['mask'].shape}")
            print(f" landmark batch \t {b['landmark'].shape}")
            # if b['image'].shape[0] != 2 or b['mask'].shape[0] != 2 or b['landmark'].shape[0] != 2:
            #     print("ha!")
            # b['image'] = b['image'][...].cuda()
            # b['mask'] = b['mask'][...].cuda()
            # b['landmark'] = b['landmark'][...].cuda()
            # deca.training_step(b, i)
            # deca.validation_step(b, i)
            # b["mask"]
            # b["landmark"]

        accelerator = None if cfg.learning.num_gpus == 1 else 'ddp'
        # if accelerator is not None:
        #     os.environ['LOCAL_RANK'] = '0'
        trainer = Trainer(gpus=cfg.learning.num_gpus, max_epochs=cfg.learning.max_epochs,
                          default_root_dir=cfg.inout.checkpoint_dir,
                          logger=wandb_logger,
                          accelerator=accelerator)

        trainer.fit(deca, train_dataloader=training_set_dl, val_dataloaders=[val_set_dl, ])

        test_set_dl = DataLoader(val_set, shuffle=False,
                                batch_size=cfg.model.batch_size_train,
                                num_workers=cfg.data.num_workers)
        wandb_logger.finalize("")
        trainer.test(deca, test_dataloaders=[test_set_dl], ckpt_path=None)
        # to make sure WANDB has the correct step
        wandb_logger.finalize("")


# @hydra.main(config_path="deca_conf", config_name="deca_finetune_coarse")
# @hydra.main(config_path="deca_conf", config_name="deca_finetune_all")
# def main(cfg : DictConfig):
def main():
    from hydra.experimental import compose, initialize
    # override = ['learning.num_gpus=2', 'model/paths=cluster']
    override = []
    initialize(config_path="deca_conf", job_name="finetune_deca")
    cfg_coarse = compose(config_name="deca_finetune_coarse", overrides=override)
    cfg_detail = compose(config_name="deca_finetune_detail", overrides=override)

    finetune_deca(cfg_coarse, cfg_detail)


if __name__ == "__main__":
    main()

