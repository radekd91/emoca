import os, sys
from pathlib import Path
sys.path += [str(Path(__file__).parent.parent)]

import numpy as np
from gdl.datasets.FaceVideoDataset import FaceVideoDataModule, \
    Expression7, AU8, expr7_to_affect_net
from gdl.datasets.AffectNetDataModule import AffectNetExpressions
from gdl.datasets.EmotionalDataModule import EmotionDataModule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
import datetime
# import hydra
import yaml
import torch
import torch.nn.functional as F
from typing import Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import pickle as pkl
from gdl.transforms.keypoints import KeypointScale, KeypointNormalization

def finetune_deca(data_params, learning_params, model_params, inout_params):

    from test_and_finetune_deca import DecaModule
    deca = DecaModule(model_params, learning_params, inout_params)
    deca.cuda()
    deca._move_extra_params_to_correct_device()
    batch_idx = 1
    with open(Path(__file__).parent / f"batch_{batch_idx}.pkl", "rb") as f:
        batch_idx = pkl.load(f)
        batch = pkl.load(f)

    lmk_transforms = KeypointNormalization(batch["image"].shape[2],
                                           batch["image"].shape[3])
    lmk = batch["landmark"]
    batch["landmark"] = lmk_transforms(lmk)

    deca.training_step(batch, batch_idx, False)
    print("Training step finished")



@hydra.main(config_path="deca_conf", config_name="deca_finetune")
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
