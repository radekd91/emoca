import os, sys
from pathlib import Path
sys.path += [str(Path(__file__).parent.parent)]

import numpy as np
from datasets.FaceVideoDataset import FaceVideoDataModule, \
    AffectNetExpressions, Expression7, AU8, expr7_to_affect_net
from datasets.EmotionalDataModule import EmotionDataModule
from pytorch_lightning import Trainer

from models.DECA import DecaModule
from pytorch_lightning.loggers import WandbLogger
import wandb
import datetime
# import hydra


def finetune_deca(data_params, learning_params, model_params, inout_params):

    fvdm = FaceVideoDataModule(Path(data_params.data_root), Path(data_params.data_root) / "processed",
                               data_params.processed_subfolder)
    dm = EmotionDataModule(fvdm, image_size=model_params.image_size,
                           with_landmarks=True, with_segmentations=True)
    dm.prepare_data()


    # index = 220
    # index = 120
    index = data_params.sequence_index
    annotation_list = data_params.annotation_list
    if index == -1:
        sequence_name = annotation_list[0]
        if annotation_list[0] == 'va':
            filter_pattern = 'VA_Set'
        elif annotation_list[0] == 'expr7':
            filter_pattern = 'Expression_Set'
    else:
        sequence_name = str(fvdm.video_list[index])
        filter_pattern = sequence_name
        if annotation_list[0] == 'va' and 'VA_Set' not in sequence_name:
            print("No GT for valence and arousal. Skipping")
            # sys.exit(0)
        if annotation_list[0] == 'expr7' and 'Expression_Set' not in sequence_name:
            print("No GT for expressions. Skipping")
            # sys.exit(0)

    deca = DecaModule(model_params, learning_params, inout_params)

    project_name = 'EMOCA_finetune'
    name = inout_params.name + '_' + str(filter_pattern) + "_" + \
           datetime.datetime.now().strftime("%b_%d_%Y_%H-%M-%S")

    train_data_loader = dm.train_dataloader(annotation_list, filter_pattern,
                                    # TODO: find a better! way to incorporate the K and the batch size
                                    batch_size=model_params.batch_size_train,
                                    num_workers=data_params.num_workers,
                                    split_ratio=data_params.split_ratio,
                                    split_style=data_params.split_style,
                                    K=model_params.train_K,
                                    K_policy=model_params.K_policy)

    val_data_loader = dm.val_dataloader(annotation_list, filter_pattern,
                                        # TODO: find a better! way to incorporate the K and the batch size
                                        batch_size=model_params.batch_size_val,
                                        num_workers=data_params.num_workers)

    # out_folder = Path(inout_params.output_dir) / name
    # out_folder.mkdir(parents=True)

    # wandb.init(project_name)
    # wandb_logger = WandbLogger(name=name, project=project_name)
    wandb_logger = None
    trainer = Trainer(gpus=1)
    # trainer = Trainer(gpus=1, logger=wandb_logger)
    # trainer.fit(deca, datamodule=dm)
    print("The training begins")
    trainer.fit(deca, train_dataloader=train_data_loader, val_dataloaders=[val_data_loader,])




import hydra
from omegaconf import DictConfig, OmegaConf


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

    cfg["inout"]['full_run_dir'] = str(full_run_dir)

    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    finetune_deca(cfg['data'], cfg['learning'], cfg['model'], cfg['inout'])


if __name__ == "__main__":
    main()
