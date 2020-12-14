from models.VOCA import Voca
from applications.FLAME.fit import load_FLAME
from datasets.MeshDataset import EmoSpeechDataModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ProgressBar
# from pytorch_lightning.callback
import torch
import numpy as np


def extract_expression_space(flame, num_dimensions):
    start_idx = flame.flame_model.shapedirs.shape[2] - 100
    end_idx = start_idx + num_dimensions
    return torch.from_numpy(np.array(flame.flame_model.shapedirs[:,:, start_idx:end_idx])).clone()


def main():
    root_dir = "/home/rdanecek/Workspace/mount/project/emotionalspeech/EmotionalSpeech/"
    processed_dir = "/home/rdanecek/Workspace/mount/scratch/rdanecek/EmotionalSpeech/"
    subfolder = "processed_2020_Dec_09_00-30-18"
    # batch_size = 64
    batch_size = 2

    dm = EmoSpeechDataModule(root_dir, processed_dir, subfolder, consecutive_frames=2, batch_size=batch_size)
    dm.prepare_data()
    dm.setup()

    # samples = dm.dataset_train[0]

    expression_parameters = 100

    flame = load_FLAME('neutral')
    epression_params = extract_expression_space(flame, expression_parameters)

    trainer = Trainer(min_epochs=10, max_steps=10, gpus=[0,])
    model = Voca(dm, epression_params)
    trainer.fit(model, datamodule=dm)

    print("Yohoho")


if __name__ == "__main__":
    main()


