from pathlib import Path
from gdl.datasets.FaceVideoDataModule import FaceVideoDataModule
from gdl.datasets.ImageDatasetHelpers import point2bbox, bbpoint_warp
import sys
import argparse
import os
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from skimage.io import imread, imsave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..' ,'..', '..', 'EMOCA')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import util
from tqdm import tqdm
import cv2






def main():
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    subfolder = 'processed_2020_Dec_21_00-30-03'
    dm = FaceVideoDataModule(str(root_path), str(output_path), processed_subfolder=subfolder)
    dm.prepare_data()
    # dm.setup()

    # gather_detections(dm)

    i = 0
    dm.create_reconstruction_video(i)
    # i = 1
    # i = 2
    # i = 3
    # seq_star = i*10
    # seq_end = (i+1)*10
    # for seq in range(seq_star, seq_end):
    #     dm.create_detection_video(seq)
    # input_video = "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/processed/processed_2020_Dec_21_00-30-03/AU_Set/reconstructions/Test_Set/88-30-360x480/video.mp4"
    # input_video_with_audio = ""
    # attach_audio_to_reconstruction_video(input_video, dm.root_dir / dm.video_list[seq])



if __name__ == "__main__":
    main()

