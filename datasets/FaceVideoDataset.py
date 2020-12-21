from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

import glob, os, sys
from pathlib import Path
import pyvista as pv
# from utils.mesh import load_mesh
# from scipy.io import wavfile
# import resampy
import numpy as np
import torch
import torchaudio
from enum import Enum
from typing import Optional, Union, List, Any
import pickle as pkl
from collections import OrderedDict
from tqdm import tqdm
import subprocess


class FaceVideoDataModule(pl.LightningDataModule):

    def __init__(self, root_dir, output_dir, processed_subfolder=None):
        super().__init__()
        self.root_dir = root_dir
        self.output_dir = output_dir

        if processed_subfolder is None:
            import datetime
            date = datetime.datetime.now()
            processed_folder = os.path.join(output_dir, "processed_%s" % date.strftime("%Y_%b_%d_%H-%M-%S"))
        else:
            processed_folder = os.path.join(output_dir, processed_subfolder)
        self.output_dir = processed_folder
        self.version = 0

    @property
    def metadata_path(self):
        return os.path.join(self.output_dir, "metadata.pkl")

    def prepare_data(self, *args, **kwargs):
        outdir = Path(self.output_dir)

        # is dataset already processed?
        if outdir.is_dir():
            print("The dataset is already processed. Loading")
            self._loadMeta()
        else:
            self._gather_data()
            self._saveMeta()
        self._unpack_videos()


    def _unpack_videos(self):
        for vi, video_file in enumerate(tqdm(self.video_list)):
            self._unpack_video(vi)

    def _unpack_video(self, video_idx):
        video_file = self.video_list[video_idx]
        suffix = Path(video_file.parts[-4]) / video_file.parts[-3] / video_file.parts[-2] / video_file.stem
        out_folder = Path(self.output_dir) / suffix

        out_folder.mkdir(exist_ok=True, parents=True)
        # ffmpeg -r 1 -i file.mp4 -r 1 "$%06d.png"

        out_format = out_folder / "%06d.png"
        out_format = '-r 1 -i %s -r 1 ' % str(video_file) + ' "' + str(out_format) + '"'
        # out_format = ' -r 1 -i %s ' % str(video_file) + ' "' + "$frame.%03d.png" + '"'
        # subprocess.call(['ffmpeg', out_format])
        os.system("ffmpeg " + out_format)

        # import ffmpeg
        # stream = ffmpeg.input(str(video_file))
        # # stream = ffmpeg.output(stream.video, str(out_format))
        # stream = ffmpeg.output(stream.video, "%06.png")
        # stream.run()

        n_frames = len(list(out_folder.glob("*.png")))
        if n_frames == self.video_metas[video_idx]['num_frames']:
            print("Successfully unpacked the video into %d frames" % self.video_metas[video_idx]['num_frames'])
        else:
            print("Expected %d frames but got %d" % (self.video_metas[video_idx]['num_frames'], n_frames))

    def _gather_data(self, exist_ok=False):
        print("Processing dataset")
        Path(self.output_dir).mkdir(parents=True, exist_ok=exist_ok)

        self.video_list = sorted(Path(self.root_dir).rglob("*.mp4"))
        self.annotation_list = sorted(Path(self.root_dir).rglob("*.txt"))

        import ffmpeg

        self.video_metas = []
        for vi, vid_file in enumerate(tqdm(self.video_list)):
            vid = ffmpeg.probe(str(vid_file))
            codec_idx = [idx for idx in range(len(vid)) if vid['streams'][idx]['codec_type'] == 'video']
            if len(codec_idx) > 1:
                raise RuntimeError("Video file has two video streams! '%s'" % str(vid_file))
            codec_idx = codec_idx[0]
            vid_info = vid['streams'][codec_idx]
            vid_meta = {}
            vid_meta['fps'] = vid_info['avg_frame_rate']
            vid_meta['width'] = vid_info['width']
            vid_meta['height'] = vid_info['height']
            vid_meta['num_frames'] = vid_info['nb_frames']

            self.video_metas += [vid_meta]


        print("Found %d video files." % len(self.video_list))


    def _loadMeta(self):
        with open(self.metadata_path, "rb") as f:
            version = pkl.load(f)
            self.video_list = pkl.load(f)
            self.video_metas = pkl.load(f)
            self.annotation_list = pkl.load(f)

    def _saveMeta(self):
        with open(self.metadata_path, "wb") as f:
            pkl.dump(self.version, f)
            pkl.dump(self.video_list,f)
            pkl.dump(self.video_metas, f)
            pkl.dump(self.annotation_list,f)


    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        pass

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass


def main():
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    subfolder = 'processed_2020_Dec_21_00-30-03'
    dm = FaceVideoDataModule(str(root_path), str(output_path), processed_subfolder=subfolder)
    dm.prepare_data()
    print("Peace out")



if __name__ == "__main__":
    main()
