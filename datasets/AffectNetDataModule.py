import os, sys
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd
from skimage.io import imread, imsave
from datasets.IO import load_segmentation, process_segmentation
from utils.image import numpy_image_to_torch
from transforms.keypoints import KeypointNormalization
import imgaug
from datasets.FaceVideoDataset import bbpoint_warp, bbox2point, FaceDataModuleBase
from datasets.EmotionalImageDataset import EmotionalImageDatasetBase
from utils.FaceDetector import save_landmark
from tqdm import auto


class AffectNetDataModule(FaceDataModuleBase):

    def __init__(self,
                 input_dir,
                 output_dir,
                 processed_subfolder = None,
                 mode="manual",
                 face_detector='fan',
                 face_detector_threshold=0.9,
                 image_size=224,
                 scale=1.25,
                 device=None):
        super().__init__(input_dir, output_dir, processed_subfolder,
                         face_detector=face_detector,
                         face_detector_threshold=face_detector_threshold,
                         image_size=image_size,
                         scale=scale,
                         device=device)
        # accepted_modes = ['manual', 'automatic', 'all'] # TODO: add support for the other images
        accepted_modes = ['manual']
        if mode not in accepted_modes:
            raise ValueError(f"Invalid mode '{mode}'. Accepted modes: {'_'.join(accepted_modes)}")
        self.mode = mode
        # self.subsets = sorted([f.name for f in (Path(input_dir) / "Manually_Annotated" / "Manually_Annotated_Images").glob("*") if f.is_dir()])
        self.input_dir = Path(self.root_dir) / "Manually_Annotated" / "Manually_Annotated_Images"
        train = pd.read_csv(self.input_dir.parent / "training.csv")
        val = pd.read_csv(self.input_dir.parent / "validation.csv")
        self.df = pd.concat([train, val], ignore_index=True, sort=False)
        self.face_detector_type = 'fan'
        self.scale = scale

    @property
    def subset_size(self):
        return 1000

    @property
    def num_subsets(self):
        num_subsets = len(self.df) // self.subset_size
        if len(self.df) % self.subset_size != 0:
            num_subsets += 1
        return num_subsets

    def _detect_faces(self):
        subset_size = 1000
        num_subsets = len(self.df) // subset_size
        if len(self.df) % subset_size != 0:
            num_subsets += 1
        for sid in range(self.num_subsets):
            self._detect_landmarks_and_segment_subset(self.subset_size * sid, min((sid + 1) * self.subset_size, len(self.df)))

    def _path_to_detections(self):
        return Path(self.output_dir) / "detections"

    def _path_to_segmentations(self):
        return Path(self.output_dir) / "segmentations"

    def _path_to_landmarks(self):
        return Path(self.output_dir) / "landmarks"

    def _detect_landmarks_and_segment_subset(self, start_i, end_i):
        self._path_to_detections().mkdir(parents=True, exist_ok=True)
        self._path_to_segmentations().mkdir(parents=True, exist_ok=True)
        self._path_to_landmarks().mkdir(parents=True, exist_ok=True)

        detection_fnames = []
        out_segmentation_folders = []

        for i in auto.tqdm(range(start_i, end_i)):
            im_file = self.df.loc[i]["subDirectory_filePath"]
            left = self.df.loc[i]["face_x"]
            top = self.df.loc[i]["face_y"]
            right = left + self.df.loc[i]["face_width"]
            bottom = top + self.df.loc[i]["face_height"]
            bb = np.array([top, left, bottom, right])

            im_fullfile = Path(self.input_dir) / im_file
            detection, _, _, bbox_type, landmarks = self._detect_faces_in_image(im_fullfile, detected_faces=[bb])

            out_detection_fname = self._path_to_detections() / Path(im_file).parent / (Path(im_file).stem + ".png")
            detection_fnames += [out_detection_fname.relative_to(self.output_dir)]
            out_detection_fname.parent.mkdir(exist_ok=True)
            imsave(out_detection_fname, detection[0])

            out_segmentation_folders += [self._path_to_landmarks() / Path(im_file).parent]

            # save landmarks
            out_landmark_fname = self._path_to_landmarks() / Path(im_file).parent / (Path(im_file).stem + ".pkl")
            out_landmark_fname.parent.mkdir(exist_ok=True)
            landmark_fnames += [out_landmark_fname.relative_to(self.output_dir)]
            save_landmark(out_landmark_fname, landmarks[0], bbox_type)

            detection_fnames += [out_detection_fname]

        self._segment_images(detection_fnames, out_segmentation_folders)

    def prepare_data(self):
        status_array_path = self.output_dir / "status.memmap"
        if not status_array_path.isfile():
            status_array = np.memmap(status_array_path,
                                     dtype=np.bool,
                                     mode='w+',
                                     shape=(dm.num_subsets,)
                                     )
            del status_array

        status_array = np.memmap(status_array_path,
                                 dtype=np.bool,
                                 mode='r+',
                                 shape=(dm.num_subsets,)
                                 )
        all_processed = status_array.all()
        del status_array
        if not all_processed:
            self._detect_faces()


    def setup(self, stage):
        self.train_dataframe_path = self.root_dir / "Manually_Annotated" / "training.csv"
        self.val_dataframe_path = self.root_dir / "Manually_Annotated" / "validation.csv"
        # if self.mode in ['all', 'manual']:
        #     # self.image_list += sorted(list((Path(self.path) / "Manually_Annotated").rglob(".jpg")))
        #     self.dataframe = pd.load_csv(self.path / "Manually_Annotated" / "Manually_Annotated.csv")
        # if self.mode in ['all', 'automatic']:
        #     # self.image_list += sorted(list((Path(self.path) / "Automatically_Annotated").rglob("*.jpg")))
        #     self.dataframe = pd.load_csv(
        #         self.path / "Automatically_Annotated" / "Automatically_annotated_file_list.csv")

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass



class AffectNetOriginal(EmotionalImageDatasetBase):

    def __init__(self,
                 image_path,
                 dataframe_path,
                 image_size,
                 scale = 1.4,
                 transforms : imgaug.augmenters.Augmenter = None,
                 use_gt_bb=True,):
        self.dataframe_path = dataframe_path
        self.image_path = image_path
        self.df = pd.read_csv(dataframe_path)
        self.image_size = image_size
        self.transforms = transforms
        self.use_gt_bb = use_gt_bb
        self.transforms = transforms or imgaug.augmenters.Identity()
        self.scale = scale
        self.landmark_normalizer = KeypointNormalization()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        im_file = self.df.loc[index]["subDirectory_filePath"]
        left = self.df.loc[index]["face_x"]
        top = self.df.loc[index]["face_y"]
        right = left + self.df.loc[index]["face_width"]
        bottom = top + self.df.loc[index]["face_height"]
        facial_landmarks = self.df.loc[index]["facial_landmarks"]
        expression = self.df.loc[index]["expression"]
        valence = self.df.loc[index]["valence"]
        arousal = self.df.loc[index]["arousal"]

        im_file = Path(self.image_path) / im_file

        input_img = imread(im_file)
        input_img_shape = input_img.shape

        old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
        size = int(old_size * self.scale)

        input_landmarks = np.array([float(f) for f in facial_landmarks.split(";")]).reshape(-1,2)
        img, landmark = bbpoint_warp(input_img, center, size, self.image_size, landmarks=input_landmarks)
        img *= 255.
        if not self.use_gt_bb:
            raise NotImplementedError()
            # landmark_type, landmark = load_landmark(
            #     self.path_prefix / self.landmark_list[index])
        landmark = landmark[np.newaxis, ...]

        # seg_image, seg_type = load_segmentation(
        #     self.path_prefix / self.segmentation_list[index])
        # seg_image = seg_image[np.newaxis, :, :, np.newaxis]
        #
        # seg_image = process_segmentation(
        #     seg_image, seg_type).astype(np.uint8)
        seg_image=None

        img, seg_image, landmark = self._augment(img, seg_image, landmark)

        sample = {
            "image": numpy_image_to_torch(img),
            "path": str(im_file),
            "expr7": torch.tensor([expression, ]),
            "va": torch.tensor([valence, arousal]),
            "label": str(im_file.stem),
        }

        if landmark is not None:
            sample["landmark"] = torch.from_numpy(landmark)
        if seg_image is not None:
            sample["mask"] = numpy_image_to_torch(seg_image)
        print(self.df.loc[index])
        return sample


if __name__ == "__main__":
    # d = AffectNetOriginal(
    #     "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/Manually_Annotated/Manually_Annotated_Images",
    #     "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/Manually_Annotated/validation.csv",
    #     224
    # )
    # print(f"Num sample {len(d)}")
    # for i in range(100):
    #     sample = d[i]
    #     d.visualize_sample(sample)

    dm = AffectNetDataModule(
             "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet//",
             "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/",
             processed_subfolder="processed_2021_Apr_02_03-13-33",
             mode="manual",
             scale=1.25)
    print(dm.num_subsets)
    dm._detect_faces()

