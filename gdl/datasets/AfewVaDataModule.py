"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""


import json
import os, sys
from enum import Enum
from pathlib import Path
import numpy as np
import scipy as sp
import torch
import pytorch_lightning as pl
import pandas as pd
import pickle as pkl
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
from gdl.datasets.IO import load_segmentation, process_segmentation, load_emotion, save_emotion
from gdl.utils.image import numpy_image_to_torch
from gdl.transforms.keypoints import KeypointNormalization
import imgaug
from gdl.datasets.FaceDataModuleBase import FaceDataModuleBase
from gdl.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp
from gdl.datasets.EmotionalImageDataset import EmotionalImageDatasetBase
from gdl.datasets.UnsupervisedImageDataset import UnsupervisedImageDataset
from gdl.utils.FaceDetector import save_landmark, load_landmark
from tqdm import auto
import traceback
from torch.utils.data.dataloader import DataLoader
from gdl.transforms.imgaug import create_image_augmenter
from torchvision.transforms import Resize, Compose
from sklearn.neighbors import NearestNeighbors
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import WeightedRandomSampler
from collections import OrderedDict
from munch import Munch
import json
import bisect
from gdl.utils.other import class_from_str
import warnings
warnings.filterwarnings('ignore')


# def make_class_balanced_sampler(labels):
#     class_counts = np.bincount(labels)
#     class_weights = 1. / class_counts
#     weights = class_weights[labels]
#     return WeightedRandomSampler(weights, len(weights))
#
# def make_va_balanced_sampler(labels):
#     class_counts = np.bincount(labels)
#     class_weights = 1. / class_counts
#     weights = class_weights[labels]
#     return WeightedRandomSampler(weights, len(weights))
#
# def make_balanced_sample_by_weights(weights):
#     return WeightedRandomSampler(weights, len(weights))



def new_affewva(class_name):
    dataset_class = class_from_str(class_name, sys.modules[__name__])
    return dataset_class


class AfewVaDataModule(FaceDataModuleBase):

    def __init__(self,
                 input_dir,
                 output_dir,
                 processed_subfolder = None,
                 face_detector='fan',
                 face_detector_threshold=0.9,
                 image_size=224,
                 scale=1.25,
                 bb_center_shift_x=0.,
                 bb_center_shift_y=0.,
                 processed_ext=".png",
                 device=None,
                 augmentation=None,
                 train_batch_size=64,
                 val_batch_size=64,
                 test_batch_size=64,
                 num_workers=0,
                 ring_type=None,
                 ring_size=None,
                 drop_last=False,
                 sampler=None,
                 split_seed=0,
                 train_fraction=0.6,
                 val_fraction=0.2,
                 test_fraction=0.2,
                 dataset_type=None,
                 ):
        super().__init__(input_dir, output_dir, processed_subfolder,
                         face_detector=face_detector,
                         face_detector_threshold=face_detector_threshold,
                         image_size=image_size,
                         bb_center_shift_x=bb_center_shift_x,
                         bb_center_shift_y=bb_center_shift_y,
                         scale=scale,
                         processed_ext=processed_ext,
                         device=device)
        self.dataset_type = dataset_type or "AfewVa"
        # # self.subsets = sorted([f.name for f in (Path(input_dir) / "Manually_Annotated" / "Manually_Annotated_Images").glob("*") if f.is_dir()])
        # self.input_dir = Path(self.root_dir) / "Manually_Annotated" / "Manually_Annotated_Images"
        # train = pd.read_csv(self.input_dir.parent / "training.csv")
        # val = pd.read_csv(self.input_dir.parent / "validation.csv")
        # self.df = pd.concat([train, val], ignore_index=True, sort=False)
        self.face_detector_type = 'fan'
        self.scale = scale
        self.use_processed = False

        if not (Path(self.output_dir) / "gt.pkl").exists():
            from omegaconf import OmegaConf, DictConfig
            video_list = sorted([p for p in Path(input_dir).glob("*") if p.is_dir()])

            video_gts = OrderedDict()
            for iv, vp in enumerate(auto.tqdm(video_list)):
                video_gts[vp.stem] = Munch( json.load(open(vp / (vp.stem + ".json"), "r")))

            with open(Path(self.output_dir) / "gt.pkl", "wb") as f:
                pkl.dump(video_gts, f)
        else:
            with open(Path(self.output_dir) / "gt.pkl", "rb") as f:
                video_gts = pkl.load(f)

        if self.use_processed:
            self.image_path = Path(self.output_dir) / "detections"
        else:
            self.image_path = Path(input_dir)

        self.seed = split_seed
        np.random.seed(self.seed)

        indices = np.arange(len(video_gts), dtype=np.int32) + 1
        np.random.shuffle(indices)

        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        assert self.train_fraction + self.val_fraction + self.test_fraction == 1.0

        train_end = int(len(indices) * self.train_fraction)
        val_end = int(len(indices) * ( self.train_fraction + self.val_fraction))

        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]

        # iterate over the training indices and create a list of the corresponding video names
        self.train_list = OrderedDict()
        self.val_list = OrderedDict()
        self.test_list = OrderedDict()
        for tr_i in self.train_indices:
            self.train_list[f"{tr_i:03d}"] = video_gts[f"{tr_i:03d}"]
        for v_i in self.val_indices:
            self.val_list[f"{v_i:03d}"] = video_gts[f"{v_i:03d}"]
        for t_i in self.test_indices:
            self.test_list[f"{t_i:03d}"] = video_gts[f"{t_i:03d}"]
        # self.ignore_invalid = ignore_invalid

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.sampler = sampler or "uniform"
        if self.sampler not in ["uniform", "balanced_videos", "balanced_expr", "balanced_va", "balanced_v", "balanced_a"]:
            raise ValueError(f"Invalid sampler type: '{self.sampler}'")
        if self.sampler in ["balanced_expr", "balanced_va", "balanced_v", "balanced_a"]:
            raise NotImplementedError()



        if ring_type not in [None, "gt_va", "augment"]:
            raise ValueError(f"Invalid ring type: '{ring_type}'")
        if ring_type == "gt_va":
            raise NotImplementedError()
        self.ring_type = ring_type
        self.ring_size = ring_size
        self.drop_last = drop_last

    @property
    def subset_size(self):
        return 1000

    # @property
    # def num_subsets(self):
    #     num_subsets = len(self.df) // self.subset_size
    #     if len(self.df) % self.subset_size != 0:
    #         num_subsets += 1
    #     return num_subsets

    def _detect_faces(self):
        subset_size = 1000
        num_subsets = len(self.df) // subset_size
        if len(self.df) % subset_size != 0:
            num_subsets += 1
        for sid in range(self.num_subsets):
            self._detect_landmarks_and_segment_subset(self.subset_size * sid, min((sid + 1) * self.subset_size, len(self.df)))

    def _extract_emotion_features(self):
        subset_size = 1000
        num_subsets = len(self.df) // subset_size
        if len(self.df) % subset_size != 0:
            num_subsets += 1
        for sid in range(self.num_subsets):
            self._extract_emotion_features_from_subset(self.subset_size * sid, min((sid + 1) * self.subset_size, len(self.df)))

    def _path_to_detections(self):
        return Path(self.output_dir) / "detections"

    def _path_to_segmentations(self):
        return Path(self.output_dir) / "segmentations"

    def _path_to_landmarks(self):
        return Path(self.output_dir) / "landmarks"

    def _path_to_emotions(self):
        return Path(self.output_dir) / "emotions"

    def _get_emotion_net(self, device):
        from gdl.layers.losses.EmonetLoader import get_emonet

        net = get_emonet()
        net = net.to(device)

        return net, "emo_net"

    def _extract_emotion_features_from_subset(self, start_i, end_i):
        self._path_to_emotions().mkdir(parents=True, exist_ok=True)

        print(f"Processing subset {start_i // self.subset_size}")
        image_file_list = []
        for i in auto.tqdm(range(start_i, end_i)):
            im_file = self.df.loc[i]["subDirectory_filePath"]
            in_detection_fname = self._path_to_detections() / Path(im_file).parent / (Path(im_file).stem + ".png")
            if in_detection_fname.is_file():
                image_file_list += [in_detection_fname]

        transforms = Compose([
            Resize((256, 256)),
        ])
        batch_size = 32
        dataset = UnsupervisedImageDataset(image_file_list, image_transforms=transforms, im_read='pil')
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        net, emotion_type = self._get_emotion_net(device)

        for i, batch in enumerate(auto.tqdm(loader)):
            # facenet_pytorch expects this stanadrization for the input to the net
            # images = fixed_image_standardization(batch['image'].to(device))
            images = batch['image'].cuda()
            # start = time.time()
            with torch.no_grad():
                out = net(images, intermediate_features=True)
            # end = time.time()
            # print(f" Inference batch {i} took : {end - start}")
            emotion_features = {key : val.detach().cpu().numpy() for key, val in out.items()}

            # start = time.time()
            for j in range(images.size()[0]):
                image_path = batch['path'][j]
                out_emotion_folder = self._path_to_emotions() / Path(image_path).parent.name
                out_emotion_folder.mkdir(exist_ok=True, parents=True)
                emotion_path = out_emotion_folder / (Path(image_path).stem + ".pkl")
                emotion_feature_j = {key: val[j] for key, val in emotion_features.items()}
                del emotion_feature_j['emo_feat'] # too large to be stored per frame = (768, 64, 64)
                del emotion_feature_j['heatmap'] # not too large but probably not usefull = (68, 64, 64)
                # we are keeping emo_feat_2 (output of last conv layer (before FC) and then the outputs of the FCs - expression, valence and arousal)
                save_emotion(emotion_path, emotion_feature_j, emotion_type)


    def _detect_landmarks_and_segment_subset(self, start_i, end_i):
        self._path_to_detections().mkdir(parents=True, exist_ok=True)
        self._path_to_segmentations().mkdir(parents=True, exist_ok=True)
        self._path_to_landmarks().mkdir(parents=True, exist_ok=True)

        detection_fnames = []
        out_segmentation_folders = []

        status_array = np.memmap(self.status_array_path,
                                 dtype=np.bool,
                                 mode='r',
                                 shape=(self.num_subsets,)
                                 )

        completed = status_array[start_i // self.subset_size]
        if not completed:
            print(f"Processing subset {start_i // self.subset_size}")
            for i in auto.tqdm(range(start_i, end_i)):
                im_file = self.df.loc[i]["subDirectory_filePath"]
                left = self.df.loc[i]["face_x"]
                top = self.df.loc[i]["face_y"]
                right = left + self.df.loc[i]["face_width"]
                bottom = top + self.df.loc[i]["face_height"]
                bb = np.array([top, left, bottom, right])

                im_fullfile = Path(self.input_dir) / im_file
                try:
                    detection, _, _, bbox_type, landmarks = self._detect_faces_in_image(im_fullfile, detected_faces=[bb])
                except Exception as e:
                # except ValueError as e:
                    print(f"Failed to load file:")
                    print(f"{im_fullfile}")
                    print(traceback.print_exc())
                    continue
                # except SyntaxError as e:
                #     print(f"Failed to load file:")
                #     print(f"{im_fullfile}")
                #     print(traceback.print_exc())
                #     continue

                out_detection_fname = self._path_to_detections() / Path(im_file).parent / (Path(im_file).stem + self.processed_ext)
                # detection_fnames += [out_detection_fname.relative_to(self.output_dir)]
                out_detection_fname.parent.mkdir(exist_ok=True)
                detection_fnames += [out_detection_fname]
                if self.processed_ext in [".jpg", ".JPG"]:
                    imsave(out_detection_fname, detection[0], quality=100)
                else:
                    imsave(out_detection_fname, detection[0])
                # out_segmentation_folders += [self._path_to_segmentations() / Path(im_file).parent]

                # save landmarks
                out_landmark_fname = self._path_to_landmarks() / Path(im_file).parent / (Path(im_file).stem + ".pkl")
                out_landmark_fname.parent.mkdir(exist_ok=True)
                # landmark_fnames += [out_landmark_fname.relative_to(self.output_dir)]
                save_landmark(out_landmark_fname, landmarks[0], bbox_type)

            self._segment_images(detection_fnames, self._path_to_segmentations(), path_depth=1)

            status_array = np.memmap(self.status_array_path,
                                     dtype=np.bool,
                                     mode='r+',
                                     shape=(self.num_subsets,)
                                     )
            status_array[start_i // self.subset_size] = True
            status_array.flush()
            del status_array
            print(f"Processing subset {start_i // self.subset_size} finished")
        else:
            print(f"Subset {start_i // self.subset_size} is already processed")

    @property
    def status_array_path(self):
        return Path(self.output_dir) / "status.memmap"

    @property
    def is_processed(self):
        status_array = np.memmap(self.status_array_path,
                                 dtype=np.bool,
                                 mode='r',
                                 shape=(self.num_subsets,)
                                 )
        all_processed = status_array.all()
        return all_processed

    def prepare_data(self):
        pass
        # if self.use_processed:
        #     if not self.status_array_path.is_file():
        #         print(f"Status file does not exist. Creating '{self.status_array_path}'")
        #         self.status_array_path.parent.mkdir(exist_ok=True, parents=True)
        #         status_array = np.memmap(self.status_array_path,
        #                                  dtype=np.bool,
        #                                  mode='w+',
        #                                  shape=(self.num_subsets,)
        #                                  )
        #         status_array[...] = False
        #         del status_array
        #
        #     all_processed = self.is_processed
        #     if not all_processed:
        #         self._detect_faces()
        #
        #
        # if self.ring_type == "emonet_feature":
        #     self._prepare_emotion_retrieval()

    def _new_training_set(self, for_training=True):
        if for_training:
            im_transforms_train = create_image_augmenter(self.image_size, self.augmentation)

            if self.ring_type == "emonet_feature":
                prefix = self.mode + "_train_"
                if self.ignore_invalid:
                    prefix += "valid_only_"
                feature_label = 'emo_net_emo_feat_2'
                self._load_retrieval_arrays(prefix, feature_label)
                nn_indices = self.nn_indices_array
                nn_distances = self.nn_distances_array
            else:
                nn_indices = None
                nn_distances = None

            return new_affewva(self.dataset_type)(self.image_path, self.train_list, self.image_size, self.scale,
                          im_transforms_train,
                          ring_type=self.ring_type,
                          ring_size=self.ring_size,
                          load_emotion_feature=False,
                          nn_indices_array=nn_indices,
                          nn_distances_array= nn_distances,
                          ext=self.processed_ext,
                          bb_center_shift_x=self.bb_center_shift_x,
                          bb_center_shift_y=self.bb_center_shift_y,
                          )

        return new_affewva(self.dataset_type)(self.image_path, self.train_list, self.image_size, self.scale,
                         None,
                         ring_type=None,
                         ring_size=None,
                         load_emotion_feature=True,
                         ext=self.processed_ext,
                          bb_center_shift_x=self.bb_center_shift_x,
                          bb_center_shift_y=self.bb_center_shift_y,
                         )

    def setup(self, stage=None):
        self.training_set = self._new_training_set()
        self.validation_set = new_affewva(self.dataset_type)(self.image_path, self.val_list, self.image_size, self.scale,
                                        None,
                                        ring_type=None,
                                        ring_size=None,
                                        ext=self.processed_ext,
                                     bb_center_shift_x=self.bb_center_shift_x,
                                     bb_center_shift_y=self.bb_center_shift_y,
                                        )

        self.test_dataframe_path = Path(self.output_dir) / "validation_representative_selection.csv"
        self.test_set = new_affewva(self.dataset_type)(self.image_path, self.test_list, self.image_size, self.scale,
                                    None,
                                  ring_type=None,
                                  ring_size=None,
                                  ext=self.processed_ext,
                               bb_center_shift_x=self.bb_center_shift_x,
                               bb_center_shift_y=self.bb_center_shift_y,
                                  )
        # if self.mode in ['all', 'manual']:
        #     # self.image_list += sorted(list((Path(self.path) / "Manually_Annotated").rglob(".jpg")))
        #     self.dataframe = pd.load_csv(self.path / "Manually_Annotated" / "Manually_Annotated.csv")
        # if self.mode in ['all', 'automatic']:
        #     # self.image_list += sorted(list((Path(self.path) / "Automatically_Annotated").rglob("*.jpg")))
        #     self.dataframe = pd.load_csv(
        #         self.path / "Automatically_Annotated" / "Automatically_annotated_file_list.csv")

    def train_dataloader(self):
        if self.sampler == "uniform":
            sampler = None
        else:
            raise NotImplementedError()
        # elif self.sampler == "balanced_expr":
        #     sampler = make_class_balanced_sampler(self.training_set.df["expression"].to_numpy())
        # elif self.sampler == "balanced_va":
        #     sampler = make_balanced_sample_by_weights(self.training_set.va_sample_weights)
        # elif self.sampler == "balanced_v":
        #     sampler = make_balanced_sample_by_weights(self.training_set.v_sample_weights)
        # elif self.sampler == "balanced_a":
        #     sampler = make_balanced_sample_by_weights(self.training_set.a_sample_weights)
        # else:
        #     raise ValueError(f"Invalid sampler value: '{self.sampler}'")
        dl = DataLoader(self.training_set, shuffle=sampler is None, num_workers=self.num_workers, pin_memory=True,
                        batch_size=self.train_batch_size, drop_last=self.drop_last, sampler=sampler)
        return dl

    def val_dataloader(self):
        return DataLoader(self.validation_set, shuffle=True, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.val_batch_size, drop_last=False)

    def test_dataloader(self):
        return [
            self.val_dataloader(),
            DataLoader(self.test_set, shuffle=True, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.test_batch_size, drop_last=False)
        ]

    def _get_retrieval_array(self, prefix, feature_label, dataset_size, feature_shape, feature_dtype, modifier='w+'):
        outfile_name = self._path_to_emotion_nn_retrieval_file(prefix, feature_label)
        if outfile_name.is_file() and modifier != 'r':
            raise RuntimeError(f"The retrieval array already exists! '{outfile_name}'")

        shape = tuple([dataset_size] + list(feature_shape))
        outfile_name.parent.mkdir(exist_ok=True, parents=True)
        array = np.memmap(outfile_name,
                         dtype=feature_dtype,
                         mode=modifier,
                         shape=shape
                         )
        return array

    def _path_to_emotion_nn_indices_file(self, prefix, feature_label):
        nn_indices_file = Path(self.output_dir) / "cache" / (prefix + feature_label + "_nn_indices.memmap")
        return nn_indices_file

    def _path_to_emotion_nn_distances_file(self,  prefix, feature_label):
        nn_distances_file = Path(self.output_dir) / "cache" / (prefix + feature_label + "_nn_distances.memmap")
        return nn_distances_file

    def _path_to_emotion_nn_retrieval_file(self,  prefix, feature_label):
        outfile_name = Path(self.output_dir) / "cache" / (prefix + feature_label + ".memmap")
        return outfile_name

    def _load_retrieval_arrays(self, prefix, feature_label):
        # prefix = self.mode + "_train_"
        # if self.ignore_invalid:
        #     prefix += "valid_only_"
        # feature_label = 'emo_net_emo_feat_2'
        nn_indices_file = self._path_to_emotion_nn_indices_file(prefix, feature_label)
        nn_distances_file = self._path_to_emotion_nn_distances_file(prefix, feature_label)
        try:
            with open(nn_indices_file.parent / (nn_indices_file.stem + "_meta.pkl"), "rb") as f:
                indices_array_dtype = pkl.load(f)
                indices_array_shape = pkl.load(f)
        except:
            indices_array_dtype = np.int64,
            indices_array_shape = (len(dataset), NUM_NEIGHBORS)

        try:
            with open(nn_distances_file.parent / (nn_distances_file.stem + "_meta.pkl"), "rb") as f:
                distances_array_dtype = pkl.load(f)
                distances_array_shape = pkl.load(f)
        except:
            distances_array_dtype = np.float32,
            distances_array_shape = (len(dataset), NUM_NEIGHBORS)

        self.nn_indices_array = np.memmap(nn_indices_file,
                                          # dtype=np.int32,
                                          dtype=indices_array_dtype,
                                          mode="r",
                                          shape=indices_array_shape
                                          )

        self.nn_distances_array = np.memmap(nn_distances_file,
                                            dtype=distances_array_dtype,
                                            # dtype=np.float64,
                                            mode="r",
                                            shape=distances_array_shape
                                            )

    def _prepare_emotion_retrieval(self):
        prefix = self.mode + "_train_"
        if self.ignore_invalid:
            prefix += "valid_only_"
        feature_label = 'emo_net_emo_feat_2'
        nn_indices_file = self._path_to_emotion_nn_indices_file(prefix, feature_label)
        nn_distances_file = self._path_to_emotion_nn_distances_file(prefix, feature_label)
        NUM_NEIGHBORS = 100
        if nn_indices_file.is_file() and nn_distances_file.is_file():
            print("Precomputed nn arrays found.")
            return
        dataset = self._new_training_set(for_training=False)
        dl = DataLoader(dataset, shuffle=False, num_workers=self.num_workers, batch_size=self.train_batch_size)

        array = None
        if self.ring_type != "emonet_feature":
            raise ValueError(f"Invalid ring type for emotion retrieval {self.ring_type}")

        outfile_name = self._path_to_emotion_nn_retrieval_file(prefix, feature_label)
        if not outfile_name.is_file():
            for bi, batch in enumerate(auto.tqdm(dl)):
                feat = batch[feature_label].numpy()
                feat_size = feat.shape[1:]
                if array is None:
                    array = self._get_retrieval_array(prefix, feature_label, len(dataset), feat_size, feat.dtype)

                # for i in range(feat.shape[0]):
                #     idx = bi*self.train_batch_size + i
                array[bi*self.train_batch_size:bi*self.train_batch_size + feat.shape[0], ...] = feat
            del array
        else:
            print(f"Feature array found in '{outfile_name}'")
            for bi, batch in enumerate(dl):
                feat = batch[feature_label].numpy()
                feat_size = feat.shape[1:]
                break

        array = self._get_retrieval_array(prefix, feature_label, len(dataset), feat_size, feat.dtype, modifier='r')

        nbrs = NearestNeighbors(n_neighbors=30, algorithm='auto', n_jobs=-1).fit(array)
        distances, indices = nbrs.kneighbors(array, NUM_NEIGHBORS)

        indices_array = np.memmap(nn_indices_file,
                         dtype=indices.dtype,
                         mode="w+",
                         shape=indices.shape
                         )
        indices_array[...] = indices
        del indices_array
        distances_array = np.memmap(nn_distances_file,
                         dtype=distances.dtype,
                         mode="w+",
                         shape=distances.shape
                         )
        distances_array[...] = distances
        del distances_array

        # save sizes a dtypes
        with open(nn_indices_file.parent / (nn_indices_file.stem + "_meta.pkl"), "wb") as f:
            pkl.dump(indices.dtype, f)
            pkl.dump(indices.shape, f)

        with open(nn_distances_file.parent / (nn_distances_file.stem + "_meta.pkl"), "wb") as f:
            pkl.dump(distances.dtype, f)
            pkl.dump(distances.shape, f)

        self.nn_indices_array = np.memmap(nn_indices_file,
                         dtype=indices.dtype,
                         mode="r",
                         shape=indices.shape
                         )

        self.nn_distances_array = np.memmap(nn_distances_file,
                         dtype=distances.dtype,
                         mode="r",
                         shape=distances.shape
                         )


class AfewVaDataVisTestModule(AfewVaDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def setup(self, stage=None):
        self.training_set = None
        self.validation_set = TestSubsetAfewVa(self.image_path, self.val_list, self.image_size, self.scale,
                                        None,
                                        ring_type=None,
                                        ring_size=None,
                                        ext=self.processed_ext,
                                     bb_center_shift_x=self.bb_center_shift_x,
                                     bb_center_shift_y=self.bb_center_shift_y,
                                        )

        self.test_dataframe_path = Path(self.output_dir) / "validation_representative_selection.csv"
        self.test_set = TestSubsetAfewVa(self.image_path, self.test_list, self.image_size, self.scale,
                                    None,
                                  ring_type=None,
                                  ring_size=None,
                                  ext=self.processed_ext,
                               bb_center_shift_x=self.bb_center_shift_x,
                               bb_center_shift_y=self.bb_center_shift_y,
                                  )

    def val_dataloader(self):
        return DataLoader(self.validation_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.val_batch_size, drop_last=False)

    def test_dataloader(self):
        return [
            self.val_dataloader(),
            DataLoader(self.test_set, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          batch_size=self.test_batch_size, drop_last=False)
        ]


class AfewVa(EmotionalImageDatasetBase):

    def __init__(self,
                 image_path,
                 sample_list,
                 image_size,
                 scale = 1.4,
                 transforms : imgaug.augmenters.Augmenter = None,
                 use_gt_bb=True,
                 bb_center_shift_x=0.0,
                 bb_center_shift_y=0.0,
                 ring_type=None,
                 ring_size=None,
                 load_emotion_feature=False,
                 nn_indices_array=None,
                 nn_distances_array=None,
                 ext=".png",
                 use_processed = None,
                 normalize_va = None,
                 ):
        self.sample_list = sample_list
        self.image_path = image_path
        self.image_size = image_size
        self.use_gt_bb = use_gt_bb
        # self.transforms = transforms or imgaug.augmenters.Identity()
        self.transforms = transforms or imgaug.augmenters.Resize((image_size, image_size))
        self.scale = scale
        self.landmark_normalizer = KeypointNormalization()
        self.use_processed = use_processed or False
        self.normalize_va = normalize_va or True # normalize va to <-1,1>
        # self.ignore_invalid = ignore_invalid
        self.load_emotion_feature = load_emotion_feature
        self.nn_distances_array = nn_distances_array
        self.ext=ext
        #
        # if ignore_invalid:
        #     # filter invalid classes
        #     ignored_classes = [AffectNetExpressions.Uncertain.value, AffectNetExpressions.Occluded.value]
        #     self.df = self.df[self.df["expression"].isin(ignored_classes) == False]
        #     # self.df = self.df.drop(self.df[self.df["expression"].isin(ignored_classes)].index)
        #
        #     # filter invalid va values
        #     self.df = self.df[self.df.valence != -2.]
        #     # self.df = self.df.drop(self.df.valence == -2.)
        #     self.df = self.df[self.df.arousal != -2.]
        #     # self.df = self.df.drop(self.df.arousal == -2.)
        #     # valid_indices = np.logical_not(pd.isnull(self.df))
        #     # valid_indices = self.df.index
        #     self.df = self.df.reset_index(drop=True)
        #     # if nn_indices_array is not None and nn_indices_array.shape[0] != len(self.df):
        #     #     nn_indices_array = nn_indices_array[valid_indices, ...]
        #     # if nn_distances_array is not None and nn_distances_array.shape[0] != len(self.df):
        #     #     nn_distances_array = nn_distances_array[valid_indices, ...]
        #
        # self.exp_weights = self.df["expression"].value_counts(normalize=True).to_dict()
        # self.exp_weight_tensor = torch.tensor([self.exp_weights[i] for i in range(len(self.exp_weights))], dtype=torch.float32)
        # self.exp_weight_tensor = 1. / self.exp_weight_tensor
        # self.exp_weight_tensor /= torch.norm(self.exp_weight_tensor)

        self.size = 0
        self.video_sizes_cumulative = [0]
        for vid_name, gt in self.sample_list.items():
            self.size += len(gt.frames)
            self.video_sizes_cumulative += [self.size]

        if ring_type not in [None, "gt_expression", "gt_va", "emonet_feature", "emonet_va", "emonet_expression", "augment"]:
            raise ValueError(f"Invalid ring type '{ring_type}'")
        if ring_type == "emonet_expression" and ( nn_indices_array is None or nn_distances_array is None ):
            raise ValueError(f"If ring type set to '{ring_type}', nn files must be specified")

        self.ring_type = ring_type
        self.ring_size = ring_size
        # self._init_sample_weights()


    def _init_sample_weights(self):

        raise NotImplementedError()
        if self.ring_type == "gt_expression":
            grouped = self.df.groupby(['expression'])
            self.expr2sample = grouped.groups
        elif self.ring_type == "emonet_expression":
            raise NotImplementedError()
        else:
            self.expr2sample = None

        va = self.df[["valence", "arousal"]].to_numpy()
        sampling_rate = 0.1
        # bin_1d = np.arange(-1.,1.+sampling_rate, sampling_rate)
        bin_1d = np.arange(-1.,1., sampling_rate)
        stat, x_ed, y_ed, va_binnumber = sp.stats.binned_statistic_2d(
            va[:, 0], va[:, 1], None, 'count', [bin_1d, bin_1d], expand_binnumbers=False)
        va_weights = 1 / va_binnumber
        va_weights /= np.linalg.norm(va_weights)
        va_weights *= np.linalg.norm(np.ones_like(va_weights))
        self.va_sample_weights = va_weights

        if self.ring_type == "gt_va":
            raise NotImplementedError()
            self.bins_to_samples = {}
            self.va_bin_indices = va_binnumber
            bin_indices = np.unique(va_binnumber)
            for bi in bin_indices:
                self.bins_to_samples[bi] = np.where(va_binnumber == bi)[0]

        elif self.ring_type == "emonet_va":
            raise NotImplementedError()
        else:
            self.bins_to_samples = {}

        if self.ring_type == "emonet_feature":

            raise NotImplementedError()
            if len(self) != self.nn_distances_array.shape[0] or len(self) != self.nn_indices_array.shape[0]:
                raise RuntimeError("The lengths of the dataset does not correspond to size of the nn_array. "
                                   "The sizes should be equal. Sth fishy is happening")
            # self.nn_indices_array = self.nn_indices_array
            self.nn_distances_array = nn_distances_array
        else:
            self.nn_indices_array = None
            self.nn_distances_array = None


        # v = self.df[["valence"]].to_numpy()
        sampling_rate = 0.1

        bin_1d = np.arange(-1.,1., sampling_rate)
        stat, x_ed, va_binnumber = sp.stats.binned_statistic(
            va[:, 0], None, 'count', bin_1d)
        v_weights = 1 / va_binnumber
        v_weights /= np.linalg.norm(v_weights)
        v_weights *= np.linalg.norm(np.ones_like(v_weights))
        self.v_sample_weights = v_weights

        bin_1d = np.arange(-1.,1., sampling_rate)
        stat, x_ed, va_binnumber = sp.stats.binned_statistic(
            va[:, 1], None, 'count', bin_1d)
        a_weights = 1 / va_binnumber
        a_weights /= np.linalg.norm(a_weights)
        a_weights *= np.linalg.norm(np.ones_like(a_weights))
        self.a_sample_weights = a_weights


    def __len__(self):
        # return 100
        return self.size

    def _load_image(self, index):
        vid_index = bisect.bisect_right(self.video_sizes_cumulative, index) - 1
        vid_first_im_index = self.video_sizes_cumulative[vid_index]
        vid_item = list(self.sample_list.items())[vid_index]
        vid_name = vid_item[0]
        vid_gt = vid_item[1]
        assert vid_gt.video_id == vid_name
        vid_frame_list = sorted(list(vid_gt['frames'].keys()))
        selected_frame = vid_frame_list[index - vid_first_im_index]
        im_rel_path = Path(vid_name) / (selected_frame + self.ext)
        im_file = Path(self.image_path) / im_rel_path
        im_file = im_file.parent / (im_file.stem + self.ext)
        input_img = imread(im_file)
        # # scale_factor_x = 1.48
        # scale_factor_x = 1.25
        # # scale_factor_x = 1
        # input_img = resize(input_img, (432, 720, 1))
        input_img = resize(input_img, (576, 960))
        scale_factor_x = 720 / 960
        valence = vid_gt['frames'][selected_frame]['valence']
        arousal = vid_gt['frames'][selected_frame]['arousal']
        facial_landmarks = np.array(vid_gt['frames'][selected_frame]['landmarks'])
        facial_landmarks[:,0] /= scale_factor_x

        if self.normalize_va:
            valence /= 10.
            arousal /= 10.

        return input_img, facial_landmarks, valence, arousal, im_file

    def _load_additional_data(self, im_rel_path):
        return {}

    def _get_sample(self, index):
        num_fails = 0
        max_fails = 50
        try:
            input_img, facial_landmarks, valence, arousal, image_path  = self._load_image(index)
            additional_data = self._load_additional_data(Path(image_path).relative_to(self.image_path))
        except Exception as e:
            # if the image is corrupted or missing (there is a few :-/), find some other one
            while True:
                num_fails += 1
                if num_fails >= max_fails:
                    # something must have gone serious wrong. Nothing loads, so throw an exception
                    raise e
                index += 1
                index = index % len(self)
                try:
                    input_img, facial_landmarks, valence, arousal, image_path = self._load_image(index)
                    additional_data = self._load_additional_data(Path(image_path).relative_to(self.image_path))
                    success = True
                except Exception as e2:
                    success = False
                if success:
                    break

        left = facial_landmarks[:,0].min()
        top =  facial_landmarks[:,1].min()
        right =  facial_landmarks[:,0].max()
        bottom = facial_landmarks[:,1].max()
        input_img_shape = input_img.shape

        if not self.use_processed:
            # Use AffectNet as is provided (their bounding boxes, and landmarks, no segmentation)
            old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            # old_size, center = bbox2point(left, right, top, bottom, type='bbox')
            size = int(old_size * self.scale)
            img, landmark = bbpoint_warp(input_img, center, size, self.image_size, landmarks=facial_landmarks)
            img *= 255.

            if not self.use_gt_bb:
                raise NotImplementedError()
                # landmark_type, landmark = load_landmark(
                #     self.path_prefix / self.landmark_list[index])
            landmark = landmark[np.newaxis, ...]
            seg_image = None
        else:
            # use AffectNet processed by me. I used their bounding boxes (to not have to worry about detecting
            # the correct face in case there's more) and I ran our FAN and segmentation over it
            img = input_img

            # the image has already been cropped in preprocessing (make sure the input root path
            # is specificed to the processed folder and not the original one

            landmark_path = Path(self.image_path).parent / "landmarks" / im_rel_path
            landmark_path = landmark_path.parent / (landmark_path.stem + ".pkl")

            landmark_type, landmark = load_landmark(
                landmark_path)
            landmark = landmark[np.newaxis, ...]

            segmentation_path = Path(self.image_path).parent / "segmentations" / im_rel_path
            segmentation_path = segmentation_path.parent / (segmentation_path.stem + ".pkl")

            seg_image, seg_type = load_segmentation(
                segmentation_path)
            seg_image = seg_image[np.newaxis, :, :, np.newaxis]

            seg_image = process_segmentation(
                seg_image, seg_type).astype(np.uint8)

            if self.load_emotion_feature:
                emotion_path = Path(self.image_path).parent / "emotions" / im_rel_path
                emotion_path = emotion_path.parent / (emotion_path.stem + ".pkl")
                emotion_features, emotion_type = load_emotion(emotion_path)
            else:
                emotion_features = None

        img, seg_image, landmark = self._augment(img, seg_image, landmark)

        sample = {
            "image": numpy_image_to_torch(img.astype(np.float32)),
            "path": str(image_path),
            # "affectnetexp": torch.tensor([expression, ], dtype=torch.long),
            "va": torch.tensor([valence, arousal], dtype=torch.float32),
            "label": str(image_path.stem),
            # "expression_weight": self.exp_weight_tensor,
            # "expression_sample_weight": torch.tensor([self.exp_weights[expression], ]),
            # "valence_sample_weight": torch.tensor([self.v_sample_weights[index],], dtype=torch.float32),
            # "arousal_sample_weight": torch.tensor([self.a_sample_weights[index],], dtype=torch.float32),
            # "va_sample_weight": torch.tensor([self.va_sample_weights[index],], dtype=torch.float32),
            **additional_data
        }

        if landmark is not None:
            sample["landmark"] = torch.from_numpy(landmark)
        if seg_image is not None:
            sample["mask"] = numpy_image_to_torch(seg_image)
        # if emotion_features is not None:
        #     for key, value in emotion_features.items():
        #         if isinstance(value, np.ndarray):
        #             sample[emotion_type + "_" + key] = torch.from_numpy(value)
        #         else:
        #             sample[emotion_type + "_" + key] = torch.tensor([value])
        # print(self.df.loc[index])
        return sample

    def __getitem__(self, index):
        if self.ring_type is None or self.ring_size == 1:
        #TODO: check if following line is a breaking change
        # if self.ring_type is None: # or self.ring_size == 1:
            return self._get_sample(index)

        sample = self._get_sample(index)

        self.ring_policy = 'random'

        # retrieve indices of the samples relevant for this ring
        if self.ring_type == "gt_expression" or self.ring_type == "emonet_expression":
            expression_label = sample["affectnetexp"]
            ring_indices = self.expr2sample[expression_label.item()]
            ring_indices = list(ring_indices)
            if len(ring_indices) > 1:
                ring_indices.remove(index)
            label = expression_label
        elif self.ring_type == "gt_va" or self.ring_type == "emonet_va":
            ring_indices = self.bins_to_samples[self.va_bin_indices[index]].tolist()
            if len(ring_indices) > 1:
                ring_indices.remove(index)
            label = self.va_bin_indices[index]
        elif self.ring_type == "emonet_feature":
            ring_indices = self.nn_indices_array[index].tolist()
            # ring_indices = [n for n in ring_indices if n < len(self)]
            max_nn = 10
            ring_indices = ring_indices[:max_nn]
            if len(ring_indices) > 1:
                ring_indices.remove(index)
            # label = index
        elif self.ring_type == "augment":
            ring_indices = (self.ring_size-1) * [index]
        else:
            raise NotImplementedError()

        # label = self.labels[index]
        # label_indices = self.label2index[label]

        if self.ring_policy == 'random':
            picked_label_indices = np.arange(len(ring_indices), dtype=np.int32)
            # print("Size of label_indices:")
            # print(len(label_indices))
            np.random.shuffle(picked_label_indices)
            if len(ring_indices) < self.ring_size - 1:
                print(
                    f"[WARNING]. Label '{label}' only has {len(ring_indices)} samples which is less than {self.ring_size}. S"
                    f"ome samples will be duplicated")
                picked_label_indices = np.concatenate(self.ring_size * [picked_label_indices], axis=0)

            picked_label_indices = picked_label_indices[:self.ring_size - 1]
            indices = [ring_indices[i] for i in picked_label_indices]
        elif self.ring_policy == 'sequential':
            indices = []
            idx = ring_indices.index(index) + 1
            idx = idx % len(ring_indices)
            while len(indices) != self.ring_size - 1:
                # if self.labels[idx] == label:
                indices += [ring_indices[idx]]
                idx += 1
                idx = idx % len(ring_indices)
        else:
            raise ValueError(f"Invalid ring policy {self.ring_policy}")

        batches = []
        batches += [sample]
        for i in range(self.ring_size - 1):
            # idx = indices[i]
            idx = indices[i]
            batches += [self._get_sample(idx)]

        try:
            combined_batch = default_collate(batches)
        except RuntimeError as e:
            print(f"Failed for index {index}")
            # print("Failed paths: ")
            for bi, batch in enumerate(batches):
                print(f"Index= {bi}")
                print(f"Path='{batch['path']}")
                print(f"Label='{batch['label']}")
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(f"{key} shape='{batch[key].shape}")
            raise e

        # end = timer()
        # print(f"Reading sample {index} took {end - start}s")
        return combined_batch


def sample_representative_set(dataset, output_file, sample_step=0.1, num_per_bin=2):
    va_array = []
    size = int(2 / sample_step)
    for i in range(size):
        va_array += [[]]
        for j in range(size):
            va_array[i] += [[]]

    print("Binning dataset")
    for i in auto.tqdm(range(len(dataset.df))):
        v = max(-1., min(1., dataset.df.loc[i]["valence"]))
        a = max(-1., min(1., dataset.df.loc[i]["arousal"]))
        row_ = int((v + 1) / sample_step)
        col_ = int((a + 1) / sample_step)
        va_array[row_][ col_] += [i]


    selected_indices = []
    for i in range(len(va_array)):
        for j in range(len(va_array[i])):
            if len(va_array[i][j]) > 0:
                # selected_indices += [va_array[i][j][0:num_per_bin]]
                selected_indices += va_array[i][j][0:num_per_bin]
            else:
                print(f"No value for {i} and {j}")

    selected_samples = dataset.df.loc[selected_indices]
    selected_samples.to_csv(output_file)
    print(f"Selected samples saved to '{output_file}'")


class AfewVaNetWithPredictions(AfewVa):

    def __init__(self, predictor, shape_name, exp_name, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.predictor = predictor
        self.shape_prediction_name = shape_name
        self.exp_prediction_name = exp_name

    def _load_additional_data(self, im_rel_path):
        rel_path = Path(im_rel_path).parent / Path(im_rel_path).stem
        prediction_path = Path(self.image_path).parent / "afew-va" / "not_processed" / "predictions" / self.predictor / rel_path
        shape_prediction_path = prediction_path / (self.shape_prediction_name + ".npy")
        exp_prediction_path = prediction_path / (self.exp_prediction_name + ".npy")

        shape_prediction = np.load(shape_prediction_path).squeeze()
        exp_prediction = np.load(exp_prediction_path).squeeze()

        additional = {}
        additional["shapecode"] = torch.from_numpy(shape_prediction)
        additional["expcode"] = torch.from_numpy(exp_prediction)
        return additional


class TestSubsetAfewVa(AfewVa):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 20

    def __len__(self):
        return super().__len__() // self.step

    def __getitem__(self,index):
        return super().__getitem__(index*self.step)


class AfewVaWithMGCNetPredictions(AfewVaNetWithPredictions):

    def __init__(self, *args, **kwargs):
        predictor = "MGCNet"
        shape_name = "shape_coeffs"
        exp_name = "exp_coeffs"
        super().__init__(predictor, shape_name, exp_name, *args, **kwargs)


class AfewVaWithExpNetPredictions(AfewVaNetWithPredictions):

    def __init__(self, *args, **kwargs):
        predictor = "ExpNet"
        shape_name = "shape"
        exp_name = "exp"
        super().__init__(predictor, shape_name, exp_name, *args, **kwargs)


class AAfewVaWithExpNetPredictionsMyCrop(AfewVaNetWithPredictions):

    def __init__(self, *args, **kwargs):
        predictor = "ExpNet_my_crops"
        shape_name = "shape"
        exp_name = "exp"
        super().__init__(predictor, shape_name, exp_name, *args, **kwargs)


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

    ## FIRST VERSION, CLASSIC FAN-LIKE problems from too tight bb (such as forehead cut in half, etc)
    # dm = AffectNetDataModule(
    #          # "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
    #          "/ps/project_cifs/EmotionalFacialAnimation/data/affectnet/",
    #          # "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/",
    #          # "/home/rdanecek/Workspace/mount/work/rdanecek/data/affectnet/",
    #          "/is/cluster/work/rdanecek/data/affectnet/",
    #          # processed_subfolder="processed_2021_Apr_02_03-13-33",
    #          processed_subfolder="processed_2021_Apr_05_15-22-18",
    #          mode="manual",
    #          scale=1.25,
    #          ignore_invalid=True,
    #          # ring_type="gt_expression",
    #          ring_type="gt_va",
    #          # ring_type="emonet_feature",
    #          ring_size=4
    #         )
    import yaml
    # augmenter = yaml.load(open(Path(__file__).parents[2] / "gdl_apps" / "EmotionRecognition" / "emodeca_conf" / "data" / "augmentations" / "default_with_resize.yaml"))["augmentation"]
    augmenter = None

    dm = AfewVaDataModule(
             # "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
             # "/ps/project_cifs/EmotionalFacialAnimation/data/afew-va/",
             # "/ps/project/EmotionalFacialAnimation/data/afew-va/",
             "/is/cluster/work/rdanecek/data/afew-va_not_processed/",
             # "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/",
             # "/home/rdanecek/Workspace/mount/work/rdanecek/data/affectnet/",
             "/is/cluster/work/rdanecek/data/afew-va/",
             # processed_subfolder="processed_2021_Nov_07_23-37-18",
             processed_subfolder="not_processed",
             processed_ext=".png",
             # scale=1.7,
             # scale=1.25,
             scale=1.4,
             # image_size=224,
             image_size=512,
             bb_center_shift_x=0,
             bb_center_shift_y=-0.5,
             # ignore_invalid=True,
             # ring_type="gt_expression",
             # ring_type="gt_va",
             # ring_type="emonet_feature",
             # ring_size=4,
            augmentation=augmenter,
            # dataset_type="AfewVaWithMGCNetPredictions",
            dataset_type="AfewVaWithExpNetPredictions",
            )

    # print(dm.num_subsets)
    dm.prepare_data()
    dm.setup()
    # dm._extract_emotion_features()
    data = dm.validation_set
    len(data)



    print(f"len training set: {len(dm.training_set)}")
    print(f"len validation set: {len(dm.validation_set)}")
    # dl = dm.train_dataloader()
    # for bi, batch in enumerate(dl):
    #     # if bi == 10:
    #     #     break
    idxs = np.arange(len(data))
    np.random.shuffle(idxs)
    for si in range(len(data)):
        sample = dm.training_set[idxs[si]]
        print(sample.keys())
        dm.training_set.visualize_sample(sample)

    # out_path = Path(dm.output_dir) / "validation_representative_selection_.csv"
    # sample_representative_set(dm.validation_set, out_path)
    #
    # validation_set = AffectNet(
    #     dm.image_path, out_path, dm.image_size, dm.scale, None,
    # )
    # for i in range(len(validation_set)):
    #     sample = validation_set[i]
    #     validation_set.visualize_sample(sample)
    #
    # # dl = DataLoader(validation_set, shuffle=False, num_workers=1, batch_size=1)

    # "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/processed_2021_Apr_02_03-13-33/validation_representative_selection_.csv"

    # dm._detect_faces()
