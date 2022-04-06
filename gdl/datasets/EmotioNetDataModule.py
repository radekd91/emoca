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


from enum import Enum


class ActionUnitTypes(Enum):
    """
    Enum that labels subsets of AUs used by EmotioNet
    """

    EMOTIONET12 = 1
    EMOTIONET23 = 2
    ALL = 3

    @staticmethod
    def AUtype2AUlist(t):
        if t == ActionUnitTypes.EMOTIONET12:
            return [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43]
        elif t == ActionUnitTypes.EMOTIONET23:
            return [1, 2, 4, 5, 6, 9, 10, 12, 15, 17, 18, 20, 24, 25, 26, 28, 43, 51, 52, 53, 54, 55, 56]
        elif t == ActionUnitTypes.ALL:
            return list(range(1,60))
        raise ValueError(f"Invalid action unit type {t}")

    @staticmethod
    def AUtype2AUstring_list(t):
        l = ActionUnitTypes.AUtype2AUlist(t)
        string_list = [f"AU{i}" for i in l]
        return string_list

    @staticmethod
    def numAUs(t):
        return len(ActionUnitTypes.AUtype2AUlist(t))

    @staticmethod
    def AU_num_2_name(num):
        d = {}
        d[1] = "Inner Brow Raiser"
        d[2] = "Outer Brow Raiser"
        d[4] = "Brow Lowerer"
        d[5] = "Upper Lid Raiser"
        d[6] = "Cheek Raiser"
        d[9] = "Nose Wrinkler"
        d[10] = "Upper Lip Raiser"
        d[12] = "Lip Corner Puller"
        d[15] = "Lip Corner Depressor"
        d[17] = "Chin Raiser"
        d[18] = "Lip Puckerer"
        d[20] = "Lip Stretcher"
        d[24] = "Lip Pressor"
        d[25] = "Lips Part"
        d[26] = "Jaw Drop"
        d[28] = "Lip Suck"
        d[43] = "Eyes Closed"
        if num not in d.keys():
            raise ValueError(f"invalid AU {num}")
        return d[num]


class EmotioNetDataModule(FaceDataModuleBase):
    """
    A data module of the EmotioNet dataset. 
    http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/ 
    """

    def __init__(self,
                 input_dir,
                 output_dir,
                 processed_subfolder = None,
                 ignore_invalid = False,
                 face_detector='fan',
                 face_detector_threshold=0.9,
                 image_size=224,
                 scale=1.25,
                 bb_center_shift_x = 0.,
                 bb_center_shift_y = 0.,
                 processed_ext=".jpg",
                 device=None,
                 augmentation=None,
                 train_batch_size=64,
                 val_batch_size=64,
                 test_batch_size=64,
                 num_workers=0,
                 # ring_type=None,
                 # ring_size=None,
                 drop_last=False,
                 au_type = ActionUnitTypes.EMOTIONET12
                 # sampler=None,
                 ):
        super().__init__(input_dir, output_dir, processed_subfolder,
                         face_detector=face_detector,
                         face_detector_threshold=face_detector_threshold,
                         image_size=image_size,
                         bb_center_shift_x=bb_center_shift_x,
                         bb_center_shift_y=bb_center_shift_y,
                         processed_ext=processed_ext,
                         scale=scale,
                         device=device)
        # self.subsets = sorted([f.name for f in (Path(input_dir) / "Manually_Annotated" / "Manually_Annotated_Images").glob("*") if f.is_dir()])
        self.input_dir = Path(self.root_dir)
        # train = pd.read_csv(self.input_dir.parent / "training.csv")
        # val = pd.read_csv(self.input_dir.parent / "validation.csv")
        # columns = ["path", ]
        # columns += [f"AU{i}" for i in range(1, 61)]
        dfs_fnames = sorted(list(self.input_dir.glob("image_list_*.csv")))
        # dfs = [pd.read_csv(dfs_fname, names=columns) for dfs_fname in dfs_fnames]
        dfs = [pd.read_csv(dfs_fname) for dfs_fname in dfs_fnames]
        self.df = pd.concat(dfs, ignore_index=True, sort=False)

        self.face_detector_type = 'fan'
        self.scale = scale
        # self.use_processed = True

        self.image_path = Path(self.output_dir) / "detections"
        self.au_type = au_type

        # self.ignore_invalid = ignore_invalid

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        # self.sampler = sampler or "uniform"
        
        # if self.sampler not in ["uniform", "balanced_expr", "balanced_va", "balanced_v", "balanced_a"]:
        #     raise ValueError(f"Invalid sampler type: '{self.sampler}'")
        # if ring_type not in [None, "gt_expression", "gt_va", "emonet_feature", "emonet_va", "emonet_expression"]:
        #     raise ValueError(f"Invalid ring type: '{ring_type}'")
        # 
        # self.ring_type = ring_type
        # self.ring_size = ring_size

        self.drop_last = drop_last

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
        num_subsets = self.num_subsets
        if len(self.df) % self.subset_size != 0:
            num_subsets += 1
        for sid in range(self.num_subsets):
            self._detect_landmarks_and_segment_subset(self.subset_size * sid, min((sid + 1) * self.subset_size, len(self.df)))

    def _extract_emotion_features(self):
        num_subsets = len(self.df) // self.subset_size
        if len(self.df) % self.subset_size != 0:
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
            # in_detection_fname = self._path_to_detections() / Path(im_file).parent / (Path(im_file).stem + ".png")
            in_detection_fname = self._path_to_detections() / Path(im_file).parent / (Path(im_file).stem + self.processed_ext)
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
                im_file = self.df.loc[i]["path"]
                # left = self.df.loc[i]["face_x"]
                # top = self.df.loc[i]["face_y"]
                # right = left + self.df.loc[i]["face_width"]
                # bottom = top + self.df.loc[i]["face_height"]
                # bb = np.array([top, left, bottom, right])

                im_fullfile = Path(self.input_dir) / "images" / im_file
                try:
                    detection, _, _, bbox_type, landmarks = self._detect_faces_in_image(im_fullfile, detected_faces=None)
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

                if len(detection) == 0:
                    print(f"Skipping file {im_fullfile} because no face was detected.")
                    continue

                out_detection_fname = self._path_to_detections() / Path(im_file).parent / (Path(im_file).stem + self.processed_ext)
                # detection_fnames += [out_detection_fname.relative_to(self.output_dir)]
                out_detection_fname.parent.mkdir(exist_ok=True)
                detection_fnames += [out_detection_fname]
                # imsave(out_detection_fname, detection[0], plugin="imageio", quality=100)
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

    def _split_train_val(self, seed=0, ratio=0.9):
        self.val_dataframe_path = Path(self.output_dir) / f"validation_set_{seed}_{ratio:0.4f}.csv"
        self.train_dataframe_path = Path(self.output_dir) / f"training_set_{seed}_{ratio:0.4f}.csv"
        self.full_dataframe_path = Path(self.output_dir) / f"full_dataset.csv"

        if self.val_dataframe_path.is_file() and self.train_dataframe_path.is_file():
            self.train_df = pd.read_csv(self.train_dataframe_path)
            self.val_df = pd.read_csv(self.val_dataframe_path)
            pass
        else:

            if self.full_dataframe_path.is_file():
                cleaned_df = pd.read_csv(self.full_dataframe_path)
            else:
                indices_to_delete = []
                for i in auto.tqdm(range(len(self.df))):
                    detection_path = Path(self.output_dir) / "detections" / self.df["path"][i]
                    if not detection_path.is_file():
                        indices_to_delete += [i]
                    # if i == 5000:
                    #     break
                cleaned_df = self.df.drop(indices_to_delete)
                cleaned_df.to_csv(self.full_dataframe_path)

                print(f"Kept {len(cleaned_df)}/{len(self.df)} images because the detection was missing. Dropping {len(indices_to_delete)}")
                # sys.exit()

            N = len(self.df)
            indices = np.arange(N, dtype=np.int32)
            np.random.seed(seed)
            np.random.shuffle(indices)
            train_indices = indices[:int(0.9*N)]
            val_indices = indices[len(train_indices):]
            self.train_df = self.df.iloc[train_indices.tolist()]
            self.val_df = self.df.iloc[val_indices.tolist()]

            self.train_df.to_csv(self.train_dataframe_path)
            self.val_df.to_csv(self.val_dataframe_path)


        # return self.train_df, self.val_df


    def _dataset_anaylysis(self):
        cleaned_df = pd.read_csv(self.full_dataframe_path)
        arr = cleaned_df[ActionUnitTypes.AUtype2AUstring_list(ActionUnitTypes.EMOTIONET12)].to_numpy(np.float)
        unique_au_configs, counts = np.unique(arr, return_counts=True, axis=0)
        print("There is {len(unique_au_configs)} configurations in the dataset. ")
        import matplotlib.pyplot  as plt
        plt.figure()
        plt.plot(counts)
        plt.figure()
        plt.plot(np.sort(counts))
        plt.show()


    def prepare_data(self):
        # if self.use_processed:
        if not self.status_array_path.is_file():
            print(f"Status file does not exist. Creating '{self.status_array_path}'")
            self.status_array_path.parent.mkdir(exist_ok=True, parents=True)
            status_array = np.memmap(self.status_array_path,
                                     dtype=np.bool,
                                     mode='w+',
                                     shape=(self.num_subsets,)
                                     )
            status_array[...] = False
            del status_array

        all_processed = self.is_processed
        if not all_processed:
            self._detect_faces()

        self._split_train_val(0,0.9)

        # if self.ring_type == "emonet_feature":
        #     self._prepare_emotion_retrieval()

    def _new_training_set(self, for_training=True):
        if for_training:
            im_transforms_train = create_image_augmenter(self.image_size, self.augmentation)

            # if self.ring_type == "emonet_feature":
            #     prefix = self.mode + "_train_"
            #     if self.ignore_invalid:
            #         prefix += "valid_only_"
            #     feature_label = 'emo_net_emo_feat_2'
            #     self._load_retrieval_arrays(prefix, feature_label)
            #     nn_indices = self.nn_indices_array
            #     nn_distances = self.nn_distances_array
            # else:
            #     nn_indices = None
            #     nn_distances = None

            return EmotioNet(self.image_path, self.train_dataframe_path, self.image_size, self.scale,
                             im_transforms_train,
                             ext = self.processed_ext,
                             au_type=self.au_type,
                             # ignore_invalid=self.ignore_invalid,
                             # ring_type=self.ring_type,
                             # ring_size=self.ring_size,
                             # load_emotion_feature=False,
                             # nn_indices_array=nn_indices,
                             # nn_distances_array= nn_distances
                             )

        return EmotioNet(self.image_path, self.train_dataframe_path, self.image_size, self.scale,
                         None,
                         ext=self.processed_ext,
                             au_type=self.au_type,
                         # ignore_invalid=self.ignore_invalid,
                         # ring_type=None,
                         # ring_size=None,
                         # load_emotion_feature=True
                         )

    def setup(self, stage=None):
        self.training_set = self._new_training_set()
        self.validation_set = EmotioNet(self.image_path, self.val_dataframe_path, self.image_size, self.scale,
                                        None,
                                        ext=self.processed_ext,
                                        au_type=self.au_type,
                                        # ignore_invalid=self.ignore_invalid,
                                        # ring_type=None,
                                        # ring_size=None
                                        )
        # self.test_set = None
        # self.test_dataframe_path = Path(self.output_dir) / "validation_representative_selection.csv"
        # self.test_set = EmotioNet(self.image_path, self.test_dataframe_path, self.image_size, self.scale,
        #                           None,
        #                           ext = self.processed_ext,
        #                           # ignore_invalid= self.ignore_invalid,
        #                           # ring_type=None,
        #                           # ring_size=None
        #                           )
        # if self.mode in ['all', 'manual']:
        #     # self.image_list += sorted(list((Path(self.path) / "Manually_Annotated").rglob(".jpg")))
        #     self.dataframe = pd.load_csv(self.path / "Manually_Annotated" / "Manually_Annotated.csv")
        # if self.mode in ['all', 'automatic']:
        #     # self.image_list += sorted(list((Path(self.path) / "Automatically_Annotated").rglob("*.jpg")))
        #     self.dataframe = pd.load_csv(
        #         self.path / "Automatically_Annotated" / "Automatically_annotated_file_list.csv")

    def train_dataloader(self):
        # if self.sampler == "uniform":
        sampler = None
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
        dl = DataLoader(self.training_set, shuffle=sampler is None, num_workers=self.num_workers,
                        batch_size=self.train_batch_size, drop_last=self.drop_last, sampler=sampler)
        return dl

    def val_dataloader(self):
        return DataLoader(self.validation_set, shuffle=False, num_workers=self.num_workers,
                          batch_size=self.val_batch_size, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, num_workers=self.num_workers,
                          batch_size=self.test_batch_size, drop_last=self.drop_last)

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


class EmotioNet(EmotionalImageDatasetBase):

    def __init__(self,
                 image_path,
                 dataframe_path,
                 image_size,
                 scale = 1.4,
                 transforms : imgaug.augmenters.Augmenter = None,
                 # use_gt_bb=True,
                 # ignore_invalid=False,
                 # ring_type=None,
                 # ring_size=None,
                 # load_emotion_feature=False,
                 nn_indices_array=None,
                 nn_distances_array=None,
                 ext=".jpg",
                 au_type = ActionUnitTypes.EMOTIONET12,
                 allow_missing_gt = None
                 ):
        self.dataframe_path = dataframe_path
        self.image_path = image_path
        self.df = pd.read_csv(dataframe_path)
        self.image_size = image_size
        # self.use_gt_bb = use_gt_bb
        # self.transforms = transforms or imgaug.augmenters.Identity()
        self.transforms = transforms or imgaug.augmenters.Resize((image_size, image_size))
        self.scale = scale
        self.landmark_normalizer = KeypointNormalization()
        # self.use_processed = True
        self.ext = ext
        self.au_type = au_type
        self.au_strs = ActionUnitTypes.AUtype2AUstring_list(self.au_type)

        self.allow_missing_gt = allow_missing_gt or au_type == ActionUnitTypes.EMOTIONET23

        # # 1 - mean occurence of action units (in case we want to use a balanced loss)
        # self.au_positive_weights = (1.-self.df[ActionUnitTypes.AUtype2AUstring_list(au_type)].to_numpy().astype(np.float64).mean(axis=0)).astype(np.float32)
        num_positive = self.df[ActionUnitTypes.AUtype2AUstring_list(au_type)].to_numpy().astype(np.float64).sum(axis=0)
        num_negative = len(self.df) - num_positive
        self.au_positive_weights = (num_negative / num_positive).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def _get_sample(self, index):
        try:
            im_rel_path = self.df.loc[index]["path"]
            im_file = Path(self.image_path) / im_rel_path
            im_file = im_file.parent / (im_file.stem + self.ext)
            input_img = imread(im_file)
        except Exception as e:
            # if the image is corrupted or missing (there might be a few :-/), find some other one
            while True:
                index += 1
                index = index % len(self)
                im_rel_path = self.df.loc[index]["path"]
                im_file = Path(self.image_path) / im_rel_path
                im_file = im_file.parent / (im_file.stem + self.ext)
                try:
                    input_img = imread(im_file)
                    success = True
                except Exception as e2:
                    success = False
                if success:
                    break

        AUs = self.df.loc[index, self.au_strs]
        AUs = np.array(AUs).astype(np.float64)

        if not self.allow_missing_gt and np.prod(np.logical_or(AUs == 1., AUs == 0.)) != 1:
            raise RuntimeError(f"It seems an AU label value in sample idx:{index}, {im_rel_path} is undefined. AUs: {AUs}")

        # input_img_shape = input_img.shape

        # if not self.use_processed:
        #     # Use AffectNet as is provided (their bounding boxes, and landmarks, no segmentation)
        #     old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
        #     size = int(old_size * self.scale)
        #     input_landmarks = np.array([float(f) for f in facial_landmarks.split(";")]).reshape(-1,2)
        #     img, landmark = bbpoint_warp(input_img, center, size, self.image_size, landmarks=input_landmarks)
        #     img *= 255.
        #
        #     # if not self.use_gt_bb:
        #     #     raise NotImplementedError()
        #     #     # landmark_type, landmark = load_landmark(
        #     #     #     self.path_prefix / self.landmark_list[index])
        #     landmark = landmark[np.newaxis, ...]
        #     seg_image = None
        # else:
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

        # if self.load_emotion_feature:
        #     emotion_path = Path(self.image_path).parent / "emotions" / im_rel_path
        #     emotion_path = emotion_path.parent / (emotion_path.stem + ".pkl")
        #     emotion_features, emotion_type = load_emotion(emotion_path)
        # else:
        #     emotion_features = None

        img, seg_image, landmark = self._augment(img, seg_image, landmark)

        sample = {
            "image": numpy_image_to_torch(img.astype(np.float32)),
            "path": str(im_file),
            "label": str(im_file.stem),
            "au": AUs,
            "au_pos_weights": self.au_positive_weights,
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
        # if self.ring_type is None or self.ring_size == 1:
        sample = self._get_sample(index)
        return sample


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



if __name__ == "__main__":
    # dm = EmotioNetDataModule(
    #          ## "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
             # "/ps/project_cifs/EmotionalFacialAnimation/data/emotionnet/emotioNet_challenge_files_server_challenge_1.2_aws_downloaded/",
             # "/is/cluster/work/rdanecek/data/emotionet/",
             # processed_subfolder="processed_2021_Aug_26_19-04-56",
             # scale=1.25,
             # ignore_invalid=True,
             # )
    import yaml
    augmenter = yaml.load(open(Path(__file__).parents[2] / "gdl_apps" / "EmotionRecognition" / "emodeca_conf" / "data" / "augmentations" / "default_with_resize.yaml"))["augmentation"]

    dm = EmotioNetDataModule(
             # "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
             "/ps/project_cifs/EmotionalFacialAnimation/data/emotionnet/emotioNet_challenge_files_server_challenge_1.2_aws_downloaded/",
             "/is/cluster/work/rdanecek/data/emotionet/",
             processed_subfolder="processed_2021_Aug_31_21-33-44",
             # processed_subfolder=None,
             scale=1.7,
             ignore_invalid=True,
             # image_size=512,
             image_size=224,
             bb_center_shift_x=0,
             bb_center_shift_y=-0.3,
            augmentation=augmenter,
            # au_type=ActionUnitTypes.EMOTIONET23
            )
    print(dm.num_subsets)
    dm.prepare_data()
    dm.setup()

    # training_set = dm.training_set
    # for i in range(len(training_set)):
    #     sample = training_set[i]
    #     # sample = training_set[0]
    #     training_set.visualize_sample(sample)
