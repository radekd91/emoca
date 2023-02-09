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


import numpy as np
import torch
from skimage.io import imread
import imgaug
from torch.utils.data._utils.collate import default_collate

# from gdl.datasets.FaceVideoDataset import FaceVideoDataModule
from gdl.transforms.keypoints import KeypointScale, KeypointNormalization
from gdl.utils.FaceDetector import load_landmark
from gdl.utils.image import numpy_image_to_torch
from .IO import load_segmentation, process_segmentation

# from timeit import default_timer as timer


### THIS CLASS IS DEPRECATED AND TO BE DELETED LATER
class EmotionalImageDatasetOld(torch.utils.data.Dataset):

    def __init__(self, image_list, annotations, labels, image_transforms,
                 path_prefix=None,
                 landmark_list=None,
                 landmark_transform=None,
                 segmentation_list=None,
                 segmentation_transform=None,
                 segmentation_discarded_lables=None,
                 K=None,
                 K_policy=None):
        self.image_list = image_list
        self.annotations = annotations
        if len(labels) != len(image_list):
            raise RuntimeError("There must be a label for every image")
        self.labels = labels
        self.image_transforms = image_transforms
        self.path_prefix = path_prefix
        if landmark_list is not None and len(landmark_list) != len(image_list):
            raise RuntimeError("There must be a landmark for every image")
        self.landmark_list = landmark_list
        self.landmark_transform = landmark_transform
        if segmentation_list is not None and len(segmentation_list) != len(segmentation_list):
            raise RuntimeError("There must be a segmentation for every image")
        self.segmentation_list = segmentation_list
        self.segmentation_transform = segmentation_transform
        self.segmentation_discarded_labels = segmentation_discarded_lables
        self.K = K # how many samples of one identity to get?
        self.K_policy = K_policy
        if self.K_policy is not None:
            if self.K_policy not in ['random', 'sequential']:
                raise ValueError(f"Invalid K policy {self.K_policy}")

        self.labels_set = sorted(list(set(self.labels)))
        self.label2index = {}

        for label in self.labels_set:
            self.label2index[label] = [i for i in range(len(self.labels))
                                       if self.labels[i] == label]

    def __len__(self):
        # return 10 #TODO: REMOVE TESTING HACK
        # return 50 #TODO: REMOVE TESTING HACK
        return len(self.image_list)

    def _get_sample(self, index):
        # sample_start = timer()
        try:
            path = self.image_list[index]
            if self.path_prefix is not None:
                path = self.path_prefix / path
            # start = timer()
            img = imread(path)
            # end = timer()
            # print(f"Image reading took {end-start} s.")
        except Exception as e:
            print(f"Failed to read '{path}'. "
                  f"File is probably corrupted. Rerun data processing")
            # return None
            raise e
        img = img.transpose([2, 0, 1]).astype(np.float32)/255.
        img_torch = torch.from_numpy(img)
        if self.image_transforms is not None:
            img_torch = self.image_transforms(img_torch)

        sample = {
            "image": img_torch,
            "path": str(self.image_list[index])
        }

        for key in self.annotations.keys():
            sample[key] = torch.tensor(self.annotations[key][index], dtype=torch.float32)

        if self.landmark_list is not None:
            # start = timer()
            landmark_type, landmark = load_landmark(
                self.path_prefix / self.landmark_list[index])
            # end = timer()
            # print(f"Landmark reading took {end - start} s.")

            landmark_torch = torch.from_numpy(landmark)

            if self.image_transforms is not None:
                if isinstance(self.landmark_transform, KeypointScale):
                    self.landmark_transform.set_scale(
                        img_torch.shape[1] / img.shape[1],
                        img_torch.shape[2] / img.shape[2])
                elif isinstance(self.landmark_transform, KeypointNormalization):
                    self.landmark_transform.set_scale(img.shape[1], img.shape[2])
                else:
                    raise ValueError(f"This transform is not supported for landmarks: "
                                     f"{type(self.landmark_transform)}")
                landmark_torch = self.landmark_transform(landmark_torch)

            sample["landmark"] = landmark_torch


        if self.segmentation_list is not None:
            if self.segmentation_list[index].stem != self.image_list[index].stem:
                raise RuntimeError(f"Name mismatch {self.segmentation_list[index].stem}"
                                   f" vs {self.image_list[index.stem]}")

            # start = timer()
            seg_image, seg_type = load_segmentation(
                self.path_prefix / self.segmentation_list[index])
            # end = timer()
            # print(f"Segmentation reading took {end - start} s.")

            # start = timer()
            seg_image = process_segmentation(
                seg_image, seg_type)
            # end = timer()
            # print(f"Segmentation processing took {end - start} s.")
            seg_image_torch = torch.from_numpy(seg_image)
            seg_image_torch = seg_image_torch.view(1, seg_image_torch.shape[0], seg_image_torch.shape[0])

            if self.image_transforms is not None:
                seg_image_torch = self.segmentation_transform(seg_image_torch)

            sample["mask"] = seg_image_torch

        # sample_end = timer()
        # print(f"Reading and processing a single sample took {sample_end-sample_start}s.")
        return sample


    def __getitem__(self, index):
        # start = timer()

        if self.K is None:
            return self._get_sample(index)
        # multiple identity samples in a single batch

        # retrieve indices of the same identity
        label = self.labels[index]
        label_indices = self.label2index[label]

        if self.K_policy == 'random':
            indices = np.arange(len(label_indices), dtype=np.int32)
            np.random.shuffle(indices)
            indices = indices[:self.K-1]
        elif self.K_policy == 'sequential':
            indices = []
            idx = label_indices.index(index) + 1
            while len(indices) != self.K-1:
                # if self.labels[idx] == label:
                indices += [label_indices[idx]]
                idx += 1
                idx = idx % len(label_indices)
        else:
            raise ValueError(f"Invalid K policy {self.K_policy}")

        batches = []
        batches += [self._get_sample(index)]
        for i in range(self.K-1):
            idx = indices[i]
            batches += [self._get_sample(idx)]

        # combined_batch = {}
        # for batch in batches:
        #     for key, value in batch.items():
        #         if key not in combined_batch.keys():
        #             combined_batch[key] = []
        #         combined_batch[key] += [value]

        combined_batch = default_collate(batches)

        # end = timer()
        # print(f"Reading sample {index} took {end - start}s")
        return combined_batch


class EmotionalImageDatasetBase(torch.utils.data.Dataset):


    def _augment(self, img, seg_image, landmark, input_img_shape=None):

        if self.transforms is not None:
            assert img.dtype == np.uint8
            # img = img.astype(np.float32) # TODO: debug this (do we get valid images when not used?)
            res = self.transforms(image=img,
                                  segmentation_maps=seg_image,
                                  keypoints=landmark)
            if seg_image is not None and landmark is not None:
                img, seg_image, landmark = res
            elif seg_image is not None:
                img, seg_image = res
            elif landmark is not None:
                img, _, landmark = res
            else:
                img = res

            
            assert img.dtype == np.uint8
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0
            
            assert img.dtype == np.float32


        if seg_image is not None:
            seg_image = np.squeeze(seg_image)[..., np.newaxis].astype(np.float32)

        if landmark is not None:
            landmark = np.squeeze(landmark)
            if isinstance(self.landmark_normalizer, KeypointScale):
                self.landmark_normalizer.set_scale(
                    img.shape[0] / input_img_shape[0],
                    img.shape[1] / input_img_shape[1])
            elif isinstance(self.landmark_normalizer, KeypointNormalization):
                self.landmark_normalizer.set_scale(img.shape[0], img.shape[1])
                # self.landmark_normalizer.set_scale(input_img_shape[0], input_img_shape[1])
            else:
                raise ValueError(f"Unsupported landmark normalizer type: {type(self.landmark_normalizer)}")
            landmark = self.landmark_normalizer(landmark)

        return img, seg_image, landmark



    def visualize_sample(self, sample):
        if isinstance(sample, int):
            sample = self[sample]

        import matplotlib.pyplot as plt
        num_images = 1
        if 'mask' in sample.keys():
            num_images += 1

        if 'landmark' in sample.keys():
            num_images += 1
        if 'landmark_mediapipe' in sample.keys():
            num_images += 1

        if len(sample["image"].shape) >= 4:
            K = sample["image"].shape[0]
            fig, axs = plt.subplots(K, num_images)
        else:
            K = None
            fig, axs = plt.subplots(1, num_images)

        # if K is not None:
        for k in range(K or 1):
            self._plot(axs, K, k, sample)
        plt.show()

    def _plot(self, axs, K, k, sample):

        from gdl.utils.DecaUtils import tensor_vis_landmarks

        def index_axis(i, k):
            if K==1 or K is None:
                return axs[i]
            return axs[k,i]

        im = sample["image"][k, ...] if K is not None else sample["image"]
        im_expanded = im[np.newaxis, ...]


        i = 0
        index_axis(i, k).imshow(im.numpy().transpose([1, 2, 0]))
        i += 1
        if 'landmark' in sample.keys():
            lmk = sample["landmark"][k, ...] if K is not None else sample["landmark"]
            lmk_expanded = lmk[np.newaxis, ...]
            lmk_im = tensor_vis_landmarks(im_expanded,
                                          self.landmark_normalizer.inv(lmk_expanded),
                                          isScale=False, rgb2bgr=False, scale_colors=True).numpy()[0] \
                .transpose([1, 2, 0])
            index_axis(i, k).imshow(lmk_im)
            i += 1

        if 'landmark_mediapipe' in sample.keys():
            lmk = sample["landmark_mediapipe"][k, ...] if K is not None else sample["landmark_mediapipe"]
            lmk_expanded = lmk[np.newaxis, ...]
            lmk_im = tensor_vis_landmarks(im_expanded,
                                          self.landmark_normalizer.inv(lmk_expanded),
                                          isScale=False, rgb2bgr=False, scale_colors=True).numpy()[0] \
                .transpose([1, 2, 0])
            index_axis(i, k).imshow(lmk_im)
            i += 1

        if 'mask' in sample.keys():
            mask = sample["mask"][k, ...] if K is not None else sample["mask"]
            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]
            index_axis(i, k).imshow(mask.numpy().transpose([1, 2, 0]).squeeze(), cmap='gray')
            i += 1


        if 'path' in sample.keys() and 'label' in sample.keys():
            if K is None:
                print(f"Path = {sample['path']}")
                print(f"Label = {sample['label']}")
            else:
                print(f"Path {k} = {sample['path'][k]}")
                print(f"Label {k} = {sample['label'][k]}")



class EmotionalImageDataset(EmotionalImageDatasetBase):

    def __init__(self,
                 image_list : list,
                 annotations,
                 labels,
                 transforms : imgaug.augmenters.Augmenter,
                 path_prefix=None,
                 landmark_list=None,
                 # landmark_transform=None,
                 segmentation_list=None,
                 # segmentation_transform=None,
                 segmentation_discarded_lables=None,
                 K=None,
                 K_policy=None
                 ):
        self.image_list = image_list
        self.annotations = annotations
        for key in annotations:
            if len(annotations[key]) != len(image_list):
                raise RuntimeError("There must be an annotation of each type for every image but "
                                   f"this is not the case for '{key}'")
        if len(labels) != len(image_list):
            raise RuntimeError("There must be a label for every image")
        self.labels = labels
        self.transforms = transforms
        self.path_prefix = path_prefix
        if landmark_list is not None and len(landmark_list) != len(image_list):
            raise RuntimeError("There must be a landmark for every image")
        self.landmark_list = landmark_list
        # self.landmark_transform = landmark_transform
        if segmentation_list is not None and len(segmentation_list) != len(segmentation_list):
            raise RuntimeError("There must be a segmentation for every image")
        self.segmentation_list = segmentation_list
        self.landmark_normalizer = KeypointNormalization()
        # self.segmentation_transform = segmentation_transform
        self.segmentation_discarded_labels = segmentation_discarded_lables
        self.K = K  # how many samples of one identity to get?
        self.K_policy = K_policy
        if self.K_policy is not None:
            if self.K_policy not in ['random', 'sequential']:
                raise ValueError(f"Invalid K policy {self.K_policy}")

        self.labels_set = sorted(list(set(self.labels)))
        self.label2index = {}

        self.include_strings_samples = False

        for label in self.labels_set:
            self.label2index[label] = [i for i in range(len(self.labels))
                                       if self.labels[i] == label]

    def __len__(self):
        # return 10 #TODO: REMOVE TESTING HACK
        # return 50 #TODO: REMOVE TESTING HACK
        return len(self.image_list)

    def _get_sample(self, index):
        # sample_start = timer()
        try:
            path = self.image_list[index]
            if self.path_prefix is not None:
                path = self.path_prefix / path
            # start = timer()
            img = imread(path)
            input_img_shape = img.shape
            # end = timer()
            # print(f"Image reading took {end-start} s.")
        except Exception as e:
            print(f"Failed to read '{path}'. "
                  f"File is probably corrupted. Rerun data processing")
            # return None
            raise e

        if self.landmark_list is not None:
            # start = timer()
            landmark_type, landmark = load_landmark(
                self.path_prefix / self.landmark_list[index])
            landmark = landmark[np.newaxis, ...]
            # end = timer()
            # print(f"Landmark reading took {end - start} s.")
        else:
            landmark = None


        if self.segmentation_list is not None:
            if self.segmentation_list[index].stem != self.image_list[index].stem:
                raise RuntimeError(f"Name mismatch {self.segmentation_list[index].stem}"
                                   f" vs {self.image_list[index.stem]}")

            # start = timer()
            seg_image, seg_type = load_segmentation(
                self.path_prefix / self.segmentation_list[index])
            seg_image = seg_image[np.newaxis, :,:,np.newaxis]
            # end = timer()
            # print(f"Segmentation reading took {end - start} s.")

            # start = timer()
            seg_image = process_segmentation(
                seg_image, seg_type).astype(np.uint8)
            # end = timer()
            # print(f"Segmentation processing took {end - start} s.")
        else:
            seg_image = None

        img, seg_image, landmark = self._augment(img, seg_image, landmark, input_img_shape)

        sample = {
            "image": numpy_image_to_torch(img)
        }

        if self.include_strings_samples:
            sample["path"] = str(self.image_list[index])
            sample["label"] = str(self.labels[index])

        for key in self.annotations.keys():
            annotation = self.annotations[key][index]
            if isinstance(annotation, int):
                annotation = [annotation]

            if annotation is None or len(annotation) == 0:
                if key == 'au8':
                    sample[key] = torch.tensor([float('nan')]*8)
                elif key == 'expr7':
                    sample[key] = torch.tensor([float('nan')]*2)[0:1]
                elif key == 'va':
                    sample[key] = torch.tensor([float('nan')]*2)
                else:
                    raise RuntimeError(f"Unknown annotation type: '{key}'")
                # print(f"{key}, size {sample[key].size()}")
                if len(sample[key].size()) == 0:
                    print(f"[WARNING] Annotation '{key}' is empty for some reason and will be invalidated")
                continue
            sample[key] = torch.tensor(annotation, dtype=torch.float32)
            if len(sample[key].size()) == 0:
                print(f"[WARNING] Annotation '{key}' is empty for some reason (even though it was not None and will be invalidated")
                print("annotation value: ")
                print(annotation)

        if landmark is not None:
            sample["landmark"] = torch.from_numpy(landmark)
        if seg_image is not None:
            sample["mask"] = numpy_image_to_torch(seg_image)

        # sample_end = timer()
        # print(f"Reading and processing a single sample took {sample_end-sample_start}s.")
        return sample


    def __getitem__(self, index):
        # start = timer()
        if self.K is None:
            return self._get_sample(index)
        # multiple identity samples in a single batch

        # retrieve indices of the same identity
        label = self.labels[index]
        label_indices = self.label2index[label]

        if self.K_policy == 'random':
            picked_label_indices = np.arange(len(label_indices), dtype=np.int32)
            # print("Size of label_indices:")
            # print(len(label_indices))
            np.random.shuffle(picked_label_indices)
            if len(label_indices) < self.K-1:
                print(f"[WARNING]. Label '{label}' only has {len(label_indices)} samples which is less than {self.K}. S"
                      f"ome samples will be duplicated")
                picked_label_indices = np.concatenate(self.K*[picked_label_indices], axis=0)

            picked_label_indices = picked_label_indices[:self.K-1]
            indices = [label_indices[i] for i in picked_label_indices]
        elif self.K_policy == 'sequential':
            indices = []
            idx = label_indices.index(index) + 1
            idx = idx % len(label_indices)
            while len(indices) != self.K-1:
                # if self.labels[idx] == label:
                indices += [label_indices[idx]]
                idx += 1
                idx = idx % len(label_indices)
        else:
            raise ValueError(f"Invalid K policy {self.K_policy}")

        batches = []
        batches += [self._get_sample(index)]
        for i in range(self.K-1):
            # idx = indices[i]
            idx = indices[i]
            batches += [self._get_sample(idx)]

        # combined_batch = {}
        # for batch in batches:
        #     for key, value in batch.items():
        #         if key not in combined_batch.keys():
        #             combined_batch[key] = []
        #         combined_batch[key] += [value]

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


