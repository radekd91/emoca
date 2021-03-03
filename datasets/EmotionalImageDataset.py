import numpy as np
import torch
from skimage.io import imread
import imgaug
from torch.utils.data._utils.collate import default_collate

# from datasets.FaceVideoDataset import FaceVideoDataModule
from transforms.keypoints import KeypointScale, KeypointNormalization
from utils.FaceDetector import load_landmark
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

        self.labels_set = set(self.labels)
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




def numpy_image_to_torch(img : np.ndarray) -> torch.Tensor:
    img = img.transpose([2, 0, 1])
    return torch.from_numpy(img)


class EmotionalImageDataset(torch.utils.data.Dataset):

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

        self.labels_set = set(self.labels)
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

        res = self.transforms(image=img,
                              segmentation_maps=seg_image,
                              keypoints=landmark)
        if seg_image is not None and landmark is not None:
            img, seg_image, landmark = res
        elif seg_image is not None:
            img, seg_image = res
        elif landmark is not None:
            img, landmark = res
        else:
            img = res

        img = img.astype(np.float32) / 255.0

        if seg_image is not None:
            seg_image = np.squeeze(seg_image)[..., np.newaxis].astype(np.float32)

        if landmark is not None:
            landmark = np.squeeze(landmark)
            self.landmark_normalizer.set_scale(img.shape[0], img.shape[1])
            landmark = self.landmark_normalizer(landmark)

        sample = {
            "image": numpy_image_to_torch(img),
            "path": str(self.image_list[index]),
        }

        for key in self.annotations.keys():
            sample[key] = torch.tensor(self.annotations[key][index], dtype=torch.float32)
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


    def visualize_sample(self, sample):
        import matplotlib.pyplot as plt
        from utils.DecaUtils import tensor_vis_landmarks

        num_images = 1
        if 'mask' in sample.keys():
            num_images += 1

        if 'landmark' in sample.keys():
            num_images += 1

        K = sample["image"].shape[0]
        fig, axs = plt.subplots(K, num_images)

        def index_axis(k, i):
            if K==1:
                return axs[i]
            return axs[k,i]

        for k in range(K):
            i = 0
            index_axis(k, i).imshow(sample["image"][k, ...].numpy().transpose([1,2,0]))
            i += 1
            if 'landmark' in sample.keys():
                lmk_im = tensor_vis_landmarks(sample['image'][k:k+1, ...],
                                              self.landmark_normalizer.inv(sample['landmark'][k:k+1, ...]),
                                              isScale=False, rgb2bgr=False, scale_colors=True).numpy()[0]\
                    .transpose([1,2,0])
                index_axis(k, i).imshow(lmk_im)
                i += 1

            if 'mask' in sample.keys():
                index_axis(k, i).imshow(sample["mask"][k, ...].numpy().transpose([1,2,0]), cmap='gray')
                i += 1

        plt.show()

