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

Parts of the code were adapted from the original DECA release: 
https://github.com/YadiraF/DECA/ 
"""


import os, sys
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from pytorch_lightning import LightningDataModule

# from . import detectors


class DatasetSplitter(Dataset):

    def __init__(self, dataset, split_ratio, split_style='random', training=True):
        self.dataset = dataset
        idxs = np.arange(len(self.dataset), dtype=np.int32)
        if split_style == 'random':
            np.random.seed(0)
            np.random.shuffle(idxs)
        elif split_style == 'sequential':
            pass
        else:
            raise ValueError(f"Invalid split style {split_style}")

        if split_ratio < 0 or split_ratio > 1:
            raise ValueError(f"Invalid split ratio {split_ratio}")

        split_idx = int(idxs.size*split_ratio)
        self.idx_train = idxs[:split_idx]
        self.idx_val = idxs[split_idx:]
        self.training = training
        self.split_ratio = split_ratio
        self.split_style = split_style

    def complementary_set(self):
        dataset = DatasetSplitter(self.dataset, self.split_ratio, self.split_style, training=not self.training )
        dataset.idx_train = self.idx_train
        dataset.idx_val = self.idx_val
        return dataset

    def __len__(self):
        # return 100
        if self.training:
            return len(self.idx_train)
        return len(self.idx_val)

    def __getitem__(self, item):
        if self.training:
            return self.dataset[ self.idx_train[item]]
        return self.dataset[ self.idx_val[item]]
        

class DecaDataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.split_ratio = config.data.split_ratio
        self.split_style = config.data.split_style

    def train_sampler(self):
        return None

    @property
    def train_batch_size(self):
        return self.config.learning.batch_size_train

    @property
    def val_batch_size(self):
        return self.config.learning.batch_size_val

    @property
    def test_batch_size(self):
        return self.config.learning.batch_size_test

    @property
    def num_workers(self):
        return self.config.data.num_workers

    def setup(self, stage=None):
        dataset = build_dataset(self.config, self.config.data.training_datasets, concat=True)
        self.training_set = DatasetSplitter(dataset, self.split_ratio, self.split_style)
        self.validation_set = []
        if self.split_ratio < 1.:
            self.validation_set += [self.training_set.complementary_set(), ]
        self.validation_set += build_dataset(self.config, self.config.data.validation_datasets, concat=False)
        self.test_set = build_dataset(self.config, self.config.data.testing_datasets, concat=False)

    def train_dataloader(self, *args, **kwargs):
        train_loader = DataLoader(self.training_set,
                                  batch_size=self.config.learning.batch_size_train, shuffle=True,
                                  num_workers=self.config.data.num_workers,
                                  pin_memory=True
                                  )
                                  # drop_last=drop_last)
        # print('---- data length: ', len(train_dataset))
        return train_loader

    def val_dataloader(self, *args, **kwargs):
        val_loaders = [DataLoader(val_set,
                                  batch_size=self.config.learning.batch_size_val, shuffle=False,
                                  num_workers=self.config.data.num_workers,
                                  pin_memory=True
                                  ) for val_set in self.validation_set]
        return val_loaders

    def test_dataloader(self):
        test_loaders = [DataLoader(test_set,
                                  batch_size=self.config.learning.batch_size_test, shuffle=False,
                                  num_workers=self.config.data.num_workers,
                                  pin_memory=True
                                   ) for test_set in self.test_set]
        return test_loaders


def build_dataset(config, dataset_list=None, concat=True):
    dataset_list = dataset_list or config.data.datasets.copy()
    data_list = []
    if 'vox1' in dataset_list:
        data_list.append(
            VoxelDataset(K=config.learning.train_K, image_size=config.model.image_size, scale=[config.data.scale_min, config.data.scale_max],
                         n_train=config.data.n_train,
                         path=config.data.path
                         )
                         # isSingle=config.isSingle)
        )
    if 'vox2' in dataset_list:
        data_list.append(VoxelDataset(dataname='vox2', K=config.learning.train_K, image_size=config.model.image_size,
                                      scale=[config.data.scale_min, config.data.scale_max], n_train=config.data.n_train,
                                      path=config.data.path)
                                      # isSingle=config.isSingle)
                         )
    if 'vggface2' in dataset_list:
        data_list.append(
            VGGFace2Dataset(K=config.learning.train_K, image_size=config.model.image_size, scale=[config.data.scale_min, config.data.scale_max],
                            trans_scale=config.data.trans_scale,
                            path=config.data.path)
            # , isSingle=config.isSingle)
        )
    if 'vggface2hq' in dataset_list:
        data_list.append(
            VGGFace2HQDataset(K=config.learning.train_K, image_size=config.model.image_size, scale=[config.data.scale_min, config.data.scale_max],
                              trans_scale=config.data.trans_scale,
                              path=config.data.path)) #, isSingle=config.isSingle))
    if 'ethnicity' in dataset_list:
        data_list.append(
            EthnicityDataset(K=config.learning.train_K, image_size=config.model.image_size, scale=[config.data.scale_min, config.data.scale_max],
                             trans_scale=config.data.trans_scale,
                             path=config.data.path
                             )
        )#, isSingle=config.isSingle))
    if 'coco' in dataset_list:
        data_list.append(COCODataset(image_size=config.model.image_size, scale=[config.data.scale_min, config.data.scale_max],
                                     trans_scale=config.data.trans_scale))
    if 'celebahq' in dataset_list:
        data_list.append(CelebAHQDataset(image_size=config.model.image_size, scale=[config.data.scale_min, config.data.scale_max],
                                         trans_scale=config.data.trans_scale))
    if 'now_eval' in dataset_list:
        data_list.append(NoWVal(path=config.data.path))
    if 'aflw2000' in dataset_list:
        data_list.append(AFLW2000())

    # if data_set_name == 'now-val':
    if 'now-val' in dataset_list:
        now = NoWVal(
            # data_path='/ps/scratch/face2d3d/texture_in_the_wild_code/NoW_validation/image_paths_ring_6_elements.npy',
            ring_elements=config.learning.val_K,
            # ring_elements=1,
            crop_size=config.model.image_size,
            path = config.data.path
        )
        data_list.append(now)

    # if data_set_name == 'ffhq-val':
    if 'ffhq-val' in dataset_list:
        ffhq = FFHQ_val(
            ring_elements=config.learning.val_K,
            # ring_elements=1,
            crop_size=config.model.image_size,
            path = config.data.path
        )
        data_list.append(ffhq)

    # if data_set_name == 'gif-val':
    if 'gif-val' in dataset_list:
        gif = GIF_val(
            # data_path='/is/cluster/scratch/partha/gif_eval_data_toheiven/gif_loadinglist.npy',
            ring_elements=config.learning.val_K,
            # ring_elements=1,
            crop_size=config.model.image_size,
            path = config.data.path
        )
        data_list.append(gif)

    # if data_set_name == 'now-test':
    if 'now-test' in dataset_list:
        now = NoWTest(
            # data_path='/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy',
            ring_elements=config.learning.test_K,
            # ring_elements=1,
            crop_size=config.model.image_size,
            path=config.data.path,
        )
        data_list.append(now)

    if 'papers-val' in dataset_list:
        papers = PaperVal(
            # data_path='/ps/scratch/face2d3d/ringnetpp/eccv(haven)/test_data/papers/papers_ring_6_elements_loadinglist.npy',
            ring_elements=config.learning.val_K,
            # ring_elements=1,
            crop_size=config.model.image_size,
            path=config.data.path
        )
        data_list.append(papers)

    # if data_set_name == 'celeb-val':
    if 'celeb-val' in dataset_list:
        celeb = CelebVal(
            # data_path='/ps/scratch/face2d3d/texture_in_the_wild_code/celeb_ring_6_elements_loadinglist.npy',
            ring_elements=config.learning.val_K,
            # ring_elements=1,
            crop_size=config.model.image_size,
            path=config.data.path
        )
        data_list.append(celeb)
    if concat:
        train_dataset = ConcatDataset(data_list)
        return train_dataset
    return data_list


def build_dataloader(config, is_train=True):
    train_dataset = build_dataset(config)
    if is_train:
        # drop_last = True
        shuffle = True
    else:
        # drop_last = False
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              )
                              # drop_last=drop_last)
    # print('---- data length: ', len(train_dataset))
    return train_dataset, train_loader


'''
images and keypoints: nomalized to [-1,1]
'''


class VoxelDataset(Dataset):
    def __init__(self, K, image_size, scale, trans_scale=0, dataname='vox2', n_train=100000,
                 path = None,
                 isTemporal=False,
                 isEval=False): #, isSingle=False):
        self.path = path or '/ps/scratch/face2d3d'
        self.K = K
        self.image_size = image_size
        self.drop_last = False
        if dataname == 'vox1':
            self.kpt_suffix = '.txt'
            self.imagefolder = '/ps/project/face2d3d/VoxCeleb/vox1/dev/images_cropped'
            self.kptfolder = '/ps/scratch/yfeng/Data/VoxCeleb/vox1/landmark_2d'

            self.face_dict = {}
            for person_id in sorted(os.listdir(self.kptfolder)):
                for video_id in os.listdir(os.path.join(self.kptfolder, person_id)):
                    for face_id in os.listdir(os.path.join(self.kptfolder, person_id, video_id)):
                        if 'txt' in face_id:
                            continue
                        key = person_id + '/' + video_id + '/' + face_id
                        # if key not in self.face_dict.keys():
                        #     self.face_dict[key] = []
                        name_list = os.listdir(os.path.join(self.kptfolder, person_id, video_id, face_id))
                        name_list = [name.split['.'][0] for name in name_list]
                        if len(name_list) < self.K:
                            continue
                        self.face_dict[key] = sorted(name_list)

        elif dataname == 'vox2':
            # clean version: filter out images with bad lanmark labels, may lack extreme pose example
            self.kpt_suffix = '.npy'
            self.imagefolder = self.path + '/VoxCeleb/vox2/dev/images_cropped_full_height'
            self.kptfolder =  self.path + '/vox2_best_clips_annotated_torch7'
            self.segfolder =  self.path + '/texture_in_the_wild_code/vox2_best_clips_cropped_frames_seg/test_crop_size_400_batch/'

            cleanlist_path =  self.path + '/texture_in_the_wild_code/VGGFace2_cleaning_codes/vox2_best_clips_info_max_normal_50_images_loadinglist.npy'
            cleanlist = np.load(cleanlist_path, allow_pickle=True)
            self.face_dict = {}
            for line in cleanlist:
                person_id, video_id, face_id, name = line.split('/')
                key = person_id + '/' + video_id + '/' + face_id
                if key not in self.face_dict.keys():
                    self.face_dict[key] = []
                else:
                    self.face_dict[key].append(name)
            # filter face
            keys = list(self.face_dict.keys())
            for key in keys:
                if len(self.face_dict[key]) < self.K:
                    del self.face_dict[key]

        self.face_list = list(self.face_dict.keys())
        n_train = n_train if n_train < len(self.face_list) else len(self.face_list)
        self.face_list = list(self.face_dict.keys())[:n_train]
        if isEval:
            self.face_list = list(self.face_dict.keys())[:n_train][-100:]
        self.isTemporal = isTemporal
        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # [dx, dy]
        # self.isSingle = isSingle
        # if isSingle:
        #     self.K = 1
        self.include_image_path = True
        # self.include_image_path = False

    def __len__(self):
        leng = len(self.face_list)
        # leng = 11
        if self.drop_last:
            return leng - leng % self.drop_last
        return len(self.face_list)

    # def __len__(self):
    #     return 11
        # return len(self.face_list)

    def __getitem__(self, idx):
        key = self.face_list[idx]
        person_id, video_id, face_id = key.split('/')
        name_list = self.face_dict[key]
        ind = np.random.randint(low=0, high=len(name_list))

        images_list = []
        kpt_list = []
        fullname_list = []
        mask_list = []
        image_path_list = []
        if self.isTemporal:
            random_start = np.random.randint(low=0, high=len(name_list) - self.K)
            sample_list = range(random_start, random_start + self.K)
        else:
            sample_list = np.array((np.random.randint(low=0, high=len(name_list), size=self.K)))

        for i in sample_list:
            name = name_list[i]
            image_path = (os.path.join(self.imagefolder, person_id, video_id, face_id, name + '.png'))
            kpt_path = (os.path.join(self.kptfolder, person_id, video_id, face_id, name + self.kpt_suffix))
            seg_path = (os.path.join(self.segfolder, person_id, video_id, face_id, name + '.npy'))

            image = imread(image_path) / 255.
            kpt = np.load(kpt_path)[:, :2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### crop information
            tform = self.crop(image, kpt)
            ## crop
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params,
                                 np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

            images_list.append(cropped_image.transpose(2, 0, 1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)
            image_path_list.append(image_path)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)  # K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype=torch.float32)  # K,224,224,3

        # if self.isSingle:
        #     images_array = images_array.squeeze()
        #     kpt_array = kpt_array.squeeze()
        #     mask_array = mask_array.squeeze()

        data_dict = {
            'image': images_array,
            'landmark': kpt_array[:,:,:2],
            'mask': mask_array
        }

        if self.include_image_path:
            data_dict['path'] = image_path_list

        # print("VoxelDataset")
        # print(data_dict['image'].shape)
        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, 0]);
        right = np.max(kpt[:, 0]);
        top = np.min(kpt[:, 1]);
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno > 0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


class VGGFace2Dataset(Dataset):
    def __init__(self, K, image_size, scale, path=None, trans_scale=0, isTemporal=False, isEval=False,):
                 # isSingle=False):
        '''
        K must be less than 6
        '''
        self.path = path or '/ps/scratch/face2d3d'
        self.image_size = image_size
        self.imagefolder =  self.path + '/train'
        self.kptfolder =  self.path + '/train_annotated_torch7'
        self.segfolder =  self.path + '/texture_in_the_wild_code/VGGFace2_seg/test_crop_size_400_batch'
        # hq:
        # datafile =  self.path + '/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_bbx_size_bigger_than_400_train_list_max_normal_100_ring_5_1_serial.npy'
        datafile =  self.path + '/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_train_list_max_normal_100_ring_5_1_serial.npy'
        self.data_lines = np.load(datafile).astype('str')
        if K == 'max':
            self.K = self.data_lines.shape[1] -1 # WARNING THE LAST COLUMN OF DATA LINES IS A DIFFERENT PERSON! (RingNet residual?)
        else:
            self.K = K
        self.isTemporal = isTemporal
        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # [dx, dy]
        # self.isSingle = isSingle
        # if isSingle:
        #     self.K = 1
        self.include_image_path = True
        self.drop_last = False
        # self.include_image_path = False

    # def __len__(self):
    #     return len(self.data_lines)
    def __len__(self):
        leng = len(self.data_lines)
        # leng = 11
        if self.drop_last:
            return leng - leng % self.drop_last
        return len(self.data_lines)

    def __getitem__(self, idx):
        images_list = []
        kpt_list = []
        mask_list = []
        image_path_list = []

        random_ind = np.random.permutation(5)[:self.K]
        # random_ind = np.random.permutation(6)[:self.K]
        # random_ind = np.arange(self.data_lines.shape[1])[:self.K]
        for i in random_ind:
            name = self.data_lines[idx, i]
            image_path = os.path.join(self.imagefolder, name + '.jpg')
            seg_path = os.path.join(self.segfolder, name + '.npy')
            kpt_path = os.path.join(self.kptfolder, name + '.npy')

            image = imread(image_path) / 255.
            kpt = np.load(kpt_path)[:, :2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### crop information
            tform = self.crop(image, kpt)
            ## crop
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params,
                                 np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

            images_list.append(cropped_image.transpose(2, 0, 1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)
            image_path_list.append(image_path)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)  # K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype=torch.float32)  # K,224,224,3

        # if self.isSingle:
        #     images_array = images_array.squeeze()
        #     kpt_array = kpt_array.squeeze()
        #     mask_array = mask_array.squeeze()
        data_dict = {
            'image': images_array,
            'landmark': kpt_array[:,:,:2],
            'mask': mask_array
        }

        if self.include_image_path:
            data_dict['path'] = image_path_list

        return data_dict


    def crop(self, image, kpt):
        left = np.min(kpt[:, 0]);
        right = np.max(kpt[:, 0]);
        top = np.min(kpt[:, 1]);
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno > 0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


class VGGFace2HQDataset(Dataset):
    def __init__(self, K, image_size, scale, path=None, trans_scale=0, isTemporal=False, isEval=False): #, isSingle=False):
        '''
        K must be less than 6
        '''
        self.path = path or '/ps/scratch/face2d3d'

        self.image_size = image_size
        self.imagefolder =  self.path + '/train'
        self.kptfolder =  self.path + '/train_annotated_torch7'
        self.segfolder =  self.path + '/texture_in_the_wild_code/VGGFace2_seg/test_crop_size_400_batch'
        # hq:
        # datafile =  self.path + '/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_bbx_size_bigger_than_400_train_list_max_normal_100_ring_5_1_serial.npy'
        datafile =  self.path + '/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_bbx_size_bigger_than_400_train_list_max_normal_100_ring_5_1_serial.npy'
        self.data_lines = np.load(datafile).astype('str')
        if K == 'max':
            self.K = self.data_lines.shape[1]
        else:
            self.K = K
        self.isTemporal = isTemporal
        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # [dx, dy]
        # self.isSingle = isSingle
        # if isSingle:
        #     self.K = 1
        self.include_image_path = True
        # self.include_image_path = False
        self.drop_last = False


    def __len__(self):
        # leng = len(self.data_lines)
        leng = 11
        if self.drop_last:
            return leng - leng % self.drop_last
        return len(self.data_lines)


    def __getitem__(self, idx):
        images_list = []
        kpt_list = []
        mask_list = []
        image_path_list = []

        for i in range(self.K):
            name = self.data_lines[idx, i]
            image_path = os.path.join(self.imagefolder, name + '.jpg')
            seg_path = os.path.join(self.segfolder, name + '.npy')
            kpt_path = os.path.join(self.kptfolder, name + '.npy')

            image = imread(image_path) / 255.
            kpt = np.load(kpt_path)[:, :2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### crop information
            tform = self.crop(image, kpt)
            ## crop
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params,
                                 np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

            images_list.append(cropped_image.transpose(2, 0, 1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)
            image_path_list.append(image_path)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)  # K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype=torch.float32)  # K,224,224,3

        # if self.isSingle:
        #     images_array = images_array.squeeze()
        #     kpt_array = kpt_array.squeeze()
        #     mask_array = mask_array.squeeze()

        data_dict = {
            'image': images_array,
            'landmark': kpt_array[:,:,:2],
            'mask': mask_array
        }

        if self.include_image_path:
            data_dict['path'] = image_path_list
        # print("VGGFace2 ")
        # print(data_dict['image'].shape)
        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, 0]);
        right = np.max(kpt[:, 0]);
        top = np.min(kpt[:, 1]);
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno > 0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


class EthnicityDataset(Dataset):
    def __init__(self, K, image_size, scale, path=None, trans_scale=0, isTemporal=False, isEval=False): #, isSingle=False):
        '''
        K must be less than 6
        '''
        self.path = path or '/ps/scratch/face2d3d'
        self.image_size = image_size
        self.imagefolder =  self.path + '/train'
        self.kptfolder =  self.path + '/train_annotated_torch7/'
        self.segfolder =  self.path + '/texture_in_the_wild_code/VGGFace2_seg/test_crop_size_400_batch/'
        # hq:
        # datafile =  self.path + '/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_bbx_size_bigger_than_400_train_list_max_normal_100_ring_5_1_serial.npy'
        datafile =  self.path + '/texture_in_the_wild_code/VGGFace2_cleaning_codes/ringnetpp_training_lists/second_cleaning/vggface2_and_race_per_7000_african_asian_2d_train_list_max_normal_100_ring_5_1_serial.npy'
        self.data_lines = np.load(datafile).astype('str')
        if K == 'max':
            self.K = self.data_lines.shape[1]
        else:
            self.K = K
        self.isTemporal = isTemporal
        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # [dx, dy]
        self.include_image_path = True
        # self.include_image_path = False
        # self.isSingle = isSingle
        # if isSingle:
        #     self.K = 1
        self.drop_last = False

    def __len__(self):
        leng = len(self.data_lines)
        if self.drop_last:
            return leng - leng % self.drop_last
        return len(self.data_lines)

    def __getitem__(self, idx):
        images_list = []
        kpt_list = []
        mask_list = []
        image_path_list = []
        for i in range(self.K):
            name = self.data_lines[idx, i]
            if name[0] == 'n':
                self.imagefolder =  self.path + '/train/'
                self.kptfolder =  self.path + '/train_annotated_torch7/'
                self.segfolder =  self.path + '/texture_in_the_wild_code/VGGFace2_seg/test_crop_size_400_batch/'
            elif name[0] == 'A':
                self.imagefolder =  self.path + '/race_per_7000/'
                self.kptfolder =  self.path + '/race_per_7000_annotated_torch7_new/'
                self.segfolder =  self.path + '/texture_in_the_wild_code/race7000_seg/test_crop_size_400_batch/'

            image_path = os.path.join(self.imagefolder, name + '.jpg')
            seg_path = os.path.join(self.segfolder, name + '.npy')
            kpt_path = os.path.join(self.kptfolder, name + '.npy')

            image = imread(image_path) / 255.
            kpt = np.load(kpt_path)[:, :2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### crop information
            tform = self.crop(image, kpt)
            ## crop
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params,
                                 np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

            images_list.append(cropped_image.transpose(2, 0, 1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)
            image_path_list.append(image_path)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)  # K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype=torch.float32)  # K,224,224,3

        # if self.isSingle:
        #     images_array = images_array.squeeze()
        #     kpt_array = kpt_array.squeeze()
        #     mask_array = mask_array.squeeze()

        data_dict = {
            'image': images_array,
            'landmark': kpt_array[:,:,:2],
            'mask': mask_array,
        }

        if self.include_image_path:
            data_dict['path'] = image_path_list

        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno > 0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


class COCODataset(Dataset):
    def __init__(self, image_size, scale, trans_scale=0, isEval=False):
        '''
        # 53877 faces
        K must be less than 6
        '''
        raise NotImplementedError("This hasn't been cleaned up yet")
        self.image_size = image_size
        self.imagefolder = '/ps/scratch/yfeng/Data/COCO/raw/train2017'
        self.kptfolder = '/ps/scratch/yfeng/Data/COCO/face/train2017_kpt'

        self.kptpath_list = os.listdir(self.kptfolder)
        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # 0.5?

    def __len__(self):
        return len(self.kptpath_list)

    def __getitem__(self, idx):
        while (100):
            kptname = self.kptpath_list[idx]
            name = kptname.split('_')[0]
            image_path = os.path.join(self.imagefolder, name + '.jpg')
            kpt_path = os.path.join(self.kptfolder, kptname)

            kpt = np.loadtxt(kpt_path)[:, :2]
            left = np.min(kpt[:, 0]);
            right = np.max(kpt[:, 0]);
            top = np.min(kpt[:, 1]);
            bottom = np.max(kpt[:, 1])
            if (right - left) < 10 or (bottom - top) < 10:
                idx = np.random.randint(low=0, high=len(self.kptpath_list))
                continue

            image = imread(image_path) / 255.
            if len(image.shape) < 3:
                image = np.tile(image[:, :, None], 3)
            ### crop information
            tform = self.crop(image, kpt)
            ## crop
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params,
                                 np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

            ###
            images_array = torch.from_numpy(cropped_image.transpose(2, 0, 1)).type(dtype=torch.float32)  # 224,224,3
            kpt_array = torch.from_numpy(cropped_kpt).type(dtype=torch.float32)  # 224,224,3

            data_dict = {
                'image': images_array * 2. - 1,
                'landmark': kpt_array[:,:,:2],
                # 'mask': mask_array
            }

            return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, 0]);
        right = np.max(kpt[:, 0]);
        top = np.min(kpt[:, 1]);
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]

        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno > 0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


class CelebAHQDataset(Dataset):
    def __init__(self, image_size, scale, trans_scale=0, isEval=False):
        '''
        # 53877 faces
        K must be less than 6
        '''
        raise NotImplementedError("this hasn't been cleaned up yet.")
        self.image_size = image_size
        self.imagefolder = '/ps/project/face2d3d/faceHQ_100K/celebA-HQ/celebahq_resized_256'
        self.kptfolder = '/ps/project/face2d3d/faceHQ_100K/celebA-HQ/celebahq_resized_256_torch'

        self.kptpath_list = os.listdir(self.kptfolder)
        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # 0.5?

    def __len__(self):
        return len(self.kptpath_list)

    def __getitem__(self, idx):
        while (100):
            kptname = self.kptpath_list[idx]
            name = kptname.split('.')[0]
            image_path = os.path.join(self.imagefolder, name + '.png')
            kpt_path = os.path.join(self.kptfolder, kptname)
            kpt = np.load(kpt_path, allow_pickle=True)
            if len(kpt.shape) != 2:
                idx = np.random.randint(low=0, high=len(self.kptpath_list))
                continue
            # print(kpt_path, kpt.shape)
            # kpt = kpt[:,:2]

            image = imread(image_path) / 255.
            if len(image.shape) < 3:
                image = np.tile(image[:, :, None], 3)
            ### crop information
            tform = self.crop(image, kpt)
            ## crop
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params,
                                 np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

            ###
            images_array = torch.from_numpy(cropped_image.transpose(2, 0, 1)).type(dtype=torch.float32)  # 224,224,3
            kpt_array = torch.from_numpy(cropped_kpt).type(dtype=torch.float32)  # 224,224,3

            data_dict = {
                'image': images_array,
                'landmark': kpt_array[:,:,:2],
                # 'mask': mask_array
            }

            return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, 0]);
        right = np.max(kpt[:, 0]);
        top = np.min(kpt[:, 1]);
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]

        size = int(old_size * scale)

        # crop image
        # src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno > 0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


########################## testing
def video2sequence(video_path):
    videofolder = video_path.split('.')[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = video_path.split('/')[-1].split('.')[0]
    # import ipdb; ipdb.set_trace()
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list


class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', face_detector_model=None):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + '/*.jpg') + glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print('please check the input path')
            exit()

        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'dlib':
            # self.face_detector = detectors.Dlib(model_path=face_detector_model)
            raise NotImplementedError()
        elif face_detector == 'fan':
            from gdl.utils.FaceDetector import FAN
            # self.face_detector = detectors.FAN()
            self.face_detector = FAN()
        else:
            print('no detector is used')

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)

        h, w, _ = image.shape
        if self.iscrop:
            if max(h, w) > 1000:
                print('image is too large, resize ',
                      imagepath)  # dlib detector will be very slow if the input image size is too large
                scale_factor = 1000 / max(h, w)
                image_small = rescale(image, scale_factor, preserve_range=True, multichannel=True)
                # print(image.shape)
                # print(image_small.shape)
                # exit()
                detected_faces = self.face_detector.run(image_small.astype(np.uint8))
            else:
                detected_faces = self.face_detector.run(image.astype(np.uint8))

            if detected_faces is None:
                print('no face detected! run original image')
                left = 0;
                right = h - 1;
                top = 0;
                bottom = w - 1
            else:
                # d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
                # left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
                kpt = detected_faces[0]
                left = np.min(kpt[:, 0]);
                right = np.max(kpt[:, 0]);
                top = np.min(kpt[:, 1]);
                bottom = np.max(kpt[:, 1])
                if max(h, w) > 1000:
                    scale_factor = 1. / scale_factor
                    left = left * scale_factor;
                    right = right * scale_factor;
                    top = top * scale_factor;
                    bottom = bottom * scale_factor
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
            size = int(old_size * self.scale)
            src_pts = np.array(
                [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                 [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': tform,
                'original_image': torch.tensor(image.transpose(2, 0, 1)).float(),
                }


class EvalData(Dataset):
    def __init__(self, testpath, kptfolder, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 face_detector_model=None):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + '/*.jpg') + glob(testpath + '/*.png')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print('please check the input path')
            exit()

        # print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'dlib':
            # self.face_detector = detectors.Dlib(model_path=face_detector_model)
            raise NotImplementedError()
        elif face_detector == 'fan':
            from gdl.utils.FaceDetector import FAN
            # self.face_detector = detectors.FAN()
            self.face_detector = FAN()
        else:
            print('no detector is used')
        self.kptfolder = kptfolder

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = imread(imagepath)[:, :, :3]

        h, w, _ = image.shape
        if self.iscrop:
            kptpath = os.path.join(self.kptfolder, imagename + '.npy')
            kpt = np.load(kptpath)
            left = np.min(kpt[:, 0]);
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1]);
            bottom = np.max(kpt[:, 1])
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
            size = int(old_size * self.scale)
            src_pts = np.array(
                [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                 [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': tform,
                'original_image': torch.tensor(image.transpose(2, 0, 1)).float(),
                }


class NoWVal(Dataset):
    def __init__(self, ring_elements=6, crop_size=224, scale=1.6, path=None):
        self.path = path or '/ps/scratch/face2d3d'
        # self.data_path =  self.path + '/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        self.data_path =  self.path + '/texture_in_the_wild_code/NoW_validation/image_paths_ring_6_elements.npy'
        self.ring_elements = ring_elements
        self.crop_size = crop_size
        self.data_lines = np.load(self.data_path).astype('str')
        self.scale = scale
        self.imagepath =  self.path + '/texture_in_the_wild_code/NoW_validation/iphone_pictures/'
        self.bbxpath =  self.path + '/texture_in_the_wild_code/NoW_validation/cropping_data/'

    def _handle_image(self, kps):
        pt_valid = []
        for pt in kps:
            if pt[2] > 0.41:  # Here 0 is not visible
                pt_valid.append(pt)
        lt, rb, valid = calc_aabb(pt_valid)
        if not valid:
            return
        else:
            return (lt, rb)

    def __len__(self):
        return self.data_lines.shape[0]

    def __getitem__(self, index):
        image_th = []
        image_names = []

        count = 0
        for i in range(self.ring_elements):
            image_path = self.imagepath + self.data_lines[index, i] + '.jpg'

            bbx_path = self.bbxpath + self.data_lines[index, i] + '.npy'
            h, w, _ = cv2.imread(image_path).shape
            bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
            # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')

            left = bbx_data['left'];
            right = bbx_data['right']
            top = bbx_data['top'];
            bottom = bbx_data['bottom']
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
            size = int(old_size * self.scale)
            src_pts = np.array(
                [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                 [center[0] + size / 2, center[1] - size / 2]])

            # print(image_path)
            # bbx_path = self.bbxpath + self.data_lines[index, i] + '.npy'
            # h, w, _ = cv2.imread(image_path).shape
            # bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
            # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
            # scale = 1.6
            # kps = np.array([[0, 0], [0, 0]])

            # image, kps = cut_image(image_path, kps, scale, box[0], box[1])
            # dst_image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC) / 255.
            # dst_image = dst_image[:, :, [2, 1, 0]].transpose(2, 0, 1)

            DST_PTS = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            image = imread(image_path)[:, :, :3]
            image = image / 255.

            dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
            dst_image = dst_image.transpose(2, 0, 1)

            #
            image_th.append(torch.tensor(dst_image).float())
            image_names.append(self.data_lines[index, i])
        image_th = torch.stack(image_th, dim=0)
        # image_th = torch.cat(image_th)
        return {'image': image_th,
                'imagenames': image_names
                }


import scipy.io


class AFLW2000(Dataset):
    def __init__(self, testpath='/ps/scratch/yfeng/Data/AFLW2000/GT', crop_size=224):
        '''
            data class for loading AFLW2000 dataset
            make sure each image has corresponding mat file, which provides cropping infromation
        '''
        if os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + '/*.jpg') + glob(testpath + '/*.png')
        elif isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png']):
            self.imagepath_list = [testpath]
        else:
            print('please check the input path')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = 1.6
        self.resolution_inp = crop_size

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]
        image = imread(imagepath)[:, :, :3]
        kpt = scipy.io.loadmat(imagepath.replace('jpg', 'mat'))['pt3d_68'].T
        left = np.min(kpt[:, 0]);
        right = np.max(kpt[:, 0]);
        top = np.min(kpt[:, 1]);
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        size = int(old_size * self.scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }


class NoWTest(Dataset):
    def __init__(self, ring_elements, crop_size, path=None):
        self.path = path or '/ps/scratch/face2d3d'
        self.data_path = self.path + '/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        self.ring_elements = ring_elements
        self.crop_size = crop_size
        self.data_lines = np.load(self.data_path).astype('str')
        self.imagepath = self.path + '/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/iphone_pictures/'
        self.bbxpath = self.path + '/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/detected_face/'

    def _handle_image(self, kps):
        pt_valid = []
        for pt in kps:
            if pt[2] > 0.41:  # Here 0 is not visible
                pt_valid.append(pt)
        lt, rb, valid = calc_aabb(pt_valid)
        if not valid:
            return
        else:
            return (lt, rb)

    def __len__(self):
        return self.data_lines.shape[0]

    def __getitem__(self, index):
        image_th = []
        image_names =[]

        count = 0
        for i in range(self.ring_elements):

            image_path = self.imagepath + self.data_lines[index, i] + '.jpg'
            bbx_path = self.bbxpath + self.data_lines[index, i] + '.npy'
            h, w, _ = cv2.imread(image_path).shape
            bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
            box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
            scale = 1.6
            kps = np.array([[0, 0], [0, 0]])
            image, kps = cut_image(image_path, kps, scale, box[0], box[1])
            dst_image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC) / 255.
            dst_image = dst_image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            #
            image_th.append(torch.tensor(dst_image).float())
            image_names.append(self.data_lines[index, i])
        image_th = torch.stack(image_th, dim=0)
        return {'image': image_th,
                'imagenames': image_names
                }


class FFHQ_val(Dataset):
    def __init__(self, ring_elements, crop_size, path=None):
        self.path = path or '/ps/scratch/face2d3d'
        self.data_path = self.path + '/faceHQ_100K/ffhq_cleaned_list_6_elements.npy'
        self.ring_elements = ring_elements
        self.crop_size = crop_size
        self.data_lines = np.load(self.data_path).astype('str')
        self.imagepath = self.path + '/faceHQ_100K/FFHQ/ffhq_resized_256/'
        self.landmarkpath = self.path + '/texture_in_the_wild_code/FFHQ/ffhq_resized_256_torch/'

    def _handle_image(self, kps):
        pt_valid = []
        for pt in kps:
            if pt[2] > 0.41:  # Here 0 is not visible
                pt_valid.append(pt)
        lt, rb, valid = calc_aabb(pt_valid)
        if not valid:
            return
        else:
            return (lt, rb)

    def __len__(self):
        return self.data_lines.shape[0]

    def __getitem__(self, index):
        image_th = []
        mask_th = []
        kp_th = []
        img_path = []
        image_names = []

        count = 0
        for i in range(self.ring_elements):
            # image_path = self.data_lines[index].split(' ')[count]
            # kps = load_openpose_landmarks(self.data_lines[index].split(' ')[count+1])
            image_path = self.imagepath + self.data_lines[index, i]
            kps = load_torch7_landmarks(self.landmarkpath + self.data_lines[index, i] + '.npy')
            h, w, _ = cv2.imread(image_path).shape
            box = self._handle_image(kps)
            scale = 1.2
            image, kps = cut_image(image_path, kps, scale, box[0], box[1])
            dst_image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC) / 255.
            dst_image = dst_image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            #
            image_th.append(torch.tensor(dst_image).float())
            image_names.append(self.data_lines[index, i][:-4])
        image_th = torch.stack(image_th, dim=0)
        return {'image': image_th,
                'imagepath': image_path,
                'imagenames': image_names
                }


from gdl.utils.DecaUtils import load_torch7_landmarks, cut_image, calc_aabb #, load_torch7_landmarks_v2

class GIF_val(Dataset):
    def __init__(self, ring_elements, crop_size, path):
        self.path = path or '/is/cluster/scratch/partha'
        self.data_path = self.path + '/gif_eval_data_toheiven/gif_loadinglist.npy'
        self.ring_elements = ring_elements
        self.crop_size = crop_size
        self.data_lines = np.load(self.data_path).astype('str')
        self.imagepath = self.path + '/gif_eval_data_toheiven/'
        self.landmarkpath = self.path + '/gif_eval_data_toheiven/results/'

    def _handle_image(self, kps):
        pt_valid = []
        for pt in kps:
            if pt[2] > 0.41:  # Here 0 is not visible
                pt_valid.append(pt)
        lt, rb, valid = calc_aabb(pt_valid)
        if not valid:
            return
        else:
            return (lt, rb)

    def __len__(self):
        return self.data_lines.shape[0]

    def __getitem__(self, index):
        image_th = []
        mask_th = []
        kp_th = []
        img_path = []
        image_names = []
        bbx_datas = []

        count = 0
        for i in range(self.ring_elements):
            # image_path = self.data_lines[index].split(' ')[count]
            # kps = load_openpose_landmarks(self.data_lines[index].split(' ')[count+1])
            img_dir = os.path.sep.join(self.data_lines[index, i].split('/')[-3:])
            image_path = self.imagepath + img_dir
            # print(image_path)
            image_folder = image_path.split('/')[-3]
            imagename = image_path.split('/')[-1]
            # print(self.landmarkpath + image_folder +'/' + imagename[:-4] + '.npy')
            kps = load_torch7_landmarks(self.landmarkpath + image_folder +'/' + imagename[:-4] + '.npy')
            h, w, _ = cv2.imread(image_path).shape
            box = self._handle_image(kps)
            scale = 1.2
            image, kps, bbx = cut_image_2(cv2.imread(image_path), kps, scale, box[0], box[1])
            dst_image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC) / 255.
            dst_image = dst_image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            #
            image_th.append(torch.tensor(dst_image).float())
            image_names.append(img_dir[:-4])
            bbx_datas.append(bbx)
        image_th = torch.stack(image_th, dim=0)
        return {'image': image_th,
                'imagepath': image_path,
                'imagenames': image_names,
                'bbx': bbx_datas
                }


class PaperVal(Dataset):
    def __init__(self, ring_elements, crop_size, path=None):
        self.path = path or '/ps/scratch/face2d3d'
        self.data_path = self.path + '/ringnetpp/eccv(haven)/test_data/papers/papers_ring_6_elements_loadinglist.npy'
        self.ring_elements = ring_elements
        self.crop_size = crop_size
        self.data_lines = np.load(self.data_path).astype('str')
        self.imagepath = self.path + '/ringnetpp/eccv(haven)/test_data/papers/'
        self.landmarkpath = self.path + '/ringnetpp/eccv(haven)/test_data/papers/'

    def _handle_image(self, kps):
        pt_valid = []
        for pt in kps:
            if pt[2] > 0.41:  # Here 0 is not visible
                pt_valid.append(pt)
        lt, rb, valid = calc_aabb(pt_valid)
        if not valid:
            return
        else:
            return (lt, rb)

    def __len__(self):
        return self.data_lines.shape[0]

    def __getitem__(self, index):
        image_th = []
        mask_th = []
        kp_th = []
        img_path = []
        image_names =[]

        count = 0
        for i in range(self.ring_elements):
            # image_path = self.data_lines[index].split(' ')[count]
            # kps = load_openpose_landmarks(self.data_lines[index].split(' ')[count+1])
            image_path = self.imagepath + self.data_lines[index, i] + '_cropped.jpg'
            # kps = load_torch7_landmarks_v2(self.landmarkpath + self.data_lines[index, i] + '_kpt_2d.txt')
            kps = load_torch7_landmarks(self.landmarkpath + self.data_lines[index, i] + '_kpt_2d.txt', allow_pickle=True)
            h, w, _ = cv2.imread(image_path).shape
            box = self._handle_image(kps)
            scale = 1.2
            image, kps = cut_image(image_path, kps, scale, box[0], box[1])
            dst_image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC) / 255.
            dst_image = dst_image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            #
            image_th.append(torch.tensor(dst_image).float())
            image_names.append(self.data_lines[index, i])

        image_th = torch.stack(image_th, dim=0)

        return {'image': image_th,
                'imagenames': image_names
                }


class CelebVal(Dataset):
    def __init__(self, ring_elements, crop_size, path=None):
        self.path = path or '/ps/scratch/face2d3d'
        self.data_path = self.path + '/texture_in_the_wild_code/celeb_ring_6_elements_loadinglist.npy'
        self.ring_elements = ring_elements
        self.crop_size = crop_size
        self.data_lines = np.load(self.data_path).astype('str')
        self.imagepath = self.path + '/texture_in_the_wild_code/celeb_images_processed/'
        self.landmarkpath = self.path + '/texture_in_the_wild_code/celeb_images_processed/'

    def _handle_image(self, kps):
        pt_valid = []
        for pt in kps:
            if pt[2] > 0.41:  # Here 0 is not visible
                pt_valid.append(pt)
        lt, rb, valid = calc_aabb(pt_valid)
        if not valid:
            return
        else:
            return (lt, rb)

    def __len__(self):
        return self.data_lines.shape[0]

    def __getitem__(self, index):
        image_th = []
        mask_th = []
        kp_th = []
        img_path = []
        image_names = []

        count = 0
        for i in range(self.ring_elements):
            # image_path = self.data_lines[index].split(' ')[count]
            # kps = load_openpose_landmarks(self.data_lines[index].split(' ')[count+1])
            image_path = self.imagepath + self.data_lines[index, i] + '.jpg'
            kps = load_torch7_landmarks(self.landmarkpath + self.data_lines[index, i] + '.npy')
            h, w, _ = cv2.imread(image_path).shape
            box = self._handle_image(kps)
            scale = 1.2
            image, kps = cut_image(image_path, kps, scale, box[0], box[1])
            dst_image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC) / 255.
            dst_image = dst_image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            #
            image_th.append(torch.tensor(dst_image).float())
            image_names.append(self.data_lines[index, i])
        # print('image shape: {}'.format(len(image_th)))
        # print('image names: {}'.format(image_names))
        image_th = torch.stack(image_th, dim=0)
        return {'image': image_th,
                'imagenames': image_names
                }

if __name__ == "__main__":
    cfg = {"data": {
        "path": "/ps/scratch/face2d3d/",
        "n_train": 10000000,
        "sampler": False,
        # "scale_max": 2.8,
        # "scale_min": 2,
        "scale_max": 1.6,
        "scale_min": 1.2,
        "data_class": "DecaDataModule",
        "num_workers": 4,
        "split_ratio": 0.9,
        "split_style": "random",
        # "trans_scale": 0.2,
        "trans_scale": 0.1,
        "testing_datasets": [
            "now-test",
            "now-val",
            "celeb-val"
        ],
        "training_datasets": [
            "vggface2",
            "ethnicity"
        ],
        "validation_datasets": [
            "now-val",
            "celeb-val"
        ]
    },
                "learning": {
                "val_K": 1,
                "test_K": 1,
                "train_K": 1,
                "num_gpus": 1,
                "optimizer": "Adam",
                "logger_type": "WandbLogger",
                "val_K_policy": "sequential",
                "learning_rate": 0.0001,
                "test_K_policy": "sequential",
                "batch_size_val": 32,
                "early_stopping": {
                    "patience": 15
                },
                "train_K_policy": "random",
                "batch_size_test": 1,
                "batch_size_train": 32,
                "gpu_memory_min_gb": 30,
                "checkpoint_after_training": "best"
            },
        "model": {
            "image_size": 224
        }
    }
    cfg = DictConfig(cfg)
    dm = DecaDataModule(cfg)
    dm.prepare_data()
    dm.setup()
    # dataset = dm.train_dataloader()
    dataset = dm.training_set

    import matplotlib.pyplot as plt

    for i in range(len(dataset)):
        batch = dataset[i]
        im = batch["image"][0].cpu().numpy().transpose([1, 2, 0])
        plt.figure()
        plt.imshow(im)
        plt.show()

