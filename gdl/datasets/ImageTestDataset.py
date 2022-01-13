import glob
import os

import cv2
import numpy as np
import scipy
import torch
from skimage.io import imread
from skimage.transform import rescale, estimate_transform, warp
from torch.utils.data import Dataset

# from gdl.datasets.FaceVideoDataModule import add_pretrained_deca_to_path
from gdl.datasets.ImageDatasetHelpers import bbox2point
from gdl.utils.FaceDetector import FAN


class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 scaling_factor=1.0):
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
            print(f'please check the test path: {testpath}')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        # add_pretrained_deca_to_path()
        # from decalib.datasets import detectors
        if face_detector == 'fan':
            self.face_detector = FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if self.scaling_factor != 1.:
            image = rescale(image, (self.scaling_factor, self.scaling_factor, 1))*255.

        h, w, _ = image.shape
        if self.iscrop:
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = imagepath.replace('.jpg', '.mat').replace('.png', '.mat')
            kpt_txtpath = imagepath.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            else:
                # bbox, bbox_type, landmarks = self.face_detector.run(image)
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 1:
                    print('no face detected! run original image')
                    left = 0
                    right = h - 1
                    top = 0
                    bottom = w - 1
                else:
                    bbox = bbox[0]
                    left = bbox[0]
                    right = bbox[2]
                    top = bbox[1]
                    bottom = bbox[3]
                old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
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
                'image_name': imagename,
                'image_path': imagepath,
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }


def video2sequence(video_path):
    videofolder = video_path.split('.')[0]
    util.check_mkdir(videofolder)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list