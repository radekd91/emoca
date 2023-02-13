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
from PIL import Image
from skimage.io import imread
from torchvision.transforms import ToTensor

from gdl.utils.FaceDetector import load_landmark
from gdl.datasets.FaceAlignmentTools import align_face

from skvideo.io import vread, vreader 
from types import GeneratorType
import pickle as pkl

class VideoFaceDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, video_name, landmark_path, image_transforms=None, 
                align_landmarks=False, vid_read=None, output_im_range=None, 
                scale_adjustment=1.25,
                target_size_height=256, 
                target_size_width=256,
                ):
        super().__init__()
        self.video_name = video_name
        self.landmark_path = landmark_path / "landmarks_original.pkl"
        # if landmark_list is not None and len(lanmark_file_name) != len(image_list):
        #     raise RuntimeError("There must be a landmark for every image")
        self.image_transforms = image_transforms
        self.vid_read = vid_read or 'skvreader' # 'skvread'
        self.prev_index = -1

        self.scale_adjustment=scale_adjustment
        self.target_size_height=target_size_height
        self.target_size_width=target_size_width

        self.video_frames = None 
        if self.vid_read == "skvread": 
            self.video_frames = vread(str(self.video_name))
        elif self.vid_read == "skvreader": 
            self.video_frames = vreader(str(self.video_name))

        with open(self.landmark_path, "rb") as f: 
            self.landmark_list = pkl.load(f)

        with open(landmark_path / "landmark_types.pkl", "rb") as f: 
            self.landmark_types = pkl.load(f)
        
        self.total_len = 0 
        self.frame_map = {} # detection index to frame map
        self.index_for_frame_map = {} # detection index to frame map
        for i in range(len(self.landmark_list)): 
            for j in range(len(self.landmark_list[i])): 
                self.frame_map[self.total_len + j] = i
                self.index_for_frame_map[self.total_len + j] = j
            self.total_len += len(self.landmark_list[i])

        self.output_im_range = output_im_range


    def __getitem__(self, index):
        # if index < len(self.image_list):
        #     x = self.mnist_data[index]
        # raise IndexError("Out of bounds")
        if index != self.prev_index+1 and self.vid_read != 'skvread': 
            raise RuntimeError("This dataset is meant to be accessed in ordered way only (and with 0 or 1 workers)")

        frame_index = self.frame_map[index]
        detection_in_frame_index = self.index_for_frame_map[index]
        landmark = self.landmark_list[frame_index][detection_in_frame_index]
        landmark_type = self.landmark_types[frame_index][detection_in_frame_index]

        if isinstance(self.video_frames, np.ndarray): 
            img = self.video_frames[frame_index, ...]
        elif isinstance(self.video_frames, GeneratorType):
            img = next(self.video_frames)
        else: 
            raise NotImplementedError() 

        # try:
        #     if self.vid_read == 'skvread':
        #         img = vread(self.image_list[index])
        #         img = img.transpose([2, 0, 1]).astype(np.float32)
        #         img_torch = torch.from_numpy(img)
        #         path = str(self.image_list[index])
        #     elif self.vid_read == 'pil':
        #         img = Image.open(self.image_list[index])
        #         img_torch = ToTensor()(img)
        #         path = str(self.image_list[index])
        #         # path = f"{index:05d}"
        #     else:
        #         raise ValueError(f"Invalid image reading method {self.im_read}")
        # except Exception as e:
        #     print(f"Failed to read '{self.image_list[index]}'. File is probably corrupted. Rerun data processing")
        #     raise e

        # crop out the face
        img = align_face(img, landmark, landmark_type, scale_adjustment=1.25, target_size_height=256, target_size_width=256,)
        if self.output_im_range == 255: 
            img = img * 255.0
        img = img.astype(np.float32)
        img_torch = ToTensor()(img)

        # # plot img with pyplot 
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        # # plot image with plotly
        # import plotly.graph_objects as go
        # fig = go.Figure(data=go.Image(z=img*255.,))
        # fig.show()


        if self.image_transforms is not None:
            img_torch = self.image_transforms(img_torch)

        batch = {"image" : img_torch,
        #         "path" : path
        }

        self.prev_index += 1
        return batch

    def __len__(self):
        return self.total_len