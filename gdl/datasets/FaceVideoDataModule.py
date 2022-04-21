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


from torch.utils.data.dataloader import DataLoader
import os, sys
from pathlib import Path
import numpy as np
import torch
# import torchaudio
from typing import Optional, Union, List
import pickle as pkl
# from collections import OrderedDict
from tqdm import tqdm, auto
# import subprocess
from torchvision.transforms import Resize, Compose
import gdl
from gdl.datasets.ImageTestDataset import TestData
from gdl.datasets.FaceDataModuleBase import FaceDataModuleBase
from gdl.datasets.ImageDatasetHelpers import point2bbox, bbpoint_warp
from gdl.datasets.UnsupervisedImageDataset import UnsupervisedImageDataset
from facenet_pytorch import InceptionResnetV1
from collections import OrderedDict
from gdl.datasets.IO import save_emotion
from PIL import Image, ImageDraw, ImageFont
import cv2
from skimage.io import imread

# from memory_profiler import profile


def add_pretrained_deca_to_path():
    deca_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DECA'))
    if deca_path not in sys.path:
        sys.path.insert(0, deca_path)


class FaceVideoDataModule(FaceDataModuleBase):
    """
    Base data module for face video datasets. Contains the functionality to unpack the videos, detect faces, segment faces, ...
    """

    def __init__(self, root_dir, output_dir, processed_subfolder=None,
                 face_detector='fan',
                 face_detector_threshold=0.9,
                 image_size=224,
                 scale=1.25,
                 device=None):
        super().__init__(root_dir, output_dir,
                         processed_subfolder=processed_subfolder,
                         face_detector=face_detector,
                         face_detector_threshold=face_detector_threshold,
                         image_size = image_size,
                         scale = scale,
                         device=device)


        # self._instantiate_detector()
        # self.face_recognition = InceptionResnetV1(pretrained='vggface2').eval().to(device)


        self.version = 2

        self.video_list = None
        self.video_metas = None
        self.annotation_list = None
        self.frame_lists = None
        self.loaded = False
        # self.detection_lists = None

        self.detection_fnames = []
        self.detection_centers = []
        self.detection_sizes = []

    @property
    def metadata_path(self):
        return os.path.join(self.output_dir, "metadata.pkl")

    def prepare_data(self, *args, **kwargs):
        outdir = Path(self.output_dir)

        # is dataset already processed?
        # if outdir.is_dir():
        if Path(self.metadata_path).is_file():
            print("The dataset is already processed. Loading")
            self._loadMeta()
            return
        # else:
        self._gather_data()
        self._unpack_videos()
        self._saveMeta()


    def _unpack_videos(self):
        self.frame_lists = []
        for vi, video_file in enumerate(tqdm(self.video_list)):
            self._unpack_video(vi)

    def get_frame_number_format(self):
        return "%06d"

    # def _get_unpacked_video_subfolder(self, video_idx):
        # return  Path(self._video_category(video_idx)) / video_file.parts[-3] /self._video_set(video_idx) / video_file.stem

    def _unpack_video(self, video_idx, overwrite=False):
        video_file = Path(self.root_dir) / self.video_list[video_idx]
        # suffix = self._get_unpacked_video_subfolder(video_idx)
        # out_folder = Path(self.output_dir) / suffix
        out_folder = self._get_path_to_sequence_frames(video_idx)

        if not out_folder.exists() or overwrite:
            print("Unpacking video to '%s'" % str(out_folder))
            out_folder.mkdir(exist_ok=True, parents=True)

            out_format = out_folder / (self.get_frame_number_format() + ".png")
            out_format = '-r 1 -i %s -r 1 ' % str(video_file) + ' "' + str(out_format) + '"'
            # out_format = ' -r 1 -i %s ' % str(video_file) + ' "' + "$frame.%03d.png" + '"'
            # subprocess.call(['ffmpeg', out_format])
            os.system("ffmpeg " + out_format)

            # import ffmpeg
            # stream = ffmpeg.input(str(video_file))
            # # stream = ffmpeg.output(stream.video, str(out_format))
            # stream = ffmpeg.output(stream.video, "%06.png")
            # stream.run()
        frame_list = sorted(list(out_folder.glob("*.png")))
        frame_list = [path.relative_to(self.output_dir) for path in frame_list]
        self.frame_lists += [frame_list]
        n_frames = len(frame_list)
        expected_frames = int(self.video_metas[video_idx]['num_frames'])
        if n_frames == expected_frames:
            pass
            # print("Successfully unpacked the video into %d frames" % expected_frames)
        else:
            print("[WARNING] Expected %d frames but got %d vor video '%s'"
                  % (expected_frames, n_frames, str(video_file)))


    def _detect_faces(self):
        for sid in range(self.num_sequences):
            self._detect_faces_in_sequence(sid)

    def _get_path_to_sequence_files(self, sequence_id, file_type, method="", suffix=""): 
        assert file_type in ['videos', 'detections', "landmarks", "segmentations", 
            "emotions", "reconstructions"]
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            file_type += "_" + method 
        if len(suffix) > 0:
            file_type += suffix

        suffix = Path(self._video_category(sequence_id)) / file_type /self._video_set(sequence_id) / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder


    def _get_path_to_sequence_frames(self, sequence_id):
        return self._get_path_to_sequence_files(sequence_id, "videos")

    def _get_path_to_sequence_detections(self, sequence_id): 
        return self._get_path_to_sequence_files(sequence_id, "detections")

    def _get_path_to_sequence_landmarks(self, sequence_id):
        return self._get_path_to_sequence_files(sequence_id, "landmarks")

    def _get_path_to_sequence_segmentations(self, sequence_id):
        return self._get_path_to_sequence_files(sequence_id, "segmentations")

    def _get_path_to_sequence_emotions(self, sequence_id):
        return self._get_path_to_sequence_files(sequence_id, "emotions")

    def _video_category(self, sequence_id):
        video_file = self.video_list[sequence_id]
        out_folder = video_file.parts[-4]
        return out_folder

    def _video_set(self, sequence_id):
        video_file = self.video_list[sequence_id]
        out_folder = video_file.parts[-2]
        return out_folder

    def _get_path_to_sequence_reconstructions(self, sequence_id, rec_method='emoca', suffix=''):
        if suffix is None:
            suffix = ''
        
        if rec_method == 'deca':
            return self._get_path_to_sequence_files(sequence_id, "reconstructions", "", suffix)
        else:
            assert rec_method in ['emoca', 'deep3dface']
            return self._get_path_to_sequence_files(sequence_id, "reconstructions", rec_method, suffix)
        # video_file = self.video_list[sequence_id]
        # if rec_method == 'deca':
        #     suffix = Path(self._video_category(sequence_id)) / f'reconstructions{suffix}' /self._video_set(sequence_id) / video_file.stem
        # elif rec_method == 'emoca':
        #     suffix = Path(self._video_category(sequence_id)) / f'reconstructions_emoca{suffix}' /self._video_set(sequence_id) / video_file.stem
        # elif rec_method == 'deep3dface':
        #     suffix = Path(self._video_category(sequence_id)) / f'reconstructions_deep3dface{suffix}' /self._video_set(sequence_id) / video_file.stem
        # else:
        #     raise ValueError("Unknown reconstruction method '%s'" % rec_method)
        # out_folder = Path(self.output_dir) / suffix
        # return out_folder

    

    # @profile
    def _detect_faces_in_sequence(self, sequence_id):
        # if self.detection_lists is None or len(self.detection_lists) == 0:
        #     self.detection_lists = [ [] for i in range(self.num_sequences)]
        video_file = self.video_list[sequence_id]
        print("Detecting faces in sequence: '%s'" % video_file)
        # suffix = Path(self._video_category(sequence_id)) / 'detections' /self._video_set(sequence_id) / video_file.stem
        out_detection_folder = self._get_path_to_sequence_detections(sequence_id)
        out_detection_folder.mkdir(exist_ok=True, parents=True)
        out_file = out_detection_folder / "bboxes.pkl"

        out_landmark_folder = self._get_path_to_sequence_landmarks(sequence_id)
        out_landmark_folder.mkdir(exist_ok=True, parents=True)

        centers_all = []
        sizes_all = []
        detection_fnames_all = []
        landmark_fnames_all = []
        # save_folder = frame_fname.parents[3] / 'detections'

        # # TODO: resuming is not tested, probably doesn't work yet
        # checkpoint_frequency = 100
        # resume = False
        # if resume and out_file.exists():
        #     detection_fnames_all, landmark_fnames_all, centers_all, sizes_all, start_fid = \
        #         FaceVideoDataModule.load_detections(out_file)
        # else:
        #     start_fid = 0
        #
        # # hack trying to circumvent memory leaks on the cluster
        # detector_instantion_frequency = 200
        start_fid = 0

        frame_list = self.frame_lists[sequence_id]
        fid = 0
        if len(frame_list) == 0:
            print("Nothing to detect in: '%s'. All frames have been processed" % self.video_list[sequence_id])
        for fid, frame_fname in enumerate(tqdm(range(start_fid, len(frame_list)))):

            # if fid % detector_instantion_frequency == 0:
            #     self._instantiate_detector(overwrite=True)

            self._detect_faces_in_image_wrapper(frame_list, fid, out_detection_folder, out_landmark_folder, out_file,
                                           centers_all, sizes_all, detection_fnames_all, landmark_fnames_all)

        FaceVideoDataModule.save_detections(out_file,
                                            detection_fnames_all, landmark_fnames_all, centers_all, sizes_all, fid)
        print("Done detecting faces in sequence: '%s'" % self.video_list[sequence_id])


    def _get_emotion_net(self, device):
        from gdl.layers.losses.EmonetLoader import get_emonet

        net = get_emonet()
        net = net.to(device)

        return net, "emo_net"

    def _segment_faces_in_sequence(self, sequence_id):
        video_file = self.video_list[sequence_id]
        print("Segmenting faces in sequence: '%s'" % video_file)
        # suffix = Path(self._video_category(sequence_id)) / 'detections' /self._video_set(sequence_id) / video_file.stem

        out_detection_folder = self._get_path_to_sequence_detections(sequence_id)
        out_segmentation_folder = self._get_path_to_sequence_segmentations(sequence_id)
        out_segmentation_folder.mkdir(exist_ok=True, parents=True)

        detection_fnames = sorted(list(out_detection_folder.glob("*.png")))

        self._segment_images(detection_fnames, out_segmentation_folder)


    def _extract_emotion_from_faces_in_sequence(self, sequence_id):

        video_file = self.video_list[sequence_id]
        print("Extracting emotion fetures from faces in sequence: '%s'" % video_file)
        # suffix = Path(self._video_category(sequence_id)) / 'detections' /self._video_set(sequence_id) / video_file.stem

        in_detection_folder = self._get_path_to_sequence_detections(sequence_id)
        out_emotion_folder = self._get_path_to_sequence_emotions(sequence_id)
        out_emotion_folder.mkdir(exist_ok=True, parents=True)
        print("into folder : '%s'" % str(out_emotion_folder))

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        net, emotion_type = self._get_emotion_net(device)

        detection_fnames = sorted(list(in_detection_folder.glob("*.png")))

        # ref_im = imread(detection_fnames[0])
        # ref_size = Resize((ref_im.shape[0], ref_im.shape[1]), interpolation=Image.NEAREST)

        transforms = Compose([
            Resize((256, 256)),
        ])
        batch_size = 64

        dataset = UnsupervisedImageDataset(detection_fnames, image_transforms=transforms,
                                           im_read='pil')
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        # import matplotlib.pyplot as plt

        for i, batch in enumerate(tqdm(loader)):
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
                emotion_path = out_emotion_folder / (Path(image_path).stem + ".pkl")
                emotion_feature_j = {key: val[j] for key, val in emotion_features.items()}
                del emotion_feature_j['emo_feat'] # too large to be stored per frame = (768, 64, 64)
                del emotion_feature_j['heatmap'] # not too large but probably not usefull = (68, 64, 64)
                # we are keeping emo_feat_2 (output of last conv layer (before FC) and then the outputs of the FCs - expression, valence and arousal)
                save_emotion(emotion_path, emotion_feature_j, emotion_type)
            # end = time.time()
            # print(f" Saving batch {i} took: {end - start}")

    @staticmethod
    def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='overlay.png'):
        # Colors for all 20 parts
        part_colors = [
            [255, 0, 0], #0
            [255, 85, 0], #1
            [255, 170, 0],#2
            [255, 0, 85], #3
            [255, 0, 170], #4
            [0, 255, 0],  #5
            [85, 255, 0], #6
            [170, 255, 0],# 7
            [0, 255, 85], # 8
            [0, 255, 170],# 9
            [0, 0, 255], # 10
            [85, 0, 255], # 11
            [170, 0, 255], # 12
            [0, 85, 255], # 13
            [0, 170, 255], # 14
            [255, 255, 0], # 15
            [255, 255, 85], #16
            [255, 255, 170], #17
            [255, 0, 255], #18
            [255, 85, 255], #19
            [255, 170, 255], #20
            [0, 255, 255],# 21
            [85, 255, 255], #22
            [170, 255, 255] #23
                       ]

        im = np.array(im)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # print(vis_parsing_anno_color.shape, vis_im.shape)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        # vis_parsing_anno_color = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.0, vis_parsing_anno_color, 1.0, 0)

        # Save result or not
        if save_im:
            cv2.imwrite(save_path[:-4] + 'seg_vis.png', vis_parsing_anno)
            cv2.imwrite(save_path[:-4] + 'seg_vis_color.png', vis_parsing_anno_color)
            cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


    def _get_emonet(self, device=None):
        from gdl.utils.other import get_path_to_externals
        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        path_to_emonet = get_path_to_externals() / "emonet"
        if not(str(path_to_emonet) in sys.path  or str(path_to_emonet.absolute()) in sys.path):
            sys.path += [str(path_to_emonet)]

        from emonet.models import EmoNet
        # n_expression = 5
        n_expression = 8

        # Loading the model
        import inspect
        state_dict_path = Path(inspect.getfile(EmoNet)).parent.parent.parent /'pretrained' / f'emonet_{n_expression}.pth'

        print(f'Loading the EmoNet model from {state_dict_path}.')
        state_dict = torch.load(str(state_dict_path), map_location='cpu')
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        net = EmoNet(n_expression=n_expression).to(device)
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        return net

    def _recognize_emotion_in_faces(self, device = None):
        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        emotion_net = self._get_emonet(device)
        for sid in range(self.num_sequences):
            self._recognize_emotion_in_sequence(sid, emotion_net, device)

    def _recognize_emotion_in_sequence(self, sequence_id, emotion_net=None, device=None):
        print("Running emotion recognition in sequence '%s'" % self.video_list[sequence_id])
        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        out_folder = self._get_path_to_sequence_detections(sequence_id)
        in_folder = self._get_path_to_sequence_detections(sequence_id)
        detections_fnames = sorted(list(in_folder.glob("*.png")))

        emotion_net = emotion_net or self._get_emonet(self.device)

        from torchvision.transforms import Resize
        transforms = Resize((256,256))
        dataset = UnsupervisedImageDataset(detections_fnames, image_transforms=transforms)
        batch_size = 64
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        all_emotions = []
        all_valence = []
        all_arousal = []

        for i, batch in enumerate(tqdm(loader)):
            # facenet_pytorch expects this stanadrization for the input to the net
            # images = fixed_image_standardization(batch['image'].to(device))
            images = batch['image'].to(device)
            out = emotion_net(images)
            expr = out['expression']
            expr = np.argmax(np.squeeze(expr.detach().cpu().numpy()), axis=1)
            val = out['valence']
            ar = out['arousal']

            all_emotions += [expr]
            all_valence += [val.detach().cpu().numpy()]
            all_arousal += [ar.detach().cpu().numpy()]

        emotion_array = np.concatenate(all_emotions, axis=0)
        valence_array = np.concatenate(all_valence, axis=0)
        arousal_array = np.concatenate(all_arousal, axis=0)
        FaceVideoDataModule._save_face_emotions(out_folder / "emotions.pkl", emotion_array, valence_array, arousal_array, detections_fnames)
        print("Done running emotion recognition in sequence '%s'" % self.video_list[sequence_id])

    def _get_recognition_net(self, device):
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        return resnet

    def _recognize_faces(self, device = None):
        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        recognition_net = self._get_recognition_net(device)
        for sid in range(self.num_sequences):
            self._recognize_faces_in_sequence(sid, recognition_net, device)

    def _recognize_faces_in_sequence(self, sequence_id, recognition_net=None, device=None, num_workers = 4):

        def fixed_image_standardization(image_tensor):
            processed_tensor = (image_tensor - 127.5) / 128.0
            return processed_tensor

        print("Running face recognition in sequence '%s'" % self.video_list[sequence_id])
        out_folder = self._get_path_to_sequence_detections(sequence_id)
        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        recognition_net = recognition_net or self._get_recognition_net(device)
        recognition_net.requires_grad_(False)
        # detections_fnames = sorted(self.detection_fnames[sequence_id])
        detections_fnames = sorted(list(out_folder.glob("*.png")))
        dataset = UnsupervisedImageDataset(detections_fnames)
        loader = DataLoader(dataset, batch_size=64, num_workers=num_workers, shuffle=False)
        # loader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)
        all_embeddings = []
        for i, batch in enumerate(tqdm(loader)):
            # facenet_pytorch expects this stanadrization for the input to the net
            images = fixed_image_standardization(batch['image'].to(device))
            embeddings = recognition_net(images)
            all_embeddings += [embeddings.detach().cpu().numpy()]

        embedding_array = np.concatenate(all_embeddings, axis=0)
        FaceVideoDataModule._save_face_embeddings(out_folder / "embeddings.pkl", embedding_array, detections_fnames)
        print("Done running face recognition in sequence '%s'" % self.video_list[sequence_id])


    def _process_everything_for_sequence(self, si, reconstruct=False):
        self._detect_faces_in_sequence(si)
        # self._recognize_faces_in_sequence(si)
        self._identify_recognitions_for_sequence(si)
        if reconstruct:
            self._reconstruct_faces_in_sequence(si)


    @staticmethod
    def _save_face_emotions(fname, emotions, valence, arousal, detections_fnames):
        with open(fname, "wb" ) as f:
            pkl.dump(emotions, f)
            pkl.dump(valence, f)
            pkl.dump(arousal, f)
            pkl.dump(detections_fnames, f)

    @staticmethod
    def _load_face_emotions(fname):
        with open(fname, "rb") as f:
            emotions = pkl.load(f)
            valence = pkl.load(f)
            arousal = pkl.load(f)
            detections_fnames = pkl.load(f)
        return emotions, valence, arousal, detections_fnames


    @staticmethod
    def _save_face_embeddings(fname, embeddings, detections_fnames):
        with open(fname, "wb" ) as f:
            pkl.dump(embeddings, f)
            pkl.dump(detections_fnames, f)

    @staticmethod
    def _load_face_embeddings(fname):
        with open(fname, "rb") as f:
            embeddings = pkl.load(f)
            detections_fnames = pkl.load(f)
        return embeddings, detections_fnames

    # def _get_reconstruction_net(self, device):
    #     add_pretrained_deca_to_path()
    #     from decalib.deca import EMOCA
    #     from decalib.utils.config import cfg as deca_cfg
    #     # deca_cfg.model.use_tex = args.useTex
    #     deca_cfg.model.use_tex = False
    #     deca = EMOCA(config=deca_cfg, device=device)
    #     return deca

    def _get_reconstruction_net(self, device, rec_method='deca'):
        if rec_method == 'deca':
            add_pretrained_deca_to_path()
            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg
            # deca_cfg.model.use_tex = args.useTex
            deca_cfg.model.use_tex = False
            deca = DECA(config=deca_cfg, device=device)
            return deca
        elif rec_method == "emoca":
            checkpoint_mode = 'best'
            mode = "test"
            from gdl.models.IO import get_checkpoint_with_kwargs
            from omegaconf import OmegaConf
            # model_path = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"
            model_path = Path(gdl.__file__).parents[1] / "assets" / "EMOCA" / "models" / "EMOCA"
            cfg = OmegaConf.load(Path(model_path) / "cfg.yaml")
            stage = 'detail'
            cfg = cfg[stage]
            checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, mode, checkpoint_mode=checkpoint_mode,
                                                                       replace_root = None, relative_to = None)
            # make sure you use the deca class of the target (for instance, if target is ExpDECA but we're starting from
            # pretrained EMOCA)

            # cfg_pretrain_.model.deca_class = cfg_coarse.model.deca_class
            # checkpoint_kwargs["config"]["model"]["deca_class"] = cfg_coarse.model.deca_class
            # load from configs
            from gdl.models.DECA import instantiate_deca
            deca_checkpoint_kwargs = {
                "model_params": checkpoint_kwargs["config"]["model"],
                "learning_params": checkpoint_kwargs["config"]["learning"],
                "inout_params": checkpoint_kwargs["config"]["inout"],
                "stage_name": "train",
            }
            deca = instantiate_deca(cfg, mode, "",  checkpoint, deca_checkpoint_kwargs )
            deca.to(device)
            deca.deca.config.detail_constrain_type = 'none'
            return deca
            # return deca.deca
        elif rec_method == "deep3dface":

            # face_model = None
            # from hydra.experimental import compose, initialize
            #
            # default = "deca_train_detail"
            # overrides = [
            #     # 'model/settings=3ddfa',
            #     # 'model/settings=3ddfa_resnet',
            #     'model/settings=deep3dface',
            #     'learning/logging=none',
            #     'data/datasets=affectnet_desktop',  # affectnet vs deca dataset
            #     # 'data/datasets=affectnet_cluster',  # affectnet vs deca dataset
            #     'data.num_workers=0',
            #     'learning.batch_size_train=4',
            # ]
            #
            # initialize(config_path="../emoca_conf", job_name="test_face_model")
            # conf = compose(config_name=default, overrides=overrides)

            from gdl.models.external.Deep3DFace import Deep3DFaceModule
            from omegaconf import DictConfig

            model_cfg = {
                # "value": {
                    "mode": "detail",
                    # "n_cam": 3,
                    # "n_exp": 50,
                    # "n_tex": 50,
                    # "n_pose": 6,
                    # "n_light": 27,
                    # "n_shape": 100,
                    # "uv_size": 256,
                    # "n_detail": 128,
                    # "tex_path": "/ps/scratch/rdanecek/data/FLAME/texture/FLAME_albedo_from_BFM.npz",
                    # "tex_type": "BFM",
                    "n_dlatent": 512,
                    "deca_class": "Deep3DFaceModule",
                    "deep3dface": {
                        "name": "face_recon_feat0.2_augment",
                        "epoch": 20,
                        "focal": 1015,
                        "model": "facerecon",
                        "phase": "test",
                        "z_far": 15,
                        "center": 112,
                        "suffix": "null",
                        "z_near": 5,
                        "gpu_ids": 0,
                        "isTrain": False,
                        "use_ddp": False,
                        "verbose": False,
                        "camera_d": 10,
                        "ddp_port": 12355,
                        "add_image": True,
                        "bfm_model": "BFM_model_front.mat",
                        "init_path": "checkpoints/init_model/resnet50-0676ba61.pth",
                        "net_recon": "resnet50",
                        "bfm_folder": "BFM",
                        "img_folder": "./datasets/examples",
                        "world_size": 1,
                        "use_last_fc": False,
                        "dataset_mode": "None",
                        "vis_batch_nums": 1,
                        "checkpoints_dir": "./checkpoints",
                        "eval_batch_nums": "inf",
                        "display_per_batch": True
                    },
                    "image_size": 224,
                    "max_epochs": 4,
                    # "n_identity": 512,
                    # "topology_path": "/ps/scratch/rdanecek/data/FLAME/geometry/head_template.obj",
                    # "face_mask_path": "/ps/scratch/rdanecek/data/FLAME/mask/uv_face_mask.png",
                    # "neural_renderer": false,
                    # "flame_model_path": "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl",
                    # "val_vis_frequency": 200,
                    # "face_eye_mask_path": "/ps/scratch/rdanecek/data/FLAME/mask/uv_face_eye_mask.png",
                    # "test_vis_frequency": 1,
                    # "val_check_interval": 0.2,
                    # "train_vis_frequency": 1000,
                    # "pretrained_modelpath": "/home/rdanecek/Workspace/Repos/DECA/data/deca_model.tar",
                    # "background_from_input": true,
                    # "fixed_displacement_path": "/ps/scratch/rdanecek/data/FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy",
                    # "flame_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy",
                    # "pretrained_vgg_face_path": "/ps/scratch/rdanecek/pretrained_vggfaceresnet/resnet50_ft_weight.pkl"
                }
            # }

            learning_cfg =  {
                "path": "/ps/scratch/face2d3d/",
                "n_train": 10000000,
                "scale_max": 1.6,
                "scale_min": 1.2,
                "data_class": "DecaDataModule",
                "num_workers": 4,
                "split_ratio": 0.9,
                "split_style": "random",
                "trans_scale": 0.1,
                "testing_datasets": [
                    "now-test",
                    "now-val",
                    "celeb-val"
                ],
                "training_datasets": [
                    "vggface2hq",
                    "vox2"
                ],
                "validation_datasets": [
                    "now-val",
                    "celeb-val"
                ]
            }

            inout_cfg = {
                "name": "ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early",
                "time": "2021_11_13_03-43-40",
                "random_id": "3038711584732653067",
                "output_dir": "/is/cluster/work/rdanecek/emoca/finetune_deca",
                "full_run_dir": "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail",
                "checkpoint_dir": "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early/detail/checkpoints"
            }


            face_model = Deep3DFaceModule(DictConfig(model_cfg), DictConfig(learning_cfg),
                                          DictConfig(inout_cfg), "")
            face_model.to(device)
            return face_model


        else:
            raise ValueError("Unknown version of the reconstruction net")


    def _reconstruct_faces(self, device = None):
        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        reconstruction_net = self._get_reconstruction_net(device)
        for sid in range(self.num_sequences):
            self._reconstruct_faces_in_sequence(sid, reconstruction_net, device)


    def _reconstruct_faces_in_sequence(self, sequence_id, reconstruction_net=None, device=None,
                                       save_obj=False, save_mat=True, save_vis=True, save_images=False,
                                       save_video=True, rec_method='emoca', retarget_from=None, retarget_suffix=None):
        # add_pretrained_deca_to_path()
        # from decalib.utils import util
        import gdl.utils.DecaUtils as util
        from scipy.io.matlab import savemat

        if retarget_from is not None:
            import datetime
            t = datetime.datetime.now()
            t_str = t.strftime("%Y_%m_%d_%H-%M-%S")
            suffix = f"_retarget_{t_str}_{str(hash(t_str))}" if retarget_suffix is None else retarget_suffix
        else:
            codedict_retarget = None
            suffix = None



        def fixed_image_standardization(image):
            return image / 255.

        print("Running face reconstruction in sequence '%s'" % self.video_list[sequence_id])
        in_folder = self._get_path_to_sequence_detections(sequence_id)
        out_folder = self._get_path_to_sequence_reconstructions(sequence_id, rec_method=rec_method, suffix=suffix)

        if retarget_from is not None:
            out_folder.mkdir(exist_ok=True, parents=True)

        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        reconstruction_net = reconstruction_net or self._get_reconstruction_net(device, rec_method=rec_method)

        if retarget_from is not None:
            image = imread(retarget_from)
            batch_r = {}
            batch_r["image"] = torch.from_numpy(image).float().unsqueeze(0).to(device)
            # to torch channel format
            batch_r["image"] = batch_r["image"].permute(0, 3, 1, 2)
            batch_r["image"] = fixed_image_standardization(batch_r["image"])
            with torch.no_grad():
                codedict_retarget = reconstruction_net.encode(batch_r, training=False)


        video_writer = None
        detections_fnames = sorted(list(in_folder.glob("*.png")))
        dataset = UnsupervisedImageDataset(detections_fnames)
        batch_size = 32
        # batch_size = 64
        # loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

        for i, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                images = fixed_image_standardization(batch['image'].to(device))#[None, ...]
                batch_ = {}
                batch_["image"] = images
                codedict = reconstruction_net.encode(batch_, training=False)
                # opdict, visdict = reconstruction_net.decode(codedict)
                if codedict_retarget is not None:
                    codedict["shapecode"] = codedict_retarget["shapecode"].repeat(batch_["image"].shape[0], 1,)
                    codedict["detailcode"] = codedict_retarget["detailcode"].repeat(batch_["image"].shape[0], 1,)
                codedict = reconstruction_net.decode(codedict, training=False)

                uv_detail_normals = None
                if 'uv_detail_normals' in codedict.keys():
                    uv_detail_normals = codedict['uv_detail_normals']
                if rec_method in ['emoca', 'deca']:
                    visdict, grid_image = reconstruction_net._visualization_checkpoint(codedict['verts'],
                                                                                codedict['trans_verts'],
                                                                                codedict['ops'],
                                                           uv_detail_normals, codedict, 0, "train", "")
                else:
                    visdict = reconstruction_net._visualization_checkpoint(batch_["image"].shape[0], batch_, codedict, i, "", "")
                values = util.dict_tensor2npy(codedict)
                #TODO: verify axis
                # vis_im = np.split(vis_im, axis=0 ,indices_or_sections=batch_size)
                for j in range(images.shape[0]):
                    path = Path(batch['path'][j])
                    name = path.stem

                    if save_obj:
                        # if i*j == 0:
                        mesh_folder = out_folder / 'meshes'
                        mesh_folder.mkdir(exist_ok=True, parents=True)
                        reconstruction_net.deca.save_obj(str(mesh_folder / (name + '.obj')), values)
                    if save_mat:
                        # if i*j == 0:
                        mat_folder = out_folder / 'mat'
                        mat_folder.mkdir(exist_ok=True, parents=True)
                        savemat(str(mat_folder / (name + '.mat')), values)
                    if save_vis or save_video:
                        # if i*j == 0:
                        vis_folder = out_folder / 'vis'
                        vis_folder.mkdir(exist_ok=True, parents=True)
                        vis_dict_j = {key: value[j:j+1, ...] for key,value in visdict.items()}
                        with torch.no_grad():
                            # vis_im = reconstruction_net.deca.visualize(vis_dict_j, savepath=None, catdim=2)
                            vis_im = reconstruction_net.visualize(vis_dict_j, savepath=None, catdim=2)
                        if save_vis:
                            # cv2.imwrite(str(vis_folder / (name + '.jpg')), vis_im)
                            cv2.imwrite(str(vis_folder / (name + '.png')), vis_im)
                        if save_video and video_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            # video_writer = cv2.VideoWriter(filename=str(vis_folder / "video.mp4"), apiPreference=cv2.CAP_FFMPEG,
                            #                                fourcc=fourcc, fps=dm.video_metas[sequence_id]['fps'], frameSize=(vis_im.shape[1], vis_im.shape[0]))
                            video_writer = cv2.VideoWriter(str(vis_folder / "video.mp4"), cv2.CAP_FFMPEG,
                                                           fourcc, int(self.video_metas[sequence_id]['fps'].split('/')[0]),
                                                           (vis_im.shape[1], vis_im.shape[0]), True)
                        if save_video:
                            video_writer.write(vis_im)
                    if save_images:
                        # if i*j == 0:
                        ims_folder = out_folder / 'ims'
                        ims_folder.mkdir(exist_ok=True, parents=True)
                        for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                            if vis_name not in visdict.keys():
                                continue
                            image = util.tensor2image(visdict[vis_name][j])
                            Path(ims_folder / vis_name).mkdir(exist_ok=True, parents=True)
                            cv2.imwrite(str(ims_folder / vis_name / (name +'.png')), image)
        if video_writer is not None:
            video_writer.release()
        print("Done running face reconstruction in sequence '%s'" % self.video_list[sequence_id])

    def _gather_data(self, exist_ok=False):
        print("Processing dataset")
        Path(self.output_dir).mkdir(parents=True, exist_ok=exist_ok)

        video_list = sorted(Path(self.root_dir).rglob("*.mp4"))
        self.video_list = [path.relative_to(self.root_dir) for path in video_list]

        annotation_list = sorted(Path(self.root_dir).rglob("*.txt"))
        self.annotation_list = [path.relative_to(self.root_dir) for path in annotation_list]

        self._gather_video_metadata()
        print("Found %d video files." % len(self.video_list))

    def _gather_video_metadata(self):
        import ffmpeg
        self.video_metas = []
        for vi, vid_file in enumerate(tqdm(self.video_list)):
            vid = ffmpeg.probe(str( Path(self.root_dir) / vid_file))
            # codec_idx = [idx for idx in range(len(vid)) if vid['streams'][idx]['codec_type'] == 'video']
            codec_idx = [idx for idx in range(len(vid)) if vid['streams'][0]['codec_type'] == 'video']
            if len(codec_idx) == 0:
                raise RuntimeError("Video file has no video streams! '%s'" % str(vid_file))
            # if len(codec_idx) > 1:
                # raise RuntimeError("Video file has two video streams! '%s'" % str(vid_file))
            print("[WARNING] Video file has %d video streams. Only the first one will be processed" % len(codec_idx))
            codec_idx = codec_idx[0]
            vid_info = vid['streams'][codec_idx]
            vid_meta = {}
            vid_meta['fps'] = vid_info['avg_frame_rate']
            vid_meta['width'] = int(vid_info['width'])
            vid_meta['height'] = int(vid_info['height'])
            vid_meta['num_frames'] = int(vid_info['nb_frames'])
            self.video_metas += [vid_meta]

    def _loadMeta(self):
        if self.loaded:
            print("FaceVideoDataset already loaded.")
            return
        print(f"Loading metadata of FaceVideoDataset from: '{self.metadata_path}'")
        with open(self.metadata_path, "rb") as f:
            version = pkl.load(f)
            self.video_list = pkl.load(f)
            self.video_metas = pkl.load(f)
            self.annotation_list = pkl.load(f)
            # try:
            self.frame_lists = pkl.load(f)
            # except Exception:
            #     pass
        self.loaded = True

    def _saveMeta(self):
        with open(self.metadata_path, "wb") as f:
            pkl.dump(self.version, f)
            pkl.dump(self.video_list, f)
            pkl.dump(self.video_metas, f)
            pkl.dump(self.annotation_list,f)
            pkl.dump(self.frame_lists, f)


    def setup(self, stage: Optional[str] = None):
        # is dataset already processed?
        if not Path(self.output_dir).is_dir():
            raise RuntimeError("The folder with the processed not not found")

        print("Loading the dataset")
        self._loadMeta()
        print("Dataset loaded")

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        pass

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self, sequence_ids, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:

        if sequence_ids is None:
            sequence_ids = list(range(self.num_sequences))
        elif isinstance(sequence_ids, int):
            sequence_ids = [sequence_ids, ]

        test_frames = []
        for sid in sequence_ids:
            test_frames += self.frame_lists[sid]
        torch.multiprocessing.set_start_method('spawn')
        test_frames = [str( Path(self.output_dir) / frame) for frame in test_frames]
        testdata = TestData(test_frames, iscrop=True, face_detector='fan')
        batch_size = 1
        return DataLoader(testdata, batch_size=batch_size, num_workers=1)

    @property
    def num_sequences(self):
        return len(self.video_list)

    def _get_detection_for_sequence(self, sid):
        out_folder = self._get_path_to_sequence_detections(sid)
        out_file = out_folder / "bboxes.pkl"
        if not out_file.exists():
            print("Detections don't exist")
            return [], [], [], 0
        detection_fnames, landmark_fnames, centers, sizes, last_frame_id = \
            FaceVideoDataModule.load_detections(out_file)

        # make the paths relative if their not, this should not be neccesary in once everythings is reprocessed
        # stuf is now being saved with relative paths.
        relative_detection_fnames = []
        relative_landmark_fnames = []

        for fname_list in detection_fnames:
            relative_detection_list = []
            for fname in fname_list:
                if fname.is_absolute():
                    try:
                        relative_detection_list += [fname.relative_to(self.output_dir)]
                    except:
                        #TODO: remove ugly hack once reprocessed
                        relative_detection_list += [fname.relative_to(Path('/ps/scratch/rdanecek/data/aff-wild2/processed/processed_2020_Dec_21_00-30-03'))]
                else:
                    relative_detection_list += [fname]
            #TODO: this hack should not be necessary anymore
            #TODO: landmark fnames should not need this

            relative_detection_fnames += [relative_detection_list]

        return relative_detection_fnames, centers, sizes, last_frame_id

    def _get_validated_annotations_for_sequence(self, sid, crash_on_failure=True):
        out_folder = self._get_path_to_sequence_detections(sid)
        out_file = out_folder / "valid_annotations.pkl"
        if not out_file.exists():
            print(f"Annotations in file {out_file} don't exist")
            if crash_on_failure:
                raise RuntimeError(f"Annotations in file {out_file} don't exist")
            else:
                return None, None, None, None, None
        detection_fnames, annotations, recognition_labels, \
        discarded_annotations, detection_not_found \
            = FaceVideoDataModule._load_annotations(out_file)

        return detection_fnames, annotations, recognition_labels, discarded_annotations, detection_not_found

    def _get_reconstructions_for_sequence(self, sid, rec_method='emoca', retarget_suffix=None):
        out_folder = self._get_path_to_sequence_reconstructions(sid, rec_method=rec_method, suffix=retarget_suffix)
        vis_fnames = sorted(list((out_folder / "vis").glob("*.png")))
        if len(vis_fnames) == 0:
            vis_fnames = sorted(list((out_folder / "vis").glob("*.jpg")))
        return vis_fnames

    def _get_frames_for_sequence(self, sid):
        out_folder = self._get_path_to_sequence_frames(sid)
        vid_frames = sorted(list(out_folder.glob("*.png")))
        return vid_frames

    def _get_annotations_for_sequence(self, sid):
        video_file = self.video_list[sid]
        suffix = Path(self._video_category(sid)) / 'annotations' /self._video_set(sequence_id)
        annotation_prefix = Path(self.root_dir / suffix)
        annotation = sorted(annotation_prefix.glob(video_file.stem + "*.txt"))
        return annotation

    def _get_processed_annotations_for_sequence(self, sid):
        pass

    def _get_recognition_for_sequence(self, sid, distance_threshold=None):
        distance_threshold = distance_threshold or self.get_default_recognition_threshold()
        # out_folder = self._get_path_to_sequence_detections(sid)
        recognition_path = self._get_recognition_filename(sid, distance_threshold)
        # recognition_path = out_folder / "recognition.pkl"
        indices, labels, mean, cov, fnames = FaceVideoDataModule._load_recognitions(recognition_path)
        return indices, labels, mean, cov, fnames

    def create_reconstruction_video(self, sequence_id, overwrite=False, distance_threshold=0.5,
                                    rec_method='emoca', image_type=None, retarget_suffix=None, cat_dim=0, include_transparent=True):
        from PIL import Image, ImageDraw
        # fid = 0
        image_type = image_type or "detail"
        detection_fnames, centers, sizes, last_frame_id = self._get_detection_for_sequence(sequence_id)
        vis_fnames = self._get_reconstructions_for_sequence(sequence_id, rec_method=rec_method, 
            retarget_suffix=retarget_suffix, image_type=image_type)

        vid_frames = self._get_frames_for_sequence(sequence_id)

        vis_fnames.sort()
        vid_frames.sort()

        if image_type == "detail":
            outfile = vis_fnames[0].parents[1] / "video.mp4"
        else:
            outfile = vis_fnames[0].parents[1] /( "video_" + image_type + ".mp4")

        print("Creating reconstruction video for sequence num %d: '%s' " % (sequence_id, self.video_list[sequence_id]))
        if outfile.exists() and not overwrite:
            print("output file already exists")
            attach_audio_to_reconstruction_video(outfile, self.root_dir / self.video_list[sequence_id],
                                                 overwrite=overwrite)
            return

        writer = None  # cv2.VideoWriter()
        broken = False
        did = 0
        for fid in tqdm(range(len(vid_frames))):
            if broken:
                break

            frame_name = vid_frames[fid]
            c = centers[fid]
            s = sizes[fid]

            frame = imread(frame_name)


            frame_pill_bb = Image.fromarray(frame)
            if retarget_suffix is None:
                frame_deca_full = Image.fromarray(frame)
                frame_deca_trans = Image.fromarray(frame)
            else:
                frame_deca_full = Image.fromarray(np.zeros_like( frame))
                frame_deca_trans = Image.fromarray(np.zeros_like( frame))

            frame_draw = ImageDraw.Draw(frame_pill_bb)

            for nd in range(len(c)):
                detection_name = detection_fnames[fid][nd]

                if did >= len(vis_fnames):
                    broken = True
                    break

                vis_name = vis_fnames[did]

                if detection_name.stem not in str(vis_name):
                    print("%s != %s" % (detection_name.stem, vis_name.stem))
                    raise RuntimeError("Detection and visualization filenames should match but they don't.")

                try:
                    detection_im = imread(self.output_dir / detection_name)
                except:
                    # ugly hack to deal with the old AffWild2 dataset
                    detection_im = imread(self.output_dir / detection_name.relative_to(detection_name.parents[4]))
                try:
                    vis_im = imread(vis_name)
                except ValueError as e:
                    continue

                im_r = vis_im.shape[0]
                im_c = vis_im.shape[1]
                num_ims = im_c // im_r

                if image_type == "coarse":
                    vis_im = vis_im[:, im_r*3:im_r*4, ...] # coarse
                elif image_type == "detail":
                    vis_im = vis_im[:, im_r*4:im_r*5, ...] # detail

                    
                # vis_im = vis_im[:, :, ...]

                # vis_mask = np.prod(vis_im, axis=2) == 0
                vis_mask = (np.prod(vis_im, axis=2) > 30).astype(np.uint8) * 255

                # vis_im = np.concatenate([vis_im, vis_mask[..., np.newaxis]], axis=2)

                warped_im = bbpoint_warp(vis_im, c[nd], s[nd], detection_im.shape[0],
                                         output_shape=(frame.shape[0], frame.shape[1]), inv=False)
                # warped_im = bbpoint_warp(vis_im, c[nd], s[nd], frame.shape[0], frame.shape[1], False)
                warped_mask = bbpoint_warp(vis_mask, c[nd], s[nd], detection_im.shape[0],
                                           output_shape=(frame.shape[0], frame.shape[1]), inv=False)
                # warped_mask = bbpoint_warp(vis_mask, c[nd], s[nd], frame.shape[0], frame.shape[1], False)

                # dst_image = bbpoint_warp(frame, c[nd], s[nd], warped_im.shape[0])
                # frame2 = bbpoint_warp(dst_image, c[nd], s[nd], warped_im.shape[0], output_shape=(frame.shape[0], frame.shape[1]), inv=False)

                # frame_pil = Image.fromarray(frame)
                # frame_pil2 = Image.fromarray(frame)
                vis_pil = Image.fromarray((warped_im * 255).astype(np.uint8))
                mask_pil = Image.fromarray((warped_mask * 255).astype(np.uint8))
                if include_transparent:
                    mask_pil_transparent = Image.fromarray((warped_mask * 196).astype(np.uint8))

                bb = point2bbox(c[nd], s[nd])

                frame_draw.rectangle(((bb[0, 0], bb[0, 1],), (bb[2, 0], bb[1, 1],)),
                                     outline='green', width=5)
                frame_deca_full.paste(vis_pil, (0, 0), mask_pil)
                if include_transparent:
                    frame_deca_trans.paste(vis_pil, (0, 0), mask_pil_transparent)
                # tform =
                did += 1

            final_im = np.array(frame_pill_bb)
            final_im2 = np.array(frame_deca_full)
            if include_transparent:
                final_im3 = np.array(frame_deca_trans)

            if cat_dim is None:
                if final_im.shape[0] > final_im.shape[1]:
                    cat_dim = 1
                else:
                    cat_dim = 0
            
            if include_transparent:
                im = np.concatenate([final_im, final_im2, final_im3], axis=cat_dim)
            else: 
                im = np.concatenate([final_im, final_im2,], axis=cat_dim)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # outfile = str(vis_folder / "video.mp4")
                
                # fps = int(self.video_metas[sequence_id]['fps'].split('/')[0])
                fps = int(self.video_metas[sequence_id]['fps'].split('/')[0]) / int(self.video_metas[sequence_id]['fps'].split('/')[1])
                writer = cv2.VideoWriter(str(outfile), cv2.CAP_FFMPEG,
                                         fourcc, fps,
                                         (im.shape[1], im.shape[0]), True)
                
            im_cv = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            writer.write(im_cv)
        writer.release()
        attach_audio_to_reconstruction_video(outfile, self.root_dir / self.video_list[sequence_id])
        # plt.figure()
        # plt.imshow(im)
        # plt.show()
        # dst_image = warp(vis_im, tform.inverse, output_shape=frame.shape[:2])


    def create_reconstruction_video_with_recognition(self, sequence_id, overwrite=False, distance_threshold=0.5):
        from PIL import Image, ImageDraw, ImageFont
        from collections import Counter
        from matplotlib.pyplot import  get_cmap
        # fid = 0
        detection_fnames, centers, sizes, last_frame_id = self._get_detection_for_sequence(sequence_id)
        vis_fnames = self._get_reconstructions_for_sequence(sequence_id)
        vid_frames = self._get_frames_for_sequence(sequence_id)
        indices, labels, mean, cov, fnames = self._get_recognition_for_sequence(sequence_id, distance_threshold=distance_threshold )

        classification = self._recognition_discriminator(indices, labels, mean, cov, fnames)
        counter = Counter(indices)

        legit_colors = [c for c in classification.keys() if classification[c] ]
        num_colors = len(legit_colors)
        cmap = get_cmap('gist_rainbow')

        if distance_threshold == self.get_default_recognition_threshold():
            baseoutfile = "video_with_labels.mp4"
        else:
            baseoutfile = "video_with_labels_thresh_%.03f.mp4" % distance_threshold
        outfile = vis_fnames[0].parents[1] / baseoutfile

        print("Creating reconstruction video for sequence num %d: '%s' " % (sequence_id, self.video_list[sequence_id]))
        if outfile.exists() and not overwrite:
            print("output file already exists")
            attach_audio_to_reconstruction_video(outfile, self.root_dir / self.video_list[sequence_id],
                                                 overwrite=overwrite)
            return

        writer = None  # cv2.VideoWriter()
        broken = False
        did = 0
        for fid in tqdm(range(len(vid_frames))):
            if broken:
                break

            frame_name = vid_frames[fid]
            c = centers[fid]
            s = sizes[fid]

            frame = imread(frame_name)

            frame_pill_bb = Image.fromarray(frame)
            frame_deca_full = Image.fromarray(frame)
            frame_deca_trans = Image.fromarray(frame)

            frame_draw = ImageDraw.Draw(frame_pill_bb)

            for nd in range(len(c)):
                detection_name = detection_fnames[fid][nd]


                if did >= len(vis_fnames):
                    broken = True
                    break

                vis_name = vis_fnames[did]
                label = indices[did]
                valid_detection = classification[label]

                if detection_name.stem != vis_name.stem:
                    print("%s != %s" % (detection_name.stem, vis_name.stem))
                    raise RuntimeError("Detection and visualization filenames should match but they don't.")

                detection_im = imread(self.output_dir / detection_name.relative_to(detection_name.parents[4]))
                try:
                    vis_im = imread(vis_name)
                except ValueError as e:
                    continue

                im_r = vis_im.shape[0]
                im_c = vis_im.shape[1]
                num_ims = im_c // im_r

                # vis_im = vis_im[:, r*3:r*4, ...] # coarse
                vis_im = vis_im[:, im_r*4:im_r*5, ...] # detail
                # vis_im = vis_im[:, :, ...]

                # vis_mask = np.prod(vis_im, axis=2) == 0
                vis_mask = (np.prod(vis_im, axis=2) > 30).astype(np.uint8) * 255

                # vis_im = np.concatenate([vis_im, vis_mask[..., np.newaxis]], axis=2)

                warped_im = bbpoint_warp(vis_im, c[nd], s[nd], detection_im.shape[0],
                                         output_shape=(frame.shape[0], frame.shape[1]), inv=False)
                # warped_im = bbpoint_warp(vis_im, c[nd], s[nd], frame.shape[0], frame.shape[1], False)
                warped_mask = bbpoint_warp(vis_mask, c[nd], s[nd], detection_im.shape[0],
                                           output_shape=(frame.shape[0], frame.shape[1]), inv=False)
                # warped_mask = bbpoint_warp(vis_mask, c[nd], s[nd], frame.shape[0], frame.shape[1], False)

                # dst_image = bbpoint_warp(frame, c[nd], s[nd], warped_im.shape[0])
                # frame2 = bbpoint_warp(dst_image, c[nd], s[nd], warped_im.shape[0], output_shape=(frame.shape[0], frame.shape[1]), inv=False)

                # frame_pil = Image.fromarray(frame)
                # frame_pil2 = Image.fromarray(frame)
                vis_pil = Image.fromarray((warped_im * 255).astype(np.uint8))
                mask_pil = Image.fromarray((warped_mask * 255).astype(np.uint8))
                mask_pil_transparent = Image.fromarray((warped_mask * 196).astype(np.uint8))

                bb = point2bbox(c[nd], s[nd])

                if valid_detection:
                    color = cmap(legit_colors.index(label)/num_colors)[:3]
                    color = tuple(int(c*255) for c in color)
                else:
                    color = 'black'

                frame_draw.rectangle(((bb[0, 0], bb[0, 1],), (bb[2, 0], bb[1, 1],)),
                                     outline=color, width=5)
                fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMonoBold.ttf", 60)
                frame_draw.text((bb[0, 0], bb[1, 1]+10,), str(label), font=fnt, fill=color)
                frame_deca_full.paste(vis_pil, (0, 0), mask_pil)
                frame_deca_trans.paste(vis_pil, (0, 0), mask_pil_transparent)
                # tform =
                did += 1

            final_im = np.array(frame_pill_bb)
            final_im2 = np.array(frame_deca_full)
            final_im3 = np.array(frame_deca_trans)

            if final_im.shape[0] > final_im.shape[1]:
                cat_dim = 1
            else:
                cat_dim = 0
            im = np.concatenate([final_im, final_im2, final_im3], axis=cat_dim)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # outfile = str(vis_folder / "video.mp4")
                writer = cv2.VideoWriter(str(outfile), cv2.CAP_FFMPEG,
                                         fourcc, int(self.video_metas[sequence_id]['fps'].split('/')[0]),
                                         (im.shape[1], im.shape[0]), True)
            im_cv = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            writer.write(im_cv)
            # imsave("test_%.05d.png" % fid, im )
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(im)
            # plt.show()
        writer.release()
        attach_audio_to_reconstruction_video(outfile, self.root_dir / self.video_list[sequence_id])

    def _draw_annotation(self, frame_draw : ImageDraw.Draw,
                         val_gt : dict, font, color):
        pass

    def create_reconstruction_video_with_recognition_and_annotations(
            self,
            sequence_id,
            overwrite=False,
            distance_threshold=None):

        from collections import Counter
        from matplotlib.pyplot import get_cmap

        distance_threshold = distance_threshold or self.get_default_recognition_threshold()

        detection_fnames, centers, sizes, last_frame_id = self._get_detection_for_sequence(sequence_id)
        vis_fnames = self._get_reconstructions_for_sequence(sequence_id)
        vid_frames = self._get_frames_for_sequence(sequence_id)
        indices, labels, mean, cov, fnames = self._get_recognition_for_sequence(sequence_id, distance_threshold=distance_threshold )

        validated_detection_fnames, validated_annotations, validated_recognition_labels, discarded_annotations, detection_not_found \
            = self._get_validated_annotations_for_sequence(sequence_id)

        classification = self._recognition_discriminator(indices, labels, mean, cov, fnames)
        counter = Counter(indices)

        legit_colors = [c for c in classification.keys() if classification[c] ]
        num_colors = len(legit_colors)
        cmap = get_cmap('gist_rainbow')
        if distance_threshold == self.get_default_recognition_threshold():
            baseoutfile = "gt_video_with_labels.mp4"
        else:
            baseoutfile = "gt_video_with_labels_thresh_%.03f.mp4" % distance_threshold

        outfolder = self._get_path_to_sequence_detections(sequence_id) / "visualizations"
        outfolder.mkdir(exist_ok=True)

        outfile = outfolder / baseoutfile

        print("Creating reconstruction video for sequence num %d: '%s' " % (sequence_id, self.video_list[sequence_id]))
        if outfile.exists() and not overwrite:
            print("output file already exists")
            attach_audio_to_reconstruction_video(outfile, self.root_dir / self.video_list[sequence_id],
                                                 overwrite=overwrite)
            return

        writer = None  # cv2.VideoWriter()
        broken = False
        did = 0
        video_name = outfolder.parent.stem
        for fid in tqdm(range(len(vid_frames))):
            if broken:
                break

            frame_name = vid_frames[fid]
            c = centers[fid]
            s = sizes[fid]

            frame = imread(frame_name)

            frame_pill_bb = Image.fromarray(frame)
            frame_deca_full = Image.fromarray(frame)
            # frame_deca_trans = Image.fromarray(frame)

            frame_draw = ImageDraw.Draw(frame_pill_bb)

            for nd in range(len(c)):
                detection_name = detection_fnames[fid][nd]

                found = False
                for annotation_name in validated_detection_fnames.keys():
                    if detection_name in validated_detection_fnames[annotation_name]:
                        found = True
                        annotation_key = annotation_name
                        break

                if not found:
                    did += 1
                    continue
                else:
                    validated_detection_index = validated_detection_fnames[annotation_key].index(detection_name)
                    val_gt = OrderedDict()
                    for key, value in validated_annotations[annotation_key].items():
                        val_gt[key] = value[validated_detection_index]


                if did >= len(vis_fnames):
                    broken = True
                    break

                vis_name = vis_fnames[did]
                label = indices[did]
                valid_detection = classification[label]

                if detection_name.stem != vis_name.stem:
                    print("%s != %s" % (detection_name.stem, vis_name.stem))
                    raise RuntimeError("Detection and visualization filenames should match but they don't.")

                detection_im = imread(self.output_dir / detection_name.relative_to(detection_name.parents[4]))
                try:
                    vis_im = imread(vis_name)
                except ValueError as e:
                    continue

                vis_im = vis_im[:, -vis_im.shape[1] // 5:, ...]

                # vis_mask = np.prod(vis_im, axis=2) == 0
                vis_mask = (np.prod(vis_im, axis=2) > 30).astype(np.uint8) * 255

                # vis_im = np.concatenate([vis_im, vis_mask[..., np.newaxis]], axis=2)

                warped_im = bbpoint_warp(vis_im, c[nd], s[nd], detection_im.shape[0],
                                         output_shape=(frame.shape[0], frame.shape[1]), inv=False)
                # warped_im = bbpoint_warp(vis_im, c[nd], s[nd], frame.shape[0], frame.shape[1], False)
                warped_mask = bbpoint_warp(vis_mask, c[nd], s[nd], detection_im.shape[0],
                                           output_shape=(frame.shape[0], frame.shape[1]), inv=False)
                # warped_mask = bbpoint_warp(vis_mask, c[nd], s[nd], frame.shape[0], frame.shape[1], False)

                # dst_image = bbpoint_warp(frame, c[nd], s[nd], warped_im.shape[0])
                # frame2 = bbpoint_warp(dst_image, c[nd], s[nd], warped_im.shape[0], output_shape=(frame.shape[0], frame.shape[1]), inv=False)

                # frame_pil = Image.fromarray(frame)
                # frame_pil2 = Image.fromarray(frame)
                vis_pil = Image.fromarray((warped_im * 255).astype(np.uint8))
                mask_pil = Image.fromarray((warped_mask * 255).astype(np.uint8))
                # mask_pil_transparent = Image.fromarray((warped_mask * 196).astype(np.uint8))

                bb = point2bbox(c[nd], s[nd])

                if valid_detection:
                    color = cmap(legit_colors.index(label)/num_colors)[:3]
                    color = tuple(int(c*255) for c in color)
                else:
                    color = 'black'

                frame_draw.rectangle(((bb[0, 0], bb[0, 1],), (bb[2, 0], bb[1, 1],)),
                                     outline=color, width=5)
                fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMonoBold.ttf", 30)
                # all_str = str(label)
                frame_draw.text((bb[0, 0]-40, bb[1, 1]-40,), str(label), font=fnt, fill=color)
                self._draw_annotation(frame_draw, val_gt, fnt, color)
                frame_deca_full.paste(vis_pil, (0, 0), mask_pil)
                # frame_deca_trans.paste(vis_pil, (0, 0), mask_pil_transparent)
                # tform =
                did += 1

            final_im = np.array(frame_pill_bb)
            final_im2 = np.array(frame_deca_full)
            # final_im3 = np.array(frame_deca_trans)

            if final_im.shape[0] > final_im.shape[1]:
                cat_dim = 1
            else:
                cat_dim = 0
            im = np.concatenate([final_im, final_im2], axis=cat_dim)
            # im = np.concatenate([final_im, final_im2, final_im3], axis=cat_dim)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # outfile = str(vis_folder / "video.mp4")
                writer = cv2.VideoWriter(str(outfile), cv2.CAP_FFMPEG,
                                         fourcc, int(self.video_metas[sequence_id]['fps'].split('/')[0]),
                                         (im.shape[1], im.shape[0]), True)
            im_cv = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            writer.write(im_cv)
            # imsave("test_%.05d.png" % fid, im )
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(im)
            # plt.show()
        writer.release()
        attach_audio_to_reconstruction_video(outfile, self.root_dir / self.video_list[sequence_id])

    def _gather_detections_for_sequence(self, sequence_id, with_recognitions=True):
        out_folder = self._get_path_to_sequence_detections(sequence_id)
        out_file_detections = out_folder / "bboxes.pkl"
        if with_recognitions:
            out_file_recognitions = out_folder / "embeddings.pkl"

        if out_file_detections.exists() and (not with_recognitions or out_file_recognitions.exists()):
            detection_fnames, landmark_fnames, centers, sizes, last_frame_id = \
                FaceVideoDataModule.load_detections(out_file_detections)
            if with_recognitions:
                embeddings, recognized_detections_fnames = FaceVideoDataModule._load_face_embeddings(out_file_recognitions)
            print("Face detections for video %d found" % sequence_id)
        else:
            print("Faces for video %d not detected" % sequence_id)
            detection_fnames = []
            centers = []
            sizes = []
        if with_recognitions:
            return detection_fnames, landmark_fnames, centers, sizes, embeddings, recognized_detections_fnames
        return detection_fnames, landmark_fnames, centers, sizes

    @staticmethod
    def _recognition_discriminator(indices, labels, means, cov, fnames):
        from collections import Counter, OrderedDict
        counter = Counter(indices)
        min_occurences = 20
        min_sequence_length = 20
        classifications = OrderedDict()

        for label in counter.keys():
            classifications[label] = True
            if label == -1:
                classifications[label] = False
                continue
            if counter[label] < min_occurences:
                classifications[label] = False
                continue

            # count sequence lengths

            # add more conditions here

        return classifications

    def _get_recognition_filename(self, sequence_id, distance_threshold=None):
        if distance_threshold is None:
            distance_threshold = self.get_default_recognition_threshold()

        out_folder = self._get_path_to_sequence_detections(sequence_id)
        if distance_threshold != self.get_default_recognition_threshold():
            out_file = out_folder / ("recognition_dist_%.03f.pkl" % distance_threshold)
        else:
            out_file = out_folder / "recognition.pkl"
        return out_file

    def _identify_recognitions_for_sequence(self, sequence_id, distance_threshold = None):
        if distance_threshold is None:
            distance_threshold = self.get_default_recognition_threshold()
        print("Identifying recognitions for sequence %d: '%s'" % (sequence_id, self.video_list[sequence_id]))

        detection_fnames, landmark_fnames, centers, sizes, embeddings, recognized_detections_fnames = \
            self._gather_detections_for_sequence(sequence_id, with_recognitions=True)

        out_folder = self._get_path_to_sequence_detections(sequence_id)
        # if distance_threshold != 0.5:
        #     out_file = out_folder / ("recognition_dist_%.03f.pkl" % distance_threshold)
        # else:
        #     out_file = out_folder / "recognition.pkl"
        out_file = self._get_recognition_filename(sequence_id, distance_threshold)

        from collections import Counter, OrderedDict
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=distance_threshold)
        labels = clustering.fit_predict(X=embeddings)
        counter = Counter(labels)

        recognition_indices = OrderedDict()
        recognition_means = OrderedDict()
        recognition_cov = OrderedDict()
        recognition_fnames = OrderedDict()

        for label in counter.keys():
            # filter out outliers
            # if label == -1:
            #     continue
            # if counter[label] < min_occurences:
            #     continue
            indices = np.where(labels == label)[0]
            features = embeddings[indices]
            mean = np.mean(features, axis=0, keepdims=True)
            cov = np.cov(features, rowvar=False)
            try:
                recognized_filenames_label = sorted([recognized_detections_fnames[i].relative_to(
                    self.output_dir) for i in indices.tolist()])
            except ValueError:
                recognized_filenames_label = sorted([recognized_detections_fnames[i].relative_to(
                    recognized_detections_fnames[i].parents[4]) for i in indices.tolist()])

            recognition_indices[label] = indices
            recognition_means[label] = mean
            recognition_cov[label] = cov
            recognition_fnames[label] = recognized_filenames_label

        FaceVideoDataModule._save_recognitions(out_file, labels, recognition_indices, recognition_means,
                                               recognition_cov, recognition_fnames)
        print("Done identifying recognitions for sequence %d: '%s'" % (sequence_id, self.video_list[sequence_id]))


    @staticmethod
    def _save_recognitions(file, labels, indices, mean, cov, fnames):
        with open(file, "wb") as f:
            pkl.dump(labels, f)
            pkl.dump(indices, f)
            pkl.dump(mean, f)
            pkl.dump(cov, f)
            pkl.dump(fnames, f)

    @staticmethod
    def _load_recognitions(file):
        with open(file, "rb") as f:
            indices = pkl.load(f)
            labels = pkl.load(f)
            mean = pkl.load(f)
            cov = pkl.load(f)
            fnames = pkl.load(f)
        return indices, labels, mean, cov, fnames


    @staticmethod
    def _save_annotations(file, detection_fnames, annotations, recognition_labels,
                          discarded_annotations, detection_not_found ):
        with open(file, "wb") as f:
            pkl.dump(detection_fnames, f)
            pkl.dump(annotations, f)
            pkl.dump(recognition_labels, f)
            pkl.dump(discarded_annotations, f)
            pkl.dump(detection_not_found, f)

    @staticmethod
    def _load_annotations(file):
        with open(file, "rb") as f:
            detection_fnames = pkl.load(f)
            annotations = pkl.load(f)
            recognition_labels = pkl.load(f)
            discarded_annotations = pkl.load(f)
            detection_not_found = pkl.load(f)
        return detection_fnames, annotations, recognition_labels, discarded_annotations, detection_not_found

    def _gather_detections(self, with_recognitions=True):
        # out_files_detections = []
        # out_files_recognitions = []
        self.detection_fnames = []
        self.detection_centers = []
        self.detection_sizes = []
        self.detection_recognized_fnames = []
        self.detection_embeddings = []

        for sid in range(self.num_sequences):

            if with_recognitions:
                detection_fnames, centers, sizes, embeddings, recognized_detections_fnames = \
                    self._gather_detections_for_sequence(sid, with_recognitions=with_recognitions)
            else:
                detection_fnames, centers, sizes = \
                    self._gather_detections_for_sequence(sid, with_recognitions=with_recognitions)
                embeddings = None
                recognized_detections_fnames = None

            self.detection_fnames += [detection_fnames]
            self.detection_centers += [centers]
            self.detection_sizes += [sizes]
            self.detection_embeddings += [embeddings]
            self.detection_recognized_fnames += [recognized_detections_fnames]

    def _assign_gt_to_detections(self):
        for sid in range(self.num_sequences):
            self.assign_gt_to_detections_sequence(sid)

    def get_default_recognition_threshold(self):
        #TODO: ensure that 0.6 is good for the most part
        return 0.6

    def _map_detections_to_gt(self, detection_filenames : list, annotation_file,
                              annotation_type,
                              # specifier = ''
                              ):

        def find_correspondng_detections(frame_number):
            pattern = self.get_frame_number_format() % frame_number
            pattern += "_"
            res = [fname for fname in detection_filenames if pattern in str(fname)]
            return res

        def unlabeled_frame(type, annotation):
            # disregarding rules are taken from the AFF2-WILD website:
            # https://ibug.doc.ic.ac.uk/resources/aff-wild2/
            if type == 'va':
                # return annotation[0] == -5 and annotation[1] == -5
                return annotation[0] == -5 or annotation[1] == -5
            if type == 'expr7':
                return annotation == - 1
            if type == 'au8':
                return -1 in annotation
            raise ValueError(f"Invalid annotation type '{type}'")

        if annotation_type == 'va':
            dtype = np.float64
        elif annotation_type == 'expr7':
            dtype = np.int32
        elif annotation_type == 'au8':
            dtype = np.int32
        else:
            raise ValueError(f"Invalid annotation type '{annotation_type}'")
        annotation = np.loadtxt(annotation_file, skiprows=1, delimiter=',', dtype=dtype)

        valid_detection_list = []
        annotation_list = []
        discarded_list = []
        detection_not_found_list = []
        for i in range(annotation.shape[0]):
            frame_idx = i+1 # frame indices are 1-based
            if unlabeled_frame(annotation_type, annotation[i]):
                print(f"Frame {frame_idx} is not labeled. Skipping")
                discarded_list += [i]
                continue

            detections = find_correspondng_detections(frame_idx)
            if len(detections) == 0:
                print(f"Frame {frame_idx} has no corresponding detection for that identity")
                #TODO: a possible extension - check if there is any other detection at all and if it is assinged?
                detection_not_found_list += [i]
                continue
            if len(detections) > 1:
                # if specifier == '':
                print(f"Frame {frame_idx} has {len(detections)} detections of the equally "
                      f"classified identity. Only one will be taken")
                # elif specifier == '_left':
                #     other_center = get_bb_center_from_fname(
                #         other_detection_file_names[0], detection_fnames, detection_centers)
                #     main_center = get_bb_center_from_fname(
                #         main_detection_file_names[0], detection_fnames, detection_centers)
                #     print(
                #         f"Frame {frame_idx} has {len(detections)} detections of the equally "
                #         f"classified identity. Specifier is {specifier}.")
                # elif specifier == "_right":
                #     print(
                #         f"Frame {frame_idx} has {len(detections)} detections of the equally "
                #         f"classified identity. Specifier is {specifier}.")
                # else:
                #     raise ValueError(f"Invalid specifier to solve ambiguity: '{specifier}'")

                #TODO: what to do if there's more?


            valid_detection_list += [detections[0]]
            annotation_list += [annotation[i:i+1, ...]]


        annotation_array = np.concatenate(annotation_list, axis=0)
        annotation_dict = OrderedDict()
        annotation_dict[annotation_type] = annotation_array

        total_annotated_frames = annotation.shape[0] - len(discarded_list)
        total_annotated_detections = len(valid_detection_list)
        print(f"For {total_annotated_frames} annotations, "
              f"{total_annotated_detections} were found. Didn't catch "
              f"{total_annotated_frames - total_annotated_detections}")
        return valid_detection_list, annotation_dict, discarded_list, detection_not_found_list

    def _get_bb_from_fname(self, fname, detection_fname_list):
        # i1 = None
        # i2 = None
        # for i, fname_list in enumerate(detection_fname_list):
        # detection_fname_list contains per every frame a list of detections
        frame_num = int(fname.stem.split('_')[0])
        fname_list = detection_fname_list[frame_num-1]
        if fname in fname_list:
            # i1 = i
            i1 = frame_num-1 # careful, the frame numbers in filenames are 1-based (ffmpeg starts at 1)
            i2 = fname_list.index(fname)
            return i1, i2
        # This should never happen. It means the cached detections are out of sync and should be reprocessed.
        raise RuntimeError(f"The specified {fname} is not in the detection fname list.")

    def _get_bb_center_from_fname(self, fname, detection_fname_list, center_list):
        i1, i2 = self._get_bb_from_fname(fname, detection_fname_list)
        return center_list[i1][i2]

    # def _create_emotional_image_dataset(self,
    #                                     annotation_list=None,
    #                                     filter_pattern=None,
    #                                     with_landmarks=False,
    #                                     with_segmentation=False,
    #                                     crash_on_missing_file=False):
    #     annotation_list = annotation_list or ['va', 'expr7', 'au8']
    #     detections_all = []
    #     annotations_all = OrderedDict()
    #     for a in annotation_list:
    #         annotations_all[a] = []
    #     recognition_labels_all = []
    #
    #
    #     import re
    #     if filter_pattern is not None:
    #         # p = re.compile(filter_pattern)
    #         p = re.compile(filter_pattern, re.IGNORECASE)
    #
    #     for si in auto.tqdm(range(self.num_sequences)):
    #         sequence_name = self.video_list[si]
    #
    #         if filter_pattern is not None:
    #             res = p.match(str(sequence_name))
    #             if res is None:
    #                 continue
    #
    #         ## TODO: or more like an idea - a solution towards duplicate videos between va/au/expression set
    #         # would be to append the video path here to serve as a key in the dictionaries (instead of just the stem
    #         # of the path)
    #
    #         detection_fnames, annotations, recognition_labels, discarded_annotations, detection_not_found = \
    #             self._get_validated_annotations_for_sequence(si, crash_on_failure=False)
    #
    #         if detection_fnames is None:
    #             continue
    #
    #         current_list = annotation_list.copy()
    #         for annotation_name, detection_list in detection_fnames.items():
    #             detections_all += detection_list
    #             # annotations_all += [annotations[key]]
    #             for annotation_key in annotations[annotation_name].keys():
    #                 if annotation_key in current_list:
    #                     current_list.remove(annotation_key)
    #                 array = annotations[annotation_name][annotation_key]
    #                 annotations_all[annotation_key] += array.tolist()
    #                 n = array.shape[0]
    #
    #             recognition_labels_all += len(detection_list)*[annotation_name + "_" + str(recognition_labels[annotation_name])]
    #             if len(current_list) != len(annotation_list):
    #                 print("No desired GT is found. Skipping sequence %d" % si)
    #
    #             for annotation_name in current_list:
    #                 annotations_all[annotation_name] += [None] * n
    #
    #     print("Data gathered")
    #     print(f"Found {len(detections_all)} detections with annotations "
    #           f"of {len(set(recognition_labels_all))} identities")
    #
    #     # #TODO: delete debug code:
    #     # N = 3000
    #     # detections = detections[:N] + detections[-N:]
    #     # recognition_labels_all = recognition_labels_all[:N] + recognition_labels_all[-N:]
    #     # for key in annotations_all.keys():
    #     #     annotations_all[key] = annotations_all[key][:N] + annotations_all[key][-N:]
    #     # # end debug code : todo remove
    #
    #     invalid_indices = set()
    #     if not with_landmarks:
    #         landmarks = None
    #     else:
    #         landmarks = []
    #         print("Checking if every frame has a corresponding landmark file")
    #         for det_i, det in enumerate(auto.tqdm(detections_all)):
    #             lmk = det.parents[3]
    #             lmk = lmk / "landmarks" / (det.relative_to(lmk / "detections"))
    #             lmk = lmk.parent / (lmk.stem + ".pkl")
    #             file_exists = (self.output_dir / lmk).is_file()
    #             if not file_exists and crash_on_missing_file:
    #                 raise RuntimeError(f"Landmark does not exist {lmk}")
    #             elif not file_exists:
    #                 invalid_indices.add(det_i)
    #             landmarks += [lmk]
    #
    #     if not with_segmentation:
    #         segmentations = None
    #     else:
    #         segmentations = []
    #         print("Checking if every frame has a corresponding segmentation file")
    #         for det_i, det in enumerate(auto.tqdm(detections_all)):
    #             seg = det.parents[3]
    #             seg = seg / "segmentations" / (det.relative_to(seg / "detections"))
    #             seg = seg.parent / (seg.stem + ".pkl")
    #             file_exists = (self.output_dir / seg).is_file()
    #             if not file_exists and crash_on_missing_file:
    #                 raise RuntimeError(f"Landmark does not exist {seg}")
    #             elif not file_exists:
    #                 invalid_indices.add(det_i)
    #             segmentations += [seg]
    #
    #     invalid_indices = sorted(list(invalid_indices), reverse=True)
    #     for idx in invalid_indices:
    #         del detections_all[idx]
    #         del landmarks[idx]
    #         del segmentations[idx]
    #         del recognition_labels_all[idx]
    #         for key in annotations_all.keys():
    #             del annotations_all[key][idx]
    #
    #     return detections_all, landmarks, segmentations, annotations_all, recognition_labels_all
    #
    #
    # def get_annotated_emotion_dataset(self,
    #                                   annotation_list = None,
    #                                   filter_pattern=None,
    #                                   image_transforms=None,
    #                                   split_ratio=None,
    #                                   split_style=None,
    #                                   with_landmarks=False,
    #                                   with_segmentations=False,
    #                                   K=None,
    #                                   K_policy=None,
    #                                   # if you add more parameters here, add them also to the hash list
    #                                   load_from_cache=True # do not add this one to the hash list
    #                                   ):
    #     # Process the dataset
    #     str_to_hash = pkl.dumps(tuple([annotation_list, filter_pattern]))
    #     inter_cache_hash = hashlib.md5(str_to_hash).hexdigest()
    #     inter_cache_folder = Path(self.output_dir) / "cache" / str(inter_cache_hash)
    #     if (inter_cache_folder / "lists.pkl").exists() and load_from_cache:
    #         print(f"Found processed filelists in '{str(inter_cache_folder)}'. "
    #               f"Reprocessing will not be needed. Loading ...")
    #         with open(inter_cache_folder / "lists.pkl", "rb") as f:
    #             detections = pkl.load(f)
    #             landmarks = pkl.load(f)
    #             segmentations = pkl.load(f)
    #             annotations = pkl.load(f)
    #             recognition_labels = pkl.load(f)
    #         print("Loading done")
    #
    #     else:
    #         detections, landmarks, segmentations, annotations, recognition_labels = \
    #             self._create_emotional_image_dataset(
    #                 annotation_list, filter_pattern, with_landmarks, with_segmentations)
    #         inter_cache_folder.mkdir(exist_ok=True, parents=True)
    #         print(f"Dataset processed. Saving into: '{str(inter_cache_folder)}'.")
    #         with open(inter_cache_folder / "lists.pkl", "wb") as f:
    #             pkl.dump(detections, f)
    #             pkl.dump(landmarks, f)
    #             pkl.dump(segmentations, f)
    #             pkl.dump(annotations, f)
    #             pkl.dump(recognition_labels, f)
    #         print(f"Saving done.")
    #
    #     if split_ratio is not None and split_style is not None:
    #
    #         hash_list = tuple([annotation_list,
    #                            filter_pattern,
    #                            split_ratio,
    #                            split_style,
    #                            with_landmarks,
    #                            with_segmentations,
    #                            K,
    #                            K_policy,
    #                            # add new parameters here
    #                            ])
    #         cache_hash = hashlib.md5(pkl.dumps(hash_list)).hexdigest()
    #         cache_folder = Path(self.output_dir) / "cache" / "tmp" / str(cache_hash)
    #         cache_folder.mkdir(exist_ok=True, parents=True)
    #         # load from cache if exists
    #         if load_from_cache and (cache_folder / "lists_train.pkl").is_file() and \
    #             (cache_folder / "lists_val.pkl").is_file():
    #             print(f"Dataset split found in: '{str(cache_folder)}'. Loading ...")
    #             with open(cache_folder / "lists_train.pkl", "rb") as f:
    #                 # training
    #                  detection_train = pkl.load(f)
    #                  landmarks_train = pkl.load(f)
    #                  segmentations_train = pkl.load(f)
    #                  annotations_train = pkl.load(f)
    #                  recognition_labels_train = pkl.load(f)
    #                  idx_train = pkl.load(f)
    #             with open(cache_folder / "lists_val.pkl", "rb") as f:
    #                 # validation
    #                  detection_val = pkl.load(f)
    #                  landmarks_val = pkl.load(f)
    #                  segmentations_val = pkl.load(f)
    #                  annotations_val = pkl.load(f)
    #                  recognition_labels_val = pkl.load(f)
    #                  idx_val = pkl.load(f)
    #             print("Loading done")
    #         else:
    #             print(f"Splitting the dataset. Split style '{split_style}', split ratio: '{split_ratio}'")
    #             if image_transforms is not None:
    #                 if not isinstance(image_transforms, list) or len(image_transforms) != 2:
    #                     raise ValueError("You have to provide image transforms for both trainng and validation sets")
    #             idxs = np.arange(len(detections), dtype=np.int32)
    #             if split_style == 'random':
    #                 np.random.seed(0)
    #                 np.random.shuffle(idxs)
    #                 split_idx = int(idxs.size * split_ratio)
    #                 idx_train = idxs[:split_idx]
    #                 idx_val = idxs[split_idx:]
    #             elif split_style == 'manual':
    #                 idx_train = []
    #                 idx_val = []
    #                 for i, det in enumerate(auto.tqdm(detections)):
    #                     if 'Train_Set' in str(det):
    #                         idx_train += [i]
    #                     elif 'Validation_Set' in str(det):
    #                         idx_val += [i]
    #                     else:
    #                         idx_val += [i]
    #
    #             elif split_style == 'sequential':
    #                 split_idx = int(idxs.size * split_ratio)
    #                 idx_train = idxs[:split_idx]
    #                 idx_val = idxs[split_idx:]
    #             elif split_style == 'random_by_label':
    #                 idx_train = []
    #                 idx_val = []
    #                 unique_labels = sorted(list(set(recognition_labels)))
    #                 np.random.seed(0)
    #                 print(f"Going through {len(unique_labels)} unique labels and splitting its samples into "
    #                       f"training/validations set randomly.")
    #                 for li, label in enumerate(auto.tqdm(unique_labels)):
    #                     label_indices = np.array([i for i in range(len(recognition_labels)) if recognition_labels[i] == label],
    #                                              dtype=np.int32)
    #                     np.random.shuffle(label_indices)
    #                     split_idx = int(len(label_indices) * split_ratio)
    #                     i_train = label_indices[:split_idx]
    #                     i_val = label_indices[split_idx:]
    #                     idx_train += i_train.tolist()
    #                     idx_val += i_val.tolist()
    #                 idx_train = np.array(idx_train, dtype= np.int32)
    #                 idx_val = np.array(idx_val, dtype= np.int32)
    #             elif split_style == 'sequential_by_label':
    #                 idx_train = []
    #                 idx_val = []
    #                 unique_labels = sorted(list(set(recognition_labels)))
    #                 print(f"Going through {len(unique_labels)} unique labels and splitting its samples into "
    #                       f"training/validations set sequentially.")
    #                 for li, label in enumerate(auto.tqdm(unique_labels)):
    #                     label_indices = [i for i in range(len(recognition_labels)) if recognition_labels[i] == label]
    #                     split_idx = int(len(label_indices) * split_ratio)
    #                     i_train = label_indices[:split_idx]
    #                     i_val = label_indices[split_idx:]
    #                     idx_train += i_train
    #                     idx_val += i_val
    #                 idx_train = np.array(idx_train, dtype= np.int32)
    #                 idx_val = np.array(idx_val, dtype= np.int32)
    #             else:
    #                 raise ValueError(f"Invalid split style {split_style}")
    #
    #             if split_ratio < 0 or split_ratio > 1:
    #                 raise ValueError(f"Invalid split ratio {split_ratio}")
    #
    #             def index_list_by_list(l, idxs):
    #                 return [l[i] for i in idxs]
    #
    #             def index_dict_by_list(d, idxs):
    #                 res = d.__class__()
    #                 for key in d.keys():
    #                     res[key] = [d[key][i] for i in idxs]
    #                 return res
    #
    #             detection_train = index_list_by_list(detections, idx_train)
    #             annotations_train = index_dict_by_list(annotations, idx_train)
    #             recognition_labels_train = index_list_by_list(recognition_labels, idx_train)
    #             if with_landmarks:
    #                 landmarks_train = index_list_by_list(landmarks, idx_train)
    #             else:
    #                 landmarks_train = None
    #
    #             if with_segmentations:
    #                 segmentations_train = index_list_by_list(segmentations, idx_train)
    #             else:
    #                 segmentations_train = None
    #
    #             detection_val = index_list_by_list(detections, idx_val)
    #             annotations_val = index_dict_by_list(annotations, idx_val)
    #             recognition_labels_val = index_list_by_list(recognition_labels, idx_val)
    #
    #             if with_landmarks:
    #                 landmarks_val = index_list_by_list(landmarks, idx_val)
    #             else:
    #                 landmarks_val = None
    #
    #             if with_segmentations:
    #                 segmentations_val = index_list_by_list(segmentations, idx_val)
    #             else:
    #                 segmentations_val = None
    #
    #             print(f"Dataset split processed. Saving into: '{str(cache_folder)}'.")
    #             with open(cache_folder / "lists_train.pkl", "wb") as f:
    #                 # training
    #                 pkl.dump(detection_train, f)
    #                 pkl.dump(landmarks_train, f)
    #                 pkl.dump(segmentations_train, f)
    #                 pkl.dump(annotations_train, f)
    #                 pkl.dump(recognition_labels_train, f)
    #                 pkl.dump(idx_train, f)
    #             with open(cache_folder / "lists_val.pkl", "wb") as f:
    #                 # validation
    #                 pkl.dump(detection_val, f)
    #                 pkl.dump(landmarks_val, f)
    #                 pkl.dump(segmentations_val, f)
    #                 pkl.dump(annotations_val, f)
    #                 pkl.dump(recognition_labels_val, f)
    #                 pkl.dump(idx_val, f)
    #             print(f"Saving done.")
    #
    #         dataset_train = EmotionalImageDataset(
    #             detection_train,
    #             annotations_train,
    #             recognition_labels_train,
    #             image_transforms[0],
    #             self.output_dir,
    #             landmark_list=landmarks_train,
    #             segmentation_list=segmentations_train,
    #             K=K,
    #             K_policy=K_policy)
    #
    #         dataset_val = EmotionalImageDataset(
    #             detection_val,
    #             annotations_val,
    #             recognition_labels_val,
    #             image_transforms[1],
    #             self.output_dir,
    #             landmark_list=landmarks_val,
    #             segmentation_list=segmentations_val,
    #             # K=K,
    #             K=1,
    #             # K=None,
    #             # K_policy=K_policy)
    #             K_policy='sequential')
    #             # K_policy=None)
    #
    #         return dataset_train, dataset_val, idx_train, idx_val
    #
    #     # dataset = EmotionalImageDataset(
    #     dataset = EmotionalImageDataset(
    #         detections,
    #         annotations,
    #         recognition_labels,
    #         image_transforms,
    #         self.output_dir,
    #         landmark_list=landmarks,
    #         segmentation_list=segmentations,
    #         K=K,
    #         K_policy=K_policy)
    #     return dataset
    #
    #
    # def test_annotations(self, net=None, annotation_list = None, filter_pattern=None):
    #     net = net or self._get_emonet(self.device)
    #
    #     dataset = self.get_annotated_emotion_dataset(annotation_list, filter_pattern)


    def assign_gt_to_detections_sequence(self, sequence_id):
        print(f"Assigning GT to sequence {sequence_id}")

        def second_most_frequent_label():
            if len(most_frequent_labels) == 2:
                second_label = most_frequent_labels[1]
            elif len(most_frequent_labels) > 2:
                raise RuntimeError(f"Too many labels occurred with the same frequency. Unclear which one to pick.")
            else:
                most_frequent_count2 = list(counts2labels.keys())[1]
                most_frequent_labels2 = counts2labels[most_frequent_count2]
                if len(most_frequent_labels2) != 1:
                    raise RuntimeError(
                        f"Too many labels occurred with the same frequency. Unclear which one to pick.")
                second_label = most_frequent_labels2[0]
            return second_label

        def correct_left_right_order(left_center, right_center):
            left_right_dim = 0 # TODO: verify if this is correct
            if left_center[left_right_dim] < right_center[left_right_dim]:
                # left is on the left
                return 1
            elif left_center[left_right_dim] == right_center[left_right_dim]:
                # same place
                return 0
            # left is on the right
            return -1

        # detection_fnames = self._get_path_to_sequence_detections(sequence_id)
        # full_frames = self._get_frames_for_sequence(sequence_id)
        annotations = self._get_annotations_for_sequence(sequence_id)
        if len(annotations) == 0:
            print(f"No GT available for video '{self.video_list[sequence_id]}'")
            return
        annotation_type = annotations[0].parent.parent.parent.stem
        if annotation_type == 'AU_Set':
            anno_type = 'au8' # AU1,AU2,AU4,AU6,AU12,AU15,AU20,AU25
        elif annotation_type == 'Expression_Set':
            anno_type = 'expr7' # Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise
        elif annotation_type == 'VA_Set':
            anno_type = 'va' # valence arousal -1 to 1
        else:
            raise ValueError(f"Unsupported annotation type: '{annotation_type}'")

        # load the recognitions:
        # recognition_file = self._get_recognition_filename(
        #     sequence_id, self.get_default_recognition_threshold())
        # indices, labels, mean, cov, recognition_fnames = FaceVideoDataModule._load_recognitions(
        #     recognition_file)
        indices, labels, mean, cov, recognition_fnames = self._get_recognition_for_sequence(sequence_id)
        counts2labels = OrderedDict()
        for key, val in labels.items():
            if key == -1: # skip invalid outliers
                continue
            count = len(val)
            if count not in counts2labels.keys():
                counts2labels[count] = []
            counts2labels[count] += [key]

        recognition_label_dict = OrderedDict()
        annotated_detection_fnames = OrderedDict()
        validated_annotations = OrderedDict()
        discarded_annotations = OrderedDict()
        detection_not_found = OrderedDict()

        # suffs = [str(Path(str(anno)[len(str(anno.parent / self.video_list[sequence_id].stem)):]).stem) for anno in
        #      annotations]
        suffs = [str(anno.stem)[len(str(self.video_list[sequence_id].stem)):] for anno in
             annotations]

        ### WARNING: HORRIBLE THINGS FOLLOW, PUT ON YOUR PROTECTIVE GOGGLES BEFORE YOU PROCEED
        # this next section is a ugly rule-based approach to assign annotation files to detected and recognized
        # faces. This assignment is not provided by the authors of aff-wild2 and therefore it's approximated
        # using these rules that are taken from the readme.

        # THERE IS ONLY ONE DOMINANT DETECTION AND ONE ANNOTATION FILE:
        if len(annotations) == 1 and suffs[0] == '':
            most_frequent_count = list(counts2labels.keys())[0]
            most_frequent_labels = counts2labels[most_frequent_count]

            if len(most_frequent_labels) != 1:
                raise ValueError("There seem to be two people at the same time in all pictures but we only "
                                 "have annotation for one")

            main_label = most_frequent_labels[0]
            main_detection_file_names = recognition_fnames[main_label]
            main_annotation_file = annotations[0]
            main_valid_detection_list, main_valid_annotation_list, main_discarded_list, main_detection_not_found_list \
                = self._map_detections_to_gt(main_detection_file_names, main_annotation_file, anno_type)

            recognition_label_dict[main_annotation_file.stem] = main_label
            annotated_detection_fnames[main_annotation_file.stem] = main_valid_detection_list
            validated_annotations[main_annotation_file.stem] = main_valid_annotation_list
            discarded_annotations[main_annotation_file.stem] = main_discarded_list
            detection_not_found[main_annotation_file.stem] = main_detection_not_found_list


            # THERE ARE TWO DOMINANT DETECTIONS BUT ONLY ONE IS ANNOTATED
        elif len(annotations) == 1 and (suffs[0] == '_left' or suffs[0] == '_right'):

            most_frequent_count = list(counts2labels.keys())[0]
            most_frequent_labels = counts2labels[most_frequent_count]

            detection_fnames, detection_centers, detection_sizes, _ = \
                self._get_detection_for_sequence(sequence_id)

            if len(most_frequent_labels) != 1:
                raise ValueError("There seem to be two people at the same time in all pictures but we only "
                                 "have annotation for one")

            main_label = most_frequent_labels[0]
            main_detection_file_names = recognition_fnames[main_label]
            main_annotation_file = annotations[0]
            main_valid_detection_list, main_valid_annotation_list, main_discarded_list, main_detection_not_found_list  \
                = self._map_detections_to_gt(main_detection_file_names, main_annotation_file, anno_type)

            other_label = second_most_frequent_label()
            other_detection_file_names = recognition_fnames[other_label]
            other_annotation_file = annotations[0] # use the same annotation, which one will be used is figured out next
            other_valid_detection_list, other_valid_annotation_list, other_discarded_list, other_detection_not_found_list\
                = self._map_detections_to_gt(other_detection_file_names, other_annotation_file, anno_type)

            other_center = self._get_bb_center_from_fname(other_detection_file_names[0], detection_fnames,
                                                          detection_centers)
            main_center = self._get_bb_center_from_fname(main_detection_file_names[0], detection_fnames,
                                                         detection_centers)
            if correct_left_right_order(other_center, main_center) == 1:
                pass # do nothing, order correct
            elif correct_left_right_order(other_center, main_center) == -1:
                # swap main and other
                print("Swapping left and right")
                other_label, main_label = main_label, other_label
                # other_valid_detection_list, main_valid_detection_list = main_valid_detection_list, other_valid_detection_list
                # other_valid_annotation_list, main_valid_annotation_list = main_valid_annotation_list, other_valid_annotation_list
            else:
                raise ValueError("Detections are in the same place. No way to tell left from right")

            # now other is on the left, and main is on the right, decide which one is annotated based on the suffix
            if suffs[0] == '_left':
                print("Choosing left")
                recognition_label_dict[other_annotation_file.stem] = other_label
                annotated_detection_fnames[other_annotation_file.stem] = other_valid_detection_list
                validated_annotations[other_annotation_file.stem] = other_valid_annotation_list
                discarded_annotations[other_annotation_file.stem] = other_discarded_list
                detection_not_found[other_annotation_file.stem] = other_detection_not_found_list
            else: # suffs[0] == '_right':
                print("Choosing right")
                recognition_label_dict[main_annotation_file.stem] = main_label
                annotated_detection_fnames[main_annotation_file.stem] = main_valid_detection_list
                validated_annotations[main_annotation_file.stem] = main_valid_annotation_list
                discarded_annotations[main_annotation_file.stem] = main_discarded_list
                detection_not_found[main_annotation_file.stem] = main_detection_not_found_list
        else:
            if len(suffs) > 2:
                print(f"Unexpected number of suffixes found {len(suffs)}")
                print(suffs)
                raise RuntimeError(f"Unexpected number of suffixes found {len(suffs)}")

            most_frequent_count = list(counts2labels.keys())[0]
            most_frequent_labels = counts2labels[most_frequent_count]

            detection_fnames, detection_centers, detection_sizes, _ = \
                self._get_detection_for_sequence(sequence_id)

            # THE CASE OF ONE DOMINANT DETECTION AND ONE SMALLER ONE (NO SUFFIX vs LEFT/RIGHT)
            if suffs[0] == '' and (suffs[1] == '_left' or suffs[1] == '_right'):
                if len(most_frequent_labels) != 1:
                    raise ValueError("There seem to be two people at the same time in all pictures but we only "
                                     "have annotation for one")

                main_label = most_frequent_labels[0]
                main_detection_file_names = recognition_fnames[main_label]
                main_annotation_file = annotations[0]
                main_valid_detection_list, main_valid_annotation_list, main_discarded_list, main_detection_not_found_list\
                    = self._map_detections_to_gt(main_detection_file_names, main_annotation_file, anno_type)

                recognition_label_dict[main_annotation_file.stem] = main_label
                annotated_detection_fnames[main_annotation_file.stem] = main_valid_detection_list
                validated_annotations[main_annotation_file.stem] = main_valid_annotation_list
                discarded_annotations[main_annotation_file.stem] = main_discarded_list
                detection_not_found[main_annotation_file.stem] = main_detection_not_found_list


                other_label = most_frequent_labels[1]
                other_detection_file_names = recognition_fnames[other_label]
                other_annotation_file = annotations[1]
                other_valid_detection_list, other_valid_annotation_list, other_discarded_list, other_detection_not_found_list \
                    = self._map_detections_to_gt(other_detection_file_names, other_annotation_file, anno_type)

                recognition_label_dict[other_annotation_file.stem] = other_label
                annotated_detection_fnames[other_annotation_file.stem] = other_valid_detection_list
                validated_annotations[other_annotation_file.stem] = other_valid_annotation_list
                discarded_annotations[other_annotation_file.stem] = other_discarded_list
                detection_not_found[other_annotation_file.stem] = other_detection_not_found_list

                other_center = self._get_bb_center_from_fname(other_detection_file_names[0], detection_fnames,
                                                        detection_centers)
                main_center = self._get_bb_center_from_fname(main_detection_file_names[0], detection_fnames,
                                                       detection_centers)
                if suffs[1] == '_left':
                    if correct_left_right_order(other_center, main_center) != 1:
                        raise RuntimeError("The main detection should be on the right and the other on the left but this is not the case")
                elif suffs[1] == '_right':
                    if correct_left_right_order(main_center, other_center) != 1:
                        raise RuntimeError(
                            "The main detection should be on the left and the other on the right but this is not the case")

            # THE CASE OF TWO ROUGHLY EQUALY DOMINANT DETECTIONS (LEFT and RIGHT)
            elif suffs[0] == '_left' and suffs[1] == '_right':
                #TODO: figure out which one is left and which one is right by loading the bboxes and comparing
                counts2labels.keys()
                left_label = most_frequent_labels[0]
                # if len(most_frequent_labels) == 2:
                #     right_label = most_frequent_labels[1]
                # elif len(most_frequent_labels) > 2:
                #     raise RuntimeError(f"Too many labels occurred with the same frequency. Unclear which one to pick.")
                # else:
                #     most_frequent_count2 = list(counts2labels.keys())[1]
                #     most_frequent_labels2 = counts2labels[most_frequent_count2]
                #     if len(most_frequent_labels2) != 1:
                #         raise RuntimeError(
                #             f"Too many labels occurred with the same frequency. Unclear which one to pick.")
                #     right_label = most_frequent_labels2[0]
                right_label = second_most_frequent_label()

                left_filename = recognition_fnames[left_label][0]
                left_center = self._get_bb_center_from_fname(left_filename, detection_fnames, detection_centers)

                right_filename = recognition_fnames[right_label][0]
                right_center = self._get_bb_center_from_fname(right_filename, detection_fnames, detection_centers)

                order = correct_left_right_order(left_center, right_center)
                # if left is not left, swap
                if order == -1:
                    left_label, right_label = right_label, left_label
                    left_filename, right_filename = right_filename, left_filename
                elif order == 0:
                    raise RuntimeError("Left and right detections have centers in the same place. "
                                       "No way to tell left from right")

                left_detection_file_names = recognition_fnames[left_label]
                left_annotation_file = annotations[0]
                left_valid_detection_list, left_annotation_list, left_discarded_list, left_detection_not_found_list \
                    = self._map_detections_to_gt(left_detection_file_names, left_annotation_file, anno_type)
                recognition_label_dict[left_annotation_file.stem] = left_label
                annotated_detection_fnames[left_annotation_file.stem] = left_valid_detection_list
                validated_annotations[left_annotation_file.stem] = left_annotation_list
                discarded_annotations[left_annotation_file.stem] = left_discarded_list
                detection_not_found[left_annotation_file.stem] = left_detection_not_found_list



                right_detection_file_names = recognition_fnames[right_label]
                right_annotation_file = annotations[1]

                right_valid_detection_list, right_valid_annotation_list, right_discarded_list, right_detection_not_found_list \
                    = self._map_detections_to_gt(right_detection_file_names, right_annotation_file, anno_type)
                recognition_label_dict[right_annotation_file.stem] = right_label
                annotated_detection_fnames[right_annotation_file.stem] = right_valid_detection_list
                validated_annotations[right_annotation_file.stem] = right_valid_annotation_list
                discarded_annotations[right_annotation_file.stem] = right_discarded_list
                detection_not_found[right_annotation_file.stem] = right_detection_not_found_list

                # THE FOLLOWING CASE SHOULD NEVER HAPPEN
            else:
                print(f"Unexpected annotation case found.")
                print(suffs)
                raise RuntimeError(f"Unexpected annotation case found: {str(suffs)}")

        out_folder = self._get_path_to_sequence_detections(sequence_id)
        out_file = out_folder / "valid_annotations.pkl"
        FaceVideoDataModule._save_annotations(out_file, annotated_detection_fnames, validated_annotations,
                                              recognition_label_dict, discarded_annotations, detection_not_found)

def attach_audio_to_reconstruction_video(input_video, input_video_with_audio, output_video=None, overwrite=False):
    output_video = output_video or (Path(input_video).parent / (str(Path(input_video).stem) + "_with_sound.mp4"))
    if output_video.exists() and not overwrite:
        return
    output_video = str(output_video)
    cmd = "ffmpeg -y -i %s -i %s -c copy -map 0:0 -map 1:1 -shortest %s" \
          % (input_video, input_video_with_audio, output_video)
    os.system(cmd)



class TestFaceVideoDM(FaceVideoDataModule): 

    def __init__(self, video_path, output_dir, processed_subfolder="processed",
                 face_detector='fan',
                 face_detector_threshold=0.9,
                 image_size=224,
                 scale=1.25,
                 batch_size=8,
                 num_workers=4,
                 device=None):
        self.video_path = Path(video_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__(self.video_path.parent, output_dir, 
                processed_subfolder,
                 face_detector,
                 face_detector_threshold,
                 image_size,
                 scale,
                 device)
        
    
    def prepare_data(self, *args, **kwargs):
        outdir = Path(self.output_dir)

        # is dataset already processed?
        # if outdir.is_dir():
        if Path(self.metadata_path).is_file():
            print("The dataset is already processed. Loading")
            self._loadMeta()
            return
        # else:
        self._gather_data()
        self._unpack_videos() 
        self._detect_faces()
        self._saveMeta()

    # def _get_unpacked_video_subfolder(self, video_idx):
    #     return  self.video_path.stem

    def _gather_data(self, exist_ok=False):
        print("Processing dataset")
        Path(self.output_dir).mkdir(parents=True, exist_ok=exist_ok)

        # video_list = sorted(Path(self.root_dir).rglob("*.mp4"))
        self.video_list = [self.video_path.relative_to(self.root_dir)]

        self.annotation_list = []
        self._gather_video_metadata()


    def _get_path_to_sequence_results(self, sequence_id, rec_method='EMOCA', suffix=''):
        return self._get_path_to_sequence_files(sequence_id, "results", rec_method, suffix)

    def _get_reconstructions_for_sequence(self, sid, rec_method='emoca', retarget_suffix=None, image_type=None):
        out_folder = self._get_path_to_sequence_results(sid, rec_method=rec_method, 
            suffix=retarget_suffix)
        if image_type is None:
            image_type = "geometry_detail"
        vis_fnames = sorted(list(out_folder.glob(f"**/{image_type}.png")))
        return vis_fnames


    def _get_path_to_sequence_files(self, sequence_id, file_type, method="", suffix=""): 
        assert file_type in ['videos', 'detections', "landmarks", "segmentations", 
            "emotions", "reconstructions", "results"]
        video_file = self.video_list[sequence_id]
        if len(method) > 0:
            file_type += "/" + method 
        if suffix is not None and len(suffix) > 0:
            file_type += "/" + suffix

        suffix = Path( video_file.stem) / file_type  
        out_folder = Path(self.output_dir) / suffix
        return out_folder

    def setup(self, stage: Optional[str] = None):
        sequence_ids = [0]
        images = []
        for sid in sequence_ids:
            detection_path = self._get_path_to_sequence_detections(sid)
            images += sorted(list(detection_path.glob("*.png")))
        self.testdata = TestData(images, iscrop=False)


    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.testdata, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)