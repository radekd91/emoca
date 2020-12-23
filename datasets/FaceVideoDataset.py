from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

import glob, os, sys
from pathlib import Path
# import pyvista as pv
# from utils.mesh import load_mesh
# from scipy.io import wavfile
# import resampy
import numpy as np
import torch
# import torchaudio
# from enum import Enum
from typing import Optional, Union, List, Any
import pickle as pkl
# from collections import OrderedDict
from tqdm import tqdm
# import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DECA')))
# from decalib.deca import DECA
# from decalib.datasets import datasets
from utils.FaceDetector import FAN, MTCNN
from facenet_pytorch import InceptionResnetV1

class FaceVideoDataModule(pl.LightningDataModule):

    def __init__(self, root_dir, output_dir, processed_subfolder=None,
                 face_detector='fan',
                 image_size=224,
                 scale=1.25,
                 device=None
                 ):
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

        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if face_detector == 'fan':
            self.face_detector = FAN(device)
        elif face_detector == 'mtcnn':
            self.face_detector = MTCNN(device)
        else:
            raise ValueError("Invalid face detector spicifier '%s'" % face_detector)

        self.face_recognition = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        self.image_size = image_size
        self.scale = scale

        self.version = 0

        self.video_list = None
        self.video_metas = None
        self.annotation_list = None
        self.frame_lists = None
        self.detection_lists = None


    @property
    def metadata_path(self):
        return os.path.join(self.output_dir, "metadata.pkl")

    def prepare_data(self, *args, **kwargs):
        outdir = Path(self.output_dir)

        # is dataset already processed?
        if outdir.is_dir():
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

    def _unpack_video(self, video_idx, overwrite=False):
        video_file = Path(self.root_dir) / self.video_list[video_idx]
        suffix = Path(video_file.parts[-4]) / video_file.parts[-3] / video_file.parts[-2] / video_file.stem
        out_folder = Path(self.output_dir) / suffix

        if not out_folder.exists() or overwrite:
            print("Unpacking video to '%s'" % str(out_folder))
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

    def _detect_faces_in_sequence(self, sequence_id):
        # if self.detection_lists is None or len(self.detection_lists) == 0:
        #     self.detection_lists = [ [] for i in range(self.num_sequences)]
        video_file = self.video_list[sequence_id]
        print("Detecting faces in sequence: '%s'" % video_file)
        suffix = Path(video_file.parts[-4]) / 'detections' / video_file.parts[-2] / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        out_folder.mkdir(exist_ok=True, parents=True)

        centers_all = []
        sizes_all = []
        detection_fnames_all = []
        # save_folder = frame_fname.parents[3] / 'detections'
        for fid, frame_fname in enumerate(tqdm(self.frame_lists[sequence_id])):
            # detect faces in each frames
            detections, centers, sizes = self._detect_faces_in_image(Path(self.output_dir) / frame_fname)
            # self.detection_lists[sequence_id][fid] += [detections]
            centers_all += [centers]
            sizes_all += [sizes]

            # save detections
            detection_fnames = []
            for di, detection in enumerate(detections):
                out_fname = out_folder / (frame_fname.stem + "_%.03d.png" % di)
                detection_fnames += [out_fname]
                imsave(out_fname, detection)
            detection_fnames_all += [detection_fnames]

        FaceVideoDataModule.save_detections(out_folder / "bboxes.pkl",
                                            detection_fnames_all, centers_all, sizes_all)
        print("Done detecting faces in sequence: '%s'" % self.video_list[sequence_id])

    @staticmethod
    def save_detections(fname, detection_fnames, centers, sizes):
        with open(fname, "wb" ) as f:
            pkl.dump(detection_fnames, f)
            pkl.dump(centers, f)
            pkl.dump(sizes, f)

    @staticmethod
    def load_detections(fname):
        with open(fname, "rb" ) as f:
            detection_fnames = pkl.load(f)
            centers = pkl.load(f)
            sizes = pkl.load(f)
        return detection_fnames, centers, sizes

    def _detect_faces_in_image(self, image_path):
        # imagepath = self.imagepath_list[index]
        # imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(imread(image_path))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape
        bounding_boxes, bbox_type = self.face_detector.run(image)
        image = image / 255.
        detection_images = []
        detection_centers = []
        detection_sizes = []
        # detection_embeddings = []
        if len(bounding_boxes) == 0:
            print('no face detected! run original image')
            return detection_images, detection_centers, detection_images
            # left = 0
            # right = h - 1
            # top = 0
            # bottom = w - 1
            # bounding_boxes += [[left, right, top, bottom]]

        for bi, bbox in enumerate(bounding_boxes):
            left = bbox[0]
            right = bbox[2]
            top = bbox[1]
            bottom = bbox[3]
            old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size * self.scale)

            dst_image = bbpoint_warp(image, center, size, self.image_size)

            # dst_image = dst_image.transpose(2, 0, 1)
            #
            detection_images += [(dst_image*255).astype(np.uint8)]
            detection_centers += [center]
            detection_sizes += [size]

        return detection_images, detection_centers, detection_sizes

    def _gather_data(self, exist_ok=False):
        print("Processing dataset")
        Path(self.output_dir).mkdir(parents=True, exist_ok=exist_ok)

        video_list = sorted(Path(self.root_dir).rglob("*.mp4"))
        self.video_list = [path.relative_to(self.root_dir) for path in video_list]

        annotation_list = sorted(Path(self.root_dir).rglob("*.txt"))
        self.annotation_list = [path.relative_to(self.root_dir) for path in annotation_list]

        import ffmpeg

        self.video_metas = []
        for vi, vid_file in enumerate(tqdm(self.video_list)):
            vid = ffmpeg.probe(str( Path(self.root_dir) / vid_file))
            codec_idx = [idx for idx in range(len(vid)) if vid['streams'][idx]['codec_type'] == 'video']
            if len(codec_idx) > 1:
                raise RuntimeError("Video file has two video streams! '%s'" % str(vid_file))
            codec_idx = codec_idx[0]
            vid_info = vid['streams'][codec_idx]
            vid_meta = {}
            vid_meta['fps'] = vid_info['avg_frame_rate']
            vid_meta['width'] = int(vid_info['width'])
            vid_meta['height'] = int(vid_info['height'])
            vid_meta['num_frames'] = int(vid_info['nb_frames'])

            self.video_metas += [vid_meta]


        print("Found %d video files." % len(self.video_list))


    def _loadMeta(self):
        with open(self.metadata_path, "rb") as f:
            version = pkl.load(f)
            self.video_list = pkl.load(f)
            self.video_metas = pkl.load(f)
            self.annotation_list = pkl.load(f)
            try:
                self.frame_lists = pkl.load(f)
            except Exception:
                pass

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


    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     pass


import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io

from decalib.datasets import detectors

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


def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    else:
        raise NotImplementedError
    return old_size, center


def point2bbox(center, size):
    size2 = size / 2

    src_pts = np.array(
        [[center[0] - size2, center[1] - size2], [center[0] - size2, center[1] + size2],
         [center[0] + size2, center[1] - size2]])
    return src_pts


def point2transform(center, size, target_size):
    src_pts = point2bbox(center, size)
    dst_pts = np.array([[0, 0], [0, target_size - 1], [target_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform


def bbpoint_warp(image, center, size, target_size):
    tform = point2transform(center, size, target_size)
    dst_image = warp(image, tform.inverse, output_shape=(target_size, target_size),order=3)
    return dst_image


class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='mtcnn'):
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
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'fan':
            self.face_detector = detectors.FAN()
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
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print('no face detected! run original image')
                    left = 0
                    right = h - 1
                    top = 0
                    bottom = w - 1
                else:
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


def main():
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    subfolder = 'processed_2020_Dec_21_00-30-03'
    dm = FaceVideoDataModule(str(root_path), str(output_path), processed_subfolder=subfolder)
    dm.prepare_data()
    # dm._detect_faces()
    dm._detect_faces_in_sequence(400)
    print("Peace out")


if __name__ == "__main__":
    main()
