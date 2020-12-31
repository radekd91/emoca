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

from memory_profiler import profile


class FaceVideoDataModule(pl.LightningDataModule):

    def __init__(self, root_dir, output_dir, processed_subfolder=None,
                 face_detector='fan',
                 face_detector_threshold=0.5,
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

        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.face_detector_type = face_detector
        self.face_detector_threshold = face_detector_threshold

        # self._instantiate_detector()
        self.face_recognition = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        self.image_size = image_size
        self.scale = scale

        self.version = 0

        self.video_list = None
        self.video_metas = None
        self.annotation_list = None
        self.frame_lists = None
        # self.detection_lists = None

        self.detection_fnames = []
        self.detection_centers = []
        self.detection_sizes = []


    def _instantiate_detector(self):
        if hasattr(self, 'detector'):
            del self.detector
        if self.face_detector == 'fan':
            self.face_detector = FAN(self.device, threshold=self.face_detector_threshold)
        elif self.face_detector == 'mtcnn':
            self.face_detector = MTCNN(self.device)
        else:
            raise ValueError("Invalid face detector specifier '%s'" % self.face_detector)


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

    def _get_path_to_sequence_frames(self, sequence_id):
        video_file = self.video_list[sequence_id]
        suffix = Path(video_file.parts[-4]) / 'videos' / video_file.parts[-2] / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder

    def _get_path_to_sequence_detections(self, sequence_id):
        video_file = self.video_list[sequence_id]
        suffix = Path(video_file.parts[-4]) / 'detections' / video_file.parts[-2] / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder

    def _get_path_to_sequence_reconstructions(self, sequence_id):
        video_file = self.video_list[sequence_id]
        suffix = Path(video_file.parts[-4]) / 'reconstructions' / video_file.parts[-2] / video_file.stem
        out_folder = Path(self.output_dir) / suffix
        return out_folder

    @profile
    def _detect_faces_in_sequence(self, sequence_id):
        # if self.detection_lists is None or len(self.detection_lists) == 0:
        #     self.detection_lists = [ [] for i in range(self.num_sequences)]
        video_file = self.video_list[sequence_id]
        print("Detecting faces in sequence: '%s'" % video_file)
        suffix = Path(video_file.parts[-4]) / 'detections' / video_file.parts[-2] / video_file.stem
        out_folder = self._get_path_to_sequence_detections(sequence_id)
        out_folder.mkdir(exist_ok=True, parents=True)
        out_file = out_folder / "bboxes.pkl"

        centers_all = []
        sizes_all = []
        detection_fnames_all = []
        # save_folder = frame_fname.parents[3] / 'detections'

        # TODO: resuming is not tested, probably doesn't work yet
        checkpoint_frequency = 100
        resume = False
        if resume and out_file.exists():
            detection_fnames_all, centers_all, sizes_all, start_fid = \
                FaceVideoDataModule.load_detections(out_file)
        else:
            start_fid = 0

        # hack trying to circumvent memory leaks on the cluster
        detector_instantion_frequency = 200

        frame_list = self.frame_lists[sequence_id]
        fid = 0
        if len(frame_list) == 0:
            print("Nothing to detect in: '%s'. All frames have been processed" % self.video_list[sequence_id])
        for fid, frame_fname in enumerate(tqdm(range(start_fid, len(frame_list)))):

            if fid % detector_instantion_frequency == 0:
                self._instantiate_detector()

            frame_fname = frame_list[fid]
            # detect faces in each frames
            detections, centers, sizes = self._detect_faces_in_image(Path(self.output_dir) / frame_fname)
            # self.detection_lists[sequence_id][fid] += [detections]
            centers_all += [centers]
            sizes_all += [sizes]

            # save detections
            detection_fnames = []
            for di, detection in enumerate(detections):
                out_fname = out_folder / (frame_fname.stem + "_%.03d.png" % di)
                detection_fnames += [out_fname.relative_to(self.output_dir)]
                imsave(out_fname, detection)
            detection_fnames_all += [detection_fnames]

            if fid % checkpoint_frequency == 0:
                FaceVideoDataModule.save_detections(out_file,
                                                    detection_fnames_all, centers_all, sizes_all, fid)

        FaceVideoDataModule.save_detections(out_file,
                                            detection_fnames_all, centers_all, sizes_all, fid)
        print("Done detecting faces in sequence: '%s'" % self.video_list[sequence_id])

    @staticmethod
    def save_detections(fname, detection_fnames, centers, sizes, last_frame_id):
        with open(fname, "wb" ) as f:
            pkl.dump(detection_fnames, f)
            pkl.dump(centers, f)
            pkl.dump(sizes, f)
            pkl.dump(last_frame_id, f)

    @staticmethod
    def load_detections(fname):
        with open(fname, "rb" ) as f:
            detection_fnames = pkl.load(f)
            centers = pkl.load(f)
            sizes = pkl.load(f)
            try:
                last_frame_id = pkl.load(f)
            except:
                last_frame_id = -1
        return detection_fnames, centers, sizes, last_frame_id

    @profile
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
            # print('no face detected! run original image')
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

    def _get_recognition_net(self, device):
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        return resnet

    def _recognize_faces(self, device = None):
        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        recognition_net = self._get_recognition_net(device)
        for sid in range(self.num_sequences):
            self._recognize_faces_in_sequence(sid, recognition_net, device)

    def _recognize_faces_in_sequence(self, sequence_id, recognition_net=None, device=None):

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
        loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
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

    def _get_reconstruction_net(self, device):
        from decalib.deca import DECA
        from decalib.utils.config import cfg as deca_cfg
        # deca_cfg.model.use_tex = args.useTex
        deca_cfg.model.use_tex = False
        deca = DECA(config=deca_cfg, device=device)
        return deca

    def _reconstruct_faces(self, device = None):
        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        reconstruction_net = self._get_reconstruction_net(device)
        for sid in range(self.num_sequences):
            self._reconstruct_faces_in_sequence(sid, reconstruction_net, device)

    def _reconstruct_faces_in_sequence(self, sequence_id, reconstruction_net=None, device=None,
                                       save_obj=False, save_mat=True, save_vis=True, save_images=False,
                                       save_video=True):
        from decalib.utils import util
        from scipy.io.matlab import savemat

        def fixed_image_standardization(image):
            return image / 255.

        print("Running face reconstruction in sequence '%s'" % self.video_list[sequence_id])
        in_folder = self._get_path_to_sequence_detections(sequence_id)
        out_folder = self._get_path_to_sequence_reconstructions(sequence_id)

        device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        reconstruction_net = reconstruction_net or self._get_reconstruction_net(device)

        video_writer = None
        detections_fnames = sorted(list(in_folder.glob("*.png")))
        dataset = UnsupervisedImageDataset(detections_fnames)
        batch_size = 64
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        for i, batch in enumerate(tqdm(loader)):
            images = fixed_image_standardization(batch['image'].to(device))
            codedict = reconstruction_net.encode(images)
            opdict, visdict = reconstruction_net.decode(codedict)
            opdict = util.dict_tensor2npy(opdict)
            #TODO: verify axis
            # vis_im = np.split(vis_im, axis=0 ,indices_or_sections=batch_size)
            for j in range(images.shape[0]):
                path = Path(batch['path'][j])
                name = path.stem

                if save_obj:
                    if i*j == 0:
                        mesh_folder = out_folder / 'meshes'
                        mesh_folder.mkdir(exist_ok=True, parents=True)
                    reconstruction_net.save_obj(str(mesh_folder / (name + '.obj')), opdict)
                if save_mat:
                    if i*j == 0:
                        mat_folder = out_folder / 'mat'
                        mat_folder.mkdir(exist_ok=True, parents=True)
                    savemat(str(mat_folder / (name + '.mat')), opdict)
                if save_vis or save_video:
                    if i*j == 0:
                        vis_folder = out_folder / 'vis'
                        vis_folder.mkdir(exist_ok=True, parents=True)
                    vis_dict_j = {key: value[j:j+1, ...] for key,value in visdict.items()}
                    vis_im = reconstruction_net.visualize(vis_dict_j)
                    if save_vis:
                        # cv2.imwrite(str(vis_folder / (name + '.jpg')), vis_im)
                        cv2.imwrite(str(vis_folder / (name + '.png')), vis_im)
                    if save_video and video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # video_writer = cv2.VideoWriter(filename=str(vis_folder / "video.mp4"), apiPreference=cv2.CAP_FFMPEG,
                        #                                fourcc=fourcc, fps=dm.video_metas[sequence_id]['fps'], frameSize=(vis_im.shape[1], vis_im.shape[0]))
                        video_writer = cv2.VideoWriter(str(vis_folder / "video.mp4"), cv2.CAP_FFMPEG,
                                                       fourcc, int(self.video_metas[sequence_id]['fps'].split('/')[0]), (vis_im.shape[1], vis_im.shape[0]), True)
                    if save_video:
                        video_writer.write(vis_im)
                if save_images:
                    if i*j == 0:
                        ims_folder = out_folder / 'ims'
                        ims_folder.mkdir(exist_ok=True, parents=True)
                    for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                        if vis_name not in visdict.keys():
                            continue
                        image = util.tensor2image(visdict[vis_name][j])
                        Path(ims_folder / vis_name).mkdir(exist_ok=True, parents=True)
                        cv2.imwrite(str(ims_folder / vis_name / (name +'.jpg')), image)
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

    def _get_detection_for_sequence(self, sid):
        out_folder = self._get_path_to_sequence_detections(sid)
        out_file = out_folder / "bboxes.pkl"
        if not out_file.exists():
            print("Detections don't exist")
            return [], [], [], 0
        detection_fnames, centers, sizes, last_frame_id = \
            FaceVideoDataModule.load_detections(out_file)

        return detection_fnames, centers, sizes, last_frame_id

    def _get_reconstructions_for_sequence(self, sid):
        out_folder = self._get_path_to_sequence_reconstructions(sid)
        vis_fnames = sorted(list((out_folder / "vis").glob("*.png")))
        if len(vis_fnames) == 0:
            vis_fnames = sorted(list((out_folder / "vis").glob("*.jpg")))
        return vis_fnames

    def _get_frames_for_sequence(self, sid):
        out_folder = self._get_path_to_sequence_frames(sid)
        vid_frames = sorted(list(out_folder.glob("*.png")))
        return vid_frames

    def _get_recognition_for_sequence(self, sid, distance_threshold=0.5):
        # out_folder = self._get_path_to_sequence_detections(sid)
        recognition_path = self._get_recognition_filename(sid, distance_threshold)
        # recognition_path = out_folder / "recognition.pkl"
        indices, labels, mean, cov, fnames = FaceVideoDataModule._load_recognitions(recognition_path)
        return indices, labels, mean, cov, fnames

    def create_reconstruction_video(self, sequence_id, overwrite=False, distance_threshold=0.5):
        from PIL import Image, ImageDraw
        # fid = 0
        detection_fnames, centers, sizes, last_frame_id = self._get_detection_for_sequence(sequence_id)
        vis_fnames = self._get_reconstructions_for_sequence(sequence_id)
        vid_frames = self._get_frames_for_sequence(sequence_id)

        outfile = vis_fnames[0].parents[1] / "video.mp4"

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
                mask_pil_transparent = Image.fromarray((warped_mask * 196).astype(np.uint8))

                bb = point2bbox(c[nd], s[nd])

                frame_draw.rectangle(((bb[0, 0], bb[0, 1],), (bb[2, 0], bb[1, 1],)),
                                     outline='green', width=5)
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
        writer.release()
        attach_audio_to_reconstruction_video(outfile, self.root_dir / self.video_list[sequence_id])
        # plt.figure()
        # plt.imshow(im)
        # plt.show()
        # dst_image = warp(vis_im, tform.inverse, output_shape=frame.shape[:2])


    def create_reconstruction_video_with_recognition(self, sequence_id, overwrite=False, distance_threshold=0.5):
        from PIL import Image, ImageDraw, ImageFont
        from collections import Counter
        from matplotlib.colors import Colormap
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

        if distance_threshold == 0.5:
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

    def _gather_detections_for_sequence(self, sequence_id, with_recognitions=True):
        out_folder = self._get_path_to_sequence_detections(sequence_id)
        out_file_detections = out_folder / "bboxes.pkl"
        if with_recognitions:
            out_file_recognitions = out_folder / "embeddings.pkl"

        if out_file_detections.exists() and (not with_recognitions or out_file_recognitions.exists()):
            detection_fnames, centers, sizes, last_frame_id = \
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
            return detection_fnames, centers, sizes, embeddings, recognized_detections_fnames
        return detection_fnames, centers, sizes

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

    def _get_recognition_filename(self, sequence_id, distance_threshold=0.5):
        out_folder = self._get_path_to_sequence_detections(sequence_id)
        if distance_threshold != 0.5:
            out_file = out_folder / ("recognition_dist_%.03f.pkl" % distance_threshold)
        else:
            out_file = out_folder / "recognition.pkl"
        return out_file

    def _identify_recognitions_for_sequence(self, sequence_id, distance_threshold = 0.5):
        print("Identifying recognitions for sequence %d: '%s'" % (sequence_id, self.video_list[sequence_id]))

        detection_fnames, centers, sizes, embeddings, recognized_detections_fnames = \
            self._gather_detections_for_sequence(sequence_id, with_recognitions=True)

        out_folder = self._get_path_to_sequence_detections(sequence_id)
        # if distance_threshold != 0.5:
        #     out_file = out_folder / ("recognition_dist_%.03f.pkl" % distance_threshold)
        # else:
        #     out_file = out_folder / "recognition.pkl"
        out_file = self._get_recognition_filename(sequence_id, distance_threshold)

        from collections import Counter, OrderedDict
        from sklearn.cluster import DBSCAN, AgglomerativeClustering

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



def attach_audio_to_reconstruction_video(input_video, input_video_with_audio, output_video=None, overwrite=False):
    output_video = output_video or (Path(input_video).parent / (str(Path(input_video).stem) + "_with_sound.mp4"))
    if output_video.exists() and not overwrite:
        return
    output_video = str(output_video)
    cmd = "ffmpeg -y -i %s -i %s -c copy -map 0:0 -map 1:1 -shortest %s" \
          % (input_video, input_video_with_audio, output_video)
    os.system(cmd)

    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     pass


class UnsupervisedImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list):
        super().__init__()
        self.image_list = image_list

    def __getitem__(self, index):
        # if index < len(self.image_list):
        #     x = self.mnist_data[index]
        # raise IndexError("Out of bounds")
        img = imread(self.image_list[index])
        img = img.transpose([2,0,1]).astype(np.float32)
        return {"image" : torch.from_numpy(img) ,
                "path" : str(self.image_list[index])}

    def __len__(self):
        return len(self.image_list)



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


def point2transform(center, size, target_size_height, target_size_width):
    target_size_width = target_size_width or target_size_height
    src_pts = point2bbox(center, size)
    dst_pts = np.array([[0, 0], [0, target_size_width - 1], [target_size_height - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform


def bbpoint_warp(image, center, size, target_size_height, target_size_width=None, output_shape=None, inv=True):
    target_size_width = target_size_width or target_size_height
    tform = point2transform(center, size, target_size_height, target_size_width)
    tf = tform.inverse if inv else tform
    output_shape = output_shape or (target_size_height, target_size_width)
    dst_image = warp(image, tf, output_shape=output_shape, order=3)
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
    # dm._detect_faces_in_sequence(102)
    # dm._detect_faces_in_sequence(5)
    # dm._detect_faces_in_sequence(16)
    # dm._detect_faces_in_sequence(21)

    # dm._identify_recognitions_for_sequence(0)
    # dm.create_reconstruction_video_with_recognition(0, overwrite=True)
    dm._identify_recognitions_for_sequence(0, distance_threshold=1.0)
    dm.create_reconstruction_video_with_recognition(0, overwrite=True, distance_threshold=1.0)
    # dm._gather_detections()

    # failed_jobs = [48,  83, 102, 135, 152, 153, 154, 169, 390]
    # failed_jobs = [48,  83, 102] #, 135, 152, 153, 154, 169, 390]
    # failed_jobs = [135, 152, 153] #, 154, 169, 390]
    # failed_jobs = [154, 169, 390]
    # for fj in failed_jobs:
    #     dm._detect_faces_in_sequence(fj)
    #     dm._recognize_faces_in_sequence(fj)
    #     dm._reconstruct_faces_in_sequence(fj)

    # dm._recognize_faces_in_sequence(400)
    # dm._reconstruct_faces_in_sequence(400)
    print("Peace out")


if __name__ == "__main__":
    main()
