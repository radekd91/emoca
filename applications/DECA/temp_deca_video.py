from pathlib import Path
from datasets.FaceVideoDataset import FaceVideoDataModule, bbpoint_warp, point2bbox
import sys
import argparse
import os
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from skimage.io import imread, imsave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..' ,'..', '..', 'DECA')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import util
from tqdm import tqdm
import cv2


def get_detection_for_sequence(self, sid):
    out_folder = self._get_path_to_sequence_detections(sid)
    out_file = out_folder / "bboxes.pkl"
    if not out_file.exists():
        print("Detections don't exist")
    detection_fnames, centers, sizes, last_frame_id = \
            FaceVideoDataModule.load_detections(out_file)

    return detection_fnames, centers, sizes, last_frame_id


def get_reconstructions_for_sequence(self, sid):
    out_folder = self._get_path_to_sequence_reconstructions(sid)
    vis_fnames = sorted(list((out_folder / "vis" ).glob("*.jpg")))
    return vis_fnames


def get_frames_for_sequence(self, sid):
    out_folder = self._get_path_to_sequence_frames(sid)
    vid_frames = sorted(list(out_folder.glob("*.png")))
    return vid_frames

def create_detection_video(self, sequence_id):
    # fid = 0
    detection_fnames, centers, sizes, last_frame_id = get_detection_for_sequence(self, sequence_id)
    vis_fnames = get_reconstructions_for_sequence(self, sequence_id)
    vid_frames = get_frames_for_sequence(self, sequence_id)

    print("Creating reconstruction video for sequence num %d: '%s' " % (sequence_id, self.video_list[sequence_id]))

    writer = None #cv2.VideoWriter()

    did = 0
    for fid in tqdm(range(len(vid_frames))):
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
            vis_name = vis_fnames[did]

            if detection_name.stem != vis_name.stem:
                print("%s != %s" % (detection_name.stem, vis_name.stem))
                raise RuntimeError("Detection and visualization filenames should match but they don't.")

            detection_im = imread(self.output_dir / detection_name.relative_to(detection_name.parents[4]))
            vis_im = imread(vis_name)
            vis_im = vis_im[:, -vis_im.shape[1]//5:, ...]

            # vis_mask = np.prod(vis_im, axis=2) == 0
            vis_mask = (np.prod(vis_im, axis=2) > 30).astype(np.uint8)*255

            # vis_im = np.concatenate([vis_im, vis_mask[..., np.newaxis]], axis=2)

            warped_im = bbpoint_warp(vis_im, c[nd], s[nd], detection_im.shape[0], output_shape=(frame.shape[0], frame.shape[1]), inv=False)
            # warped_im = bbpoint_warp(vis_im, c[nd], s[nd], frame.shape[0], frame.shape[1], False)
            warped_mask = bbpoint_warp(vis_mask, c[nd], s[nd], detection_im.shape[0], output_shape=(frame.shape[0], frame.shape[1]), inv=False)
            # warped_mask = bbpoint_warp(vis_mask, c[nd], s[nd], frame.shape[0], frame.shape[1], False)

            # dst_image = bbpoint_warp(frame, c[nd], s[nd], warped_im.shape[0])
            # frame2 = bbpoint_warp(dst_image, c[nd], s[nd], warped_im.shape[0], output_shape=(frame.shape[0], frame.shape[1]), inv=False)

            # frame_pil = Image.fromarray(frame)
            # frame_pil2 = Image.fromarray(frame)
            vis_pil = Image.fromarray((warped_im * 255).astype(np.uint8))
            mask_pil = Image.fromarray((warped_mask* 255).astype(np.uint8))
            mask_pil_transparent = Image.fromarray((warped_mask* 196).astype(np.uint8))

            bb = point2bbox(c[nd], s[nd])

            frame_draw.rectangle(( (bb[0,0] ,  bb[0,1] , ), ( bb[2,0], bb[1,1],)) , outline='green')
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
            outfile = str(vis_fnames[0].parents[1] / "video.mp4")
            writer = cv2.VideoWriter(outfile, cv2.CAP_FFMPEG,
                                           fourcc, int(self.video_metas[sequence_id]['fps'].split('/')[0]),
                                           (im.shape[1], im.shape[0]), True)
        im_cv = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        writer.write(im_cv)
    writer.release()

    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # dst_image = warp(vis_im, tform.inverse, output_shape=frame.shape[:2])


def gather_detections(self):
    out_files = []

    detection_fnames_all = []
    centers_all = []
    sizes_all = []

    for sid in range(self.num_sequences):
        out_folder = self._get_path_to_sequence_detections(sid)
        out_file = out_folder / "bboxes.pkl"
        out_files += [out_file]

        if out_file.exists():
            # detection_fnames, centers, sizes, last_frame_id = \
            #     FaceVideoDataModule.load_detections(out_file)
            # print("Face detections for video %d found" % sid)
            pass
        else:
            print("Faces for video %d not detected" % sid)
            # detection_fnames = []
            # centers = []
            # sizes = []
        #
        # detection_fnames_all += [detection_fnames]
        # centers_all += [centers]
        # sizes_all += [sizes]






def main():
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    subfolder = 'processed_2020_Dec_21_00-30-03'
    dm = FaceVideoDataModule(str(root_path), str(output_path), processed_subfolder=subfolder)
    dm.prepare_data()
    # dm.setup()

    # gather_detections(dm)

    seq = 10
    create_detection_video(dm, seq)




if __name__ == "__main__":
    main()
