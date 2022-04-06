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


from gdl.datasets.FaceVideoDataModule import FaceVideoDataModule
from gdl.datasets.AffectNetDataModule import AffectNetExpressions
from gdl.datasets.EmotionalImageDataset import EmotionalImageDataset
from enum import Enum
import pickle as pkl
from pathlib import Path
import hashlib
from tqdm import auto
import numpy as np
import PIL
from collections import OrderedDict


class Expression7(Enum):
    Neutral = 0
    Anger = 1
    Disgust = 2
    Fear = 3
    Happiness = 4
    Sadness = 5
    Surprise = 6
    None_ = 7


def affect_net_to_expr7(aff : AffectNetExpressions) -> Expression7:
    # try:
    if aff == AffectNetExpressions.Happy:
        return Expression7.Happiness
    if aff == AffectNetExpressions.Sad:
        return Expression7.Sadness
    if aff == AffectNetExpressions.Contempt:
        return Expression7.None_
    return Expression7[aff.name]
    # except KeyError as e:
    #     return Expression7.None_


def expr7_to_affect_net(expr : Expression7) -> AffectNetExpressions:
    # try:
    if isinstance(expr, int) or isinstance(expr, np.int32) or isinstance(expr, np.int64):
        expr = Expression7(expr)
    if expr == Expression7.Happiness:
        return AffectNetExpressions.Happy
    if expr == Expression7.Sadness:
        return AffectNetExpressions.Sad
    return AffectNetExpressions[expr.name]
    # except KeyError as e:
    #     return AffectNetExpressions.None_


class AU8(Enum):
    AU1 = 0
    AU2 = 1
    AU4 = 2
    AU6 = 3
    AU12 = 4
    AU15 = 5
    AU20 = 6
    AU25 = 7


class AffWild2DMBase(FaceVideoDataModule):
    """
    A data module which implements a wrapper for the AffWild2 dataset.
    https://ibug.doc.ic.ac.uk/resources/aff-wild2/ 
    """

    def _get_processed_annotations_for_sequence(self, sid):
        pass
        video_file = self.video_list[sid]
        suffix = Path(video_file.parts[-4]) / 'detections' / video_file.parts[-2]
        annotation = Path(self.root_dir / suffix) / "valid_annotations.pkl"
        emotions, valence, arousal, detections_fnames = FaceVideoDataModule._load_face_emotions(annotation)
        return emotions, valence, arousal, detections_fnames

    def _create_emotional_image_dataset(self,
                                        annotation_list=None,
                                        filter_pattern=None,
                                        with_landmarks=False,
                                        with_segmentation=False,
                                        crash_on_missing_file=False):
        annotation_list = annotation_list or ['va', 'expr7', 'au8']
        detections_all = []
        annotations_all = OrderedDict()
        for a in annotation_list:
            annotations_all[a] = []
        recognition_labels_all = []


        import re
        if filter_pattern is not None:
            # p = re.compile(filter_pattern)
            p = re.compile(filter_pattern, re.IGNORECASE)

        for si in auto.tqdm(range(self.num_sequences)):
            sequence_name = self.video_list[si]

            if filter_pattern is not None:
                res = p.match(str(sequence_name))
                if res is None:
                    continue

            ## TODO: or more like an idea - a solution towards duplicate videos between va/au/expression set
            # would be to append the video path here to serve as a key in the dictionaries (instead of just the stem
            # of the path)

            detection_fnames, annotations, recognition_labels, discarded_annotations, detection_not_found = \
                self._get_validated_annotations_for_sequence(si, crash_on_failure=False)

            if detection_fnames is None:
                continue

            current_list = annotation_list.copy()
            for annotation_name, detection_list in detection_fnames.items():
                detections_all += detection_list
                # annotations_all += [annotations[key]]
                for annotation_key in annotations[annotation_name].keys():
                    if annotation_key in current_list:
                        current_list.remove(annotation_key)
                    array = annotations[annotation_name][annotation_key]
                    annotations_all[annotation_key] += array.tolist()
                    n = array.shape[0]

                recognition_labels_all += len(detection_list)*[annotation_name + "_" + str(recognition_labels[annotation_name])]
                if len(current_list) != len(annotation_list):
                    print("No desired GT is found. Skipping sequence %d" % si)

                for annotation_name in current_list:
                    annotations_all[annotation_name] += [None] * n

        print("Data gathered")
        print(f"Found {len(detections_all)} detections with annotations "
              f"of {len(set(recognition_labels_all))} identities")

        # #TODO: delete debug code:
        # N = 3000
        # detections = detections[:N] + detections[-N:]
        # recognition_labels_all = recognition_labels_all[:N] + recognition_labels_all[-N:]
        # for key in annotations_all.keys():
        #     annotations_all[key] = annotations_all[key][:N] + annotations_all[key][-N:]
        # # end debug code : todo remove

        invalid_indices = set()
        if not with_landmarks:
            landmarks = None
        else:
            landmarks = []
            print("Checking if every frame has a corresponding landmark file")
            for det_i, det in enumerate(auto.tqdm(detections_all)):
                lmk = det.parents[3]
                lmk = lmk / "landmarks" / (det.relative_to(lmk / "detections"))
                lmk = lmk.parent / (lmk.stem + ".pkl")
                file_exists = (self.output_dir / lmk).is_file()
                if not file_exists and crash_on_missing_file:
                    raise RuntimeError(f"Landmark does not exist {lmk}")
                elif not file_exists:
                    invalid_indices.add(det_i)
                landmarks += [lmk]

        if not with_segmentation:
            segmentations = None
        else:
            segmentations = []
            print("Checking if every frame has a corresponding segmentation file")
            for det_i, det in enumerate(auto.tqdm(detections_all)):
                seg = det.parents[3]
                seg = seg / "segmentations" / (det.relative_to(seg / "detections"))
                seg = seg.parent / (seg.stem + ".pkl")
                file_exists = (self.output_dir / seg).is_file()
                if not file_exists and crash_on_missing_file:
                    raise RuntimeError(f"Landmark does not exist {seg}")
                elif not file_exists:
                    invalid_indices.add(det_i)
                segmentations += [seg]

        invalid_indices = sorted(list(invalid_indices), reverse=True)
        for idx in invalid_indices:
            del detections_all[idx]
            del landmarks[idx]
            del segmentations[idx]
            del recognition_labels_all[idx]
            for key in annotations_all.keys():
                del annotations_all[key][idx]

        return detections_all, landmarks, segmentations, annotations_all, recognition_labels_all

    def get_annotated_emotion_dataset(self,
                                      annotation_list = None,
                                      filter_pattern=None,
                                      image_transforms=None,
                                      split_ratio=None,
                                      split_style=None,
                                      with_landmarks=False,
                                      with_segmentations=False,
                                      K=None,
                                      K_policy=None,
                                      # if you add more parameters here, add them also to the hash list
                                      load_from_cache=True # do not add this one to the hash list
                                      ):
        # Process the dataset
        str_to_hash = pkl.dumps(tuple([annotation_list, filter_pattern]))
        inter_cache_hash = hashlib.md5(str_to_hash).hexdigest()
        inter_cache_folder = Path(self.output_dir) / "cache" / str(inter_cache_hash)
        if (inter_cache_folder / "lists.pkl").exists() and load_from_cache:
            print(f"Found processed filelists in '{str(inter_cache_folder)}'. "
                  f"Reprocessing will not be needed. Loading ...")
            with open(inter_cache_folder / "lists.pkl", "rb") as f:
                detections = pkl.load(f)
                landmarks = pkl.load(f)
                segmentations = pkl.load(f)
                annotations = pkl.load(f)
                recognition_labels = pkl.load(f)
            print("Loading done")

        else:
            detections, landmarks, segmentations, annotations, recognition_labels = \
                self._create_emotional_image_dataset(
                    annotation_list, filter_pattern, with_landmarks, with_segmentations)
            inter_cache_folder.mkdir(exist_ok=True, parents=True)
            print(f"Dataset processed. Saving into: '{str(inter_cache_folder)}'.")
            with open(inter_cache_folder / "lists.pkl", "wb") as f:
                pkl.dump(detections, f)
                pkl.dump(landmarks, f)
                pkl.dump(segmentations, f)
                pkl.dump(annotations, f)
                pkl.dump(recognition_labels, f)
            print(f"Saving done.")

        if split_ratio is not None and split_style is not None:

            hash_list = tuple([annotation_list,
                               filter_pattern,
                               split_ratio,
                               split_style,
                               with_landmarks,
                               with_segmentations,
                               K,
                               K_policy,
                               # add new parameters here
                               ])
            cache_hash = hashlib.md5(pkl.dumps(hash_list)).hexdigest()
            cache_folder = Path(self.output_dir) / "cache" / "tmp" / str(cache_hash)
            cache_folder.mkdir(exist_ok=True, parents=True)
            # load from cache if exists
            if load_from_cache and (cache_folder / "lists_train.pkl").is_file() and \
                (cache_folder / "lists_val.pkl").is_file():
                print(f"Dataset split found in: '{str(cache_folder)}'. Loading ...")
                with open(cache_folder / "lists_train.pkl", "rb") as f:
                    # training
                     detection_train = pkl.load(f)
                     landmarks_train = pkl.load(f)
                     segmentations_train = pkl.load(f)
                     annotations_train = pkl.load(f)
                     recognition_labels_train = pkl.load(f)
                     idx_train = pkl.load(f)
                with open(cache_folder / "lists_val.pkl", "rb") as f:
                    # validation
                     detection_val = pkl.load(f)
                     landmarks_val = pkl.load(f)
                     segmentations_val = pkl.load(f)
                     annotations_val = pkl.load(f)
                     recognition_labels_val = pkl.load(f)
                     idx_val = pkl.load(f)
                print("Loading done")
            else:
                print(f"Splitting the dataset. Split style '{split_style}', split ratio: '{split_ratio}'")
                if image_transforms is not None:
                    if not isinstance(image_transforms, list) or len(image_transforms) != 2:
                        raise ValueError("You have to provide image transforms for both trainng and validation sets")
                idxs = np.arange(len(detections), dtype=np.int32)
                if split_style == 'random':
                    np.random.seed(0)
                    np.random.shuffle(idxs)
                    split_idx = int(idxs.size * split_ratio)
                    idx_train = idxs[:split_idx]
                    idx_val = idxs[split_idx:]
                elif split_style == 'manual':
                    idx_train = []
                    idx_val = []
                    for i, det in enumerate(auto.tqdm(detections)):
                        if 'Train_Set' in str(det):
                            idx_train += [i]
                        elif 'Validation_Set' in str(det):
                            idx_val += [i]
                        else:
                            idx_val += [i]

                elif split_style == 'sequential':
                    split_idx = int(idxs.size * split_ratio)
                    idx_train = idxs[:split_idx]
                    idx_val = idxs[split_idx:]
                elif split_style == 'random_by_label':
                    idx_train = []
                    idx_val = []
                    unique_labels = sorted(list(set(recognition_labels)))
                    np.random.seed(0)
                    print(f"Going through {len(unique_labels)} unique labels and splitting its samples into "
                          f"training/validations set randomly.")
                    for li, label in enumerate(auto.tqdm(unique_labels)):
                        label_indices = np.array([i for i in range(len(recognition_labels)) if recognition_labels[i] == label],
                                                 dtype=np.int32)
                        np.random.shuffle(label_indices)
                        split_idx = int(len(label_indices) * split_ratio)
                        i_train = label_indices[:split_idx]
                        i_val = label_indices[split_idx:]
                        idx_train += i_train.tolist()
                        idx_val += i_val.tolist()
                    idx_train = np.array(idx_train, dtype= np.int32)
                    idx_val = np.array(idx_val, dtype= np.int32)
                elif split_style == 'sequential_by_label':
                    idx_train = []
                    idx_val = []
                    unique_labels = sorted(list(set(recognition_labels)))
                    print(f"Going through {len(unique_labels)} unique labels and splitting its samples into "
                          f"training/validations set sequentially.")
                    for li, label in enumerate(auto.tqdm(unique_labels)):
                        label_indices = [i for i in range(len(recognition_labels)) if recognition_labels[i] == label]
                        split_idx = int(len(label_indices) * split_ratio)
                        i_train = label_indices[:split_idx]
                        i_val = label_indices[split_idx:]
                        idx_train += i_train
                        idx_val += i_val
                    idx_train = np.array(idx_train, dtype= np.int32)
                    idx_val = np.array(idx_val, dtype= np.int32)
                else:
                    raise ValueError(f"Invalid split style {split_style}")

                if split_ratio < 0 or split_ratio > 1:
                    raise ValueError(f"Invalid split ratio {split_ratio}")

                def index_list_by_list(l, idxs):
                    return [l[i] for i in idxs]

                def index_dict_by_list(d, idxs):
                    res = d.__class__()
                    for key in d.keys():
                        res[key] = [d[key][i] for i in idxs]
                    return res

                detection_train = index_list_by_list(detections, idx_train)
                annotations_train = index_dict_by_list(annotations, idx_train)
                recognition_labels_train = index_list_by_list(recognition_labels, idx_train)
                if with_landmarks:
                    landmarks_train = index_list_by_list(landmarks, idx_train)
                else:
                    landmarks_train = None

                if with_segmentations:
                    segmentations_train = index_list_by_list(segmentations, idx_train)
                else:
                    segmentations_train = None

                detection_val = index_list_by_list(detections, idx_val)
                annotations_val = index_dict_by_list(annotations, idx_val)
                recognition_labels_val = index_list_by_list(recognition_labels, idx_val)

                if with_landmarks:
                    landmarks_val = index_list_by_list(landmarks, idx_val)
                else:
                    landmarks_val = None

                if with_segmentations:
                    segmentations_val = index_list_by_list(segmentations, idx_val)
                else:
                    segmentations_val = None

                print(f"Dataset split processed. Saving into: '{str(cache_folder)}'.")
                with open(cache_folder / "lists_train.pkl", "wb") as f:
                    # training
                    pkl.dump(detection_train, f)
                    pkl.dump(landmarks_train, f)
                    pkl.dump(segmentations_train, f)
                    pkl.dump(annotations_train, f)
                    pkl.dump(recognition_labels_train, f)
                    pkl.dump(idx_train, f)
                with open(cache_folder / "lists_val.pkl", "wb") as f:
                    # validation
                    pkl.dump(detection_val, f)
                    pkl.dump(landmarks_val, f)
                    pkl.dump(segmentations_val, f)
                    pkl.dump(annotations_val, f)
                    pkl.dump(recognition_labels_val, f)
                    pkl.dump(idx_val, f)
                print(f"Saving done.")

            dataset_train = EmotionalImageDataset(
                detection_train,
                annotations_train,
                recognition_labels_train,
                image_transforms[0],
                self.output_dir,
                landmark_list=landmarks_train,
                segmentation_list=segmentations_train,
                K=K,
                K_policy=K_policy)

            dataset_val = EmotionalImageDataset(
                detection_val,
                annotations_val,
                recognition_labels_val,
                image_transforms[1],
                self.output_dir,
                landmark_list=landmarks_val,
                segmentation_list=segmentations_val,
                # K=K,
                K=1,
                # K=None,
                # K_policy=K_policy)
                K_policy='sequential')
                # K_policy=None)

            return dataset_train, dataset_val, idx_train, idx_val

        # dataset = EmotionalImageDataset(
        dataset = EmotionalImageDataset(
            detections,
            annotations,
            recognition_labels,
            image_transforms,
            self.output_dir,
            landmark_list=landmarks,
            segmentation_list=segmentations,
            K=K,
            K_policy=K_policy)
        return dataset

    def _draw_annotation(self, frame_draw : PIL.ImageDraw.Draw, val_gt : dict, font, color):
        all_str = ''
        for gt_type, val in val_gt.items():
            if gt_type == 'va':
                va_str = "V: %.02f  A: %.02f" % (val[0], val[1])
                all_str += "\n" + va_str
                # frame_draw.text((bb[1, 0] - 60, bb[0, 1] - 30,), va_str, font=fnt, fill=color)
            elif gt_type == 'expr7':
                frame_draw.text((bb[0, 0], bb[0, 1] - 30,), Expression7(val).name, font=font, fill=color)
            elif gt_type == 'au8':
                au_str = ''
                for li, label in enumerate(val):
                    if label:
                        au_str += AU8(li).name + ' '
                all_str += "\n" + au_str
                # frame_draw.text((bb[0, 0], bb[1, 1] + 30,), au_str, font=fnt, fill=color)
            else:
                raise ValueError(f"Unable to visualize this gt_type: '{gt_type}")
            frame_draw.text((bb[0, 0], bb[1, 1] + 10,), str(all_str), font=font, fill=color)


    def test_annotations(self, net=None, annotation_list = None, filter_pattern=None):
        net = net or self._get_emonet(self.device)

        dataset = self.get_annotated_emotion_dataset(annotation_list, filter_pattern)



def main():
    # root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    # root = Path("/is/cluster/work/rdanecek/data/aff-wild2/")
    root = Path("/ps/project/EmotionalFacialAnimation/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    # subfolder = 'processed_2020_Dec_21_00-30-03'
    subfolder = 'processed_2021_Jan_19_20-25-10'
    dm = AffWild2DMBase(str(root_path), str(output_path), processed_subfolder=subfolder)
    dm.prepare_data()
    #
    # test_videos = [
    #     '9-15-1920x1080.mp4', # smiles, sadness, tears, girl with glasses
    #     '19-24-1920x1080.mp4', # angry young black guy on stage
    #     '17-24-1920x1080.mp4', # black guy on stage, difficult light
    #     '23-24-1920x1080.mp4', # white woman, over-articulated expressions
    #     '24-30-1920x1080-2.mp4', # white woman, over-articulated expressions
    #     '28-30-1280x720-1.mp4', # angry black guy
    #     '31-30-1920x1080.mp4', # crazy white guy, beard, view from the side
    #     '34-25-1920x1080.mp4', # white guy, mostly neutral
    #     '50-30-1920x1080.mp4', # baby
    #     '60-30-1920x1080.mp4', # smiling asian woman
    #     '61-24-1920x1080.mp4', # very lively white woman
    #     '63-30-1920x1080.mp4', # smiling asian woman
    #     '66-25-1080x1920.mp4', # white girl acting out an emotional performance
    #     '71-30-1920x1080.mp4', # old white woman, camera shaking
    #     '83-24-1920x1080.mp4', # excited black guy (but expressions mostly neutral)
    #     '87-25-1920x1080.mp4', # white guy explaining stuff, mostly neutral
    #     '95-24-1920x1080.mp4', # white guy explaining stuff, mostly neutral
    #     '122-60-1920x1080-1.mp4', # crazy white youtuber, lots of overexaggerated expressiosn
    #     '135-24-1920x1080.mp4', # a couple watching a video, smiles, sadness, tears
    #     '82-25-854x480.mp4', # Rachel McAdams, sadness, anger
    #     '111-25-1920x1080.mp4', # disgusted white guy
    #     '121-24-1920x1080.mp4', # white guy scared and happy faces
    # ]
    #
    # indices = [dm.video_list.index(Path('VA_Set/videos/Train_Set') / name) for name in test_videos]
    #
    # # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/9-15-1920x1080.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/119-30-848x480.mp4')) # black lady with at Oscars
    # i = dm.video_list.index(Path('Expression_Set/videos/Train_Set/1-30-1280x720.mp4')) # black lady with at Oscars
    # # dm._process_everything_for_sequence(i)
    # # dm._detect_faces_in_sequence(i)
    # # dm._segment_faces_in_sequence(i)

    # dm._extract_emotion_from_faces_in_sequence(0)

    # rpoblematic indices
    # dm._segment_faces_in_sequence(30)
    # dm._segment_faces_in_sequence(156)
    # dm._segment_faces_in_sequence(399)

    # dm._create_emotional_image_dataset(['va'], "VA_Set")
    # dm._recognize_emotion_in_sequence(0)
    # i = dm.video_list.index(Path('AU_Set/videos/Train_Set/130-25-1280x720.mp4'))
    # i = dm.video_list.index(Path('AU_Set/videos/Train_Set/52-30-1280x720.mp4'))
    # i = dm.video_list.index(Path('Expression_Set/videos/Train_Set/46-30-484x360.mp4'))
    # i = dm.video_list.index(Path('Expression_Set/videos/Train_Set/135-24-1920x1080.mp4'))
    # i = dm.video_list.index(Path('VA_Set/videos/Train_Set/30-30-1920x1080.mp4'))
    # dm._recognize_faces_in_sequence(i)
    # dm._identify_recognitions_for_sequence(i)
    # for i in range(7,8):
    # for i in range(8, 30):
    #     dm._recognize_faces_in_sequence(i, num_workers=8)
    #     dm._identify_recognitions_for_sequence(i)
    #     print("----------------------------------")
    #     print(f"Assigning GT to detections for seq: {i}")
    #     dm.assign_gt_to_detections_sequence(i)
    # dm._detect_faces()
    # dm._detect_faces_in_sequence(30)
    # dm._detect_faces_in_sequence(107)
    # dm._detect_faces_in_sequence(399)
    # dm._detect_faces_in_sequence(21)
    # dm.create_reconstruction_video_with_recognition_and_annotations(100, overwrite=True)
    # dm._identify_recognitions_for_sequence(0)
    # dm.create_reconstruction_video_with_recognition(0, overwrite=True)
    # dm._identify_recognitions_for_sequence(0, distance_threshold=1.0)
    # dm.create_reconstruction_video_with_recognition(0, overwrite=True, distance_threshold=1.0)
    # dm._gather_detections()

    # failed_jobs = [48,  83, 102, 135, 152, 153, 154, 169, 390]
    # failed_jobs = [48,  83, 102] #, 135, 152, 153, 154, 169, 390]
    # failed_jobs = [135, 152, 153] #, 154, 169, 390]
    # failed_jobs = [154, 169, 390]
    # for fj in failed_jobs:

    fj = 9
    # dm._detect_faces_in_sequence(fj)
    # dm._recognize_faces_in_sequence(fj)
    retarget_from = None
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/" \
    #                 "processed_2021_Jan_19_20-25-10/AU_Set/detections/Test_Set/82-25-854x480/000001_000.png" ## Rachel McAdams
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/detections/Test_Set/30-30-1920x1080/000880_000.png" # benedict
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/detections/Train_Set/11-24-1920x1080/000485_000.png" # john cena
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/detections/Train_Set/26-60-1280x720/000200_000.png" # obama
    # retarget_from = "/ps/project/EmotionalFacialAnimation/data/random_images/soubhik.jpg" # obama
    # dm._reconstruct_faces_in_sequence(fj, rec_method="emoca", retarget_from=retarget_from, retarget_suffix="soubhik")
    dm._reconstruct_faces_in_sequence(fj, rec_method="emoca", retarget_from=retarget_from, retarget_suffix="_retarget_cena")
    # dm._reconstruct_faces_in_sequence(fj, rec_method='deep3dface')
    # dm.create_reconstruction_video(fj, overwrite=False)
    # dm.create_reconstruction_video(fj, overwrite=False, rec_method='emoca')
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='emoca', retarget_suffix="_retarget_soubhik")
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='emoca', retarget_suffix="_retarget_obama")
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='emoca', retarget_suffix="_retarget_cumberbatch")
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='emoca', retarget_suffix="_retarget_cena")
    # dm.create_reconstruction_video(fj, overwrite=False, rec_method='emoca', image_type="coarse")
    # dm.create_reconstruction_video(fj, overwrite=False, rec_method='deep3dface')
    # dm.create_reconstruction_video(fj, overwrite=False, rec_method='deep3dface')
    # dm.create_reconstruction_video_with_recognition(fj, overwrite=True)
    # dm._identify_recognitions_for_sequence(fj)
    # dm.create_reconstruction_video_with_recognition(fj, overwrite=True, distance_threshold=0.6)

    # dm._recognize_faces_in_sequence(400)
    # dm._reconstruct_faces_in_sequence(400)
    print("Peace out")


if __name__ == "__main__":
    main()