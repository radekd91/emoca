import numpy as np
# import torch
import pickle as pkl
from gdl.utils.FaceDetector import FaceDetector
import os, sys
# from gdl.utils.other import get_path_to_externals 
from pathlib import Path

import mediapipe as mp
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark
# from google.protobuf.pyext._message import RepeatedCompositeContainer

def mediapipe2np(landmarks): 
    # d = protobuf_to_dict(landmarks)
    array = np.zeros(shape=(len(landmarks), 3))
    for i in range(len(landmarks)):
        array[i, 0] = landmarks[i].x
        array[i, 1] = landmarks[i].y
        array[i, 2] = landmarks[i].z
    return array


def np2mediapipe(array): 
    # from munch import Munch
    landmarks = NormalizedLandmarkList()
    for i in range(len(array)):
        # landmarks += [ Munch(landmark=Munch(x=array[i, 0], y=array[i, 1], z=array[i, 2]))]
        # landmarks += [Munch(x=array[i, 0], y=array[i, 1], z=array[i, 2])]
        if array.shape[1] == 3:
            lmk = NormalizedLandmark(x=array[i, 0], y=array[i, 1], z=array[i, 2])
        else: 
            lmk = NormalizedLandmark(x=array[i, 0], y=array[i, 1], z=0.)
        landmarks.landmark.append(lmk)
    return landmarks


class MediaPipeLandmarkDetector(FaceDetector):

    def __init__(self, threshold=0.1, max_faces=1, video_based=False):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        # self.mp_face_mesh_options = mp.FaceMeshCalculatorOptions()

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=not video_based,
            refine_landmarks=True,
            max_num_faces=max_faces,
            min_detection_confidence=threshold)
        # self.face_mesh =   self.mp_face_mesh.FaceMesh(
        # static_image_mode=False,
        # refine_landmarks=True,
        # max_num_faces=1,
        # min_detection_confidence=0.1)

        # # Load drawing_utils and drawing_styles
        # self.mp_drawing = mp.solutions.drawing_utils 
        # self.mp_drawing_styles = mp.solutions.drawing_styles

        # self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
        #     model_selection=1, min_detection_confidence=0.1)
        # self.mp_face_detection_ = mp.solutions.face_detection.FaceDetection(
        #     model_selection=0, min_detection_confidence=0.1)

    
    def run(self, image, with_landmarks=False, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list (in image coordinates), landmarks list (in image coordinates)
        '''
        # if detected_faces is None: 
        results = self.face_mesh.process(image)
        # else:
            # print("Image size: {}".format(image.shape)) 
            # bboxes = [np.array([0, 0, image.shape[1], image.shape[0]])]
        if not results.multi_face_landmarks: 
            # this is a really weird thing, but somehow (especially when switching from one video to another) nothing will get picked up on the 
            # first run but it will be after the second run.
            results = self.face_mesh.process(image) 


        if not results.multi_face_landmarks:
            # det_results = self.mp_face_detection.process(image)
            # det_results_ = self.mp_face_detection_.process(image)
            print("no face detected by mediapipe")
            if with_landmarks:
                return [],  'mediapipe', [] 
            else:
                return [],  'mediapipe'


        all_landmarks = []
        all_boxes = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = mediapipe2np(face_landmarks.landmark)
            
            # scale landmarks to image size
            landmarks = landmarks * np.array([image.shape[1], image.shape[0], 1])
            
            all_landmarks += [landmarks]

            left = np.min(landmarks[:, 0])
            right = np.max(landmarks[:, 0])
            top = np.min(landmarks[:, 1])
            bottom = np.max(landmarks[:, 1])

            bbox = [left, top, right, bottom]
            all_boxes += [bbox]

        if with_landmarks:
            return all_boxes, 'mediapipe', all_landmarks
        else:
            return all_boxes, 'mediapipe'


