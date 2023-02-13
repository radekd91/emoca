from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
import numpy as np


def unpack_mediapipe_set(edge_set): 
    vertex_set = set()
    for i in edge_set: 
        vertex_set.add(i[0])
        vertex_set.add(i[1])
    return vertex_set


def left_eye_eyebrow_landmark_indices(sorted=True): 
    left_eye = list(unpack_mediapipe_set(FACEMESH_LEFT_EYE) \
        .union(unpack_mediapipe_set(FACEMESH_LEFT_IRIS)) \
        .union(unpack_mediapipe_set(FACEMESH_LEFT_EYEBROW)))
    if sorted: 
        left_eye.sort()
    left_eye = np.array(left_eye, dtype=np.int32)
    return left_eye

def right_eye_eyebrow_landmark_indices(sorted=True): 
    right_eye = list(unpack_mediapipe_set(FACEMESH_RIGHT_EYE) \
        .union(unpack_mediapipe_set(FACEMESH_RIGHT_IRIS)) \
        .union(unpack_mediapipe_set(FACEMESH_RIGHT_EYEBROW)))
    if sorted: 
        right_eye.sort()
    right_eye = np.array(right_eye, dtype=np.int32)
    return right_eye

def left_eye_landmark_indices(sorted=True): 
    left_eye = list(unpack_mediapipe_set(FACEMESH_LEFT_EYE))
    if sorted: 
        left_eye.sort()
    left_eye = np.array(left_eye, dtype=np.int32)
    return left_eye

def right_eye_landmark_indices(sorted=True): 
    right_eye = list(unpack_mediapipe_set(FACEMESH_RIGHT_EYE))
    if sorted:
        right_eye.sort()
    right_eye = np.array(right_eye, dtype=np.int32)
    return right_eye

def mouth_landmark_indices(sorted=True): 
    mouth = list(unpack_mediapipe_set(FACEMESH_LIPS)) 
    if sorted: 
        mouth.sort()
    mouth = np.array(mouth, dtype=np.int32)
    return mouth

def face_oval_landmark_indices(sorted=True): 
    face_oval = list(unpack_mediapipe_set(FACEMESH_FACE_OVAL))
    if sorted: 
        face_oval.sort()
    face_oval = np.array(face_oval, dtype=np.int32)
    return face_oval

def face_oval_landmark_indices(sorted=True): 
    face_oval = list(unpack_mediapipe_set(FACEMESH_FACE_OVAL))
    if sorted: 
        face_oval.sort()
    face_oval = np.array(face_oval, dtype=np.int32)
    return face_oval

def all_face_landmark_indices(sorted=True): 
    face_all = list(unpack_mediapipe_set(FACEMESH_TESSELATION))
    if sorted: 
        face_all.sort()
    face_all = np.array(face_all, dtype=np.int32)
    return face_all
