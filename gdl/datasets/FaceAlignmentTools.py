import numpy as np
from pathlib import Path
from gdl.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp
import skvideo
import types


def align_face(image, landmarks, landmark_type, scale_adjustment, target_size_height, target_size_width=None,):
    """
    Returns an image with the face aligned to the center of the image.
    :param image: The full resolution image in which to align the face. 
    :param landmarks: The landmarks of the face in the image (in the original image coordinates).
    :param landmark_type: The type of landmarks. Such as 'kpt68' or 'bbox' or 'mediapipe'.
    :param scale_adjustment: The scale adjustment to apply to the image.
    :param target_size_height: The height of the output image.
    :param target_size_width: The width of the output image. If not provided, it is assumed to be the same as target_size_height.
    :return: The aligned face image. The image will be in range [0,1].
    """
    # landmarks_for_alignment = "mediapipe"
    left = landmarks[:,0].min()
    top =  landmarks[:,1].min()
    right =  landmarks[:,0].max()
    bottom = landmarks[:,1].max()

    old_size, center = bbox2point(left, right, top, bottom, type=landmark_type)
    size = (old_size * scale_adjustment).astype(np.int32)

    img_warped, lmk_warped = bbpoint_warp(image, center, size, target_size_height, target_size_width, landmarks=landmarks)

    return img_warped


def align_video(video, centers, sizes, landmarks, target_size_height, target_size_width=None, ):
    """
    Returns a video with the face aligned to the center of the image.
    :param video: The full resolution video in which to align the face. 
    :param landmarks: The landmarks of the face in the video (in the original video coordinates).
    :param target_size_height: The height of the output video.
    :param target_size_width: The width of the output video. If not provided, it is assumed to be the same as target_size_height.
    :return: The aligned face video. The video will be in range [0,1].
    """
    if isinstance(video, (str, Path)):
        video = skvideo.io.vread(video)
    elif isinstance(video, (np.ndarray, types.GeneratorType)):
        pass
    else:
        raise ValueError("video must be a string, Path, or numpy array")

    aligned_video = []
    warped_landmarks = []
    if isinstance(video, np.ndarray):
        for i in range(len(centers)): 
            img_warped, lmk_warped = bbpoint_warp(video[i], centers[i], sizes[i], 
                    target_size_height=target_size_height, target_size_width=target_size_width, 
                    landmarks=landmarks[i])
            aligned_video.append(img_warped)
            warped_landmarks += [lmk_warped]
            
    elif isinstance(video, types.GeneratorType): 
        for i, frame in enumerate(video):
            img_warped, lmk_warped = bbpoint_warp(frame, centers[i], sizes[i], 
                    target_size_height=target_size_height, target_size_width=target_size_width, 
                    landmarks=landmarks[i])
            aligned_video.append(img_warped)
            warped_landmarks += [lmk_warped] 

    aligned_video = np.stack(aligned_video, axis=0)
    return aligned_video, warped_landmarks


def align_and_save_video(video, out_video_path, centers, sizes, landmarks, target_size_height, target_size_width=None, output_dict=None):
    """
    Returns a video with the face aligned to the center of the image.
    :param video: The full resolution video in which to align the face. 
    :param landmarks: The landmarks of the face in the video (in the original video coordinates).
    :param target_size_height: The height of the output video.
    :param target_size_width: The width of the output video. If not provided, it is assumed to be the same as target_size_height.
    :return: The aligned face video. The video will be in range [0,1].
    """
    if isinstance(video, (str, Path)):
        video = skvideo.io.vread(video)
    elif isinstance(video, (np.ndarray, types.GeneratorType)):
        pass
    else:
        raise ValueError("video must be a string, Path, or numpy array")

    writer = skvideo.io.FFmpegWriter(str(out_video_path), outputdict=output_dict)
    warped_landmarks = []
    if isinstance(video, np.ndarray):
        for i in range(len(centers)): 
            img_warped, lmk_warped = bbpoint_warp(video[i], centers[i], sizes[i], 
                    target_size_height=target_size_height, target_size_width=target_size_width, 
                    landmarks=landmarks[i])
            img_warped = (img_warped * 255).astype(np.uint8)
            writer.writeFrame(img_warped)
            warped_landmarks += [lmk_warped]
            
    elif isinstance(video, types.GeneratorType): 
        for i, frame in enumerate(video):
            img_warped, lmk_warped = bbpoint_warp(frame, centers[i], sizes[i], 
                    target_size_height=target_size_height, target_size_width=target_size_width, 
                    landmarks=landmarks[i])
            img_warped = (img_warped * 255).astype(np.uint8)
            writer.writeFrame(img_warped)
            warped_landmarks += [lmk_warped] 
    writer.close()

    return warped_landmarks