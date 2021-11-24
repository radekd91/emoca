import os,sys
from pathlib import Path
import numpy as np
from skimage.io import imread, imsave
import shutil

def main():
    detections_path = Path("/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/detections/")
    # reconstructions_path = Path("/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/reconstructions_emoca")
    # reconstructions_path = Path("/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/reconstructions_emoca_retarget_soubhik")
    reconstructions_path = Path("/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/reconstructions_emoca_retarget_obama")
    # reconstructions_path = Path("/ps/project/EmotionalFacialAnimation/data/aff-wild2/processed/processed_2021_Jan_19_20-25-10/AU_Set/reconstructions_emoca_retarget_cumberbatch/")

    output_dir = Path("output/inputs")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_rec_dir = Path("output/coarse")
    output_rec_dir.mkdir(exist_ok=True, parents=True)

    output_rec_detail_dir = Path("output/detail")
    output_rec_detail_dir.mkdir(exist_ok=True, parents=True)

    video_name = "Test_Set/82-25-854x480"
    image_indices = [105, 1680, 2360,  2466, 3255, 3502, 3780, 3910, 4105, 4150]
    image_indices.sort()

    for idx in image_indices:
        name = f"{idx:06d}_000.png"

        detection_image_path = detections_path / video_name / name
        reconstruction_image_path = reconstructions_path / video_name / "vis"/ name

        output_image_path = output_dir / name
        output_image_rec_path = output_rec_dir / name
        output_image_detail_rec_path = output_rec_detail_dir / name

        # copy the detection image
        shutil.copy(detection_image_path, output_image_path)

        # load the reconstruction image
        reconstruction_image = imread(reconstruction_image_path)
        rows , cols, _ = reconstruction_image.shape
        num_images = cols // rows
        coarse_image = reconstruction_image[:, 3*rows:4*rows, :]
        detail_image = reconstruction_image[:, 4*rows:5*rows, :]

        # save the reconstruction image
        imsave(output_image_rec_path, coarse_image)
        imsave(output_image_detail_rec_path, detail_image)

    white_image = detail_image*0 + 255
    imsave(output_dir / "white.png", white_image)


if __name__ == "__main__":
    main()