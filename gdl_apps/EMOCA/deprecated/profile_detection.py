import os, sys
# print(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
sys.path = [os.path.abspath(os.path.join(__file__, "..", "..", ".."))] + sys.path
from pathlib import Path

from gdl.datasets.FaceVideoDataset import FaceVideoDataModule

def main():
    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    subfolder = 'processed_det_thresh_0.80'
    dm = FaceVideoDataModule(str(root_path), str(output_path), processed_subfolder=subfolder,
                             face_detector_threshold=0.8)
    # dm.prepare_data()
    dm._loadMeta()
    print("Dataset loaded")
    dm._detect_faces_in_sequence(0)


if __name__ == "__main__":
    main()
