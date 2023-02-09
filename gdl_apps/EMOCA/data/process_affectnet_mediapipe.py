import sys, os 
import math
sys.path = [os.path.abspath("../../..")] + sys.path

from pathlib import Path

if len(sys.argv) > 1:
    sid = int(sys.argv[1])
else:
    sid = 0


from gdl.datasets.AffectNetDataModule import AffectNetDataModule, AffectNetEmoNetSplitModule



"""
Author: Radek Danecek
Copyright (c) 2023, Radek Danecek
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
import sys
from gdl.datasets.AffectNetDataModule import AffectNetDataModule


def main(): 
    if len(sys.argv) < 2: 
        print("Usage: python process_affectnet.py <input_folder> <output_folder> <optional_processed_subfolder> <optional_subset_index>")
        print("input_folder ... folder where you downloaded and extracted AffectNet")
        print("output_folder ... folder where you want to process AffectNet")
        print("optional_processed_subfolder ... if AffectNet is partly processed, it created a subfolder, which you can specify here to finish processing")
        print("optional_subset_index ... index of subset of AffectNet if you want to process many parts in parallel (recommended)")

    downloaded_affectnet_folder = sys.argv[1]
    processed_output_folder = sys.argv[2]

    if len(sys.argv) >= 3: 
        processed_subfolder = sys.argv[3]
    else: 
        processed_subfolder = None


    if len(sys.argv) >= 4: 
        sid = int(sys.argv[4])
    else: 
        sid = None


    dm = AffectNetDataModule(
            downloaded_affectnet_folder,
            processed_output_folder,
            processed_subfolder=processed_subfolder,
            mode="manual",
            scale=1.25,
            ignore_invalid=True,
            )

    if sid is not None:
        if sid >= dm.num_subsets: 
            print(f"Subset index {sid} is larger than number of subsets. Terminating")
            sys.exit()
        print("Detecting mediapipe landmarks in subset %d" % sid)
        dm._detect_landmarks_mediapipe(dm.subset_size * sid, min((sid + 1) * dm.subset_size, len(dm.df)))
        print("Finished decting faces")
    else:
        dm.prepare_data() 

    
    

if __name__ == "__main__":
    main()

