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


from gdl_apps.EmotionRecognition.utils.package_model import package_model, test_loading
from gdl_apps.EmotionRecognition.utils.io import load_model, get_path_to_assets
import gdl
from pathlib import Path
from gdl.utils.other import get_path_to_assets

def test_loading(output_dir):
    return load_model(output_dir)


def main():
    asset_dir = get_path_to_assets()
    input_dir = Path("/ps/project/EmotionalFacialAnimation/emoca/emorec_models_comparison/affectnet/new_split/validation") 
    output_dir = asset_dir / "EmotionRecognition" / "facerec_based_models_2"
    output_dir.mkdir(exist_ok=True, parents=True)
    model_dirs = {
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_10_16-33-08_2929045501486288941_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early": "EMOCA-emorec",  
        # "/is/cluster/work/rdanecek/emoca/emodeca/2022_01_29_21-59-07_3822537833951552856_EmotionRecognition_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early": "EMOCA_detail-emorec",  
    }

    for model, name in model_dirs.items():
        print(f"Packing model {name}")
        package_model(input_dir / model, output_dir / name, asset_dir, overwrite=True, remove_bfm_textures=True)
        test_loading(output_dir / name)
        print("Model loading tested")


if __name__ == "__main__": 
    main() 
