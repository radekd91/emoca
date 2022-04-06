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
import gdl
from pathlib import Path
from gdl.utils.other import get_path_to_assets


def main():
    asset_dir =  get_path_to_assets()
    input_dir = Path("/ps/project/EmotionalFacialAnimation/emoca/emotion_network_models/new_affectnet_split/image_based_networks") 
    output_dir = get_path_to_assets() / "EmotionRecognition" / "image_based_models_2"
    output_dir.mkdir(exist_ok=True, parents=True)
    model_dirs = {
        "2021_11_09_04-02-49_-1360894345964690046_EmoCnn_vgg19_bn_shake_samp-balanced_expr_Aug_early": "VGG19BN",  
        "2021_11_09_04-05-57_1011354483695245068_EmoSwin_swin_tiny_patch4_window7_224_shake_samp-balanced_expr_Aug_early": "SWIN-T", 
        "2021_11_09_04-04-01_-3592833751800073730_EmoSwin_swin_base_patch4_window7_224_shake_samp-balanced_expr_Aug_early": "SWIN-B",   
        "2021_11_09_04-12-56_7559763461347220097_EmoNet_shake_samp-balanced_expr_Aug_early": "EmoNet",
        "2021_11_09_04-04-52_-2546023918050637211_EmoSwin_swin_small_patch4_window7_224_shake_samp-balanced_expr_Aug_early": "SWIN-S", 
        "2021_11_09_05-15-38_-8198495972451127810_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early": "ResNet50", 
    }

    for model, name in model_dirs.items():
        print(f"Packing model {name}")
        package_model(input_dir / model, output_dir / name, asset_dir, overwrite=True)
        test_loading(output_dir / name)
        print("Model loading tested")


if __name__ == "__main__": 
    main() 
