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


from gdl.models.EmoDECA import EmoDECA 
from gdl.models.EmoCnnModule import EmoCnnModule 
try:
    from gdl.models.EmoSwinModule import EmoSwinModule 
except ImportError as e: 
    print(f"Could not import EmoSwinModule. SWIN models will not be available.  Make sure you pull the repository with submodules to enable Swin.")
try:
    from gdl.models.EmoNetModule import EmoNetModule
except ImportError as e: 
    print(f"Could not import EmoNetModule. EmoNet models will not be available.  Make sure you pull the repository with submodules to enable EmoNet.")
from gdl.models.IO import locate_checkpoint
from gdl.utils.other import class_from_str
from pathlib import Path
import sys 
from gdl.utils.other import get_path_to_assets


def load_configs(run_path):
    from omegaconf import OmegaConf
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    if run_path != conf.inout.full_run_dir: 
        conf.inout.output_dir = str(Path(run_path).parent)
        conf.inout.full_run_dir = str(run_path)
        conf.inout.checkpoint_dir = str(Path(run_path) / "checkpoints")
    return conf



def replace_asset_dirs(cfg, output_dir : Path, ): 
    asset_dir = get_path_to_assets()
    if 'deca_cfg' in cfg.keys():
        # cfg.model.deca_cfg.inout.output_dir = str(output_dir.parent)
        # cfg.model.deca_cfg.inout.full_run_dir = str(output_dir / mode)
        # cfg.model.deca_cfg.inout.checkpoint_dir = str(output_dir / mode / "checkpoints")
        cfg.model.deca_cfg.inout.output_dir = ""
        cfg.model.deca_cfg.inout.full_run_dir = ""
        cfg.model.deca_cfg.inout.checkpoint_dir = ""

        cfg.model.deca_cfg.model.tex_path = str(asset_dir / "FLAME/texture/FLAME_albedo_from_BFM.npz")
        cfg.model.deca_cfg.model.topology_path = str(asset_dir / "FLAME/geometry/head_template.obj")
        cfg.model.deca_cfg.model.fixed_displacement_path = str(asset_dir / 
                "FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy")
        cfg.model.deca_cfg.model.flame_model_path = str(asset_dir / "FLAME/geometry/generic_model.pkl")
        cfg.model.deca_cfg.model.flame_lmk_embedding_path = str(asset_dir / "FLAME/geometry/landmark_embedding.npy")
        cfg.model.deca_cfg.model.face_mask_path = str(asset_dir / "FLAME/mask/uv_face_mask.png")
        cfg.model.deca_cfg.model.face_eye_mask_path  = str(asset_dir / "FLAME/mask/uv_face_eye_mask.png")
        cfg.model.deca_cfg.model.pretrained_modelpath = str(asset_dir / "DECA/data/deca_model.tar")
        cfg.model.deca_cfg.model.pretrained_vgg_face_path = str(asset_dir /  "FaceRecognition/resnet50_ft_weight.pkl") 
        # cfg.model.emonet_model_path = str(asset_dir /  "EmotionRecognition/image_based_networks/ResNet50")
        cfg.model.deca_cfg.model.emonet_model_path = ""

    return cfg



def load_model(output_dir):
    
    cfg = load_configs(output_dir)
    replace_asset_dirs(cfg, output_dir)

    if 'emodeca_type' in cfg.model:
        model_class = class_from_str(cfg.model.emodeca_type, sys.modules[__name__])
    else:
        model_class = EmoDECA

    checkpoint_kwargs = {'config': cfg}
    checkpoint = locate_checkpoint(cfg, mode="best")
    model = model_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
    return model


def test(model, batch): 
    output = model(batch)