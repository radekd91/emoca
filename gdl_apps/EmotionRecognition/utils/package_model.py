import sys
import os 
from pathlib import Path
from typing import overload
import distutils.dir_util
from omegaconf import OmegaConf, DictConfig
import shutil

from gdl.layers.losses.EmoNetLoss import emo_network_from_path

def package_model(input_dir, output_dir, asset_dir, overwrite=False):
    input_dir = Path(input_dir) 
    output_dir = Path(output_dir)
    asset_dir = Path(asset_dir)

    if output_dir.exists(): 
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory '{output_dir}' already exists.")
            sys.exit()

    if not input_dir.is_dir(): 
        print(f"Input directory '{input_dir}' does not exist.")
        sys.exit()

    if not input_dir.is_dir(): 
        print(f"Input directory '{asset_dir}' does not exist.")
        sys.exit()


    # copy all files and folders from input_dir to output_dir using distutils.dir_util.copy_tree
    distutils.dir_util.copy_tree(str(input_dir), str(output_dir), preserve_symlinks=True)

    things_to_remove = ["wandb", "submission", "test"]
    for thing in things_to_remove: 
        thing_path = output_dir / thing
        if thing_path.exists() or thing_path.is_symlink():
            print(f"Removing {thing_path}")
            if thing_path.is_dir(): 
                shutil.rmtree(str(thing_path))
            else:
                os.remove(thing_path)

    with open(Path(output_dir) / "cfg.yaml", "r") as f:
        cfg = OmegaConf.load(f)
    
    cfg.inout.output_dir = str(output_dir.parent)
    cfg.inout.full_run_dir = str(output_dir)
    cfg.inout.checkpoint_dir = str(output_dir / "checkpoints")

    if 'deca_cfg' in cfg.model.keys(): 
        cfg.model.deca_cfg.inout.output_dir = "todo"
        cfg.model.deca_cfg.inout.full_run_dir = "todo"
        cfg.model.deca_cfg.inout.checkpoint_dir = "todo"
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
        # cfg.model.deca_cfg.model.emonet_model_path = str(asset_dir /  "EmotionRecognition/image_based_networks/ResNet50")
        cfg.model.deca_cfg.model.emonet_model_path = ""


    with open(output_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)


def test_loading(outpath):
    emonet = emo_network_from_path(str(outpath))

def main():

    if len(sys.argv) < 4:
        # print("Usage: package_model.py <model_dir> <output_packaged_model_dir>")
        # sys.exit()
        input_dir = "/ps/project/EmotionalFacialAnimation/emoca/emotion_network_models/new_affectnet_split/image_based_networks" \
            "/2021_11_09_04-04-52_-2546023918050637211_EmoSwin_swin_small_patch4_window7_224_shake_samp-balanced_expr_Aug_early" 
        output_dir = "/ps/project/EmotionalFacialAnimation/emoca/emotion_network_models/new_affectnet_split/image_based_networks/packaged/SWIN-S"

        asset_dir = "/home/rdanecek/Workspace/Repos/gdl/assets/"


    # input_dir = sys.argv[1]
    # output_dir = sys.argv[2]
    # asset_dir = sys.argv[3]

    if len(sys.argv) >= 4:
        overwrite = bool(int(sys.argv[3]))
    else: 
        overwrite = True


    package_model(input_dir, output_dir, asset_dir, overwrite)
    print("Model packaged.")

    test_loading(output_dir)
    print("Model loading tested")


if __name__ == "__main__":
    main()