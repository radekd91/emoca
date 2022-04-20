import sys
import os 
from pathlib import Path
from typing import overload
import distutils.dir_util
from omegaconf import OmegaConf, DictConfig
import shutil
from gdl_apps.EMOCA.utils.load import load_model, replace_asset_dirs
from gdl.models.IO import locate_checkpoint

def package_model(input_dir, output_dir, asset_dir, overwrite=False, remove_bfm_textures=False):
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

    with open(Path(input_dir) / "cfg.yaml", "r") as f:
        cfg = OmegaConf.load(f)

    # copy all files and folders from input_dir to output_dir using distutils.dir_util.copy_tree
    # distutils.dir_util.copy_tree(str(input_dir), str(output_dir), preserve_symlinks=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(input_dir / "cfg.yaml"), str(output_dir / "cfg.yaml"))
    checkpoints_dir = output_dir / "detail" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # distutils.dir_util.copy_tree(str(input_dir / "detail" / "checkpoints"), str(checkpoints_dir), preserve_symlinks=True)

    checkpoint = Path(locate_checkpoint(cfg["detail"], mode="best"))
    
    # copy checkpoint file
    dst_checkpoint = checkpoints_dir / ( Path(checkpoint).relative_to(cfg.detail.inout.checkpoint_dir) )
    dst_checkpoint.parent.mkdir(parents=True, exist_ok=overwrite)
    shutil.copy(str(checkpoint), str(dst_checkpoint))


    # things_to_remove = ["wandb", "coarse", "detail/wandb", "detail/affectnet_validation_new_split_detail_test" 
    #     , "detail/detail_test", "detail/detail_train", "detail/detail_val"
    #     "submission", "test"]
    # for thing in things_to_remove: 
    #     thing_path = output_dir / thing
    #     if thing_path.exists() or thing_path.is_symlink():
    #         print(f"Removing {thing_path}")
    #         if thing_path.is_dir(): 
    #             shutil.rmtree(str(thing_path))
    #         else:
    #             os.remove(thing_path)


    cfg = replace_asset_dirs(cfg, output_dir)

    if remove_bfm_textures: 
        for mode in ["coarse", "detail"]:
            cfg[mode].model.use_texture = False

    # for mode in ["coarse", "detail"]:

    #     cfg[mode].inout.output_dir = str(output_dir.parent)
    #     cfg[mode].inout.full_run_dir = str(output_dir / mode)
    #     cfg[mode].inout.checkpoint_dir = str(output_dir / mode / "checkpoints")

    #     cfg[mode].model.tex_path = str(asset_dir / "FLAME/texture/FLAME_albedo_from_BFM.npz")
    #     cfg[mode].model.topology_path = str(asset_dir / "FLAME/geometry/head_template.obj")
    #     cfg[mode].model.fixed_displacement_path = str(asset_dir / 
    #             "FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy")
    #     cfg[mode].model.flame_model_path = str(asset_dir / "FLAME/geometry/generic_model.pkl")
    #     cfg[mode].model.flame_lmk_embedding_path = str(asset_dir / "FLAME/geometry/landmark_embedding.npy")
    #     cfg[mode].model.face_mask_path = str(asset_dir / "FLAME/mask/uv_face_mask.png")
    #     cfg[mode].model.face_eye_mask_path  = str(asset_dir / "FLAME/mask/uv_face_eye_mask.png")
    #     cfg[mode].model.pretrained_modelpath = str(asset_dir / "DECA/data/deca_model.tar")
    #     cfg[mode].model.pretrained_vgg_face_path = str(asset_dir /  "FaceRecognition/resnet50_ft_weight.pkl") 
    #     # cfg.model.emonet_model_path = str(asset_dir /  "EmotionRecognition/image_based_networks/ResNet50")
    #     cfg[mode].model.emonet_model_path = ""

    # if remove_bfm_textures: 
    #     emoca = load_model(str(output_dir.parent), output_dir.name, stage="detail")
    #     emoca.

    with open(output_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    if remove_bfm_textures: 
        # if we are removing BFM textures (distributed release), we need to remove the texture weights (which are not publicly available)
        emoca, _ = load_model(str(output_dir), output_dir, stage="detail")
        emoca.deca._disable_texture(remove_from_model=True)
        from pytorch_lightning import Trainer
        trainer = Trainer(resume_from_checkpoint=dst_checkpoint)
        trainer.model = emoca
        # overwrite the checkpoint with the new one without textures
        trainer.save_checkpoint(dst_checkpoint)
    
    # save the model 
    # emoca.save(dst_checkpoint)



def test_loading(outpath):
    outpath = Path(outpath)
    emoca = load_model(str(outpath.parent), outpath.name, stage="detail")
    print("Model loaded")

def main():

    if len(sys.argv) < 4:
        # print("Usage: package_model.py <model_dir> <output_packaged_model_dir>")
        # sys.exit()
        # EMOA
        input_dir = "/ps/project/EmotionalFacialAnimation/emoca/face_reconstruction_models/new_affectnet_split/final_models" \
            "/2021_11_13_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early" 
        output_dir = "/ps/project/EmotionalFacialAnimation/emoca/face_reconstruction_models/new_affectnet_split/final_models/packaged2/EMOCA"

        # DECA
        # input_dir = "/is/cluster/work/rdanecek/emoca/finetune_deca/2021_11_15_17-20-34_-4851631063966731039_Orig_DECA2" 
        # output_dir = "/ps/project/EmotionalFacialAnimation/emoca/face_reconstruction_models/new_affectnet_split/final_models/packaged2/DECA"


        asset_dir = "/home/rdanecek/Workspace/Repos/gdl/assets/"


    # input_dir = sys.argv[1]
    # output_dir = sys.argv[2]
    # asset_dir = sys.argv[3]

    if len(sys.argv) >= 4:
        overwrite = bool(int(sys.argv[3]))
    else: 
        overwrite = True


    package_model(input_dir, output_dir, asset_dir, overwrite, remove_bfm_textures=True)
    print("Model packaged.")

    test_loading(output_dir)
    print("Model loading tested")


if __name__ == "__main__":
    main()