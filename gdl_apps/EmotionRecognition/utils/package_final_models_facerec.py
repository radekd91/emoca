from gdl_apps.EmotionRecognition.utils.package_model import package_model, test_loading
import gdl
from pathlib import Path
from gdl.utils.other import class_from_str
import sys
from gdl.models.EmoDECA import EmoDECA
from gdl.models.IO import locate_checkpoint


def load_configs(run_path):
    from omegaconf import OmegaConf
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    if run_path != conf.inout.full_run_dir: 
        conf.inout.output_dir = str(Path(run_path).parent)
        conf.inout.full_run_dir = str(run_path)
        conf.inout.checkpoint_dir = str(Path(run_path) / "checkpoints")
    return conf


def test_loading(output_dir):
    
    cfg = load_configs(output_dir)

    if 'emodeca_type' in cfg.model:
        deca_class = class_from_str(cfg.model.emodeca_type, sys.modules[__name__])
    else:
        deca_class = EmoDECA

    checkpoint_kwargs = {'config': cfg}
    checkpoint = locate_checkpoint(cfg, mode="best")
    deca = deca_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
    return deca


def main():
    asset_dir =  Path(gdl.__file__).parents[1] / "assets"
    input_dir = Path("/ps/project/EmotionalFacialAnimation/emoca/emorec_models_comparison/affectnet/new_split/validation") 
    output_dir = Path(gdl.__file__).parents[1] / "assets" / "EmotionRecognition" / "facerec_based_models"
    output_dir.mkdir(exist_ok=True, parents=True)
    model_dirs = {
        "2021_11_10_16-33-08_2929045501486288941_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early": "EMOCA-emorec",  
    }

    for model, name in model_dirs.items():
        print(f"Packing model {name}")
        package_model(input_dir / model, output_dir / name, asset_dir, overwrite=True)
        test_loading(output_dir / name)
        print("Model loading tested")


if __name__ == "__main__": 
    main() 
