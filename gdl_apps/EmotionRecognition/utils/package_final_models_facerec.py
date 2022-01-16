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
    output_dir = asset_dir / "EmotionRecognition" / "facerec_based_models"
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
