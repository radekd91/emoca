from gdl.models.EmoDECA import EmoDECA 
from gdl.models.EmoCnnModule import EmoCnnModule 
from gdl.models.EmoSwinModule import EmoSwinModule 
from gdl.models.EmoNetModule import EmoNetModule
from gdl.models.IO import locate_checkpoint
from gdl.utils.other import class_from_str
from pathlib import Path
import sys 


def load_configs(run_path):
    from omegaconf import OmegaConf
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    if run_path != conf.inout.full_run_dir: 
        conf.inout.output_dir = str(Path(run_path).parent)
        conf.inout.full_run_dir = str(run_path)
        conf.inout.checkpoint_dir = str(Path(run_path) / "checkpoints")
    return conf


def load_model(output_dir):
    
    cfg = load_configs(output_dir)

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