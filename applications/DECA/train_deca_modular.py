from applications.DECA.test_and_finetune_deca import single_stage_deca_pass
from datasets.DecaDataModule import DecaDataModule
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import datetime


project_name = 'EmotionalDeca'

# def prepare_data(cfg):
#     dm = DecaDataModule(cfg)
#     sequence_name = "ClassicDECA"
#     return dm, sequence_name


def create_experiment_name():
    return "DECA_training"


def train_deca(configs: list, stage_types: list, stage_prefixes: list, stage_names: list, start_i=0, prepare_data=None):
    # configs = [cfg_coarse_pretraining, cfg_coarse_pretraining, cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    # stages = ["train", "test", "train", "test", "train", "test"]
    # stages_prefixes = ["pretrain", "pretrain", "", "", "", ""]

    cfg_first = configs[start_i]

    if cfg_first.inout.full_run_dir == 'todo':
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        experiment_name = time + "_" + create_experiment_name()
        full_run_dir = Path(configs[0].inout.output_dir) / experiment_name
        exist_ok = False # a path for a new experiment should not yet exist
    else:
        experiment_name = cfg_first.inout.name
        len_time_str = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        time = experiment_name[:len_time_str]
        full_run_dir = Path(cfg_first.inout.full_run_dir).parent
        exist_ok = True # a path for an old experiment should exist

    full_run_dir.mkdir(parents=True, exist_ok=exist_ok)
    print(f"The run will be saved  to: '{str(full_run_dir)}'")
    with open("out_folder.txt", "w") as f:
        f.write(str(full_run_dir))

    cfg_first_checkpoint_dir = full_run_dir / stage_names[start_i] / "checkpoints"
    cfg_first_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_first.inout.full_run_dir = str(cfg_first_checkpoint_dir.parent)
    cfg_first.inout.checkpoint_dir = str(cfg_first_checkpoint_dir)
    cfg_first.inout.name = experiment_name

    for i in range(start_i+1, len(configs)):
        cfg = configs[i]
        checkpoint_dir = full_run_dir / stage_names[i] / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

        cfg.inout.full_run_dir = str(checkpoint_dir.parent)
        cfg.inout.checkpoint_dir = str(checkpoint_dir)
        cfg.inout.name = experiment_name

    # save config to target folder
    conf = DictConfig({})
    for i in range(len(configs)):
        conf[stage_names[i]] = configs[i]

    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    wandb_logger = WandbLogger(name=experiment_name,
                         project=project_name,
                         config=dict(conf),
                         version=time,
                         save_dir=full_run_dir)

    deca = None
    checkpoint = None
    checkpoint_kwargs = None
    if start_i > 0:
        print(f"Looking for checkpoint in '{configs[start_i-1].inout.checkpoint_dir}'")
        checkpoints = sorted(list(Path(configs[start_i-1].inout.checkpoint_dir).glob("*.ckpt")))
        print(f"Found {len(checkpoints)} checkpoints")
        for ckpt in checkpoints:
            print(f" - {str(ckpt)}")
        checkpoint = str(checkpoints[-1])
        print(f"Loading a checkpoint: {checkpoint} and starting from stage {start_i}")
        checkpoint_kwargs = {
            "model_params": configs[start_i-1].model,
            "learning_params": configs[start_i-1].learning,
            "inout_params": configs[start_i-1].inout,
            "stage_name":  stage_prefixes[start_i - 1],
        }

    for i in range(start_i, len(configs)):
        cfg = configs[i]
        deca = single_stage_deca_pass(deca, cfg, stage_types[i], stage_prefixes[i], dm=None, logger=wandb_logger,
                                      data_preparation_function=prepare_data,
                                      checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs)
        checkpoint = None


def configure(cfg_default, cfg_overrides):
    from hydra.experimental import compose, initialize
    initialize(config_path="deca_conf", job_name="train_deca")
    cfg = compose(config_name=cfg_default, overrides=cfg_overrides)
    return cfg


def configure_stages(config_default_list, overrides_list):
    config_list = []
    for i in range(len(config_default_list)):
        cfg = configure(config_default_list[i], overrides_list[i])
        config_list += [cfg]
    return config_list


def resume_training(run_path, start_at_stage):
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    cfgs = []
    stage_names = []
    stage_types = []
    stage_prefixes = []
    for key in conf.keys():
        cfgs += [conf.keys()]
        stage_names += [key]

    train_deca(cfgs, stage_types, stage_prefixes, stage_names, start_i=start_at_stage)

def main():
    pass


if __name__ == "__main__":
    main()

