from applications.DECA.test_and_finetune_deca import single_stage_deca_pass, get_checkpoint_with_kwargs
from datasets.DecaDataModule import DecaDataModule
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import datetime


project_name = 'EmotionalDeca'

def prepare_data(cfg):
    dm = DecaDataModule(cfg)
    sequence_name = "ClassicDECA"
    return dm, sequence_name


def create_experiment_name():
    return "DECA_training"


def train_deca(cfg_coarse_pretraining, cfg_coarse, cfg_detail, start_i=0, resume_from_previous = True,
               force_new_location=False):
    configs = [cfg_coarse_pretraining, cfg_coarse_pretraining, cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    stages = ["train", "test", "train", "test", "train", "test"]
    stages_prefixes = ["pretrain", "pretrain", "", "", "", ""]
    # configs = [cfg_coarse_pretraining, cfg_coarse, cfg_detail]
    # stages = ["train", "train", "train",]
    # stages_prefixes = ["pretrain", "", ""]

    if start_i > 0 or force_new_location:
        if resume_from_previous:
            resume_i = start_i - 1
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the next stage {start_i})")
        else:
            resume_i = start_i
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the same stage {start_i})")
        checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(configs[resume_i], stages_prefixes[resume_i])
    else:
        checkpoint, checkpoint_kwargs = None, None

    if cfg_coarse.inout.full_run_dir == 'todo':
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        experiment_name = create_experiment_name()
        full_run_dir = Path(configs[0].inout.output_dir) / (time + "_" + experiment_name)
        exist_ok = False # a path for a new experiment should not yet exist
    else:
        experiment_name = cfg_coarse.inout.name
        len_time_str = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        if hasattr(cfg_coarse.inout, 'time') and cfg_coarse.inout.time is not None:
            time = cfg_coarse.inout.time
        else:
            time = experiment_name[:len_time_str]
        full_run_dir = Path(cfg_coarse.inout.full_run_dir).parent
        exist_ok = True # a path for an old experiment should exist

    full_run_dir.mkdir(parents=True, exist_ok=exist_ok)
    print(f"The run will be saved  to: '{str(full_run_dir)}'")
    with open("out_folder.txt", "w") as f:
        f.write(str(full_run_dir))

    coarse_pretrain_checkpoint_dir = full_run_dir / "coarse_pretrain" / "checkpoints"
    coarse_pretrain_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_coarse_pretraining.inout.full_run_dir = str(coarse_pretrain_checkpoint_dir.parent)
    cfg_coarse_pretraining.inout.checkpoint_dir = str(coarse_pretrain_checkpoint_dir)
    cfg_coarse_pretraining.inout.name = experiment_name
    cfg_coarse_pretraining.inout.time = time

    coarse_checkpoint_dir = full_run_dir / "coarse" / "checkpoints"
    coarse_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_coarse.inout.full_run_dir = str(coarse_checkpoint_dir.parent)
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_coarse.inout.name = experiment_name
    cfg_coarse.inout.time = time

    # if cfg_detail.inout.full_run_dir == 'todo':
    detail_checkpoint_dir = full_run_dir / "detail" / "checkpoints"
    detail_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_detail.inout.full_run_dir = str(detail_checkpoint_dir.parent)
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)
    cfg_detail.inout.name = experiment_name
    cfg_detail.inout.time = time

    # save config to target folder
    conf = DictConfig({})
    conf.coarse_pretraining = cfg_coarse_pretraining
    conf.coarse = cfg_coarse
    conf.detail = cfg_detail
    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    wandb_logger = WandbLogger(name=experiment_name,
                         project=project_name,
                         config=OmegaConf.to_container(conf),
                         version=time,
                         save_dir=full_run_dir)

    deca = None
    if start_i > 0 or force_new_location:
        print(f"Loading a checkpoint: {checkpoint} and starting from stage {start_i}")

    for i in range(start_i, len(configs)):
        cfg = configs[i]
        deca = single_stage_deca_pass(deca, cfg, stages[i], stages_prefixes[i], dm=None, logger=wandb_logger,
                                      data_preparation_function=prepare_data,
                                      checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs)
        checkpoint = None


def configure(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
              coarse_cfg_default, coarse_overrides,
              detail_cfg_default, detail_overrides):
    from hydra.experimental import compose, initialize
    initialize(config_path="deca_conf", job_name="train_deca")
    cfg_coarse_pretrain = compose(config_name=coarse_pretrain_cfg_default, overrides=coarse_pretrain_overrides)
    cfg_coarse = compose(config_name=coarse_cfg_default, overrides=coarse_overrides)
    cfg_detail = compose(config_name=detail_cfg_default, overrides=detail_overrides)
    return cfg_coarse_pretrain, cfg_coarse, cfg_detail



def configure_and_train(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                        coarse_cfg_default, coarse_overrides,
                        detail_cfg_default, detail_overrides):
    cfg_coarse_pretrain, cfg_coarse, cfg_detail = configure(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                                       coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)
    train_deca(cfg_coarse_pretrain, cfg_coarse, cfg_detail)


def configure_and_resume(run_path,
                         coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                         coarse_cfg_default, coarse_overrides,
                         detail_cfg_default, detail_overrides,
                         start_at_stage):
    cfg_coarse_pretrain, cfg_coarse, cfg_detail = configure(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                                       coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)

    cfg_coarse_pretrain_, cfg_coarse_, cfg_detail_ = load_configs(run_path)

    if start_at_stage < 2:
        raise RuntimeError("Resuming before stage 2 makes no sense, that would be training from scratch")
    if start_at_stage == 2:
        cfg_coarse_pretrain = cfg_coarse_pretrain_
    elif start_at_stage == 3:
        raise RuntimeError("Resuming for stage 3 makes no sense, that is a testing stage")
    elif start_at_stage == 4:
        cfg_coarse_pretrain = cfg_coarse_pretrain_
        cfg_coarse = cfg_coarse_
    elif start_at_stage == 5:
        raise RuntimeError("Resuming for stage 5 makes no sense, that is a testing stage")
    else:
        raise RuntimeError(f"Cannot resume at stage {start_at_stage}")

    train_deca(cfg_coarse_pretrain, cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=True, #important, resume from previous stage's checkpoint
               force_new_location=True)


def load_configs(run_path):
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    cfg_coarse_pretraining = conf.coarse_pretraining
    cfg_coarse = conf.coarse
    cfg_detail = conf.detail
    return cfg_coarse_pretraining, cfg_coarse, cfg_detail


def resume_training(run_path, start_at_stage, resume_from_previous, force_new_location):
    # with open(Path(run_path) / "cfg.yaml", "r") as f:
    #     conf = OmegaConf.load(f)
    # cfg_coarse_pretraining = conf.coarse_pretraining
    # cfg_coarse = conf.coarse
    # cfg_detail = conf.detail
    cfg_coarse_pretraining, cfg_coarse, cfg_detail = load_configs(run_path)
    train_deca(cfg_coarse_pretraining, cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=resume_from_previous,
               force_new_location=force_new_location)

# @hydra.main(config_path="deca_conf", config_name="deca_finetune")
# def main(cfg : DictConfig):
def main():
    configured = False
    if len(sys.argv) >= 4:
        if Path(sys.argv[1]).is_file():
            configured = True
            with open(sys.argv[1], 'r') as f:
                coarse_pretrain_conf = OmegaConf.load(f)
            with open(sys.argv[2], 'r') as f:
                coarse_conf = OmegaConf.load(f)
            with open(sys.argv[3], 'r') as f:
                detail_conf = OmegaConf.load(f)
        else:
            coarse_pretrain_conf = sys.argv[1]
            coarse_conf = sys.argv[2]
            detail_conf = sys.argv[3]
    else:
        coarse_pretrain_conf = "deca_train_coarse_pretrain"
        coarse_conf = "deca_train_coarse"
        detail_conf = "deca_train_detail"

    if len(sys.argv) >= 7:
        coarse_pretrain_override = sys.argv[4]
        coarse_override = sys.argv[5]
        detail_override = sys.argv[6]
    else:
        coarse_pretrain_override = []
        coarse_override = []
        detail_override = []
    if configured:
        train_deca(coarse_pretrain_conf, coarse_conf, detail_conf)
    else:
        configure_and_train(coarse_pretrain_conf, coarse_pretrain_override,
                            coarse_conf, coarse_override, detail_conf, detail_override)


if __name__ == "__main__":
    main()

