from applications.DECA.test_and_finetune_deca import single_stage_deca_pass
from datasets.DecaDataModule import DecaDataModule
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import datetime


project_name = 'Deca'

def prepare_data(cfg):
    dm = DecaDataModule(cfg)
    sequence_name = "ClassicDECA"
    return dm, sequence_name


def create_experiment_name():
    return "DECA_training"


def train_deca(cfg_coarse_pretraining, cfg_coarse, cfg_detail):
    conf = DictConfig({})
    conf.coarse_pretraining = cfg_coarse_pretraining
    conf.coarse = cfg_coarse
    conf.detail = cfg_detail
    # configs = [cfg_coarse_pretraining, cfg_coarse_pretraining, cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    # stages = ["train", "test", "train", "test", "train", "test"]
    # stages_prefixes = ["pretrain", "pretrain", "", "", "", ""]
    configs = [cfg_coarse_pretraining, cfg_coarse, cfg_detail]
    stages = ["train", "train", "train",]
    stages_prefixes = ["pretrain", "", ""]

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    experiment_name = time + "_" + create_experiment_name()

    if cfg_coarse.inout.full_run_dir == 'todo':
        full_run_dir = Path(configs[0].inout.output_dir) / experiment_name
    else:
        full_run_dir = cfg_coarse.inout.full_run_dir

    full_run_dir.mkdir(parents=True)
    print(f"The run will be saved  to: '{str(full_run_dir)}'")
    with open("out_folder.txt", "w") as f:
        f.write(str(full_run_dir))

    coarse_pretrain_checkpoint_dir = full_run_dir / "coarse_pretrain"
    coarse_pretrain_checkpoint_dir.mkdir(parents=True)

    cfg_coarse_pretraining.inout.full_run_dir = str(full_run_dir / "coarse_pretrain")
    cfg_coarse_pretraining.inout.checkpoint_dir = str(coarse_pretrain_checkpoint_dir)
    cfg_coarse_pretraining.inout.name = experiment_name

    coarse_checkpoint_dir = full_run_dir / "coarse"
    coarse_checkpoint_dir.mkdir(parents=True)

    cfg_coarse.inout.full_run_dir = str(full_run_dir / "coarse")
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_coarse.inout.name = experiment_name

    # if cfg_detail.inout.full_run_dir == 'todo':
    detail_checkpoint_dir = full_run_dir / "detail"
    detail_checkpoint_dir.mkdir(parents=True)

    cfg_detail.inout.full_run_dir = str(full_run_dir / "detail")
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)
    cfg_detail.inout.name = experiment_name

    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    wandb_logger = WandbLogger(name=experiment_name,
                         project=project_name,
                         config=dict(conf),
                         version=time,
                         save_dir=full_run_dir)

    #TODO: create a proper config for each stage
    #1) first train coarse without photometric
    #2) then train coarse with photometric
    #3) then train detail

    deca = None
    for i, cfg in enumerate(configs):
        deca = single_stage_deca_pass(deca, cfg, stages[i], stages_prefixes[i], dm=None, logger=wandb_logger,
                                      data_preparation_function=prepare_data)


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

