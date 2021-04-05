from test_and_finetune_deca import single_stage_deca_pass
from interactive_deca_decoder import load_deca
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datasets.AffectNetDataModule import AffectNetDataModule, AffectNet, AffectNetTestModule


def load_model(path_to_models,
              run_name,
              stage,
              relative_to_path=None,
              replace_root_path=None,
              mode='best'
              ):
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    deca = load_deca(conf,
              stage,
              mode,
              relative_to_path,
              replace_root_path,
              )
    return deca, conf



def data_preparation_function(cfg):
    dm = AffectNetTestModule(
            "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/",
             "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/",
             # processed_subfolder="processed_2021_Apr_02_03-13-33",
             processed_subfolder="processed_2021_Apr_05_15-22-18",
             mode="manual",
             scale=1.25,
             test_batch_size=1
    )
    sequence_name = "affect_custom_val"
    return dm, sequence_name



def main():
    path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    path_to_now = '/home/rdanecek/Workspace/Data/now/NoW_Dataset/final_release_version/'

    run_names = []
    run_names += ['2021_03_25_19-42-13_DECA_training'] # DECA EmoNet
    # run_names += ['2021_03_18_21-10-25_DECA_training'] # Basic DECA
    # run_names += ['2021_03_26_15-05-56_DECA__DeSegFalse_DwC_early'] # Detail with coarse
    # run_names += ['2021_03_26_14-36-03_DECA__DeSegFalse_DeNone_early'] # No detail exchange


    for run_name in run_names:

        mode = 'detail'
        relative_to_path = '/ps/scratch/'
        replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'
        deca, conf = load_model(path_to_models, run_name, mode, relative_to_path, replace_root_path)

        # deca.deca.config.resume_training = True
        # deca.deca._load_old_checkpoint()
        # run_name = "Original_DECA"

        deca.eval()

        dm = data_preparation_function(conf[mode])
        conf[mode].model.test_vis_frequency = 1
        conf[mode].inout.name = "affectnet_test"
        single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_net",
                               dm=dm)
    print("We're done y'all")


if __name__ == '__main__':
    main()
