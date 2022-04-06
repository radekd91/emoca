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


from gdl_apps.EMOCA.training.test_and_finetune_deca import single_stage_deca_pass
from gdl_apps.EMOCA.utils.load import load_deca
from omegaconf import DictConfig, OmegaConf
import os, sys
from pathlib import Path
from gdl.datasets.AffectNetDataModule import AffectNetEmoNetSplitTestModule
from omegaconf import open_dict

def load_model(path_to_models,
              run_name,
              stage,
              relative_to_path=None,
              replace_root_path=None,
              mode='best',
              allow_stage_revert=False, # allows to load coarse if detail checkpoint not found
              ):
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    deca = load_deca(conf,
              stage,
              mode,
              relative_to_path,
              replace_root_path,
              terminate_on_failure= not allow_stage_revert
              )
    if deca is None and allow_stage_revert:
        deca = load_deca(conf,
                         "coarse",
                         mode,
                         relative_to_path,
                         replace_root_path,
                         )

    return deca, conf



def data_preparation_function(cfg,path_to_affectnet, path_to_processed_affectnet):
    dm = AffectNetEmoNetSplitTestModule(
            path_to_affectnet,
             path_to_processed_affectnet,
             # processed_subfolder="processed_2021_Apr_02_03-13-33",
             processed_subfolder="processed_2021_Apr_05_15-22-18",
             mode="manual",
             scale=1.25,
             test_batch_size=1
    )
    return dm



def main():
    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_affectnet = "/ps/project/EmotionalFacialAnimation/data/affectnet/"
    # path_to_processed_affectnet = "/ps/scratch/rdanecek/data/affectnet/"
    path_to_processed_affectnet = "/is/cluster/work/rdanecek/data/affectnet/"
    # run_name = sys.argv[1]

    # if len(sys.argv) > 2:
    #     mode = sys.argv[2]
    # else:
    #     mode = 'coarse'
    mode = 'detail'


    run_name = '2021_03_26_15-05-56_Orig_DECA2'  # Detail with coarse

    # deca, conf = load_model(path_to_models, run_name, mode, allow_stage_revert=True)

    import datetime
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    # now that we loaded, let's reconfigure to detail
    # if mode != 'detail':
    #     mode = 'detail'
    #     deca.reconfigure(conf[mode].model, conf[mode].inout, conf[mode].learning, stage_name="",
    #                      downgrade_ok=False, train=False)
    # deca.eval()

    relative_to_path = None
    replace_root_path = None

    # for run_name in run_names:
    # print(f"Beginning testing for '{run_name}' in mode '{mode}'")
    deca, conf = load_model(path_to_models, run_name, mode, relative_to_path, replace_root_path)
    # conf.learning.logger_type = None
    # conf.data.root_dir = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"
    # conf.data.root_dir = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"
    # conf[mode].data.root_dir = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"

    deca.deca.config.resume_training = True
    # deca.deca.config.pretrained_modelpath = "/home/rdanecek/Workspace/Repos/DECA/data/deca_model.tar"
    deca.deca.config.pretrained_modelpath = "/lustre/home/rdanecek/workspace/repos/DECA/data/deca_model.tar"
    deca.deca._load_old_checkpoint()
    run_name = "Original_DECA"

    dm = data_preparation_function(conf, path_to_affectnet, path_to_processed_affectnet)
    conf[mode].model.test_vis_frequency = 1
    # conf[mode].inout.name = "EMOCA"
    with open_dict(conf["coarse"].model):
        conf["coarse"].model["deca_class"] = "EMOCA"
    with open_dict(conf["detail"].model):
        conf["detail"].model["deca_class"] = "EMOCA"
    conf[mode].inout.random_id = str(hash(time))
    conf[mode].inout.time = time
    conf[mode].inout.full_run_dir = str(Path( conf[mode].inout.output_dir) / (time + "_" + conf[mode].inout.random_id + "_" + conf[mode].inout.name) /  mode)
    conf[mode].inout.checkpoint_dir = str(Path(conf[mode].inout.full_run_dir) / "checkpoints")
    Path(conf[mode].inout.full_run_dir).mkdir(parents=True)

    print(f"Beginning testing for '{run_name}' in mode '{mode}'")
    single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affectnet_validation_new_split", dm=dm, project_name_="AffectNetTestsNewSplit")
    print("We're done y'all")

if __name__ == '__main__':
    main()
