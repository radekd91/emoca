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


from gdl_apps.EMOCA.training.test_and_finetune_deca  import single_stage_deca_pass
from gdl_apps.EMOCA.utils.load import load_deca
from omegaconf import DictConfig, OmegaConf
import os, sys
from pathlib import Path
from gdl.datasets.AfewVaDataModule import AfewVaDataVisTestModule


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
    dm = AfewVaDataVisTestModule(
            path_to_affectnet,
             path_to_processed_affectnet,
             processed_subfolder="processed_2021_Nov_07_23-37-18",
             scale=1.25,
        val_batch_size=1,
             test_batch_size=1,
             processed_ext=".png",
                split_seed=0,
                train_fraction=0.6,
                val_fraction=0.2,
                test_fraction=0.2,
        num_workers=4,
    )
    return dm



def main():
    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    # path_to_affectnet = "/ps/project/EmotionalFacialAnimation/data/affectnet/"
    path_to_affectnet = "/is/cluster/work/rdanecek/data/afew-va_not_processed/"
    # path_to_processed_affectnet = "/ps/scratch/rdanecek/data/affectnet/"
    path_to_processed_affectnet = "/is/cluster/work/rdanecek/data/afew-va/"
    run_name = sys.argv[1]

    if len(sys.argv) > 2:
        mode = sys.argv[2]
    else:
        mode = 'coarse'

    deca, conf = load_model(path_to_models, run_name, mode, allow_stage_revert=True)

    # now that we loaded, let's reconfigure to detail
    if mode != 'detail':
        mode = 'detail'
        deca.reconfigure(conf[mode].model, conf[mode].inout, conf[mode].learning, stage_name="",
                         downgrade_ok=False, train=False)
    deca.eval()

    import wandb
    api = wandb.Api()
    name = str(Path(run_name).name)
    idx = name.find("ExpDECA")
    run_id = name[:idx - 1]
    run = api.run("rdanecek/EmotionalDeca/" + run_id)
    tags = run.tags
    conf["coarse"]["learning"]["tags"] = tags
    conf["detail"]["learning"]["tags"] = tags

    dm = data_preparation_function(conf[mode], path_to_affectnet, path_to_processed_affectnet)
    conf[mode].model.test_vis_frequency = 1
    # conf[mode].inout.name = "affectnet_test"
    # conf[mode].inout.name = "affectnet_test_" + conf[mode].inout.name
    # conf[mode].inout.name = "afft_" + conf[mode].inout.name
    conf[mode].inout.name = conf[mode].inout.name
    import datetime
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    conf[mode].inout.random_id = str(hash(time))
    print(f"Beginning testing for '{run_name}' in mode '{mode}'")
    single_stage_deca_pass(deca, conf[mode], stage="test", prefix="afew_test_new_split",
                           dm=dm, project_name_="AfewVATestVis")
    print("We're done y'all")


if __name__ == '__main__':
    main()
