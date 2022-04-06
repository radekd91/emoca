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
from gdl_apps.EMOCA.utils.load import load_model
import sys
from gdl.datasets.AffectNetDataModule import AffectNetTestModule


def data_preparation_function(cfg,path_to_affectnet, path_to_processed_affectnet):
    dm = AffectNetTestModule(
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
    run_name = sys.argv[1]

    if len(sys.argv) > 2:
        mode = sys.argv[2]
    else:
        mode = 'detail'
    deca, conf = load_model(path_to_models, run_name, mode, allow_stage_revert=True)

    deca.eval()

    dm = data_preparation_function(conf[mode], path_to_affectnet, path_to_processed_affectnet)
    conf[mode].model.test_vis_frequency = 1
    # conf[mode].inout.name = "affectnet_test"
    # conf[mode].inout.name = "affectnet_test_" + conf[mode].inout.name
    conf[mode].inout.name = "afft_" + conf[mode].inout.name
    import datetime
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    conf[mode].inout.random_id = str(hash(time))
    print(f"Beginning testing for '{run_name}' in mode '{mode}'")
    single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_net", dm=dm, project_name_="AffectNetTests")
    print("We're done y'all")


if __name__ == '__main__':
    main()
