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


# from gdl.models.external.Face_3DDFA_v2 import Face3DDFAModule
# from gdl.models.external.Deep3DFace import Deep3DFaceModule
import time as t
from affectnet_mturk import *
from omegaconf import open_dict
# def str2module(class_name):
#     if class_name in ["3ddfa", "Face3DDFAModule"]:
#         return Face3DDFAModule
#     if class_name in ["deep3dface", "Deep3DFaceModule"]:
#         return Deep3DFaceModule
#     raise NotImplementedError(f"Not supported for {class_name}")


# def instantiate_other_face_models(cfg, stage, prefix, checkpoint=None, checkpoint_kwargs=None):
#     module_class = str2module(cfg.model.deca_class)
#
#     if checkpoint is None:
#         face_model = module_class(cfg.model, cfg.learning, cfg.inout, prefix)
#
#     else:
#         checkpoint_kwargs = checkpoint_kwargs or {}
#         face_model = module_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
#         # if stage == 'train':
#         #     mode = True
#         # else:
#         #     mode = False
#         # face_model.reconfigure(cfg.model, cfg.inout, cfg.learning, prefix, train=mode)
#     return face_model



def main():
    # path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    #
    # path_to_affectnet = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"
    path_to_affectnet = "/ps/project/EmotionalFacialAnimation/data/affectnet/"
    # path_to_affectnet = "/ps/project_cifs/EmotionalFacialAnimation/data/affectnet/"
    # path_to_processed_affectnet = "/home/rdanecek/Workspace/mount/scratch/rdanecek/data/affectnet/"

    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    # path_to_affectnet = "/ps/project/EmotionalFacialAnimation/data/affectnet/"
    # path_to_processed_affectnet = "/ps/scratch/rdanecek/data/affectnet/"
    path_to_processed_affectnet = "/is/cluster/work/rdanecek/data/affectnet/"

    run_names = []
    # run_names += ['2021_03_26_15-05-56_Orig_DECA']  # Detail with coarse
    run_names += ['2021_03_26_15-05-56_Orig_DECA2']  # Detail with coarse

    mode = 'detail'
    # mode = 'coarse'

    # for run_name in run_names:
    # print(f"Beginning testing for '{conf.model.deca_class}'.")

    # relative_to_path = None
    # replace_root_path = None
    #
    # face_model, conf = load_model(path_to_models, run_name, mode, relative_to_path, replace_root_path)
    # # # face_model, conf = load_model(path_to_models, run_name, mode)
    # #
    # # run_name = conf[mode].inout.name
    # #
    # # face_model.face_model.config.resume_training = True
    # # face_model.face_model.config.pretrained_modelpath = "/home/rdanecek/Workspace/Repos/DECA/data/deca_model.tar"
    # # face_model.face_model._load_old_checkpoint()
    # # run_name = "Original_DECA"

    import datetime
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    mode = 'detail'
    # mode = 'coarse'

    relative_to_path = None
    replace_root_path = None

    for run_name in run_names:
        print(f"Beginning testing for '{run_name}' in mode '{mode}'")
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

        single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_net_mturk", dm=dm, project_name_="AffectNetMTurkTests",
                               )
        # t.sleep(3600)
        print("We're done y'all")


if __name__ == '__main__':
    main()
