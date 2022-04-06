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


from gdl_apps.EMOCA.training.train_deca import resume_training
import sys, os


def main():

    if len(sys.argv) >= 4:
        resume_from = sys.argv[1]
        stage = int(sys.argv[2])
        resume_from_previous = bool(int(sys.argv[3]))
        force_new_location = bool(int(sys.argv[4]))
    else:
        raise RuntimeError("INvalid arugments passed.")
        stage = 2
        resume_from_previous = True
        force_new_location = False
        # resume_from = '/ps/scratch/rdanecek/emoca/finetune_deca/2021_03_09_10-04-28_vaCoPhotoCoLMK_IDW-0.15_Aug_early'
        # resume_from = '/ps/scratch/rdanecek/emoca/finetune_deca/2021_04_02_18-46-51_va_DeSegFalse_DeNone_Aug_DwC_early'
        resume_from = '/is/cluster/work/rdanecek/emoca/finetune_deca/2021_08_29_10-31-15_DECAStar_DecaD_VGGl_DeSegrend_Deex_early/'
    resume_training(resume_from, start_at_stage=stage,
                    resume_from_previous=resume_from_previous, force_new_location=force_new_location)


if __name__ == "__main__":
    main()

