from gdl_apps.DECA.train_stardeca import resume_training
import sys, os


def main():
    resume_from = sys.argv[1]
    stage = int(sys.argv[2])
    resume_from_previous = bool(int(sys.argv[3]))
    force_new_location = bool(int(sys.argv[4]))

    # resume_from = '/ps/scratch/rdanecek/emoca/finetune_deca/2021_03_09_10-04-28_vaCoPhotoCoLMK_IDW-0.15_Aug_early'
    # resume_from = '/ps/scratch/rdanecek/emoca/finetune_deca/2021_04_02_18-46-51_va_DeSegFalse_DeNone_Aug_DwC_early'
    resume_training(resume_from, start_at_stage=stage,
                    resume_from_previous=resume_from_previous, force_new_location=force_new_location)


if __name__ == "__main__":
    main()

