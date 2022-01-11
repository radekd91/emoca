from gdl_apps.EmotionRecognition.training.train_emodeca import resume_training
import sys, os


def main():
    resume_from = sys.argv[1]
    stage = int(sys.argv[2])
    resume_from_previous = bool(int(sys.argv[3]))
    force_new_location = bool(int(sys.argv[4]))

    resume_training(resume_from, start_at_stage=stage,
                    resume_from_previous=resume_from_previous, force_new_location=force_new_location)


if __name__ == "__main__":
    main()
