import os, sys
from pathlib import Path

# import NoW related stuff
sys.path += [str(Path(__file__).absolute().parents[3] / "now_evaluation")]
from main import generating_cumulative_error_plots


def main():
    stage = 'detail'

    run_names = {}
    run_names['Original_DECA'] = "Paper DECA" # Original DECA from Yao
    run_names['2021_03_18_21-10-25_DECA_training'] = "My retrained DECA" # Basic DECA
    run_names['2021_03_25_19-42-13_DECA_training'] = "My retrained with EmoNet loss" # DECA EmoNet
    run_names['2021_03_26_15-05-56_DECA__DeSegFalse_DwC_early'] = "Detail with coarse jointly" # Detail with coarse
    run_names['2021_03_26_14-36-03_DECA__DeSegFalse_DeNone_early'] = "DECA no detail exchange" # No detail exchange

    use_dense_topology = False
    # use_dense_topology = True

    path_to_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/finetune_deca'


    run_files = []
    nicks = []
    for run_name, nick in run_names.items():
        if use_dense_topology:
            savefolder = Path(path_to_models) / run_name / stage / "NoW_dense"
        else:
            savefolder = Path(path_to_models) / run_name / stage / "NoW_flame"

        run_files += [str(savefolder / "results" / "_computed_distances.npy")]
        nicks += [nick]

    generating_cumulative_error_plots(run_files, nicks, "out.png")


if __name__ == "__main__":
    main()
