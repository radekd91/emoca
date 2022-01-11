import wandb
import pandas
from pathlib import Path
import json
from omegaconf import DictConfig, OmegaConf

def main():
    api = wandb.Api()
    root = Path("/is/cluster/work/rdanecek/emoca/emodeca/")

    run_names = []
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-35-37_-1161116375494981412_EmoMGCNet_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-35-36_-8046457176211889687_EmoExpNet_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-33-07_1843147884226165688_EmoExpNet_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-15-27_7268323253792821996_EmoDeep3DFace_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-13_6949616983506395245_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-13_-4341500954807723397_EmoDECA_AfewV_Orig_nl-4BatchNorm1d_id_exp_jaw_detail_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-12_-883171165053951196_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-11_-8771568993008847876_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-11_-5001598437032731747_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-11_-4648497253982293620_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-10_-4988270973026329720_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-07_-1357828887451007047_Emo3DDFA_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-04_7382187467192733770_EmoDECA_AfewV_Orig_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-04_68940662102047876_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-03_2775975653132874343_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-03-03_-8396122457639155494_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-02-51_4002177680479129605_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_21-02-51_-6723661694728668670_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_20-58-59_7591881836062511055_EmoDECA_AfewV_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-38-34_-5108247086344313851_EmoSwin_swin_base_patch4_window7_224_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-38-34_-2551392040533364774_EmoCnn_resnet50_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-38-14_-4363274065263711458_EmoCnn_resnet50_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-38-10_-675529272828243504_EmoSwin_swin_tiny_patch4_window7_224_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-38-08_5239516111744668401_EmoSwin_swin_base_patch4_window7_224_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-38-05_-1021932498572718565_EmoSwin_swin_tiny_patch4_window7_224_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-38-05_-3698006624804482910_EmoCnn_vgg19_bn_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_15_02-37-41_-4513972209100694206_EmoCnn_vgg19_bn_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-39-11_8967464683573230563_EmoSwin_swin_base_patch4_window7_224_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-39-00_4363551435501292966_EmoCnn_vgg19_bn_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-39-00_5784706256688858056_EmoCnn_resnet50_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_17-38-43_-2897877075877919183_EmoSwin_swin_tiny_patch4_window7_224_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-52_5063894700880742144_EmoSwin_swin_base_patch4_window7_224_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-06_5824575168153882277_EmoSwin_swin_tiny_patch4_window7_224_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-51-03_1650355822631362210_EmoCnn_vgg19_bn_shake_Aug_early"]
    run_names += [
        "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_14_04-50-45_-1078162224083866132_EmoCnn_resnet50_shake_Aug_early"]

    leave_out_not_metioned_runs = True
    # leave_out_not_metioned_runs = False

    # for run_key, run_dir in run_names.items():
    runs = api.runs("rdanecek/EmoDECA_Afew-VA")

    for run in runs:
        cfg = DictConfig(json.loads(run.json_config))
        entry_dict = {}
        # entry_dict["run_id"] = cfg.inout.value.name
        id = cfg.inout.value.random_id

        if cfg.inout.value.full_run_dir in run_names:
            print(f'final_model_nicknames["{id}"]= "{cfg.inout.value.name}"')

        # print(f"random_id: {id}")
        # print(f"name: {cfg.inout.value.name}" )
        # print(f"link: {run.url}")
        # print("--------------------------")




def format(num):
    return f"{num:.02f}"

if __name__ == '__main__':
    main()
