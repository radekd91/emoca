import wandb
import pandas
from pathlib import Path
import json
from omegaconf import DictConfig, OmegaConf

def main():
    api = wandb.Api()
    root = Path("/is/cluster/work/rdanecek/emoca/emodeca/")

    # run_names = {}
    # # some of the best candidates
    # run_names["0"] =\
    #     "2021_11_12_19-56-13_704003715291275370_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["1"] =\
    #     "2021_11_11_15-42-30_8680779076656978317_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["2"] =\
    #     "2021_11_11_12-13-16_-8024089022190881636_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["3"] =\
    #     "2021_11_11_01-59-07_-9007648997833454518_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["4"] =\
    #     "2021_11_11_01-58-56_1043302978911105834_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["5"] =\
    #     "2021_11_10_20-58-31_-7948033884851958030_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["6"] =\
    #     "2021_11_10_20-58-27_-5553059236244394333_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["7"] =\
    #     "2021_11_10_20-57-28_-4957717700349337532_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["8"] =\
    #     "2021_11_10_16-34-49_8015192522733347822_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["9"] =\
    #     "2021_11_10_16-33-02_-5975857231436227431_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["10"] =\
    #     "2021_11_10_16-33-00_-1889770853677981780_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"
    # run_names["11"] =\
    #     "2021_11_10_16-32-49_-6879167987895418873_EmoDECA_Affec_ExpDECA_nl-4BatchNorm1d_id_exp_jaw_shake_samp-balanced_expr_early"

    # create a pandas table with the following colums: run_name, run_folder, score
    metrics_pretty_names = ["Valence PCC", "Valence CCC", "Valence RMSE", "Valence SAGR",
                            "Arousal PCC", "Arousal CCC", "Arousal RMSE", "Arousal SAGR",
                            "Expression Accuracy"]
    metrics_columns = ["v_pcc", "v_ccc", "v_rmse", "v_sagr", "a_pcc", "a_ccc", "a_rmse", "a_sagr", "expr_acc"]

    table = pandas.DataFrame(columns=["run_name", *metrics_columns ])

    prefix = "test_metric_"
    # dl = "_epoch/dataloader_idx_1"
    dl = "_epoch"


    keys_to_recover = [ prefix + key + dl for key in metrics_columns]

    runs = api.runs("rdanecek/AffectNetEmoDECATest")

    final_model_nicknames = {}
    final_model_nicknames["1035135348370037204"] = "ResNet 50"
    final_model_nicknames["3217986143633947716"] = "VGG19 BN"
    final_model_nicknames["2984454094000662996"] = "SWIN-B"
    final_model_nicknames["-6722575657717322954"] = "SWIN-T"
    final_model_nicknames["-365025347384386566"] = "EmoNet Ours"
    # final_model_nicknames["8867238002216919009"] = "EmoNet Original"
    final_model_nicknames["-4553010808430891931"] = "EmoNet Original"
    # final_model_nicknames["-1082829187662313332"] = "EMOCA-SWIN 10" #without identity
    final_model_nicknames["-8892368489464891714"] = "EMOCA-SWIN 10" # with identity
    # final_model_nicknames["5167931985213549187"] = "EMOCA-SWIN 5" # without identity
    final_model_nicknames["-4070931720609577319"] = "EMOCA-SWIN 5" # with identity
    final_model_nicknames["2839023693996900189"] = "EMOCA-SWIN 1"
    # final_model_nicknames["1338802996723128481"] = "EMOCA-SWIN 1 DecaD" # without identity
    final_model_nicknames["1916228855864926169"] = "EMOCA-SWIN 1 DecaD" # with identity
    # final_model_nicknames["-7671276478970211630"] = "EMOCA-ResNet 5" # without identity
    final_model_nicknames["-3448168656212712789"] = "EMOCA-ResNet 5" # with identity
    final_model_nicknames["-1667632836893565667"] = "EMOCA-ResNet 1"
    final_model_nicknames["-7287099288123125386"] = "EMOCA-ResNet 1 DecaD"
    final_model_nicknames["-7163854018263734313"] = "EMOCA-ResNet cos"
    final_model_nicknames["-4010187096645607381"] = "EMOCA-ResNet L1"
    # final_model_nicknames["4154598636886285872"] = "EMOCA-ResNet not balanced" # without identity
    final_model_nicknames["-2939789230769256174"] = "EMOCA-ResNet not balanced" # with identity
    final_model_nicknames["-7787195888843808313"] = "DECA" # with identity
    final_model_nicknames["7017738612092177501"] = "DECA detail" # with identity and detail
    # final_model_nicknames["-3858966055357166666"] = "MGCNet exp"
    final_model_nicknames["5479712542688490481"] = "MGCNet exp+id"
    # final_model_nicknames["683485161921968157"] = "ExpNet exp" # why nan?
    # final_model_nicknames["6513288339971951709"] = "ExpNet exp+id" # why nan?
    final_model_nicknames["4468883309025587637"] = "ExpNet exp+id" # no nan
    # final_model_nicknames["6235434408676196359"] = "3DDFA-v2 mobilenet exp"
    # final_model_nicknames["6239115371183456618"] = "3DDFA-v2 mobilenet exp+id"
    # final_model_nicknames["5168057079614915647"] = "3DDFA-v2 exp"
    final_model_nicknames["-6233569188012021071"] = "3DDFA-v2 exp+id"
    # final_model_nicknames["8190019668768274600"] = "Deep3DFace exp"
    final_model_nicknames["-8709498652981536266"] = "Deep3DFace exp+id"

    order = {}
    idx = 0
    order["ResNet 50"] = idx; idx+=1
    order["VGG19 BN"] =  idx; idx+=1
    order["SWIN-B"] =  idx; idx+=1
    order["SWIN-T"] =  idx; idx+=1
    order["EmoNet Ours"] =  idx; idx+=1
    # order["EmoNet Original"] =  idx; idx+=1
    order["EmoNet Original"] = idx; idx+=1
    # order["-EMOCA-SWIN 10"] =  idx; idx+=1 #without identity
    order["EMOCA-SWIN 10"] =  idx; idx+=1 # with identity
    # order["EMOCA-SWIN 5"] =  idx; idx+=1 # without identity
    order["EMOCA-SWIN 5"] =  idx; idx+=1 # with identity
    order["EMOCA-SWIN 1"] =  idx; idx+=1
    # order["EMOCA-SWIN 1 DecaD"] =  idx; idx+=1# without identity
    order["EMOCA-SWIN 1 DecaD"] = idx; idx+=1 # with identity
    # order["EMOCA-ResNet 5"] =  idx; idx+=1 # without identity
    order["EMOCA-ResNet 5"] =  idx; idx+=1 # with identity
    order["EMOCA-ResNet 1"] =  idx; idx+=1
    order["EMOCA-ResNet 1 DecaD"] =  idx; idx+=1
    order["EMOCA-ResNet cos"] =  idx; idx+=1
    order["EMOCA-ResNet L1"] =  idx; idx+=1
    # order["EMOCA-ResNet not balanced"] =  idx; idx+=1 # without identity
    order["EMOCA-ResNet not balanced"] =  idx; idx+=1 # with identity
    order["DECA"] = idx; idx+=1
    order["DECA detail"] =  idx; idx+=1
    # order["MGCNet exp"] =  idx; idx+=1
    order["MGCNet exp+id"] =  idx; idx+=1
    # order["ExpNet exp"] =  idx; idx+=1
    order["ExpNet exp+id"] =  idx; idx+=1
    # order["3DDFA-v2 mobilenet exp"] =  idx; idx+=1
    # order["DDFA-v2 mobilenet exp+id"] =  idx; idx+=1
    # order["3DDFA-v2 exp"] =  idx; idx+=1
    order["3DDFA-v2 exp+id"] =  idx; idx+=1
    # order["Deep3DFace exp"] =  idx; idx+=1
    order["Deep3DFace exp+id"] =  idx; idx+=1

    leave_out_not_metioned_runs = True
    # leave_out_not_metioned_runs = False

    # for run_key, run_dir in run_names.items():
    for run in runs:
        i = len(table)
        cfg = DictConfig(json.loads(run.json_config))
        entry_dict = {}
        # entry_dict["run_id"] = cfg.inout.value.name
        id = cfg.inout.value.random_id
        # print(f"random_id: {id}")
        # print(f"name: {cfg.inout.value.name}" )
        # print(f"link: {run.url}")
        # print("--------------------------")

        if id in final_model_nicknames.keys():
            entry_dict["run_name"] = final_model_nicknames[id]
        elif not leave_out_not_metioned_runs:
            entry_dict["run_name"] = run.name
        else:
            continue
        #
        if 'deca_cfg' in cfg.model.value.keys():
            print(f"run_name: {entry_dict['run_name']}: ")
            print(cfg.model.value.deca_cfg.inout.full_run_dir)

        for ki, key in enumerate(keys_to_recover):
            entry_dict[metrics_columns[ki]] = run.summary_metrics[key]

        table = table.append(entry_dict, ignore_index=True)

    if leave_out_not_metioned_runs:
        new_order = []
        for ind in range(len(table)):
            new_order.append(order[table.iloc[ind]["run_name"]])
        table.rename(index=dict(zip(list(range(len(table))), new_order)), inplace=True)


    #     # table = table.reindex(new_order)
        table = table.sort_index(ascending=True)
    #     table_ordered = pandas.DataFrame(columns=["run_name", *metrics_columns])
    #     for i in range(new_order):
    #         table_ordered = table_ordered.append(table.iloc[ind], ignore_index=True)
    #     table = table_ordered
    print(table)
    table.to_csv("affectnet_test_metrics_.csv")



    with open("affectnet_test_metrics_.tex", "w") as f:
        table.to_latex(f,
                       index_names=False,
                       # float_format="{:0.2f}",
                       float_format=format,
                       # column_formatstr=len(table.colums) * "l" + "r",
                       )
        # f.write(
        #         )

def format(num):
    return f"{num:.02f}"

if __name__ == '__main__':
    main()
