import wandb
import pandas
from pathlib import Path
import json
from omegaconf import DictConfig, OmegaConf

def main():
    api = wandb.Api()
    root = Path("/is/cluster/work/rdanecek/emoca/emodeca/")
    # create a pandas table with the following colums: run_name, run_folder, score
    metrics_pretty_names = ["Valence PCC", "Valence CCC", "Valence RMSE", "Valence SAGR",
                            "Arousal PCC", "Arousal CCC", "Arousal RMSE", "Arousal SAGR",
                            "Expression Accuracy"]
    metrics_columns = ["v_pcc", "v_ccc", "v_rmse", "v_sagr", "a_pcc", "a_ccc", "a_rmse", "a_sagr",]

    table = pandas.DataFrame(columns=["run_name", *metrics_columns ])

    prefix = "test_metric_"
    dl = "_epoch/dataloader_idx_1"
    # dl = "_epoch"


    keys_to_recover = [ prefix + key + dl for key in metrics_columns]

    runs = api.runs("rdanecek/EmoDECA_Afew-VA")

    final_model_nicknames = {}
    final_model_nicknames["-1161116375494981412"] = "MGCNet exp+id"
    final_model_nicknames["-8046457176211889687"] = "ExpNet exp+id" #todo
    # final_model_nicknames["1843147884226165688"] = "ExpNet 2" #todo
    final_model_nicknames["7268323253792821996"] = "Deep3DFace exp+id" # Deep3DFace
    final_model_nicknames["6949616983506395245"] = "EMOCA-ResNet 1"
    final_model_nicknames["-4341500954807723397"] = "DECA detail"
    final_model_nicknames["-883171165053951196"] = "EMOCA-ResNet L1"
    final_model_nicknames["-8771568993008847876"] = "EMOCA-ResNet 5"
    final_model_nicknames["-5001598437032731747"] = "EMOCA-ResNet cos"
    final_model_nicknames["-4648497253982293620"] = "EMOCA-SWIN 5"
    final_model_nicknames["-4988270973026329720"] = "EMOCA-EmoNet 1"
    final_model_nicknames["-1357828887451007047"] = "3DDFA-v2 exp+id"
    final_model_nicknames["7382187467192733770"] = "DECA"
    final_model_nicknames["68940662102047876"] = "EMOCA-SWIN 1 DecaD"
    final_model_nicknames["2775975653132874343"] =  "EMOCA-ResNet 1 DecaD"
    final_model_nicknames["-8396122457639155494"] = "EMOCA-SWIN 1"
    # final_model_nicknames["4002177680479129605"] = "EMOCA-ResNet 1 v2"
    # final_model_nicknames["-6723661694728668670"] = "EMOCA-SWIN 0.5"
    # final_model_nicknames["7591881836062511055"] = "EMOCA-ResNet 1 v3"
    # final_model_nicknames["-5108247086344313851"] = "SWIN-B"
    final_model_nicknames["-2551392040533364774"] = "ResNet 50"
    # final_model_nicknames["-4363274065263711458"] = "ResNet 50"
    final_model_nicknames["-675529272828243504"] = "SWIN-T"
    final_model_nicknames["5239516111744668401"] = "SWIN-B"
    # final_model_nicknames["-1021932498572718565"] = "SWIN-T"
    final_model_nicknames["-3698006624804482910"] = "VGG19 BN"
    final_model_nicknames["-963597621400333018"] = "EmoNet Original"
    # final_model_nicknames["-4513972209100694206"] = "VGG19BN"
    # final_model_nicknames["8967464683573230563"] = "SWIN-B"
    # final_model_nicknames["4363551435501292966"] = "EmoCnn_vgg19_bn_shake_Aug_early"
    # final_model_nicknames["5784706256688858056"] = "EmoCnn_resnet50_shake_Aug_early"
    # final_model_nicknames["-2897877075877919183"] = "EmoSwin_swin_tiny_patch4_window7_224_shake_Aug_early"
    # final_model_nicknames["5063894700880742144"] = "SWIN-B"
    # final_model_nicknames["5824575168153882277"] = "SWIN-T"
    # final_model_nicknames["1650355822631362210"] = "VGG19"
    # final_model_nicknames["-1078162224083866132"] = "Resnet"

    order = {}
    idx = 0
    order["ResNet 50"] = idx; idx+=1
    order["VGG19 BN"] =  idx; idx+=1
    order["SWIN-B"] =  idx; idx+=1
    order["SWIN-T"] =  idx; idx+=1
    order["EmoNet Original"] =  idx; idx+=1
    # order["EmoNet Ours"] =  idx; idx+=1
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
    # order["EMOCA-ResNet 1"] =  idx; idx+=1
    order["EMOCA-ResNet 1"] =  idx; idx+=1
    order["EMOCA-ResNet 1 DecaD"] =  idx; idx+=1
    order["EMOCA-ResNet cos"] =  idx; idx+=1
    order["EMOCA-ResNet L1"] =  idx; idx+=1
    order["EMOCA-EmoNet 1"] =  idx; idx+=1
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

        for ki, key in enumerate(keys_to_recover):
            entry_dict[metrics_columns[ki]] = run.summary_metrics[key]

        # if 'deca_cfg' in cfg.model.value.keys():
        #     print(f"deca run name: {entry_dict['run_name']}: ")
        #     print(cfg.model.value.deca_cfg.inout.full_run_dir)
        # # print(f"run_name: {entry_dict['run_name']}: ")
        # # print(cfg.inout.value.full_run_dir)

        table = table.append(entry_dict, ignore_index=True)

    if leave_out_not_metioned_runs:
        new_order = []
        for ind in range(len(table)):
            new_order.append(order[table.iloc[ind]["run_name"]])
        table.rename(index=dict(zip(list(range(len(table))), new_order)), inplace=True)
        table = table.sort_index(ascending=True)

    # print(table)
    # table.to_csv("affewva_test_metrics_.csv")
    #
    # with open("affewva_test_metrics_.tex", "w") as f:
    #     table.to_latex(f,
    #                    index_names=False,
    #                    # float_format="{:0.2f}",
    #                    float_format=format,
    #                    # column_formatstr=len(table.colums) * "l" + "r",
    #                    )


def format(num):
    return f"{num:.02f}"

if __name__ == '__main__':
    main()
