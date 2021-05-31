import os, sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from datasets.AffectNetDataModule import AffectNetDataModule, AffectNetExpressions
from applications.DECA.train_expdeca import prepare_data, create_logger
from applications.DECA.train_deca_modular import get_checkpoint, locate_checkpoint

from models.DECA import DECA, ExpDECA, DecaModule
from models.EmoNetModule import EmoNetModule
from utils.other import class_from_str
import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from applications.DECA.interactive_deca_decoder import hack_paths
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import wandb
from emotion_disentanglement import exchange_and_decode, exchange_codes, decode, \
    load_deca, test, load_affectnet, plot_comparison
from tqdm import auto
from utils.lightning_logging import  _log_wandb_image
from pytorch_lightning.loggers import WandbLogger
from train_expdeca import get_checkpoint_with_kwargs

project_name = 'EmotionalDECA'



def eliminate_unwanted_visualization(d, inputs=False, landmarks=False):
    keys_to_remove = []
    for key in d.keys():
        if 'normals' in key:
            keys_to_remove += [key]
        if 'albedo' in key:
            keys_to_remove += [key]

        if inputs and 'inputs' in key:
            keys_to_remove += [key]

        if landmarks and 'landmarks' in key:
            keys_to_remove += [key]


    for key in keys_to_remove:
        del d[key]
    return key


def validation_set_pass(cfg,
                        deca,
                        dm,
                        visualization_freq,
                        num_epochs,
                        codes_to_exchange,
                        # stage, prefix,
                        # dm=None,
                        # logger=None,
                        # data_preparation_function=None,
                        # checkpoint=None, checkpoint_kwargs=None
    ):


    logger = None

    if logger is not None:
        logger.finalize("")

    # checkpoint_kwargs = checkpoint_kwargs or {}
    # deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)

    dm.val_batch_size = 2
    dm.prepare_data()
    dm.setup()

    deca.cuda()
    deca.eval()
    deca_name = deca.inout_params.name

    exchange_acronym = ''
    for code in codes_to_exchange:
        exchange_acronym += code[0]

    result_dir = Path(deca.inout_params.full_run_dir).parent / "tests" / "AffectNetDisentangle" / exchange_acronym
    result_dir.mkdir(exist_ok=True, parents=True)

    visualization_dir = result_dir / "visualizations"
    visualization_dir1 = visualization_dir / "1"
    visualization_dir2 = visualization_dir / "2"
    visualization_dir12 = visualization_dir / "12"
    visualization_dir21 = visualization_dir / "21"

    import pytorch_lightning as pl
    # pl.utilities.seed.seed_everything(0, workers=True)

    losses_all_original = {}
    losses_all_exchanged = {}

    # visualization_freq = 50

    if logger is None:
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        cfg.inout.time = time
        N = len(time)
        cfg.inout.random_id = str(hash(time))

        if hasattr(cfg.inout, 'time') and hasattr(cfg.inout, 'random_id'):
            version = cfg.inout.time + "_" + cfg.inout.random_id
        elif hasattr(cfg.inout, 'time'):
            version = cfg.inout.time # + "_" + cfg.inout.name
        else:
            version = "AffNet"[:N] # unfortunately time doesn't cut it if two jobs happen to start at the same time

        exp_conf = DictConfig({})
        exp_conf.model = cfg
        exp_conf.result_dir = str(result_dir)
        exp_conf.num_epochs = num_epochs
        exp_conf.visualization_freq = visualization_freq
        exp_conf.codes_to_exchange = codes_to_exchange

        logger = create_logger(
                    cfg.learning.logger_type,
                    name=cfg.inout.name,
                    project_name=project_name,
                    version=version,
                    save_dir=result_dir,
                    config=OmegaConf.to_container(exp_conf))

    deca.logger = logger

    step = 0

    for ie in range(num_epochs):

        pl.utilities.seed.seed_everything(ie)
        dl = DataLoader(dm.validation_set, shuffle=True, num_workers=dm.num_workers,
                              batch_size=dm.val_batch_size, drop_last=True)

        for bi, batch in enumerate(auto.tqdm(dl)):
            # if bi == 50:
            #     break
            visualize = step % visualization_freq == 0

            batch1 = {}
            batch2 = {}
            batch_size = None

            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()
                    if batch_size is None:
                        batch_size = batch[key].shape[0]
                # else:
                #     raise ValueError(f"Wtf is this {type(batch[key])}")
                batch1[key] = batch[key][:batch_size//2]
                batch2[key] = batch[key][batch_size//2:]


            with torch.no_grad():
                values_img1, visdict1, losses1 = test(deca, batch1, visualize=visualize, #stage="1",
                                                      output_vis_path=str(visualization_dir1))
                values_img2, visdict2, losses2 = test(deca, batch2, visualize=visualize, #stage="2",
                                                      output_vis_path=str(visualization_dir2))

                vals21_de, vals12_de = exchange_and_decode(deca, values_img1, values_img2,
                                                           codes_to_exchange
                                                          , batch1,
                                                           batch2, visualize=visualize,
                                                           output_vis_path12=str(visualization_dir12),
                                                           output_vis_path21=str(visualization_dir21)
                                                           )
                values_21, vis_dict_21, losses_21 = vals21_de
                values_12, vis_dict_12, losses_12 = vals12_de

            # correct for the annoying weight
            if 'emonet_weight' in cfg.model.keys():
                for key in losses1.keys():
                    if '_emonet_' in key:
                        losses1[key] /= cfg.model.emonet_weight
                        losses2[key] /= cfg.model.emonet_weight
                for key in losses_21.keys():
                    if '_emonet_' in key:
                        losses_21[key] /= cfg.model.emonet_weight
                        losses_12[key] /= cfg.model.emonet_weight

            # if step % visualization_freq == 0:
            #     results1 = [[values_img1, visdict1, losses1], ]
            #     results1 += [[values_img2, visdict2, losses2], ]
            #     results1 += [vals12_de]
            #
            #     results2 = [[values_img2, visdict2, losses2], ]
            #     results2 += [[values_img1, visdict1, losses1], ]
            #     results2 += [vals21_de]
            #
            #     names = [f"Input {bi}", f"Target {bi}", f"Exchange {bi}"]
            #     # names = [f"Input {bi}", f"Target {bi}", f"Exchange {bi}"]
            #
            #     fig12 = plot_comparison(names, results1, batch1, batch2, deca_name)
            #     fig21 = plot_comparison(names, results2, batch2, batch1, deca_name)
            #     fig12.savefig(result_dir / f"{bi:04d}_{bi - 1:04d}.png", bbox_inches='tight')  # , dpi = 300)
            #     fig21.savefig(result_dir / f"{bi - 1:04d}_{bi:04d}.png", bbox_inches='tight')  # , dpi = 300)

            if len(losses_all_original) == 0:
                for key in losses1.keys():
                    losses_all_original[key] = []

                for key in losses_21.keys():
                    losses_all_exchanged[key] = []

            for key in losses1.keys():
                losses_all_original[key] += [losses1[key].detach().cpu().item()]
                losses_all_original[key] += [losses2[key].detach().cpu().item()]
            for key in losses_21.keys():
                losses_all_exchanged[key] += [losses_21[key].detach().cpu().item()]
            for key in losses_12.keys():
                losses_all_exchanged[key] += [losses_12[key].detach().cpu().item()]

            if logger is not None:
                logger.log_metrics({ "1_" + key: value for key,value in losses1.items()}, step=step)
                logger.log_metrics({ "2_" + key: value for key,value in losses2.items()}, step=step)
                logger.log_metrics({ "12_" + key: value for key,value in losses_12.items()}, step=step)
                logger.log_metrics({ "21_" + key: value for key,value in losses_21.items()}, step=step)

                if visualize:
                    eliminate_unwanted_visualization(visdict1)
                    eliminate_unwanted_visualization(visdict2)
                    eliminate_unwanted_visualization(vis_dict_12, inputs=True, landmarks=True)
                    eliminate_unwanted_visualization(vis_dict_21, inputs=True, landmarks=True)

                    # for key in visdict1.keys():
                    #     savepath = Path(
                    #         f'{result_dir}/{key}/{step:04d}.png')
                    #     savepath.parent.mkdir(exist_ok=True, parents=True)
                    #
                    #     if isinstance(logger, WandbLogger):
                    #         im2log = _log_wandb_image(savepath, visdict1, caption)

                    logger.log_metrics({"1_" + key: value for key,value in visdict1.items()}, step=step)
                    logger.log_metrics({"2_" + key: value for key,value in visdict2.items()}, step=step)
                    logger.log_metrics({"12_" + key: value for key,value in vis_dict_12.items()}, step=step)
                    logger.log_metrics({"21_" + key: value for key,value in vis_dict_21.items()}, step=step)

            step += 1


    final_losses_orig = {}
    final_losses_exchanged = {}
    for key in losses_all_original.keys():
        final_losses_orig["final_orig_" + key] = np.array(losses_all_original[key]).mean()
        final_losses_exchanged["final_exchanged_" + key] = np.array(losses_all_exchanged[key]).mean()

    if logger is not None:
        logger.log_metrics(final_losses_orig)
        logger.log_metrics(final_losses_exchanged)

    # names = [AffectNetExpressions(i).name for i in range(9)]

    print("Peace")


def load_configs(run_path):
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    return conf




def main():

    print(sys.argv)

    if len(sys.argv) <= 2:
        path_to_models = "/is/cluster/work/rdanecek/emoca/finetune_deca/"
        ## relative_to_path = None
        ## replace_root_path = None

        # path_to_models = "/ps/scratch/rdanecek/emoca/finetune_deca/"
        relative_to_path = '/ps/scratch/'
        replace_root_path = '/home/rdanecek/Workspace/mount/scratch/'

        ## ExpDECA on AffectNet, emotion loss
        # run_name = "2021_04_20_18-36-33_ExpDECA_Affec_para_Jaw_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"

        # DECA on DECA dataset, no ring, emotion loss
        # run_name = "2021_04_23_17-06-29_ExpDECA_DecaD_NoRing_EmoLossB_F2VAEw-0.00150_DeSegrend_DwC_early"

        # ExpDECA on AffectNet, Expression ring without geometric losses (exchange punished for emotion mismatched only)
        # run_name += '2021_05_02_12-43-06_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_early'
        # run_name += '2021_05_02_12-42-01_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_DwC_early'
        # run_name += '2021_05_02_12-37-20_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_expression_CoNone_DeNone_DwC_early'
        # run_name += '2021_05_02_12-36-00_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_DwC_early'
        # run_name += '2021_05_02_12-35-44_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_early'
        # run_name += '2021_05_02_12-34-47_ExpDECA_Affec_para_Jaw_EmoLossB_F2VAEw-0.00150_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_early'

        # ExpDECA on AffectNet, Expression ring with geometric losses
        # run_name = '2021_05_07_20-48-30_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_DwC_early'
        # run_name = '2021_05_07_20-46-09_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exgt_va_CoNone_DeNone_early'
        # run_name = '2021_05_07_20-45-33_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_DwC_early'
        # run_name = '2021_05_07_20-36-43_ExpDECA_Affec_para_Jaw_EmoB_F2VAE_GeEx_DeSegrend_BlackB_Exemonet_feature_CoNone_DeNone_early'

        run_name = '2021_05_24_12-22-17_ExpDECA_Affec_para_Jaw_NoRing_DeSegrend_Exnone_MLP_0.1_DwC_early'

        run_path = Path(path_to_models) / run_name
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)

        project = "/home/rdanecek/Workspace/mount/project/"
        dm = load_affectnet(project=project)

        codes_to_exchange = ['detailcode', 'expcode', 'jawpose']

    else: # > 1
        cfg_path = sys.argv[1]
        relative_to_path = None
        replace_root_path = None

        codes_to_exchange = sys.argv[2]
        codes_to_exchange = sorted(codes_to_exchange.split(','))

        dm = load_affectnet()

        with open(Path(cfg_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)

    if len(sys.argv) > 3:
        num_epochs = int(sys.argv[3])
    else:
        num_epochs = 1

    if len(sys.argv) > 4:
        visualization_freq = int(sys.argv[4])
    else:
        visualization_freq = 50

    stage = 'detail'


    exchange_acronym = ''
    for code in codes_to_exchange:
        exchange_acronym += code[0]

    conf["coarse"].inout.name = "an_tangle_" + exchange_acronym + '_' + conf["coarse"].inout.name
    conf["detail"].inout.name = "an_tangle_" + exchange_acronym + '_' + conf["detail"].inout.name

    deca = load_deca(conf, stage, 'best', relative_to_path, replace_root_path)
    deca.cuda()
    deca.eval()

    stages_prefix = ""


    # checkpoint_mode = 'best' # resuming in the same stage, we want to pick up where we left of
    # checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(conf[stage],
    #                                                            checkpoint_mode=checkpoint_mode,
    #                                                            prefix=stages_prefix
    #                                                            # relative_to=relative_to_path,
    #                                                            # replace_root=replace_root_path
    #                                                            )




    validation_set_pass(conf[stage],
                        deca,
                        dm,
                        visualization_freq,
                        num_epochs,
                        codes_to_exchange
                        # data_preparation_function=prepare_data,
                        # checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs
                        )


if __name__ == '__main__':
    main()
