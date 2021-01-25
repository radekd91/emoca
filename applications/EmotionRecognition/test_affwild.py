import os, sys
from pathlib import Path
sys.path += [str(Path(__file__).parent.parent)]

import numpy as np

from datasets.FaceVideoDataset import FaceVideoDataModule, \
    AffectNetExpressions, Expression7, AU8, expr7_to_affect_net
from datasets.EmotionalDataModule import EmotionDataModule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb


class EmotionModule(LightningModule):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.net.to(self.device)

    def forward(self, x):
        return self.net(x)

    def training_step(self, *args, **kwargs):
        pass

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None
        images = batch['image'].to(self.device)
        # pred = self.net(images/255.)
        pred = self.net(images/255.)
        # pred = self.net( (images/255. - 0.5)*2)
        # pred = self.net(images/255.)
        val = pred['valence']
        ar = pred['arousal']
        expr = pred['expression']
        # expr = np.argmax(np.squeeze(expr.detach().cpu().numpy()), axis=1)
        expr = expr.detach().cpu().numpy()

        images_to_log = set()

        if 'expr7' in batch.keys():
            expr_gt = batch['expr7'].detach().cpu().numpy()
            expr_gt_ = []
            for i in range(len(expr_gt)):
                expr_gt_ += [expr7_to_affect_net(int(expr_gt[i])).value]
            expr_gt = np.array(expr_gt_)
            expr = np.argmax(expr, axis=1)
            class_match = (expr_gt == expr).astype(int)
            self.log('expr_match', class_match.mean(), on_step=True, on_epoch=True)
            self.log('expr_gt', expr_gt.mean(), on_step=True, on_epoch=True)
            self.log('expr_pred', expr.mean(), on_step=True, on_epoch=True)

            for i in range(len(class_match)):
                if not class_match[i]:
                    # images_to_log.add(batch["path"][i])
                    images_to_log.add(i)

        if 'va' in batch:
            val_gt = batch['va'][0]
            ar_gt = batch['va'][1]

            val_err = (val - val_gt).abs()
            ar_err = (ar - ar_gt).abs()

            self.log('valence_error', val_err.mean(), on_step=True, on_epoch=True)
            self.log('arousal_error', ar_err.mean(), on_step=True, on_epoch=True)
            self.log('valence_gt', val_gt.mean(), on_step=True, on_epoch=True)
            self.log('arousal_gt', ar_gt.mean(), on_step=True, on_epoch=True)
            self.log('valence', val.mean(), on_step=True, on_epoch=True)
            self.log('arousal', ar.mean(), on_step=True, on_epoch=True)

            for i in range(len(val_err)):
                thresh = 0.4
                if val_err[i] > thresh:
                    # images_to_log.add(batch["path"][i])
                    images_to_log.add(i)

                if ar_err[i] > thresh:
                    # images_to_log.add(batch["path"][i])
                    images_to_log.add(i)

        if 'au8' in batch.keys():
            pass

        to_log = {}
        to_log["fails"] = []
        for i in images_to_log:
            im = images[i, ...].detach().cpu().numpy()
            im = np.transpose(im, [1,2,0])
            # print("Logging image:")
            # print(im.shape)
            path = batch["path"][i]
            # print(path)
            cap = path + "\n"
            if 'va' in batch:
                cap += f" Val_gt = {val_gt[i]}\n"
                cap += f" Ar_gt = {ar_gt[i]}\n"
            cap += f" Val_pred = {val[i]}\n"
            cap += f" Ar_pred = {ar[i]}\n"

            if 'expr7' in batch:
                cap += f" Expr_gt = { AffectNetExpressions(expr_gt[i]).name}\n"
            cap += f" Epr_pred = {AffectNetExpressions(expr[i]).name}\n"

            to_log["fails"] += [wandb.Image(im, caption=cap)]

        self.logger.experiment.log(to_log)

    def test_step_end(self, *args, **kwargs):
        pass

    def test_epoch_end(self, outputs) -> None:
        pass


def main():
    import datetime
    if len(sys.argv) >= 2:
        annotation_list = [sys.argv[1]]
    if len(sys.argv) >= 3:
        index = int(sys.argv[2])

    root = Path("/home/rdanecek/Workspace/mount/scratch/rdanecek/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    # subfolder = 'processed_2020_Dec_21_00-30-03'
    subfolder = 'processed_2021_Jan_19_20-25-10'
    fvdm = FaceVideoDataModule(str(root_path), str(output_path), processed_subfolder=subfolder)
    dm = EmotionDataModule(fvdm)
    dm.prepare_data()
    em = EmotionModule(fvdm._get_emonet())

    # index = 220
    # index = 120
    sequence_name = str(fvdm.video_list[index])
    if annotation_list[0] == 'va' and 'VA_Set' not in sequence_name:
        print("No GT for valence and arousal. Skipping")
        sys.exit(0)
    if annotation_list[0] == 'expr7' and 'Expression_Set' not in sequence_name:
        print("No GT for expressions. Skipping")
        sys.exit(0)

    filter_pattern = sequence_name
    project_name = 'EmoNetOnAffWild2Test'
    name = subfolder + "_" + str(filter_pattern) + "_" + \
           datetime.datetime.now().strftime("%b_%d_%Y_%H-%M-%S") + "_"
    # wandb.init(project_name)
    wandb_logger = WandbLogger(name=name, project=project_name)
    # wandb_logger = None
    # annotation_list = ['va']
    # annotation_list = ['expr7']
    data_loader = dm.test_dataloader(annotation_list, filter_pattern, batch_size=1, num_workers=4)
    trainer = Trainer(gpus=1, logger=wandb_logger)
    trainer.test(em, data_loader)


if __name__ == "__main__":
    main()
