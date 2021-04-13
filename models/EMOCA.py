import torch
import pytorch_lightning as pl

from utils.other import class_from_str
from .DECA import DecaModule
from omegaconf import DictConfig, OmegaConf
from torch.functional import F
from torch.nn import Linear, LeakyReLU, Sequential


class MLP(torch.nn.Module):
    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__()
        self.in_size = config.in_size
        self.out_size = config.out_size
        self.metric = config.metric
        self.hidden_layer_sizes = OmegaConf.to_container(config.hidden_layer_sizes)
        self.loss = class_from_str(config.loss, F)
        self._build_network()

    def _build_network(self):
        layers = []
        layers += [Linear(self.in_size, self.hidden_layer_sizes[0])]
        layers += [LeakyReLU()]
        for i in range(1, len(self.hidden_layer_sizes)):
            layers += [
                Linear(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i])
            ]
            layers += [LeakyReLU()]
        layers += [Linear(self.hidden_layer_sizes[-1], self.out_size)]
        self.model = Sequential(*layers)

    def forward(self, sample):
        # print(sample.keys())
        x = sample[self.metric]
        y = self.model(x)
        return y


class Emoca(pl.LightningModule):

    def __init__(self, config, deca_kwargs, deca_checkpoint):
        super().__init__()
        self.deca = DecaModule.load_from_checkpoint(checkpoint_path=deca_checkpoint, **deca_kwargs)
        self.emonet = self.deca.emonet_loss

        if config.deca2emonet:
            self.deca2emonet = MLP(config.deca2emonet)
        else:
            self.deca2emonet = None

        if config.emonet2deca:
            self.emonet2deca = MLP(config.emonet2deca)
        else:
            self.emonet2deca = None

        # if config.deca2emonet and config.emonet2deca and config.bidirectional:
        #     pass


    def forward(self, batch):
        values = self.encode(batch)
        if self.bidirectional:
            values = self.decode(values)

    def encode(self, batch):
        images = batch["image"]
        values = self.deca.encode(batch, training=False)
        emonet_out = self.emonet()

        deca_z = []
        if self.config.use_identity:
            deca_z += values["shapecode"]
        if self.config.use_expression:
            deca_z += values["expcode"]
        if self.config.use_pose:
            deca_z += values["posecode"]
        if self.config.use_detail:
            deca_z += values["detailcode"]

        if self.deca2emonet is not None:
            deca_z = torch.cat(deca_z, dim=1)
            emonet_z_pred = self.deca2emonet(deca_z)
        else:
            emonet_z_pred = None

        emonet_z = []
        if self.config.use_f1:
            emonet_z += values["emo_feat"]
        if self.config.use_f2:
            emonet_z += values["emo_feat_2"]
        if self.config.use_valence:
            emonet_z += values["valence"]
        if self.config.use_arousal:
            emonet_z += values["arousal"]
        if self.config.use_expression:
            emonet_z += values["expression"]
        emonet_z = torch.cat(emonet_z, dim=1)

        if self.emonet2deca is not None:
            deca_z_pred = self.emonet2deca(emonet_z)
        else:
            deca_z_pred = None

        values = {}
        values["deca_z"] = deca_z
        values["deca_z_pred"] = deca_z_pred
        values["emonet_z"] = emonet_z
        values["emonet_z_pred"] = emonet_z_pred
        return values


    def decode(self, values):
        deca_z = values["deca_z"]
        deca_z_pred = values["deca_z_pred"]
        emonet_z = values["emonet_z"]
        emonet_z_pred = values["emonet_z_pred"]

        if self.config.bidirectional:
            deca_z_cycle = self.emonet2deca(emonet_z_pred)
            emonet_z_cycle = self.deca2emonet(deca_z_pred)
            values["deca_z_cycle"] = deca_z_cycle
            values["emonet_z_cycle"] = emonet_z_cycle

        return values




    def compute_loss(self, values):
        deca_z = values["deca_z"]
        deca_z_pred = values["deca_z_pred"]
        deca_z_cycle = values["deca_z_cycle"]
        emonet_z = values["emonet_z"]
        emonet_z_pred = values["emonet_z_pred"]
        emonet_z_cycle = values["emonet_z_cycle"]

        losses = {}
        if self.config.deca2emonet:
            losses["emonet_rec"] = F.l1_loss(emonet_z, emonet_z_pred)

        if self.config.emonet2deca:
            losses["deca_rec"] = F.l1_loss(deca_z, deca_z_pred)

        if self.config.bidirectional:
            losses["deca_cycle"] = F.l1_loss(deca_z, deca_z_cycle)
            losses["emonet_cycle"] = F.l1_loss(emonet_z, emonet_z_cycle)

        return losses



    def training_step(self, batch, batch_idx):  # , debug=True):
        values = self.encode(batch)
        values = self.decode(values)
        losses_and_metrics = self.compute_loss(values)