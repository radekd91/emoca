import os, sys
# from fastai.torch_core import Module
# from fastai.layers import ConvLayer, BatchNorm
import torch
from torch.optim import Adam
from torch.nn import BatchNorm1d, BatchNorm2d, Conv1d, Conv2d, ReLU, Module, Sequential, Linear, Tanh
import torch.nn.functional as F
from datasets.MeshDataset import EmoSpeechDataModule

from pytorch_lightning.core import LightningModule

#@delegates()


class SpeechEncoder(Module):

    def __init__(self, speech_encoding_dim, condition_speech_features, speech_encoder_size_factor, num_training_subjects,
                 num_feats, size_factor=1):
        super().__init__()

        # self.bn = BatchNorm()

        # num_feats = 5
        # size_factor = 1
        # speech_encoding_dim = 1

        self._speech_encoding_dim = speech_encoding_dim
        self._condition_speech_features = condition_speech_features
        self._speech_encoder_size_factor = speech_encoder_size_factor
        self._num_training_styles = num_training_subjects

        self._with_emotions = 9
        self._with_identities = 4

        # self.bn = BatchNorm1d(eps=1e-5, momentum=0.9, num_features=num_feats)
        self.bn = BatchNorm2d(eps=1e-5, momentum=0.9, num_features=num_feats)
        # self.conv1_time = Conv1d(in_channels=1e-5, kernel_size=5, num_feat)
        self.conv1_time = Conv2d(kernel_size=(3, 1), stride=(2,1), in_channels=1, out_channels=32*size_factor)
        self.conv2_time = Conv2d(kernel_size=(3, 1), stride=(2,1), in_channels=self.conv1_time.out_channels, out_channels=32*size_factor)
        self.conv3_time = Conv2d(kernel_size=(3, 1), stride=(2,1), in_channels=self.conv2_time.out_channels, out_channels=64*size_factor)

        self.conv4_time = Conv2d(kernel_size=(3, 1), stride=(2,1), in_channels=self.conv3_time.out_channels, out_channels=64*size_factor)
        # output should be 1x1x(64*size_factor)

        self.fc1 = Linear(in_features=self.conv4_time.out_channels + self._with_emotions + self._with_identities, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=speech_encoding_dim)



    def forward(self, x, identity=None, emotion=None):
        x = F.relu(self.bn(x))

        if self._condition_speech_features:
            if self._with_identities:
                assert identity is not None
                x = torch.cat([x, identity], dim=2)

            if self._with_emotions:
                assert emotion is not None
                x = torch.cat([x, emotion], dim=2)

        x = F.relu(self.conv1_time(x))
        x = F.relu(self.conv2_time(x))
        x = F.relu(self.conv3_time(x))
        x = F.relu(self.conv4_time(x))

        if self._with_identities:
            assert identity is not None
            x = torch.cat([x, identity], dim=2)

        if self._with_emotions:
            assert emotion is not None
            x = torch.cat([x, emotion], dim=2)

        x = x.view(-1, x.shape(1) * x.shape(2) * x.shape(3))

        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class ExpressionDecoder(torch.nn.Module):

    def __init__(self, expression_basis, trainable=False):
        super().__init__()
        self.decoder = torch.nn.Linear(expression_basis.shape[2], expression_basis.shape[1]*expression_basis.shape[0], bias = False)
        self.decoder.requires_grad_(trainable)
        self.decoder.weight[...] = expression_basis.view(-1, expression_basis.shape[2])


    def forward(self, x):
        return self.decoder(x)



class Voca(LightningModule):


    def __init__(self, dm : EmoSpeechDataModule,
                 expression_space,
                 condition_speech_features=True,
                 speech_encoder_size_factor=32):
        super().__init__()
        self.consecutive_frames = dm.consecutive_frames

        self.temporal_window = dm.temporal_window

        self.speech_encoder = SpeechEncoder(self.temporal_window,condition_speech_features, dm.num_audio_samples_per_scan, dm.num_training_subjects, speech_encoder_size_factor)
        self.expression_decoder = ExpressionDecoder(expression_space)

        self._position_loss_weight = 1
        self._velocity_loss_weigt = 0.


    def forward(self, x, template, identity=None, emotion=None):
        print("YOYOYO")
        x_reshaped = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        encoded = self.speech_encoder.forward(x_reshaped, identity, emotion)
        decoded = self.expression_decoder(encoded)
        decoded = decoded.view(-1, self.consecutive_frames, decoded.shape[2], decoded.shape[3])
        return decoded + template


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        ds = batch["deep_speech"]
        gt_verts = batch["vertices"]
        template_verts = batch["template_vertices"]
        emo = None

        if self.speech_encoder._with_emotions:
            emo = batch["emotion"]
        id = None
        if self.speech_encoder._with_identities:
            id = batch["identity"]

        predicted_verts = self.forward(ds, template_verts, id, emo)


    def validation_step(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def _position_loss(self, gt_verts, predicted_verts):
        return F.mse_loss(gt_verts, predicted_verts)

    def _velocity_loss(self, gt_verts, predicted_verts):
        id1 = torch.arange(gt_verts.shape[1]-1, dtype=torch.int32)
        id2 = torch.arange(1, gt_verts.shape[1], dtype=torch.int32)

        gt_velocities = gt_verts[:, id2, ...] - gt_verts[:, id1, ...]
        predicted_velocities = predicted_verts[:, id2, ...] - predicted_verts[:, id1, ...]

        return F.mse_loss(gt_velocities, predicted_velocities)

    def _compute_loss(self, gt_verts, predicted_verts):
        loss = 0
        if self._position_loss_weight > 0:
            loss += self.position_loss_weight * self._position_loss(gt_verts, predicted_verts)

        if self._velocity_loss_weigt > 0:
            loss += self.velocity_loss_weight * self._velocity_loss(gt_verts, predicted_verts)

    def _create_pairs(self):

        pass



