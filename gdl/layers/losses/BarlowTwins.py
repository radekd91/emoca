"""
Borrowed and modified from official Pytorch implementation of Barlow Twins (paper):
https://github.com/facebookresearch/barlowtwins/blob/a655214c76c97d0150277b85d16e69328ea52fd9/main.py
"""
import torch
from torch import nn, optim


class BarlowTwins(nn.Module):

    def __init__(self, args, backbone=None):
        super().__init__()
        self.args = args
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity()

        self.bt_loss = BarlowTwinsLoss(self.args)

    def forward(self, y1, y2):
        loss = self.bt_loss(self.backbone(y1), self.backbone(y2))
        return loss


class BarlowTwinsLoss(nn.Module):
    def __init__(self, feature_size=2048, layer_sizes=None, final_reduction='mean_on_diag'):
        super().__init__()
        if layer_sizes is None:
            # layer_sizes = 3*[2048]
            layer_sizes = 3*[8192]

        # # projector
        # if args.use_projector:
        # sizes = [feature_size] + list(map(int, args.projector.split('-')))
        sizes = [feature_size] + layer_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1])) # here the BN layer of the projector is learnable
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        # else:
        #     self.projector = None

        self.bt_loss_headless = BarlowTwinsLossHeadless(sizes[-1], final_reduction=final_reduction)


    def forward(self, y1, y2, batch_size=None, ring_size=None):
        if self.projector is not None:
            z1 = self.projector(y1)
            z2 = self.projector(y2)
        else:
            z1 = y1
            z2 = y2
        loss = self.bt_loss_headless(z1, z2, batch_size=batch_size, ring_size=ring_size)
        return loss


class BarlowTwinsLossHeadless(nn.Module):

    def __init__(self, feature_size, batch_size=None, lambd=0.005, final_reduction='mean_on_diag'):
        super().__init__()
        # normalization layer for the representations z1 and z2
        # the affine=False means there are no learnable weights in the BN layer
        self.bn = nn.BatchNorm1d(feature_size, affine=False)
        self.lambd = lambd
        self.batch_size = batch_size
        if final_reduction not in ["sum", "mean", "mean_on_diag", "mean_off_diag"]:
            raise ValueError(f"Invalid reduction operation for Barlow Twins: '{self.final_reduction}'")
        self.final_reduction = final_reduction

    def forward(self, z1, z2, batch_size=None, ring_size=None):
        assert not (batch_size is not None and self.batch_size is not None)
        if ring_size is not None and ring_size > 1:
            raise NotImplementedError("Barlow Twins with rings are not yet supported.")
        if batch_size is None:
            if self.batch_size is not None:
                batch_size = self.batch_size
            else:
                print("[WARNING] Batch size for Barlow Twins loss not explicitly set. "
                      "This can make problems in multi-gpu training.")
                batch_size = z1.shape[0]

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(batch_size)

        # sum the cross-correlation matrix between all gpus (if multi-gpu training)
        if torch.distributed.is_initialized():
            torch.distributed.nn.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2)
        off_diag = off_diagonal(c).pow_(2)

        # implementation note:
        # The original implementation uses 'sum' for final reduction (in fact they never did mean). However,
        # if you're using additional losses apart from this one, the 'sum' reduction can significantly change
        # the influence of your loss depending on how many elements does the diagonal matrix have. In those cases,
        # 'mean' should be more appropriate.
        if self.final_reduction == 'sum':
            # the original paper
            on_diag = on_diag.sum()
            off_diag = off_diag.sum()
        elif self.final_reduction == 'mean':
            # mean of the on diag and off diag elements
            # there is much more of off diag elemetns and therefore the mean can add up to disproportionally less
            # than what the original implementation intended
            on_diag = on_diag.mean()
            off_diag = off_diag.mean()
        elif self.final_reduction == 'mean_on_diag':
            # normalized by number of elements on diagonal
            # off diag elements are normalized by number of on diag elements so the proportionality is preserved
            n = on_diag.numel()
            on_diag = on_diag.mean()
            off_diag = off_diag.sum() / n
        elif self.final_reduction == 'mean_off_diag':
            # normalized by number of elements off diagonal
            # on diag elements are normalized by number of off diag elements so the proportionality is preserved
            n = off_diag.numel()
            on_diag = on_diag.sum() / n
            off_diag = off_diag.mean()
        else:
            raise ValueError(f"Invalid reduction operation for Barlow Twins: '{self.final_reduction}'")
        loss = on_diag + self.lambd * off_diag
        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()