import torch
import torch.nn.functional as F
from munch import Munch
from omegaconf import DictConfig
from .BarlowTwins import BarlowTwinsLossHeadless, BarlowTwinsLoss


def cosine_sim_negative(*args, **kwargs):
    return (1. - F.cosine_similarity(*args, **kwargs)).mean()


def metric_from_str(metric, **kwargs):
    if metric == "cosine_similarity":
        return cosine_sim_negative
    elif metric in ["l1", "l1_loss", "mae"]:
        return torch.nn.functional.l1_loss
    elif metric in ["mse", "mse_loss", "l2", "l2_loss"]:
        return torch.nn.functional.mse_loss
    elif metric == "barlow_twins_headless":
        return BarlowTwinsLossHeadless(**kwargs)
    elif metric == "barlow_twins":
        return BarlowTwinsLoss(**kwargs)
    else:
        raise ValueError(f"Invalid metric for deep feature loss: {metric}")


def metric_from_cfg(metric):
    if metric.type == "cosine_similarity":
        return cosine_sim_negative
    elif metric.type in ["l1", "l1_loss", "mae"]:
        return torch.nn.functional.l1_loss
    elif metric.type in ["mse", "mse_loss", "l2", "l2_loss"]:
        return torch.nn.functional.mse_loss
    elif metric.type == "barlow_twins_headless":
        return BarlowTwinsLossHeadless(metric.feature_size)
    elif metric.type == "barlow_twins":
        layer_sizes = metric.layer_sizes if 'layer_sizes' in metric.keys() else None
        return BarlowTwinsLoss(metric.feature_size, layer_sizes)
    else:
        raise ValueError(f"Invalid metric for deep feature loss: {metric}")


def get_metric(metric):
    if isinstance(metric, str):
        return metric_from_str(metric)
    if isinstance(metric, (DictConfig, Munch)):
        return metric_from_cfg(metric)
    if isinstance(metric, dict):
        return metric_from_cfg(Munch(metric))
    raise ValueError(f"invalid type: '{type(metric)}'")