import torch


class NormalizeGeometricData(object):

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')
        self.mean = torch.as_tensor(self.mean, dtype=data.x.dtype, device=data.x.device)
        self.std = torch.as_tensor(self.std, dtype=data.x.dtype, device=data.x.device)
        data.x = (data.x - self.mean)/self.std
        data.y = (data.y - self.mean)/self.std
        return data


