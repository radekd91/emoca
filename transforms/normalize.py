import torch


class NormalizeGeometricData(object):

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def _init(self, dtype, device):
        assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')
        self.mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        self.std = torch.as_tensor(self.std, dtype=dtype, device=device)

    def __call__(self, data):
        self._init(data.x.dtype, data.x.device)
        data.x = (data.x - self.mean)/self.std
        data.y = (data.y - self.mean)/self.std
        return data

    def inv(self, data):
        self._init(data.x.dtype, data.x.device)
        data.x = (data.x*self.std) + self.mean
        data.y = (data.y*self.std) + self.mean
        return data
