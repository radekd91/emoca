import torch


class KeypointScale(torch.nn.Module):

    def __init__(self, scale_x=1., scale_y=1.):
        super().__init__()
        self.scale_x = scale_x
        self.scale_y = scale_y

    def set_scale(self, scale_x=1., scale_y=1.):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def forward(self, points):
        points_ = points.clone()
        points_[:, 0] *= self.scale_x
        points_[:, 1] *= self.scale_y
        return points_