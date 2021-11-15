# implementation borrowed from: https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py

import torch
import torch.nn as nn

class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label = None):
        if label is not None:
            index = torch.where(label != -1)[0]
        else:
            index = torch.ones()
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine

