import torch
from torch import nn

class NGL(nn.Module):

    def __init__(self):
        super(NGL, self).__init__()

    def forward(self, x, target):
        target = torch.nn.functional.one_hot(target, num_classes=x.size(1))
        x = torch.softmax(x, dim = -1)
        loss = torch.mean(torch.exp(2.4092 - x - x*target) - torch.cos(torch.cos(torch.sin(x))))
        return loss
