import os

from bin_modules import *

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


class SegNet_Binary(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SegNet_Binary, self).__init__(*args, **kwargs)

    def forward(self):
        pass

    def train(self, *args, **kwargs):
        pass