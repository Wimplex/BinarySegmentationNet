import os

import bin_modules as bin_nn

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


class SegNet_Binary(nn.Module):
    def __init__(self, in_shape, device, name="segnet"):
        super(SegNet_Binary, self).__init__()

        self.in_shape = in_shape
        self.device = device
        self.name = name

        channels, width, height = in_shape
        init_ch = 32

        self.encoder = nn.Sequential(
            bin_nn.BinarizedConv2d(channels, init_ch, 3),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),

            bin_nn.BinarizedConv2d(init_ch, init_ch * 2, 3),
            nn.BatchNorm2d(init_ch * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            bin_nn.BinarizedConv2d(init_ch * 2, init_ch * 4, 3),
            nn.BatchNorm2d(init_ch * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            bin_nn.BinarizedConv2d(init_ch * 4, init_ch * 6, 3),
            nn.BatchNorm2d(init_ch * 6),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(

        )

    def forward(self):
        pass

    def train(self, *args, **kwargs):
        pass
