from bin_modules import BinaryConv2d, BinaryConvTranspose2d, BinaryTanh

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegnetDownsampleUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegnetDownsampleUnit, self).__init__()
        self.conv = BinaryConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        # x = BinaryTanh(x)
        x = F.relu(x)
        return x


class SegnetUpsampleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super(SegnetUpsampleUnit, self).__init__()
        self.conv_tr = BinaryConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None

    def forward(self, x):
        x = self.conv_tr(x)

        if self.bn is not None:
            x = self.bn(x)

        # x = BinaryTanh(x)
        x = F.relu(x)
        return x


class SegNet(nn.Module):
    def __init__(self, input_channels, name):
        super(SegNet, self).__init__()

        channels = input_channels
        self.name = name
        self.device = 'cuda:0'

        exp_fact = 2
        init_layers = 16

        self.ds_00 = SegnetDownsampleUnit(channels, init_layers)
        self.ds_01 = SegnetDownsampleUnit(init_layers, init_layers)

        self.ds_10 = SegnetDownsampleUnit(init_layers, init_layers * exp_fact)
        self.ds_11 = SegnetDownsampleUnit(init_layers * exp_fact, init_layers * exp_fact)

        self.ds_20 = SegnetDownsampleUnit(init_layers * exp_fact, init_layers * exp_fact * 2)
        self.ds_21 = SegnetDownsampleUnit(init_layers * exp_fact * 2, init_layers * exp_fact * 2)
        self.ds_22 = SegnetDownsampleUnit(init_layers * exp_fact * 2, init_layers * exp_fact * 2)

        self.ds_30 = SegnetDownsampleUnit(init_layers * exp_fact * 2, init_layers * exp_fact * 4)
        self.ds_31 = SegnetDownsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)
        self.ds_32 = SegnetDownsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)

        self.ds_40 = SegnetDownsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)
        self.ds_41 = SegnetDownsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)
        self.ds_42 = SegnetDownsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)

        self.us_42 = SegnetUpsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)
        self.us_41 = SegnetUpsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)
        self.us_40 = SegnetUpsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)

        self.us_32 = SegnetUpsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)
        self.us_31 = SegnetUpsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 4)
        self.us_30 = SegnetUpsampleUnit(init_layers * exp_fact * 4, init_layers * exp_fact * 2)

        self.us_22 = SegnetUpsampleUnit(init_layers * exp_fact * 2, init_layers * exp_fact * 2)
        self.us_21 = SegnetUpsampleUnit(init_layers * exp_fact * 2, init_layers * exp_fact * 2)
        self.us_20 = SegnetUpsampleUnit(init_layers * exp_fact * 2, init_layers * exp_fact)

        self.us_11 = SegnetUpsampleUnit(init_layers * exp_fact, init_layers * exp_fact)
        self.us_10 = SegnetUpsampleUnit(init_layers * exp_fact, init_layers)

        self.us_01 = SegnetUpsampleUnit(init_layers, init_layers)
        self.us_00 = SegnetUpsampleUnit(init_layers, channels, use_bn=False)

    def forward(self, x):
        # Энкодер

        dim_0 = x.shape

        x = self.ds_00(x)
        x = self.ds_01(x)
        x, indices_0 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        dim_1 = x.shape

        x = self.ds_10(x)
        x = self.ds_11(x)
        x, indices_1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        dim_2 = x.shape

        x = self.ds_20(x)
        x = self.ds_21(x)
        x = self.ds_22(x)
        x, indices_2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        dim_3 = x.shape

        x = self.ds_30(x)
        x = self.ds_31(x)
        x = self.ds_32(x)
        x, indices_3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        dim_4 = x.shape

        x = self.ds_40(x)
        x = self.ds_41(x)
        x = self.ds_42(x)
        x, indices_4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        dim_d = x.shape

        # Декодер
        x = F.max_unpool2d(x, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x = self.us_42(x)
        x = self.us_41(x)
        x = self.us_40(x)
        dim_d4 = x.shape

        x = F.max_unpool2d(x, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x = self.us_32(x)
        x = self.us_31(x)
        x = self.us_30(x)
        dim_d3 = x.shape

        x = F.max_unpool2d(x, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x = self.us_22(x)
        x = self.us_21(x)
        x = self.us_20(x)
        dim_d2 = x.shape

        x = F.max_unpool2d(x, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x = self.us_11(x)
        x = self.us_10(x)
        dim_d1 = x.shape

        x = F.max_unpool2d(x, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x = self.us_01(x)
        x = self.us_00(x)
        dim_d0 = x.shape

        x_softmax = F.softmax(x, dim=1)
        return x, x_softmax
