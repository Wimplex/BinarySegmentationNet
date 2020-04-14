import math

from functions import binarize

import torch
import torch.nn as nn
import torch.nn.functional as F


# Функция бинаризации активации гиперболического тангенса
class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, x):
        output = self.hardtanh(x)
        output = binarize(output)
        return output


# Бинаризованный полносвязный слой (weights + bias)
class BinaryLinear(nn.Linear):
    def forward(self, x):
        binary_weight = binarize(self.weight)
        if self.bias is None:
            return F.linear(x, binary_weight)
        else:
            return F.linear(x, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


# Бинаризованный сверточный слой (weights + bias)
class BinaryConv2d(nn.Conv2d):
    def forward(self, x):
        bw = binarize(self.weight)
        return F.conv2d(x, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


# Бинаризованный слой транспонированной свертки (развертки) (weights, bias)
class BinaryConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, x):
        bw = binarize(self.weight)
        return F.conv_transpose2d(x, bw, self.bias, self.stride, self.padding,
                                  self.output_padding, self.groups, self.dilation)

    def reset_parameters(self):
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


# (эксперементально) Слой бинаризации входа.
# Принимает на вход изображение и преобразует его в бинарный вид с заданными порогами.
# BinarizedInput2d([bs, l, w, h], [p]) -> [bs, l * p, w, h]
class BinarizedInput2d(nn.Module):
    def __init__(self, percentiles):
        super(BinarizedInput2d, self).__init__()
        self.percentiles = percentiles

    def forward(self, x):
        if len(x.shape) != 4:
            raise RuntimeError("BinarizedInput layer: Input should have 3 dimensions")

        batch_size, channels, width, height = x.shape
        new_x = torch.empty([batch_size, channels * len(self.percentiles), width, height]).cuda()

        i = 0
        for channel in range(channels):
            for perc in self.percentiles:
                new_x[:, i, :, :] = (x[:, channel, :, :] > perc)
                i += 1
        return new_x
