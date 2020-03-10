import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable, Function


def binarize(tensor):
    return tensor.sign()


def quantize(tensor, quant_mode='det', params=None, num_bits=8):
    tensor.clamp_(-2 ** (num_bits - 1), 2 ** (num_bits - 1))
    if quant_mode == 'det':
        tensor = tensor.mul(2 ** (num_bits - 1)).round().div(2 ** (num_bits - 1))
    else:
        tensor = tensor.mul(2 ** (num_bits - 1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2 ** (num_bits - 1))
    return tensor


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

        # self.margin - минимум ширины разделяющей полосы
        self.marging = 1.0

    def loss(self, input, target):
        # Формула для hinge_loss:          l = max(0, 1 - yxw)
        # Формула для grad_hinge_loss: dl/dw = -yx if yxw < 1 else 0

        # input.mul(target) - считает величину отступа
        output = self.marging - input.mul(target)

        # Проверяет, какие элементы имеют отрицательный отступ
        # (предсказание на каких элементах ошибочно)
        output[output.le(0)] = 0

        return output.mean()

    def forward(self, input, target):
        return self.loss(input, target)


class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction, self).__init__()

        # Аналогично, как и в вышеописанном классе
        self.margin = 1.0

    def forward(self, input, target):
        # По аналогии с функцией loss из вышеописанного класса
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0

        # Сохранение активаций для обратного распространения
        # В объекте self.saved_tensors
        self.save_for_backward(input, target)

        # Поэлементное умножение матрицы output на саму себя,
        # суммирование всех элементов получившейся матрицы,
        # деление на количество этих элементов.
        # Таким образом в строке ниже происходит вычисление среднего квадрата элементов
        loss = output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self, grad_output):
        # Получение сохраненных при активации функции forward тензоров
        input, target = self.saved_tensors

        # Вычисление hinge_loss (по аналогии с предыдущей функцией)
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0

        # Вычисление градиента output-а из слоя.
        # Градиент должен быть по форме, как и input.
        # ...
        grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(input.numel()) # получение среднего

        return grad_output, grad_output


class BinarizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinarizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if input.shape[1] != 784:
            input.data = binarize(input.data)

        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.reshape(1, -1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(BinarizeConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if input.shape[1] != 3: