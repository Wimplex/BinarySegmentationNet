import os
import argparse

from execute_model import train, predict
from lenet5_binary import LeNet5BinaryConv, Net

import torch
import torchvision
import torchvision.transforms as transforms


BATCH_SIZE = 64
LEARNING_RATE = 2e-3
N_EPOCHS = 50
DATA_PATH = "../../../[Datasets]/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-c', '--test', action='store_true')
    # parser.add_argument('-m', '--model', "Модель для обучения [lenet5_bin, segnet_bin, unet_bin]")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.train:
        # Загружаем тренировочные данные
        train_data = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH), transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        # if args.model == 'lenet5_bin':
        #     lenet5 = LeNet5BinaryConv(input_shape=[1, 28, 28])
        #     # lenet5 = LeNet5(input_shape=[1, 28, 28])
        #     train(lenet5, train_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE)

        net = Net()
        train(net, train_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE)

    elif args.test:
        # Загружаем тестовые данные
        test_data = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH), train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

        # Загружаем обученную модель
        # lenet5 = LeNet5(input_shape=[1, 28, 28])
        # lenet5.load("models/lenet5.pth")
        lenet5_bin = LeNet5BinaryConv(input_shape=[1, 28, 28])
        lenet5_bin.load("models/bin_lenet5.pth")

    else:
        print("Укажите один из двух флагов: --train или --test")
