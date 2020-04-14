import argparse

from config import Config
from execute import train, predict
from segnet_binary import SegNet
from dataset import get_dataset_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-c', '--test', action='store_true')
    args = parser.parse_args()

    options = Config.get_options(0)

    if args.train:
        # Загружаем тренировочные данные
        print("Загрузка данных")
        train_loader = get_dataset_loader(Config.DATA_DIR, options['batch_size'], train=True)

        print("Построение модели")
        net = SegNet(input_channels=3, name='segnet0')

        print("Начало тренировки")
        train(net=net, data_loader=train_loader, n_epochs=options['n_epochs'], lr=options['lr'])

    elif args.test:
        # Загружаем тестовые данные
        test_loader = get_dataset_loader(Config.DATA_DIR, options['batch_size'], train=False)
        predict

    else:
        print("Укажите один из двух флагов: --train или --test")
