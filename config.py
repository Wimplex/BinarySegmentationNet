

# Класс-конфигурация запуска модели
class Config:
    __train_options = [
        {'batch_size': 128, 'lr': 5e-3, 'n_epochs': 40, 'momentum': 1e-5, 'eps': 1e-5},
        {'batch_size': 64, 'lr': 2e-3, 'n_epochs': 20, 'momentum': 1e-5, 'eps': 1e-5},
        {'batch_size': 128, 'lr': 2e-3, 'n_epochs': 10, 'momentum': 1e-5, 'eps': 1e-5},  # для быстрого тестирования
        {'batch_size': 64, 'lr': 5e-3, 'n_epochs': 30, 'momentum': 1e-5, 'eps': 1e-5},
    ]

    DATA_DIR = "../../../[Datasets]/VOC2012/"

    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    NUM_CLASSES = len(VOC_CLASSES) + 1

    @staticmethod
    def get_options(idx):
        return Config.__train_options[idx]
