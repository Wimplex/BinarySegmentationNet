import os

from config import Config

import numpy as np
from PIL import Image
import torch


# Класс-датасет
class PascalVOC_Segmentation(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, train=True):
        super(PascalVOC_Segmentation, self).__init__()

        self.set_dir = dataset_dir
        self.file_names = []
        self.__train = train
        self.__initialize_dataset()
        self.counts = self.__compute_class_probability()

    def __initialize_dataset(self):
        names_file = 'train.txt' if self.__train else 'val.txt'
        with open(os.path.join(self.set_dir, 'ImageSets/Segmentation', names_file)) as file:
            for line in file.readlines():
                self.file_names.append(line.strip())
        np.random.shuffle(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Подгружаем маску и соответствующее ей изображение
        image_path = os.path.join(self.set_dir, 'JPEGImages', self.file_names[idx] + '.jpg')
        mask_path = os.path.join(self.set_dir, 'SegmentationClass', self.file_names[idx] + '.png')

        image = self.__load_image(image_path)
        mask = self.__load_mask(mask_path)

        return torch.FloatTensor(image), torch.LongTensor(mask)

    def __load_image(self, path):
        img = Image.open(path).resize((224, 224))
        img = np.transpose(img, (2, 1, 0))
        img_scaled = np.array(img, dtype=np.float32) / 255.0
        return img_scaled

    def __load_mask(self, path):
        img = Image.open(path).resize((224, 224))
        img = np.array(img)
        img[img == 255] = len(Config.VOC_CLASSES)
        return img

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(Config.NUM_CLASSES))

        for name in self.file_names:
            mask_path = os.path.join(self.set_dir, 'SegmentationClass', name + '.png')

            raw_image = Image.open(mask_path).resize((224, 224))
            img = np.array(raw_image).reshape(224 * 224)
            img[img == 255] = len(Config.VOC_CLASSES)

            for i in range(Config.NUM_CLASSES):
                counts[i] += np.sum(img == i)

        return counts

    def get_class_porbability(self):
        values = np.array(list(self.counts.values()))
        values = values / np.sum(values)
        return torch.Tensor(values)


# Возвращает DataLoader-объект выбранного сегмента датасета
def get_dataset_loader(dataset_dir, batch_size, train=True):
    voc_set = PascalVOC_Segmentation(dataset_dir=dataset_dir, train=train)
    loader = torch.utils.data.DataLoader(voc_set,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4)
    return loader


if __name__ == '__main__':
    ds = PascalVOC_Segmentation(Config.DATA_DIR, train=True)
    # print(ds.get_class_porbability())

    import matplotlib.pyplot as plt
    image, mask = ds[1]
    image.transpose_(0, 2)

    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow(image)

    a = fig.add_subplot(1, 2, 2)
    plt.imshow(mask)

    plt.show()
