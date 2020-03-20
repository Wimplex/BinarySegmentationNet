import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


def plot_samples(samples, y_true, y_pred):
    fig, axes = plt.subplots(2, 5)
    i = 0
    for row in axes:
        for ax in row:
            ax.imshow(np.squeeze(samples[i]), cmap='gray')
            ax.get_yaxis().set_visible(False)
            ax.set_xlabel(f"true: {y_true[i]}, pred: {y_pred[i]}")
            i += 1
    plt.show()


def train(net, data_loader, n_epochs, lr):
    # Переводим модель на выбранное устройство (cuda:0)
    net.to(net.device)

    # Выбираем оптимизатор и функцию потерь
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # Запускаем цикл обучения
    for epoch in range(n_epochs):

        train_loss = 0.0
        processed = 0

        # Проходимся по бачам данных
        for X_batch, y_batch in data_loader:

            # Переводим батч на устройство
            X_batch = X_batch.to(net.device)
            y_batch = y_batch.to(net.device)

            # Рассчитываем выход из сети на батче
            batch_output = net(X_batch)

            # Обнуляем градиент
            optimizer.zero_grad()

            # Рассчитываем значение функции потерь
            loss = loss_func(batch_output, y_batch)

            # Пропускаем градиент в обратном направлении в сеть (backpropagation)
            loss.backward()

            # Шаг оптимизатора: обновление весов в сети на основе значений,
            # полученных при обратном распространении градиента
            optimizer.step()

            # Вывод текущих значений
            train_loss += loss.item() * data_loader.batch_size
            processed += data_loader.batch_size
            print(f"Epoch {epoch}: {round(100.0 * processed / (len(data_loader) * data_loader.batch_size), 2)}%, "
                  f"loss: {round(train_loss / processed, 5)}",
                  end='\r')

        # Вывод значений за эпоху
        print(f"Epoch: {epoch}, train_loss: {round(train_loss / processed, 5)}")

    # Сохранение сети
    torch.save(net.state_dict(), f"models/{net.name}.pth")


def predict(net, data_loader):
    res = np.empty([data_loader.batch_size, 10])
    y_true = np.empty([data_loader.batch_size, 1])
    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(net.device)
        output = net(X_batch).cpu().detach().numpy()

        res = np.append(res, output, axis=0)
        y_true = np.append(y_true, y_batch.reshape(-1, 1))

    X_batch = X_batch.cpu().detach().numpy()
    plot_samples(X_batch[-10:], y_true[:-10], np.argmax(res[-10:], axis=1))
    return res, y_true


def classification_report():
    # Смотрим результат на тестовых данных
    y_pred, y_true = predict(lenet5_bin, test_loader)
    y_pred = np.argmax(y_pred, axis=1)