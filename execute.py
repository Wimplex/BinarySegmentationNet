import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# Функция отрисовки примеров после предсказания
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


# Функция обучения модели
def train(net, data_loader, n_epochs, lr):
    net.to(net.device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):

        train_loss = 0.0
        processed = 0
        for X_batch, y_batch in data_loader:
            image_batch = Variable(X_batch).cuda()
            mask_batch = Variable(y_batch).cuda()

            predicted, softmaxed = net(image_batch)

            optimizer.zero_grad()
            loss = loss_func(predicted, mask_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.float()
            processed += data_loader.batch_size
            print(f"Epoch {epoch}: {round(100.0 * processed / (len(data_loader) * data_loader.batch_size), 2)}%,"
                  f"loss: {round(train_loss / processed, 5)}")

        print(f"---> Epoch: {epoch}, train_loss: {round(train_loss / processed, 5)}; Checkpoint...")
        with open(f'models/{net.name}.pth', 'w'):
            torch.save(net.state_dict(), f"{net.name}.pth")


# Функция совершения предсказания при помощи обученной модели
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
