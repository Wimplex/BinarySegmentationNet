


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-c', '--test', action='store_true')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.train:
        # Загружаем тренировочные данные
        train_data = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH), transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        lenet5 = LeNet5Binary(input_shape=[1, 28, 28])
        #lenet5 = LeNet5(input_shape=[1, 28, 28])
        train(lenet5, train_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE)

    elif args.test:
        # Загружаем тестовые данные
        test_data = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH), train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

        # Загружаем обученную модель
        #lenet5 = LeNet5(input_shape=[1, 28, 28])
        #lenet5.load("models/lenet5.pth")
        lenet5_bin = LeNet5Binary(input_shape=[1, 28, 28])
        lenet5_bin.load("models/bin_lenet5.pth")

        # Смотрим результат на тестовых данных
        y_pred, y_true = predict(lenet5_bin, test_loader)
        y_pred = np.argmax(y_pred, axis=1)

        print(y_true[-10:])
        print(y_pred[-10:])

        print(classification_report(y_true, y_pred))

    else:
        print("Укажите один из двух флагов: --train или --test")