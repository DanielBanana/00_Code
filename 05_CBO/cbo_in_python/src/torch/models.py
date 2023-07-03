import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_726x10(nn.Module):
    def __init__(self):
        super(MNIST_726x10, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(1, 3),
            nn.Linear(28 ** 2, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10, affine=False, momentum=None),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.model(x)

class MNIST_726x20(nn.Module):
    def __init__(self):
        super(MNIST_726x20, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(1, 3),
            nn.Linear(28 ** 2, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20, affine=False, momentum=None),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.model(x)

class MNIST_726x10x10(nn.Module):
    def __init__(self):
        super(MNIST_726x10x10, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(1, 3),
            nn.Linear(28 ** 2, 10),
            nn.BatchNorm1d(10, affine=False),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10, affine=False),
            nn.Linear(10, 10),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.model(x)

class PARA_5x5x5(nn.Module):
    def __init__(self):
        super(PARA_5x5x5, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.BatchNorm1d(5, affine=False, momentum=None),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.BatchNorm1d(5, affine=False, momentum=None),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.BatchNorm1d(5, affine=False, momentum=None),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        return self.model(x)

class PARA_7x7(nn.Module):
    def __init__(self):
        super(PARA_7x7, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 7),
            nn.ReLU(),
            nn.BatchNorm1d(7, affine=False, momentum=None),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.BatchNorm1d(7, affine=False, momentum=None),
            nn.Linear(7, 1),
        )

    def forward(self, x):
        return self.model(x)

class PARA_25(nn.Module):
    def __init__(self):
        super(PARA_25, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 25),
            nn.ReLU(),
            nn.BatchNorm1d(25, affine=False, momentum=None),
            nn.Linear(25, 1),
        )

    def forward(self, x):
        return self.model(x)


class CustomMNIST(nn.Module):
    def __init__(self, layers):
        super(CustomMNIST, self).__init__()

        parsed_layers = []
        parsed_layers.append(nn.Flatten(1, 3))
        parsed_layers.append(nn.Linear(28 ** 2, layers[0]))
        for i in range(1, len(layers)-1):
            parsed_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                nn.BatchNorm1d(layers[i+1], affine=False),
                parsed_layers.append(nn.ReLU())

        self.model = nn.Sequential(*parsed_layers)

    def forward(self, x):
        return self.model(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, 5),
            nn.BatchNorm2d(4, affine=False, momentum=None),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(4, 12, 5),
            nn.BatchNorm2d(12, affine=False, momentum=None),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(12 * 4 * 4, 10),
            nn.LogSoftmax())

    def forward(self, x):
        output = self.cnn(x)
        return self.mlp(output.view(x.shape[0], -1))


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.BatchNorm2d(6, affine=False),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16, affine=False),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.BatchNorm1d(120, affine=False),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84, affine=False),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax())

    def forward(self, x):
        output = self.cnn(x)
        return self.mlp(output.view(x.shape[0], -1))
