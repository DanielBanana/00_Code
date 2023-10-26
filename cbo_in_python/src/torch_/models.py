import torch.nn as nn
from typing import List
import numpy as np


class TinyMLP(nn.Module):
    def __init__(self):
        super(TinyMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(1, 3),
            nn.Linear(28 ** 2, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10, affine=False, momentum=None),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.model(x)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, x):
        return self.model(x)

class CustomMLP(nn.Module):
    def __init__(self, layers:List = [1,10,1]):
        super(CustomMLP, self).__init__()

        parsed_layers = []
        for i in range(len(layers)-1):
            parsed_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                nn.BatchNorm1d(layers[i+1], affine=False),
                parsed_layers.append(nn.ReLU())

        self.model = nn.Sequential(*parsed_layers)

        self.init_weights()

        self.double()

    def forward(self, x):
        return self.model(x)

    def init_weights(self):

        def weights_init_uniform_rule(m):
            classname = m.__class__.__name__
            # for every Linear layer in a model..
            if classname.find('Linear') != -1:
                # get the number of the inputs
                n = m.in_features
                y = 1.0/np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                m.bias.data.uniform_(-y, y)

        self.model.apply(weights_init_uniform_rule)



class SmallMLP(nn.Module):
    def __init__(self):
        super(SmallMLP, self).__init__()

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
