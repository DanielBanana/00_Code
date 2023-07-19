import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, Subset
from torchvision import transforms, datasets



import numpy as np

from cbo_in_python.src import DATASETS_DIR

"""
A list of helper functions for loading the popular datasets. By default, the datasets are downloaded and saved
to the `saved_datasets` folder.
"""


def load_mnist_dataloaders(train_batch_size, test_batch_size, data_dir=None):
    """
    Downloads and loads the MNIST datasets. Returns the dataloaders for the train and test subsets respectively.
    :param train_batch_size: train dataloader batch size.
    :param test_batch_size: test dataloader batch size.
    :param data_dir: directory to save the downloaded dataset to. Optional.
    :return: train and test MNIST dataloaders (respectively).
    """
    if data_dir is None:
        data_dir = DATASETS_DIR
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        # Copied from this example https://github.com/pytorch/examples/blob/main/mnist/main.py#L114
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=mnist_transforms)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=mnist_transforms)

    # indices = torch.arange(1000)
    # mnist_train = Subset(mnist_train, indices)

    train_dataloader = DataLoader(mnist_train, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(mnist_test, batch_size=test_batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def get_mnist_dataset(data_dir=None):
    """
    Downloads and loads the MNIST datasets.
    :param train_batch_size: train dataloader batch size.
    :param test_batch_size: test dataloader batch size.
    :param data_dir: directory to save the downloaded dataset to. Optional.
    :return: train and test MNIST dataloaders (respectively).
    """
    if data_dir is None:
        data_dir = DATASETS_DIR
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        # Copied from this example https://github.com/pytorch/examples/blob/main/mnist/main.py#L114
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=mnist_transforms)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=mnist_transforms)

    return mnist_train, mnist_test


class GenericDataset(Dataset):
    def __init__(self, x:torch.Tensor, y:torch.Tensor):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

def create_generic_dataset(input, output):
    return GenericDataset(input, output)

def load_generic_dataloaders(train_dataset:GenericDataset, train_batch_size:int, test_dataset:GenericDataset, test_batch_size:int, shuffle=True, drop_last=True):
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, drop_last=drop_last)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle, drop_last=drop_last)
    return train_dataloader, test_dataloader

def load_parabola_dataloaders(train_batch_size, test_batch_size, data_dir=None, n_train_samples=60000, n_test_samples=10000, shuffle=True):
    x_train = torch.Tensor(np.random.uniform(-5, 5, n_train_samples)).reshape(-1,1)
    x_test = torch.Tensor(np.random.uniform(-5, 5, n_test_samples)).reshape(-1,1)

    def f(x):
        return x*x

    y_train = f(x_train)
    train_dataset = GenericDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle)

    y_test = f(x_test)
    test_dataset = GenericDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloader

def load_shifted_parabola_dataloaders(train_batch_size, test_batch_size, data_dir=None, n_train_samples=60000, n_test_samples=10000, shuffle=True):
    x_train = torch.Tensor(np.random.uniform(-5, 5, n_train_samples)).reshape(-1,1)
    x_test = torch.Tensor(np.random.uniform(-5, 5, n_test_samples)).reshape(-1,1)

    def f(x):
        return 8.53*(1-x*x)

    y_train = f(x_train)
    train_dataset = GenericDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle)

    y_test = f(x_test)
    test_dataset = GenericDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloader