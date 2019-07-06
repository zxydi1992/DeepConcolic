import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),])


CIFAR10_RESNET_MEAN = [0.485, 0.456, 0.406]
CIFAR10_RESNET_STD = [0.229, 0.224, 0.225]


CIFAR10_VGG_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_VGG_STD = [0.2023, 0.1994, 0.2010]


def cifar10_resnet_normalize(t, mean=None, std=None):
    if mean is None:
        mean = CIFAR10_RESNET_MEAN
    if std is None:
        std= CIFAR10_RESNET_STD

    ts = []
    for i in range(3):
        ts.append(torch.unsqueeze((t[:, i] - mean[i]) / std[i], 1))
    return torch.cat(ts, dim=1)


def cifar10_vgg_normalize(t, mean=None, std=None):
    if mean is None:
        mean = CIFAR10_VGG_MEAN
    if std is None:
        std = CIFAR10_VGG_STD

    ts = []
    for i in range(3):
        ts.append(torch.unsqueeze((t[:, i] - mean[i]) / std[i], 1))
    return torch.cat(ts, dim=1)


def cifar10_vgg_normalize_np(t, mean=None, std=None):
    if mean is None:
        mean = CIFAR10_VGG_MEAN
    mean = np.asarray(mean, dtype=np.float32).reshape((1, -1, 1, 1))
    if std is None:
        std = CIFAR10_VGG_STD
    std = np.asarray(std, dtype=np.float32).reshape((1, -1, 1, 1))
    return (t - mean) / std


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_cifar10_dataset():
    return trainset, testset


def get_cifar10_dataset_np():
    trainset, testset = get_cifar10_dataset()
    train_data = (np.asarray(trainset.train_data, np.float32) / 255.).transpose((0, 3, 1, 2))
    train_labels = np.asarray(trainset.train_labels, dtype=np.int64)

    test_data = (np.asarray(testset.test_data, np.float32) / 255.).transpose((0, 3, 1, 2))
    test_labels = np.asarray(testset.test_labels, dtype=np.int64)
    return train_data, train_labels, test_data, test_labels

