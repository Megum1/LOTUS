import os
import random
import numpy as np

import torch
from torchvision import datasets, transforms

from models import *


# Set random seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Dataset configurations (mean, std, size, num_classes)
_dataset_name = ['cifar10']

_mean = {
    'cifar10':  [0.4914, 0.4822, 0.4465],
}

_std = {
    'cifar10':  [0.2023, 0.1994, 0.2010],
}

_size = {
    'cifar10':  (32, 32),
}

_num = {
    'cifar10':  10,
}


def get_config(dataset):
    assert dataset in _dataset_name, _dataset_name
    config = {}
    config['mean'] = _mean[dataset]
    config['std']  = _std[dataset]
    config['size'] = _size[dataset]
    config['num_classes'] = _num[dataset]
    return config


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_transform(dataset, augment=False, tensor=False):
    transforms_list = []
    if augment:
        transforms_list.append(transforms.Resize(_size[dataset]))
        transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))

        # Horizontal Flip
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(transforms.Resize(_size[dataset]))

    # To Tensor
    if not tensor:
        transforms_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms_list)
    return transform


def get_augment(dataset):
    transforms_list = []
    transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
    transforms_list.append(transforms.RandomHorizontalFlip())
    transform = transforms.Compose(transforms_list)
    return transform


# Get dataset
def get_dataset(dataset, datadir='data', train=True, augment=True):
    transform = get_transform(dataset, augment=train & augment)
    
    if dataset == 'cifar10':
        dataset = datasets.CIFAR10(datadir, train, download=True, transform=transform)

    return dataset


# Get model
def get_model(dataset, network):
    num_classes = _num[dataset]

    if network == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif network == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif network == 'vgg11':
        model = vgg11(num_classes=num_classes)
    elif network == 'vgg13':
        model = vgg13(num_classes=num_classes)
    else:
        raise NotImplementedError

    return model
