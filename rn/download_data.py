from __future__ import print_function

import torch


import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import sys
from pathlib import Path

import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--cutout', default=32)
args = parser.parse_args()

bs = int(args.bs)
imsize = int(args.size)
size = imsize

num_classes = 10
transform_train = transforms.Compose([
    transforms.RandomCrop(args.cutout, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
transform_train.transforms.insert(0, torchvision.transforms.TrivialAugmentWide())

# Prepare dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset_eval = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

# Calculate the sizes for train and validation sets
train_size = int(0.75 * len(trainset))
val_size = len(trainset) - train_size

# Split the dataset
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
_, val_dataset_eval = torch.utils.data.random_split(trainset_eval, [train_size, val_size])

# Create the DataLoaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=4)
valloader_eval = torch.utils.data.DataLoader(val_dataset_eval, batch_size=bs, shuffle=False, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4)

num_classes = 100
transform_train = transforms.Compose([
    transforms.RandomCrop(args.cutout, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
transform_train.transforms.insert(0, torchvision.transforms.TrivialAugmentWide())

# Prepare dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainset_eval = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)

# Calculate the sizes for train and validation sets
train_size = int(0.75 * len(trainset))
val_size = len(trainset) - train_size

# Split the dataset
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
_, val_dataset_eval = torch.utils.data.random_split(trainset_eval, [train_size, val_size])

# Create the DataLoaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=4)
valloader_eval = torch.utils.data.DataLoader(val_dataset_eval, batch_size=bs, shuffle=False, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4)

num_classes = 10
transform_train = transforms.Compose([
    transforms.RandomCrop(args.cutout, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4379, 0.4440, 0.4729), (0.1980, 0.2010, 0.1970)),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4379, 0.4440, 0.4729), (0.1980, 0.2010, 0.1970)),
])

# Add RandAugment with N, M(hyperparameter)
transform_train.transforms.insert(0, torchvision.transforms.TrivialAugmentWide())

trainset = torchvision.datasets.SVHN(root='./data', split="train", download=True, transform=transform_train)
trainset_eval = torchvision.datasets.SVHN(root='./data', split="train", download=True, transform=transform_train)

# Calculate the sizes for train and validation sets
train_size = int(0.75 * len(trainset))
val_size = len(trainset) - train_size

# Split the dataset
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
_, val_dataset_eval = torch.utils.data.random_split(trainset_eval, [train_size, val_size])

# Create the DataLoaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=4)
valloader_eval = torch.utils.data.DataLoader(val_dataset_eval, batch_size=bs, shuffle=False, num_workers=4)

testset = torchvision.datasets.SVHN(root='./data', split="test", download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
