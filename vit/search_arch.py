# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import random

import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils.utils import progress_bar
# from randomaug import RandAugment
# from models.res import resnet20
from models.vit import ViT
from models.convmixer import ConvMixer
from torch.utils.data import DataLoader, random_split

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from architecture.architecture import Architecture
from activation_cell.ac_cell import ActivationCell
from utils.utils import replace_ac_function

import time

import types

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--arch_lr', default=0.09, type=float, help='arch learning rate')  # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--arch_weight_decay', default=0.001, type=float, help='arch_weight_decay')  # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--data', default="cifar10")
parser.add_argument('--comp', default="gelu")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='enable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--zero_loss', action='store_true', help='add loss')
parser.add_argument('--disc_loss', action='store_true', help='add loss')
parser.add_argument('--monotonic_loss', action='store_true', help='add loss')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--start_shrinking', default=2, type=int)
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--search_epochs', type=int, default='35')
parser.add_argument('--warmstart_epoch', type=int, default='1')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--cutout', default=32, type=int)
parser.add_argument('--grad_acumm', default=1, type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--ac', default='relu', type=str, help="activation function")
parser.add_argument('--id', default=1, type=int, help="job id")
parser.add_argument('--results_folder', default='exps', type=str, help="results folder")
parser.add_argument('--log-wandb', action='store_true', default=False, help='log training and validation metrics to wandb')
parser.add_argument('--wandb_project', default='cifar10-challange', type=str, help="wandb project")
parser.add_argument('--wandb_name', default='search', type=str, help="wandb name")
parser.add_argument('--train_portion', default=0.75, type=float, help='train portion in train-val split')
parser.add_argument('--start_bilevel', default=0, type=int)
args = parser.parse_args()

def random_seed(seed=1337, rank=0):
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    # if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    # torch.use_deterministic_algorithms(True)

# os.environ["WANDB_MODE"]="offline"

# take in args
usewandb = args.log_wandb
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project=args.wandb_project,
            name=args.wandb_name)
    wandb.config.update(args)

random_seed(args.id)


bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.aug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = -1  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

t = time.time()

# Data
print('==> Preparing data..')
if args.net == "vit_timm":
    size = 384
else:
    size = imsize

def worker_init_fn(worker_id):
    np.random.seed(args.id + worker_id)
    random.seed(args.id + worker_id)
    torch.manual_seed(args.id + worker_id)


if args.data == "cifar10":
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(args.cutout, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:
        N = 2;
        M = 14;
        transform_train.transforms.insert(0, torchvision.transforms.TrivialAugmentWide())

    # Prepare dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    num_train = len(trainset) # remove
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader( # remove
        trainset, batch_size=bs,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader( # remove
        trainset, batch_size=bs,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

elif args.data == "cifar100":
    num_classes = 100
    transform_train = transforms.Compose([
        transforms.RandomCrop(args.cutout, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:
        N = 2;
        M = 14;
        transform_train.transforms.insert(0, torchvision.transforms.TrivialAugmentWide())

    # Prepare dataset
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        trainset, batch_size=bs,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        trainset, batch_size=bs,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

elif args.data == "svhn_core":
    num_classes = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(args.cutout, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:
        N = 2;
        M = 14;
        transform_train.transforms.insert(0, torchvision.transforms.TrivialAugmentWide())

    trainset = torchvision.datasets.SVHN(root='./data', split="train", download=True, transform=transform_train)
    
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        trainset, batch_size=bs,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        trainset, batch_size=bs,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

else:
    raise KeyError

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net == 'res18':
    net = ResNet18(num_classes=num_classes)
elif args.net == 'vgg11':
    net = VGG('VGG11')
elif args.net == 'vgg13':
    net = VGG('VGG13')
elif args.net == 'vgg16':
    net = VGG('VGG16')
elif args.net == 'vgg19':
    net = VGG('VGG19')
elif args.net == 'res34':
    net = ResNet34(num_classes=num_classes)
elif args.net == 'res50':
    net = ResNet50(num_classes=num_classes)
elif args.net == 'res101':
    net = ResNet101(num_classes=num_classes)
elif args.net == "convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=num_classes)
elif args.net == "mlpmixer":
    from models.mlpmixer import MLPMixer

    net = MLPMixer(
        image_size=32,
        channels=3,
        patch_size=args.patch,
        dim=512,
        depth=6,
        num_classes=num_classes
    )
elif args.net == "vit_small":
    from models.vit_small import ViT

    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "vit_tiny":
    from models.vit_small import ViT

    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=4,
        heads=6,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "simplevit":
    from models.simplevit import SimpleViT

    net = SimpleViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512
    )
elif args.net == "vit":
    # ViT for cifar10
    net = ViT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
elif args.net == "vit_timm":
    import timm

    net = timm.create_model("vit_base_patch16_384", pretrained=False)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net == "cait":
    from models.cait import CaiT

    net = CaiT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,  # depth of transformer for patch to patch attention only
        cls_depth=2,  # depth of cross attention of CLS tokens to patch
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05
    )
elif args.net == "cait_small":
    from models.cait import CaiT

    net = CaiT(
        image_size=size,
        patch_size=args.patch,
        num_classes=num_classes,
        dim=int(args.dimhead),
        depth=6,  # depth of transformer for patch to patch attention only
        cls_depth=2,  # depth of cross attention of CLS tokens to patch
        heads=6,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05
    )
elif args.net == "swin":
    from models.swin import swin_t

    net = swin_t(window_size=args.patch,
                 num_classes=num_classes,
                 downscaling_factors=(2, 2, 2, 1))
elif args.net == 'wresnet40_2':
    net = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_classes, adaptive_dropouter_creator=None,
                     adaptive_conv_dropouter_creator=None, groupnorm=False, examplewise_bn=False,
                     virtual_bn=False)
elif args.net == 'wresnet28_10':
    net = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_classes, adaptive_dropouter_creator=None,
                     adaptive_conv_dropouter_creator=None, groupnorm=False, examplewise_bn=False,
                     virtual_bn=False)
elif args.net == 'wresnet28_2':
    net = WideResNet(28, 2, dropout_rate=0.0, num_classes=num_classes, adaptive_dropouter_creator=None,
                     adaptive_conv_dropouter_creator=None, groupnorm=False, examplewise_bn=False,
                     virtual_bn=False)
elif args.net == 'restiny':
    net = ResNet_tiny(num_classes=num_classes)
elif args.net == 'res20':
    net = resnet20(num_classes=num_classes)
elif args.net == 'res32':
    net = resnet32(num_classes=num_classes)
elif args.net == 'res44':
    net = resnet44(num_classes=num_classes)
else:
    raise KeyError("Network is not supported")



net.cuda()

print('net original =', net)
    

n_params = sum(p.numel() for p in net.parameters())
print('n_params before:', n_params)


ac_func = ActivationCell().to(device)

replace_ac_function(net, nn.ReLU, ac_func)
replace_ac_function(net, nn.GELU, ac_func)

print('net =', net)

n_params = sum(p.numel() for p in net.parameters())
print('n_params after:', n_params)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


if not os.path.exists(f"results/{args.results_folder}") or not os.path.isdir(f"results/{args.results_folder}"):
    os.makedirs(f"results/{args.results_folder}", exist_ok=True)


# Loss is CE
criterion = nn.CrossEntropyLoss()

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def train(epoch, arch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_queue): ## restore trainloader
        ac_func.forward_type = "drnas"

        batch_arch = next(iter(valid_queue))

        if epoch > args.start_bilevel - 1:

            arch.step(net, batch_arch, args.grad_acumm, batch_idx, len(train_queue))

        if epoch < args.warmstart_epoch:
            if args.comp == "gelu":
                ac_func.forward_type = "gelu"
            elif args.comp == "relu":
                ac_func.forward_type = "relu"
            elif args.comp == "elu":
                ac_func.forward_type = "elu"
            elif args.comp == "silu":
                ac_func.forward_type = "silu"
            elif args.comp == "leakyrelu":
                ac_func.forward_type = "leakyrelu"
        
        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
        # Train with amp
        # todo also in arch. step
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets) / args.grad_acumm
        scaler.scale(loss).backward()
        # clip_grad_norm_(net.parameters(), 5.)
        if batch_idx % args.grad_acumm == args.grad_acumm - 1 or batch_idx == len(train_queue) - 1: ## restore trainloader
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_queue), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' ## restore trainloader
                     % (args.grad_acumm * train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    ac_func.forward_type = "drnas"
    return args.grad_acumm * train_loss / (batch_idx + 1)


##### Validation
def test(epoch, arch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_queue): ## restore trainloader
            # inputs, targets = batch_arch
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            ac_func.forward_type == "discretized"
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valid_queue), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' ## restore trainloader
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.net + '-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
        ac_func.best_genotype = ac_func.genotype()

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss / (batch_idx + 1):.5f}, acc: {(acc):.5f}'
    print(content)
    with open(
            f'log/log_{args.net}_patch{args.patch}_{args.id}_{args.zero_loss}_{args.disc_loss}_{args.monotonic_loss}.txt',
            'a') as appender:
        appender.write(content + "\n")
    ac_func.forward_type = "drnas"
    return test_loss / (batch_idx + 1), acc


list_val_loss = []
list_val_acc = []
list_train_loss = []
list_train_acc = []




if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)  # , weight_decay=5e-4)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  # , weight_decay=5e-4, momentum=0.9)
else:
    raise KeyError

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, 0.)

arch = Architecture(net, ac_func, valid_queue, criterion, args.n_epochs, scaler,
                    lr=args.arch_lr, arch_weight_decay=args.arch_weight_decay) ### CHANGED (arch_lr 0.001-->0.1)

condition_array = args.start_shrinking - 1 + np.logspace(np.log10(1), np.log10(args.search_epochs-args.start_shrinking), len(ac_func) - 6, base=10).astype(int)
print(condition_array, len(condition_array))
condition_idx = 0

condition_array_u = args.start_shrinking - 1 + np.logspace(np.log10(1), np.log10(args.search_epochs-args.start_shrinking), len(ac_func.ops[0]) - 1, base=10).astype(int)
condition_array_b = args.start_shrinking - 1 + np.logspace(np.log10(1), np.log10(args.search_epochs-args.start_shrinking), len(ac_func.ops[-1]) - 1, base=10).astype(int)


os.makedirs("plots", exist_ok=True)
os.makedirs(f"results/{args.results_folder}/search_{args.data}_{args.net}_{args.start_shrinking}_{args.arch_weight_decay>0}", exist_ok=True)
with open(
        f"results/{args.results_folder}/search_{args.data}_{args.net}_{args.start_shrinking}_{args.arch_weight_decay>0}/arch_{args.id}_{args.lr}_{args.search_epochs}_{args.n_epochs}_{args.warmstart_epoch}_{args.comp}.csv",
        "w") as f:
    
    f.write(f"epoch,train_loss,val_loss,val_acc,activation,best_activation,time")

ac_func.best_genotype = ac_func.genotype()
start_all = time.time()
for epoch in range(start_epoch, args.search_epochs):

    # shrinking method
    if epoch in condition_array:
        ac_func.drop_op(np.count_nonzero(condition_array == epoch))

    if all([sum(m)==1 for m in ac_func.mask]):
        print('fully discretized architecture found before max epochs reached')
        break

    start = time.time()
    trainloss = train(epoch, arch)
    val_loss, acc = test(epoch, arch)

    scheduler.step()  # step cosine scheduling

    if usewandb:
        wandb.log({
            'epoch': epoch,
            'train_loss': trainloss,
            'val_loss': val_loss,
            'val_acc': acc,
            'time': time.time() - start,
            'total search time': time.time() - start_all,
            "id": args.id,
            "num remaining ops": sum([sum(m) for m in ac_func.mask]),
            "num remaining unary ops 0": sum(ac_func.mask[0]),
            "num remaining unary ops 1": sum(ac_func.mask[1]),
            "num remaining binary ops 2": sum(ac_func.mask[2]),
            "num remaining unary ops 3": sum(ac_func.mask[3]),
            "num remaining unary ops 1": sum(ac_func.mask[4]),
            "num remaining binary ops 2": sum(ac_func.mask[5]),
            "run_name": args.wandb_name,
        })

    
    torch.save(ac_func.best_genotype, f"results/{args.results_folder}/search_{args.id}.pth")
    activation = [type(op).__name__ if len(tuple((float(p.cpu()) for p in op.parameters())))==0 else type(op).__name__+f"-{tuple((float(p.cpu()) for p in op.parameters()))}" for op in ac_func.genotype()]
    best_activation = [type(op).__name__ if len(tuple((float(p.cpu()) for p in op.parameters())))==0 else type(op).__name__+f"-{tuple((float(p.cpu()) for p in op.parameters()))}" for op in ac_func.best_genotype]
    print(f'activation at epoch {epoch} =', activation)
    print(f'best activation till epoch {epoch} =', best_activation)

    with open(
            f"results/{args.results_folder}/search_{args.data}_{args.net}_{args.start_shrinking}_{args.arch_weight_decay>0}/arch_{args.id}_{args.lr}_{args.search_epochs}_{args.n_epochs}_{args.warmstart_epoch}_{args.comp}.csv",
            "a") as f:
        f.write(
            f"\n{epoch},{trainloss},{val_loss},{acc},{activation},{best_activation},{time.time()-start}"
        )

        
end = time.time()