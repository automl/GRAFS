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

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time
import inspect

from models import *
from utils.utils import progress_bar
# from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer

from utils.utils import replace_ac_function

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from architecture.architecture import Architecture
from activation_cell.ac_cell import ActivationCell
from utils.utils import replace_ac_function
import time

import random

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--data', default="cifar10")
parser.add_argument('--aug', action='store_true', help='enable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--cutout', default=32, type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--num_workers', default='4', type=int, help="num dataloader workers")
parser.add_argument('--ac', default=None, type=str, help="activation function")
parser.add_argument('--seed', default=1, type=int, help="random seed")
parser.add_argument('--id', default=-1, type=int, help="search id")
parser.add_argument('--eval', action='store_true', help='eval modus')
parser.add_argument('--grad_acumm', default=1, type=int)
parser.add_argument('--results_folder', default='exps', type=str, help="results folder")
parser.add_argument('--log-wandb', action='store_true', default=False, help='log training and validation metrics to wandb')
parser.add_argument('--wandb_project', default='cifar10-challange', type=str, help="wandb project")
parser.add_argument('--wandb_name', default='eval', type=str, help="wandb name")
args = parser.parse_args()

def random_seed(seed=1337, rank=0):
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    # if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed + rank)

# os.environ["WANDB_MODE"]="offline"

# take in args
usewandb = args.log_wandb
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project=args.wandb_project,
            name=args.wandb_name)
    wandb.config.update(args)

random_seed(seed=args.seed)

# torch.manual_seed(args.id)
# np.random.seed(args.id)
# random.seed(args.id)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.aug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

t = time.time()

# Data
print('==> Preparing data..')
if args.net=="vit_timm" or args.net=="vit_timm_tiny":
    size = 384
else:
    size = imsize

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
        transforms.Resize(32),
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=args.num_workers)
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=args.num_workers)
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
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:
        N = 2;
        M = 14;
        transform_train.transforms.insert(0, torchvision.transforms.TrivialAugmentWide())

    trainset = torchvision.datasets.SVHN(root='./data', split="train", download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.SVHN(root='./data', split="test", download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=args.num_workers)
else:
    raise KeyError

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18(num_classes=num_classes)
elif args.net == 'vgg11':
    net = VGG('VGG11')
elif args.net == 'vgg13':
    net = VGG('VGG13')
elif args.net == 'vgg16':
    net = VGG('VGG16')
elif args.net == 'vgg19':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34(num_classes=num_classes)
elif args.net=='res50':
    net = ResNet50(num_classes=num_classes)
elif args.net=='res101':
    net = ResNet101(num_classes=num_classes)
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=num_classes)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes=num_classes
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes=num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes=num_classes,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes=num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes=num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="vit_timm_tiny":
    import timm
    net = timm.create_model("vit_tiny_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes=num_classes,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes=num_classes,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                 num_classes=num_classes,
                 downscaling_factors=(2, 2, 2, 1))
elif args.net == 'wresnet40_2':
    net = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_classes, adaptive_dropouter_creator=None,
                                      adaptive_conv_dropouter_creator=None, groupnorm=False, examplewise_bn=False,
                                      virtual_bn=False)
elif args.net== 'wresnet28_10':
    net = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_classes, adaptive_dropouter_creator=None,
                                      adaptive_conv_dropouter_creator=None, groupnorm=False, examplewise_bn=False,
                                      virtual_bn=False)
elif args.net== 'wresnet28_2':
    net = WideResNet(28, 2, dropout_rate=0.0, num_classes=num_classes, adaptive_dropouter_creator=None,
                                      adaptive_conv_dropouter_creator=None, groupnorm=False, examplewise_bn=False,
                                      virtual_bn=False)
elif args.net == 'res20':
    net = resnet20(num_classes=num_classes)
elif args.net == 'res32':
    net = resnet32(num_classes=num_classes)
elif args.net == 'res44':
    net = resnet44(num_classes=num_classes)
else:
    raise KeyError("Network is not supported")

# # For Multi-GPU
# if 'cuda' in device:
#     print(device)
#     print("using data parallel")
#     net = torch.nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets) / args.grad_acumm
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        if batch_idx % args.grad_acumm == args.grad_acumm - 1 or batch_idx == len(trainloader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log_test_cifar10", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)

    with open(f'log_test_cifar10/log_{args.net}_patch{args.patch}_{args.ac}_{args.id}_{args.lr}_{args.aug}_{args.cutout}_{args.opt}_{args.n_epochs}_test.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss/(batch_idx+1), acc

list_loss = []
list_acc = []

# if usewandb:
#     wandb.watch(net)
    
net.cuda()

print(net)

os.makedirs(f"results/{args.results_folder}/eval_{args.data}_{args.net}", exist_ok=True)


# f_name = args.ac
if args.ac is None:
    print('Evaluation with default activation function')
elif args.ac == 'discovered':
    ac_func = ActivationCell().to(device)

    # ## remove clamping of operations
    for ub in ac_func.ops:
        for op in ub:
            if hasattr(op, 'inf'):
                op.inf = torch.tensor(1e+7) # torch.inf
            if hasattr(op, 'eps'):
                op.eps = torch.tensor(0.0) # torch.inf


    ac_func.eval_genotype = torch.load(f"results/{args.results_folder}/search_{args.id}.pth")
    ac_func.forward_type = "eval"
    replace_ac_function(net, torch.nn.GELU, ac_func)
    replace_ac_function(net, torch.nn.ReLU, ac_func)
    eval_activation = [type(op).__name__ if len(tuple((float(p.cpu()) for p in op.parameters())))==0 else type(op).__name__+f"-{tuple((float(p.cpu()) for p in op.parameters()))}" for op in ac_func.eval_genotype]
    print(f'Evaluation with discovered activation {eval_activation}')
    with open(f"results/{args.results_folder}/eval_{args.data}_{args.net}/arch_{args.id}_{args.seed}_{args.lr}_{args.n_epochs}_{args.ac}.csv", "w") as f:
        f.write(
            f"epoch,train_loss,val_loss,val_acc,time,{ac_func.eval_genotype}"
        )
else:
    if args.ac == "relu":
        ac = nn.ReLU
    elif args.ac == "gelu":
        ac = nn.GELU
    elif args.ac == "silu":
        ac = nn.SiLU
    elif args.ac == "leakyrelu":
        ac = nn.LeakyReLU
    elif args.ac == "elu":
        ac = nn.ELU
    elif args.ac == "sigmoid":
        ac = nn.Sigmoid
    else:
        classes = {cls_name: cls_obj for cls_name, cls_obj in inspect.getmembers(sys.modules['models.custom_activations'])}
        ac = classes[args.ac]

    ac_func = ac().to(device)

    replace_ac_function(net, torch.nn.GELU, ac_func)
    replace_ac_function(net, torch.nn.ReLU, ac_func)
    print(f'Evaluation with baseline activation {type(ac_func).__name__}')
    with open(f"results/{args.results_folder}/eval_{args.data}_{args.net}/arch_{args.id}_{args.seed}_{args.lr}_{args.n_epochs}_{args.ac}.csv", "w") as f:
        f.write(
            f"epoch,train_loss,val_loss,val_acc,time,{type(ac_func).__name__}"
        )

print(net)


start_all = time.time()
for epoch in range(start_epoch, args.n_epochs):

    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    if usewandb:
        wandb.log({
            'epoch': epoch,
            'train_loss': trainloss,
            'test_loss': val_loss,
            'test_acc': acc,
            'epoch time': time.time() - start,
            'total time': time.time() - start_all,
            "id": args.id,
            "seed": args.seed,
            "run_name": args.wandb_name,
        })
    
    # scheduler.step(epoch-1)
    scheduler.step()

    with open(f"results/{args.results_folder}/eval_{args.data}_{args.net}/arch_{args.id}_{args.seed}_{args.lr}_{args.n_epochs}_{args.ac}.csv", "a") as f:
        f.write(
            f"\n{epoch},{trainloss},{val_loss},{acc},{time.time()-start}"
        )

end = time.time()  
