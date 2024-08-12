# https://github.com/akamaster/pytorch_resnet_cifar10/tree/master

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.resnet as resnet
from models import *
# from models.resnet import *
import numpy as np
import random


import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from architecture.architecture import Architecture
from activation_cell.ac_cell import ActivationCell
from utils.utils import replace_ac_function

import time



model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--model', default='res20')
parser.add_argument('--data', default="cifar10")
parser.add_argument('--arch_lr', default=0.09, type=float, help='arch learning rate')  # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--arch_weight_decay', default=0.001, type=float, help='arch_weight_decay')  # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--warmstart_epoch', type=int, default='1')
parser.add_argument('--grad_acumm', default=1, type=int)
parser.add_argument('--ac', default='relu', type=str, help="activation function")
parser.add_argument('--comp', default="relu")
parser.add_argument('--id', default=1, type=int, help="job id")
parser.add_argument('--results_folder', default='exps', type=str, help="results folder")
parser.add_argument('--log-wandb', action='store_true', default=False, help='log training and validation metrics to wandb')
parser.add_argument('--wandb_project', default='cifar10-challange', type=str, help="wandb project")
parser.add_argument('--wandb_name', default='search', type=str, help="wandb name")
parser.add_argument('--train_portion', default=0.75, type=float, help='train portion in train-val split')
parser.add_argument('--start_shrinking', default=2, type=int)
parser.add_argument('--start_bilevel', default=0, type=int)
best_prec1 = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# use_amp = not args.noamp
scaler = torch.cuda.amp.GradScaler(enabled=True)

def main():
    global args, best_prec1
    args = parser.parse_args()

    def random_seed(seed=1337, rank=0):
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        torch.manual_seed(seed + rank)

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
        watermark = "{}_lr{}".format(args.model, args.lr)
        wandb.init(project=args.wandb_project,
                name=args.wandb_name)
        wandb.config.update(args)

    random_seed(args.id)


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    # model.cuda()
    args.arch = args.model


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    if args.data == "cifar10":
        num_classes = 10
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        trainset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size, 
            # shuffle=True,
            num_workers=args.workers, 
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size, 
            # shuffle=True,
            num_workers=args.workers, 
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True)

    elif args.data == "cifar100":
        num_classes = 100
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
        
        trainset = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size, 
            # shuffle=True,
            num_workers=args.workers, 
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size, 
            # shuffle=True,
            num_workers=args.workers, 
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True)

    elif args.data == "svhn_core":
        num_classes = 10
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                        std=[0.1980, 0.2010, 0.1970])
        
        trainset = datasets.SVHN(root='./data', split="train", transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size, 
            # shuffle=True,
            num_workers=args.workers, 
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size, 
            # shuffle=True,
            num_workers=args.workers, 
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True)


    


    if args.model == 'res20':
        model = resnet20(num_classes=num_classes)
    elif args.model == 'res32':
        model = resnet32(num_classes=num_classes)
    elif args.model == 'res44':
        model = resnet44(num_classes=num_classes)
    elif args.model == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_classes, adaptive_dropouter_creator=None,
                        adaptive_conv_dropouter_creator=None, groupnorm=False, examplewise_bn=False,
                        virtual_bn=False)
    elif args.model == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_classes, adaptive_dropouter_creator=None,
                        adaptive_conv_dropouter_creator=None, groupnorm=False, examplewise_bn=False,
                        virtual_bn=False)
    elif args.model == 'wresnet28_2':
        model = WideResNet(28, 2, dropout_rate=0.0, num_classes=num_classes, adaptive_dropouter_creator=None,
                        adaptive_conv_dropouter_creator=None, groupnorm=False, examplewise_bn=False,
                        virtual_bn=False)
    else:
        raise KeyError("Network is not supported")
    
    model.cuda()


    print('model original =', model)

    n_params = sum(p.numel() for p in model.parameters())
    print('n_params before:', n_params)


    ac_func = ActivationCell().to(device)
    replace_ac_function(model, nn.ReLU, ac_func)
    replace_ac_function(model, nn.GELU, ac_func)

    print('model after =', model)

    n_params = sum(p.numel() for p in model.parameters())
    print('n_params after:', n_params)

    if not os.path.exists(f"results/{args.results_folder}") or not os.path.isdir(f"results/{args.results_folder}"):
        os.makedirs(f"results/{args.results_folder}", exist_ok=True)



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)
    
    arch = Architecture(model, ac_func, val_loader, criterion, args.epochs, scaler,
                        lr=args.arch_lr, arch_weight_decay=args.arch_weight_decay)

    condition_array = args.start_shrinking - 1 + np.logspace(np.log10(1), np.log10(args.epochs-args.start_shrinking), len(ac_func) - 6, base=10).astype(int)
    print(condition_array, len(condition_array))


    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    start_all = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()

        # shrinking
        if epoch in condition_array:
            ac_func.drop_op(np.count_nonzero(condition_array == epoch))

        if all([sum(m)==1 for m in ac_func.mask]):
            print('fully discretized architecture found before max epochs reached')
            break

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_loss = train(train_loader, val_loader, model, ac_func, arch, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1, val_loss = validate(val_loader, model, ac_func, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            ac_func.best_genotype = ac_func.genotype()

        if usewandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': prec1,
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


def train(train_loader, val_loader, model, ac_func, arch, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        ac_func.forward_type = "drnas"

        batch_arch = next(iter(val_loader))
    
        if epoch > args.start_bilevel - 1:
            arch.step(model, batch_arch, args.grad_acumm, i, len(train_loader))

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


        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()


        with torch.cuda.amp.autocast(enabled=True):
            output = model(input_var)
            loss = criterion(output, target_var) / args.grad_acumm
        scaler.scale(loss).backward()
        # clip_grad_norm_(net.parameters(), 5.)
        if i % args.grad_acumm == args.grad_acumm - 1 or i == len(train_loader) - 1: ## restore trainloader
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()




        output = output.float()
        loss = loss.float()
       
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            
    return losses.avg


def validate(val_loader, model, ac_func, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            ac_func.forward_type == "discretized"

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    
    ac_func.forward_type = "drnas"

    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
