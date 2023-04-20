# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import config as cf

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import sys
import time
import argparse
import datetime
import copy
from tqdm import *

from models import Wide_ResNet_orthognal,mixup_criterion,OverfittingNet



parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument("--savedir",type=str, default='./model/')
parser.add_argument("--alpha",type=float, default= '1.0', help='hyper_paramters for alpha')
parser.add_argument("--noise",type=float, default= '0.05', help='hyper_paramters for the std of noise')

args = parser.parse_args()


use_cuda = torch.cuda.is_available()

time_point = time.strftime("%a_%b_%d_%H_%M_%S", time.localtime())

args.savedir = os.path.join(args.savedir,time_point)
#Parame Initialization

start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
# Transform
transform_attack_train, transform_attack_test = cf.get_transform_attack()

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=transform_attack_train)
    testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_attack_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./', train=True, download=False, transform=transform_attack_train)
    testset = torchvision.datasets.CIFAR100(root='./', train=False, download=False, transform=transform_attack_test)
    num_classes = 100


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


net = Wide_ResNet_orthognal(args.depth,
                            args.widen_factor,
                            num_classes).cuda()
net = OverfittingNet(net, mu=cf.mean[args.dataset], sigma=cf.std[args.dataset])

criterion = nn.CrossEntropyLoss()

acc_list = [0,0,0,0,0]

def test_acc():
    net.eval()
    net.training = False
    correct = 0.
    total = 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
    acc = correct/total
    return acc

def warm_up(num,threshold_num=10):
    if num > threshold_num:
        return 1.
    else:
        return num / threshold_num

def train_clean(epoch):
    global acc_list
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    mix_loss=0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
    t = tqdm(trainloader)
    t.set_description("Epoch [{}/{}]".format(epoch,num_epochs))
    for batch_idx, (inputs, targets) in enumerate(t):
        net.train()
        net.training = True

        inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        index = torch.randperm(inputs.size(0)).cuda()

        x_a = inputs
        targets_a = targets
        x_b = copy.deepcopy(inputs[index,:])
        targets_b = copy.deepcopy(targets[index])

        out_mixed, lam, mix_consist = net.adaptive_mix(x_a,x_b,alpha=args.alpha,noise_std=args.noise)

        optimizer.zero_grad()

        mix_cate_loss = mixup_criterion(criterion,out_mixed,targets_a,targets_b,lam)

        total_loss =  mix_cate_loss + mix_consist

        total_loss.backward()

        optimizer.step()

        cate_loss = mix_cate_loss.item()
        mix_loss = mix_consist.item()

        if batch_idx % 30 == 0:
            acc = test_acc()
            if acc> min(acc_list):
                acc_list[acc_list.index(min(acc_list))] = acc
                acc_list = sorted(acc_list,reverse=True)
                name = 'TOP_' + str(acc_list.index(acc)+1)+'_Net.pth'
                save_model(args.savedir,name)
                print("| Save model! Path:{} acc: {:4f}".format(os.path.join(args.savedir,name), acc))

        t.set_postfix_str('loss_cls: {:4f},loss_mix: {:4f} acc@1: {:4f}'.format(cate_loss, mix_loss, acc))

        t.update()

def save_model(dir_name,name):
    torch.save(net.net.state_dict(),os.path.join(dir_name,name))

if __name__ == '__main__':
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    for i in range(num_epochs):
        train_clean(i+1)
