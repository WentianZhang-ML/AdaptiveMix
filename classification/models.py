import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import random
import sys
import numpy as np
import copy
import math
from torch.autograd.function import InplaceFunction
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def mixup_process(out, indices, lam):
    return out*lam + out[indices]*(1-lam)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def Hloss(res):
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    b = S(res) * LS(res)
    b = -1 * torch.mean(b)
    return b


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))
        out += self.shortcut(x)

        return out


class Wide_ResNet_orthognal(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, dropout_rate=0, gain=1.0):
        super(Wide_ResNet_orthognal, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes, bias=False)

        self.linear.weight.requires_grad = False

        self.nChannels = nStages[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data,gain)   # Initializing with orthogonal rows

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = F.normalize(out, dim=1, p=2)
        out = torch.abs(self.linear(out))

        return out
    def _embedding(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0),-1)
        out = F.normalize(out, dim=1, p=2)
        return out
    def _linear_classifier(self,x):
        out = torch.abs(self.linear(x))
        return out
    def adaptive_mix(self,x_a,x_b,alpha=1.0,noise_std=0.05):

        lam = np.random.beta(alpha, alpha)

        batch_size = x_a.size()[0]

        x_mixed = x_a * lam + x_b * (1-lam)

        out_mixed = self._embedding(x_mixed)

        out_a = self._embedding(x_a)

        out_b = self._embedding(x_b)

        mix_z = out_a * lam  + out_b * (1-lam)

        mix_z_ = mix_z + torch.normal(mean = 0., std = noise_std, size= (mix_z.size())).cuda()

        loss_mix = F.mse_loss(out_mixed, mix_z_.detach())

        mix_result = self._linear_classifier(mix_z)+ self._linear_classifier(out_mixed)

        return mix_result, lam, loss_mix


class OverfittingNet(nn.Module):
    def __init__(self, net, mu, sigma):
        super(OverfittingNet, self).__init__()
        self.mu = torch.Tensor(mu).float().view(3, 1, 1).cuda()
        self.sigma = torch.Tensor(sigma).float().view(3, 1, 1).cuda()
        self.net = net
    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return self.net(x)
    def adaptive_mix(self,x1,x2,alpha,noise_std=0.05):
        x1 = (x1 - self.mu) / self.sigma
        x2 = (x2 - self.mu) / self.sigma
        return self.net.adaptive_mix(x1,x2,alpha=alpha,noise_std=noise_std)
    def embedding(self,x1):
        x1 = (x1 - self.mu) / self.sigma
        return self.net._embedding(x1)
