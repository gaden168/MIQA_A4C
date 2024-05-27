import torch.nn as nn
from torchvision import models

from torch import  nn
import  torchvision
from torch.nn import functional as F
import numpy as np
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)



model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}
class Baseline(nn.Module):
    """Linear regressor"""
    def __init__(self, name='resnet50'):
        super(Baseline, self).__init__()
        model_fun, feat_dim = model_dict[name]
        self.encoder = model_fun()
        self.fc  = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,1)
        )
    def forward(self, x):
        return self.fc(self.encoder(x))
    
    
class MTS(nn.Module):
    def __init__(self,numclass,name='resnet50'):
        super(MTS, self).__init__()
        model_fun, feat_dim = model_dict[name]
        self.encoder = model_fun()
        self.classifier = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,numclass)
        )
        self.sigm = nn.Sigmoid()
        
        self.fc = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,1)
        )
        self.con = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256)
        )
    def forward(self, x):
        # print("x1...",x.size())
        x = self.encoder(x)
        x1 = self.classifier(x) #class
        x1 = self.sigm(x1)
        x2 = self.fc(x) ##score
        x3= self.con(x)#contrast loss
        # print("x1...", x1.size())
        # print("x2...", x2.size())
        return x1,x2,x3


class Baseline_PB(nn.Module):
    """Linear regressor"""

    def __init__(self,name='resnet50'):
        super(Baseline_PB, self).__init__()
        model_fun, feat_dim = model_dict[name]
        self.encoder = model_fun()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        # self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)  # class
        return x

class Utr_model(nn.Module):
    def __init__(self,numclass):
        super(Utr_model, self).__init__()
        resnet = models.resnext50_32x4d(pretrained=False)
        self.base = nn.Sequential(*list(resnet.children())[:-2])

        self.bn1 = nn.BatchNorm2d(2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Linear(256,numclass)
        )

    def forward(self, x):

        base = self.base(x)

        x1 = self.bn1(base)
        x1 = self.relu1(x1)

        x1 = self.conv1(x1)
        x1 = F.avg_pool2d(x1, x1.size()[2:])
        x1 = x1.view(x1.size(0), -1)
        
        x1 = self.classifier(x1)
        # x1 = self.sigm(x1)

        return x1

#搭建模型
# class SiameseNetwork(nn.Module):
#     def __init__(self):
#             super(SiameseNetwork, self).__init__()
#             resnet = models.resnext50_32x4d(pretrained=True)
#             self.base = nn.Sequential(*list(resnet.children())[:-2])
#
#             self.bn1 = nn.BatchNorm2d(2048)
#             self.relu1 = nn.ReLU(inplace=True)
#             self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=True)
#
#             self.mlp = nn.Sequential(
#                 nn.Linear(1024, 256),
#                 nn.BatchNorm1d(256),
#                 nn.Linear(256,128)
#             )
#             self.regressor = nn.Sequential(
#                 nn.Linear(128, 1),
#             )
#             self.sigm = nn.Sigmoid()
#     def forward_once(self, x):
#         base = self.base(x)
#         x = self.bn1(base)
#         x = self.relu1(x)
#         x = self.conv1(x)
#         x = F.avg_pool2d(x, x.size()[2:])
#         x = x.view(x.size(0), -1)#[16, 1024]
#         return x
#
#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         # print('output1',output1.shape)
#         # print('output2',output2.shape)
#         # output2 = output2.transpose(0, 1)
#         # output = torch.mul(output1, output2)
#         output = torch.cat((output1,output2),1)
#         output = self.regressor(output)
#         return self.sigm(output)

class RankNet(nn.Module):
    def __init__(self):
        super(RankNet, self).__init__()
        resnet = models.resnext50_32x4d(pretrained=False)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.regressor = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Linear(256,1)
        )

        self.output_sig = nn.Sigmoid()
        
    def forward_once(self, x):
        base = self.base(x)
        x = self.bn1(base)
        x = self.relu1(x)
        x = self.conv1(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)#[16, 1024]
        x= self.regressor(x)
        return x
    
    def forward(self, input_1,input_2):
        s1 = self.forward_once(input_1)
        s2 = self.forward_once(input_2)
        out = self.output_sig(s1-s2)
        return out
    
    
#自定义ContrastiveLoss
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

    