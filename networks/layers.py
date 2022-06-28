#coding:utf8
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

class depthwise(nn.Module):
    '''
    depthwise convlution
    '''
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=3, dilation=1, depth=False):
        super(depthwise, self).__init__()
        if depth:
                self.Conv=nn.Sequential(OrderedDict([('conv1_1_depth', nn.Conv3d(cin, cin,
                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=cin)),
                        ('conv1_1_point', nn.Conv3d(cin, cout, 1))]))
        else:
            if stride>=1:
                self.Conv=nn.Conv3d(cin, cout, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, dilation=dilation)
            else:
                stride = int(1//stride)
                self.Conv = nn.ConvTranspose3d(cin, cout, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, dilation=dilation)
    def forward(self, x):
        return self.Conv(x)

class DoubleResConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', droprate=0, depth=False):
        super(DoubleResConv3D, self).__init__()
        if norm == 'bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2_depth', depthwise(cout, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU()),
        ]))
        self.norm = Norm(cout)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class TribleResConv3D(nn.Module):

    def __init__(self, cin, cout, norm='in', droprate=0, depth=False):
        super(TribleResConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2_depth', depthwise(cout, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_3', nn.ReLU()),
            ('conv1_3_depth', depthwise(cout, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU()),
        ]))
        self.norm = Norm(cout)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class TriResSeparateConv3D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3,norm='in', droprate=0, depth=False, pad=0, dilat=1, active=True, separate_direction='axial'):
        super(TriResSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()

        if separate_direction == 'axial':
            if pad == 'same':
                padding = [[0, dilat, dilat], [dilat, 0, 0]]
            else:
                padding = [[0, 1, 1], [1, 0, 0]]
            conv = [[1, kernel_size, kernel_size], [kernel_size, 1, 1]]
            dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        elif separate_direction == 'sagittal':
            if pad == 'same':
                padding = [[dilat, 0, dilat], [0, dilat, 0]]
            else:
                padding = [[1, 0, 1], [0, 1, 0]]
            conv = [[kernel_size, 1, kernel_size], [1, kernel_size, 1]]
            dilation = [[dilat, 1, dilat], [1, dilat, 1]]
        elif separate_direction == 'coronal':
            if pad == 'same':
                padding = [[dilat, dilat, 0], [0, 0, dilat]]
            else:
                padding = [[1, 1, 0], [0, 0, 1]]
            conv = [[kernel_size, kernel_size, 1], [1, 1, kernel_size]]
            dilation = [[dilat, dilat, 1], [1, 1, dilat]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_1_depth_deep', depthwise(cout, cout, conv[1], padding=padding[1], depth=depth, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(cout)),
            ('relu1_1_deep', nn.ReLU(inplace=True)),
            ('conv1_2_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU(inplace=True)),
        ]))

        self.norm = Norm(cout)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)