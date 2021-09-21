from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch as t
import torch.nn as nn
from networks.layers import TriResSeparateConv3D
import torch.nn.functional as F


class Pnet(nn.Module):
    def __init__(self, conv_inc=1, base_chns=24, norm='in', 
            depth = False, dilation=1, separate_direction='axial',
            output_distance=False, fc_inc=1,        
            classes=3, droprate=0, dis_range=300):
        super(Pnet, self).__init__()

        self.dropout = nn.Dropout(droprate)
        self.downsample = nn.MaxPool3d(2, 2)
        self.output_distance = output_distance

        self.conv1 = TriResSeparateConv3D(conv_inc, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv2 = TriResSeparateConv3D(2*base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv3 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv4 = TriResSeparateConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv5 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.fc1 = nn.Linear(fc_inc, 8 * base_chns)
        self.fc2 = nn.Linear(8 * base_chns, 4 * base_chns)
        self.fc3 = nn.Linear(4 * base_chns, classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.downsample(conv1)  # 1/2
        conv2 = self.conv2(out)  #
        out = self.downsample(conv2)  # 1/4
        conv3 = self.conv3(out)  #
        out = self.downsample(conv3)  # 1/8
        out = self.conv4(out)
        g_feature = self.conv5(out)

        out = g_feature.view(g_feature.shape[0],-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return {'fc_position':out, 'g_feature':g_feature}
