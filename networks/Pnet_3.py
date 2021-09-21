from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch as t
import numpy as np
import torch.nn as nn
from networks.layers import TribleResConv3D
import torch.nn.functional as F


class Pnet_3(nn.Module):
    def __init__(self, inc=1, base_chns=24, norm='in', 
            depth = False, dilation=1, patch_size=1,        
            n_classes=3, droprate=0, dis_range=300):
        super(Pnet_3, self).__init__()

        self.dropout = nn.Dropout(droprate)
        self.downsample = nn.MaxPool3d(2, 2)

        self.conv1 = TribleResConv3D(inc, 2*base_chns, norm=norm, depth=depth)

        self.conv2 = TribleResConv3D(2*base_chns, 2 * base_chns, norm=norm, depth=depth)

        self.conv3 = TribleResConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth)

        self.conv4 = TribleResConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth)
        self.conv5 = TribleResConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth)
        fc_inc = int(np.asarray(patch_size).prod()/4096)*4*base_chns
        self.fc1 = nn.Linear(fc_inc, 8 * base_chns)
        self.fc2 = nn.Linear(8 * base_chns, 4 * base_chns)
        self.fc3 = nn.Linear(4 * base_chns, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.downsample(out)  # 1/2
        out = self.conv2(out)  #
        out = self.downsample(out)  # 1/4
        out = self.conv3(out)  #
        out = self.downsample(out)  # 1/8
        out = self.conv4(out)
        out = self.downsample(out)
        g_feature = self.conv5(out)

        out = g_feature.view(g_feature.shape[0],-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return {'fc_position':out, 'feature':g_feature}
