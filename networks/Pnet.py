from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch as t
import torch.nn as nn
import numpy as np
from networks.layers import TriResSeparateConv3D
import torch.nn.functional as F


class Pnet(nn.Module):
    def __init__(self, inc=1, base_chns=24, norm='in', 
            depth = False, separate_direction='axial',
            patch_size=1, n_classes=3):
        super(Pnet, self).__init__()

        self.downsample = nn.MaxPool3d(2, 2)

        self.conv1 = TriResSeparateConv3D(inc, 2*base_chns, norm=norm, depth=depth, pad='same', separate_direction=separate_direction)

        self.conv2 = TriResSeparateConv3D(2*base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', separate_direction=separate_direction)

        self.conv3 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', separate_direction=separate_direction)

        self.conv4 = TriResSeparateConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, pad='same', separate_direction=separate_direction)
        self.conv5 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', separate_direction=separate_direction)
        fc_inc = int(np.asarray(patch_size).prod()/512)*4*base_chns
        self.fc1 = nn.Linear(fc_inc, 8 * base_chns)
        self.fc2 = nn.Linear(8 * base_chns, 4 * base_chns)
        self.fc3 = nn.Linear(4 * base_chns, n_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.downsample(conv1)  # 1/2
        conv2 = self.conv2(out)  #
        out = self.downsample(conv2)  # 1/4
        conv3 = self.conv3(out)  #
        out = self.downsample(conv3)  # 1/8
        out = self.conv4(out)
        out = self.conv5(out)

        out = out.view(out.shape[0],-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return {'fc_position':out}
