#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
from Pnet import Pnet
from Pnet_2 import Pnet_2
from Pnet_3 import Pnet_3

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'Pnet':
            return Pnet

        if name == 'Pnet_2':
            return Pnet_2
        
        if name == 'Pnet_3':
            return Pnet_3
            
        print('unsupported network:', name)
        exit()
