#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from os.path import abspath, join, dirname
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, join(abspath(dirname(__file__)), 'src'))
from scipy import ndimage
import  numpy as np
import matplotlib
from data_process.data_process_func import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import morphology

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

class Relative_distance(object):
    def __init__(self,network, out_mode='fc_position', distance_mode='linear', distance_ratio=100):
        self.network = network
        self.distance_ratio = distance_ratio
        self.distance_mode = distance_mode
        self.out_mode = out_mode
    def cal_support_position(self, support_patch, idx=0):
        '''
        support_patch: [b*1*d*w*h] 
        '''
        self.support_patch = support_patch
        self.full_shape = support_patch.squeeze().shape # 6, D, W, H
        self.support_all = self.network(torch.from_numpy(support_patch).float().half())
        if idx ==0:
            self.support_position = self.support_all[self.out_mode].cpu().numpy()
        else:
            self.support_position += self.support_all[self.out_mode].cpu().numpy()
    
      
    def cal_RD(self, query_patch, mean=False):
        '''
        query_patch:[b*1*d*w*h]
        '''
        result = {}
        query_all = self.network(torch.from_numpy(query_patch).float().half())
        quer_position = query_all[self.out_mode].cpu().numpy()# [b, 3]
        if mean:
            quer_position = np.mean(quer_position, axis=0)
        if self.distance_mode=='linear':
            relative_position = self.distance_ratio*(self.support_position-quer_position)
        elif self.distance_mode=='tanh':
            relative_position = self.distance_ratio*np.tanh(self.support_position-quer_position)
        else:
            raise ValueError('Please select a correct distance mode!!!')
        result['relative_position']=relative_position
        return result


def random_all(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def get_center_cor(img):
    '''
    get 2d binary img center corardiate
    :param img: 2d binary img
    :return:
    '''
    mask = np.nonzero(img)
    half = len(mask[0])//2
    center_x = np.asarray(mask[0][half:half+1])
    center_y = np.asarray(mask[1][half:half+1])
    # center_x = np.round(np.mean(np.asarray(mask[0])))
    # center_y = np.round(np.mean(np.asarray(mask[1])))
    return center_x, center_y


def extract_fg_cor(label, extreme_point_num=2, erosion_num=False, pad=[0, 0, 0]):
    if erosion_num:
        label= morphology.binary_erosion(label, structure=np.ones([1,1,1]), iterations=erosion_num)
    extre_cor = get_bound_coordinate(label, pad=pad) #[minpoint, maxpoint]
    if extreme_point_num==2:
        return np.asarray(extre_cor)
    elif extreme_point_num==6:
        real_extre_point = np.zeros([6,3])
        for i in range(len(extre_cor)):
            for ii in range(len(extre_cor[i])):
                slice_label = label.transpose(ii,ii-2,ii-1)[extre_cor[i][ii]]
                #(center_x, center_y) = ndimage.center_of_mass(slice_label)
                center_x, center_y = get_center_cor(img=slice_label)
                cor=np.zeros(3)
                cor[ii]=extre_cor[i][ii]
                cor[ii-2]=center_x
                cor[ii-1]=center_y
                real_extre_point[i*3+ii] = cor
        return np.int16(real_extre_point)



def save_detection(support_img, query_img, support_point_position, query_point_position, predicted_point_position, fname):
    plt.figure(figsize=(16, 6))
    plt.subplot(131)
    plt.imshow(support_img[support_point_position[0]].astype(np.float32), cmap='gray')
    plt.plot(support_point_position[2], support_point_position[1], '*', c='r')
    plt.title('support position {0:}'.format(support_point_position))
    print('Saving ...', fname)
    plt.subplot(132)
    plt.imshow(query_img[query_point_position[0]].astype(np.float32), cmap='gray')
    plt.plot(query_point_position[2], query_point_position[1], '*', c='r')
    plt.title('query position {0:}'.format(query_point_position))
    plt.subplot(133)
    plt.imshow(query_img[predicted_point_position[0]].astype(np.float32), cmap='gray')
    plt.plot(predicted_point_position[2], predicted_point_position[1], '*', c='r')
    plt.title('detected position {0:}'.format(predicted_point_position))
    plt.savefig(fname)
    plt.close()

def pad(img, pad_size):
    img = np.pad(img, [(pad_size[0] // 2, pad_size[0] // 2), (pad_size[1] // 2, pad_size[1] // 2),
                (pad_size[2] // 2, pad_size[2] // 2)], mode='constant', constant_values=0)
    return img

def crop(img, pad_size):
    img = img[ pad_size[0] // 2: -pad_size[0] // 2, pad_size[1] // 2: -pad_size[1] // 2,
                pad_size[2] // 2: -pad_size[2] // 2]
    return img

def iou(box1, box2):
    'compute 3D iou,box=[h_min,w_min,d_min,h_max,w_max,d_max]'
    box1 = np.asarray(box1).reshape([-1,1])
    box2 = np.asarray(box2).reshape([-1,1])
    in_h = min(box1[3], box2[3]) - max(box1[0], box2[0])
    in_w = min(box1[4], box2[4]) - max(box1[1], box2[1])
    in_d =min(box1[5], box2[5]) - max(box1[2], box2[2])
    inter = 0 if in_h<0 or in_w<0 or in_d<0 else in_h*in_w*in_d
    union = (box1[3] - box1[0]) * (box1[4] - box1[1])*(box1[5] - box1[2]) + \
            (box2[3] - box2[0]) * (box2[4] - box2[1])*(box2[5] - box2[2]) - inter
    iou = inter / union
    return iou
