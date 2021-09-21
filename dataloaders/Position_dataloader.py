#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import numbers
from scipy import ndimage
from glob import glob
from torch.utils.data import Dataset
import tqdm
from tqdm import trange
import random
import h5py
import itertools
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data.sampler import Sampler
from data_process.data_process_func import *

class PositionDataloader(Dataset):
    def __init__(self, config=None, split='train', num=None, transform=None, 
                random_sample=True, load_aug=False, load_memory=True):
        self._data_root = config['data_root']
        self._image_filename = config['image_name']
        self._iternum = config['iter_num']
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.image_dic = {}
        self.iternum = 0
        self.load_aug = load_aug
        self.load_memory = load_memory
        self.random_sample = random_sample
        self.image_name_list = os.listdir(os.path.join(self._data_root,split))
        if load_memory:
            for i in trange(len(self.image_name_list)):
                image_name = self.image_name_list[i]
                if self._image_filename is None:
                    image_path = os.path.join(self._data_root, self.split, image_name)
                else:
                    image_path = os.path.join(self._data_root, self.split, image_name, self._image_filename)
                image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
                spacing = np.asarray(spacing)
                shape = list(image.shape)
                cor_x,cor_z,cor_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # the cor map of z,x,y
                cor_z,cor_x,cor_y = cor_z.astype(np.float32),cor_x.astype(np.float32),cor_y.astype(np.float32)
                cor = np.concatenate((cor_z[np.newaxis,:], cor_x[np.newaxis,:], cor_y[np.newaxis,:]), axis=0)
                sample = {'image': image.astype(np.float16), 'coordinate':cor, 'spacing':spacing,'image_path':image_path}
                self.image_dic[image_name]=sample
        
        if num is not None:
            self.image_name_list = self.image_name_list[:num]
        print("total {} samples".format(len(self.image_name_list)))

    def __len__(self):
        if self.random_sample:
            return self._iternum
        else:
            return len(self.image_name_list)

    def __getitem__(self, idx):
        if self.load_memory:
            sample = self.image_dic[random.sample(self.image_name_list, 1)[0]]
        else:
            if self.random_sample:
                image_name = random.sample(self.image_name_list, 1)[0]
            else:
                image_name = self.image_name_list[idx]
            if self._image_filename is None:
                image_path = os.path.join(self._data_root, self.split, image_name)
            else:
                image_path = os.path.join(self._data_root, self.split, image_name, self._image_filename)
            image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
            spacing = np.asarray(spacing)
            shape = list(image.shape)
            cor_x,cor_z,cor_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  
            cor_z,cor_x,cor_y = cor_z.astype(np.float32),cor_x.astype(np.float32),cor_y.astype(np.float32)
            cor = np.concatenate((cor_z[np.newaxis,:], cor_x[np.newaxis,:], cor_y[np.newaxis,:]), axis=0)
            sample = {'image': image.astype(np.float16), 'coordinate':cor, 'spacing':spacing,'image_path':image_path}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomPositionDoubleCrop(object):
    """
    Randomly crop several images in one sample;
    distance is a vector(could be positive or pasitive), representing the offset
    from image1 to image2.
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, padding=True, foreground_only=True, small_move=False, fluct_range=[0,0,0]):
        self.output_size = output_size
        self.padding = padding
        self.foregroung_only = foreground_only
        self.fluct_range=fluct_range
        self.small_move = small_move
    def random_position(self, shape, initial_position=[0,0,0], fluct_range=[5, 10,10], small_move=False):
        position = []
        for i in range(len(shape)):
            if small_move:
                position.append(np.random.randint(max(0, initial_position[i]-fluct_range[i]),
                                                  min(shape[i] - self.output_size[i], initial_position[i]+fluct_range[i])))
            else:
                position.append(
                    np.random.randint(0, shape[i] - self.output_size[i]))
        return np.asarray(position)


    def __call__(self, sample):
        image,spacing, cor= sample['image'],sample['spacing'], sample['coordinate']
        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        rela_position = np.zeros(3)
        if self.padding:
            image = np.pad(image, [(self.output_size[0]//2, self.output_size[0]//2), (self.output_size[1]//2, 
                self.output_size[1]//2), (self.output_size[2]//2, self.output_size[2]//2)], mode='constant', constant_values=0)

        shape = list(image.shape)
        image_n = image.copy()
        cor_n = cor.copy()

        background_chosen = True
        shape_n = image_n.shape
        while background_chosen:
            random_pos0 = self.random_position(shape_n)
            if image_n[random_pos0[0] + self.output_size[0]//2, random_pos0[1] + self.output_size[1]//2, random_pos0[2] + self.output_size[2]//2]!=0:
                background_chosen = False
        sample['random_crop_image_0']=image_n[random_pos0[0]:random_pos0[0] + self.output_size[0],
                                                    random_pos0[1]:random_pos0[1] + self.output_size[1],
                                                    random_pos0[2]:random_pos0[2] + self.output_size[2]]
        sample['random_position_0'] = cor_n[:,random_pos0[0],random_pos0[1],random_pos0[2]]
        sample['random_fullsize_position_0'] = cor_n[:, random_pos0[0]:random_pos0[0] + self.output_size[0],
                                                    random_pos0[1]:random_pos0[1] + self.output_size[1],
                                                    random_pos0[2]:random_pos0[2] + self.output_size[2]]


        cor = np.concatenate((cor_z[np.newaxis,:], cor_x[np.newaxis,:], cor_y[np.newaxis,:]), axis=0)
        background_chosen = True
        shape = image.shape
        while background_chosen:
            random_pos1= self.random_position(shape,sample['random_position_0'],fluct_range=self.fluct_range, small_move=self.small_move)
            if image[random_pos1[0] + self.output_size[0] // 2, random_pos1[1] + self.output_size[1] // 2,
                     random_pos1[2] + self.output_size[2] // 2] != 0:
                background_chosen = False
        sample['random_crop_image_1'] = image[random_pos1[0]:random_pos1[0] + self.output_size[0],
                                                     random_pos1[1]:random_pos1[1] + self.output_size[1],
                                                     random_pos1[2]:random_pos1[2] + self.output_size[2]]
        sample['random_position_1'] = cor[:, random_pos1[0],random_pos1[1],random_pos1[2]]
        sample['random_fullsize_position_1'] = cor[:, random_pos1[0]:random_pos1[0] + self.output_size[0],
                                                     random_pos1[1]:random_pos1[1] + self.output_size[1],
                                                     random_pos1[2]:random_pos1[2] + self.output_size[2]]

        for i in range(len(rela_position)):
            if sample['random_position_0'][i]<sample['random_position_1'][i]:
                rela_position[i] = 1
        spacing = np.asarray(spacing).squeeze()
        sample['rela_distance']=(sample['random_position_0']-sample['random_position_1'])*spacing
        sample['rela_fullsize_distance'] = (sample['random_fullsize_position_0']-sample['random_fullsize_position_1'])*spacing[:,np.newaxis,np.newaxis,np.newaxis]
        sample['rela_poi']=rela_position
        return sample



class ToPositionTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        nsample = {}
        key_float_list = ['rela_distance','rela_fullsize_distance','rela_poi','random_position','random_fullsize_position']
        for key in sample.keys():
            if 'random_crop_image' in key or 'random_crop_aug_image' in key:
                image = sample[key]
                image= image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
                nsample[key] = torch.from_numpy(image)
            else:
                for key_float in  key_float_list:
                    if key_float in key:
                        nsample[key]=torch.from_numpy(sample[key]).float()
        return nsample

