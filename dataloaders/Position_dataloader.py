#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import trange
import random
from data_process.data_process_func import *

class PositionDataloader(Dataset):
    def __init__(self, config=None, image_list=None, num=None, transform=None, 
                random_sample=True, load_memory=True):
        self._iternum = config['iter_num']
        self.transform = transform
        self.image_dic = {}
        self.load_memory = load_memory
        self.random_sample = random_sample
        self.image_list = read_file_list(image_list)
        if load_memory:
            for i in trange(len(self.image_list)):
                image_path = self.image_list[i]
                image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
                spacing = np.asarray(spacing)
                sample = {'image': image.astype(np.float16), 'spacing':spacing,'image_path':image_path}
                self.image_dic[image_path]=sample
        
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        if self.random_sample:
            return self._iternum
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        if self.load_memory:
            sample = self.image_dic[random.sample(self.image_list, 1)[0]]
        else:
            if self.random_sample:
                image_path = random.sample(self.image_list, 1)[0]
            else:
                image_path = self.image_list[idx]
            image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
            spacing = np.asarray(spacing)
            sample = {'image': image.astype(np.float16), 'spacing':spacing,'image_path':image_path}

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
        image,spacing = sample['image'],sample['spacing']
        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        if self.padding:
            image = np.pad(image, [(self.output_size[0]//2, self.output_size[0]//2), (self.output_size[1]//2, 
                self.output_size[1]//2), (self.output_size[2]//2, self.output_size[2]//2)], mode='constant', constant_values=0)

        shape = list(image.shape)
        image_n = image.copy()

        background_chosen = True
        shape_n = image_n.shape
        while background_chosen:
            random_pos0 = self.random_position(shape_n)
            if image_n[random_pos0[0] + self.output_size[0]//2, random_pos0[1] + self.output_size[1]//2, random_pos0[2] + self.output_size[2]//2]!=0:
                background_chosen = False
        sample['random_crop_image_0']=image_n[random_pos0[0]:random_pos0[0] + self.output_size[0],
                                                    random_pos0[1]:random_pos0[1] + self.output_size[1],
                                                    random_pos0[2]:random_pos0[2] + self.output_size[2]]
        sample['random_position_0'] = random_pos0

        background_chosen = True
        while background_chosen:
            random_pos1= self.random_position(shape_n,sample['random_position_0'],fluct_range=self.fluct_range, small_move=self.small_move)
            if image[random_pos1[0] + self.output_size[0] // 2, random_pos1[1] + self.output_size[1] // 2,
                     random_pos1[2] + self.output_size[2] // 2] != 0:
                background_chosen = False
        sample['random_crop_image_1'] = image[random_pos1[0]:random_pos1[0] + self.output_size[0],
                                                     random_pos1[1]:random_pos1[1] + self.output_size[1],
                                                     random_pos1[2]:random_pos1[2] + self.output_size[2]]
        sample['random_position_1'] = random_pos1

        spacing = np.asarray(spacing).squeeze()
        sample['rela_distance']=(sample['random_position_0']-sample['random_position_1']).squeeze()*spacing
        return sample



class ToPositionTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        nsample = {}
        for key in sample.keys():
            if 'random_crop_image' in key:
                nsample[key] = torch.from_numpy(sample[key][np.newaxis].astype(np.float32))
            elif 'rela_distance' in key:
                nsample[key]=torch.from_numpy(sample[key]).float()
        return nsample

