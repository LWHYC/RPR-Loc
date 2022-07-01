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
    def __init__(self, config=None, query_image_list=None, query_label_list=None,
                support_image_list=None, support_label_list=None, 
                transform=None, random_sample=True):
        self.transform = transform
        self.image_dic = {}
        self.random_sample = random_sample
        self.query_image_list = read_file_list(query_image_list)
        self.query_label_list = read_file_list(query_label_list)
        self.support_image_list = read_file_list(support_image_list)
        self.support_label_list = read_file_list(support_label_list)

        print("total {} samples".format(len(self.query_image_list)))

    def __len__(self):
        return len(self.query_image_list)

    def get_support_sample(self, idx):
        image_path = self.support_image_list[idx]
        image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
        label = load_nifty_volume_as_array(self.support_label_list[idx])
        spacing = np.asarray(spacing)
        sample = {'image': image.astype(np.float16), 'label':label, 'spacing':spacing, 'image_path':image_path}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        if self.random_sample:
            image_path = random.sample(self.query_image_list, 1)[0]
        else:
            image_path = self.query_image_list[idx]
        image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
        label = load_nifty_volume_as_array(self.query_label_list[idx])
        spacing = np.asarray(spacing)
        sample = {'image': image.astype(np.float16), 'label':label, 'spacing':spacing,'image_path':image_path}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomPositionCrop(object):
    """
    Randomly crop an image in one sample;
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, padding=True, foreground_only=True, small_move=False, fluct_range=[0,0,0]):
        self.output_size = np.array(output_size)
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
                position.append(np.random.randint(0, shape[i] - self.output_size[i]))
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

        background_chosen = True
        shape_n = image.shape
        while background_chosen:
            random_pos = self.random_position(shape_n)
            if image[random_pos[0] + self.output_size[0]//2, random_pos[1] + self.output_size[1]//2, random_pos[2] + self.output_size[2]//2]!=0:
                background_chosen = False
        sample['random_crop_image']=image[random_pos[0]:random_pos[0] + self.output_size[0],
                                                    random_pos[1]:random_pos[1] + self.output_size[1],
                                                    random_pos[2]:random_pos[2] + self.output_size[2]]
        sample['random_position'] = (random_pos+self.output_size//2)*spacing
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

