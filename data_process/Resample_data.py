#!/usr/bin/env python
from __future__ import absolute_import, print_function
import os
import sys
import argparse
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data_process.data_process_func import * 
from multiprocessing import Pool
import os
import SimpleITK as sitk
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage import measure
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




def crop_and_resample(data_save_path, label_save_path, data_path, label_path, class_num, target_spacing):
    data = sitk.ReadImage(data_path)
    img = sitk.GetArrayFromImage(data).astype(np.float32)
    label = sitk.ReadImage(label_path)
    mask = sitk.GetArrayFromImage(label).astype(np.int8)
    binary = img>-600
    [minc, maxc]=get_bound_coordinate(binary)
    print(minc, maxc)
    img = img[minc[0]:maxc[0],minc[1]:maxc[1],minc[2]:maxc[2]]
    mask = mask[minc[0]:maxc[0],minc[1]:maxc[1],minc[2]:maxc[2]]
    zoom_factor = np.array(data.GetSpacing())/np.array(target_spacing)
    zoom_img = resize_ND_volume_to_given_shape(img, zoom_factor, order=1)
    zoom_mask = resize_Multi_label_to_given_shape(mask, zoom_factor, class_num=class_num)
    zoom_data = sitk.GetImageFromArray(img)
    zoom_data.SetSpacing(data.GetSpacing())
    zoom_data.SetOrigin(data.GetOrigin())
    zoom_data.SetDirection(data.GetDirection())
    sitk.WriteImage(zoom_data, data_save_path)
    zoom_label = sitk.GetImageFromArray(mask)
    zoom_label.SetSpacing(data.GetSpacing())
    zoom_label.SetOrigin(data.GetOrigin())
    zoom_label.SetDirection(data.GetDirection())
    sitk.WriteImage(zoom_label, label_save_path)
    return


data_root = '../../../Data/HaN'
imgname = 'data.nii.gz'
img_save_name = 'rdata.nii.gz'
labelname = 'label.nii.gz'
label_save_name = 'rlabel.nii.gz'
modelist = ['train', 'valid', 'test']
target_spacing = [3,1,1]
class_num = 8   # total class num of label, include background
for mode in modelist:
    filelist =os.listdir(os.path.join(data_root, mode))
    for file in filelist:
        data_save_path = os.path.join(data_root, mode, file, img_save_name)
        data_path = os.path.join(data_root, mode, file, imgname)
        label_save_path = os.path.join(data_root, mode, file, label_save_name)
        label_path = os.path.join(data_root, mode, file, labelname)
        crop_and_resample(data_save_path, label_save_path, data_path, label_path, class_num, target_spacing)
        print('Successfully resample', file)
print('---done!')