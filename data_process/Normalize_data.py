#!/usr/bin/env python
import os
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from data_process.data_process_func import *
import SimpleITK as sitk

data_root = '../../../Data/HaN/'
filename = 'rdata.nii.gz'
savename = 'norm_rdata.nii.gz'
modelist = ['train','valid','test']
save_as_nifty = True
norm = True
thresh_lis = [-500,-100,100, 1500]
norm_lis = [0, 0.2,0.8,1]
normalize = img_multi_thresh_normalized


for mode in modelist:
    filelist = os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for ii in range(filenum):
        data_path = os.path.join(data_root, mode, filelist[ii], filename)
        data_norm_save_path = os.path.join(data_root, mode, filelist[ii], savename)
        data = sitk.ReadImage(data_path)
        img = sitk.GetArrayFromImage(data)
        img = normalize(img,thresh_lis=thresh_lis, norm_lis=norm_lis)
        ndata = sitk.GetImageFromArray(img)
        ndata.SetSpacing(data.GetSpacing())
        ndata.SetOrigin(data.GetOrigin())
        ndata.SetDirection(data.GetDirection())
        sitk.WriteImage(ndata, data_norm_save_path)

        print('Successfully process', filelist[ii])