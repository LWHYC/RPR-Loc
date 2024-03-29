#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.backends.cudnn as cudnn
from dataloaders.Position_detection_dataloader import *
from torch.utils.data import DataLoader
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
import matplotlib
# matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from data_process.data_process_func import *
from prefetch_generator import BackgroundGenerator
from detection.detection_functions import *
import numpy as np


def test(config_file):#, label_wanted):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data = config['data'] 
    config_coarse_pnet = config['coarse_pnetwork']
    config_fine_pnet = config['fine_pnetwork']
    config_test = config['testing']

    patch_size = np.asarray(config_data['patch_size'])
    multi_run_ensemble_num = config_data['multi_run_ensemble_num']
    label_wanted_ls = config_data['label_wanted_ls']
    fine_detection = config_data['fine_detection']

    coarse_dis_ratio = np.array(config_coarse_pnet['distance_ratio'])
    fine_dis_ratio = np.array(config_fine_pnet['distance_ratio'])
    
    random_seed = config_test.get('random_seed', 2)
    random_all(random_seed)
    
     
    random_crop = RandomPositionCrop(patch_size, padding=False)

    cudnn.benchmark = True
    cudnn.deterministic = True

    # 2, load data
    validData = PositionDataloader(config=config_data, \
                                    query_image_list=config_data['query_image_list'],
                                    query_label_list=config_data['query_label_list'],
                                    support_image_list=config_data['support_image_list'],
                                    support_label_list=config_data['support_label_list'],
                                    transform=None,
                                    random_sample = False)
    validLoader = DataLoader(validData, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    support_volume_num = len(read_file_list(config_data['support_image_list']))

    # 3. creat model
    coarse_net_type = config_coarse_pnet['net_type']
    coarse_net_class = NetFactory.create(coarse_net_type)
    fine_net_type = config_fine_pnet['net_type']
    fine_net_class = NetFactory.create(fine_net_type)
    patch_size = np.asarray(config_data['patch_size'])
    Coarse_Pnet = coarse_net_class(
                    inc=config_coarse_pnet.get('input_channel', 1),
                    patch_size = patch_size,
                    base_chns= config_coarse_pnet.get('base_feature_number', 16),
                    norm='in',
                    n_classes = config_coarse_pnet['class_num']
                    )
    Fine_Pnet = fine_net_class(
                    inc=config_fine_pnet.get('input_channel', 1),
                    patch_size = patch_size,
                    base_chns= config_fine_pnet.get('base_feature_number', 16),
                    norm='in',
                    n_classes = config_fine_pnet['class_num']
                    )
    Coarse_Pnet = torch.nn.DataParallel(Coarse_Pnet).half().cuda()
    Fine_Pnet = torch.nn.DataParallel(Fine_Pnet).half().cuda()
    if config_test['load_weight']:
        if os.path.isfile(config_test['coarse_pnet_load_path']):
            print("=> loading checkpoint '{}'".format(config_test['coarse_pnet_load_path']))
            if config_test['coarse_pnet_load_path'].endswith('.tar'):
                coarse_pnet_weight = torch.load(config_test['coarse_pnet_load_path'])  # position net
                Coarse_Pnet.load_state_dict(coarse_pnet_weight['state_dict'])
            elif config_test['coarse_pnet_load_path'].endswith('.pkl'):
                coarse_pnet_weight = torch.load(config_test['coarse_pnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # position net
                Coarse_Pnet.load_state_dict(coarse_pnet_weight)
            print("=> loaded checkpoint '{}' ".format(config_test['coarse_pnet_load_path']))
        else:
            raise(ValueError("=> no checkpoint found at '{}'".format(config_test['coarse_pnet_load_path'])))
        if os.path.isfile(config_test['fine_pnet_load_path']):
            print("=> loading checkpoint '{}'".format(config_test['fine_pnet_load_path']))
            if config_test['fine_pnet_load_path'].endswith('.tar'):
                fine_pnet_weight = torch.load(config_test['fine_pnet_load_path'])  # position net
                Fine_Pnet.load_state_dict(fine_pnet_weight['state_dict'])
            elif config_test['fine_pnet_load_path'].endswith('.pkl'):
                fine_pnet_weight = torch.load(config_test['fine_pnet_load_path'],
                                 map_location=lambda storage, loc: storage)  # position net
                Fine_Pnet.load_state_dict(fine_pnet_weight)
            print("=> loaded checkpoint '{}' ".format(config_test['fine_pnet_load_path']))
        else:
            raise(ValueError("=> no checkpoint found at '{}'".format(config_test['fine_pnet_load_path'])))
    Coarse_Pnet.eval()
    Coarse_RD = Relative_distance(Coarse_Pnet,out_mode='fc_position', distance_mode='tanh', distance_ratio=coarse_dis_ratio)
    Fine_Pnet.eval()
    Fine_RD = Relative_distance(Fine_Pnet,out_mode='fc_position', distance_mode='tanh', distance_ratio=fine_dis_ratio)

    # 4, start to detect
    mean_iou_ls, mean_error_ls = [],[]
    for label_wanted in label_wanted_ls:
        iou_ls, relative_cor_ls, cor_error_ls = [],[],[]
        print('lw', label_wanted)
        save_bad_case = False
        det_time = 0
        with torch.no_grad():
            for idx in range(support_volume_num):
                support_sample = validData.get_support_sample(idx=idx)
                support_sample['label'] = pad(support_sample['label'], patch_size)
                support_sample['label'] = 1*(support_sample['label'] == label_wanted)
                support_sample['image'] = pad(support_sample['image'], patch_size)
                support_extreme_cor = extract_fg_cor(support_sample['label'], 6, erosion_num=False)
                support_image_batch, support_label_batch = [], []
                for i in range(support_extreme_cor.shape[0]):
                    support_cor = support_extreme_cor[i]
                    support_image_batch.append(support_sample['image'][support_cor[0] - patch_size[0] // 2:support_cor[0] + patch_size[0] // 2,
                                support_cor[1] - patch_size[1] // 2:support_cor[1] + patch_size[1] // 2,
                                support_cor[2] - patch_size[2] // 2:support_cor[2] + patch_size[2] // 2][np.newaxis])
                    support_label_batch.append(support_sample['label'][support_cor[0] - patch_size[0] // 2:support_cor[0] + patch_size[0] // 2,
                                support_cor[1] - patch_size[1] // 2:support_cor[1] + patch_size[1] // 2,
                                support_cor[2] - patch_size[2] // 2:support_cor[2] + patch_size[2] // 2][np.newaxis])
                support_image_batch = np.asarray(support_image_batch)
                support_label_batch = torch.tensor(np.array(support_label_batch)).cuda()
                Coarse_RD.cal_support_position(support_image_batch, idx=idx)
                Fine_RD.cal_support_position(support_image_batch, idx=idx)
            Coarse_RD.support_position /= support_volume_num
            Fine_RD.support_position /= support_volume_num
            
            for ii_batch, query_sample in enumerate(validLoader):
                    print(query_sample['image_path'])
                    query_sample['spacing'] = query_sample['spacing'].cpu().data.numpy().squeeze()
                    spacing = query_sample['spacing']
                    query_label = pad(query_sample['label'].cpu().data.numpy().squeeze(), patch_size)
                    query_label = 1*(query_label==label_wanted)
                    ss = query_label.shape
                    predic_extreme_cor = np.zeros([support_extreme_cor.shape[0],3])
                    query_sample['image'] = pad(query_sample['image'].cpu().data.numpy().squeeze(), patch_size)
                    real_corner_cor = extract_fg_cor(query_label,extreme_point_num=2)
                    real_extreme_cor = extract_fg_cor(query_label,extreme_point_num=6)
                    cur_position,predic_position,query_batch = [],[],[]
                    for ii in range(multi_run_ensemble_num): # 
                        sample = random_crop(query_sample)
                        random_position = np.int16(sample['random_position']).squeeze()
                        cur_position.append(random_position)
                        query_batch.append(sample['random_crop_image'][np.newaxis,:])
                    query_batch = np.asarray(query_batch)
                    relative_position = Coarse_RD.cal_RD(query_patch=query_batch, mean=True)['relative_position']
                    cur_position = np.mean(np.asarray(cur_position), axis=0)
                    cur_position = cur_position + relative_position# [6, 3]
                    if fine_detection:
                        fine_query_batch = []
                        cur_cor = np.round(cur_position/spacing).astype(np.int16) 
                        cur_cor[:,0] = np.minimum(np.maximum(cur_cor[:,0], patch_size[0]//2), ss[0]-patch_size[0]//2-1)
                        cur_cor[:,1] = np.minimum(np.maximum(cur_cor[:,1], patch_size[1]//2), ss[1]-patch_size[1]//2-1)
                        cur_cor[:,2] = np.minimum(np.maximum(cur_cor[:,2], patch_size[2]//2), ss[2]-patch_size[2]//2-1)
                        
                        for iii in range(cur_position.shape[0]):
                            fine_query_batch.append(query_sample['image'][
                                        cur_cor[iii,0] - patch_size[0] // 2:cur_cor[iii,0] + patch_size[0] // 2,
                                        cur_cor[iii,1] - patch_size[1] // 2:cur_cor[iii,1] + patch_size[1] // 2,
                                        cur_cor[iii,2] - patch_size[2] // 2:cur_cor[iii,2] + patch_size[2] // 2][np.newaxis])
                        
                        fine_query_batch = np.asarray(fine_query_batch) # [6, 1, D, W, H]
                        relative_position = Fine_RD.cal_RD(query_patch=fine_query_batch)['relative_position']
                        print('rela pos\n', relative_position)
                        relative_cor_ls.append(relative_position)
                        cur_position += relative_position # [6,3]
                    predic_extreme_cor = cur_position.copy()/spacing
                    predic_corner_cor = np.asarray([np.min(predic_extreme_cor,axis=0),np.max(predic_extreme_cor, axis=0)])
                    pred_iou = iou(real_corner_cor,  predic_corner_cor)
                    pred_error = spacing*(real_corner_cor-predic_corner_cor) # 2*3
                    print(ii_batch,'predic iou:',pred_iou, '\npredic error:\n',pred_error, \
                        '\npredic cor:\n',predic_corner_cor,'\nreal cor:\n', real_corner_cor)
                    iou_ls.append(pred_iou)
                    det_time +=1
                    cor_error_ls.append(pred_error)
                    if save_bad_case and pred_iou<0.5:
                        in_predic_extreme_cor = np.int16(np.around(predic_extreme_cor))
                        for i in range(support_extreme_cor.shape[0]):
                            print('Saving img', 'results/{0:}_{1:}_{2:}_f.png'.format(idx, ii_batch, i))
                            support_img = crop(support_sample['image'], patch_size)
                            query_img = crop(query_sample['image'], patch_size)
                            support_point_position = support_extreme_cor[i]-patch_size//2
                            query_point_position = real_extreme_cor[i] - patch_size//2
                            predicted_point_position = in_predic_extreme_cor[i]-patch_size//2
                            save_detection(support_img, query_img, support_point_position,query_point_position, \
                                        predicted_point_position, fname='results/{0:}_{1:}_{2:}_f.png'.format(idx, ii_batch, i))
            mean_iou = np.around(np.array(iou_ls).mean(), decimals=3)
            mean_error = np.around(np.abs(np.array(cor_error_ls)).mean(), decimals=2)
            mean_iou_ls.append(mean_iou)
            mean_error_ls.append(mean_error)
            print(label_wanted, 'mean iou:\n',mean_iou, '\nmean corner error:\n',mean_error)
        
    print(mean_iou_ls, mean_error_ls)
    



if __name__ == '__main__':
    config_file = str(sys.argv[1])
    assert (os.path.isfile(config_file))
    test(config_file) 
