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
    config_data = config['data']  # 包含数据的各种信息,如data_shape,batch_size等
    config_coarse_pnet = config['coarse_pnetwork']
    coarse_dis_ratio = np.array(config_coarse_pnet['distance_ratio'])
    config_fine_pnet = config['fine_pnetwork']
    fine_dis_ratio = np.array(config_fine_pnet['distance_ratio'])
    config_test = config['testing']
    patch_size = np.asarray(config_data['patch_size'])
    max_scale = patch_size
    random_seed = config_test.get('random_seed', 2)
    random_all(random_seed)  # 给定seed value,决定了后面的伪随机序列
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
    volume_num = len(read_file_list(config_data['query_image_list']))

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
    for ognb in [1]:
        iou_ls, relative_cor_ls, cor_error_ls = [],[],[]
        label_wanted = ognb
        print('lw', label_wanted)
        show = False
        det_time = 0
        error_sum = np.zeros([2, 3])
        ex_error_sum = np.zeros([6, 3])
        iter_patch_num = 10
        iter_move_num = [1,0]
        with torch.no_grad():
            for idx in range(10):
                support_sample = validData.get_support_sample(idx=idx)
                support_sample['label'] = pad(support_sample['label'], max_scale)
                support_sample['label'] = 1*(support_sample['label'] == label_wanted)
                support_sample['image'] = pad(support_sample['image'], max_scale)
                support_extreme_cor = extract_fg_cor(support_sample['label'], 6, erosion_num=False)
                support_image_batch, support_label_batch = [], []
                for i in range(support_extreme_cor.shape[0]): #分别裁减几个support极端点所在patch，预测其坐标
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
            Coarse_RD.support_position /= 10
            Fine_RD.support_position /= 10
            
            for ii_batch, query_sample in enumerate(validLoader):
                #try:
                    print(query_sample['image_path'])
                    query_sample['spacing'] = query_sample['spacing'].cpu().data.numpy().squeeze()
                    spacing = query_sample['spacing']
                    query_label = pad(query_sample['label'].cpu().data.numpy().squeeze(), max_scale)
                    query_label = 1*(query_label==label_wanted)
                    ss = query_label.shape
                    predic_extreme_cor = np.zeros([support_extreme_cor.shape[0],3])
                    query_sample['image'] = pad(query_sample['image'].cpu().data.numpy().squeeze(), max_scale)
                    real_corner_cor = extract_fg_cor(query_label,extreme_point_num=2)
                    real_extreme_cor = extract_fg_cor(query_label,extreme_point_num=6)
                    cur_position,predic_position,query_batch = [],[],[]
                    for ii in range(iter_patch_num): # 多次随机裁减预测距离，最终取平均
                        sample = random_crop(query_sample)
                        random_position = np.int16(sample['random_position']).squeeze()
                        cur_position.append(random_position)
                        query_batch.append(sample['random_crop_image'][np.newaxis,:])
                    query_batch = np.asarray(query_batch)
                    relative_position = Coarse_RD.cal_RD(query_patch=query_batch, mean=True)['relative_position']
                    cur_position = np.mean(np.asarray(cur_position), axis=0)
                    cur_position = cur_position + relative_position# [6, 3]
                    for move_step in range(iter_move_num[0]):
                        fine_query_batch = []
                        cur_cor = np.round(cur_position/spacing).astype(np.int16) #像素坐标
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
                    #predic_corner_cor = transfer_multi_extremepoint_to_cornerpoint(predic_extreme_cor, point_per_slice=1, slice_range=1)
                    predic_corner_cor = np.asarray([np.min(predic_extreme_cor,axis=0),np.max(predic_extreme_cor, axis=0)])
                    pred_iou = iou(real_corner_cor,  predic_corner_cor)
                    pred_error = spacing*(real_corner_cor-predic_corner_cor) # 2*3
                    print(ii_batch,'predic iou:',pred_iou, '\npredic error:\n',pred_error, \
                        '\npredic cor:\n',predic_corner_cor,'\nreal cor:\n', real_corner_cor)
                    iou_ls.append(pred_iou)
                    det_time +=1
                    cor_error_ls.append(pred_error)
                    if show and pred_iou<0.5:
                        in_predic_extreme_cor = np.int16(np.around(predic_extreme_cor))
                        for i in range(support_extreme_cor.shape[0]):
                            print('Saving img', 'results/{0:}_{1:}_{2:}_f.png'.format(idx, ii_batch, i))
                            support_img = crop(support_sample['image'], max_scale)
                            query_img = crop(query_sample['image'], max_scale)
                            support_point_position = support_extreme_cor[i]-max_scale//2
                            query_point_position = real_extreme_cor[i] - max_scale//2
                            predicted_point_position = in_predic_extreme_cor[i]-max_scale//2
                            save_detection(support_img, query_img, support_point_position,query_point_position, \
                                        predicted_point_position, fname='results/{0:}_{1:}_{2:}_f.png'.format(idx, ii_batch, i))
                # except:
                #     print('Something goes wrong with:', query_sample['image_path'],'in class', ognb)
                #     pass
                    
            # error_dis[-1]=np.asarray(error_dis[-1])
            # fig,(ax0,ax1,ax2, ax3,ax4,ax5) = plt.subplots(nrows=6,figsize=(9,6)) 
            # ax0.hist(error_dis[-1][:,0,0],10,histtype='bar',facecolor='yellowgreen',alpha=0.75)
            # ax1.hist(error_dis[-1][:,0,1],10,histtype='bar',facecolor='pink',alpha=0.75)
            # ax2.hist(error_dis[-1][:,0,2],10,histtype='bar',facecolor='red',alpha=0.75)
            # ax3.hist(error_dis[-1][:,1,0],10,histtype='bar',facecolor='yellowgreen',alpha=0.75)
            # ax4.hist(error_dis[-1][:,1,1],10,histtype='bar',facecolor='pink',alpha=0.75)
            # ax5.hist(error_dis[-1][:,1,2],10,histtype='bar',facecolor='red',alpha=0.75)       
            # fig.subplots_adjust(hspace=0.4)  
            # plt.show()
            mean_iou = np.around(np.array(iou_ls).mean(), decimals=3)
            mean_error = np.around(np.abs(np.array(cor_error_ls)).mean(), decimals=2)
            mean_iou_ls.append(mean_iou)
            mean_error_ls.append(mean_error)
            print(label_wanted, 'mean iou:\n',mean_iou, '\nmean corner error:\n',mean_error)
        
    print(mean_iou_ls, mean_error_ls)
    



if __name__ == '__main__':
    config_file = str('/home/ps/leiwenhui/Code/RPR-Loc-main/config/test/test_c2f_pancreas_detection.txt')
    assert (os.path.isfile(config_file))
    test(config_file) 
