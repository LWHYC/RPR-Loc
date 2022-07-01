#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from os.path import abspath, join, dirname
sys.path.append(os.path.abspath(__file__))  #返回当前.py文件的绝对路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   #当前文件的绝对路径目录，不包括当前 *.py 部分，即只到该文件目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, join(abspath(dirname(__file__)), 'src'))
import scipy
from scipy import ndimage
import  numpy as np
import matplotlib
matplotlib.use('WebAgg')
from numpy.lib.function_base import extract
from data_process.data_process_func import *
import cv2
import torch
import torch.nn as nn
import time
from sklearn.metrics.pairwise import cosine_similarity
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
    random.seed(random_seed)  # 给定seed value,决定了后面的伪随机序列
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def move(image, inher_position, test_patch_size):
    cur_position = np.asarray(image.shape)*np.asarray(inher_position)
    cur_position = cur_position.astype(np.int16)
    cur_image_patch = image[cur_position[0]:cur_position[0]+test_patch_size[0], cur_position[1]:cur_position[1]+test_patch_size[1],
                      cur_position[2]:cur_position[2]+test_patch_size[2]]
    return cur_image_patch

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

def get_random_cor(img):
    '''
    randomly select one point from 2d binary img and return its coordinate
    :param img: 2d binary img
    :return:
    '''
    mask = np.nonzero(img)
    random_cor = random.randint(0, len(mask[0])-1)
    cor_x = np.asarray(mask[0][random_cor:random_cor+1])
    cor_y = np.asarray(mask[1][random_cor:random_cor+1])
    # cor_x = np.round(np.mean(np.asarray(mask[0])))
    # cor_y = np.round(np.mean(np.asarray(mask[1])))
    return cor_x, cor_y

def transfer_extremepoint_to_cornerpoint(extremepoint):
    cornerpoint = [[],[]]
    for i in range(3):  
        cornerpoint[0].append(extremepoint[i, i])
        cornerpoint[1].append(extremepoint[3+i, i]) 
    return np.asarray(cornerpoint)

def transfer_multi_extremepoint_to_cornerpoint(extremepoint, point_per_slice, slice_range):
    '''
    extremepoint: min, max
    '''
    num = extremepoint.shape[0]
    cor_offset = np.arange(slice_range).repeat(point_per_slice)
    cornerpoint = [[],[]]
    for i in range(3):
        min_p = np.median(extremepoint[i*num//6: (i+1)*num//6, i]-cor_offset)
        max_p = np.median(extremepoint[(i+3)*num//6: (i+4)*num//6, i]+cor_offset)
        cornerpoint[0].append(min_p)
        cornerpoint[1].append(max_p)
    return np.asarray(cornerpoint)


def crop_patch_around_center(img, r):
    '''
    img: array, c*w*h*d
    r: list
    crop a patch around the center point with shape r
    '''
    if len(img.shape)==4:
        shape = img.shape[1::]
        patch = img[:, shape[0]//2-r[0]:shape[0]//2+r[0], shape[1]//2-r[1]:shape[1]//2+r[1], shape[2]//2-r[2]:shape[2]//2+r[2]]
    elif len(img.shape)==3:
        shape = img.shape
        patch = img[shape[0]//2-r[0]:shape[0]//2+r[0], shape[1]//2-r[1]:shape[1]//2+r[1], shape[2]//2-r[2]:shape[2]//2+r[2]]
    elif len(img.shape)==5:
        shape = img.shape[2::]
        patch = img[:, :, shape[0]//2-r[0]:shape[0]//2+r[0], shape[1]//2-r[1]:shape[1]//2+r[1], shape[2]//2-r[2]:shape[2]//2+r[2]]
    return patch

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

def select_extreme_support_cor(label, point_per_slice=2, erosion_num=False, slice_range=2):
    if erosion_num:
        label= morphology.binary_erosion(label, structure=np.ones([1,1,1]), iterations=erosion_num)
    extre_cor = get_bound_coordinate(label) #[minpoint, maxpoint]
    support_cor = []
    for i in range(2): # 2
        for ii in range(3): # 3
            for iii in range(slice_range):
                slice_label = label.transpose(ii,ii-2,ii-1)[extre_cor[i][ii]+iii*(-1)**i]
                #(center_x, center_y) = ndimage.center_of_mass(slice_label)
                for _ in range(point_per_slice):
                    random_x, random_y = get_random_cor(img=slice_label)
                    cor=np.zeros(3)
                    cor[ii]=extre_cor[i][ii]+iii*(-1)**i
                    cor[ii-2]=random_x
                    cor[ii-1]=random_y
                    support_cor.append(cor)
    return np.array(support_cor, dtype=np.int16)

def cal_average_except_minmax(predicted_point_position, extract_m = False):
    

    predicted_point_position = np.asarray(predicted_point_position)
    position_each_axis = [predicted_point_position[:,i].tolist() for i in range(predicted_point_position.shape[1])]
    n = len(position_each_axis[0])//10
    for i in range(len(position_each_axis)):
        position_each_axis[i].sort()
        if not extract_m == False:
            position_each_axis[i] = position_each_axis[i][:-n]
            position_each_axis[i] = position_each_axis[i][n:]
        position_each_axis[i] = np.mean(position_each_axis[i])
    return np.asarray(position_each_axis)


def expand_cor_if_nessary(corner_cor, patch_size):
    w,h,d = corner_cor[1]-corner_cor[0]
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        corner_cor[0]-=np.asarray([wl_pad,hl_pad, dl_pad])
        corner_cor[1]+=np.asarray([wr_pad,hr_pad, dr_pad])  
    return corner_cor


def show_detection(sample_batch, support_point_position, query_point_position,predicted_point_position):
    plt.figure(figsize=(20, 20))
    plt.subplot(131)
    plt.imshow(sample_batch['image_1'][support_point_position[0]].astype(np.float32), cmap='gray')
    plt.plot(support_point_position[2], support_point_position[1], '*', c='r')
    plt.title('support position')
    plt.subplot(132)
    plt.imshow(sample_batch['image_0'][query_point_position[0]].astype(np.float32), cmap='gray')
    plt.plot(query_point_position[2], query_point_position[1], '*', c='r')
    plt.title('query position')
    plt.subplot(133)
    plt.imshow(sample_batch['image_0'][predicted_point_position[0]].astype(np.float32), cmap='gray')
    plt.plot(predicted_point_position[2], predicted_point_position[1], '*', c='r')
    plt.title('detected position')
    plt.show()
    plt.close()

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
    '计算三维iou,box=[h_min,w_min,d_min,h_max,w_max,d_max]'
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

def sliding_window(image, stepSize, windowSize):
    pd, ph, pw = windowSize[0]//2, windowSize[1]//2, windowSize[2]//2
    pad_size = ((pd, pd), (ph, ph), (pw, pw))
    image_pad = np.pad(image, pad_size, mode='constant', constant_values=0)
    # slide a window across the image
    for z in range(0, image.shape[0], stepSize[0]):
        for y in range(0, image.shape[1], stepSize[1]):
            for x in range(0, image.shape[2], stepSize[2]):
                # yield the current window
                yield (z, y, x, image_pad[z:z + windowSize[0], y:y + windowSize[1], x:x + windowSize[2]])

def crop_sliding_window(image, stepSize, windowSize, cur_position):
    # pd, ph, pw = windowSize[0]//2, windowSize[1]//2, windowSize[2]//2
    # pad_size = ((pd, pd), (ph, ph), (pw, pw))
    # image_pad = np.pad(image, pad_size, mode='constant', constant_values=0)
    # slide a window across the image
    for z in range(max(windowSize[0]//2, cur_position[0]-5), min(image.shape[0]-windowSize[0]//2, cur_position[0]+5), stepSize):
        for y in range(max(windowSize[1]//2, cur_position[1]-10), min(image.shape[1]-windowSize[1]//2, cur_position[1]+10), stepSize):
            for x in range(max(windowSize[2]//2, cur_position[2]-10), min(image.shape[2]-windowSize[2]//2, cur_position[2]+10), stepSize):
                # yield the current window
                yield (z, y, x, image[z- windowSize[0]//2:z + windowSize[0]//2,
                                y- windowSize[1]//2:y + windowSize[1]//2,
                                x- windowSize[2]//2:x + windowSize[2]//2])

def image_similarity(org_img, pred_img):
    assert org_img.shape == pred_img.shape
    score = np.square(np.subtract(org_img, pred_img)).mean()
    return score

def image_similarity_torch(org_img, pred_img):
    assert org_img.size() == pred_img.size()
    score = torch.mean(torch.square((org_img - pred_img))).cpu()
    return score.numpy()


def get_coords(score_map):
    '''
    get x, y and z coordinate from scoremap
    '''
    sm_d, sm_h, sm_w = score_map.shape[0], score_map.shape[1], score_map.shape[2]

    # sm_coor = np.argmin(score_map)
    sm_coor = np.argmax(score_map)
    # print(sm_coor)
    z = sm_coor // (sm_h*sm_w)
    y = (sm_coor % (sm_h*sm_w)) // sm_w
    x = (sm_coor % (sm_h*sm_w)) % sm_w
    coords = [z, y, x]
    return coords

def get_position(target_patch, image, stepSize=1):
    score_map = np.zeros_like(image)
    windowSize = target_patch.shape
    for (z,y,x,img) in sliding_window(image, stepSize, windowSize):
        a = np.mean(np.multiply((target_patch-np.mean(target_patch)),(img-np.mean(img))))
        b = np.sqrt(np.mean((target_patch-np.mean(target_patch))**2))*np.sqrt(np.mean((img-np.mean(img))**2))
        score = a/(b+10**(-5))
        # print(score)
        # print(np.sum(target_patch * img))
        # score_map[z,y,x] = np.sum(target_patch * img)

        # score_map[z,y,x] = signal.convolve(target_patch, img, mode='same',method = "direct")
        # score_map[z,y,x] = np.correlate(target_patch, img,'full')/np.dot(abs(target_patch),abs(img),'full')
        # score_map[z,y,x] = np.mean(np.multiply((target_patch-np.mean(target_patch)),(img-np.mean(img))))/(np.std(target_patch)*np.std(img))
        # score_map[z,y,x] = np.square(np.subtract(target_patch, img)).mean()
        # score_map[z,y,x] = cos(torch.from_numpy(target_patch.reshape(1,-1)), torch.from_numpy(img.reshape(1,-1))).cpu().numpy()
        # score_map[z,y,x] = distance.cosine(target_patch.reshape(1,-1), img.reshape(1,-1))
        score_map[z,y,x] = 100*score
    # score_map = signal.convolve(image, target_patch, mode='same',method = "direct")
    # print(score_map.shape)
    coors = np.unravel_index(np.argmax(score_map, axis=None), score_map.shape)
    # coors = get_coords(score_map)
    return coors


def get_position_mse(target_patch, image, stepSize=1):
    score_map = np.ones_like(image)*1000
    windowSize = target_patch.shape
    for (z,y,x,img) in sliding_window(image, stepSize, windowSize):
        # score = np.square(np.subtract(target_patch, img)).mean()/1000
        # score_map[z,y,x] = score
        # print(score)
        score_map[z,y,x] = np.square(np.subtract(target_patch, img)).mean()/1000

        # score_map[z,y,x] = cos(torch.from_numpy(target_patch.reshape(1,-1)), torch.from_numpy(img.reshape(1,-1))).cpu().numpy()
        # score_map[z,y,x] = distance.cosine(target_patch.reshape(1,-1), img.reshape(1,-1))

    # score_map = signal.convolve(image, target_patch, mode='same',method = "direct")
    # print(score_map.shape)
    coors = np.unravel_index(np.argmin(score_map, axis=None), score_map.shape)
    # coors = get_coords(score_map)
    return coors

def get_position_cosine(target_patch, image, stepSize=1):
    score_map = np.zeros_like(image)
    windowSize = target_patch.shape
    for (z,y,x,img) in sliding_window(image, stepSize, windowSize):
        # score = 10*cos(torch.from_numpy(target_patch.reshape(1,-1)).to(torch.double), torch.from_numpy(img.reshape(1,-1)).to(torch.double)).cpu().numpy()
        # score_map[z,y,x] = score
        # print(score)
        score_map[z,y,x] = 10*cos(torch.from_numpy(target_patch.reshape(1,-1)).to(torch.double), torch.from_numpy(img.reshape(1,-1)).to(torch.double)).cpu().numpy()
    coors = np.unravel_index(np.argmax(score_map, axis=None), score_map.shape)
    # print(np.max(score_map))
    return coors




def get_position_feature_based(target_feature, image, cur_position,
                               fnet, pnet,windowSize, stepSize=1):
    score_map = np.ones_like(image)
    cur_position[0]=cur_position[0]//3
    for (z,y,x,img) in crop_sliding_window(image, stepSize, windowSize, cur_position):
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda().float()
        cur_feature = fnet(img)['8x']
        cur_feature = pnet(cur_feature, cur_feature)['feature_1'].cpu().data.numpy().squeeze()
        similarity_score = image_similarity(target_feature, cur_feature)
        score_map[z,y,x] = similarity_score
    coors = np.asarray(get_coords(score_map))
    coors[0]*=3
    return coors


def get_position_feature(target_patch, image, net, stepSize=[1,1,1]):
    # print('%',target_patch.shape, np.max(target_patch), np.max(image), np.min(image))
    score_map = np.ones_like(image)*10000
    # score_map = np.zeros_like(image)
    # print(score_map.shape)
    windowSize = target_patch.shape
    target_feature = net(torch.tensor(target_patch).unsqueeze(0).unsqueeze(0).type(torch.cuda.FloatTensor))
    # print(target_feature.size(), torch.max(target_feature), torch.min(target_feature))
    for (z,y,x,img) in sliding_window(image, stepSize, windowSize):
        # print('sliding...')
        # tt = time.time()
        img_feature = net(torch.tensor(img).unsqueeze(0).unsqueeze(0).type(torch.cuda.FloatTensor))
        # print(img_feature.size(), torch.max(img_feature))
        # MSE
        score_map[z,y,x] = torch.mean(torch.square((target_feature - img_feature))).cpu().numpy()
        # print(score)
        # 
        # cosine
        # score = cos(target_feature.view(1,-1), img_feature.view(1,-1)).cpu().numpy()
        # print(score)
        # score_map[z,y,x] = score
        # print(time.time() - tt)
    # print('Done')

    coors = np.unravel_index(np.argmin(score_map, axis=None), score_map.shape)
    # coors = get_coords(score_map)
    return coors



def get_position_sift(target_patch, image, stepSize=1):
    score_map = np.zeros_like(image)
    windowSize = target_patch.shape

    # [c,h,w] = target_patch.shape
    # print(c,h,w)
    target_patch = target_patch.reshape(6*48,-1).astype('uint8')
    image = image.astype('uint8')
    # target_patch = cv2.normalize(target_patch, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # print(target_patch.shape, type(target_patch), np.max(target_patch))
    #创建SIFT特征提取器
    sift = cv2.SIFT_create()
    #创建FLANN匹配对象
    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams=dict(checks=50)
    flann=cv2.FlannBasedMatcher(indexParams,searchParams)
    kp1, des1 = sift.detectAndCompute(target_patch, None) #提取样本图片的特征
    # print(len(kp1),des1.shape)

    for (z,y,x,img) in sliding_window(image, stepSize, windowSize):
        # img = cv2.normalize(img.reshape(6*48,-1), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # print(img.shape, np.min(img), np.max(img))
        if np.max(img) == 255:
            kp2, des2 = sift.detectAndCompute(img.reshape(6*48,-1), None) #提取比对图片的特征
            # print(np.max(img),'**',len(kp2))
            if len(kp2) >= 200:
                # print(len(kp2))
                # matches=flann.knnMatch(des1,des2,k=2) #匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
                matches=flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), 2)
                matchNum=getMatchNum(matches,0.9) #通过比率条件，计算出匹配程度
                # matchRatio=matchNum*100/len(matches)
                score_map[z,y,x] = matchNum*100/len(matches)
                # print(matchNum*100/len(matches))
            else:
                continue
    # score_map = signal.convolve(image, target_patch, mode='same',method = "direct")
    # print(score_map.shape)
    coors = np.unravel_index(np.argmax(score_map, axis=None), score_map.shape)
    # coors = get_coords(score_map)
    return coors

    
def getMatchNum(matches,ratio):
    '''返回特征点匹配数量和匹配掩码'''
    # matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
            # matchesMask[i]=[1,0]
            matchNum+=1
    # return (matchNum,matchesMask)
    return matchNum