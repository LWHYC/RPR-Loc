#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import sys
import os
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch.optim as optim
import torch.tensor
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms
from dataloaders.Position_dataloader import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
from losses.loss_function import TestDiceLoss, AttentionExpDiceLoss
from util.visualization.visualize_loss import loss_visualize
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def random_all(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def extend_dimension(label,predic):
    for i in range(predic.shape[1]):
        one_pred = torch.ones_like(predic)


def train(config_file):
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data  = config['data']   
    config_pnet  = config['pnetwork'] 
    config_train = config['training']

    patch_size = config_data['patch_size']
    batch_size = config_data.get('batch_size', 4)
    class_num   = config_pnet['class_num']
    cur_loss = 0
    lr = config_train.get('learning_rate', 1e-3)
    best_loss = config_train.get('best_loss', 0.5)
    random_seed = config_train.get('random_seed', 1)
    small_move = config_train.get('small_move', False)
    fluct_range = config_train.get('fluct_range')
    num_worker = config_train.get('num_worker')
    random_all(random_seed)  
    random_crop = RandomPositionDoubleCrop(patch_size, small_move=small_move, fluct_range=fluct_range)
    to_tensor = ToPositionTensor()

    cudnn.benchmark = True
    cudnn.deterministic = True

    # 2, load data
    print('2.Load data')
    trainData = PositionDataloader(config=config_data,
                                   split='train',
                                   transform=transforms.Compose([
                                       RandomPositionDoubleCrop(patch_size, small_move=small_move, fluct_range=fluct_range),
                                       ToPositionTensor(),
                                   ]))
    validData = PositionDataloader(config=config_data,
                                   split='valid',
                                   transform=None)
    trainLoader = DataLoaderX(trainData, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
    validLoader = DataLoaderX(validData, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # 3. creat model
    print('3.Creat model')
    net_type   = config_pnet['net_type']
    net_class = NetFactory.create(net_type)
    distance_ratio= config_pnet['distance_ratio']
    pnet = net_class(
                    inc=config_pnet.get('input_channel', 1),
                    base_chns= config_pnet.get('base_feature_number', 16),
                    norm='in',
                    depth=config_pnet.get('depth', False),
                    dilation=config_pnet.get('dilation', 1),
                    patch_size=patch_size,
                    n_classes = config_pnet['class_num'],
                    droprate=config_pnet.get('drop_rate', 0.2)
                    )
    pnet = torch.nn.DataParallel(pnet).cuda()
    if config_train['load_weight']:
        pnet_weight = torch.load(config_train['pnet_load_path'], map_location=lambda storage, loc: storage) # position net
        pnet.load_state_dict(pnet_weight)

    loss_func = nn.MSELoss()
    show_loss = loss_visualize(class_num)
    Adamoptimizer = optim.Adam(pnet.parameters(), lr=lr, weight_decay=config_train.get('decay', 1e-7))
    Adamscheduler = torch.optim.lr_scheduler.StepLR(Adamoptimizer, step_size=5, gamma=0.9)

    # 4, start to train
    print('4.Start to train')
    start_it  = config_train.get('start_iteration', 0)
    dice_save = {}
    lr_retain_epoch = 0
    for epoch in range(start_it, config_train['maximal_epoch']): 
        print('#######epoch:', epoch)
        optimizer = Adamoptimizer
        pnet.train()
        for i_batch, sample_batch in enumerate(trainLoader):
            img_batch0,img_batch1, label_batch = sample_batch['random_crop_image_0'], \
                                                 sample_batch['random_crop_image_1'],sample_batch['rela_distance']
            img_batch0,img_batch1, label_batch = img_batch0.cuda(),img_batch1.cuda(), label_batch.cuda()
            predic_cor_0 = pnet(img_batch0)['fc_position']
            predic_cor_1 = pnet(img_batch1)['fc_position']
            predic = distance_ratio*torch.tanh(predic_cor_0-predic_cor_1)
            train_loss = loss_func(predic, label_batch)
            optimizer.zero_grad() 
            train_loss.backward() 
            optimizer.step() 
            if epoch%config_train['train_step']==0 and i_batch%config_train['print_step']==0:
                train_loss = train_loss.cpu().data.numpy()
                if i_batch ==0:
                    train_loss_array=train_loss
                else:
                    train_loss_array = np.append(train_loss_array, train_loss)
                print('train batch:',i_batch,'train loss:', train_loss, '\npredic:', predic[0].cpu().data.numpy(), 'label:', label_batch[0].cpu().data.numpy())
        Adamscheduler.step()
        if  epoch % config_train['test_step']==0:
            with torch.no_grad():
                pnet.eval()
                for ii_batch, sample_batch in enumerate(validLoader):
                    sample_batch['image']=sample_batch['image'].cpu().data.numpy().squeeze()
                    for  ii_iter in range(config_train['test_iter']):
                        sample = random_crop(sample_batch)
                        sample = to_tensor(sample)
                        img_batch0,img_batch1,label_batch=sample['random_crop_image_0'].cuda().unsqueeze(0), \
                                                          sample['random_crop_image_1'].cuda().unsqueeze(0),sample['rela_distance'].cuda()
                        predic_cor_0 = pnet(img_batch0)['fc_position']
                        predic_cor_1 = pnet(img_batch1)['fc_position']
                        predic = distance_ratio*torch.tanh(predic_cor_0-predic_cor_1)
                        valid_loss = loss_func(predic,label_batch ).cpu().data.numpy()
                        if ii_batch ==0 and ii_iter==0:
                            valid_loss_array = valid_loss
                        else:
                            valid_loss_array = np.append(valid_loss_array, valid_loss)
                    print('valid batch:',ii_batch,' valid loss:', valid_loss)

            epoch_loss = {'valid_loss':valid_loss_array.mean(), 'train_loss':train_loss_array.mean()}
            t = time.strftime('%X %x %Z')
            print(t, 'epoch', epoch, 'loss:', epoch_loss)
            show_loss.plot_loss(epoch, epoch_loss)
            dice_save[epoch] = epoch_loss

            'save current model'
            if os.path.exists(config_train['pnet_save_name'] + "_cur_{0:}.pkl".format(cur_loss)):
                os.remove(config_train['pnet_save_name'] + "_cur_{0:}.pkl".format(cur_loss))
            cur_loss = epoch_loss['valid_loss']
            torch.save(pnet.state_dict(), config_train['pnet_save_name'] + "_cur_{0:}.pkl".format(cur_loss))

            'save the best model'
            if epoch_loss['valid_loss'] < best_loss:
                if os.path.exists(config_train['pnet_save_name'] + "_{0:}.pkl".format(best_loss)):
                    os.remove(config_train['pnet_save_name'] + "_{0:}.pkl".format(best_loss))
                best_loss = epoch_loss['valid_loss']
                torch.save(pnet.state_dict(), config_train['pnet_save_name'] + "_{0:}.pkl".format(best_loss))



if __name__ == '__main__':
    config_file = str('config/train_position_pancreas_coarse.txt')
    assert(os.path.isfile(config_file))
    train(config_file)
