# train.py
#!/usr/bin/env	python3

""" train network using pytorch

"""

import csv,os
import random
import sys
import argparse
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score,roc_curve,roc_auc_score,auc
import numpy as np
import pandas as pd
from utils import get_train_dataloader,get_val_dataloader,get_test_dataloader,calculate_metrics,calEuclidean
# from FocalLoss import FocalLoss
from torch.autograd import Variable
from sklearn.metrics import classification_report,confusion_matrix
from model import Baseline
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from BL_test import test
from scipy import stats
from memory_profiler import profile
from ptflops import get_model_complexity_info

import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ['TZ'] = 'UTC-8'
time.tzset()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2022)

def train_test(train_dataloader,val_dataloader,val_freq,test_dataloader,test_dataset):
    best_loss = 9999
    model.train()
    # Run training
    epoch = 0
    iteration = 0
    best_score = 999
    best_score_epoch = 0
    
    srcc_all = np.zeros(max_epoch_number, dtype=np.float)
    plcc_all = np.zeros(max_epoch_number, dtype=np.float)

    best_srcc = 0.0
    while True:
        batch_losses = []
        for imgs, targets,scores in train_dataloader:
            imgs, targets,scores = imgs.to(device), targets.to(device), scores.to(device)
            optimizer.zero_grad()
            model_result = model(imgs)
            # print(imgs.shape)
            loss = criterion(model_result, scores.type(torch.float)) 
            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()

            logger.add_scalar('train_loss', batch_loss_value, iteration)
            batch_losses.append(batch_loss_value)
            
            # 使用 ptflops 库计算 GFLOPs
            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(model, (3, 128, 128), as_strings=False,
                                                         print_per_layer_stat=True, verbose=True)
                gflops = (macs * 2) / 1e9
                print(f"Computational complexity: {gflops:.2f} GFLOPs")
                print(f"Number of parameters: {params / 1e6:.2f} M")
            
            if iteration % val_freq == 0:
                start_time = time.time()
                model.eval()
                with torch.no_grad():
                    pred_scores = []
                    gt_scores = []
                    for imgs, batch_targets,batch_scores in val_dataloader:
                        imgs = imgs.to(device)
                        model_batch_result = model(imgs)
                        pred_scores.extend(model_batch_result.cpu().numpy())
                        gt_scores.extend(batch_scores.cpu().numpy())
                pred_scores = np.array(list(np.array(pred_scores).flatten()))
                result_score = calEuclidean(np.array(pred_scores), np.array(gt_scores))
                
                test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
                test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        
                if result_score <  best_score:
                    best_score = result_score
                    best_score_epoch = epoch
                print("mean_result_score: {:.3f} ".format(
                                                  result_score
                                                  ))
                if test_srcc > best_srcc:
                    best_srcc = test_srcc
                    best_plcc = test_plcc
                print('epoch:%d, Val: SRCC %f, PLCC %f' % (epoch, best_srcc, best_plcc))
                
                model.train()
                end_time = time.time()
            iteration += 1
            print(f"Test Time: {end_time - start_time} seconds")
        loss_value = np.mean(batch_losses)
        print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
        # if epoch % save_freq == 0:
        if result_score <  best_score:
            checkpoint_save(model, save_path, epoch)
        #end one epoch 
        epoch += 1
        srcc_all[epoch], plcc_all[epoch] = test(model,test_dataloader,test_dataset)
        if max_epoch_number < epoch:
            break
    
    print('**************best_score_epoch in *********',best_score_epoch)
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)
    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))

    
def checkpoint_save(model, save_path, epoch):
    f = os.path.join(save_path+'/BL/', 'checkpoint-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)

# @profile
# def model_inference(x):
#     # 仅查看这一行的内存使用情况
#     output = model(x)
#     return output

if __name__ == '__main__':
    # Initialize the training parameters.
    num_workers = 2 
    lr = 1e-4 
    batch_size = 32
    save_freq = 1
    val_freq = 20 
    max_epoch_number = 200 

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    device = torch.device('cuda')
    save_path = 'chekpoints/'
    logdir = 'logs/'

    #data preprocessing:
    echo_training_loader = get_train_dataloader(
        mean = mean,
        std = std,
        img_folder = 'images',
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True
    )
    echo_val_loader = get_val_dataloader(
        mean = mean,
        std = std,
        img_folder = 'images',
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False
    )
    echo_test_dataloader,test_dataset = get_test_dataloader(
        mean = mean,
        std = std,
        img_folder = 'images',
        num_workers=num_workers,
        batch_size=1,
        shuffle=False
    )
    
    # Initialize the model
    model = Baseline()
    
    # 计算参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f} M")  # 转换为百万参数数量 (M)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    os.makedirs(save_path, exist_ok=True)
    


    
    # Loss function
    # criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    
    logger = SummaryWriter(logdir)
    train_test(echo_training_loader,echo_val_loader,val_freq,
               echo_test_dataloader,test_dataset)
    
