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
from utils import get_train_dataloader,get_val_dataloader,\
    get_test_dataloader,calculate_metrics,calEuclidean,get_mse,mini_batch_rank
# from FocalLoss import FocalLoss
from torch.autograd import Variable
from sklearn.metrics import classification_report,confusion_matrix
from model import MTS
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from scipy import stats
from ML_test import test
from losses import  ContrastiveLoss
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

def train_test(train_dataloader,val_dataloader,val_freq,test_dataloader,test_dataset,alph=0.5):
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
    loss2 = 0
    loss3 = 0
    while True:
        batch_losses = []
        for imgs, targets,scores in train_dataloader:
            imgs, targets,scores = imgs.to(device), targets.to(device), scores.to(device)
            optimizer.zero_grad()
            model_result,out2,out_feature = model(imgs)
            x1, x2, y_ = mini_batch_rank(out2, scores)
            loss1 = criterion(model_result, targets.type(torch.float))
            loss2 = loss_func(out2, scores.type(torch.float).unsqueeze(1))
            loss3 += loss_rank(x1, x2, y_)
            loss = loss1 +loss3 + loss2
            # loss = alph*loss1 +(1-alph)*loss23

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
                
            with torch.no_grad():
                result = calculate_metrics(model_result.cpu().numpy(), targets.cpu().numpy())
                for metric in result:
                    logger.add_scalar('train/' + metric, result[metric], iteration)
            if iteration % val_freq == 0:
                start_time = time.time()
                model.eval()
                with torch.no_grad():
                    model_result = []
                    targets = []
                    model_result2 = []
                    scores = []
                    calculated_scores = []
                    for imgs, batch_targets,batch_scores in val_dataloader:
                        imgs = imgs.to(device)
                        model_batch_result,model_batch_out2,out_feature = model(imgs)
                        
                        temp = np.array(model_batch_result.cpu().numpy() > 0.5, dtype=float)
                        pre_batch_slice = np.count_nonzero(temp == 1, axis=1)
                        model_result.extend(model_batch_result.cpu().numpy())
                        model_result2.extend(model_batch_out2.cpu().numpy())
                        targets.extend(batch_targets.cpu().numpy())
                        scores.extend(batch_scores.cpu().numpy())
                        calculated_scores.extend(pre_batch_slice*0.45)
                        
                pre_scores = np.array(list(np.array(model_result2).flatten()))
                mean_scores = (pre_scores+np.array(calculated_scores))/2

                test_srcc, _ = stats.spearmanr(mean_scores, scores)
                test_plcc, _ = stats.pearsonr(mean_scores, scores)

                result = calculate_metrics(np.array(model_result), np.array(targets))
                score_classifier = calEuclidean(np.array(calculated_scores), np.array(scores))
                score_regressor = calEuclidean(np.array(pre_scores), np.array(scores))
                result_score = calEuclidean(np.array(mean_scores), np.array(scores))
                result_socore_mse = get_mse(np.array(scores), np.array(mean_scores))

                test_srcc, _ = stats.spearmanr(mean_scores, scores)
                test_plcc, _ = stats.pearsonr(mean_scores, scores)

                if result_socore_mse < best_score:
                    best_score = result_socore_mse
                    best_score_epoch = epoch
                print("mean_result_mse: {:.3f} ".format(
                    result_socore_mse
                ))
                if test_srcc > best_srcc:
                    best_srcc = test_srcc
                    best_plcc = test_plcc
                # print('epoch:%d, Test: SRCC %f, PLCC %f' % (epoch, best_srcc, best_plcc))


                if result_socore_mse <  best_score:
                    best_score = result_socore_mse
                    best_score_epoch = epoch
                print("epoch:{:2d} iter:{:3d} test: "
                      "micro f1: {:.3f} "
                      "macro f1: {:.3f} "
                      "samples f1: {:.3f}".format(epoch, iteration,
                                                  result['micro/f1'],
                                                  result['macro/f1'],
                                                  result['samples/f1'],
                                                  ))
                print("score_regressor: {:.3f} "
                      "score_classifier: {:.3f} "
                      "mean_result_score: {:.3f} ".format(
                                                  score_regressor,
                                                  score_classifier,
                                                  result_socore_mse
                                                  ))
                # if test_srcc > best_srcc:
                #     best_srcc = test_srcc
                #     best_plcc = test_plcc
                # print('epoch:%d, Val: SRCC %f, PLCC %f' % (epoch, best_srcc, best_plcc))
                
                model.train()
                end_time = time.time()
            iteration += 1
            print(f"Training Time: {end_time - start_time} seconds")
        loss_value = np.mean(batch_losses)
        
        print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
        # if result_score <  best_score:
        if epoch % save_freq == 0 and epoch > 50:
            checkpoint_save(model, save_path, epoch)
        epoch += 1
        if max_epoch_number < epoch:
            break
    print('**************best_score_epoch in *********',best_score_epoch)

def checkpoint_save(model, save_path, epoch):
    f = os.path.join(save_path+'ML/', 'checkpoint-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        # torch.save(model,f)
        torch.save(model.module.state_dict(), f)
    else:
        # torch.save(model, f)
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)

# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.

if __name__ == '__main__':
    # Initialize the training parameters.
    num_workers = 2
    lr = 1e-4 
    batch_size = 16
    save_freq = 1 
    val_freq = 20 
    max_epoch_number = 100

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    n_classes = 10
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
    echo_test_dataloader, test_dataset = get_test_dataloader(
        mean=mean,
        std=std,
        img_folder='images',
        num_workers=num_workers,
        batch_size=1,
        shuffle=False
    )
    # Initialize the model
    model = MTS(n_classes)
    model = model.to(device)
    
    # 计算参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f} M")  # 转换为百万参数数量 (M)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    os.makedirs(save_path, exist_ok=True)

    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCELoss() #class
    loss_func = nn.MSELoss() #score
    loss_rank = nn.MarginRankingLoss()
    # con_loss = ContrastiveLoss()
    logger = SummaryWriter(logdir)
    # train(echo_training_loader,echo_val_loader,val_freq,alph=0.5)
    train_test(echo_training_loader, echo_val_loader, val_freq,
               echo_test_dataloader, test_dataset,alph=0.5)