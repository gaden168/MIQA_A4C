from utils import get_test_dataloader,calculate_metrics,calEuclidean,get_mse
import torch
from model import Baseline
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os,csv
from model import Baseline

from scipy import stats

def test(test_dataloader,test_dataset):
    model.eval()
    # Run test
    # srcc_all = np.zeros(len(test_dataset), dtype=np.float)
    # plcc_all = np.zeros(len(test_dataset), dtype=np.float)
    epoch = 0
    iteration = 0
    with torch.no_grad():
        result_socore_mses = []
        result_scores = []
        pred_scores = []
        gt_scores = []
        scores = []
        print('Start Test....')
        for i,(imgs, batch_target,label) in enumerate(test_dataloader):
            pred = model(imgs)
            pre_score = np.array(list(np.array(pred.cpu().numpy()).flatten()))
            result_score = calEuclidean(np.array(pre_score), np.array(label))

            result_socore_mse = get_mse(np.array(label), np.array(pre_score))

            result_scores.append(result_score.item())
            result_socore_mses.append(result_socore_mse.item())
            # print("score: {:.3f} ".format(result_score))
            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()
            
    
    test_srcc, p_srcc = stats.spearmanr(np.array(pred_scores), np.array(gt_scores))
    test_plcc, p_plcc = stats.pearsonr(np.array(pred_scores), np.array(gt_scores))
    print("test: "
          "mean_all_score: {:.3f} "
          "max_all_score: {:.3f} "
          "min_all_score: {:.3f}".format(
                                      np.mean(result_socore_mses),
                                      np.max(result_socore_mses),
                                      np.min(result_socore_mses)))

    # result_socore_mses
    # print('Testing  SRCC %4.4f,\tmedian P%4.4%', test_srcc,p_srcc)
    # print('Testing  PLCC %4.4f,\tmedian P%4.4%', test_plcc,p_plcc)
    # srcc_med = np.median(srcc_all)
    # plcc_med = np.median(plcc_all)
    # print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
    # srcc_mean = np.mean(srcc_all)
    # plcc_mean = np.mean(plcc_all)
    # print('Testing mean SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_mean, plcc_mean))
    return test_srcc, test_plcc


if __name__ == '__main__':
    num_workers = 2 # Number of CPU processes for data preprocessing
    lr = 1e-4 # Learning rate
    batch_size = 1
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    logdir = 'logs/'
    EPOCH = 10
    #data preprocessing:
    echo_test_dataloader,test_dataset = get_test_dataloader(
        mean = mean,
        std = std,
        img_folder = 'images',
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize the model
    model = Baseline()
    model.load_state_dict(torch.load("./chekpoints/BL/checkpoint-000035.pth"))#029
    # Switch model to the training mode and move it to GPU.
    # model = model.to(device)
    logger = SummaryWriter(logdir)

    srcc_all = np.zeros(EPOCH, dtype=np.float)
    plcc_all = np.zeros(EPOCH, dtype=np.float)

    for epoch in range(EPOCH):
        srcc_all[epoch], plcc_all[epoch] = test(echo_test_dataloader,test_dataset)

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    srcc_max = np.max(srcc_all)
    plcc_max = np.max(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
    print('Testing max SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_max, plcc_max))
    # print(len(test_dataset))
    # test_show(test_dataset)
    