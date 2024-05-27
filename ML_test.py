from utils import get_test_dataloader,calculate_metrics,calEuclidean,get_mse
import torch
from model import MTS
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os,csv
from scipy import stats

def test(test_dataloader,test_dataset):
    model.eval()
    # Run test
    epoch = 0
    iteration = 0
    with torch.no_grad():
        result_socore_mses = []
        model_result = []
        targets = []
        result_scores = []
        pred_scores = []
        gt_scores = []

        for imgs, batch_targets, batch_scores in test_dataloader:

            model_batch_result, model_batch_out2,out3  = model(imgs)
            
            temp = np.array(model_batch_result.cpu().numpy() > 0.5, dtype=float)
            pre_batch_slice = np.count_nonzero(temp == 1, axis=1)
            calculated_score = pre_batch_slice*0.45
            
            pre_scores = np.array(list(np.array(model_batch_out2.cpu().numpy()).flatten()))
            mean_scores = (pre_scores+np.array(calculated_score))/2


            
            score_classifier = calEuclidean(np.array(calculated_score), np.array(batch_scores))
            score_regressor = calEuclidean(np.array(pre_scores), np.array(batch_scores))
            result_score = calEuclidean(np.array(mean_scores), np.array(batch_scores))

            result_socore_mse = get_mse(np.array(batch_scores), np.array(mean_scores))

            result_scores.append(result_score.item())
            result_socore_mses.append(result_socore_mse.item())

            pred_scores.append(float(mean_scores.item()))
            gt_scores = gt_scores + batch_scores.cpu().tolist()


            model_result.extend(model_batch_result.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())
            # print("score_regressor: {:.3f} "
            #       "score_classifier: {:.3f} "
            #       "mean_result_score: {:.3f} ".format(
            #                                   score_regressor,
            #                                   score_classifier,
            #                                   result_score
            #                                   ))
        result = calculate_metrics(np.array(model_result), np.array(targets))
        test_srcc, p_srcc = stats.spearmanr(np.array(pred_scores), np.array(gt_scores))
        test_plcc, p_plcc = stats.pearsonr(np.array(pred_scores), np.array(gt_scores))

        print("test: "
              "micro f1: {:.3f} "
              "macro f1: {:.3f} "
              "samples f1: {:.3f}".format(
                                          result['micro/f1'],
                                          result['macro/f1'],
                                          result['samples/f1']))
        print("test: "
              "mean_all_score: {:.3f} "
              "max_all_score: {:.3f} "
              "min_all_score: {:.3f}".format(
                                          np.mean(result_socore_mses),
                                          np.max(result_socore_mses),
                                          np.min(result_socore_mses)))
        return test_srcc,test_plcc

if __name__ == '__main__':
    num_workers = 0
    lr = 1e-4 
    batch_size = 1
    save_freq = 1 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    logdir = 'logs/'
    n_classes = 10
    
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
    model = MTS(n_classes)
    # model.load(torch.load("./chekpoints/echo/checkpoint-000100.pth"))
    # model=torch.load("./chekpoints/echo/checkpoint-000097.pth")
    # model = torch.load("./chekpoints/ML/checkpoint-000065.pth")
    model.load_state_dict(torch.load("./chekpoints/ML/checkpoint-000084.pth"))


    # Switch model to the training mode and move it to GPU.
    # model = model.to(device)
    logger = SummaryWriter(logdir)
    EPOCH = 10

    # srcc_all = np.zeros(EPOCH, dtype=np.float)
    # plcc_all = np.zeros(EPOCH, dtype=np.float)

    for epoch in range(EPOCH):
        test(echo_test_dataloader,test_dataset)
    # # test_show(test_dataset)
    # srcc_med = np.median(srcc_all)
    # plcc_med = np.median(plcc_all)
    #
    # srcc_max = np.max(srcc_all)
    # plcc_max = np.max(plcc_all)

    # print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
    # print('Testing max SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_max, plcc_max))
