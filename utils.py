""" helper function

author baiyu
"""
import os
import time
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torchvision import transforms
from torchvision import models
import torch
from torch.utils.data.dataloader import DataLoader
from dataset import EchoDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools

def get_train_dataloader(mean, std, img_folder, batch_size, num_workers=0, shuffle=True):
    """ return training dataloader
    Returns: train_data_loader:torch dataloader object
    """
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5),
                                shear=None, fill=tuple(np.array(np.array(mean)*255).astype(int).tolist())),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_annotations = os.path.join(img_folder,'train.json')
    train_dataset = EchoDataset(train_annotations, train_transform)
    
    echo_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    return echo_train_dataloader

def get_val_dataloader(mean, std, img_folder, batch_size, num_workers=0, shuffle=True):
    """ return training dataloader
    Returns: train_data_loader:torch dataloader object
    """
    # val preprocessing
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    val_annotations = os.path.join(img_folder, 'val.json')
    val_dataset = EchoDataset(val_annotations, val_transform)
    echo_val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    
    return echo_val_dataloader

def get_test_dataloader(mean, std, img_folder, batch_size, num_workers=0, shuffle=True):
    """ return training dataloader
    Returns: train_data_loader:torch dataloader object
    """
    # Test preprocessing
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_annotations = os.path.join(img_folder, 'test.json')
    test_dataset = EchoDataset(test_annotations, test_transform)
    echo_test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    return echo_test_dataloader,test_dataset

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }

def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def calEuclidean(x, y):
    dist = np.sqrt(np.sum(np.square(x-y)))   
    # 注意：np.array 类型的数据可以直接进行向量、矩阵加减运算。np.square 是对每个元素求平均~~~~
    return dist

def mini_batch_rank(model_result, scores):
    tem_list = list(itertools.combinations(scores, 2))
    model_result_list = list(itertools.combinations(model_result, 2))
    x1_list = []
    x2_list = []
    y_list =[]
    for i,item in enumerate(tem_list):
        a,b = item
        if a > b or a == b:
            y = 1
        else:
            y =-1
        c,d = model_result_list[i]
        x1_list.append(c)
        x2_list.append(d)
        y_list.append(y)
    return torch.tensor(x1_list),torch.tensor(x2_list),torch.tensor(y_list)


import numpy as np
import numpy as np


def voc12_mAP(imagessetfile, num):
    with open(imagessetfile, 'r') as f:
        lines = f.readlines()

    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_label = seg[:, num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims=True)
    threshold = 1 / (num_target + 1e-6)

    predict_result = seg[:, 0:num] > threshold

    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []
    per_class_recall = []

    for class_id in range(class_num):
        confidence = seg[:, class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i] > 0)
            fp[i] = (sorted_label[i] <= 0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=3, suppress=True)
    mAP = np.mean(aps)
    return mAP


def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

import numpy as np


def patk(actual, pred, k):
    # we return 0 if k is 0 because
    #   we can't divide the no of common values by 0
    if k == 0:
        return 0

    # taking only the top k predictions in a class
    k_pred = pred[:k]

    # taking the set of the actual values
    actual_set = set(actual)
    # print(list(actual_set))
    # taking the set of the predicted values
    pred_set = set(k_pred)
    # print(list(pred_set))

    # 求预测值与真实值得交集
    common_values = actual_set.intersection(pred_set)
    # print(common_values)

    return len(common_values) / len(pred[:k])


def apatk(acutal, pred, k):
    #creating a list for storing the values of precision for each k
    precision_ = []
    for i in range(1, k+1):
        #calculating the precision at different values of k
        #      and appending them to the list
        precision_.append(patk(acutal, pred, i))

    #return 0 if there are no values in the list
    if len(precision_) == 0:
        return 0

    #returning the average of all the precision values
    return np.mean(precision_)



def mapk(acutal, pred, k):

    #creating a list for storing the Average Precision Values
    average_precision = []
    #interating through the whole data and calculating the apk for each
    for i in range(len(acutal)):
        ap = apatk(acutal[i], pred[i], k)
        # print(f"AP@{i}: {ap}")
        average_precision.append(ap)

    #returning the mean of all the data
    return np.mean(average_precision)
#defining the values of the actual and the predicted class
y_true = [[1,2,0,1], [0,4], [3], [1,2]]
y_pred = [[1,1,0,1], [1,4], [2], [1,3]]

def get_apatk(y_true,y_pred):
    for i in range(len(y_true)):
        for j in range(1, 9):
            print(
                f"""
                AP@{j} = {apatk(y_true[i], y_pred[i], k=j)}
                """
            )
if __name__ == "__main__":
    get_apatk(y_true, y_pred)
    # print(mapk(y_true, y_pred,3))

# if __name__ == '__main__':
#     scores = [0.45,0.33,0.45,0.34,0.21,0.78]
#     model_result = [0.55,0.33,0.45,0.34,0.89,0.78]
#     model_result_list= mini_batch_rank(model_result,scores)
#     print(model_result_list)