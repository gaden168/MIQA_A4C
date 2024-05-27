from torch.utils.data.dataset import Dataset
from numpy import printoptions
import requests
import tarfile
import random
import json
from shutil import copyfile
import os
import time
import numpy as np
from PIL import Image
import itertools
import torch

class EchoDataset(Dataset):
    def __init__(self, anno_path, transforms):
        self.transforms = transforms
        with open(anno_path) as fp:
            json_data = json.load(fp)
        samples = json_data['samples']
        self.classes = json_data['labels']
        self.imgs = []
        self.annos = []
        self.score = []
        print('loading', anno_path)
        for sample in samples:
            self.imgs.append(sample['image_name'])
            self.annos.append(sample['image_labels'])
            self.score.append(np.array(sample['image_score'], dtype=float))
        for item_id in range(len(self.annos)):
            item = self.annos[item_id]
            vector = [cls in item for cls in self.classes]
            self.annos[item_id] = np.array(vector, dtype=float)

    def __getitem__(self, item):
        anno = self.annos[item]
        score = self.score[item]
        img_path = os.path.join(self.imgs[item])
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, anno,score

    def __len__(self):
        return len(self.imgs)

    
class SiameseNetworkDataset(Dataset):
    def __init__(self, anno_path, transforms):
        self.transforms = transforms
        with open(anno_path) as fp:
            json_data = json.load(fp)
        samples = json_data['samples']
        self.imgs = []
        self.score = []
        print('loading', anno_path)
        for sample in samples:
            self.imgs.append(sample['image_name'])
            # self.score.append(np.array(sample['image_score'], dtype=float))[消耗内存太大]
        # self.combin_imgs = list(itertools.combinations(self.imgs, 2))
    def __getitem__(self, item):
        img0, img1 =  self.combin_imgs[item]
        score0 = self.score[self.imgs.index(img0)]
        score1= self.score[self.imgs.index(img1)]                  
        # img0_path = os.path.join(self.imgs[item])
        img0 = Image.open(img0)
        img1 = Image.open(img1)
        
        if self.transforms is not None:
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
        return img0, img1, torch.from_numpy(np.array([int(score0 > score1)],dtype=np.float32))
    def __len__(self):
        return len(self.imgs)
