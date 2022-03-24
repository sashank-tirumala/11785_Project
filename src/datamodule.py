import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as ttf

import os
import os.path as osp
import os
from typing import List
from typing import Optional

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np
import pytorch_lightning as pl

import logging

import matplotlib.pyplot as plt
from init import *
import numpy as np
import random
def get_file_paths(data_dirs, reskin_file = "reskin_data.csv" ):
    """
    Input: is a ratio of training to validation split. They should sum up to 100 and paths to directories.
    Output: Paths to maintain this split across each class (The main idea being to approximately
    maintain the same ratio. 0 will still be dominant but rest of the classes should be equal in datapoints roughly )
    """
    if(type(data_dirs) == str):
        data_dirs = [data_dirs]
    reskin_paths = {"-1cloth":[],"0cloth":[], "1cloth":[], "2cloth": [], "3cloth": []}
    for data_dir in data_dirs:
        class_dirs = os.listdir(data_dir)
        for class_dir in class_dirs:
            temp = []
            path_dirs = os.listdir(data_dir + "/" + class_dir)
            for path_dir in path_dirs:
                reskin_file_path = data_dir + "/" + class_dir + "/" + path_dir + "/" + reskin_file
                if("0cloth" in class_dir):
                    reskin_paths["0cloth"].append(reskin_file_path)
                elif("1cloth" in class_dir):
                    reskin_paths["1cloth"].append(reskin_file_path)
                elif("2cloth" in class_dir):
                    reskin_paths["2cloth"].append(reskin_file_path)
                elif("3cloth" in class_dir):
                    reskin_paths["3cloth"].append(reskin_file_path)
    return reskin_paths

def get_context(data, context, offset=-1, time_idx = -1, class_idx = -2, include_transition = False, context_type = "double"):
    """
    Input: 2D np array -> raw data from reskin csv file
    Output: dictionary with class followed context values
    """
    res = []
    start_idx = context
    if(context_type == "double"):
        end_idx = data.shape[0] - context - 1
    elif(context_type == "left"):
        end_idx =  data.shape[0]
    else:
        print("Invalid context type")
        return None
    for i in np.arange(start_idx, end_idx, 1):
        
        if(context_type == "double"):
            cur_val = np.array(data[i-context: i+context+1,:])
        elif(context_type == "left"):
            cur_val = np.array(data[i-context:i,:])
        else:
            print("Invalid context type")
            return None
        bool_arr = cur_val[:,class_idx] == cur_val[0,class_idx]
        if(np.all(bool_arr) or include_transition): #Checks if the value is in a transition or not --> For now removing, I can disable this 
            label = int(cur_val[0,class_idx])+offset
            res.append([np.delete(cur_val, [time_idx, class_idx], axis=1), label])
    return res

def std_normalizer(data):
    x_vals = []
    for dat in data:
        x_vals.append(dat[0])
    x_vals = np.hstack(x_vals)
    x_mean  = np.mean(x_vals, axis=0).reshape(1,-1)
    x_std = np.std(x_vals, axis=0).reshape(1,-1)
    for dat in data:
        res_x = np.zeros(dat[0].shape)
        for col in range(dat[0].shape[1]):
            res_x[:,col] = dat[0][:,col] - x_mean[0, col]
            res_x[:,col] = res_x[:,col] /(x_std[0, col]+1e-6)
        dat[0] = res_x
    return data

def get_data(paths):
    data = []
    for path in paths:
        temp = np.loadtxt(path, delimiter=",")
        data.append(temp)
    return data

class ClothDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, shuffle, batch_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.train_transform = ttf.Compose([ttf.ToTensor()])
        self.val_transform = ttf.Compose([ttf.ToTensor()])
        self.test_transform = ttf.Compose([ttf.ToTensor()])
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = 4
        self.shuffle = shuffle
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.face_recognition_train = torchvision.datasets.ImageFolder(self.train_dir,
                                                    transform=self.train_transform)
        if stage == "test" or stage is None:
            self.face_recognition_test = ClassificationTestSet(self.test_dir, self.test_transform)

        if stage == "validate" or stage is None:
            self.face_recognition_val = torchvision.datasets.ImageFolder(self.val_dir, transform = self.val_transform)

    def train_dataloader(self):
        # tensor_image = self.face_recognition_train[0][0]
        # plt.figure()
        # plt.imshow(tensor_image.permute(1, 2, 0))
        # plt.show()
        return DataLoader(self.face_recognition_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers= self.num_workers) 

    def test_dataloader(self):
        return DataLoader(self.face_recognition_test, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers= self.num_workers) 
    
    def val_dataloader(self):
        return DataLoader(self.face_recognition_val, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers= self.num_workers)


class ClothDataSet(Dataset):
    def __init__(self, paths, transforms, context = 5, normalizer = std_normalizer, get_context = get_context, label_offset = -1,
    time_idx = -1, class_idx = -2, include_transition = False, context_type = "double", shuffle=True):
        self.paths = paths
        self.transforms = transforms
        self.normalizer = std_normalizer
        self.get_context = get_context
        self.label_offset = label_offset
        self.time_idx = time_idx
        self.class_idx = class_idx
        self.include_transition = include_transition
        self.context_type = context_type
        self.data = 0
        self.shuffle = shuffle
        self.context = context
        self.setup()

    def setup(self):
        """
        store it in an array of arrays [[ x, y], [x, y] and so] x is a 2D Array btw, 
        normalize the entire dataset (only x)
        shuffle the data completely (shuffling within a particular dataset is fine, just not across datasets)
        store it in some variable
        """
        self.data = get_data(self.paths)
        temp = []
        for data in self.data:
            temp=temp+get_context(data, self.context, self.label_offset, self.time_idx, self.class_idx, self.include_transition, self.context_type)
        self.data = temp
        self.data = self.normalizer(self.data)
        if(self.shuffle):
            random.shuffle(self.data)
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """
        Think of what data augmentations you want to perform before finally outputting your code
        """
        #TODO perform transforms
        return self.data[idx]


if(__name__ == "__main__"):
    dirn = str(ROOT_DIR) + "/data/angled_feb25_all"
    res = get_file_paths(dirn)
    paths = res["0cloth"]
    # dat = get_data(res["0cloth"])
    # temp = []
    # for data in dat:
    #     temp=temp+get_context(data, 5)
    # data = temp
    # data = std_normalizer(data)
    data = ClothDataSet(paths, None)
    print(data[10])