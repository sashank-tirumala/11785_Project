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
    # It's possible to load test set data using ImageFolder without making a custom class.
    # See if you can think it through!

    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms


    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]))

def get_file_paths(data_dirs, reskin_file = "reskin_data.csv" ):
    """
    Input: is a ratio of training to validation split. They should sum up to 100 and paths to directories.
    Output: Paths to maintain this split across each class (The main idea being to approximately
    maintain the same ratio. 0 will still be dominant but rest of the classes should be equal in datapoints roughly )
    """
    if(type(data_dirs) == str):
        data_dirs = [data_dirs]
    reskin_paths = {"0cloth":[], "1cloth":[], "2cloth": [], "3cloth": []}
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

def create_data_array(reskin_paths, context):
    """
    Input: paths to reskin files dictionary
    Output: dictionary with data bunched up by context. (Context values before and Context values after)
    """
    
    pass

if(__name__ == "__main__"):
    dirn = str(ROOT_DIR) + "/data/angled_feb25_all"
    res = get_file_paths(dirn)
    arr = np.loadtxt(res["0cloth"][0], delimiter=",")
    print(arr[500:600,:])
    arr = np.loadtxt(res["1cloth"][0], delimiter=",")
    print(arr[500:600,:])
    arr = np.loadtxt(res["2cloth"][0], delimiter=",")
    print(arr[500:600,:])
    arr = np.loadtxt(res["3cloth"][0], delimiter=",")
    print(arr[500:600,:])
    
    pass