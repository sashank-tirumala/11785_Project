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

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloading import FaceRecognitionDataModule
from pytorch_lightning.loggers import WandbLogger

from timm.models.layers import trunc_normal_, DropPath

class MLPModel(pl.LightningModule):
    def __init__(self, input_context = 10, num_classes = 5, depths=5, lr = 4e-3, weight_decay=1e-4):
        super().__init__()
        self.lr = self.lr
        self.weight_decay = self.weight_decay
        inp_layer = nn.Linear(input_context*15, input_context*15*8),
        temp=[]
        temp.append(inp_layer)
        temp.append(nn.ReLU())
        for i in range(depths):
            temp.append(nn.Linear(input_context*15*8, input_context*15*8))
            temp.append(nn.ReLU())
        temp.append(nn.Linear(input_context*15*8, num_classes))
        self.layers = nn.Sequential(*temp)      

    def forward(self,x):
        outs = self.layers(x)
        return outs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.CrossEntropyLoss()
        output = loss(y_hat, y)
        accuracy_percent = 100*int((torch.argmax(y_hat, axis=1) == y).sum())/x.shape[0]
        self.log("train_loss", output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy_percent, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return output
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.CrossEntropyLoss()
        output = loss(y_hat, y)
        accuracy_percent = 100*int((torch.argmax(y_hat, axis=1) == y).sum())/x.shape[0]
        self.log("val_loss", output,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy_percent, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return y_hat

    def predict_step(self, batch, batch_idx):
        x = batch
        pred = self(x)
        return pred
if(__name__ == "__main__"):
    model = MLPModel()
    

