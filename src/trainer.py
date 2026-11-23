import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn

from tqdm import tqdm
from os import replace
#from numpy.core.numeric import Inf
from numpy import inf

from src.utils import create_dir

import sys

import segmentation_models_pytorch as smp
from src.modelGenerator import generate_model

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.nn.functional as F
import pdb

import icecream as ic

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score


def get_loss(loss_type="CrossEntropyLoss"):

    if loss_type == "CrossEntropyLoss":
        weights = [0.1, 0.9, 0]            
        class_weights = torch.FloatTensor(weights).cuda()
        loss_segmentation = nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "FocalLoss":
        loss_segmentation = smp.losses.FocalLoss(mode ="multiclass", gamma=3,ignore_index=[2])
    return loss_segmentation


def get_optimizer(config, net):

    if config['General']['optim'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=config['General']['lr'])
    elif config['General']['optim'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=config['General']['lr'], momentum=config['General']['momentum'])
    return optimizer

def get_scheduler(optimizer):
    return [ReduceLROnPlateau(optimizer)]

class EarlyStopping():
    def __init__(self, patience):
        self.patience = patience
        self.restartCounter()
    def restartCounter(self):
        self.counter = 0
    def increaseCounter(self):
        self.counter += 1
    def checkStopping(self):
        if self.counter >= self.patience:
            return True
        else:
            return False
class Trainer(object):
        
    def __init__(self, config, tm=""):
        super().__init__()
        self.config = config

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']


        self.model = generate_model(config)

        #ic(self.model)
                
        self.model.to(self.device)

        # print(self.model)

        # print(self.model.encoder.model.blocks_1.stack)

        # pdb.set_trace()
        # exit(0)
        # print("input shape: ", (3,resize,resize))
        # print(resize)
        # summary(self.model, (3,resize,resize))
        # exit(0)

        loss_type = self.config.get("loss_type",-1) 
        if loss_type == -1:
            self.loss_segmentation = get_loss()
        else:
            self.loss_segmentation = get_loss(loss_type)
            
        self.optimizer = get_optimizer(config, self.model)
        self.schedulers = get_scheduler(self.optimizer)

        self.model_output_folder = self.config['General']['path_model']
        self.path_model = os.path.join(self.config['General']['path_model'], 
            self.model.__class__.__name__ + 
            '_' + str(self.config['General']['exp_id']) + tm + '.pth')
        if config['General']['model_type'] == 'resunet':
            self.path_model = self.path_model.replace("Unet","ResUNet")

        print("self.path_model", self.path_model)
    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['General']['epochs']

        val_loss = inf
        es = EarlyStopping(10)
        for epoch in range(epochs):  # loop over the dataset multiple times
            print("Epoch ", epoch+1)
            running_loss = 0.0
            self.model.train()
            pbar = tqdm(train_dataloader)
            pbar.set_description("Training")
            for i, (X, Y_segmentations) in enumerate(pbar):
                # get the inputs; data is a list of [inputs, labels]
                X, Y_segmentations = X.to(self.device), Y_segmentations.to(self.device)
                # zero the parameter gradients

                self.optimizer.zero_grad()
                # forward + backward + optimizer

                output_segmentations = self.model(X)
                Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW
                # get loss
                loss = self.loss_segmentation(output_segmentations, Y_segmentations)
                loss.backward()
                # step optimizer
                self.optimizer.step()

                running_loss += loss.item()
                if np.isnan(running_loss):
                    print('\n',
                        X.min().item(), X.max().item(),'\n',
                        loss.item(),
                    )
                    exit(0)

                pbar.set_postfix({'training_loss': running_loss/(i+1)})

            new_val_loss = self.run_eval(val_dataloader)

            if new_val_loss < val_loss:
                self.save_model()
                val_loss = new_val_loss
                es.restartCounter()
            else:
                es.increaseCounter()

            self.schedulers[0].step(new_val_loss)

            if es.checkStopping() == True:
                print("Early stopping")
                print(es.counter, es.patience)
                print('Finished Training')
                # sys.exit(0)
                break

        print('Finished Training')

    def run_eval(self, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- val_dataloader -: torch dataloader
        """
        val_loss = 0.
        self.model.eval()
        X_1 = None
        Y_depths_1 = None
        Y_segmentations_1 = None
        output_depths_1 = None
        output_segmentations_1 = None

        all_preds = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")
            for i, (X, Y_segmentations) in enumerate(pbar):
                X, Y_segmentations = X.to(self.device), Y_segmentations.to(self.device)


                output_depths, output_segmentations = (None, self.model(X))

                output_depths = output_depths.squeeze(1) if output_depths != None else None
                Y_segmentations = Y_segmentations.squeeze(1)
                if i==0:
                    X_1 = X
                    Y_segmentations_1 = Y_segmentations
                    output_depths_1 = output_depths
                    output_segmentations_1 = output_segmentations
                # get loss
                loss = self.loss_segmentation(output_segmentations, Y_segmentations)
                val_loss += loss.item()

                preds = torch.argmax(output_segmentations, dim=1)
                
                all_preds.extend(preds.cpu().numpy().ravel())
                all_targets.extend(Y_segmentations.cpu().numpy().ravel())
                
                
                pbar.set_postfix({'validation_loss': val_loss/(i+1)})
        all_targets_array = np.array(all_targets, dtype=int)
        all_preds_array = np.array(all_preds, dtype=int)
        per_class_f1 = f1_score(all_targets_array, all_preds_array, average=None, zero_division=0)
        f1_str = " | ".join([f"Class {i}: {score:.4f}" for i, score in enumerate(per_class_f1)])
        print(f"Val F1 - {f1_str}")
        return val_loss/(i+1)


    def _save_model(self):
        path = os.path.join(self.path_model)
        print('Model saved at : {}'.format(path))        
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        torch.save(self.model.cpu().state_dict(), path)

    def save_model(self):
        create_dir(self.model_output_folder)
        torch.save(self.model.state_dict(), self.path_model)
        print('Model saved at : {}'.format(self.path_model))