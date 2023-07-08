import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import cv2
import torch.nn as nn

from tqdm import tqdm
from os import replace
from numpy.core.numeric import Inf

from src.utils import get_losses, get_optimizer, get_schedulers, create_dir

import sys

import segmentation_models_pytorch as smp

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.nn.functional as F
import pdb

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_loss():

    weights = [0.1, 0.9, 0]            
    class_weights = torch.FloatTensor(weights).cuda()
    loss_segmentation = nn.CrossEntropyLoss(weight=class_weights)
    return loss_segmentation


def get_optimizer(config, net):

    if config['General']['optim'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=config['General']['lr'])
    elif config['General']['optim'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=config['General']['lr'], momentum=config['General']['momentum'])
    return optimizer

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
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = self.config['General']['type']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']


        if config['General']['model_type'] == 'unet':        
            
            self.model = smp.Unet('xception', encoder_weights='imagenet', in_channels=self.config['Train']['input_bands'],
                encoder_depth=4, decoder_channels=[128, 64, 32, 16], classes=self.config['Dataset']['class_n'])
        elif config['General']['model_type'] == 'deeplab': # use this one        
            
            self.model = smp.DeepLabV3Plus('tu-xception41', encoder_weights='imagenet', 
                                        in_channels=self.config['Train']['input_bands'],
                                        classes=self.config['Dataset']['class_n'])

                
        self.model.to(self.device)

        # print(self.model)

        # print(self.model.encoder.model.blocks_1.stack)

        # pdb.set_trace()
        # exit(0)
        # print("input shape: ", (3,resize,resize))
        # print(resize)
        # summary(self.model, (3,resize,resize))
        # exit(0)

        self.loss_depth, self.loss_segmentation = get_loss(config)
        self.optimizer = get_optimizer(config, self.model)
        self.optimizer = ReduceLROnPlateau(self.optimizer)

        self.path_model = os.path.join(self.config['General']['path_model'], 
            self.model.__class__.__name__ + 
            '_' + str(self.config['General']['exp_id']))

            
    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['General']['epochs']

        val_loss = Inf
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

                self.optimizer_scratch.zero_grad()
                # forward + backward + optimizer

                output_segmentations = self.model(X)
                Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW
                # get loss
                loss = self.loss_segmentation(output_segmentations, Y_segmentations)
                loss.backward()
                # step optimizer
                self.optimizer_scratch.step()

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
                exit(0)

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
                pbar.set_postfix({'validation_loss': val_loss/(i+1)})

        return val_loss/(i+1)

    def save_model(self):
        
        create_dir(self.path_model)
        torch.save({'model_state_dict': self.model.state_dict(),
                    # 'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                    'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                    }, self.path_model+'.p')
        print('Model saved at : {}'.format(self.path_model))