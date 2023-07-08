import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from src.Trainer import Trainer
from src.dataset import RasterDataset

import time
import pdb
import copy

import src.utils as utils

import pandas as pd
import os


with open('config.json', 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

dataset_name = config['Dataset']['paths']['path_dataset']


print(config['Dataset']['splits'])



config['seed'] = config['General']['seed']

## train set
# =============================================================================
dataset_config = config   
dataset_config['split'] = 'train'

train_data = RasterDataset(dataset_config)
train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True, 
    drop_last=True)

## validation set
# =============================================================================
dataset_config = config       
dataset_config['split'] = 'val'

val_data = RasterDataset(dataset_config)
val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True)
# =============================================================================

trainer = Trainer(config)
t0 = time.time()
trainer.train(train_dataloader, val_dataloader)
print(time.time() - t0)