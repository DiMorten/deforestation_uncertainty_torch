import os
import random
from glob import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import pdb
import pandas as pd

# from src.Custom_augmentation import ToMask


class RasterDataset(Dataset):
	def __init__(self, input_image, reference, coords, config):
		self.reference = reference
		self.input_image = input_image
		self.coords = coords
		self.patch_size = config['Dataset']['patch_size']
		self.class_n = config['Dataset']['class_n']
		
		self.config = config
		self.split = config['split']

		assert (self.split in ['train', 'test', 'val']), "Invalid split!"


		# assert (len(self.paths_images) == len(self.paths_segmentations)), "Different number of instances between the input and the segmentation maps"
		# check for segmentation

		# Get the transforms
		self.transform_image, self.transform_seg, self.transform_augmentation = self.get_transforms()

		# get p_flip from config
		self.p_flip = config['Dataset']['transforms']['p_flip'] if self.split=='train' else 0
		self.p_crop = config['Dataset']['transforms']['p_crop'] if self.split=='train' else 0
		self.p_rot = config['Dataset']['transforms']['p_rot'] if self.split=='train' else 0
		self.resize = config['Dataset']['transforms']['resize']

	def get_transforms(self):

		transform_image = transforms.Compose([
			transforms.ToTensor()
		])
		transform_seg = transforms.Compose([
			transforms.ToTensor()
		])                           
		return transform_image, transform_seg, None
	
	def __len__(self):
		"""
			Function to get the number of images using the given list of images
		"""
		return len(self.coords)

	def __getitem__(self, idx):
		"""
			Getter function in order to get the triplet of images and segmentation masks
		"""
		if torch.is_tensor(idx):
			idx = idx.tolist()

		coord = self.coords[idx].astype(np.uint16)
		# batch_coords = np.squeeze(batch_coords.astype(np.uint16))
		
		# batch_img = np.zeros((self.patch_size, self.patch_size, self.input_image.shape[-1]), dtype = np.float32)

		batch_img = self.input_image[coord[0] : coord[0] + self.patch_size,
				coord[1] : coord[1] + self.patch_size] 
		batch_ref_int = self.reference[coord[0] : coord[0] + self.patch_size,
				coord[1] : coord[1] + self.patch_size]


	
		if np.random.rand()<0.3:
			batch_img = np.rot90(batch_img, 1)
			batch_ref_int = np.rot90(batch_ref_int, 1)
			
		if np.random.rand() >= 0.3 and np.random.rand() <= 0.5:
			batch_img = np.flip(batch_img, 0)
			batch_ref_int = np.flip(batch_ref_int, 0)
		
		if np.random.rand() > 0.5 and np.random.rand() <= 0.7:
			batch_img = np.flip(batch_img, 1)
			batch_ref_int = np.flip(batch_ref_int, 1)
			
		if np.random.rand() > 0.7:
			batch_img = batch_img
			batch_ref_int = batch_ref_int
		# batch_ref[i] = tf.keras.utils.to_categorical(batch_ref_int, self.class_n)
		# print("1 batch_img.shape: ", batch_img.shape)

		# I have shape (batch_size, patch_size, patch_size, 3). Change shape to (batch_size, 3, patch_size, patch_size)
		batch_img = np.transpose(batch_img, (2, 0, 1))
		batch_ref_int = np.transpose(batch_ref_int, (2, 0, 1))
  
		# print("2 batch_img.shape: ", batch_img.shape)
		# print("3 batch_ref_int.shape: ", batch_ref_int.shape)
		# transform batch_img to tensor
		batch_img = torch.from_numpy(batch_img.copy()).float()
		# transform batch_ref_int to tensor
		batch_ref_int = torch.from_numpy(batch_ref_int.copy()).long()

		# print("4 batch_img.dtype: ", batch_img.dtype)
		# print("5 batch_ref_int.dtype: ", batch_ref_int.dtype)
		# pdb.set_trace()
		return batch_img, batch_ref_int





		'''
		image = Image.open(self.paths_images[idx])
		segmentation = Image.open(self.paths_segmentations[idx])
		
		if self.split == 'train':
			# Apply train transformations
			state = torch.get_rng_state()
			image = self.transform_augmentation(image)
			torch.set_rng_state(state)
			segmentation = self.transform_augmentation(segmentation)

		image = self.transform_image(image)
		segmentation = self.transform_seg(segmentation)



		return image, segmentation
		'''