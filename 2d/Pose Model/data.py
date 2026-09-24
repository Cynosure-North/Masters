import os
import glob
import csv
import torch
from pathlib import Path
import numpy as np
import pandas as pd

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']
SEED = 0


class PoseDataset(torch.utils.data.Dataset):
	"""
	Data is in the format
	sentence, list of [25 principle component analysis output vectors] for each keypress
	One frame is marked as the pressing frame each time a key is pressed, that is what's recorded
	"""
	def __init__(self, path):
		pass
		# TODO: Dataset for pose

	def __len__(self):
		pass

	def __getitem__(self, idx):
		pass

####################################

def transform():
	pass
	# TODO: Transform for pose model

	# Data into vector format
		# To estimate the pose of each finger, we measured the angle ($$ \theta $$) between the hand joint vectors as a
		# representative metric. It is indicative of the degree of finger bending. The angle was calculated as
		# a cosine function as follows. (MW vectors point to the wrist, and MF to the fingertip)
		
		
		# $$ cos\theta_n = \frac{\overrightarrow{MW_n}; \overrightarrow{MF_n}}{\norm{\overrightarrow{MW_n}} \cross \norm{{\overrightarrow{MF_n}}}}, n \in \{\text{all fingers}\}$$
		
		
		# Exceptionally, there were no differences in finger angles between when entering the Y and U keys for
		# all fingers including the touching finger (p > 0.05). To differentiate these two keys, we were
		# required to analyze another hand characteristic that affects the global hand position, such as the
		# hand direction. We checked whether the hand direction was different depending on input keys using
		# the same analysis used for analyzing finger angles. The hand direction was estimated as follows
		
		
		# $$ \text{hand direction} = \frac{\overrightarrow{WM_{index}} + \overrightarrow{WM_{little}}}{\norm{\overrightarrow{WM_{index}} + \overrightarrow{WM_{little}}}} $$
	
	
	# normalise
		# During the first stage of the preprocessing, the Normalizer transformed each hand joint vector to
		# be a unit vector. As the Normalizer extracted the unit vectors independently from each hand size,
		# it reduced the variance of our typing data.
	
	# scale
		# Then, in the second stage, the StandardScaler made each feature of the unit vector follow
		# the standard normal distribution.															Why force things into a normal distribution?
	
	# pca, output 25 compontents
	# Unclear if this is per hand or not


def preprocess():
	data_folder = Path(__file__).parent.parent.joinpath("data")
	pass
	# TODO: preprocess for pose
	# Use finger assignment function in main


if __name__ == "__main__":
	preprocess()
