import os
import glob
import csv
import torch
from pathlib import Path
import numpy as np
import pandas as pd

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']

SEED = 0

class MotionDataset(torch.utils.data.Dataset):
	"""
	Data is in the format
	sentence, list of [list of [x, z, y] coordinates for each marker] for each frame
	"""

	def __init__(self, path):
		# TODO: Dataset for motion
		# labels in the format [a, b, c, ..., -]
		pass

	def __len__(self):
		pass

	def __getitem__(self, idx):
		pass

####################################

def transform():
	pass
	# TODO: transform for motion
	# The input features to the network are frame-to-frame deltas of wrist position and rotation along 
	# with 3D fingertip positions. All positions are represented in the coordinate frame of the keyboard

def preprocess(path):
	pass
	# TODO: preprocess for motion

if __name__ == "__main__":
	preprocess()