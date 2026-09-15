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
	def __init__(self, path):
		self.path = Path(path)
		if not self.path.exists():
			raise FileNotFoundError(f"Pose data file not found: {self.path}")
		frame = pd.read_csv(self.path)
		if {"key", "finger"}.issubset(frame.columns):
			feature_columns = [column for column in frame.columns if str(column).lower().startswith(("pc", "principal"))]
			if len(feature_columns) != 25:
				feature_columns = list(frame.columns[2:27])
			key_column, finger_column = "key", "finger"
		else:
			if frame.shape[1] < 27:
				raise ValueError("Pose data must contain key, finger, and 25 PCA columns")
			key_column, finger_column = frame.columns[:2]
			feature_columns = list(frame.columns[2:27])

		self.keys = frame[key_column].astype(str).tolist()
		self.fingers = frame[finger_column].astype(str).tolist()
		self.features = torch.tensor(
			frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
			dtype=torch.float32,
		)
		if self.features.shape[1] != 25:
			raise ValueError(f"Expected 25 PCA features, found {self.features.shape[1]}")

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, idx):
		return self.keys[idx], self.fingers[idx], self.features[idx]

####################################

def preprocess():
	data_folder = Path(__file__).parent.parent.joinpath("data")
	pass
	# TODO


if __name__ == "__main__":
	preprocess()

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
	# the standard normal distribution.															Why force things into a normal distribution

# pca, output 25 compontents