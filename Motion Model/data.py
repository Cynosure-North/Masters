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
	def __init__(self, path, min_seq_len=8):

		self.text_column = "typed"
		self.feature_columns = [
			"Hands_L_Cout_x", "Hands_L_Cout_y", "Hands_L_Cout_z",
			"Hands_R_Cout_x", "Hands_R_Cout_y", "Hands_R_Cout_z",
			"Hands_L_Wout_x", "Hands_L_Wout_y", "Hands_L_Wout_z",
			"Hands_R_Wout_x", "Hands_R_Wout_y", "Hands_R_Wout_z",
			"Hands_L_Ain_x", "Hands_L_Ain_y", "Hands_L_Ain_z",
			"Hands_R_Ain_x", "Hands_R_Ain_y", "Hands_R_Ain_z",
			"Hands_L_Win_x", "Hands_L_Win_y", "Hands_L_Win_z",
			"Hands_R_Win_x", "Hands_R_Win_y", "Hands_R_Win_z",
		]
		files = sorted(glob.glob(os.path.join(path, "*.csv")))
		if not files:
			raise FileNotFoundError(f"No CSV files found in {path}")

		use_cols = ["uid", "condition", "stimulus", "input_index", "time", self.text_column] + self.feature_columns
		frames = []
		for filename in files:
			frames.append(pd.read_csv(filename, sep='\t', skiprows=2, usecols=use_cols, quoting=csv.QUOTE_NONE, low_memory=False))
		self.data = pd.concat(frames, axis=0, ignore_index=True)

		self.data[self.text_column] = self.data[self.text_column].fillna("").astype(str)
		self.data["uid"] = self.data["uid"].fillna("").astype(str)
		self.data["stimulus"] = self.data["stimulus"].fillna("").astype(str)
		self.data["condition"] = self.data["condition"].fillna("").astype(str)
		self.data["input_index"] = pd.to_numeric(self.data["input_index"], errors="coerce")
		self.data["time"] = pd.to_numeric(self.data["time"], errors="coerce")
		self.data["sample_id"] = self.data["uid"].astype(str) + "::" + self.data["stimulus"].astype(str) + "::" + self.data["condition"].astype(str)

		samples = []
		for _, group in self.data.groupby("sample_id", sort=False):
			group = group.sort_values(by=["input_index", "time"], kind="mergesort")
			text_values = [str(value).strip() for value in group[self.text_column].tolist() if str(value).strip()]
			if not text_values:
				continue

			feature_values = group[self.feature_columns].to_numpy(dtype="float32")
			feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
			if len(feature_values) < min_seq_len:
				continue

			# Frame-to-frame deltas, matching the paper's motion feature description.
			deltas = np.diff(feature_values, axis=0, prepend=feature_values[:1], append=feature_values[-1:])
			samples.append((deltas, text_values[-1]))

		self.samples = samples
		self.vocabulary = ["<blank>", "<unk>"] + sorted({char for _, text in samples for char in text})
		self.char_to_index = {char: idx for idx, char in enumerate(self.vocabulary)}
		self.num_features = len(self.feature_columns)
		print(f"Loaded {len(self.samples)} typed-text samples with {len(self.vocabulary)} symbols")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		features, target_text = self.samples[idx]
		encoded = torch.tensor([self.char_to_index.get(char, self.char_to_index["<unk>"]) for char in target_text], dtype=torch.long)
		return torch.from_numpy(features).float(), encoded, torch.tensor(len(encoded), dtype=torch.long)

####################################

def preprocess():
	pass
	# TODO
	# The input features to the network are frame-to-frame deltas of wrist position and rotation along 
	# with 3D fingertip positions. All positions are represented in the coordinate frame of the keyboard

if __name__ == "__main__":
	preprocess()