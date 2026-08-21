import os
import glob
import csv
import torch
import collections
from pathlib import Path
import numpy as np
import pandas as pd

data_folder = Path(__file__).parent.parent.joinpath("data")

# Load marker positions from the files in C:\Users\mno64\Datasets\How we type\Motion Capture and split into training and test sets
# files are in the format 005307_sentences_mocap_matched.csv
# they are tsv files, despite the extension
# the columns are uid	condition	time	stimulus_index	input_index	iki	input	key_symbol	stimulus	typed	finger	right_hand	Hands_L_Cout_x	Hands_L_Cout_y	Hands_L_Cout_z	Hands_R_Cout_x	Hands_R_Cout_y	Hands_R_Cout_z	Hands_L_R2_x	Hands_L_R2_y	Hands_L_R2_z	Hands_L_M1_x	Hands_L_M1_y	Hands_L_M1_z	Hands_L_I1_x	Hands_L_I1_y	Hands_L_I1_z	Hands_L_Cin_x	Hands_L_Cin_y	Hands_L_Cin_z	Hands_L_R1_x	Hands_L_R1_y	Hands_L_R1_z	Hands_L_L1_x	Hands_L_L1_y	Hands_L_L1_z	Hands_L_L2_x	Hands_L_L2_y	Hands_L_L2_z	Hands_L_Aout_x	Hands_L_Aout_y	Hands_L_Aout_z	Hands_L_I2_x	Hands_L_I2_y	Hands_L_I2_z	Hands_R_L2_x	Hands_R_L2_y	Hands_R_L2_z	Hands_L_M2_x	Hands_L_M2_y	Hands_L_M2_z	Hands_L_T2_x	Hands_L_T2_y	Hands_L_T2_z	Hands_R_R3_x	Hands_R_R3_y	Hands_R_R3_z	Hands_L_I3_x	Hands_L_I3_y	Hands_L_I3_z	Hands_L_R4_x	Hands_L_R4_y	Hands_L_R4_z	Hands_L_T1_x	Hands_L_T1_y	Hands_L_T1_z	Hands_L_R3_x	Hands_L_R3_y	Hands_L_R3_z	Hands_L_M4_x	Hands_L_M4_y	Hands_L_M4_z	Hands_L_L4_x	Hands_L_L4_y	Hands_L_L4_z	Hands_L_L3_x	Hands_L_L3_y	Hands_L_L3_z	Hands_L_M3_x	Hands_L_M3_y	Hands_L_M3_z	Hands_R_Win_x	Hands_R_Win_y	Hands_R_Win_z	Hands_R_Cin_x	Hands_R_Cin_y	Hands_R_Cin_z	Hands_R_L1_x	Hands_R_L1_y	Hands_R_L1_z	Hands_L_Win_x	Hands_L_Win_y	Hands_L_Win_z	Hands_L_Wout_x	Hands_L_Wout_y	Hands_L_Wout_z	Hands_R_I1_x	Hands_R_I1_y	Hands_R_I1_z	Hands_L_Ain_x	Hands_L_Ain_y	Hands_L_Ain_z	Hands_R_I2_x	Hands_R_I2_y	Hands_R_I2_z	Hands_R_L3_x	Hands_R_L3_y	Hands_R_L3_z	Hands_R_R2_x	Hands_R_R2_y	Hands_R_R2_z	Hands_R_R1_x	Hands_R_R1_y	Hands_R_R1_z	Hands_R_M3_x	Hands_R_M3_y	Hands_R_M3_z	Hands_R_Wout_x	Hands_R_Wout_y	Hands_R_Wout_z	Hands_R_Ain_x	Hands_R_Ain_y	Hands_R_Ain_z	Hands_R_T2_x	Hands_R_T2_y	Hands_R_T2_z	Hands_R_T3_x	Hands_R_T3_y	Hands_R_T3_z	Hands_R_T1_x	Hands_R_T1_y	Hands_R_T1_z	Hands_R_M1_x	Hands_R_M1_y	Hands_R_M1_z	Hands_R_M2_x	Hands_R_M2_y	Hands_R_M2_z	Hands_R_Aout_x	Hands_R_Aout_y	Hands_R_Aout_z	Hands_R_M4_x	Hands_R_M4_y	Hands_R_M4_z	Hands_R_L4_x	Hands_R_L4_y	Hands_R_L4_z	Hands_R_R4_x	Hands_R_R4_y	Hands_R_R4_z	Hands_L_I4_x	Hands_L_I4_y	Hands_L_I4_z	Hands_L_T4_x	Hands_L_T4_y	Hands_L_T4_z	Hands_R_I3_x	Hands_R_I3_y	Hands_R_I3_z	Hands_R_T4_x	Hands_R_T4_y	Hands_R_T4_z	Hands_R_I4_x	Hands_R_I4_y	Hands_R_I4_z	Hands_L_T3_x	Hands_L_T3_y	Hands_L_T3_z
# I will need key symbol and the columns from Hands_L_Cout_x to the right

class KeyPressDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir="C:\\Users\\mno64\\Datasets\\How we type\\Motion Capture", min_seq_len=8):
		def default():
			return "Float32"
		types = collections.defaultdict(default)
		types["input_index"] = pd.Int64Dtype()

		self.data_dir = data_dir
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
		files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
		if not files:
			raise FileNotFoundError(f"No CSV files found in {data_dir}")

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

# TODO: load dataset into numpy array (or other appropriate format)
# TODO: Deterministically split out the test data
# Could be easiest to randomly split on import and put them in different folders

test_data = ()

# Format
# particpant_id	timestamp	phrase_id	phrase	...hand positions