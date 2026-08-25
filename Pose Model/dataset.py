import os
import glob
import csv
import torch
from pathlib import Path
import numpy as np
import pandas as pd

data_folder = Path(__file__).parent.parent.joinpath("data")

generator = torch.Generator()
generator.seed(1234567890)

class PoseDataset(torch.utils.data.Dataset):
	def __init__(self, min_seq_len=8):

		self.text_column = "typed"
		self.feature_columns = [ "TODO" ]
		files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
		if not files:
			raise FileNotFoundError(f"No CSV files found in {data_folder}")

		frames = []
		for filename in files:
			frames.append(pd.read_csv(filename, sep='\t', skiprows=2, quoting=csv.QUOTE_NONE, low_memory=False))
		self.data = pd.concat(frames, axis=0, ignore_index=True)

		self.data[self.text_column] = self.data[self.text_column].fillna("").astype(str)
		self.data["uid"] = self.data["uid"].fillna("").astype(str)
		self.data["stimulus"] = self.data["stimulus"].fillna("").astype(str)
		self.data["condition"] = self.data["condition"].fillna("").astype(str)
		self.data["input_index"] = pd.to_numeric(self.data["input_index"], errors="coerce")
		self.data["time"] = pd.to_numeric(self.data["time"], errors="coerce")
		self.data["sample_id"] = self.data["uid"].astype(str) + "::" + self.data["stimulus"].astype(str) + "::" + self.data["condition"].astype(str)

		# TODO: Idk about this
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

			samples.append((feature_values, text_values[-1]))

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


dataset = PoseDataset()
train_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)], generator=generator)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0)

# TODO: Preprocess data once on load