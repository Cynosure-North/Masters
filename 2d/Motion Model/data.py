import re
import torch
from pathlib import Path
import numpy as np

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']
SEED = 0

class MotionDataset(torch.utils.data.Dataset):
	def __init__(self, path):
		self.path = Path(path)
		if not self.path.exists():
			raise FileNotFoundError(f"Motion dataset not found: {self.path}")

		self.files = sorted(self.path.glob("*.csv")) if self.path.is_dir() else None
		if self.files is None:
			raise ValueError(f"Motion dataset directory is empty: {self.path}")

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		file_path = self.files[idx]
		label = re.sub(r"_\d+$", "", file_path.stem).lower()
		try:
			features = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
		except (OSError, ValueError) as exc:
			raise ValueError(f"Could not load numeric motion features from {file_path}") from exc

		if features.ndim != 2 or features.shape[0] == 0:
			raise ValueError(f"Motion record {idx} must contain at least one feature frame")

		# Class zero is reserved for the CTC blank token.
		targets = torch.tensor([chars.index(char) +1 for char in label], dtype=torch.long)
		return torch.from_numpy(features), targets, len(targets)

####################################

def transform(data):
	data = np.asarray(data, dtype=float).reshape(2, 9, 3)

	# The final four markers per hand are Wout, Win, Aout, and Ain.
	wrist_positions = (data[:, 5] + data[:, 6]) / 2.0
	arm_positions = (data[:, 7] + data[:, 8]) / 2.0
	rotation_vectors = arm_positions - wrist_positions
	lengths = np.linalg.norm(rotation_vectors, axis=1, keepdims=True)
	wrist_rotations = rotation_vectors / lengths

	return np.hstack((data[:, :5], [wrist_positions, wrist_rotations]))
	



def preprocess(path):
	pass
	# TODO: preprocess for motion

if __name__ == "__main__":
	preprocess()