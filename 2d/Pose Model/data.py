import csv
import torch
from pathlib import Path
import numpy as np

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']
SEED = 0

finger_list = ["l_little", "l_ring", "l_middle", "l_index", "l_thumb", "r_little", "r_ring", "r_middle", "r_index", "r_thumb"]
finger_to_keys = {
	"l_little": "qaz",
	"l_ring": "qazwsxe",
	"l_middle": "wsxedcrx",
	"l_index": "edcrfvtgbyh",
	"l_thumb": " ",
	"r_thumb": " ",
	"r_index": "yhbujnikm",
	"r_middle": "uikm,ol",
	"r_ring": "ik,ol.p",
	"r_little": "lp." }

# key_to_finger = {}
# for char in 'abcdefghijklmnopqrstuvwxyz':
# 	tmp = []
# 	for key, value in finger_to_keys.items():
# 		if char in value:
# 			tmp.append(key)
# 	key_to_finger[char] = tmp

class PoseDataset(torch.utils.data.Dataset):
	"""Dataset of individual keypresses from a generated train/val/test CSV."""

	def __init__(self, path):
		self.path = Path(path)
		self.feature_columns = [f"feature_{i}" for i in range(25)]
		self.features = []
		self.fingers = []
		self.labels = []

		weird_finger_assignments = 0

		with self.path.open("r", newline="", encoding="utf-8-sig") as file:
			reader = csv.DictReader(file)
			required = {"label", "finger", *self.feature_columns}
			missing = required - set(reader.fieldnames or ())
			if missing:
				raise ValueError(
					f"{self.path} is missing columns: {', '.join(sorted(missing))}"
				)

			for row_number, row in enumerate(reader, start=2):
				label = row["label"]
				if label not in chars:
					raise ValueError(
						f"{self.path}, row {row_number}: unknown label {label!r}"
					)

				try:
					features = [float(row[column]) for column in self.feature_columns]
				except (TypeError, ValueError) as err:
					raise ValueError(
						f"{self.path}, row {row_number}: invalid feature value"
					) from err

				finger = row['finger']

				try:
					self.labels.append(finger_to_keys[finger].index(label))
					self.features.append(features)
					self.fingers.append(finger)
				except ValueError:
					weird_finger_assignments += 1
					print(f"Someone pressed {label} with {finger}")

		self.features = torch.tensor(self.features, dtype=torch.float32).reshape(-1, 25)
		self.labels = torch.tensor(self.labels, dtype=torch.long)

		print(f"There were {weird_finger_assignments} times someone pressed a button with an unusual finger, which is {(weird_finger_assignments/(len(self.labels) + weird_finger_assignments)):.2%} of the total dataset")

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		return self.labels[idx], self.fingers[idx], self.features[idx]


####################################

from sklearn.preprocessing import StandardScaler # pyright: ignore[reportMissingModuleSource]
from  sklearn.decomposition import PCA # pyright: ignore[reportMissingModuleSource]

scaler = StandardScaler()
pca = PCA(n_components=25)

def transform(data, online_scaling=False):
	"""
	Little, Ring, Index, Middle, Thumb - 0 is knuckle, 4 is tip of finger
	Input format: ['L0_x', 'L0_y', 'L0_z', 'L4_x', 'L4_y', 'L4_z', 'R0_x', 'R0_y', 'R0_z', 'R4_x', 'R4_y', 'R4_z', 'M0_x', 'M0_y', 'M0_z', 'M4_x', 'M4_y', 'M4_z', 'I0_x', 'I0_y', 'I0_z', 'I4_x', 'I4_y', 'I4_z', 'T0_x', 'T0_y', 'T0_z', 'T4_x', 'T4_y', 'T4_z', 'Wout_x', 'Wout_y', 'Wout_z', 'Win_x', 'Win_y', 'Win_z']
	"""
	
	try:
		points = np.asarray(data, dtype=float).reshape(12, 3)
	except (TypeError, ValueError) as err:
		raise ValueError("Expected 36 numeric coordinates (12 points * 3 axes)") from err

	# Use the midpoint between the two wrist markers.
	wrist = (points[10] + points[11]) / 2.0

	# Each pair contains a finger's knuckle and fingertip.
	finger_pairs = (
		(points[0], points[1]),  # Little
		(points[2], points[3]),  # Ring
		(points[4], points[5]),  # Middle
		(points[6], points[7]),  # Index
		(points[8], points[9]),  # Thumb
	)

	vectors = []
	for knuckle, fingertip in finger_pairs:
		# MF points from the knuckle to the fingertip; MW points to the wrist.
		mf = fingertip - knuckle
		mw = wrist - knuckle
		vectors += [[mf, mw]]

	if online_scaling:
		vectors = np.array(vectors)
		np.append(vectors, np.ones(5,), axis=-1)
		vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
		scaled = scaler.transform(data)
		return pca.transform(scaled)

	return vectors

def calculate_params(data):
	data = data.reshape((-1, 10, 3))

	# This constant tracks the changes made in the following steps
	# I'm unsure how useful it is
	np.concatenate((data, np.ones((data.shape[0], 10, 1))), axis=-1)

	# Normalise each vector
	data = data / np.linalg.norm(data, axis=1, keepdims=True)

	data = data.reshape(-1, 30)

	# Also unsure how necessary this step is but the paper does it
	scaler.fit(data)
	scaled = scaler.transform(data)
	
	pca.fit(scaled)
	pca_out = pca.transform(scaled)

	return scaler.get_params(), pca.get_params(), pca_out
	
####################################

def preprocess():
	data_folder = Path(__file__).parent.parent.joinpath("data")
	pass
	# TODO: preprocess for pose
	# 	Determine which finger pressed
	# 	Determine scale  + pca params from test data


if __name__ == "__main__":
	preprocess()
