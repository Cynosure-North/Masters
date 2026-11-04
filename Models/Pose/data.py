import csv
import torch
from pathlib import Path
import numpy as np

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
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
	"r_middle": "uikmol",
	"r_ring": "ikolp",
	"r_little": "lp" }

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
		log_weird_presses = False

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
					if log_weird_presses: print(f"Someone pressed {label} with {finger}")

		self.features = torch.tensor(self.features, dtype=torch.float32).reshape(-1, 25)
		self.labels = torch.tensor(self.labels, dtype=torch.long)

		if log_weird_presses: print(f"There were {weird_finger_assignments} times someone pressed a button with an unusual finger, which is {(weird_finger_assignments/(len(self.labels) + weird_finger_assignments)):.2%} of the total dataset")

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		return self.labels[idx], self.fingers[idx], self.features[idx]


####################################

from sklearn.preprocessing import StandardScaler	# pyright: ignore[reportMissingModuleSource]
from sklearn.decomposition import PCA	# pyright: ignore[reportMissingModuleSource]

scaler = None
pca = None

def sample_to_vectors(sample):
	try:
		points = np.asarray(sample, dtype=float).reshape(12, 3)
	except (TypeError, ValueError) as err:
		raise ValueError("Expected 36 numeric coordinates (12 points * 3 axes)") from err

	# Use the midpoint between the two wrist markers.
	wrist = (points[10] + points[11]) / 2.0
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
		vectors.append(np.concatenate([mf, mw]))
	return np.asarray(vectors, dtype=np.float32).reshape(-1)

def fit_preprocessor(train_data, *, n_components=25):		# CONCERN: Maybe pipeline?
	scaler.fit(train_data)
	scaled = scaler.transform(train_data)

	pca = PCA(n_components=min(n_components, scaled.shape[1]))
	pca.fit(scaled)

def get_preprocessor_state():
	return {
		"scaler": {
			"mean": np.asarray(scaler.mean_, dtype=np.float32).tolist(),
			"scale": np.asarray(scaler.scale_, dtype=np.float32).tolist(),
			"var": np.asarray(scaler.var_, dtype=np.float32).tolist(),
		},
		"pca": {
			"mean": np.asarray(pca.mean_, dtype=np.float32).tolist(),
			"components": np.asarray(pca.components_, dtype=np.float32).tolist(),
			"explained_variance": np.asarray(pca.explained_variance_, dtype=np.float32).tolist(),
			"n_components": int(pca.n_components_),
		},
	}

def set_preprocessor_state(state):
	global scaler, pca

	scaler = StandardScaler()
	scaler.mean_ = np.asarray(state["scaler"]["mean"], dtype=np.float32)
	scaler.scale_ = np.asarray(state["scaler"]["scale"], dtype=np.float32)
	scaler.var_ = np.asarray(state["scaler"]["var"], dtype=np.float32)
	scaler.n_features_in_ = len(scaler.mean_)

	pca = PCA(n_components=int(state["pca"]["n_components"]))
	pca.mean_ = np.asarray(state["pca"]["mean"], dtype=np.float32)
	pca.components_ = np.asarray(state["pca"]["components"], dtype=np.float32)
	pca.explained_variance_ = np.asarray(state["pca"]["explained_variance"], dtype=np.float32)
	pca.n_features_in_ = len(pca.mean_)

def _pose_adjustment(frame):
	if not hasattr(scaler, "mean_"):
		raise ValueError("Scaler has not been fitted")
	if not hasattr(pca, "components_"):
		raise ValueError("PCA has not been fitted")

	scaled = scaler.transform(frame)
	return pca.transform(scaled)

def transform(frames, _left_hand, _right_hand):
	return np.vectorize(_pose_adjustment)(frames)	

def interactive_transform(frame, _left_hand, _right_hand):
	return _pose_adjustment(frame)

def preprocess():
	# data_folder = Path(__file__).parent.parent.parent / "Data" / "phase1"
	raw_data_folder = Path(r"H:\Research_Data\Morgana North(MHIT)\phase1")
	processed_data_folder = Path(__file__).parent / "Data" / "phase1"

	processed_data_folder.mkdir(exist_ok=True)

	for subject in raw_data_folder.iterdir():
		for surface in subject.iterdir():
			pass
			
	# Iterate through dirs and subdirs,
	# Parse to python lists
	# filter out 2% most extreme lengths (max and mix) (different values for calibration and typing)
	# for each surface calibrate surface 
	# Run preprocessing on each phrase
	# save to corresponding dir in 
	
	# TODO: preprocess for pose
	# 	Determine which finger pressed
	# 	Determine scale  + pca params from test data


if __name__ == "__main__":
	preprocess()
