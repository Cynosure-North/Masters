import csv
import torch
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA

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

scaler = None
pca = None

def fit_scaler(train_data):
	vectors = frame_to_vectors(train_data)

	global scaler
	if scaler is None:
		scaler = StandardScaler()

	scaler.partial_fit(vectors)

def fit_pca(train_data, n_components=25):
	vectors = frame_to_vectors(train_data)
	scaled = scaler.transform(vectors)

	global pca
	if pca is None:
		pca = IncrementalPCA(n_components=min(n_components, scaled.shape[1]))

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

def load_preprocessor_state(state):
	params_path = Path(__file__).parent / "data" / "preprocessor_params"
	state = np.load(params_path, allow_pickle=True).item()
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

def frame_to_vectors(frame, test=False):
	if not hasattr(scaler, "mean_"):
		raise ValueError("Scaler has not been fitted")
	if not hasattr(pca, "components_"):
		raise ValueError("PCA has not been fitted")

	if test:
		# Use the midpoint between the two wrist markers.
		wrist = (frame[10] + frame[11]) / 2.0
		finger_pairs = (
			(frame[0], frame[1], wrist),  # Little
			(frame[2], frame[3], wrist),  # Ring
			(frame[4], frame[5], wrist),  # Middle
			(frame[6], frame[7], wrist),  # Index
			(frame[8], frame[9], wrist) ) # Thumb
	else:
		wrist_l, wrist_r = frame[0], frame[7]
		finger_pairs = None	# todo: I forgot to record knuckle positions...

	vectors = []
	for knuckle, fingertip, wrist in finger_pairs:
		# MF points from the knuckle to the fingertip; MW points to the wrist.
		mf = fingertip - knuckle
		mw = wrist - knuckle
		vectors.append(np.concatenate([mf, mw]))

	return np.asarray(vectors, dtype=np.float32).reshape(-1)

def _pose_adjustment(frame):
	return pca.transform(scaler.transform(frame_to_vectors(frame)))

def transform(frames):
	return np.vectorize(_pose_adjustment)(frames)

def interactive_transform(frame):
	return _pose_adjustment(frame)
