# Test the BiGRU and SANDC models using an existing datasource to sanity check the implementation
#
# Data from https://github.com/google-research-datasets/tap-typing-with-touch-sensing-images
# Column structure: participant_id, task_id, trial_id, timestamp_ms, ref_char, ref_char_index_in_prompt, first_frame_touch_x, first_frame_touch_y, first_frame_touch_major, first_frame_touch_minor, first_frame_touch_orientation, first_frame_touch_heatmap, first_frame_heatmap_overlap_vector, was_deleted, lm_scores

import csv
import random
from pathlib import Path
import BiGRU
import SANCD
import torch
from torch.utils.data import DataLoader, Dataset


csv_path = Path(r"C:\Users\mno64\Datasets\tap-typing-with-touch-sensing-images\touch_data.csv")


def random_split(path):
	train_path = path.with_name(f"{path.stem}_train.csv")
	test_path = path.with_name(f"{path.stem}_test.csv")

	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Dataset not found: {path}")

	if not (train_path.exists() and test_path.exists()):
		with path.open("r", newline="", encoding="utf-8") as f:
			reader = csv.reader(f)
			rows = list(reader)

		if len(rows) < 2:
			raise ValueError(f"CSV file has no data rows: {path}")

		header = rows[0]
		data_rows = rows[1:]

		rng = random.Random(42)
		rng.shuffle(data_rows)

		split_idx = max(1, int(len(data_rows) * 0.8))
		train_rows = data_rows[:split_idx]
		test_rows = data_rows[split_idx:]


		for output_path, rows_to_write in [(train_path, train_rows), (test_path, test_rows)]:
			with output_path.open("w", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow(header)
				writer.writerows(rows_to_write)

	return train_path, test_path


class PoseDataset(Dataset):
	def __init__(self, path):
		self.path = Path(path)
		self.features = []
		self.labels = []

		with self.path.open("r", newline="", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			for row in reader:
				x = float(row[" first_frame_touch_x"]) / 1440.0
				y = float(row[" first_frame_touch_y"]) / 854.0
				feature = [x, y]
				label = row[" ref_char"]

				self.features.append(feature)
				self.labels.append(label)

		self.classes = sorted(set(self.labels))
		self.label_to_idx = {char: idx for idx, char in enumerate(self.classes)}
		self.targets = [self.label_to_idx[label] for label in self.labels]

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		# Column structure: participant_id, task_id, trial_id, timestamp_ms, ref_char, ref_char_index_in_prompt,
		# first_frame_touch_x, first_frame_touch_y, first_frame_touch_major, first_frame_touch_minor,
		# first_frame_touch_orientation, first_frame_touch_heatmap, first_frame_heatmap_overlap_vector,
		# was_deleted, lm_scores
		# ref_char is the label
		# (first_frame_touch_x/1440, first_frame_touch_y/854) is the feature
		# The other columns aren't needed

		x = torch.tensor(self.features[idx], dtype=torch.float32)
		y = torch.tensor(self.targets[idx], dtype=torch.long)
		return x, y


def main():
	train_path, test_path = random_split(csv_path)

	train_dataset = PoseDataset(train_path)
	test_dataset = PoseDataset(test_path)

	loader_kwargs = {"batch_size": 64, "num_workers": 2, "pin_memory": True}
	train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
	test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
	# NOTE: I wonder if there are issues from using the same training+test data for both models

	project_dir = Path(__file__).resolve().parent
	bigru_path = project_dir / "trained" / "test_bigru_weights.pth"
	sacnd_path = project_dir / "trained" / "test_sacnd_weights.pth"

	print("######### Training BiGRU")
	BiGRU.main(train_loader, test_loader, _validation_dataloader=None, validate=False, _save_path=bigru_path)
	print("######### Training SANDC")
	SANCD.main(train_loader, test_loader, _validation_dataloader=None, validate=False, _bigru_path=bigru_path, _save_path=sacnd_path)


if __name__ == "__main__":
	main()