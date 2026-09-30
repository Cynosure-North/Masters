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
	val_path = path.with_name(f"{path.stem}_val.csv")
	test_path = path.with_name(f"{path.stem}_test.csv")

	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Dataset not found: {path}")

	if not (train_path.exists() and val_path.exists() and test_path.exists()):
		with path.open("r", newline="", encoding="utf-8-sig") as f:
			reader = csv.reader(f)
			rows = list(reader)

		header = rows[0]
		data_rows = rows[1:]

		rng = random.Random(0)
		rng.shuffle(data_rows)

		train_end = max(1, int(len(data_rows) * 0.7))
		val_end = max(train_end + 1, int(len(data_rows) * 0.85))

		train_rows = data_rows[:train_end]
		val_rows = data_rows[train_end:val_end]
		test_rows = data_rows[val_end:]

		for output_path, rows_to_write in [
			(train_path, train_rows),
			(val_path, val_rows),
			(test_path, test_rows),
		]:
			with output_path.open("w", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow(header)
				writer.writerows(rows_to_write)

	return train_path, val_path, test_path


class PoseDataset(Dataset):
	def __init__(self, path):
		self.path = Path(path)
		self.features = []
		self.labels = []

		with self.path.open("r", newline="", encoding="utf-8-sig") as f:
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
		x = torch.tensor(self.features[idx], dtype=torch.float32)
		y = torch.tensor(self.targets[idx], dtype=torch.long)
		return x, y


def main():
	train_path, val_path, test_path = random_split(csv_path)

	train_dataset = PoseDataset(train_path)
	val_dataset = PoseDataset(val_path)
	test_dataset = PoseDataset(test_path)

	loader_kwargs = {"batch_size": 64, "num_workers": 2, "pin_memory": True}
	train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
	val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
	test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

	# NOTE: I wonder if there are issues from using the same training+test data for both models

	project_dir = Path(__file__).resolve().parent
	bigru_path = project_dir / "trained" / "test_bigru_weights.pth"
	sacnd_path = project_dir / "trained" / "test_sacnd_weights.pth"

	print("######### Training BiGRU")
	BiGRU.main(train_loader, test_loader, _validation_dataloader=val_loader, _save_path=bigru_path)
	print("######### Training SANDC")
	SANCD.main(train_loader, test_loader, _validation_dataloader=val_loader, _bigru_path=bigru_path, _save_path=sacnd_path)


if __name__ == "__main__":
	main()