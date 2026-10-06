# Test the BiGRU and SANCD models using an existing datasource to sanity check the implementation
#
# Data from https://github.com/google-research-datasets/tap-typing-with-touch-sensing-images

import csv
import random
from pathlib import Path
import BiGRU
import SANCD

SEED = 0
csv_path = Path(r"C:\Users\mno64\Datasets\tap-typing-with-touch-sensing-images\touch_data.csv")

def random_split(path):
	train_path = path.with_name(f"{path.stem}_train.csv")
	val_path = path.with_name(f"{path.stem}_val.csv")
	test_path = path.with_name(f"{path.stem}_test.csv")

	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Dataset not found: {path}")

	if not (train_path.exists() and val_path.exists() and test_path.exists()):
		with path.open('r', newline='', encoding="utf-8-sig") as f:

			reader = csv.DictReader(f)
			rows = []

			sentence, xs, ys = "", [], []
			task_id, trial_id = "", ""

			for row in reader:
				task = row[" task_id"]
				trial = row[" trial_id"]
				if task_id == "": task_id, trial_id = task, trial

				x = float(row[" first_frame_touch_x"]) / 1440.0
				y = float(row[" first_frame_touch_y"]) / 854.0
				char = row[" ref_char"].strip()
				if char == "SPACE": char = ' '

				if task == task_id and trial == trial_id:
					sentence += char
					xs.append(x)
					ys.append(y)
				else:
					rows.append([sentence, xs, ys])
					sentence, xs, ys = char, [x], [y]
					task_id, trial_id = task, trial
			else:
				sentence += char
				xs.append(x)
				ys.append(y)
				rows.append([sentence, xs, ys])


		random.Random(SEED).shuffle(rows)

		train_end = max(1, int(len(rows) * 0.7))
		val_end = max(train_end + 1, int(len(rows) * 0.85))

		train_rows = rows[:train_end]
		val_rows = rows[train_end:val_end]
		test_rows = rows[val_end:]

		for output_path, split_rows in [
			(train_path, train_rows),
			(val_path, val_rows),
			(test_path, test_rows) ]:

			with output_path.open("w", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow(["sentence", "x_list", "y_list"])
				writer.writerows(split_rows)

	return train_path, val_path, test_path

def main():
	train_path, val_path, test_path = random_split(csv_path)

	project_dir = Path(__file__).resolve().parent
	bigru_path = project_dir / "trained" / "test_BiGRU_weights.pth"
	sacnd_path = project_dir / "trained" / "test_SACND_weights.pth"

	print("######### Training BiGRU")
	BiGRU.main(train_path, val_path, test_path, _save_path=bigru_path)

	# BERT trains on separate data (even in the final implementation), and can be tested on its own
	
	print("######### Training SANCD")
	SANCD.main(train_path, val_path, test_path, _bigru_path=bigru_path, _save_path=sacnd_path)


if __name__ == "__main__":
	main()