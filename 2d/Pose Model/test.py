# Test the implementation using data from an existing source
#
# Data from https://userinterfaces.aalto.fi/how-we-type/resources/HowWeType_CHI16.pdf
# NOTE: as published the data is in csv files, despite being in a tsv format. I changed the file extensions to reflect that

import csv
from pathlib import Path
import numpy as np
from data import chars, transform, calculate_params, MotionDataset
import MLP

SEED = 0
dir_path = Path(r"C:\Users\mno64\Datasets\How we type\Motion Capture")


def random_split(path):
	path = Path(path)
	data_dir = path if path.is_dir() else path.parent
	train_path = data_dir / "train.csv"
	val_path = data_dir / "val.csv"
	test_path = data_dir / "test.csv"
	raw_path = data_dir / "raw.csv"
	output_paths = {train_path.resolve(), val_path.resolve(), test_path.resolve(), raw_path.resolve()}

	if not (train_path.exists() and val_path.exists() and test_path.exists()):
		if path.is_file():
			source_files = [path]
		elif data_dir.is_dir():
			source_files = sorted(
				file for file in data_dir.iterdir()
				if file.is_file()
				and file.suffix.lower() in {".csv", ".tsv"}
				and file.resolve() not in output_paths
			)
		else:
			raise FileNotFoundError(f"Input path does not exist: {path}")

		if not source_files:
			raise FileNotFoundError(f"No TSV/CSV source files found in: {data_dir}")

		fingers = [digit + str(phlange)
			for digit in "LRMIT"			# little, ring, middle, index, thumb
			for phlange in [1, 4]]		# 1 is knuckle, 4 is tip
		# The data I'm using in this study provides more markers locations than the model requires

		markers = [ f"{point}_{axis}"
			for point in fingers + ["Wout", "Win"]
			for axis in "xyz" ]

		right_columns = [
			f"Hands_R_{marker}"
			for marker in markers ]
		left_columns = [
			f"Hands_L_{marker}"
			for marker in markers ]

		my_finger_names = {
			"Hands_L_L4": "l_little",
			"Hands_L_R4": "l_ring",
			"Hands_L_M4": "l_middle",
			"Hands_L_I4": "l_index",
			"Hands_L_T4": "l_thumb",
			"Hands_R_L4": "r_little",
			"Hands_R_R4": "r_ring",
			"Hands_R_M4": "r_middle",
			"Hands_R_I4": "r_index",
			"Hands_R_T4": "r_thumb",
		}

		rows = []
		for source_file in source_files:
			print(f"processing {source_file}")
			with source_file.open('r', newline='', encoding="utf-8-sig") as source:
				next(source)	# Skip extra info
				next(source)
				reader = csv.DictReader(source, delimiter="\t")		# Despite being labeled csv they're actually tsv files
				required_columns = {
					"stimulus", "key_symbol", "finger", "right_hand",
					*right_columns, *left_columns,
				}
				missing = required_columns - set(reader.fieldnames or ())
				if missing:
					raise ValueError(
						f"{source_file} is missing columns: {', '.join(sorted(missing))}"
					)

				for row in reader:
					stimulus = row["stimulus"] or ""
					if any(char not in chars for char in stimulus):
						continue

					key_symbol = row["key_symbol"]
					if key_symbol not in chars:
						continue

					hand_value = (row["right_hand"] or "").strip().lower()
					if hand_value == '1':
						feature_columns = right_columns
					elif hand_value == '0':
						feature_columns = left_columns
					else:
						continue	 # No button was pressed this frame

					transformed_values = np.asarray(
						transform([row[column] for column in feature_columns]),
						dtype=float ).reshape(-1)

					if transformed_values.size != 30:
						raise ValueError(
							f"{source_file}, row {reader.line_num}: expected 30 features, "
							f"got {transformed_values.size}"
						)

					rows.append([key_symbol, my_finger_names[row["finger"]]] +
								transformed_values.tolist())

		print("data loaded")

		rows = np.array(rows)
		labels = np.asarray([row[:2] for row in rows], dtype=str)
		feature_matrix = np.asarray([row[2:] for row in rows], dtype=float)

		_, _, transformed_data = calculate_params(feature_matrix)
		transformed_data = np.asarray(transformed_data, dtype=float)

		if transformed_data.ndim != 2 or transformed_data.shape != (len(rows), 25):
			raise ValueError(
				f"Expected transformed features with shape ({len(rows)}, 25), "
				f"got {transformed_data.shape}"
			)

		rows = np.column_stack((labels, transformed_data))

		np.random.default_rng(SEED).shuffle(rows)

		train_end = int(len(rows) * 0.7)
		val_end = train_end + int(len(rows) * 0.15)

		output_header = ["label"] + ["finger"] + [f"feature_{index}" for index in range(25)]

		for output_path, split_rows in (
			(train_path, rows[:train_end]),
			(val_path, rows[train_end:val_end]),
			(test_path, rows[val_end:])):

			with output_path.open('w', encoding="utf-8", newline='') as output:
				writer = csv.writer(output, delimiter=",")
				writer.writerow(output_header)
				writer.writerows(split_rows)

		print("Processing complete")
	else:
		print("Loaded existing files")

	return train_path, val_path, test_path

def main():
	train_path, val_path, test_path = random_split(dir_path)

	project_dir = Path(__file__).resolve().parent
	save_path = project_dir / "trained" / "test_MLP_weights.pth"

	if True:
		MLP.main(train_path, val_path, test_path, save_path)
	else:
		print("MLP")
		loss, accuracy = test_model(MLP.instantiate_models(save_path), test_dataloader)
		print(f"Test Loss: {loss}")
		print(f"Test Accuracy: {accuracy:.4%}")
		

if __name__ == "__main__":
	main()

# TODO: The test accuracy for the pose model is way lower than was reported in the paper, I wonder if it's because of the dataset I'm using?