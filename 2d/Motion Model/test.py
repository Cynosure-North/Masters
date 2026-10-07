# Test the implementation using data from an existing source
#
# Data from https://userinterfaces.aalto.fi/how-we-type/resources/HowWeType_CHI16.pdf
# NOTE: as published the data is in csv files, despite being in a tsv format. I changed the file extensions to reflect that

import editdistance
from collections import defaultdict
import random
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import chars, transform, MotionDataset
from BeamSearch import prefix_beam_search
import TCN
import LLM

SEED = 0
dir_path = Path(r"C:\Users\mno64\Datasets\How we type\Motion Capture")

fingers = [digit + "4"				# They only use the fingertips
	for digit in "LRMIT"]			# little, ring, middle, index, thumb
markers = [ f"Hands_{hand}_{point}_{axis}"
	for hand in "LR"
	for point in fingers + ["Wout", "Win", "Aout", "Ain"]
	for axis in "xyz" ]

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


def random_split(path):
	path = Path(path)
	data_dir = path if path.is_dir() else path.parent

	train_path = data_dir / "train"
	val_path = data_dir / "validation"
	test_path = data_dir / "test"
	output_dirs = (train_path, val_path, test_path)

	if not all(directory.is_dir() and any(directory.glob("*.csv")) for directory in output_dirs):
		if path.is_file():
			source_files = [path]
		elif data_dir.is_dir():
			source_files = sorted(
				file for file in data_dir.iterdir()
				if file.is_file()
				and file.suffix.lower() == ".tsv"
			)
		else:
			raise FileNotFoundError(f"Input path does not exist: {path}")
		if not source_files:
			raise FileNotFoundError(f"No TSV source files found in: {data_dir}")

		rows = []
		for source_file in source_files:
			print(f"processing {source_file}")
			with source_file.open('r', newline='', encoding="utf-8-sig") as source:
				next(source)	# Skip extra info
				next(source)
				reader = csv.DictReader(source, delimiter="\t")		# Despite being labeled csv they're actually tsv files

				required_columns = { "uid", "stimulus", *markers }
				missing = required_columns - set(reader.fieldnames or ())
				if missing:
					raise ValueError( f"{source_file} is missing columns: {', '.join(sorted(missing))}")

				batch = []
				prev_uid, prev_stimulus, prev_frame = None, None, None

				def save_batch():
					if prev_stimulus is not None:
						rows.append([prev_stimulus, batch])

				for row in reader:
					uid = row['uid']
					label = (row['stimulus'] or "").strip().lower()
					if any(char not in chars for char in label):
						continue

					transformed_values = transform([row[column] for column in markers])

					if uid != prev_uid or label != prev_stimulus:
						save_batch()
						prev_uid, prev_stimulus = uid, label
						prev_frame = transformed_values
						batch = []
						continue

					deltas = np.reshape((transformed_values - prev_frame), -1).tolist()
					prev_frame = transformed_values
					batch.append(deltas)

				save_batch()


		print("data loaded")
		if not rows:
			raise ValueError("No valid motion sequences were found in the input data")

		random.Random(SEED).shuffle(rows)

		train_end = int(len(rows) * 0.7)
		val_end = train_end + int(len(rows) * 0.15)

		for output_dir in output_dirs:
			output_dir.mkdir(parents=True, exist_ok=True)
			for stale_file in output_dir.glob("*.csv"):
				stale_file.unlink()

		feature_header = [f"feature_{index}" for index in range(len(markers))]
		for output_dir, split_rows in (
			(train_path, rows[:train_end]),
			(val_path, rows[train_end:val_end]),
			(test_path, rows[val_end:])):
			name_counts = defaultdict(lambda: 0)
			for stimulus, timesteps in split_rows:
				name_counts[stimulus] += 1
				duplicate_suffix = "" if name_counts[stimulus] == 1 else f"_{name_counts[stimulus]}"
				output_path = output_dir / f"{stimulus}{duplicate_suffix}.csv"
				with output_path.open("w", newline="", encoding="utf-8") as f:
					writer = csv.writer(f)
					writer.writerow(feature_header)
					writer.writerows(timesteps)

		print("Processing complete")
	else:
		print("Loaded existing split directories")

	return train_path, val_path, test_path

def main():
	project_dir = Path(__file__).resolve().parent
	train_path, val_path, test_path = random_split(dir_path)
	llm_path = project_dir / "downloaded" / "gemma-4-e2b-q4_k_m.gguf"
	save_path = project_dir / "trained" / "test_TCN_weights.pth"

	if True:
		print("######### Training TCN")
		TCN.main(train_path, val_path, test_path, save_path)
	
	print("######### Testing TCN with CTC prefix beam search")
	tcn = TCN.instantiate_model(save_path).eval()
	llm = LLM.LLM(llm_path)
	loader = DataLoader(MotionDataset(test_path), shuffle=False)

	incorrect_chars = 0
	total_chars = 0
	incorrect_sequences = 0
	total_sequences = 0

	with torch.inference_mode():
		for (features, [label], len_label) in loader:
			predicted_text = prefix_beam_search(
				tcn(features.to(next(tcn.parameters()).device)).tolist()[0],
				llm.get_word_probability)
			expected_text = "".join(chars[token - 1] for token in label.tolist())

			incorrect_chars += editdistance.eval(expected_text, predicted_text)
			total_chars += len_label.item()
			incorrect_sequences += int(expected_text != predicted_text)
			total_sequences += 1

	character_accuracy = max(0.0, 1.0 - incorrect_chars / total_chars)
	word_accuracy = (max(0.0, 1.0 - incorrect_sequences / total_sequences))

	print(
		f"Character accuracy: {character_accuracy:.2%} "
		f"({incorrect_chars} edit(s) / {total_chars} reference characters)"
	)
	print(
		f"Sequence: {word_accuracy:.2%} "
		f"({incorrect_sequences} edit(s) / {total_sequences} reference sequences)"
	)

if __name__ == "__main__":
	main()