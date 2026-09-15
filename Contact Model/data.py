import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from random import randint
import os
from pathlib import Path
import numpy as np
import csv

SEED = 0

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


class VariableLengthDataset(Dataset):
	"""Data loader for the long-term decoder.
	DataLongTerm parses the raw data and extracts
	(seq. of user input, g.t. seq. of characters) pairs of each full sentence.
	"""
	def __init__(self, csv_path, min_length=13, full_sentence=False, augment=False):
		df = pd.read_csv(csv_path)
		self.full_sentence = full_sentence
		self.min_length = min_length
		data_file = df[df['length'] >= self.min_length]
		self.df = data_file
		self.data_length = self.df.shape[0]
		self.augment = augment

	def process_csv(self, data, idx):
		# Each row contains comma-separated coordinates and its target character string.
		sample = data.iloc[idx]

		sentence = sample['sentence'][:]
		x_list = np.asarray(sample['x_list'].split(','), dtype=np.float32)
		y_list = np.asarray(sample['y_list'].split(','), dtype=np.float32)
		if len(x_list) != len(y_list) or len(x_list) != len(sentence):
			raise ValueError(
				f"Row {idx} has mismatched lengths: "
				f"x={len(x_list)}, y={len(y_list)}, sentence={len(sentence)}"
			)

		coordinates = torch.tensor(np.column_stack((x_list, y_list)), dtype=torch.float32)
		char_list = torch.tensor(np.array([char_to_idx[char] for char in sentence]))

		return coordinates, char_list

	def __getitem__(self, idx):
		data_arr, label = self.process_csv(self.df, idx)
		return data_arr, label

	def __len__(self):
		return self.data_length

def pad_variable(batch):
	# Padding lets a batch contain variable-length sentences while preserving time order.
	sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)             # sort the batch in descending order
	sequences = [x[0] for x in sorted_batch]                                            # length of sequence
	sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)     # pad the sequence
	labels = [x[1] for x in sorted_batch]
	labels_padded = torch.nn.utils.rnn.pad_sequence(
		labels,
		batch_first=True,
		padding_value=-100,
	)

	if len(sorted_batch[0]) > 2:
		full_labels = [x[3] for x in sorted_batch]
		full_labels_padded = torch.nn.utils.rnn.pad_sequence(
			full_labels,
			batch_first=True,
			padding_value=-100,
		)

	else:
		full_labels_padded = labels_padded

	return sequences_padded, full_labels_padded.long()

def get_dataloader(data_path, batch_size, test=False):
	dataset = VariableLengthDataset(data_path, full_sentence=False, min_length=9, augment=False)
	return DataLoader(dataset, batch_size=batch_size, shuffle=not test, collate_fn=pad_variable)

####################################

class MaskedDataset(Dataset):
	"""Memory-mapped, pre-tokenized version of the benchmark dataset."""

	DATASET_DIR = Path(r"C:\Users\mno64\Datasets\1-billion-word-benchmark")
	train_path = DATASET_DIR / "train.txt"
	test_path = DATASET_DIR / "test.txt"

	def __init__(self, train, tokenizer, cache_dir=None, max_length=512):
		self.tokenizer = tokenizer
		self.vocab_size = len(tokenizer)
		self.max_length = max_length
		self.source_path = MaskedDataset.train_path if train else MaskedDataset.test_path
		self.cache_dir = (
			Path(cache_dir)
			if cache_dir is not None
			else Path(__file__).resolve().parent / "_cache" )
		name = "train" if train else "test"
		self.tokens_path = os.path.join(self.cache_dir, f"{name}_tokens.uint8")
		self.offsets_path = os.path.join(self.cache_dir, f"{name}_offsets.int64.npy")
		self.signature_path = os.path.join(self.cache_dir, f"{name}_vocab_signature.int64.npy")

		if not self._cache_is_valid():
			self._build_cache()

		self._tokens = None
		self._offsets = np.load(self.offsets_path, mmap_mode="r")
		self.lengths = np.diff(self._offsets).astype(np.int32, copy=False)

	def _cache_is_valid(self):
		if not (os.path.exists(self.tokens_path) and os.path.exists(self.offsets_path) and os.path.exists(self.signature_path)):
			return False
		try:
			signature = np.load(self.signature_path, allow_pickle=False)
			return signature.size == 1 and int(signature[0]) == self.vocab_size
		except Exception:
			return False

	def _build_cache(self):
		# Store token bytes and offsets separately so large corpora can be memory-mapped.
		os.makedirs(self.cache_dir, exist_ok=True)
		# Invalidate stale cache files when the tokenizer vocabulary changes.
		for stale_path in (self.tokens_path, self.offsets_path, self.signature_path):
			if os.path.exists(stale_path):
				try:
					os.remove(stale_path)
				except OSError:
					pass
		offsets = [0]
		with open(self.source_path, "r", encoding="utf-8") as source, open(self.tokens_path, "wb") as target:
			for line in source:
				sequence = self.tokenizer.encode(line)
				if len(sequence) > self.max_length:
					sequence = sequence[:self.max_length - 1] + [self.tokenizer.sep_token_id]
				target.write(bytes(sequence))
				offsets.append(offsets[-1] + len(sequence))
		np.save(self.offsets_path, np.asarray(offsets, dtype=np.int64))
		np.save(self.signature_path, np.asarray([self.vocab_size], dtype=np.int64))

	def _ensure_tokens_open(self):
		if self._tokens is None:
			self._tokens = np.memmap(self.tokens_path, mode="r", dtype=np.uint8)

	def __len__(self):
		return len(self._offsets) - 1

	def __getitem__(self, idx):
		# Open the memory map lazily; this is important when DataLoader creates workers.
		self._ensure_tokens_open()
		start, end = self._offsets[idx], self._offsets[idx + 1]
		sequence = np.array(self._tokens[start:end], dtype=np.uint8, copy=True)
		if sequence.size and int(sequence.max()) >= self.vocab_size:
			raise ValueError(
				f"Cached token sequence for record {idx} contains invalid token IDs: "
				f"max={int(sequence.max())}, vocab_size={self.vocab_size}. "
				"Delete the cache files under the Contact Model/_cache folder and rerun."
			)
		return torch.from_numpy(sequence)

	def __getstate__(self):
		state = self.__dict__.copy()
		state["_tokens"] = None
		return state

####################################

def preproces():
	DATA_DIR = Path(__file__).resolve().parent / "data" / "geometric"
	INPUT_FILE = DATA_DIR / "IMK_data.csv"
	TRAIN_FILE = DATA_DIR / "train.csv"
	VALIDATION_FILE = DATA_DIR / "validation.csv"
	TEST_FILE = DATA_DIR / "test.csv"

	TEST_VAL_RATIO = 0.1
	SEED = 0


	def normalize_coordinates(values, divisor):
		# Clamp normalized coordinates so malformed measurements cannot leave the [0, 1] range.
		result = []

		for value in values.split(","):
			normalized = float(value) / divisor
			normalized = max(0.0, min(1.0, normalized))
			result.append(f"{normalized:.12f}")

		return ",".join(result)


	with INPUT_FILE.open("r", newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		rows = list(reader)
		fieldnames = reader.fieldnames

	for row in rows:
		width = float(row["width"])
		height = float(row["height"])

		row["x_list"] = normalize_coordinates(row["x_list"], width)
		row["y_list"] = normalize_coordinates(row["y_list"], height)

	random.Random(SEED).shuffle(rows)

	test_size = int(len(rows) * TEST_VAL_RATIO)
	val_rows = rows[:test_size]
	test_rows = rows[test_size:test_size*2]
	train_rows = rows[test_size*2:]


	def write_csv(path, data):
		with path.open("w", newline="", encoding="utf-8") as file:
			writer = csv.DictWriter(file, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(data)


	write_csv(TRAIN_FILE, train_rows)
	write_csv(VALIDATION_FILE, val_rows)
	write_csv(TEST_FILE, test_rows)

	print(f"Training rows: {len(train_rows)}")
	print(f"Validation rows: {len(val_rows)}")
	print(f"Test rows: {len(test_rows)}")
	print(f"Written: {TRAIN_FILE}, {VALIDATION_FILE}, {TEST_FILE}")

if __name__ == "__main__":
	preproces()