import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from random import randint
import os
from pathlib import Path
import numpy as np
import torch

np.random.seed(1)
random.seed(1)
# Character IDs are shared by the geometric and semantic models.
chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
DATASET_DIR = Path(r"C:\Users\mno64\Datasets\1-billion-word-benchmark")


class DataVariableLength(Dataset):
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
	dataset = DataVariableLength(data_path, full_sentence=False, min_length=9, augment=False)
	return DataLoader(dataset, batch_size=batch_size, shuffle=not test, collate_fn=pad_variable)

####################################

class TokenizedBWDataset(Dataset):
	"""Memory-mapped, pre-tokenized version of the benchmark dataset."""

	train_path = DATASET_DIR / "train.txt"
	test_path = DATASET_DIR / "test.txt"

	def __init__(self, train, tokenizer, cache_dir=None, max_length=512):
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.source_path = TokenizedBWDataset.train_path if train else TokenizedBWDataset.test_path
		self.cache_dir = (
			Path(cache_dir)
			if cache_dir is not None
			else Path(__file__).resolve().parent / "tokenized_cache" )
		name = "train" if train else "test"
		self.tokens_path = os.path.join(self.cache_dir, f"{name}_tokens.uint8")
		self.offsets_path = os.path.join(self.cache_dir, f"{name}_offsets.int64.npy")

		if not (os.path.exists(self.tokens_path) and os.path.exists(self.offsets_path)):
			self._build_cache()

		self._tokens = None
		self._offsets = np.load(self.offsets_path, mmap_mode="r")
		self.lengths = np.diff(self._offsets).astype(np.int32, copy=False)

	def _build_cache(self):
		# Store token bytes and offsets separately so large corpora can be memory-mapped.
		os.makedirs(self.cache_dir, exist_ok=True)
		offsets = [0]
		with open(self.source_path, "r", encoding="utf-8") as source, open(self.tokens_path, "wb") as target:
			for line in source:
				sequence = self.tokenizer.encode(line)
				if len(sequence) > self.max_length:
					sequence = sequence[:self.max_length - 1] + [self.tokenizer.sep_token_id]
				target.write(bytes(sequence))
				offsets.append(offsets[-1] + len(sequence))
		np.save(self.offsets_path, np.asarray(offsets, dtype=np.int64))

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
		return torch.from_numpy(sequence)

	def __getstate__(self):
		state = self.__dict__.copy()
		state["_tokens"] = None
		return state