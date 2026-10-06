import torch
from pathlib import Path
import numpy as np
import csv

SEED = 0

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


class GeometricDataset(torch.utils.data.Dataset):
	"""
	Data is in the format
	sentence, list of [normalised x coordinates], list of [normalised y coordinates]
	"""
	def __init__(self, csv_path):
		self.path = Path(csv_path)
		self.labels = []
		self.features = []

		with self.path.open("r", newline="", encoding="utf-8") as file:
			for row in csv.DictReader(file):

			# Each row contains comma-separated coordinates and its target character string.

				sentence = row['sentence'][:]
				x_list = np.asarray(row['x_list'].strip("[]").split(','), dtype=np.float32)
				y_list = np.asarray(row['y_list'].strip("[]").split(','), dtype=np.float32)
				if len(x_list) != len(y_list) or len(x_list) != len(sentence):
					raise ValueError(
						f"Row has mismatched lengths: "
						f"x={len(x_list)}, y={len(y_list)}, sentence={len(sentence)}"
					)

				coordinates = torch.tensor(np.column_stack((x_list, y_list)), dtype=torch.float32)
				char_list = torch.tensor([char_to_idx[char] for char in sentence], dtype=torch.long)

				self.labels.append(char_list)
				self.features.append(coordinates)
				

	def __getitem__(self, idx):
		return self.labels[idx], self.features[idx]

	def __len__(self):
		return len(self.labels)

def pad_variable(batch):
	# Each sample is (character labels, (x, y) features); models consume features first.
	sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
	labels = [sample[0] for sample in sorted_batch]
	features = [sample[1] for sample in sorted_batch]
	labels_padded = torch.nn.utils.rnn.pad_sequence(
		labels,
		batch_first=True,
		padding_value=-100,
	)
	features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)

	return labels_padded.long(), features_padded

####################################

class MaskedDataset(torch.utils.data.Dataset):
	"""Memory-mapped, pre-tokenized version of the 1 Billion Words benchmark dataset."""

	data_dir = Path(r"C:\Users\mno64\Datasets\1-billion-word-benchmark")
	train_path = data_dir / "train.txt"
	test_path = data_dir / "test.txt"

	def __init__(self, train, tokenizer, cache_dir=None, max_length=512):
		self.tokenizer = tokenizer
		self.vocab_size = len(tokenizer)
		self.max_length = max_length
		self.source_path = MaskedDataset.train_path if train else MaskedDataset.test_path
		self.cache_dir = (
			Path(cache_dir)
			if cache_dir is not None
			else Path(__file__).resolve().parent / "_cache"
		)
		name = "train" if train else "test"
		self.tokens_path = self.cache_dir / f"{name}_tokens.uint8"
		self.offsets_path = self.cache_dir / f"{name}_offsets.int64.npy"
		self.signature_path = self.cache_dir / f"{name}_vocab_signature.int64.npy"

		if not self._cache_is_valid():
			self._build_cache()

		self._tokens = None
		self._offsets = np.load(self.offsets_path, mmap_mode="r")
		self.lengths = np.diff(self._offsets).astype(np.int32, copy=False)

	def _cache_is_valid(self):
		if not (self.tokens_path.exists() and self.offsets_path.exists() and self.signature_path.exists()):
			return False
		try:
			signature = np.load(self.signature_path, allow_pickle=False)
			return signature.size == 1 and int(signature[0]) == self.vocab_size
		except Exception:
			return False

	def _build_cache(self):
		# Store token bytes and offsets separately so large corpora can be memory-mapped.
		self.cache_dir.mkdir(parents=True, exist_ok=True)
		# Invalidate stale cache files when the tokenizer vocabulary changes.
		for stale_path in (self.tokens_path, self.offsets_path, self.signature_path):
			if stale_path.exists():
				try:
					stale_path.unlink()
				except OSError:
					pass
		offsets = [0]
		with self.source_path.open("r", encoding="utf-8") as source, self.tokens_path.open("wb") as target:
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
	pass
	# TODO: Preprocess for contact

if __name__ == "__main__":
	preproces()