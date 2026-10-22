"""Character-level language model compatible with BeamSearcher."""

import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import chars


class CharacterLanguageModel(nn.Module):
	"""Small recurrent character LM with length-normalized log scoring."""

	def __init__(self, *, hidden_size=256, num_layers=2, dropout=0.1):
		super().__init__()
		self.characters = tuple(chars)
		self.bos_id = len(self.characters)
		self.vocab_size = self.bos_id + 1
		self.embedding = nn.Embedding(self.vocab_size, hidden_size)
		self.rnn = nn.GRU(
			hidden_size,
			hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		self.output = nn.Linear(hidden_size, self.vocab_size)
		self.device = torch.device("cpu")
		self.inference_dtype = torch.float32
		self._score_cache = OrderedDict()
		self._cache_size = 4096

	@property
	def config(self):
		return type("Config", (), {"vocab_size": self.vocab_size})()

	@staticmethod
	def filter_text(text):
		return " ".join(
			"".join(character if character in chars else " " for character in str(text).lower()).split()
		)

	def to(self, *args, **kwargs):
		result = super().to(*args, **kwargs)
		parameters = list(self.parameters())
		if parameters:
			self.device = parameters[0].device
		self.inference_dtype = (
			torch.float16 if self.device.type == "cuda" else torch.float32
		)
		return result

	def _autocast(self):
		return torch.autocast(
			device_type=self.device.type,
			dtype=self.inference_dtype,
			enabled=self.device.type == "cuda",
		)

	def forward(self, token_ids, hidden=None):
		embedded = self.embedding(token_ids)
		output, hidden = self.rnn(embedded, hidden)
		return self.output(output), hidden

	def _ids(self, text):
		return [self.characters.index(character) for character in text]

	def get_word_log_probabilities(self, texts, *, batch_size=16, use_cache=True):
		if isinstance(texts, str):
			texts = [texts]
		results = []
		for original_text in texts:
			text = self.filter_text(original_text)
			if use_cache and text in self._score_cache:
				score = self._score_cache.pop(text)
				self._score_cache[text] = score
				results.append(score)
				continue
			if not text:
				results.append(0.0)
				continue
			prefix, word = text.rsplit(" ", 1) if " " in text else ("", text)
			context = [self.bos_id, *self._ids((prefix + " ") if prefix else "")]
			targets = self._ids(word + " ")
			input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
			with torch.inference_mode(), self._autocast():
				logits, hidden = self.forward(input_ids)
				log_probability = 0.0
				next_logits = logits[0, -1]
				for target in targets:
					log_probability += F.log_softmax(next_logits.float(), dim=-1)[target].item()
					token = torch.tensor([[target]], dtype=torch.long, device=self.device)
					next_logits, hidden = self.forward(token, hidden)
					next_logits = next_logits[0, -1]
			score = log_probability / len(targets)
			self._score_cache[text] = score
			results.append(score)
		while len(self._score_cache) > self._cache_size:
			self._score_cache.popitem(last=False)
		return results

	def get_word_probabilities(self, texts, *, batch_size=16):
		return [math.exp(score) for score in self.get_word_log_probabilities(
			texts, batch_size=batch_size
		)]

	def get_word_probability(self, text):
		return self.get_word_probabilities([text])[0]


def instantiate_model(model_path=None, *, device=None, quantize=False):
	"""Create a character LM and optionally load a state-dict checkpoint."""
	model = CharacterLanguageModel()
	if model_path is not None:
		checkpoint_path = Path(model_path)
		if not checkpoint_path.exists():
			raise FileNotFoundError(f"Character LM checkpoint not found: {checkpoint_path}")
		checkpoint = torch.load(
			checkpoint_path, map_location="cpu", weights_only=True
		)
		state_dict = (
			checkpoint["model_state_dict"]
			if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
			else checkpoint
		)
		model.load_state_dict(state_dict, strict=True)
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	if quantize:
		if model.device.type != "cpu":
			raise ValueError("Dynamic quantization is supported only for CPU inference")
		model.rnn = torch.quantization.quantize_dynamic(
			model.rnn, {nn.GRU}, dtype=torch.qint8
		)
	model.eval()
	return model
