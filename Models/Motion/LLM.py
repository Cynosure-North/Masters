"""Character-level language model compatible with BeamSearcher."""

import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data import CCNewsDataset, chars

SEED = 0
torch.manual_seed(SEED)
class CharacterLanguageModel(nn.Module):
	"""Small recurrent character LM with length-normalized log scoring."""

	def __init__(self, *, hidden_size=256, num_layers=2, dropout=0.1, characters=chars):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.characters = tuple(characters)
		self._character_ids = {character: index for index, character in enumerate(self.characters)}

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

		self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

	@property
	def config(self):
		return type("Config", (), {"vocab_size": self.vocab_size})()

	@staticmethod
	def filter_text(text):
		return " ".join(
			"".join(character if character in chars else "" for character in str(text).lower()).split()
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
		return [self._character_ids[character] for character in text]

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

def save_checkpoint(model, destination, cleanup_previous=None):
	destination = Path(destination)
	destination.parent.mkdir(parents=True, exist_ok=True)

	checkpoint = {
		"model_config": {
			"hidden_size": model.hidden_size,
			"num_layers": model.num_layers,
			"dropout": model.dropout,
			"characters": model.characters },
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": model.optimizer.state_dict() }

	temporary_path = destination.with_name(f".{destination.name}.tmp")
	torch.save(checkpoint, temporary_path)
	temporary_path.replace(destination)

	if cleanup_previous:
		for candidate in destination.parent.iterdir():
			if candidate.is_file() and candidate.suffix == ".pth" and candidate.name.startswith(cleanup_previous.stem):
				candidate.unlink()

def instantiate_model(model_path):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_path = Path(model_path)
	if not model_path.exists():
		raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

	checkpoint = torch.load(model_path, map_location=device, weights_only=False)
	config = checkpoint["model_config"]
	model = CharacterLanguageModel(
		hidden_size=config["hidden_size"],
		num_layers=config["num_layers"],
		dropout=config["dropout"],
		characters=config["characters"],
	).to(device)

	model.load_state_dict(checkpoint["model_state_dict"], strict=True)
	model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

	return model


def test_model(model, corpus, *, block_size=256):
	"""Return average next-character loss and perplexity for an encoded corpus."""
	model.eval()
	total_loss = 0.0
	total_tokens = 0
	with torch.inference_mode(), model._autocast():
		for inputs, targets in corpus.batches(block_size, model.device):
			logits, _ = model(inputs)
			total_loss += F.cross_entropy(
				logits.reshape(-1, model.vocab_size),
				targets.reshape(-1),
				reduction="sum",
			).item()
			total_tokens += targets.numel()
	if total_tokens == 0:
		raise ValueError("Evaluation corpus has no tokens to score")
	loss = total_loss / total_tokens
	return loss, math.exp(loss)


def train_model(
	model,
	train_corpus,
	validation_corpus,
	checkpoint_path,
	*,
	steps=None,
	batch_size=32,
	block_size=256,
	validation_interval=5000,
	compile_model=False,
):
	"""Train the character LM and restore its best validation checkpoint."""
	if validation_interval < 1:
		raise ValueError("validation_interval must be at least 1")
	if steps is not None and steps < 1:
		raise ValueError("steps must be at least 1")

	best_loss = math.inf
	best_checkpoint = None
	step = 0
	network = model
	if compile_model:
		if not hasattr(torch, "compile"):
			raise RuntimeError("This PyTorch version does not provide torch.compile")
		network = torch.compile(model)
	use_amp = model.device.type == "cuda"
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
	try:
		while steps is None or step < steps:
			step += 1
			model.train()
			inputs, targets = train_corpus.sample_batch(batch_size, block_size, model.device)
			with model._autocast():
				logits, _ = network(inputs)
				loss = F.cross_entropy(
					logits.reshape(-1, model.vocab_size), targets.reshape(-1)
				)
			model.optimizer.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			scaler.unscale_(model.optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			scaler.step(model.optimizer)
			scaler.update()

			if step % validation_interval == 0:
				validation_loss, _ = test_model(
					model, validation_corpus, block_size=block_size
				)
				if validation_loss < best_loss:
					checkpoint = checkpoint_path.with_stem(f"{checkpoint_path.stem}_{step // validation_interval}")

					best_loss = validation_loss
					best_checkpoint = checkpoint
					save_checkpoint(model, best_checkpoint)

				print(f"step {step:>6} -- train loss {loss.item():.4f} -- val loss {validation_loss:.4f}")

		print("Training complete")
	except KeyboardInterrupt:
		print("Training interrupted")
	finally:
		close = getattr(train_corpus, "close", None)
		if close is not None:
			close()
	if best_checkpoint is not None:
		model = instantiate_model(best_checkpoint)
	return model


def main(steps=None, *, batch_size=128, compile_model=False):
	"""Train and evaluate a character LM using Common Crawl News."""
	save_path = Path(__file__).resolve().parent / "trained" / "final_CharacterLM.pth"

	if save_path.exists():
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = instantiate_model(save_path).to(device)
		print("loaded saved weights")
	else:
		model = CharacterLanguageModel()

	tokenize = lambda document: model._ids(model.filter_text(document))
	train_corpus = CCNewsDataset(
		"train",
		tokenize,
		bos_id=model.bos_id,
		min_document_count=10_000,
		vocab_size=model.vocab_size)
	validation_corpus = CCNewsDataset(
		"validation",
		tokenize,
		bos_id=model.bos_id,
		vocab_size=model.vocab_size)
	test_corpus = CCNewsDataset(
		"test",
		tokenize,
		bos_id=model.bos_id,
		vocab_size=model.vocab_size)
	print("data loaded")

	trained_model = train_model(
		model,
		train_corpus,
		validation_corpus,
		checkpoint_path=save_path.with_stem(save_path.stem.replace("final_", "")),
		steps=steps,
		batch_size=batch_size,
		compile_model=compile_model)
	save_checkpoint(model, save_path)

	print("training complete")
	loss, perplexity = test_model(trained_model, test_corpus)
	print(f"Test Loss: {loss:.4f}")
	print(f"Test Perplexity: {perplexity:.4f}")
	return trained_model


if __name__ == "__main__":
	main()
