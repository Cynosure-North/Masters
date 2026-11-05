"""Character-level language model compatible with BeamSearcher."""

import math
import hashlib
import threading
import queue
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CommonCrawlNewsStream, chars

SEED = 0
torch.manual_seed(SEED)
TOKEN_BYTES = torch.tensor([], dtype=torch.long).element_size()


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

class CharacterTextCorpus:
	"""Encoded text documents used for finite evaluation."""

	def __init__(self, documents):
		self.documents = [str(document) for document in documents]
		if not self.documents:
			raise ValueError("CharacterTextCorpus cannot be empty")
		self.tokens = None

	def encode(self, model):
		tokens = [model.bos_id]
		for document in self.documents:
			text = model.filter_text(document)
			if text:
				tokens.extend(model._ids(text))
				tokens.append(model.bos_id)
		if len(tokens) < 2:
			raise ValueError("Character corpus has no supported characters")
		self.tokens = torch.tensor(tokens, dtype=torch.long)
		return self.tokens

	def sample_batch(self, batch_size, block_size, device):
		if self.tokens is None:
			raise RuntimeError("Call encode(model) before requesting token batches")
		if self.tokens.numel() <= block_size:
			raise ValueError("Corpus is too short for the configured context window")
		starts = torch.randint(self.tokens.numel() - block_size, (batch_size,))
		inputs = torch.stack([self.tokens[start:start + block_size] for start in starts])
		targets = torch.stack([
			self.tokens[start + 1:start + block_size + 1] for start in starts
		])
		return inputs.to(device), targets.to(device)

	def batches(self, block_size, device):
		if self.tokens is None:
			raise RuntimeError("Call encode(model) before requesting token batches")
		for start in range(0, self.tokens.numel() - 1, block_size):
			end = min(start + block_size + 1, self.tokens.numel())
			inputs = self.tokens[start:end - 1].unsqueeze(0).to(device)
			targets = self.tokens[start + 1:end].to(device)
			if targets.numel():
				yield inputs, targets


class StreamingCharacterCorpus:
	"""Prefetch character batches from a Common Crawl document stream."""

	def __init__(
		self,
		document_stream,
		model,
		*,
		max_documents=None,
		prefetch_batches=16,
		max_token_cache_bytes=64 * 1024 * 1024,
	):
		if prefetch_batches < 1:
			raise ValueError("prefetch_batches must be at least 1")
		if max_token_cache_bytes < 1:
			raise ValueError("max_token_cache_bytes must be at least 1")
		self.document_stream = document_stream
		self.model = model
		self.max_documents = max_documents
		self._iterator = None
		self._buffer = []
		self._queue = queue.Queue(maxsize=prefetch_batches)
		self._stop = threading.Event()
		self._thread = None
		self._config = None
		self._token_cache = OrderedDict()
		self._token_cache_bytes = 0
		self._max_token_cache_bytes = max_token_cache_bytes

	def _tokens(self):
		yield self.model.bos_id
		for document in self.document_stream.iter_documents(max_documents=self.max_documents):
			document_key = hashlib.sha256(document.encode("utf-8")).digest()
			token_ids = self._token_cache.pop(document_key, None)
			if token_ids is None:
				text = self.model.filter_text(document)
				token_ids = tuple(self.model._ids(text)) if text else ()
				token_bytes = len(token_ids) * TOKEN_BYTES
				if token_ids and token_bytes <= self._max_token_cache_bytes:
					self._token_cache[document_key] = token_ids
					self._token_cache_bytes += token_bytes
					while self._token_cache_bytes > self._max_token_cache_bytes:
						_, evicted = self._token_cache.popitem(last=False)
						self._token_cache_bytes -= len(evicted) * TOKEN_BYTES
			else:
				self._token_cache[document_key] = token_ids
			if token_ids:
				yield from token_ids
				yield self.model.bos_id

	def _sample_cpu_batch(self, batch_size, block_size):
		if self._iterator is None:
			self._iterator = iter(self._tokens())
		needed = batch_size * (block_size + 1)
		while len(self._buffer) < needed:
			try:
				self._buffer.append(next(self._iterator))
			except StopIteration as exc:
				raise ValueError(
					"Common Crawl stream ended before a training batch could be formed"
				) from exc
		sequences = torch.tensor(self._buffer[:needed], dtype=torch.long)
		self._buffer = self._buffer[needed:]
		sequences = sequences.reshape(batch_size, block_size + 1)
		return sequences[:, :-1], sequences[:, 1:]

	def _worker(self, batch_size, block_size):
		try:
			while not self._stop.is_set():
				if not self._put(("batch", self._sample_cpu_batch(batch_size, block_size))):
					return
		except Exception as exc:
			self._put(("error", exc))

	def _put(self, item):
		while not self._stop.is_set():
			try:
				self._queue.put(item, timeout=0.1)
				return True
			except queue.Full:
				continue
		return False

	def sample_batch(self, batch_size, block_size, device):
		config = (batch_size, block_size)
		if self._thread is None:
			self._config = config
			self._thread = threading.Thread(
				target=self._worker, args=config, name="character-lm-prefetch", daemon=True
			)
			self._thread.start()
		elif config != self._config:
			raise ValueError("batch_size and block_size cannot change after prefetching starts")
		kind, result = self._queue.get()
		if kind == "error":
			raise result
		inputs, targets = result
		if device.type == "cuda":
			inputs = inputs.pin_memory()
			targets = targets.pin_memory()
		return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

	def close(self):
		self._stop.set()


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
	train_source = CommonCrawlNewsStream("train", seed=SEED, max_warcs=16)
	validation_source = CommonCrawlNewsStream("validation", seed=SEED + 1, max_warcs=8)
	test_source = CommonCrawlNewsStream("test", seed=SEED + 2, max_warcs=8)

	if save_path.exists():
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = instantiate_model(save_path).to(device)
		print("loaded saved weights")
	else:
		model = CharacterLanguageModel()

	validation_corpus = CharacterTextCorpus(
		list(validation_source.iter_documents(max_documents=250))
	)
	test_corpus = CharacterTextCorpus(list(test_source.iter_documents(max_documents=250)))
	validation_corpus.encode(model)
	test_corpus.encode(model)
	train_corpus = StreamingCharacterCorpus(train_source, model)

	print("######### Training Character LM")
	trained_model = train_model(
		model,
		train_corpus,
		validation_corpus,
		checkpoint_path=save_path.with_stem(save_path.stem.replace("final_", "")),
		steps=steps,
		batch_size=batch_size,
		compile_model=compile_model)
	save_checkpoint(model, save_path)

	print("######### Testing Character LM")
	loss, perplexity = test_model(trained_model, test_corpus)
	print(f"Test Loss: {loss:.4f}")
	print(f"Test Perplexity: {perplexity:.4f}")
	return trained_model


if __name__ == "__main__":
	main()


# TODO: Training is so slow. Also there's probably no need to continuously stream data. A bank should be fine (maybe expand in background)