import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel

from data import CommonCrawlNewsStream, StreamingTextCorpus, TextCorpus, chars

SEED = 0
torch.manual_seed(SEED)


class BPETokenizer:
	BOS_TOKEN = "<|bos|>"
	UNK_TOKEN = "<|unk|>"
	ALLOWED_CHARS = frozenset(chars)

	def __init__(self, vocab_size=5000):
		if vocab_size < 258:
			raise ValueError("vocab_size must include the byte alphabet and two special tokens")
		self.vocab_size = vocab_size
		self.tokenizer = None
		self.BOS_ID = None

	@classmethod
	def filter_text(cls, text):
		"""Discard every character outside the motion model's supported alphabet."""
		return "".join(character for character in str(text) if character in cls.ALLOWED_CHARS)

	def fit(self, texts):
		"""Train byte-level BPE merges from an iterable of text documents."""
		if isinstance(texts, str):
			texts = [texts]
		texts = [self.filter_text(text) for text in texts]
		texts = [text for text in texts if text]
		if not texts or not any(texts):
			raise ValueError("Cannot train BPE: corpus has no supported characters")

		self.tokenizer = Tokenizer(models.BPE(unk_token=self.UNK_TOKEN))
		self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
		self.tokenizer.decoder = decoders.ByteLevel()
		trainer = trainers.BpeTrainer(
			vocab_size=self.vocab_size,
			special_tokens=[self.BOS_TOKEN, self.UNK_TOKEN],
			initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
			show_progress=False,
		)
		self.tokenizer.train_from_iterator(texts, trainer=trainer)
		actual_vocab_size = self.tokenizer.get_vocab_size()
		if actual_vocab_size > self.vocab_size:
			raise ValueError(
				f"Corpus produced {actual_vocab_size} BPE tokens; "
				f"{self.vocab_size} is too small. Supply a larger vocab_size or corpus."
			)
		self.vocab_size = actual_vocab_size
		self.BOS_ID = self.tokenizer.token_to_id(self.BOS_TOKEN)
		return self

	def encode(self, text, add_bos=False):
		if self.tokenizer is None:
			raise RuntimeError("Train or load the tokenizer before encoding text")
		token_ids = self.tokenizer.encode(self.filter_text(text)).ids
		if add_bos:
			token_ids.insert(0, self.BOS_ID)
		return token_ids

	def decode(self, token_ids):
		if self.tokenizer is None:
			raise RuntimeError("Train or load the tokenizer before decoding token IDs")
		return self.filter_text(self.tokenizer.decode(list(token_ids), skip_special_tokens=True))

	def save(self, path):
		if self.tokenizer is None:
			raise RuntimeError("Train or load the tokenizer before saving it")
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		self.tokenizer.save(str(path))

	@classmethod
	def load(cls, path):
		tokenizer = cls.__new__(cls)
		tokenizer.tokenizer = Tokenizer.from_file(str(path))
		tokenizer.vocab_size = tokenizer.tokenizer.get_vocab_size()
		tokenizer.BOS_ID = tokenizer.tokenizer.token_to_id(cls.BOS_TOKEN)
		if tokenizer.BOS_ID is None:
			raise ValueError("Tokenizer file does not contain the BOS token")
		return tokenizer


class SmallTwoTransformerLM(nn.Module):
	"""GPT-2 causal LM configured with the Small Two architecture."""

	def __init__(
		self,
		tokenizer,
		block_size=500,
		n_layer=6,
		n_head=8,
		n_embd=256,
		ffn_dim=1024,
		dropout=0.1):

		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.tokenizer = tokenizer
		config = GPT2Config(
			vocab_size=tokenizer.vocab_size,
			n_positions=block_size,
			n_ctx=block_size,
			n_layer=n_layer,
			n_head=n_head,
			n_embd=n_embd,
			n_inner=ffn_dim,
			activation_function="gelu",
			resid_pdrop=dropout,
			embd_pdrop=dropout,
			attn_pdrop=dropout,
			bos_token_id=tokenizer.BOS_ID,
			eos_token_id=tokenizer.BOS_ID,
		)
		self.model = GPT2LMHeadModel(config)
		self.model.to(self.device)
		self._probability_cache = OrderedDict()
		self._probability_cache_size = 4096

	@property
	def config(self):
		return self.model.config

	def to(self, *args, **kwargs):
		result = super().to(*args, **kwargs)
		parameters = list(self.parameters())
		if parameters:
			self.device = parameters[0].device
		return result

	def forward(self, token_ids):
		if token_ids.ndim != 2:
			raise ValueError("token_ids must have shape (batch, sequence)")
		if token_ids.shape[1] == 0 or token_ids.shape[1] > self.model.config.n_positions:
			raise ValueError(
				f"Sequence length must be between 1 and {self.model.config.n_positions}"
			)
		return self.model(input_ids=token_ids, use_cache=False).logits

	def get_word_probability(self, text):
		text = self.tokenizer.filter_text(text)
		cached = self._probability_cache.pop(text, None)
		if cached is not None:
			self._probability_cache[text] = cached
			return cached
		if " " in text:
			prefix, word = text.rsplit(" ", 1)
			prefix += " "
		else:
			prefix, word = "", text
		if not word:
			return 1.0

		prefix_ids = self.tokenizer.encode(prefix)
		full_ids = self.tokenizer.encode(prefix + word)
		first_word_token = 0
		for prefix_token, full_token in zip(prefix_ids, full_ids):
			if prefix_token != full_token:
				break
			first_word_token += 1
		if first_word_token >= len(full_ids):
			return 1.0

		log_probability = 0.0
		with torch.inference_mode():
			for position in range(first_word_token, len(full_ids)):
				context = [self.tokenizer.BOS_ID, *full_ids[:position]][-self.model.config.n_positions :]
				input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
				logits = self.model(input_ids=input_ids).logits[0, -1]
				log_probability += F.log_softmax(logits, dim=-1)[full_ids[position]].item()

		probability = math.exp(log_probability / max(1, len(full_ids) - first_word_token))
		self._probability_cache[text] = probability
		if len(self._probability_cache) > self._probability_cache_size:
			self._probability_cache.popitem(last=False)
		return probability

def instantiate_model(
	model_path,
	*,
	tokenizer_training_documents=None,
	vocab_size=5000,
):
	"""Load a model/tokenizer pair, or initialize and save a new tokenizer/model."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint_path = Path(model_path)
	tokenizer_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_tokenizer.json")

	if tokenizer_path.exists():
		print(f"Loading tokenizer from {tokenizer_path}")
		tokenizer = BPETokenizer.load(tokenizer_path)
	else:
		if checkpoint_path.exists():
			raise FileNotFoundError(f"Weights exist at {checkpoint_path}, but their tokenizer is missing: {tokenizer_path}")
		elif tokenizer_training_documents is None:
			raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}; supply training documents to fit a new tokenizer")

		print("Fitting a new tokenizer from the training stream")
		tokenizer = BPETokenizer(vocab_size=vocab_size)
		tokenizer.fit(tokenizer_training_documents)
		tokenizer.save(tokenizer_path)

	model = SmallTwoTransformerLM(tokenizer).to(device)
	if checkpoint_path.exists():
		print(f"Loading model weights from {checkpoint_path}")
		checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
		state_dict = (
			checkpoint["model_state_dict"]
			if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
			else checkpoint
		)
		model.load_state_dict(state_dict, strict=True)
	model.eval()
	return model


def test_model(model, corpus):
	model.eval()
	total_loss = 0.0
	total_tokens = 0
	with torch.inference_mode():
		for inputs, targets in corpus.batches(model.config.n_positions, model.device):
			logits = model(inputs)
			total_loss += F.cross_entropy(
				logits.reshape(-1, model.config.vocab_size),
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
	batch_size=4,
	validation_interval=1000,
):
	if validation_interval < 1:
		raise ValueError("validation_interval must be at least 1")

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
	best_validation_loss = math.inf
	best_state = None
	step = 0

	try:
		while steps is None or step < steps:
			step += 1
			model.train()
			inputs, targets = train_corpus.sample_batch(
				batch_size, model.config.n_positions, model.device
			)
			logits = model(inputs)
			loss = F.cross_entropy(
				logits.reshape(-1, model.config.vocab_size), targets.reshape(-1)
			)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()

			if step % validation_interval == 0:
				validation_loss, _ = test_model(model, validation_corpus)
				print(f"step {step:>6} -- train loss {loss.item():.4f} -- val loss {validation_loss:.4f}")

				if validation_loss < best_validation_loss:
					best_validation_loss = validation_loss
					best_state = {
						key: value.detach().cpu().clone()
						for key, value in model.state_dict().items()
					}
					torch.save(model.state_dict(), checkpoint_path.with_name(f"LLM_{step//validation_interval}.pth"))
				
					# clean up previous checkpoints
					checkpoint_path.with_name(f"LLM_{(step//validation_interval)-2}.pth").unlink(missing_ok=True)

	except KeyboardInterrupt:
		print("Training interrupted")
	finally:
		close = getattr(train_corpus, "close", None)
		if close is not None:
			close()

	if best_state is not None:
		model.load_state_dict(best_state)
	return model


def main():
	print()
	save_path = Path(__file__).resolve().parent / "trained" / "best_LLM.pth"
	train_source = CommonCrawlNewsStream("train", seed=SEED, max_warcs=16)
	validation_source = CommonCrawlNewsStream("validation", seed=SEED + 1, max_warcs=8)
	test_source = CommonCrawlNewsStream("test", seed=SEED + 2, max_warcs=8)
	model = instantiate_model(
		save_path,
		tokenizer_training_documents=train_source.iter_documents(max_documents=1000),
	)
	tokenizer = model.tokenizer

	validation_corpus = TextCorpus(list(validation_source.iter_documents(max_documents=250)))
	test_corpus = TextCorpus(list(test_source.iter_documents(max_documents=250)))
	validation_corpus.encode(tokenizer)
	test_corpus.encode(tokenizer)

	train_corpus = StreamingTextCorpus(train_source, tokenizer)

	print("######### Training LLM")
	trained_model = train_model(model, train_corpus, validation_corpus, save_path)

	print("training complete")
	torch.save(trained_model.state_dict(), save_path)

	print("######### Testing LLM")
	loss, perplexity = test_model(trained_model, test_corpus)
	print(f"Test Loss: {loss:.4f}")
	print(f"Test Perplexity: {perplexity:.4f}")
	return trained_model


if __name__ == "__main__":
	main()
