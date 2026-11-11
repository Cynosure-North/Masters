import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader
from functools import partial
from pathlib import Path

from data import MaskedDataset, chars

SEED = 0
if torch.cuda.is_available():
	torch.set_float32_matmul_precision("high")

class CharTokenizer:
	def __init__(self):
		# Base vocabulary: ASCII printable characters + special tokens
		# special_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"]
		special_tokens = ["=", "?", "<CLS>", "<SEP>", "_"]
	   
		self.vocab = chars + special_tokens
		self.char2idx = {c: i for i, c in enumerate(self.vocab)}
		self.idx2char = {i: c for i, c in enumerate(self.vocab)}
	   
		self.pad_token_id = self.char2idx["="]
		self.unk_token_id = self.char2idx["?"]
		self.cls_token_id = self.char2idx["<CLS>"]
		self.sep_token_id = self.char2idx["<SEP>"]
		self.mask_token_id = self.char2idx["_"]

	def __len__(self):
		return len(self.vocab)

	def encode(self, text):
		"""Converts raw text string into token IDs with <CLS> and <SEP>."""
		ids = [self.cls_token_id]
		for char in text:
			if char in self.vocab:
				ids.append(self.char2idx[char])
		ids.append(self.sep_token_id)
		return ids

	def decode(self, ids):
		"""Converts token IDs back to a string."""
		return "".join([self.idx2char.get(i, "?") for i in ids])

def create_mlm_inputs(input_ids, tokenizer, generator, mask_prob=0.15):
	"""
	Applies the BERT 80/10/10 masking logic across non-special character tokens.
	"""

	labels = input_ids.clone()
	masked_inputs = input_ids.clone()

	# Do not train the MLM objective on padding or sequence boundary tokens.
	special_ids = {tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id}
	eligible_mask = torch.ones_like(input_ids, dtype=torch.bool)
	for sp_id in special_ids:
		eligible_mask &= (input_ids != sp_id)

	# Sample 15% of eligible positions
	prob_matrix = torch.full(input_ids.shape, mask_prob, device=input_ids.device)
	prob_matrix.masked_fill_(~eligible_mask, 0.0)
	masked_indices = (torch.rand(input_ids.shape, device=input_ids.device, generator=generator) < prob_matrix).bool()

	# Positions not selected get -100 label so CrossEntropyLoss ignores them
	labels[~masked_indices] = -100

	# Apply the standard 80/10/10 rule: mask, random token, or unchanged token.
	indices_replaced = (
		torch.rand(input_ids.shape, device=input_ids.device, generator=generator) < 0.8
	) & masked_indices
	masked_inputs[indices_replaced] = tokenizer.mask_token_id

	# 10% of selected -> replaced with random character ID
	indices_random = (
		(torch.rand(input_ids.shape, device=input_ids.device, generator=generator) < 0.5)
		& masked_indices
		& ~indices_replaced
	)
	random_chars = torch.randint(
		len(tokenizer), input_ids.shape, dtype=torch.long,
		device=input_ids.device, generator=generator,
	)
	masked_inputs[indices_random] = random_chars[indices_random]

	# Remaining 10% of selected -> left unchanged in masked_inputs

	return labels, masked_inputs

class CharBERTForMLM(nn.Module):
	def __init__(self, vocab_size=len(CharTokenizer()), d_model=256, nhead=8, num_layers=6, max_len=512, dropout=0.1):
		super().__init__()
		self.d_model = d_model
	   
		# Character & Learned Positional Embeddings
		self.char_embedding = nn.Embedding(vocab_size, d_model)
		self.pos_embedding = nn.Embedding(max_len, d_model)
		self.layer_norm = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

		# Encoder Layers
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=d_model * 4,
			dropout=dropout,
			activation="gelu",
			batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		# MLM Head
		self.mlm_head = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.GELU(),
			nn.LayerNorm(d_model),
			nn.Linear(d_model, vocab_size)
		)
		self.optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
		self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

	def forward(self, input_ids, padding_mask=None):
		# input_ids has shape (batch, sequence); logits has shape (batch, sequence, vocabulary).
		batch_size, seq_len = input_ids.size()
		if seq_len > self.pos_embedding.num_embeddings:
			raise ValueError(
				f"Sequence length {seq_len} exceeds the model maximum "
				f"of {self.pos_embedding.num_embeddings}."
			)
		pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

		# Token + Position embeddings
		embeddings = self.char_embedding(input_ids) + self.pos_embedding(pos_ids)
		x = self.dropout(self.layer_norm(embeddings))

		# Pass through Transformer (src_key_padding_mask requires True for padded locations)
		hidden_states = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
	   
		# Predict character logits
		logits = self.mlm_head(hidden_states)
		return logits


def save_checkpoint(model, destination, cleanup_previous=None):
	destination = Path(destination)
	destination.parent.mkdir(parents=True, exist_ok=True)
	checkpoint = {
		"model_config": {
			"vocab_size": model.char_embedding.num_embeddings,
			"d_model": model.d_model,
			"nhead": model.transformer_encoder.layers[0].self_attn.num_heads,
			"num_layers": len(model.transformer_encoder.layers),
			"max_len": model.pos_embedding.num_embeddings,
			"dropout": model.dropout.p,
		},
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": model.optimizer.state_dict(),
		"scaler_state_dict": model.scaler.state_dict() }

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
	model = CharBERTForMLM(
		vocab_size=config["vocab_size"],
		d_model=config["d_model"],
		nhead=config["nhead"],
		num_layers=config["num_layers"],
		max_len=config["max_len"],
		dropout=config["dropout"],
	).to(device)

	model.load_state_dict(checkpoint["model_state_dict"], strict=True)
	model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	model.scaler.load_state_dict(checkpoint["scaler_state_dict"])

	return model

def _prepare_batch(batch, tokenizer, device, max_length=None):
	# Normalize tensor, string, and (token IDs, padding mask) batches to one format.
	if isinstance(batch, torch.Tensor):
		input_ids = batch[:, :max_length] if max_length is not None else batch
		padding_mask = input_ids.eq(tokenizer.pad_token_id)
	elif isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], str):
		sequences = [tokenizer.encode(text) for text in batch]
		if max_length is not None:
			if max_length < 2:
				raise ValueError("max_length must be at least 2 for <CLS> and <SEP>.")
			sequences = [
				sequence[:max_length - 1] + [tokenizer.sep_token_id]
				if len(sequence) > max_length else sequence
				for sequence in sequences
			]
		max_length = max(len(sequence) for sequence in sequences)
		input_ids = torch.full(
			(len(sequences), max_length),
			tokenizer.pad_token_id,
			dtype=torch.long,
		)
		for row, sequence in enumerate(sequences):
			input_ids[row, :len(sequence)] = torch.tensor(sequence)
		padding_mask = input_ids.eq(tokenizer.pad_token_id)
	elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
		input_ids = batch[0]
		if max_length is not None:
			input_ids = input_ids[:, :max_length]
		padding_mask = batch[1] if len(batch) > 1 else input_ids.eq(tokenizer.pad_token_id)
		if len(batch) > 1 and max_length is not None:
			padding_mask = padding_mask[:, :max_length]
	else:
		raise TypeError("Each batch must contain token IDs or raw strings.")

	input_ids = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
	padding_mask = padding_mask.to(device=device, dtype=torch.bool, non_blocking=True)
	return input_ids, padding_mask

def pad_collate(batch, pad_token_id):
	input_ids = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
	return input_ids, input_ids.eq(pad_token_id)

class LengthBucketBatchSampler(BatchSampler):
	def __init__(self, lengths, batch_size, shuffle=True, bucket_size_multiplier=50):
		self.lengths = torch.as_tensor(lengths, dtype=torch.long)
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.bucket_size = batch_size * bucket_size_multiplier

	def __iter__(self):
		# Sorting within shuffled buckets reduces padding while retaining batch randomness.
		num_buckets = (len(self.lengths) + self.bucket_size - 1) // self.bucket_size
		bucket_order = torch.randperm(num_buckets).tolist() if self.shuffle else range(num_buckets)
		for bucket_number in bucket_order:
			start = bucket_number * self.bucket_size
			stop = min(start + self.bucket_size, len(self.lengths))
			bucket = list(range(start, stop))
			if self.shuffle:
				local_order = torch.randperm(len(bucket)).tolist()
				bucket = [bucket[index] for index in local_order]
			bucket.sort(key=self.lengths.__getitem__)
			for offset in range(0, len(bucket), self.batch_size):
				yield bucket[offset:offset + self.batch_size]

	def __len__(self):
		return (len(self.lengths) + self.batch_size - 1) // self.batch_size

def _mlm_loss(logits, labels, loss_fn):
	return loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

def _is_triton_missing(error):
	"""Return whether an error is the optional Triton compiler failure."""
	error_type = type(error)
	return (
		error_type.__name__ == "TritonMissing"
		and error_type.__module__ == "torch._inductor.exc"
	)

def train_model(
	model,
	dataloader,
	tokenizer,
	checkpoint_path,
	*,
	epochs=3,
	mask_prob=0.15,
	gradient_clip=None,
	scheduler=None,
	loss_fn=None,
	use_amp=True,
	use_compile=False,
	checkpoint_interval=50_000,
	seed=SEED,
):
	"""Train a character MLM and return average loss for each epoch."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	compiled_model = None
	if use_compile and hasattr(torch, "compile"):
		compiled_model = model
		model = torch.compile(model, dynamic=True)
	loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-100)
	amp_enabled = use_amp and device.type == "cuda"

	mask_generator = torch.Generator(device=device).manual_seed(seed)

	for epoch in range(epochs):
		model.train()
		batch_count = 0

		for batch_count, batch in enumerate(dataloader):
			# Masking is regenerated each batch, providing a new MLM task each epoch.
			input_ids, padding_mask = _prepare_batch(
				batch, tokenizer, device, max_length=model.pos_embedding.num_embeddings
			)
			labels, masked_input_ids= create_mlm_inputs(
				input_ids,
				tokenizer,
				mask_generator,
				mask_prob,
			)

			model.optimizer.zero_grad(set_to_none=True)
			try:
				with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
					logits = model(masked_input_ids, padding_mask=padding_mask)
					loss = _mlm_loss(logits, labels, loss_fn)
			except RuntimeError as e:
				if compiled_model is None or not _is_triton_missing(e):
					raise
				print("torch.compile is unavailable; continuing with eager execution")
				model = compiled_model
				compiled_model = None
				with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
					logits = model(masked_input_ids, padding_mask=padding_mask)
					loss = _mlm_loss(logits, labels, loss_fn)
			model.scaler.scale(loss).backward()
			if gradient_clip is not None:
				model.scaler.unscale_(model.optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
			model.scaler.step(model.optimizer)
			model.scaler.update()

			batch_loss = loss.detach().item()

			if batch_count % 500 == 0: print(f"batch {batch_count:<6n} of epoch {epoch} finished -- loss: {batch_loss:.4f}")
			if checkpoint_path is not None and batch_count % checkpoint_interval == 0:
				checkpoint = checkpoint_path.with_stem(f"{checkpoint_path.stem}_{epoch}_{batch_count // checkpoint_interval}")
				save_checkpoint(model, checkpoint)
				torch.save(model.state_dict(), checkpoint)
				print(f"Saved checkpoint, epoch: {epoch}_{batch_count // checkpoint_interval}")


		if batch_count == 0:
			raise ValueError("The dataloader must contain at least one batch.")
		if scheduler is not None:
			scheduler.step()

		print(f"Epoch {epoch} finished")

	return model

@torch.no_grad()
def test_model(
	model,
	dataloader,
	tokenizer,
	mask_prob=0.15,
	use_amp=True,
	use_compile=False,
	seed=SEED,
):
	"""Evaluate a character MLM and return average loss and masked-token accuracy."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	compiled_model = None
	if use_compile and hasattr(torch, "compile"):
		compiled_model = model
		model = torch.compile(model, dynamic=True)
	model.eval()
	loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
	amp_enabled = use_amp and device.type == "cuda"
	total_loss = 0.0
	total_correct = 0
	total_masked = 0
	batch_count = 0
	mask_generator = torch.Generator(device=device).manual_seed(seed)

	for batch in dataloader:
			# Evaluate on freshly sampled masked positions, matching the training objective.
		input_ids, padding_mask = _prepare_batch(
			batch, tokenizer, device, max_length=model.pos_embedding.num_embeddings
		)
		labels, masked_input_ids = create_mlm_inputs(
			input_ids,
			tokenizer,
			mask_generator,
			mask_prob)
		try:
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				logits = model(masked_input_ids, padding_mask=padding_mask)
		except RuntimeError as error:
			if compiled_model is None or not _is_triton_missing(error):
				raise
			print("torch.compile is unavailable; continuing with eager evaluation")
			model = compiled_model
			compiled_model = None
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				logits = model(masked_input_ids, padding_mask=padding_mask)
		total_loss += _mlm_loss(logits, labels, loss_fn).item()
		valid_labels = labels.ne(-100)
		predictions = logits.argmax(dim=-1)
		total_correct += ((predictions == labels) & valid_labels).sum().item()
		total_masked += valid_labels.sum().item()
		batch_count += 1

	if batch_count == 0:
		raise ValueError("The dataloader must contain at least one batch.")
	accuracy = total_correct / total_masked if total_masked else 0.0
	return total_loss / batch_count, accuracy


def main():
	project_dir = Path(__file__).resolve().parent
	save_path = project_dir / "trained" / "final_BERT.pth"

	if save_path.exists():
		model = instantiate_model(save_path)
		print("loaded saved weights")
	else:
		model = CharBERTForMLM()
	tokenizer = CharTokenizer()

	train_dataset = MaskedDataset(train=True, tokenizer=tokenizer, max_length=model.pos_embedding.num_embeddings)
	train_sampler = LengthBucketBatchSampler(train_dataset.lengths, batch_size=64, shuffle=True)

	loader_kwargs = {
		"collate_fn": partial(pad_collate, pad_token_id=tokenizer.pad_token_id),
		"num_workers": 2,
		"pin_memory": True,
		"persistent_workers": True, }
	dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
	print("data loaded")

	trained_model = train_model(
		model,
		dataloader,
		tokenizer,
		checkpoint_path=save_path.with_stem(save_path.stem.replace("final_", "")),
		use_amp=True,
		use_compile=False)
		

	print("training complete")
	save_checkpoint(trained_model, save_path,
		cleanup_previous=save_path.with_stem(save_path.stem.replace("final_", "")))

	test_dataset = MaskedDataset(train=False, tokenizer=tokenizer, max_length=model.pos_embedding.num_embeddings)
	test_sampler = LengthBucketBatchSampler(test_dataset.lengths, batch_size=64, shuffle=False)
	test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, **loader_kwargs)

	loss, accuracy = test_model(trained_model, test_dataloader, tokenizer)
	print(f"Test Loss: {loss:.4f}")
	print(f"Test Accuracy: {accuracy:.4%}")

if __name__ == "__main__":
	main()