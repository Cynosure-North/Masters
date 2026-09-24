import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader
from functools import partial
from pathlib import Path

from data import MaskedDataset, chars

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
				ids.append(self.char2idx.get(char))		# Siletly ignore/remove extra characters (I won't need to predict their existence)
		ids.append(self.sep_token_id)
		return ids

	def decode(self, ids):
		"""Converts token IDs back to a string."""
		return "".join([self.idx2char.get(i, "?") for i in ids])

def create_mlm_inputs(input_ids, tokenizer, mask_prob=0.15, generator=None):
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

	return masked_inputs, labels

class CharBERTForMLM(nn.Module):
	def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_len=512, dropout=0.1):
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


def instantiate_model(model_path=None, *, vocab_size=None, d_model=128, nhead=4, num_layers=3, max_len=512, dropout=0.1, device=None):
	"""Create a CharBERTForMLM and optionally load weights from a saved checkpoint path."""
	if vocab_size is None:
		vocab_size = len(CharTokenizer())
	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model = CharBERTForMLM(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, max_len=max_len, dropout=dropout)
	model.to(device)
	if model_path is not None:
		model_path = Path(model_path)
		if model_path.exists():
			state_dict = torch.load(model_path, map_location=device, weights_only=False)
			if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
				state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
			model.load_state_dict(state_dict, strict=True)
		else:
			raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
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

def checkpoint_path_for(save_path, epoch, batch_count, *, checkpoint_interval=50_000):
	"""Return a path in the same directory as save_path with a checkpoint-specific basename."""
	save_path = Path(save_path)
	checkpoint_step = batch_count / checkpoint_interval
	return save_path.with_name(f"{save_path.stem}_{epoch}_{checkpoint_step}{save_path.suffix}")

def train_model(
	model,
	dataloader,
	tokenizer,
	epochs=3,
	optimizer=None,
	device=None,
	mask_prob=0.15,
	gradient_clip=None,
	scheduler=None,
	loss_fn=None,
	use_amp=True,
	use_compile=False,
	save_path=None,
	checkpoint_interval=50_000,
):
	"""Train a character MLM and return average loss for each epoch."""
	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model.to(device)
	compiled_model = None
	if use_compile and hasattr(torch, "compile"):
		compiled_model = model
		model = torch.compile(model, dynamic=True)
	if optimizer is None:
		optimizer_kwargs = {"lr": 1e-4}
		if device.type == "cuda":
			optimizer_kwargs["fused"] = True
		optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
	loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-100)
	amp_enabled = use_amp and device.type == "cuda"
	scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

	for epoch in range(epochs):
		model.train()
		batch_count = 0

		for batch_count, batch in enumerate(dataloader):
			# Masking is regenerated each batch, providing a new MLM task each epoch.
			input_ids, padding_mask = _prepare_batch(
				batch, tokenizer, device, max_length=model.pos_embedding.num_embeddings
			)
			masked_input_ids, labels = create_mlm_inputs(input_ids, tokenizer, mask_prob)

			optimizer.zero_grad(set_to_none=True)
			try:
				with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
					logits = model(masked_input_ids, padding_mask=padding_mask)
					loss = _mlm_loss(logits, labels, loss_fn)
			except torch._inductor.exc.TritonMissing:
				if compiled_model is None:
					raise
				print("torch.compile is unavailable; continuing with eager execution")
				model = compiled_model
				compiled_model = None
				with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
					logits = model(masked_input_ids, padding_mask=padding_mask)
					loss = _mlm_loss(logits, labels, loss_fn)
			scaler.scale(loss).backward()
			if gradient_clip is not None:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
			scaler.step(optimizer)
			scaler.update()

			batch_loss = loss.detach().item()

			if batch_count % 500 == 0: print(f"batch {batch_count} of epoch {epoch} finished -- loss: {batch_loss}")
			if save_path is not None and batch_count % checkpoint_interval == 0:
				checkpoint = checkpoint_path_for(save_path, epoch, batch_count, checkpoint_interval=checkpoint_interval)
				torch.save(model.state_dict(), checkpoint)
				print(f"Saved checkpoint, epoch: {epoch} batch {batch_count / checkpoint_interval}")

				# Keep the most recent checkpoint in the current save directory while
				# preserving the configured basename pattern for the active run.
				previous_checkpoint = checkpoint_path_for(
					save_path,
					epoch,
					max(batch_count - checkpoint_interval, 0),
					checkpoint_interval=checkpoint_interval,
				)
				previous_checkpoint.unlink(missing_ok=True)


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
	device=None,
	mask_prob=0.15,
	loss_fn=None,
	use_amp=True,
	use_compile=False,
	seed=0,
):
	"""Evaluate a character MLM and return average loss and masked-token accuracy."""
	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model.to(device)
	compiled_model = None
	if use_compile and hasattr(torch, "compile"):
		compiled_model = model
		model = torch.compile(model, dynamic=True)
	model.eval()
	loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-100)
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
		masked_input_ids, labels = create_mlm_inputs(
			input_ids, tokenizer, mask_prob, generator=mask_generator
		)
		try:
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				logits = model(masked_input_ids, padding_mask=padding_mask)
		except torch._inductor.exc.TritonMissing:
			if compiled_model is None:
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
	save_path = project_dir / "trained" / "best_BERT.pth"

	tokenizer = CharTokenizer()
	model = CharBERTForMLM(vocab_size=len(tokenizer), d_model=128, nhead=4, num_layers=3)

	if save_path.exists():
		model.load_state_dict(torch.load(save_path, weights_only=True, map_location="cpu"))
		print("loaded saved weights")

	train_dataset = MaskedDataset(train=True, tokenizer=tokenizer, max_length=model.pos_embedding.num_embeddings)
	train_sampler = LengthBucketBatchSampler(train_dataset.lengths, batch_size=64, shuffle=True)
	loader_kwargs = {
		"collate_fn": partial(pad_collate, pad_token_id=tokenizer.pad_token_id),
		"num_workers": 2,
		"pin_memory": True,
		"persistent_workers": True,
	}
	dataloader = DataLoader(
		train_dataset,
		batch_sampler=train_sampler,
		**loader_kwargs,
	)
	print("data loaded")

	trained_model = train_model(model, dataloader, tokenizer, use_amp=True, use_compile=False, save_path=save_path)

	print("training complete")
	torch.save(trained_model.state_dict(), save_path)

	test_dataset = MaskedDataset(train=False, tokenizer=tokenizer, max_length=model.pos_embedding.num_embeddings)
	test_sampler = LengthBucketBatchSampler(test_dataset.lengths, batch_size=64, shuffle=False)
	test_dataloader = DataLoader(
		test_dataset,
		batch_sampler=test_sampler,
		**loader_kwargs,
	)

	loss, accuracy = test_model(trained_model, test_dataloader, tokenizer)
	print(f"Test Loss: {loss}")
	print(f"Test Accuracy: {accuracy:.4%}")

if __name__ == "__main__":
	main()