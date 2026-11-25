import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from data import TestMotionDataset, MotionDataset, chars
import editdistance

VARIANT = "3d-raw"
CTC_BLANK = len(chars)

# Ideas:
# Chomp
# Diff numbers of layers
class ResidualTCNBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
		super().__init__()
		self.conv1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
		self.conv2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation))
		self.norm1 = nn.GroupNorm(1, out_channels)
		self.norm2 = nn.GroupNorm(1, out_channels)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout1 = nn.Dropout(dropout)
		self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
		self.padding = (kernel_size - 1) * dilation

	def forward(self, x):
		residual = x
		if self.projection is not None:
			residual = self.projection(residual)

		x = nn.functional.pad(x, (self.padding, 0))
		x = self.conv1(x)
		x = self.norm1(x)
		x = nn.LeakyReLU()(x)
		x = self.dropout1(x)

		x = nn.functional.pad(x, (self.padding, 0))
		x = self.conv2(x)
		x = self.norm2(x)
		x = nn.LeakyReLU()(x)
		x = self.dropout1(x)
		return x + residual


# Not sure how they got 46 frames, their parameters only give 19
class TCN(nn.Module):
	# num_classes has one extra character for the non-character class (nothing is pressed)
	def __init__(self, input_features=42, num_classes=len(chars)+1, hidden_channels=(64, 64, 32), kernel_size=2, dilation=3, dropout=0.4):
		super().__init__()

		self.input_features = input_features
		self.num_classes = num_classes
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation = dilation
		self.dropout = dropout

		self.input_projection = nn.Conv1d(self.input_features, self.hidden_channels[0], kernel_size=1)
		self.blocks = nn.ModuleList()
		for idx, out_channels in enumerate(self.hidden_channels):
			in_channels = self.hidden_channels[0] if idx == 0 else self.hidden_channels[idx - 1]
			self.blocks.append(
				ResidualTCNBlock(
					in_channels,
					out_channels,
					kernel_size=self.kernel_size,
					dilation=self.dilation,
					dropout=dropout + idx/10
				)
			)
		self.output_projection = nn.Linear(self.hidden_channels[-1], self.num_classes)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

	def forward(self, x):
		# Convert [Batch, Timestep, Features] to Conv1d's [B, F, T] layout.
		x = x.transpose(1, 2)

		x = self.input_projection(x)
		for block in self.blocks:
			x = block(x)

		# Convert from [B, Channels, T] back to [B, T, F]
		x = x.transpose(1, 2)
		return self.output_projection(x)


def collate_batch(batch):
	features, targets, target_lengths = zip(*batch)
	padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True)
	padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
	input_lengths = torch.tensor([feature.size(0) for feature in features], dtype=torch.long)
	target_lengths = torch.tensor(target_lengths, dtype=torch.long)
	return padded_features, padded_targets, input_lengths, target_lengths


def save_checkpoint(model, destination, cleanup_previous=None):
	destination = Path(destination)
	destination.parent.mkdir(parents=True, exist_ok=True)
	checkpoint = {
		"model_config": {
			"input_features": model.input_features,
			"num_classes": model.num_classes,
			"hidden_channels": model.hidden_channels,
			"kernel_size": model.kernel_size,
			"dilation": model.dilation,
			"dropout": model.dropout },
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": model.optimizer.state_dict(),
		"scaler_state_dict": model.scaler.state_dict() }

	temporary_path = destination.with_name(f".{destination.name}.tmp")
	torch.save(checkpoint, temporary_path)
	temporary_path.replace(destination)

	if cleanup_previous:
		cleanup_previous = Path(cleanup_previous)
		for candidate in destination.parent.iterdir():
			if candidate.is_file() and candidate.suffix == ".pth" and candidate.name.startswith(cleanup_previous.stem):
				candidate.unlink()

def instantiate_model(model_path, **kwargs):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_path = Path(model_path)
	if not model_path.exists():
		print(f"Model checkpoint not found: {model_path}, instantiating new model")
		return TCN(**kwargs)

	checkpoint = torch.load(model_path, map_location=device, weights_only=False)
	config = checkpoint["model_config"]
	model = TCN(
		input_features=config["input_features"],
		num_classes=config["num_classes"],
		hidden_channels=config["hidden_channels"],
		kernel_size=config["kernel_size"],
		dilation=config["dilation"],
	).to(device)

	model.load_state_dict(checkpoint["model_state_dict"], strict=True)
	model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	model.scaler.load_state_dict(checkpoint["scaler_state_dict"])

	print("loaded model from saved weights")
	
	return model


def train_model(
	model,
	dataloader,
	checkpoint_path,
	*,
	validation_dataloader=None,
	epochs=500,
	patience=5,
	min_delta=0.0,
	gradient_clip=None,
):
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	best_validation_loss = float("inf")
	best_checkpoint = None
	stale_epochs = 0
	epoch = 0

	try:
		while epoch != epochs:
			model.train()
			epoch_loss = 0.0
			batch_count = 0
			for batch in dataloader:
				model.optimizer.zero_grad(set_to_none=True)

				features, targets, input_lengths, target_lengths = batch
				features = features.to(device, non_blocking=True)
				loss = F.ctc_loss(
					F.log_softmax(
						model(features.to(device, non_blocking=True)),
						dim=-1).transpose(0, 1),
					targets.to(device, non_blocking=True),
					input_lengths.to(device, non_blocking=True),
					target_lengths.to(device, non_blocking=True),
					blank=CTC_BLANK,
				)
				if not torch.isfinite(loss):
					raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
				model.scaler.scale(loss).backward()
				if gradient_clip is not None:
					model.scaler.unscale_(model.optimizer)
					nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
				model.scaler.step(model.optimizer)
				model.scaler.update()
				epoch_loss += loss.item()
				batch_count += 1

			epoch_loss /= max(batch_count, 1)
			message = f"Epoch {epoch:<5n} -- train loss: {epoch_loss:9.6f}"

			if validation_dataloader is not None:
				validation_loss, char_accuracy, _ = test_model(model, validation_dataloader)
				message += f" -- validation loss: {validation_loss:9.6f}"
				if validation_loss < best_validation_loss - min_delta:
					if patience == -1:
						message += f" new best, char accuracy: {char_accuracy}"

					checkpoint = checkpoint_path.with_stem(f"{checkpoint_path.stem}_{epoch}")

					best_validation_loss = validation_loss
					best_checkpoint = checkpoint
					stale_epochs = 0

					save_checkpoint(model, checkpoint)
				elif patience != -1:
					stale_epochs += 1
					message += f", patience: {stale_epochs}/{patience}"
					if stale_epochs >= patience:
						print(message)
						print("Early stopping")
						break

			print(message)
			epoch += 1
		
		print("Training complete")

	except KeyboardInterrupt:
		print("Training interrupted")

	finally:
		save_checkpoint(model, checkpoint_path.with_stem(f"{checkpoint_path.stem}_last"))

		if best_checkpoint is not None:
			model = instantiate_model(best_checkpoint)
	return model
		

@torch.no_grad()
def test_model(model, dataloader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	total_loss = 0.0
	correct_sequences = 0
	total_sequences = 0
	incorrect_chars = 0
	total_chars = 0
	batch_count = 0
	for batch in dataloader:
		features, targets, input_lengths, target_lengths = batch
		features = features.to(device, non_blocking=True)
		logits = model(features)
		log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
		total_loss += F.ctc_loss(
			log_probs,
			targets.to(device, non_blocking=True),
			input_lengths.to(device, non_blocking=True),
			target_lengths.to(device, non_blocking=True),
			blank=CTC_BLANK,
		).item()

		predictions = logits.argmax(dim=-1).cpu()
		for prediction, target, in zip(predictions, targets):
			total_sequences += 1
			collapsed = []
			previous = None
			for token in prediction.tolist():
				if token != CTC_BLANK and token != previous:
					collapsed.append(token)
				previous = token
			expected = target.tolist()
			correct_sequences += int(collapsed == expected)
			total_chars += len(expected)
			incorrect_chars += editdistance.eval(collapsed, expected)
		batch_count += 1

	char_accuracy = 1 - (incorrect_chars / total_chars)
	accuracy = correct_sequences / max(total_sequences, 1)
	return total_loss / max(batch_count, 1), accuracy, char_accuracy


def main(train_path_=None, validation_path_=None, test_path_=None, save_path_=None, variant=VARIANT, epochs=-1):
	project_dir = Path(__file__).resolve().parent
	save_path = save_path_ or project_dir / "trained" / "best_TCN.pth"
	data_dir = project_dir / "data" / variant
	train_path = train_path_ or data_dir / "train"
	validation_path = validation_path_ or data_dir / "validation"
	test_path = test_path_ or data_dir / "test"

	loader_kwargs = {"batch_size": 64, "num_workers": 2, "pin_memory": True}
	if not variant:
		model = instantiate_model(save_path)

		train_dataloader = DataLoader(
			TestMotionDataset(train_path), shuffle=True, collate_fn=collate_batch, **loader_kwargs)
		validation_dataloader = DataLoader(
			TestMotionDataset(validation_path), shuffle=False, collate_fn=collate_batch, **loader_kwargs)
		test_dataloader = DataLoader(
			TestMotionDataset(test_path), shuffle=False, collate_fn=collate_batch, **loader_kwargs)
	else:
		save_path = save_path.with_stem(f"{save_path.stem}_{variant}")
		train_dataset = MotionDataset(train_path)
		# The feature width depends on the selected preprocessing variant.
		model = instantiate_model(save_path, input_features=train_dataset.input_features)
		if model.input_features != train_dataset.input_features:
			raise ValueError(
				f"Model expects {model.input_features} features per frame, "
				f"but the dataset provides {train_dataset.input_features}")

		train_dataloader = DataLoader(
			train_dataset, shuffle=True, collate_fn=collate_batch, **loader_kwargs)
		validation_dataloader = DataLoader(
			MotionDataset(validation_path), shuffle=False, collate_fn=collate_batch, **loader_kwargs)
		test_dataloader = DataLoader(
			MotionDataset(test_path), shuffle=False, collate_fn=collate_batch, **loader_kwargs)
		
	print("data loaded")

	trained_model = train_model(
		model,
		train_dataloader,
		checkpoint_path=save_path.with_stem(save_path.stem.replace("best_", "")),
		epochs=epochs,
		validation_dataloader=validation_dataloader,
		patience=-1)

	# print("training complete")
	save_checkpoint(trained_model, save_path,
		cleanup_previous=save_path.stem.replace("best_", ""))

	loss, accuracy, char_accuracy = test_model(trained_model, test_dataloader)
	print(f"Test Loss: {loss:.4f}")
	print(f"Test Accuracy: {accuracy:.4%}")
	print(f"Test Character Accuracy: {char_accuracy:.4%}")


if __name__ == "__main__":
	main()
