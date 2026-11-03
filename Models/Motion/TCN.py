import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from data import MotionDataset, chars

THREE_DIMENSIONS = False

class ResidualTCNBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=2, dilation=3):
		super().__init__()
		self.conv1 = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
		self.conv2 = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation))
		self.norm1 = torch.nn.GroupNorm(1, out_channels)
		self.norm2 = torch.nn.GroupNorm(1, out_channels)
		self.projection = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
		self.padding = (kernel_size - 1) * dilation

	def forward(self, x):
		residual = x
		if self.projection is not None:
			residual = self.projection(residual)

		x = torch.nn.functional.pad(x, (self.padding, 0))
		x = self.conv1(x)
		x = self.norm1(x)
		x = torch.relu(x)

		x = torch.nn.functional.pad(x, (self.padding, 0))
		x = self.conv2(x)
		x = self.norm2(x)
		return torch.relu(x + residual)


class TCN(torch.nn.Module):
	# num_classes has one extra character for the non-character class (nothing is pressed)
	def __init__(self, input_features=42, num_classes=len(chars)+1, hidden_channels=(64, 64, 32), kernel_size=2, dilation=3):
		super().__init__()

		self.input_features = input_features
		self.num_classes = num_classes
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation = dilation

		self.input_projection = torch.nn.Conv1d(self.input_features, self.hidden_channels[0], kernel_size=1)
		self.blocks = torch.nn.ModuleList()
		for index, out_channels in enumerate(self.hidden_channels):
			in_channels = self.hidden_channels[0] if index == 0 else self.hidden_channels[index - 1]
			self.blocks.append(
				ResidualTCNBlock(
					in_channels,
					out_channels,
					kernel_size=self.kernel_size,
					dilation=self.dilation,
				)
			)
		self.output_projection = torch.nn.Linear(self.hidden_channels[-1], self.num_classes)

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
	padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
	padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
	input_lengths = torch.tensor([feature.size(0) for feature in features], dtype=torch.long)
	target_lengths = torch.tensor(target_lengths, dtype=torch.long)
	return padded_features, padded_targets, input_lengths, target_lengths


def save_checkpoint(model, destination, *, epoch=None, validation_loss=None, cleanup_previous=None):
	destination = Path(destination)
	destination.parent.mkdir(parents=True, exist_ok=True)
	checkpoint = {
		"model_config": {
			"input_features": model.input_features,
			"num_classes": model.num_classes,
			"hidden_channels": model.hidden_channels,
			"kernel_size": model.kernel_size,
			"dilation": model.dilation },
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": model.optimizer.state_dict(),
		"scaler_state_dict": model.scaler.state_dict(),
		"epoch": epoch,
		"validation_loss": validation_loss }

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
	
	return model


def train_model(
	model,
	dataloader,
	checkpoint_path,
	*,
	validation_dataloader=None,
	epochs=3,
	patience=5,
	min_delta=0.0,
	gradient_clip=None,
):
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	best_validation_loss = float("inf")
	best_checkpoint = None
	stale_epochs = 0

	for epoch in range(epochs):
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
				target_lengths.to(device, non_blocking=True)
			)
			if not torch.isfinite(loss):
				raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
			model.scaler.scale(loss).backward()
			if gradient_clip is not None:
				model.scaler.unscale_(model.optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
			model.scaler.step(model.optimizer)
			model.scaler.update()
			epoch_loss += loss.item()
			batch_count += 1

		epoch_loss /= max(batch_count, 1)
		message = f"Epoch {epoch:<3n} -- train loss: {epoch_loss:.6f}"

		if validation_dataloader is not None:
			validation_loss, _ = test_model(model, validation_dataloader)
			if validation_loss < best_validation_loss - min_delta:
				checkpoint = f"{checkpoint_path.stem}_{epoch}"

				best_validation_loss = validation_loss
				best_checkpoint = checkpoint
				stale_epochs = 0

				save_checkpoint(
					model,
					checkpoint_path.with_stem(checkpoint),
					epoch=epoch,
					validation_loss=validation_loss)
			else:
				stale_epochs += 1
				message += f", patience: {stale_epochs}/{patience}"
				if stale_epochs >= patience and patience != -1:
					print(message)
					print("Early stopping")
					break

		print(message)

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
			target_lengths.to(device, non_blocking=True)
		).item()

		predictions = logits.argmax(dim=-1).cpu()
		for prediction, target, in zip(predictions, targets):
			total_sequences += 1
			collapsed = []
			previous = None
			for token in prediction.tolist():
				if token != 0 and token != previous:
					collapsed.append(token)
				previous = token
			expected = target.tolist()
			correct_sequences += int(collapsed == expected)
		batch_count += 1

	accuracy = correct_sequences / max(total_sequences, 1)
	return total_loss / max(batch_count, 1), accuracy


def main(_train_path=None, _validation_path=None, _test_path=None, _save_path=None):
	project_dir = Path(__file__).resolve().parent
	data_dir = project_dir / "data"
	save_path = _save_path or project_dir / "trained" / "final_TCN.pth"
	train_path = _train_path or data_dir / "train"
	validation_path = _validation_path or data_dir / "validation"
	test_path = _test_path or data_dir / "test"

	if not THREE_DIMENSIONS:
		model = TCN()
	else:
		# FIXME: Maybe add another layer for the 3d TCN?
		# Adds in a vector pointing from each fingertip to the nearest surface point, and also compressibility
		model = TCN(input_features=73)
		save_path = save_path.with_stem(f"{save_path.stem}_3d")

	if save_path.exists():
		model = instantiate_model(save_path)
		print("loaded saved weights")

	loader_kwargs = {"batch_size": 64, "num_workers": 2, "pin_memory": True}
	train_dataloader = DataLoader(
		MotionDataset(train_path), shuffle=True, collate_fn=collate_batch, **loader_kwargs)
	validation_dataloader = DataLoader(
		MotionDataset(validation_path), shuffle=False, collate_fn=collate_batch, **loader_kwargs)
	test_dataloader = DataLoader(
		MotionDataset(test_path), shuffle=False, collate_fn=collate_batch, **loader_kwargs)
	print("data loaded")

	trained_model = train_model(
		model,
		train_dataloader,
		checkpoint_path=save_path.with_stem(save_path.stem.replace("final_", "")),
		epochs=200,
		validation_dataloader=validation_dataloader)

	print("training complete")
	save_checkpoint(trained_model, save_path,
		cleanup_previous=save_path.stem.replace("final_", ""))

	loss, accuracy = test_model(trained_model, test_dataloader)
	print(f"Test Loss: {loss:.4f}")
	print(f"Test Accuracy: {accuracy:.4%}")


if __name__ == "__main__":
	main()

# TODO: TCN is cooked, perhaps because I'm not using the full keyboard? IDK needs more testing

# IDEA: It could also be good to test the motion model with only the motion data and none of the surface data