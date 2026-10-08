import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from data import MotionDataset, chars

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
		self.input_projection = torch.nn.Conv1d(input_features, hidden_channels[0], kernel_size=1)
		self.blocks = torch.nn.ModuleList([
			ResidualTCNBlock(hidden_channels[0], hidden_channels[0], kernel_size=kernel_size, dilation=dilation),
			ResidualTCNBlock(hidden_channels[0], hidden_channels[1], kernel_size=kernel_size, dilation=dilation),
			ResidualTCNBlock(hidden_channels[1], hidden_channels[2], kernel_size=kernel_size, dilation=dilation),
		])
		self.output_projection = torch.nn.Linear(hidden_channels[-1], num_classes)

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


def instantiate_model(model_path):
	"""Create a KeyPressModel and optionally load weights from a checkpoint."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = TCN().to(device)

	model_path = Path(model_path)
	if model_path.exists():
		state_dict = torch.load(model_path, map_location=device, weights_only=True)
		if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict):
			state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
	else:
		raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

	model.load_state_dict(state_dict, strict=True)
	
	return model


def _ctc_loss(model, batch, device):
	features, targets, input_lengths, target_lengths = batch
	features = features.to(device, non_blocking=True)
	targets = targets.to(device, non_blocking=True)
	input_lengths = input_lengths.to(device, non_blocking=True)
	target_lengths = target_lengths.to(device, non_blocking=True)
	log_probs = F.log_softmax(model(features), dim=-1).transpose(0, 1)
	return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0)


def train_model(
	model,
	dataloader,
	validation_dataloader=None,
	epochs=3,
	patience=5,
	min_delta=0.0,
	gradient_clip=None,
	checkpoint_path=None,
):
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	best_validation_loss = float("inf")
	best_state = None
	stale_epochs = 0

	for epoch in range(epochs):
		model.train()
		epoch_loss = 0.0
		batch_count = 0
		for batch in dataloader:
			optimizer.zero_grad(set_to_none=True)

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
			_ctc_loss(model, batch, device)
			if not torch.isfinite(loss):
				raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
			loss.backward()
			if gradient_clip is not None:
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
			optimizer.step()
			epoch_loss += loss.item()
			batch_count += 1

		epoch_loss /= max(batch_count, 1)
		message = f"Epoch {epoch:<3n} -- train loss: {epoch_loss:.6f}"

		if validation_dataloader is not None:
			validation_loss, _ = test_model(model, validation_dataloader)
			message += f", validation loss: {validation_loss:.6f}"
			if validation_loss < best_validation_loss - min_delta:
				best_validation_loss = validation_loss
				best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
				stale_epochs = 0
				if checkpoint_path is not None:
					torch.save(best_state, checkpoint_path)
			else:
				stale_epochs += 1
				message += f", patience: {stale_epochs}/{patience}"
				if stale_epochs >= patience and patience != -1:
					print(message)
					print("Early stopping")
					break

		print(message)

	if best_state is not None:
		model.load_state_dict(best_state)
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
		for prediction, target, input_length, target_length in zip(
			predictions, targets, input_lengths, target_lengths):

			total_sequences += 1
			collapsed = []
			previous = None
			for token in prediction[:input_length].tolist():
				if token != 0 and token != previous:
					collapsed.append(token)
				previous = token
			expected = target[:target_length].tolist()
			correct_sequences += int(collapsed == expected)
		batch_count += 1

	accuracy = correct_sequences / max(total_sequences, 1)
	return total_loss / max(batch_count, 1), accuracy


def main(_train_path=None, _validation_path=None, _test_path=None, _save_path=None):
	project_dir = Path(__file__).resolve().parent
	save_path = _save_path or project_dir / "trained" / "best_TCNe.pth"
	data_dir = project_dir / "data"
	train_path = _train_path or data_dir / "train"
	validation_path = _validation_path or data_dir / "validation"
	test_path = _test_path or data_dir / "test"

	model = TCN()

	if save_path.exists():
		model.load_state_dict(torch.load(save_path, weights_only=True, map_location="cpu"))
		print("loaded saved weights")

	loader_kwargs = {"batch_size": 64, "num_workers": 2, "pin_memory": True}
	train_dataloader = DataLoader(
		MotionDataset(train_path), shuffle=True, collate_fn=collate_batch, **loader_kwargs
	)
	validation_dataloader = DataLoader(
		MotionDataset(validation_path), shuffle=False, collate_fn=collate_batch, **loader_kwargs
	)
	test_dataloader = DataLoader(
		MotionDataset(test_path), shuffle=False, collate_fn=collate_batch, **loader_kwargs
	)
	print("data loaded")

	trained_model = train_model(
		model,
		train_dataloader,
		epochs=200,
		validation_dataloader=validation_dataloader,
	)

	print("training complete")
	torch.save(trained_model.state_dict(), save_path)

	loss, accuracy = test_model(trained_model, test_dataloader)
	print(f"Test Loss: {loss:.4f}")
	print(f"Test Accuracy: {accuracy:.4%}")


if __name__ == "__main__":
	main()
