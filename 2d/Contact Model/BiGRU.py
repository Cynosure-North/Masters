import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from pathlib import Path
from data import chars, GeometricDataset

class BiGRU(nn.Module):
	def __init__(self, input_size=2, hidden_size=128, num_layers=2, output_size=len(chars)):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		# Set bidirectional=True for a BiGRU
		self.gru = nn.GRU(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			bidirectional=True
		)
		
		# Multiply hidden_size by 2 because it is bidirectional
		self.fc = nn.Linear(hidden_size * 2, output_size)

	def forward(self, data):
		# Input is (batch, sequence, (x,y) coordinates); output is (batch, sequence, probabilities for each vocab token).
		out, _ = self.gru(data)
		
		out = self.fc(out)
		return out


def instantiate_model(model_path=None, *, input_size=2, hidden_size=128, num_layers=2, output_size=None, device=None):
	"""Create a BiGRU and optionally load weights from a saved checkpoint path."""
	if output_size is None:
		output_size = len(chars)
	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model = BiGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
	model.to(device)
	if model_path is not None:
		# Checkpoints are loaded onto the selected device so CPU evaluation also works.
		model_path = Path(model_path)
		if model_path.exists():
			state_dict = torch.load(model_path, map_location=device, weights_only=False)
			if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
				state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
			model.load_state_dict(state_dict, strict=True)
		else:
			raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
	return model


def _loss_for_logits(logits, labels, loss_fn):
	"""Flatten sequence logits and labels at the loss boundary."""
	return loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def train_model(
	model,
	dataloader,
	validation_dataloader=None,
	epochs=3,
	patience=3,
	min_delta=0.0,
	optimizer=None,
	device=None,
	gradient_clip=None,
	loss_fn=None,
	use_amp=True,
	use_compile=False,
	checkpoint_path=None,
):
	"""Train, optionally early-stop on validation loss, and return the model."""
	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model.to(device)
	compiled_model = None
	if use_compile and hasattr(torch, "compile"):
		compiled_model = model
		model = torch.compile(model, dynamic=True)
	if optimizer is None:
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-100)
	amp_enabled = use_amp and device.type == "cuda"
	scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
	best_validation_loss = float("inf")
	best_state = None
	stale_epochs = 0
	model.train()

	for epoch in range(epochs):
		batch_count = 0
		epoch_loss = 0
		for data, label in dataloader:
			# Flatten only at the loss boundary; the recurrent model keeps sequence structure.
			data = data.to(device, non_blocking=True)
			label = label.flatten().to(device, non_blocking=True)
			optimizer.zero_grad(set_to_none=True)
			try:
				with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
					output = model(data)
					loss = _loss_for_logits(output, label, loss_fn)
				if not torch.isfinite(loss):
					raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
			except torch._inductor.exc.TritonMissing:
				if compiled_model is None:
					raise
				print("torch.compile is unavailable; continuing with eager execution")
				model = compiled_model
				compiled_model = None
				with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
					output = model(data)
					loss = _loss_for_logits(output, label, loss_fn)

			scaler.scale(loss).backward()
			if gradient_clip is not None:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
			scaler.step(optimizer)
			scaler.update()

			batch_count += 1
			epoch_loss += loss.item()

		epoch_loss /= batch_count
		message = f"Epoch {epoch} -- train loss: {epoch_loss:.6f}"

		if validation_dataloader is not None:
			# Validation selects the best checkpoint without updating model parameters.
			validation_loss, _ = test_model(
				model,
				validation_dataloader,
				device=device,
				loss_fn=loss_fn,
				use_amp=use_amp,
			)
			model.train()
			message += f", validation loss: {validation_loss:.6f}"

			if validation_loss < best_validation_loss - min_delta:
				best_validation_loss = validation_loss
				best_state = deepcopy(model.state_dict())
				stale_epochs = 0
				if checkpoint_path is not None:
					torch.save(best_state, checkpoint_path)
			else:
				stale_epochs += 1
				message += f", patience: {stale_epochs}/{patience}"
				if stale_epochs >= patience:
					print(message)
					print("Early stopping")
					break

		print(message)

	if best_state is not None:
		model.load_state_dict(best_state)

	return model

@torch.no_grad()
def test_model(
	model,
	dataloader,
	device=None,
	loss_fn=None,
	use_amp=True,
	use_compile=False,
):
	"""Evaluate and return average loss and masked-token accuracy."""
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
	total_count = 0
	batch_count = 0

	for data, label in dataloader:
		data = data.to(device, non_blocking=True)
		label = label.flatten().to(device, non_blocking=True)
		try:
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				output = model(data)
				loss = _loss_for_logits(output, label, loss_fn)
		except torch._inductor.exc.TritonMissing:
			if compiled_model is None:
				raise
			print("torch.compile is unavailable; continuing with eager execution")
			model = compiled_model
			compiled_model = None
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				output = model(data)
				loss = _loss_for_logits(output, label, loss_fn)

		total_loss += loss.item()
		batch_count += 1
		prediction = output.argmax(dim=-1).reshape(-1)
		valid_labels = label.ne(-100)
		total_correct += ((prediction == label) & valid_labels).sum().item()
		total_count += valid_labels.sum().item()

	accuracy = total_correct / total_count if total_count else 0.0
	return total_loss / batch_count, accuracy

def main(_train_dataloader=None, _test_dataloader=None, _validation_dataloader=None, _save_path=None):
	project_dir = Path(__file__).resolve().parent
	save_path = _save_path or project_dir / "trained" / "best_weights.pth"
	data_dir = project_dir / "data"
	train_path = data_dir / "train.csv"
	validation_path = data_dir / "validation.csv"
	test_path = data_dir / "test.csv"

	model = BiGRU()

	if save_path.exists():
		model.load_state_dict(torch.load(save_path, weights_only=True, map_location="cpu"))
		print("loaded saved weights")

	loader_kwargs = {"batch_size": 64, "num_workers": 2, "pin_memory": True}
	train_dataloader = _train_dataloader or DataLoader(GeometricDataset(train_path), shuffle=True, **loader_kwargs)
	validation_dataloader = _validation_dataloader or DataLoader(GeometricDataset(validation_path), shuffle=False, **loader_kwargs)
	test_dataloader = _test_dataloader or DataLoader(GeometricDataset(test_path), shuffle=False, **loader_kwargs)
	print("data loaded")

	trained_model = train_model(
		model,
		train_dataloader,
		epochs=200,
		validation_dataloader=validation_dataloader,
		gradient_clip=0.5,
		use_amp=True,
		checkpoint_path=save_path,
	)

	print("training complete")
	torch.save(trained_model.state_dict(), save_path)

	loss, accuracy = test_model(trained_model, test_dataloader)
	print(f"Test Loss: {loss}")
	print(f"Test Accuracy: {accuracy:.4%}")

if __name__ == "__main__":
	main()