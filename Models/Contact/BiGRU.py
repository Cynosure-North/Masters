import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from pathlib import Path
from data import chars, pad_variable, GeometricDataset

SEED = 0
THREE_DIMENSIONS = False

def augment_coordinates(data, generator, *, jitter_pixels=3.0, noise_std=0.01, p=0.5):
	"""Apply coordinate jitter and gaussian noise"""
	if data.ndim != 3 or data.size(-1) != 2:
		return data
	if p <= 0:
		return data
	if not torch.is_floating_point(data):
		data = data.float()

	mask = torch.rand(data.shape[:2], generator=generator, device=data.device) < p
	if not mask.any():
		return data

	jitter_x = ((torch.rand(data.shape[:2], generator=generator, device=data.device) - 0.5) * 2.0) * (jitter_pixels / 1440.0)
	jitter_y = ((torch.rand(data.shape[:2], generator=generator, device=data.device) - 0.5) * 2.0) * (jitter_pixels / 854.0)
	noise = torch.randn(data.shape, generator=generator, device=data.device, dtype=data.dtype) * noise_std

	augmented = data.clone()
	augmented[:, :, 0] = torch.where(mask, augmented[:, :, 0] + jitter_x + noise[:, :, 0], augmented[:, :, 0])
	augmented[:, :, 1] = torch.where(mask, augmented[:, :, 1] + jitter_y + noise[:, :, 1], augmented[:, :, 1])
	return augmented

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
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

	def forward(self, data):
		# Input is (batch, sequence, (x,y) coordinates); output is (batch, sequence, probabilities for each vocab token).
		out, _ = self.gru(data)
		
		out = self.fc(out)
		return out


def save_checkpoint(
	model,
	destination,
	*,
	epoch=None,
	validation_loss=None,
	cleanup_previous=None,
):

	destination = Path(destination)
	destination.parent.mkdir(parents=True, exist_ok=True)
	checkpoint = {
		"model_config": {
			"input_size": model.gru.input_size,
			"hidden_size": model.hidden_size,
			"num_layers": model.num_layers,
			"output_size": model.fc.out_features },
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
			if candidate.is_file() and candidate.suffix == ".pth" and candidate.name.startswith(cleanup_previous):
				candidate.unlink()



def instantiate_model(model_path):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_path = Path(model_path)
	if not model_path.exists():
		raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

	checkpoint = torch.load(model_path, map_location=device, weights_only=False)
	config = checkpoint["model_config"]
	model = BiGRU(
		input_size=config["input_size"],
		hidden_size=config["hidden_size"],
		num_layers=config["num_layers"],
		output_size=config["output_size"],
	).to(device)

	model.load_state_dict(checkpoint["model_state_dict"], strict=True)
	model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	model.scaler.load_state_dict(checkpoint["scaler_state_dict"])
	
	return model

def _loss_for_logits(logits, labels, loss_fn):
	"""Flatten sequence logits and labels at the loss boundary."""
	return loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def train_model(
	model,
	dataloader,
	checkpoint_path,
	*,
	validation_dataloader=None,
	epochs=3,
	patience=3,
	min_delta=0.0,
	device=None,
	gradient_clip=None,
	use_amp=True,
	use_compile=False,
	seed=SEED,
	use_augmentation=True,
):
	"""Train, optionally early-stop on validation loss, and return the model."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	compiled_model = None
	if use_compile and hasattr(torch, "compile"):
		compiled_model = model
		model = torch.compile(model, dynamic=True)

	loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
	amp_enabled = use_amp and device.type == "cuda"
	generator = torch.Generator(device=device).manual_seed(seed)

	best_validation_loss = float("inf")
	best_checkpoint = None
	stale_epochs = 0

	for epoch in range(epochs):
		model.train()
		batch_count = 0
		epoch_loss = 0
		for labels, data in dataloader:
			# Flatten only at the loss boundary; the recurrent model keeps sequence structure.
			data = data.to(device, non_blocking=True)
			if use_augmentation:
				data = augment_coordinates(data, generator)
			labels = labels.flatten().to(device, non_blocking=True)
			model.optimizer.zero_grad(set_to_none=True)
			try:
				with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
					output = model(data)
					loss = _loss_for_logits(output, labels, loss_fn)
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
					loss = _loss_for_logits(output, labels, loss_fn)

			model.scaler.scale(loss).backward()
			if gradient_clip is not None:
				model.scaler.unscale_(model.optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
			model.scaler.step(model.optimizer)
			model.scaler.update()

			batch_count += 1
			epoch_loss += loss.item()

		epoch_loss /= batch_count
		message = f"Epoch {epoch:<3n} -- train loss: {epoch_loss:.6f}"

		if validation_dataloader is not None:
			validation_loss, _ = test_model(
				model,
				validation_dataloader,
				use_amp=use_amp)
			model.train()
			message += f", validation loss: {validation_loss:.6f}"

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
				if stale_epochs >= patience:
					print(message)
					print("Early stopping")
					break

		print(message)

	if best_checkpoint is not None:
		model = instantiate_model(best_checkpoint)

	return model

@torch.no_grad()
def test_model(
	model,
	dataloader,
	use_amp=True,
	use_compile=False,
):
	"""Evaluate and return average loss and masked-token accuracy."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	compiled_model = None
	if use_compile and hasattr(torch, "compile"):
		compiled_model = model
		model = torch.compile(model, dynamic=True)

	loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
	amp_enabled = use_amp and device.type == "cuda"

	total_loss = 0.0
	total_correct = 0
	total_count = 0
	batch_count = 0

	model.eval()

	for labels, data in dataloader:
		data = data.to(device, non_blocking=True)
		labels = labels.flatten().to(device, non_blocking=True)
		try:
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				output = model(data)
				loss = _loss_for_logits(output, labels, loss_fn)
		except torch._inductor.exc.TritonMissing:
			if compiled_model is None:
				raise
			print("torch.compile is unavailable; continuing with eager execution")
			model = compiled_model
			compiled_model = None
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				output = model(data)
				loss = _loss_for_logits(output, labels, loss_fn)

		total_loss += loss.item()
		batch_count += 1
		prediction = output.argmax(dim=-1).reshape(-1)
		valid_labels = labels.ne(-100)
		total_correct += ((prediction == labels) & valid_labels).sum().item()
		total_count += valid_labels.sum().item()

	accuracy = total_correct / total_count if total_count else 0.0
	return total_loss / batch_count, accuracy

def main(_train_path=None, _validation_path=None, _test_path=None, _save_path=None):
	project_dir = Path(__file__).resolve().parent
	save_path = _save_path or project_dir / "trained" / "final_BiGRU.pth"
	data_dir = project_dir / "data" / "geometric"
	train_path = _train_path or data_dir / "train.csv"
	validation_path = _validation_path or data_dir / "validation.csv"
	test_path = _test_path or data_dir / "test.csv"

	if not THREE_DIMENSIONS:
		model = BiGRU()
	else:
		# Also has [uv, normal, principal curvatures, compressibility]
		# FIXME: maybe adding an extra layer to the 3d contact model will help. We will see
		model = BiGRU(input_size=16, num_layers=3)
		save_path = save_path.with_stem(f"{save_path.stem}_3d")

	if save_path.exists():
		model = instantiate_model(save_path)
		print("loaded saved weights")

	loader_kwargs = {"batch_size": 64, "collate_fn": pad_variable, "num_workers": 2, "pin_memory": True}
	train_dataloader = DataLoader(GeometricDataset(train_path), shuffle=True, **loader_kwargs)
	validation_dataloader = DataLoader(GeometricDataset(validation_path), shuffle=False, **loader_kwargs)
	test_dataloader = DataLoader(GeometricDataset(test_path), shuffle=False, **loader_kwargs)
	print("data loaded")

	trained_model = train_model(
		model,
		train_dataloader,
		checkpoint_path=save_path.with_stem(save_path.stem.replace("final_", "")),
		epochs=200,
		validation_dataloader=validation_dataloader,
		gradient_clip=0.5,
		use_amp=True)

	print("training complete")
	save_checkpoint(trained_model, save_path,
		cleanup_previous=save_path.stem.replace("final_", ""))

	loss, accuracy = test_model(trained_model, test_dataloader)
	print(f"Test Loss: {loss:.4f}")
	print(f"Test Accuracy: {accuracy:.4%}")

if __name__ == "__main__":
	main()
	
# For the 3d version I should test if I can just use the UV coordinates