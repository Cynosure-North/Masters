# https://ieeexplore.ieee.org/document/8764534



# Pre-allocation keyboard (or P-key) is designed based on Key Pre-allocation. When a touch event
# occurs, it brings the keys allocated to that touching finger and selects the one closest to the
# touch point. As the identity of touching finger limits the candidates of inputtable keys, the P-key
# performs better than the Normal keyboard. Specifically, the P-key was effective in reducing
# horizontal typing errors. Pre-allocation and Hand Pose aware keyboard (or HP-key) is designed based
# on both Key Pre-allocation and Key Inference based on Hand Poses. When a touch event occurs, a
# target key is inferred through the touching finger's key inference model. Therefore, we expected
# that the HP-key reduced the horizontal and vertical typing errors at the same time


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from data import PoseDataset, finger_list, finger_to_keys, get_preprocessor_state, set_preprocessor_state

THREE_DIMENSIONS=False

class Inidiv_PoseMLP(nn.Module):
	def __init__(self, *, inputs=25, neurons=60, outputs=9, layers=1):
		super().__init__()
		self.inputs = inputs
		self.neurons = neurons
		self.outputs = outputs
		self.layers = layers
		
		self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

		network = nn.Sequential(nn.Linear(inputs, neurons))
		for _ in range(layers - 1):
			network.append(nn.LeakyReLU())
			network.append(nn.Linear(neurons, neurons))
		network.append(nn.LeakyReLU())
		network.append(nn.Linear(neurons, outputs))
		self.network = network

	def forward(self, inputs):
		if self.outputs == 0:
			raise ValueError("Finger model with no assigned keys called")
		return self.network(inputs)

class PoseMLP(nn.Module):
	def __init__(self, models=None, *, inputs=25, neurons=60, layers=1):
		super().__init__()

		if models is None:
			models = {
				key: Inidiv_PoseMLP(inputs=inputs, neurons=neurons, outputs=len(value), layers=layers)
				for key, value in finger_to_keys.items()
				if finger_to_keys[key] }
		self.models = nn.ModuleDict(models)

	def forward(self, data, finger=None):
		if finger is None:
			raise ValueError("finger is required because each finger has a different key vocabulary")
		if finger not in self.models:
			raise KeyError(f"No pose model is configured for finger: {finger}")
		return self.models[finger](data)
	
def save_checkpoint(model, destination, cleanup_previous=None):
	destination = Path(destination)
	destination.mkdir(parents=True, exist_ok=True)
	temporary_path = destination.with_name(f".{destination.name}.tmp")
	
	# TODO: What is the point of saving validation loss in the checkpoint, or storing the epoch as it's in the name
	
	preprocess_checkpoint = get_preprocessor_state()
	torch.save(preprocess_checkpoint, temporary_path / "preprocess.pth")

	for name, finger in model.models.items():
		finger_checkpoint = {
			"model_config": {
				"inputs": finger.inputs,
				"neurons": finger.neurons,
				"outputs": finger.outputs,
				"layers": finger.layers,
			},
			"model_state_dict": finger.state_dict(),
			"optimizer_state_dict": finger.optimizer.state_dict() }
		torch.save(finger_checkpoint, temporary_path / f"{name}.pth")

	temporary_path.replace(destination)

	if cleanup_previous:
		for candidate in destination.parent.iterdir():
			if candidate.is_dir() and candidate.name.startswith(cleanup_previous.name):
				candidate.rmdir()

def instantiate_model(model_path):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_path = Path(model_path)
	if not model_path.exists():
		raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

	models = {}
	for file in model_path.iterdir():
		name = file.stem
		checkpoint = torch.load(file, map_location=device, weights_only=False)
		
		if name == "preprocess":
			set_preprocessor_state(checkpoint)
		else:
			config = checkpoint["model_config"]
			finger_model = Inidiv_PoseMLP(
				inputs=config["inputs"],
				neurons=config["neurons"],
				outputs=config["outputs"],
				layers=config["layers"])

			finger_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
			finger_model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

			models[name] = finger_model
	
	model = PoseMLP(models) 
	
	return model


def train_model(
		model,
		dataloader,
		checkpoint_path,
		*,
		epochs=10,
		validation_dataloader=None,
		patience=3):
	"""Train all ten finger classifiers on interleaved key/finger datapoints."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	loss_fn = nn.CrossEntropyLoss()
	best_state = None
	best_loss = float("inf")
	stale_epochs = 0

	for epoch in range(epochs):
		model.train()
		finger_losses = {finger: [] for finger in model.models.keys()}

		for labels, fingers, features in dataloader:
			features = features.to(device, non_blocking=True).float()
			labels = labels.to(device, non_blocking=True).long()
			
			for name, finger in model.models.items():
				selected = torch.tensor(
					[current_finger == finger for current_finger in fingers],
					dtype=torch.bool,
					device=device)

				if not selected.any():
					continue

				this_finger_features = features[selected]
				this_finger_labels = labels[selected]

				finger.optimizer.zero_grad(set_to_none=True)
				loss = loss_fn(model(this_finger_features, finger), this_finger_labels)
				loss.backward()
				finger.optimizer.step()
				finger_losses[name].append(loss.item())

		train_loss = {
			finger: f"{(sum(losses) / len(losses)):.3f}"
			for finger, losses in finger_losses.items()
			if losses }
		message = f"Epoch {epoch:<3n} -- train loss: {train_loss:.4f}"

		if validation_dataloader is not None:
			validation_loss, validation_accuracy = test_model(model, validation_dataloader)
			if validation_loss < best_loss:
				checkpoint = checkpoint_path.with_stem(f"{checkpoint_path.stem}_{epoch}")

				best_loss = validation_loss
				best_checkpoint = checkpoint
				stale_epochs = 0

				save_checkpoint(model, checkpoint)
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
	"""Evaluate each interleaved datapoint with its assigned finger model."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device).eval()
	loss_fn = nn.CrossEntropyLoss()
	total_loss = 0.0
	total_correct = 0
	total_count = 0

	with torch.no_grad():
		for labels, fingers, features in dataloader:
			features = features.to(device, non_blocking=True).float()
			labels = labels.to(device, non_blocking=True).long()

			for finger in fingers:
				selected = torch.tensor(
					[current_finger == finger for current_finger in fingers],
					dtype=torch.bool,
					device=device)

				if not selected.any():
					continue
				
				this_finger_features = features[selected]
				this_finger_labels = labels[selected]

				logits = model(this_finger_features, finger)
				batch_size = this_finger_labels.numel()
				total_loss += loss_fn(logits, this_finger_labels).item() * batch_size
				total_correct += (logits.argmax(dim=-1) == this_finger_labels).sum().item()
				total_count += batch_size

	return total_loss / max(total_count, 1), total_correct / max(total_count, 1)

def main(_train_path=None, _validation_path=None, _test_path=None, _save_path=None):
	project_dir = Path(__file__).resolve().parent
	save_path = _save_path or project_dir / "trained" / "final_MLP.pth"
	data_dir = project_dir / "data" 
	train_path = _train_path or data_dir / "train.csv"
	validation_path = _validation_path or data_dir / "validation.csv"
	test_path = _test_path or data_dir / "test.csv"

	if not THREE_DIMENSIONS:
		model = PoseMLP()
	else:
		model = PoseMLP(layers=2)
		save_path = save_path.with_stem(f"{save_path.stem}_3d")

	if save_path.exists():
		model.load_state_dict(torch.load(save_path, weights_only=True, map_location="cpu"))
		print("loaded saved weights")

	loader_kwargs = {"batch_size": 64, "num_workers": 2, "pin_memory": True}
	train_dataloader = DataLoader(PoseDataset(train_path), shuffle=True, **loader_kwargs)
	validation_dataloader = DataLoader(PoseDataset(validation_path), shuffle=False, **loader_kwargs)
	test_dataloader = DataLoader(PoseDataset(test_path), shuffle=False, **loader_kwargs)
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

# https://karpathy.github.io/2019/04/25/recipe/
#
# I'll probably want to increase the number of neurons/layers compared to the 2d one
# I can test and see how it goes