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
from data import PoseDataset, finger_list, finger_to_keys


class Inidiv_PoseMLP(nn.Module):
	def __init__(self, *, inputs=25, neurons=60, outputs=9):
		super().__init__()
		self.outputs = outputs

		self.network = nn.Sequential(
			nn.Linear(inputs, neurons),
			nn.Sigmoid(),
			nn.Linear(neurons, outputs),
		)

	def forward(self, inputs):
		if self.outputs == 0:
			raise ValueError("Finger model with no assigned keys called")
		return self.network(inputs)

class PoseMLP(nn.Module):
	def __init__(self, models=None, *, inputs=25, neurons=60):
		super().__init__()

		self.models = nn.ModuleDict()
		if models is None:
			models = {
				key: Inidiv_PoseMLP(inputs=inputs, neurons=neurons, outputs=len(value))
				for key, value in finger_to_keys.items()
			}
		self.models.update(models)

	def forward(self, data, finger=None):
		if finger is None:
			raise ValueError("finger is required because each finger has a different key vocabulary")
		if finger not in self.models:
			raise KeyError(f"No pose model is configured for finger: {finger}")
		return self.models[finger](data)

def instantiate_models(model_path):
	"""Create the multi-finger model and optionally load a checkpoint."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = PoseMLP().to(device)

	model_path = Path(model_path)
	if model_path.exists():
		state_dict = torch.load(model_path, map_location=device, weights_only=True)
		model.load_state_dict(state_dict, strict=True)
	else:
		raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

	return model


def train_model(
		model,
		dataloader,
		*,
		epochs=10,
		learning_rate=1e-3,
		checkpoint_path=None,
		validation_dataloader=None,
		patience=3
		):
	"""Train all ten finger classifiers on interleaved key/finger datapoints."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizers = {
		finger: torch.optim.Adam(model.models[finger].parameters(), lr=learning_rate)
		for finger in finger_list
		if finger_to_keys[finger]
	}
	best_state = None
	best_loss = float("inf")
	stale_epochs = 0

	for epoch in range(epochs):
		model.train()
		finger_losses = {finger: [] for finger in optimizers}

		for labels, fingers, features in dataloader:
			features = features.to(device, non_blocking=True).float()
			labels = labels.to(device, non_blocking=True).long()
			
			for finger, optimizer in optimizers.items():
				selected = torch.tensor(
					[current_finger == finger for current_finger in fingers],
					dtype=torch.bool,
					device=device)

				if not selected.any():
					continue

				this_finger_features = features[selected]
				this_finger_labels = labels[selected]

				optimizer.zero_grad(set_to_none=True)
				loss = loss_fn(model(this_finger_features, finger), this_finger_labels)
				loss.backward()
				optimizer.step()
				finger_losses[finger].append(loss.item())

		train_loss = {
			finger: f"{(sum(losses) / len(losses)):.3f}"
			for finger, losses in finger_losses.items()
			if losses }

		if validation_dataloader is None:
			print(f"Epoch {epoch + 1:3n} -- train loss: {train_loss:.4f}")
			continue
		validation_loss, validation_accuracy = test_model(model, validation_dataloader)
		if validation_loss < best_loss:
			best_loss = validation_loss
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
			stale_epochs = 0
			if checkpoint_path is not None:
				torch.save(best_state, checkpoint_path)
		else:
			stale_epochs += 1
		print(f"Epoch {epoch + 1:<3n} -- train loss: {train_loss:.4f}, validation loss: {validation_loss:.4f}, validation accuracy: {validation_accuracy:.4%}")
		if stale_epochs >= patience:
			break

	if best_state is not None:
		model.load_state_dict(best_state)
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
	data_dir = project_dir / "data" 
	train_path = _train_path or data_dir / "train.csv"
	validation_path = _validation_path or data_dir / "validation.csv"
	test_path = _test_path or data_dir / "test.csv"
	save_path = _save_path or project_dir / "trained" / "best_MLP.pth"

	model = PoseMLP()

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
