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
from data import PoseDataset


fingers = ["l_little", "l_ring", "l_middle", "l_index", "l_thumb", "r_little", "r_ring", "r_middle", "r_index", "r_thumb"]
finger_to_keys = {
	"l_little": "qaz",
	"l_ring": "qwsxz",
	"l_middle": "edcrx",
	"l_index": "rtgfgcvby",
	"l_thumb": " ",
	"r_thumb": " ",
	"r_index": "yuhjbnmi",
	"r_middle": "ikm,ol",
	"r_ring": "lo.p",
	"r_little": "" }

# key_to_finger = {}
# for char in 'abcdefghijklmnopqrstuvwxyz':
# 	tmp = []
# 	for key, value in finger_to_keys.items():
# 		if char in value:
# 			tmp.append(key)
# 	key_to_finger[char] = tmp
				



class Inidiv_PoseMLP(nn.Module):
	def __init__(self, *, inputs=25, nuerons=60, outputs=9):
		super().__init__()
		self.outputs = outputs

		self.network = nn.Sequential(
			nn.Linear(inputs, nuerons),
			nn.Sigmoid(),
			nn.Linear(nuerons, outputs),
		)

	def forward(self, inputs):
		if self.outputs == 0:
			return 0
			# TODO
		if self.outputs == 1:
			return 1
			# TODO
			# return " "
		return self.network(inputs)

class PoseMLP(nn.Module):
	def __init__(self, models=None, *, inputs=25, neurons=60):
		super().__init__()

		self.models = nn.ModuleDict()
		if models is None:
			models = {
				key: Inidiv_PoseMLP(inputs=inputs, nuerons=neurons, outputs=len(value))
				for key, value in finger_to_keys.items()
			}
		self.models.update(models)

	def forward(self, data, finger=None):
		if finger is None:
			raise ValueError("finger is required because each finger has a different key vocabulary")
		if finger not in self.models:
			raise KeyError(f"No pose model is configured for finger: {finger}")
		return self.models[finger](data)

def instantiate_models(model_path, *, inputs=25, nuerons=60):
	"""Create the multi-finger model and optionally load a checkpoint."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = PoseMLP(inputs=inputs, neurons=nuerons)
	checkpoint = Path(model_path)
	if not checkpoint.exists():
		raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
	state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
	model.load_state_dict(state_dict, strict=True)
	return model.to(device)



def _key_indices(keys, finger):
	key_to_index = {key: index for index, key in enumerate(finger_to_keys[finger])}
	if isinstance(keys, torch.Tensor):
		return keys.long()
	try:
		return torch.tensor([key_to_index[key] for key in keys], dtype=torch.long)
	except KeyError as error:
		raise ValueError(f"Key {error.args[0]!r} is not assigned to {finger}") from error


def train_model(model, dataloader, *, epochs=10, learning_rate=1e-3,
		device=None, checkpoint_path=None, validation_dataloader=None,
		patience=3, loss_fn=None):
	"""Train all ten finger classifiers on interleaved key/finger datapoints."""
	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model.to(device)
	loss_fn = loss_fn or nn.CrossEntropyLoss()
	optimizers = {
		finger: torch.optim.Adam(model.models[finger].parameters(), lr=learning_rate)
		for finger in fingers
		if finger_to_keys[finger]
	}
	best_state = None
	best_loss = float("inf")
	stale_epochs = 0

	for epoch in range(epochs):
		model.train()
		finger_losses = {finger: [] for finger in optimizers}

		for keys, batch_fingers, features in dataloader:
			features = features.to(device, non_blocking=True).float()
			for finger, optimizer in optimizers.items():
				selected = torch.tensor(
					[current_finger == finger for current_finger in batch_fingers],
					dtype=torch.bool,
					device=device,
				)
				if not selected.any():
					continue
				finger_features = features[selected]
				finger_keys = _key_indices(
					[key for key, keep in zip(keys, selected.cpu().tolist()) if keep], finger,
				).to(device)
				optimizer.zero_grad(set_to_none=True)
				loss = loss_fn(model(finger_features, finger), finger_keys)
				loss.backward()
				optimizer.step()
				finger_losses[finger].append(loss.item())

		train_loss = {
			finger: sum(losses) / len(losses)
			for finger, losses in finger_losses.items()
			if losses
		}

		if validation_dataloader is None:
			print(f"Epoch {epoch + 1} -- train loss: {train_loss}")
			continue
		validation_loss, validation_accuracy = test_model(
			model, validation_dataloader, device=device, loss_fn=loss_fn
		)
		if validation_loss < best_loss:
			best_loss = validation_loss
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
			stale_epochs = 0
			if checkpoint_path is not None:
				torch.save(best_state, checkpoint_path)
		else:
			stale_epochs += 1
		print(f"Epoch {epoch + 1} -- train loss: {train_loss}, validation loss: {validation_loss:.6f}, validation accuracy: {validation_accuracy:.4%}")
		if stale_epochs >= patience:
			break

	if best_state is not None:
		model.load_state_dict(best_state)
	return model


@torch.no_grad()
def test_model(model, dataloader, *, device=None, loss_fn=None):
	"""Evaluate each interleaved datapoint with its assigned finger model."""
	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model.to(device).eval()
	loss_fn = loss_fn or nn.CrossEntropyLoss()
	total_loss = 0.0
	total_correct = 0
	total_count = 0

	with torch.no_grad():
		for keys, batch_fingers, features in dataloader:
			features = features.to(device, non_blocking=True).float()
			for finger in fingers:
				if not finger_to_keys[finger]:
					continue
				selected = torch.tensor(
					[current_finger == finger for current_finger in batch_fingers],
					dtype=torch.bool,
					device=device,
				)
				if not selected.any():
					continue
				finger_keys = _key_indices(
					[key for key, keep in zip(keys, selected.cpu().tolist()) if keep], finger,
				).to(device)
				logits = model(features[selected], finger)
				batch_size = finger_keys.numel()
				total_loss += loss_fn(logits, finger_keys).item() * batch_size
				total_correct += (logits.argmax(dim=-1) == finger_keys).sum().item()
				total_count += batch_size

	return total_loss / max(total_count, 1), total_correct / max(total_count, 1)

def main():
	project_dir = Path(__file__).resolve().parent
	save_path = project_dir / "best_weights.pth"
	data_dir = project_dir / "data"
	train_path = data_dir / "train.csv"
	validation_path = data_dir / "validation.csv"
	test_path = data_dir / "test.csv"

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
	print(f"Test Loss: {loss}")
	print(f"Test Accuracy: {accuracy:.4%}")

if __name__ == "__main__":
	main()
