import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import dataset as KPdataset

model_path = Path(__file__).parent.parent.joinpath("downloaded", "TNN_weights.pt")
sns.set_theme(style="whitegrid")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class ResidualTCNBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=2, dilation=3):
		super().__init__()
		padding = (kernel_size - 1) * dilation
		self.conv1 = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
		self.conv2 = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation))
		self.norm1 = torch.nn.GroupNorm(1, out_channels)
		self.norm2 = torch.nn.GroupNorm(1, out_channels)
		self.projection = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
		self.padding = padding

	def forward(self, x):
		residual = x
		if self.projection is not None:
			residual = self.projection(residual)

		x = torch.nn.functional.pad(x, (self.padding, 0))
		x = self.conv1(x)
		x = torch.relu(self.norm1(x))

		x = torch.nn.functional.pad(x, (self.padding, 0))
		x = self.conv2(x)
		x = self.norm2(x)
		return torch.relu(x + residual)


class KeyPressModel(torch.nn.Module):
	def __init__(self, input_features=20, num_classes=None, hidden_channels=(64, 64, 32), kernel_size=2, dilation=3):
		super().__init__()
		self.input_features = input_features
		self.num_classes = num_classes if num_classes is not None else 75
		self.input_projection = torch.nn.Conv1d(input_features, hidden_channels[0], kernel_size=1)
		self.blocks = torch.nn.ModuleList([
			ResidualTCNBlock(hidden_channels[0], hidden_channels[0], kernel_size=kernel_size, dilation=dilation),
			ResidualTCNBlock(hidden_channels[0], hidden_channels[1], kernel_size=kernel_size, dilation=dilation),
			ResidualTCNBlock(hidden_channels[1], hidden_channels[2], kernel_size=kernel_size, dilation=dilation),
		])
		self.output_projection = torch.nn.Linear(hidden_channels[-1], self.num_classes)

	def forward(self, x):
		if x.dim() == 2:
			x = x.unsqueeze(1)
		elif x.dim() != 3:
			raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

		# x: [B, T, F]
		if x.dim() == 3 and x.shape[1] != self.input_features:
			x = x.transpose(1, 2)
		elif x.dim() == 3:
			x = x.transpose(1, 2)

		x = self.input_projection(x)
		for block in self.blocks:
			x = block(x)

		# x: [B, C, T]
		x = x.transpose(1, 2)
		return self.output_projection(x)


def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	model.train()
	running_loss = 0.0
	num_batches = 0
	for batch, (X, y, input_lengths, target_lengths) in enumerate(dataloader):
		log_probs = model(X)
		log_probs = F.log_softmax(log_probs, dim=-1).transpose(0, 1)
		loss = loss_fn(log_probs, y, input_lengths, target_lengths)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		running_loss += loss.item()
		num_batches += 1

		if batch % 100 == 0:
			print(f"loss: {loss.item():>7f}  [{(batch + 1):>5d}/{size:>5d}]")

	return running_loss / max(num_batches, 1)


def test(dataloader, model, loss_fn):
	num_batches = len(dataloader)
	model.eval()
	test_loss = 0.0
	with torch.no_grad():
		for X, y, input_lengths, target_lengths in dataloader:
			log_probs = model(X)
			log_probs = F.log_softmax(log_probs, dim=-1).transpose(0, 1)
			test_loss += loss_fn(log_probs, y, input_lengths, target_lengths).item()
	test_loss /= max(num_batches, 1)
	print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
	return test_loss


def main():
	train_loader = KPdataset.train_loader
	test_loader = KPdataset.test_loader

	model = KeyPressModel(input_features=KPdataset.dataset.num_features, num_classes=len(KPdataset.dataset.vocabulary)).to(device)

	loss_fn = torch.nn.CTCLoss(blank=0)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	epochs = 2
	train_losses = []
	test_losses = []
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		epoch_train_loss = train(train_loader, model, loss_fn, optimizer)
		train_losses.append(epoch_train_loss)
		test_loss = test(test_loader, model, loss_fn)
		test_losses.append(test_loss)
		print(f"Epoch {t+1} train loss: {epoch_train_loss:.6f}, test loss: {test_loss:.6f}")

	print("Saving model to {model_path}")
	torch.save(model.state_dict(), model_path)

	plt.figure(figsize=(8, 5))
	sns.lineplot(x=list(range(1, len(train_losses) + 1)), y=train_losses, label="Training loss", marker="o")
	sns.lineplot(x=list(range(1, len(test_losses) + 1)), y=test_losses, label="Test loss", marker="s")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training and Test Loss")
	plt.legend()
	plt.tight_layout()
	plt.show()
	print("Done!")

if __name__ == "__main__":
	main()

# TODO: Determine num_features in advance.
# num_classes will be 29 (26 letters + space (_) + no-char (-) + end-char (>) )
model = KeyPressModel(input_features=KPdataset.dataset.num_features, num_classes=len(KPdataset.dataset.vocabulary)).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# TODO use the hand representation they did
# The input features to the network are frame-to-frame deltas of wrist position and rotation along 
# with 3D fingertip positions. All positions are represented in the coordinate frame of the keyboard