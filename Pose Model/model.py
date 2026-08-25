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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import dataset as dataset

model_path = Path(__file__).joinpath("trained", "TNN_weights.pt")	# TODO: I'll need to update this to have one model per finger
sns.set_theme(style="whitegrid")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class PoseMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(25, 60),
			nn.Sigmoid(),
			nn.Linear(60, 9),
		)

	def forward(self, inputs):
		return self.network(inputs)

def preprocess(todo):
	pass

	# Data into vector format
		# To estimate the pose of each finger, we measured the angle ($$ \theta $$) between the hand joint vectors as a
		# representative metric. It is indicative of the degree of finger bending. The angle was calculated as
		# a cosine function as follows. (MW vectors point to the wrist, and MF to the fingertip)
		
		
		# $$ cos\theta_n = \frac{\overrightarrow{MW_n}; \overrightarrow{MF_n}}{\norm{\overrightarrow{MW_n}} \cross \norm{{\overrightarrow{MF_n}}}}, n \in \{\text{all fingers}\}$$
		
		
		# Exceptionally, there were no differences in finger angles between when entering the Y and U keys for
		# all fingers including the touching finger (p > 0.05). To differentiate these two keys, we were
		# required to analyze another hand characteristic that affects the global hand position, such as the
		# hand direction. We checked whether the hand direction was different depending on input keys using
		# the same analysis used for analyzing finger angles. The hand direction was estimated as follows
		
		
		# $$ \text{hand direction} = \frac{\overrightarrow{WM_{index}} + \overrightarrow{WM_{little}}}{\norm{\overrightarrow{WM_{index}} + \overrightarrow{WM_{little}}}} $$


	# normalise
		# During the first stage of the preprocessing, the Normalizer transformed each hand joint vector to
		# be a unit vector. As the Normalizer extracted the unit vectors independently from each hand size,
		# it reduced the variance of our typing data.

	# scale
		# Then, in the second stage, the StandardScaler made each feature of the unit vector follow
		# the standard normal distribution.															Why force things into a normal distribution

	# pca, output 25 compontents
	
def assign_finger(todo):
	pass


def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	model.train()
	running_loss = 0.0
	num_batches = 0
	for batch, (X, y, input_lengths, target_lengths) in enumerate(dataloader):
		log_probs = model(X)
		log_probs = nn.functional.log_softmax(log_probs, dim=-1).transpose(0, 1)
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
			log_probs = nn.functional.log_softmax(log_probs, dim=-1).transpose(0, 1)
			test_loss += loss_fn(log_probs, y, input_lengths, target_lengths).item()
	test_loss /= max(num_batches, 1)
	print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
	return test_loss


def main():
	train_loader = dataset.train_loader
	test_loader = dataset.test_loader

	model = PoseMLP().to(device)

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

model = PoseMLP().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()