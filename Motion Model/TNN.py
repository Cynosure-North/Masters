import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import glob
import csv
import collections
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Load marker positions from the files in C:\Users\mno64\Datasets\How we type\Motion Capture and split into training and test sets
# files are in the format 005307_sentences_mocap_matched.csv
# they are tsv files, despite the extension
# the columns are uid	condition	time	stimulus_index	input_index	iki	input	key_symbol	stimulus	typed	finger	right_hand	Hands_L_Cout_x	Hands_L_Cout_y	Hands_L_Cout_z	Hands_R_Cout_x	Hands_R_Cout_y	Hands_R_Cout_z	Hands_L_R2_x	Hands_L_R2_y	Hands_L_R2_z	Hands_L_M1_x	Hands_L_M1_y	Hands_L_M1_z	Hands_L_I1_x	Hands_L_I1_y	Hands_L_I1_z	Hands_L_Cin_x	Hands_L_Cin_y	Hands_L_Cin_z	Hands_L_R1_x	Hands_L_R1_y	Hands_L_R1_z	Hands_L_L1_x	Hands_L_L1_y	Hands_L_L1_z	Hands_L_L2_x	Hands_L_L2_y	Hands_L_L2_z	Hands_L_Aout_x	Hands_L_Aout_y	Hands_L_Aout_z	Hands_L_I2_x	Hands_L_I2_y	Hands_L_I2_z	Hands_R_L2_x	Hands_R_L2_y	Hands_R_L2_z	Hands_L_M2_x	Hands_L_M2_y	Hands_L_M2_z	Hands_L_T2_x	Hands_L_T2_y	Hands_L_T2_z	Hands_R_R3_x	Hands_R_R3_y	Hands_R_R3_z	Hands_L_I3_x	Hands_L_I3_y	Hands_L_I3_z	Hands_L_R4_x	Hands_L_R4_y	Hands_L_R4_z	Hands_L_T1_x	Hands_L_T1_y	Hands_L_T1_z	Hands_L_R3_x	Hands_L_R3_y	Hands_L_R3_z	Hands_L_M4_x	Hands_L_M4_y	Hands_L_M4_z	Hands_L_L4_x	Hands_L_L4_y	Hands_L_L4_z	Hands_L_L3_x	Hands_L_L3_y	Hands_L_L3_z	Hands_L_M3_x	Hands_L_M3_y	Hands_L_M3_z	Hands_R_Win_x	Hands_R_Win_y	Hands_R_Win_z	Hands_R_Cin_x	Hands_R_Cin_y	Hands_R_Cin_z	Hands_R_L1_x	Hands_R_L1_y	Hands_R_L1_z	Hands_L_Win_x	Hands_L_Win_y	Hands_L_Win_z	Hands_L_Wout_x	Hands_L_Wout_y	Hands_L_Wout_z	Hands_R_I1_x	Hands_R_I1_y	Hands_R_I1_z	Hands_L_Ain_x	Hands_L_Ain_y	Hands_L_Ain_z	Hands_R_I2_x	Hands_R_I2_y	Hands_R_I2_z	Hands_R_L3_x	Hands_R_L3_y	Hands_R_L3_z	Hands_R_R2_x	Hands_R_R2_y	Hands_R_R2_z	Hands_R_R1_x	Hands_R_R1_y	Hands_R_R1_z	Hands_R_M3_x	Hands_R_M3_y	Hands_R_M3_z	Hands_R_Wout_x	Hands_R_Wout_y	Hands_R_Wout_z	Hands_R_Ain_x	Hands_R_Ain_y	Hands_R_Ain_z	Hands_R_T2_x	Hands_R_T2_y	Hands_R_T2_z	Hands_R_T3_x	Hands_R_T3_y	Hands_R_T3_z	Hands_R_T1_x	Hands_R_T1_y	Hands_R_T1_z	Hands_R_M1_x	Hands_R_M1_y	Hands_R_M1_z	Hands_R_M2_x	Hands_R_M2_y	Hands_R_M2_z	Hands_R_Aout_x	Hands_R_Aout_y	Hands_R_Aout_z	Hands_R_M4_x	Hands_R_M4_y	Hands_R_M4_z	Hands_R_L4_x	Hands_R_L4_y	Hands_R_L4_z	Hands_R_R4_x	Hands_R_R4_y	Hands_R_R4_z	Hands_L_I4_x	Hands_L_I4_y	Hands_L_I4_z	Hands_L_T4_x	Hands_L_T4_y	Hands_L_T4_z	Hands_R_I3_x	Hands_R_I3_y	Hands_R_I3_z	Hands_R_T4_x	Hands_R_T4_y	Hands_R_T4_z	Hands_R_I4_x	Hands_R_I4_y	Hands_R_I4_z	Hands_L_T3_x	Hands_L_T3_y	Hands_L_T3_z
# I will need key symbol and the columns from Hands_L_Cout_x to the right

class KeyPressDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir="C:\\Users\\mno64\\Datasets\\How we type\\Motion Capture", min_seq_len=8):
		def default():
			return "Float32"
		types = collections.defaultdict(default)
		types["input_index"] = pd.Int64Dtype()

		self.data_dir = data_dir
		self.text_column = "typed"
		self.feature_columns = [
			"Hands_L_Cout_x", "Hands_L_Cout_y", "Hands_L_Cout_z",
			"Hands_R_Cout_x", "Hands_R_Cout_y", "Hands_R_Cout_z",
			"Hands_L_Wout_x", "Hands_L_Wout_y", "Hands_L_Wout_z",
			"Hands_R_Wout_x", "Hands_R_Wout_y", "Hands_R_Wout_z",
			"Hands_L_Ain_x", "Hands_L_Ain_y", "Hands_L_Ain_z",
			"Hands_R_Ain_x", "Hands_R_Ain_y", "Hands_R_Ain_z",
			"Hands_L_Win_x", "Hands_L_Win_y", "Hands_L_Win_z",
			"Hands_R_Win_x", "Hands_R_Win_y", "Hands_R_Win_z",
		]
		files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
		if not files:
			raise FileNotFoundError(f"No CSV files found in {data_dir}")

		use_cols = ["uid", "condition", "stimulus", "input_index", "time", self.text_column] + self.feature_columns
		frames = []
		for filename in files:
			frames.append(pd.read_csv(filename, sep='\t', skiprows=2, usecols=use_cols, quoting=csv.QUOTE_NONE, low_memory=False))
		self.data = pd.concat(frames, axis=0, ignore_index=True)

		self.data[self.text_column] = self.data[self.text_column].fillna("").astype(str)
		self.data["uid"] = self.data["uid"].fillna("").astype(str)
		self.data["stimulus"] = self.data["stimulus"].fillna("").astype(str)
		self.data["condition"] = self.data["condition"].fillna("").astype(str)
		self.data["input_index"] = pd.to_numeric(self.data["input_index"], errors="coerce")
		self.data["time"] = pd.to_numeric(self.data["time"], errors="coerce")
		self.data["sample_id"] = self.data["uid"].astype(str) + "::" + self.data["stimulus"].astype(str) + "::" + self.data["condition"].astype(str)

		samples = []
		for _, group in self.data.groupby("sample_id", sort=False):
			group = group.sort_values(by=["input_index", "time"], kind="mergesort")
			text_values = [str(value).strip() for value in group[self.text_column].tolist() if str(value).strip()]
			if not text_values:
				continue

			feature_values = group[self.feature_columns].to_numpy(dtype="float32")
			feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
			if len(feature_values) < min_seq_len:
				continue

			# Frame-to-frame deltas, matching the paper's motion feature description.
			deltas = np.diff(feature_values, axis=0, prepend=feature_values[:1], append=feature_values[-1:])
			samples.append((deltas, text_values[-1]))

		self.samples = samples
		self.vocabulary = ["<blank>", "<unk>"] + sorted({char for _, text in samples for char in text})
		self.char_to_index = {char: idx for idx, char in enumerate(self.vocabulary)}
		self.num_features = len(self.feature_columns)
		print(f"Loaded {len(self.samples)} typed-text samples with {len(self.vocabulary)} symbols")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		features, target_text = self.samples[idx]
		encoded = torch.tensor([self.char_to_index.get(char, self.char_to_index["<unk>"]) for char in target_text], dtype=torch.long)
		return torch.from_numpy(features).float(), encoded, torch.tensor(len(encoded), dtype=torch.long)


class LegacyKeyPressDataset(torch.utils.data.Dataset):
	def __init__(self):
		def default():
			return "Float32"
		types = collections.defaultdict(default)
		types["input_index"] = pd.Int64Dtype()

		dir = "C:\\Users\\mno64\\Datasets\\How we type\\Motion Capture"
		use_cols = ["input_index", "Hands_L_Cout_x", "Hands_L_Cout_y", "Hands_L_Cout_z", "Hands_R_Cout_x", "Hands_R_Cout_y", "Hands_R_Cout_z", "Hands_L_R2_x", "Hands_L_R2_y", "Hands_L_R2_z", "Hands_L_M1_x", "Hands_L_M1_y", "Hands_L_M1_z", "Hands_L_I1_x", "Hands_L_I1_y", "Hands_L_I1_z", "Hands_L_Cin_x", "Hands_L_Cin_y", "Hands_L_Cin_z", "Hands_L_R1_x", "Hands_L_R1_y", "Hands_L_R1_z", "Hands_L_L1_x", "Hands_L_L1_y", "Hands_L_L1_z", "Hands_L_L2_x", "Hands_L_L2_y", "Hands_L_L2_z", "Hands_L_Aout_x", "Hands_L_Aout_y", "Hands_L_Aout_z", "Hands_L_I2_x", "Hands_L_I2_y", "Hands_L_I2_z", "Hands_R_L2_x", "Hands_R_L2_y", "Hands_R_L2_z", "Hands_L_M2_x", "Hands_L_M2_y", "Hands_L_M2_z", "Hands_L_T2_x", "Hands_L_T2_y", "Hands_L_T2_z", "Hands_R_R3_x", "Hands_R_R3_y", "Hands_R_R3_z", "Hands_L_I3_x", "Hands_L_I3_y", "Hands_L_I3_z", "Hands_L_R4_x", "Hands_L_R4_y", "Hands_L_R4_z", "Hands_L_T1_x", "Hands_L_T1_y", "Hands_L_T1_z", "Hands_L_R3_x", "Hands_L_R3_y", "Hands_L_R3_z", "Hands_L_M4_x", "Hands_L_M4_y", "Hands_L_M4_z", "Hands_L_L4_x", "Hands_L_L4_y", "Hands_L_L4_z", "Hands_L_L3_x", "Hands_L_L3_y", "Hands_L_L3_z", "Hands_L_M3_x", "Hands_L_M3_y", "Hands_L_M3_z", "Hands_R_Win_x", "Hands_R_Win_y", "Hands_R_Win_z", "Hands_R_Cin_x", "Hands_R_Cin_y", "Hands_R_Cin_z", "Hands_R_L1_x", "Hands_R_L1_y", "Hands_R_L1_z", "Hands_L_Win_x", "Hands_L_Win_y", "Hands_L_Win_z", "Hands_L_Wout_x", "Hands_L_Wout_y", "Hands_L_Wout_z", "Hands_R_I1_x", "Hands_R_I1_y", "Hands_R_I1_z", "Hands_L_Ain_x", "Hands_L_Ain_y", "Hands_L_Ain_z", "Hands_R_I2_x", "Hands_R_I2_y", "Hands_R_I2_z", "Hands_R_L3_x", "Hands_R_L3_y", "Hands_R_L3_z", "Hands_R_R2_x", "Hands_R_R2_y", "Hands_R_R2_z", "Hands_R_R1_x", "Hands_R_R1_y", "Hands_R_R1_z", "Hands_R_M3_x", "Hands_R_M3_y", "Hands_R_M3_z", "Hands_R_Wout_x", "Hands_R_Wout_y", "Hands_R_Wout_z", "Hands_R_Ain_x", "Hands_R_Ain_y", "Hands_R_Ain_z", "Hands_R_T2_x", "Hands_R_T2_y", "Hands_R_T2_z", "Hands_R_T3_x", "Hands_R_T3_y", "Hands_R_T3_z", "Hands_R_T1_x", "Hands_R_T1_y", "Hands_R_T1_z", "Hands_R_M1_x", "Hands_R_M1_y", "Hands_R_M1_z", "Hands_R_M2_x", "Hands_R_M2_y", "Hands_R_M2_z", "Hands_R_Aout_x", "Hands_R_Aout_y", "Hands_R_Aout_z", "Hands_R_M4_x", "Hands_R_M4_y", "Hands_R_M4_z", "Hands_R_L4_x", "Hands_R_L4_y", "Hands_R_L4_z", "Hands_R_R4_x", "Hands_R_R4_y", "Hands_R_R4_z", "Hands_L_I4_x", "Hands_L_I4_y", "Hands_L_I4_z", "Hands_L_T4_x", "Hands_L_T4_y", "Hands_L_T4_z", "Hands_R_I3_x", "Hands_R_I3_y", "Hands_R_I3_z", "Hands_R_T4_x", "Hands_R_T4_y", "Hands_R_T4_z", "Hands_R_I4_x", "Hands_R_I4_y", "Hands_R_I4_z", "Hands_L_T3_x", "Hands_L_T3_y", "Hands_L_T3_z"]
		self.data = pd.concat([pd.read_csv(filename, sep='\t', skiprows=2, usecols=use_cols, dtype=types, quoting=csv.QUOTE_NONE) for filename in glob.glob(dir + "/*.csv")], axis=0)

		labels = self.data.iloc[:, 0]
		labels = labels.fillna(0).to_numpy(dtype="int64")
		features = self.data.iloc[:, 1:].to_numpy(dtype="float32")

		self.labels = torch.from_numpy(labels).to(device)
		self.features = torch.from_numpy(features).to(device)
		print("init complete")

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		return self.features[idx], self.labels[idx]


class ResidualTCNBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=2, dilation=3):
		super().__init__()
		padding = (kernel_size - 1) * dilation
		self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
		self.conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation))
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

def collate_batch(batch):
	features, targets, target_lengths = zip(*batch)
	padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
	padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
	input_lengths = torch.tensor([x.size(0) for x in features], dtype=torch.long)
	target_lengths = torch.tensor(target_lengths, dtype=torch.long)
	return padded_features, padded_targets, input_lengths, target_lengths


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
	size = len(dataloader.dataset)
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
	dataset = KeyPressDataset()
	train_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_batch)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_batch)

	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	model = KeyPressModel(input_features=dataset.num_features, num_classes=len(dataset.vocabulary)).to(device)

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


# Skeleton to text	- Temporal Convolutional Network (TCN)
# We follow the TCN architecture proposed by Bai et al. 	https://arxiv.org/pdf/1803.01271
# which consists of causal dilated convolutions and weight normalization arranged in residual blocks.
# We use three layers of residual blocks with 64, 64, and 32 hidden units respectively, a kernel 
# size of 2 and a dilation factor of 3.

# They run it at 60hz

# The input features to the network are frame-to-frame deltas of wrist position and rotation along 
# with 3D fingertip positions. All positions are represented in the coordinate frame of the keyboard

# The network is trained with a batch size of 32, where each individual sample consists of input
# features $$ \{u_i\}_{i=1}^{T} $$, where T depends on how long it took the participant to type the
# phrase. The target for each sample is the sequence of keys $$ \{\hat{w}_j\}_{j=1}^{N} $$ that were 
# prompted to the participant. The network is trained using the CTC loss function, 		https://dl.acm.org/doi/pdf/10.1145/1143844.1143891
# which allows for sequence level labels without needing known alignment of labels to individual
# frames of input data. The output of the network $$ V= \{v_i\}_{i=1}^{T} $$ is a corresponding 
# sequence of T frames, each containing a probability distribution $$ v_i(k) $$ over the set of
# the K + 1 possible keys (with one extra for the blank label).




# Language Model

# For our language model, we use a Transformer model similar to the “small-two” model described in https://arxiv.org/pdf/1910.11450
# with # 4.76M parameters. This language model was trained on using a window of 500 characters on
# text sampled from 2 million articles in the CC-News dataset [20].



# Decoder
# We can do better than greedy decoding by using a prefix beam search decoder.		https://arxiv.org/pdf/1408.2873
# A prefix beam search decoder approximately maximizes p(W ;V) by incrementally constructing
# W by tracking a set of B best candidates (beams) at any given step. Furthermore, we can incorporate
# a joint probability from both the likelihood of the beam according to the motion model p(W ;V) as 
# well as the likelihood of the compacted text according to a language model $$ p_{lm}(W) $$, i.e.,
# $$ p_{total} = (p(W;V)p_{lm}(W)^\gamma)^{\frac{1}{1+\gamma}} $$ where γ is a hyperparameter
# to control the balance between the two likelihoods. This allows the language model to steer
# decoding when the motion model is uncertain. As has been shown in the speech recognition community,
# beam search decoding is also well-suited to the CTC loss with which we train the network. We use 
# a beam search implementation with beam compaction which maximizes the objective
# $$ W=\text{argmax}_Wp_{total} $$ After decoding, we compute the uncorrected error rate
# $$ \text{UER}(W,\hat{W}) $$ as the Levenstein edit distance between the decoded and prompted strings
# divided by the number of characters of the longer of the two strings.
#
# https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306


# 1. 2D Motion model
# :
# 2. Language model
# 2a. Find one
# 2b. Load it
# 3. Prefix beam search
# 3a. extract likliehoods from motion model and language model
# 3b. implement it