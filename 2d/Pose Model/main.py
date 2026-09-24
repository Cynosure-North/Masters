# https://ieeexplore.ieee.org/document/8764534

# To estimate the pose of each finger, we measured the angle ($$ \theta $$) between the hand joint vectors as a
# representative metric. It is indicative of the degree of finger bending. The angle was calculated
# as a cosine function as follows. (MW vectors point to the wrist, and MF to the fingertip)


# $$ cos\theta_n = \frac{\overrightarrow{MW_n}; \overrightarrow{MF_n}}{\norm{\overrightarrow{MW_n}} \cross \norm{{\overrightarrow{MF_n}}}}, n \in \{\text{all fingers}\}$$


# Exceptionally, there were no differences in finger angles between when entering the Y and U keys
# for all fingers including the touching finger (p > 0.05). To differentiate these two keys, we were
# required to analyze another hand characteristic that affects the global hand position, such as the
# hand direction. We checked whether the hand direction was different depending on input keys using
# the same analysis used for analyzing finger angles. The hand direction was estimated as follows


# $$ \text{hand direction} = \frac{\overrightarrow{WM_{index}} + \overrightarrow{WM_{little}}}{\norm{\overrightarrow{WM_{index}} + \overrightarrow{WM_{little}}}} $$


# MLP is one of most famous machine learning algorithms, and it has the capability to learn both
# linear and non-linear models. The basic structure consists of an input, hidden, and output layers
# where each layer has a plurality of neurons, and each neuron has its weighted connections to all
# neurons at the next layer. Each element of an input vector, i.e., the hand joint vector in our
# case, is inserted into the corresponding neuron at the input layer. Each of the elements is
# multiplied by the weight on its connection to the next layer and then, all the multiplied values
# are summed up into one accumulated value. The accumulated value is eventually passed through a
# specific activation function to the corresponding neuron of the next layer. When an input vector
# reaches the output layer, the neuron values of the output layer become the predicted result. In
# this work, we used a MLP with one hidden layer. To determine the optimal MLP model, we then
# compared the combinations of the hyper parameters: activation function = {tanh, relu, sigmoid},
# the number of neurons in hidden layer = {all multiples of 10 between 10 and 100}

# Pre-allocation keyboard (or P-key) is designed based on Key Pre-allocation. When a touch event
# occurs, it brings the keys allocated to that touching finger and selects the one closest to the
# touch point. As the identity of touching finger limits the candidates of inputtable keys, the
# P-key performs better than the Normal keyboard. Specifically, the P-key was effective in reducing
# horizontal typing errors. Pre-allocation and Hand Pose aware keyboard (or HP-key) is designed
# based on both Key Pre-allocation and Key Inference based on Hand Poses. When a touch event occurs,
# a target key is inferred through the touching finger's key inference model. Therefore, we expected
# that the HP-key reduced the horizontal and vertical typing errors at the same time

import torch
from torch.utils.data import DataLoader
import editdistance # pyright: ignore[reportMissingModuleSource]
from pathlib import Path

import MLP
from data import PoseDataset

def interactive():
	pass
	# TODO: Interactive mode for pose

@torch.no_grad()
def test():
	project_dir = Path(__file__).resolve().parent
	model_path = project_dir / "trained" / "model_weights.pth"
	test_path = project_dir / "data" / "train.csv"

	dataset = PoseDataset(test_path)
	loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
	model = MLP.instantiate_model(model_path).eval()

	total_length
	err_distance = 0
	correct_count = 0

	for data, label in loader:
		prediction = model(data)

		levenshtein = editdistance.eval(prediction, label)
		total_length += len(label)
		if prediction == label:
			correct_count += 1
		# print(f"CORRECT - {prediction}")
		else:
			err_distance += levenshtein
			# print(f"{levenshtein} - prediction: {prediction}, correct answer {label}")

	word_accuracy = correct_count / len(dataset)
	distance_accuracy = err_distance / total_length
	print(f"Test accuracy (Correct phrases) {word_accuracy:.2%}")
	print(f"Test accuracy (Levenshtien distance) {distance_accuracy:.2%}")

if __name__ == "__main__":
	test()