# https://dl.acm.org/doi/10.1145/3379337.3415816

# The problem of generating text from hand motion has strong analogs to automatic speech recognition
#  (ASR), and we use ASR as motivation to design a multi-component system with the following three
# pieces.
#
# 1. A motion model, analogous to an acoustic model, which takes a sequence of hand poses and for
# each frame outputs a likelihood over an alphabet of tokens, i.e., keys plus a blank token
# corresponding to no key.
# 2. A language model which can give the likelihood of an additional token given a prefix of tokens.
# 3. A decoder which can optimize an objective function combining the likelihoods from both the
# motion and the language models.
# The motion model captures the mapping from finger trajectories to intended key presses. This
# mapping is inherently ambiguous, for example, when a finger strikes the boundary between two keys,
# and is exacerbated by drift of the users’ fingers due to the lack of haptic feedback of physical
# keys. We apply a beam search decoder to resolve these ambiguities using the language model as a
# prior.

# For temporal modeling, we opt to use a temporal convolutional network (TCN) rather than a
# recurrent model because a fixed window of hand motion data is typically sufficient to make
# predictions about key presses. Longer term context is useful for prediction in the context of
# language, but since our system design separates motion modeling from language modeling, a TCN
# works well for the former. An advantage of using a TCN is efficiency during training, since TCNs
# can process entire sequences in parallel rather than sequentially as with recurrent neural
# networks. We follow the TCN architecture proposed by Bai et al. [2] which consists of causal
# dilated convolutions and weight normalization arranged in residual blocks. We use three layers of
# residual blocks with 64, 64, and 32 hidden units respectively, a kernel size of 2 and a dilation
# factor of 3. This architecture allows for 46 past frames of hand features (around 0.75 seconds of
# motion at 60Hz) to be referenced to generate the current probability distribution over the set of
# keys.
# The input features to the network are frame-to-frame deltas of wrist position and rotation along
# with 3D fingertip positions. All positions are represented in the coordinate frame of the keyboard
# (the touchpad, or a virtual keyboard in the runtime). We chose these features because they are
# somewhat invariant to differences in hand scale across people.
# The network is trained with a batch size of 32, where each individual sample consists of input
# features $$ ī\{u_i\}_{i=1}^T $$, where T depends on how long it took the participant to type the phrase. The
# target for each sample is the sequence of keys $$ \hat{W} = \{\hat{W}_j\}_{j=1}^N $$ that were prompted to the participant.
# The network is trained using the CTC loss function, which allows for sequence level labels without
# needing known alignment of labels to individual frames of input data. The output of the network
# $$ V = \{v_i\}_{i=1}^T $$ is a corresponding sequence of T frames, each containing a probability distribution
# $$ v_i(k) $$ over the set of the $$ K + 1 $$ possible keys (with one extra for the blank label). This output can
# be interpreted as a $$ T \times (K + 1) $$ heat map containing the prediction of which key was hit at which
# time. To transcribe the probabilities V into typed text, this output must be decoded into a
# sequence of keys W.
#
# A naive greedy approach to decoding is to generate a label for each frame $$ t \in T $$ by taking
# $$ \ell_t = \text{argmax}_k v_t(k) $$. These labels $$ \{\ell\}_{t=1}^T $$ are compacted into a transcription W by first combining
# adjacent frames with the same label into one of that label, and then by removing all blank labels.
# We can do better than greedy decoding by using a prefix beam search decoder. A prefix beam search
# decoder approximately maximizes $$ p(W; V) $$ by incrementally constructing W by tracking a set of B
# best candidates (beams) at any given step. Furthermore, we can incorporate a joint probability
# from both the likelihood of the beam according to the motion model $$ p(W; V) $$ as well as the
# likelihood of the compacted text according to a language model $$ p_{lm}(W)$$, i.e.
# $$ p_{total} = (p(W; V)p_lm(W)^\gamma)^{\frac{1}{a+\gamma}} $$ where $$ \gamma $$ is a hyperparameter to control the balance between the two
# likelihoods. This allows the language model to steer decoding when the motion model is uncertain.
# As has been shown in the speech recognition community, beam search decoding is also well-suited to
# the CTC loss with which we train the network.
# We use a beam search implementation with beam compaction which maximizes the objective
# $$ W = \text{argmax}_W p_{total} $$. After decoding, we compute the uncorrected error rate $$ UER(W, \hat{W}) $$ as the
# Levenstein edit distance between the decoded and prompted strings divided by the number of
# characters of the longer of the two strings. For all of our results, we use $$ B = 100 $$ beams. For our
# language model, we use a Transformer model similar to the “small-two” model described in Hongzhao
# Huang and Fuchun Peng. 2019. An Empirical Study of Efficient ASR Rescoring with Transformers. with
# 4.76M parameters. This language model was trained on using a window of 500 characters on text
# sampled from 2 million articles in the CC-News dataset.

import torch
from torch.utils.data import DataLoader
import editdistance # pyright: ignore[reportMissingModuleSource]
from pathlib import Path

import LLM
import TCN
from BeamSearch import prefix_beam_search
from data import MotionDataset

def interactive(tnn, llm):
	pass
	# TODO: Interactive mode
	# Run at 60fps
	# If a new keypress has come in from the TNN update the next char
	# Else use the off-frames to run the llm in batches (of 4?) to manage performance

@torch.no_grad()
def test():
	project_dir = Path(__file__).resolve().parent
	tcn_path = project_dir / "trained" / "tnn.pth"
	llm_path = project_dir / "downloaded" / "gemma-4-e2b-q4_k_m.gguf"
	test_path = project_dir / "data" / "train.csv"

	dataset = MotionDataset(test_path)
	loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
	tcn = TCN.instantiate_model(tcn_path).eval()
	llm = LLM.LLM(llm_path).get_word_probability

	total_length
	err_distance = 0
	correct_count = 0

	for data, label in loader:
		tnn_prediction = tcn(data)
		prediction = prefix_beam_search(tnn_prediction, llm)

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