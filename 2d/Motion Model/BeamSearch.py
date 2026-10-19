# Modified version of this code
# https://zhaozeyu1995.github.io/CTC-Prefix-Beam-Search-Decoding-Algorithm-with-Language-Model/
# https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306

from collections import defaultdict, deque
import editdistance
import re
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from data import chars, MotionDataset
import LLM
import TCN
import pickle


_alphabet = chars + ['-', '>']	# Blank character and end character
num_features = len(chars) + 1	# Also the non-character (when no key is pressed)
convergence_delay=6

# Init
prev_Pb, prev_Pnb = defaultdict(float), defaultdict(float)
prev_Pb[''] = 1		# Probability of blank character
prev_Pnb[''] = 0		# Probability of non-blank character
previous_prefixes = None


def incremental_prefix_beam_search(
	ctc,
	lm,
	*,
	k=100,
	alpha=0.30,
	beta=5,		# TODO: Tune llm compensation factor
	prune=0.001,
	alphabet=_alphabet,
	convergence_delay=6
):
	"""
	Args:
		ctc (array): The CTC output from the latest frame. Should be a 1D array (alphabet_size + '-')
		lm (func): Language model function. Should take as input a string and output a probability.
		k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
		alpha (float): The language model weight. Should usually be between 0 and 1.
		beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
		prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.

	Returns:
		string: The decoded CTC output.
	"""
	global prev_Pb, prev_Pnb, previous_prefixes

	if previous_prefixes is None:
		previous_prefixes = deque(maxlen=(convergence_delay if convergence_delay != 0 else 1))
		previous_prefixes.append([''])

	pruned_alphabet = [alphabet[idx] for idx, val in enumerate(ctc) if val > prune]
	Pb, Pnb = defaultdict(float), defaultdict(float)
	blank_prob = ctc[-1]

	for prefix in previous_prefixes[-1]:
		
		# Once you hit the end character stick with that
		if len(prefix) > 0 and prefix[-1] == '>':
			Pb[prefix] = prev_Pb[prefix]
			Pnb[prefix] = prev_Pnb[prefix]
			continue  

		for c in pruned_alphabet:
			char_prob = ctc[alphabet.index(c)]
			extended = prefix + c
			
			# Extending with a blank
			if c == '-':
				Pb[prefix] += blank_prob * (prev_Pb[prefix] + prev_Pnb[prefix])
			
			else:
				# Extending with the previous character
				if len(prefix) > 0 and c == prefix[-1]:
					Pnb[extended] += char_prob * prev_Pb[prefix]
					Pnb[prefix] += char_prob * prev_Pnb[prefix]

				# Extending with space/end character - triggers LM likelihood check
				elif len(prefix.replace(' ', '')) > 0 and c in (' ', '>'):
					# With a convergence_delay of only a handful of frames I'm unsure how much
					# this can really do
					lm_prob = lm(extended.strip(' >')) ** alpha
					Pnb[extended] += lm_prob * char_prob * (prev_Pb[prefix] + prev_Pnb[prefix])
				# Extending with any other character
				else:
					Pnb[extended] += char_prob * (prev_Pb[prefix] + prev_Pnb[prefix])

				# Make use of discarded prefixes if they're available
				if extended not in previous_prefixes:
					Pb[extended] += blank_prob * (prev_Pb[extended] + prev_Pnb[extended])
					Pnb[extended] += char_prob * prev_Pnb[extended]


	def merge(dict1, dict2):
		new_dict = {}
		keys = dict1.keys() | dict2.keys()
		for key in keys:
			if key in dict1 and key in dict2:
				new_dict[key] = dict1[key] + dict2[key]
			elif key in dict1:
				new_dict[key] = dict1[key]
			else:
				new_dict[key] = dict2[key]
		return new_dict

	current_prefixes = merge(Pb, Pnb)
	
	# Select most probable prefixes
	word_count = lambda l: re.findall(r'\w+[\s>]', l)
	scorer = lambda l: current_prefixes[l] * (len(word_count(l)) + 1) ** beta
		
	if convergence_delay == 0:
		prev_Pb, prev_Pnb = Pb, Pnb
		current_prefixes = sorted(current_prefixes, key=scorer, reverse=True)[:min(k, len(current_prefixes))]
		previous_prefixes.append(current_prefixes)
		return current_prefixes[0].strip('>')

	# Force convergence
	if len(previous_prefixes) < convergence_delay:
		prev_Pb, prev_Pnb = Pb, Pnb
		current_prefixes = sorted(current_prefixes, key=scorer, reverse=True)
		previous_prefixes.append(current_prefixes[:min(k, len(current_prefixes))])
		return ''

	# The reason I select prefixes from convergence_delay frames ago is because in the prefixes from
	# this frame I can't tell when each character was added, so I can't select a stable sub-prefix
	relevance = {}
	for old_prefix in previous_prefixes[0]:
		score = 0
		for new_prefix in current_prefixes:
			levenshtein = editdistance.eval(old_prefix, new_prefix)
			score += (Pb[new_prefix] + Pnb[new_prefix]) * levenshtein
		relevance[old_prefix] = score * scorer(old_prefix)
	converged_prefix = sorted(relevance, key=relevance.get, reverse=True)[0]

	# Drop dead paths
	prev_Pb, prev_Pnb = defaultdict(float), defaultdict(float)
	for k, v in Pb.items():
		if k.startswith(converged_prefix):
			prev_Pb[k] = v
	for k, v in Pnb.items():
		if k.startswith(converged_prefix):
			prev_Pnb[k] = v
	current_prefixes = list(filter(lambda l: l.startswith(converged_prefix), current_prefixes))

	current_prefixes = sorted(current_prefixes, key=scorer, reverse=True)[:min(k, len(current_prefixes))]
	previous_prefixes.append(current_prefixes)

	return previous_prefixes[0][0].strip('>')

def reset_incremental_prefix_beam_search():
	previous_prefixes = None

def prefix_beam_search(
	ctc,
	lm,
	*,
	k=100,
	alpha=0.30,
	beta=5,
	prune=0.001,
	alphabet=_alphabet
):

	final_output = ""
	for frame in ctc:
		final_output = incremental_prefix_beam_search(frame, lm, k=k, alpha=alpha, beta=beta, prune=prune, convergence_delay=0, alphabet=alphabet)

	reset_incremental_prefix_beam_search()

	return final_output
	
def tune_alpha_beta(
	ctc_sequences,
	lm,
	texts,
	*,
	alpha_values=(0, 0.001, 0.005, 0.1, 0.2, 0.3, 0.5, 0.7),
	beta_values=(0, 0.5, 1, 1.5, 3, 5, 8, 12),
):
	"""Grid-search alpha/beta pairs on a set of target strings and return the best pair."""
	if len(ctc_sequences) != len(texts):
		raise ValueError("ctc_sequences and texts must have the same length.")
	best_result = (-float('inf'), -1, -1)
	for alpha in alpha_values:
		for beta in beta_values:
			matches = 0
			for ctc, expected in zip(ctc_sequences, texts):
				predicted = prefix_beam_search(ctc, lm, alpha=alpha, beta=beta)
				matches += int(predicted == expected)
			score = matches / max(len(texts), 1)
			candidate = (score, alpha, beta)
			print(f"a: {alpha:<5.f}, b:{beta:<3.f} - {score:.3f}")
			if best_result is None or candidate[0] > best_result[0]:
				best_result = candidate
	return (best_result[1], best_result[2], best_result[0])

def run_tuning():
	from test import random_split, dir_path		# Import here to avoid circular imports

	train_path, _, _ = random_split(dir_path)
	project_dir = Path(__file__).resolve().parent
	llm_path = project_dir / "downloaded" / "gemma-4-e2b-q4_k_m.gguf"
	tcn_path = project_dir / "trained" / "test_TCN.pth"

	llm = LLM.LLM(llm_path).get_word_probability
	tcn = TCN.instantiate_model(tcn_path).eval()
	device = next(tcn.parameters()).device
	loader_kwargs = {"batch_size": 64, "num_workers": 2, "pin_memory": True}
	loader = DataLoader(
		MotionDataset(train_path),
		collate_fn=TCN.collate_batch,
		**loader_kwargs)

	ctc_outputs = []
	ctc_labels = []
	for features, labels, _, _ in loader:
		logits = tcn(features.to(device, non_blocking=True))

		for sample_logits, label in zip(logits, labels):
			sample_logits = sample_logits.cpu().tolist()

			ctc_outputs.append(sample_logits)
			ctc_labels.append(label)

	print("Loaded data, running sweep")

	alpha, beta, score = tune_alpha_beta(ctc_outputs, llm, ctc_labels)

	print(f"Best alpha: {alpha}")
	print(f"Best beta: {beta}")
	print(f"Best result (correct sequences/total sequences): {score:.2%}")

def greedy_decoder(ctc, alphabet=_alphabet):
	"""
	Performs greedy decoding (max decoding) on the output of a CTC network.

	Args:
	ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)

	Returns:
	string: The decoded CTC output.
	"""

	# Collapse repeating characters
	arg_max = np.argmax(ctc, axis=1)
	repeat_filter = arg_max[1:] != arg_max[:-1]
	repeat_filter = np.concatenate([[True], repeat_filter])
	collapsed = arg_max[repeat_filter]

	# discard blank tokens (the blank is always last in the alphabet)
	blank_filter = np.where(collapsed < (len(alphabet) - 1))[0]
	final_sequence = collapsed[blank_filter]
	full_decode = ''.join([alphabet[letter_idx] for letter_idx in final_sequence])
	return full_decode.strip('>')

def test():
	# Test data from https://github.com/corticph/prefix-beam-search
	# llm_path = Path(__file__).resolve().parent / "downloaded" / "gemma-4-e2b-q4_k_m.gguf"
	llm_path = Path(__file__).resolve().parent / "downloaded" / "Qwen3-0.6B-Base.Q4_K_S.gguf"
	llm = LLM.LLM(llm_path).get_word_probability

	alphabet = 'abcdefghijklmnopqrstuvwxyz >-'

	test_dir = Path(r"C:\Users\mno64\Datasets\prefix-beam-search\examples")
	for example_file in test_dir.iterdir():
		example = pickle.load(open(example_file, 'rb'))
		greedy_out = greedy_decoder(example, alphabet)
		pbs_out = prefix_beam_search(example, llm, alphabet=alphabet)
		print(f"{example_file}")
		print(f"Greedy Decode: {greedy_out}")
		print(f"Prefix Beam Search: {pbs_out}")
	

if __name__ == "__main__":
	test()