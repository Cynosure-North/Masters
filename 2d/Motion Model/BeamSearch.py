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
import TCN

import ref_beamsearch



alphabet = chars + ['-']	# CTC blank (last column)
char_to_index = {character: index for index, character in enumerate(alphabet)}


class BeamSearcher():
	def __init__(self, language_model):
		if language_model is None:
			raise ValueError("language_model is required")
		self.llm = language_model

		self.prev_Pb, self.prev_Pnb = defaultdict(float), defaultdict(float)
		self.prev_Pb[''] = 1		# Probability of blank character
		self.prev_Pnb[''] = 0		# Probability of non-blank character
		self.previous_prefixes = None

	def incremental_prefix_beam_search(
		self,
		ctc,
		*,
		beam_width=100,
		alpha=0.30,
		beta=12,
		prune=0.001,
		convergence_delay=6
	):
		"""
		Args:
			ctc (array): The CTC output from the latest frame. Should be a 1D array (alphabet_size + '-')
			beam_width (int): Keep the n most likely candidates at each timestep.
			alpha (float): The language model weight. Should usually be between 0 and 1.
			beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
			prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.

		Returns:
			string: The decoded CTC output.
		"""

		if self.previous_prefixes is None:
			self.previous_prefixes = deque(maxlen=(convergence_delay if convergence_delay != 0 else 1))
			self.previous_prefixes.append([''])

		pruned_alphabet = [
			character for character, value in zip(alphabet, ctc)
			if value > prune
		]

		Pb, Pnb = defaultdict(float), defaultdict(float)
		blank_prob = ctc[-1]

		lm_candidates = list(dict.fromkeys(		# Precalculate lm probabilities in batches
			(prefix + c).strip()
			for prefix in self.previous_prefixes[-1]
			for c in pruned_alphabet
			if c != '-' and not (prefix and c == prefix[-1])
			and len(prefix.replace(' ', '')) > 0 and c == ' '
		))
		lm_log_probabilities = dict(zip(
			lm_candidates,
			self.llm.get_word_log_probabilities(lm_candidates),
		))

		for prefix in self.previous_prefixes[-1]:
			for c in pruned_alphabet:
				char_prob = ctc[char_to_index[c]]
				extended = prefix + c

				# Extending with a blank
				if c == '-':
					Pb[prefix] += blank_prob * (self.prev_Pb[prefix] + self.prev_Pnb[prefix])
				
				else:
					# Extending with the self.previous character
					if len(prefix) > 0 and c == prefix[-1]:
						Pnb[extended] += char_prob * self.prev_Pb[prefix]
						Pnb[prefix] += char_prob * self.prev_Pnb[prefix]

					# Extending with space/end character - triggers LM likelihood check
					elif len(prefix.replace(' ', '')) > 0 and c == ' ':
						# With a convergence_delay of only a handful of frames I'm unsure how much this can really do
						lm_prob = np.exp(alpha * lm_log_probabilities[extended.strip()])
						Pnb[extended] += lm_prob * char_prob * (self.prev_Pb[prefix] + self.prev_Pnb[prefix])
					# Extending with any other character
					else:
						Pnb[extended] += char_prob * (self.prev_Pb[prefix] + self.prev_Pnb[prefix])

					# Make use of discarded prefixes if they're available
					if extended not in self.previous_prefixes[-1]:
						Pb[extended] += blank_prob * (self.prev_Pb[extended] + self.prev_Pnb[extended])
						Pnb[extended] += char_prob * self.prev_Pnb[extended]


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
		word_count = lambda l: re.findall(r'\w+\s', l)
		scorer = lambda l: current_prefixes[l] * (len(word_count(l)) + 1) ** beta
		ranked_prefixes = sorted(current_prefixes, key=scorer, reverse=True)
			
		if convergence_delay == 0:
			self.prev_Pb, self.prev_Pnb = Pb, Pnb
			current_prefixes = ranked_prefixes[:min(beam_width, len(ranked_prefixes))]
			self.previous_prefixes.append(current_prefixes)
			return current_prefixes[0].strip()

		# Force convergence
		if len(self.previous_prefixes) < convergence_delay:
			self.prev_Pb, self.prev_Pnb = Pb, Pnb
			current_prefixes = sorted(current_prefixes, key=scorer, reverse=True)
			self.previous_prefixes.append(current_prefixes[:min(beam_width, len(current_prefixes))])
			return ''

		# The reason I select prefixes from convergence_delay frames ago is because in the prefixes from
		# this frame I can't tell when each character was added, so I can't select a stable sub-prefix
		irrelevance = {}
		for old_prefix in self.previous_prefixes[0]:
			score = 0
			for new_prefix in current_prefixes:
				levenshtein = editdistance.eval(old_prefix, new_prefix)
				score += (Pb[new_prefix] + Pnb[new_prefix]) * levenshtein
			irrelevance[old_prefix] = score * (len(old_prefix) ** beta)
		converged_prefix = sorted(irrelevance, key=irrelevance.get)[0]

		# Drop dead paths
		self.prev_Pb, self.prev_Pnb = defaultdict(float), defaultdict(float)
		for k, v in Pb.items():
			if k.startswith(converged_prefix):
				self.prev_Pb[k] = v
		for k, v in Pnb.items():
			if k.startswith(converged_prefix):
				self.prev_Pnb[k] = v
		current_prefixes = {k: v for k, v in current_prefixes.items() if k.startswith(converged_prefix)}

		current_prefixes = sorted(current_prefixes, key=scorer, reverse=True)[:min(beam_width, len(current_prefixes))]
		self.previous_prefixes.append(current_prefixes)

		return self.previous_prefixes[0][0].strip()

	def finalize(self, *, alpha=0.30, beta=12):
		"""Finalize an utterance, including a word that has no CTC separator.

		Interactive CTC input often stops immediately after the final character.
		Score each retained hypothesis as if a word boundary followed it so the
		last word receives the same language-model treatment as earlier words.
		"""
		if self.previous_prefixes is None or not self.previous_prefixes[-1]:
			return ''

		prefixes = self.previous_prefixes[-1]
		acoustic_scores = {
			prefix: self.prev_Pb[prefix] + self.prev_Pnb[prefix]
			for prefix in prefixes
		}
		unfinished = [
			prefix.rstrip()
			for prefix in prefixes
			if prefix and prefix.rstrip()
		]
		lm_log_scores = dict(zip(
			unfinished,
			self.llm.get_word_log_probabilities(unfinished),
		))

		def score(prefix):
			acoustic_score = acoustic_scores[prefix]
			if acoustic_score <= 0:
				return float('-inf')
			final_word = prefix.rstrip()
			lm_log_score = lm_log_scores.get(final_word, 0.0)
			word_count = len(re.findall(r'\w+\s', prefix))
			return (
				np.log(acoustic_score)
				+ alpha * lm_log_score
				+ beta * np.log(word_count + 1)
			)

		return max(prefixes, key=score).strip()

	def reset(self):
		self.previous_prefixes = None
		self.prev_Pb, self.prev_Pnb = defaultdict(float), defaultdict(float)
		self.prev_Pb[''] = 1
		self.prev_Pnb[''] = 0

def prefix_beam_search(ctc, beam_searcher, **kwargs):
	"""Decode a complete CTC sequence and finalize its last word."""
	final_output = ''
	for frame in ctc:
		final_output = beam_searcher.incremental_prefix_beam_search(
			frame, convergence_delay=0, **kwargs
		)

	if len(ctc) > 0:
		final_output = beam_searcher.finalize(
			alpha=kwargs.get('alpha', 0.30),
			beta=kwargs.get('beta', 12),
		)

	beam_searcher.reset()
	return final_output
	
def tune_alpha_beta(
	ctc_sequences,
	texts,
	language_model,
	*,
	alpha_values=(0, 0.001, 0.005, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9),
	beta_values=(0, 0.5, 1, 1.5, 3, 5, 8, 12),
):
	"""Grid-search decoder weights on arrays or sentence-named .npy exports.

	Pass a directory path as ``ctc_sequences`` to load the ``*.npy`` files and
	derive reference strings from their filenames. Otherwise, provide CTC arrays
	and matching ``texts`` as before.
	"""
	ctc_sequences, texts = ctc_sequences[:10], texts[:10]

	if ctc_sequences is None:
		raise ValueError("CTC sequences are required")
	if texts is None:
		raise ValueError("texts are required")
	if not texts:
		raise ValueError("At least one CTC sequence is required for tuning")
	# if len(ctc_sequences) != len(texts):
	# 	raise ValueError("ctc_sequences and texts must have the same length.")


	beam_searcher = BeamSearcher(language_model)
	best_result = (0, -1, -1)

	for alpha in alpha_values:
		for beta in beta_values:
			score = 1 - (sum(
				editdistance.eval(ref_beamsearch.prefix_beam_search(ctc, beam_searcher.llm.get_word_probability, alpha=alpha, beta=beta), expected)
				# prefix_beam_search(ctc, beam_searcher, alpha=alpha, beta=beta, alphabet=alphabet) == expected
				for ctc, expected in zip(ctc_sequences, texts)
			) / sum (len(text) for text in texts))

			print(f"alpha: {alpha:<5g}, beta: {beta:<3g} - matching characters: {score:.2%}")

			if score > best_result[0]:
				best_result = (score, alpha, beta)

	return (best_result[1], best_result[2], best_result[0])


def run_tuning():
	project_dir = Path(__file__).resolve().parent
	llm_path = project_dir / "trained" / "best_LLM.pth"
	test_dir = project_dir / "data" / "beamsearch_test"

	if True:
		print("Using pregenerated CTC output")
		ctc_outputs, ctc_labels = _load_sentence_named_ctc(test_dir)
	else:
		from test import random_split, dir_path		# Import here to avoid circular imports
		train_path, _, _ = random_split(dir_path)

		print("Using TCN output")
		tcn_path = project_dir / "trained" / "best_TCN.pth"

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

	import LLM
	language_model = LLM.instantiate_model(
		llm_path,
		compile_model=True)
	alpha, beta, score = tune_alpha_beta(ctc_outputs, ctc_labels, language_model)

	print(f"Best alpha: {alpha}")
	print(f"Best beta: {beta}")
	print(f"Best result (correct characters/total characters): {score:.2%}")

def greedy_decoder(ctc):
	# Collapse repeating characters
	arg_max = np.argmax(ctc, axis=1)
	repeat_filter = arg_max[1:] != arg_max[:-1]
	repeat_filter = np.concatenate([[True], repeat_filter])
	collapsed = arg_max[repeat_filter]

	# discard blank tokens (the blank is always last in the alphabet)
	blank_filter = np.where(collapsed < (len(alphabet) - 1))[0]
	final_sequence = collapsed[blank_filter]
	full_decode = ''.join([alphabet[letter_idx] for letter_idx in final_sequence])
	return full_decode.strip()


def _load_sentence_named_ctc(test_dir):
	"""Load filename-labeled wav2vec CTC arrays in the decoder's alphabet order."""
	import torchaudio # pyright: ignore[reportMissingImports]

	test_dir = Path(test_dir)
	if not test_dir.is_dir():
		raise FileNotFoundError(f"CTC test directory not found: {test_dir}")

	bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
	model_labels = bundle.get_labels()
	beam_alphabet = alphabet
	beam_indices = {character: index for index, character in enumerate(beam_alphabet)}
	ctc_files = sorted(test_dir.glob("*.npy"))
	if not ctc_files:
		raise FileNotFoundError(f"No .npy CTC arrays found in: {test_dir}")

	ctc_sequences = []
	texts = []
	for example_file in ctc_files[:10]:		# FIXME: I'm using a subset of the files to test BeamSearch because it runs slow, I'd like to speed it up and run a full test
		model_ctc = np.load(example_file, allow_pickle=False)
		if model_ctc.ndim != 2 or model_ctc.shape[1] != len(model_labels):
			raise ValueError(
				f"Expected a (time, {len(model_labels)}) CTC array in {example_file}; "
				f"got shape {model_ctc.shape}"
			)

		# Wav2vec uses blank-first uppercase labels; the local decoder uses
		# lowercase labels and places blank last. Fold unsupported symbols into blank.
		ctc = np.zeros((model_ctc.shape[0], len(beam_alphabet)), dtype=np.float32)
		for model_index, label in enumerate(model_labels):
			if label == "-":
				beam_index = beam_indices["-"]
			elif label == "|":
				beam_index = beam_indices[" "]
			else:
				beam_index = beam_indices.get(label.lower(), beam_indices["-"])
			ctc[:, beam_index] += model_ctc[:, model_index]

		# gen.py suffixes repeated transcript filenames with _2, _3, etc.
		text = re.sub(r"_\d+$", "", example_file.stem)
		ctc_sequences.append(ctc)
		texts.append(re.sub(f"[^{chars}]", " ", text.lower()))

	return ctc_sequences, texts


def test():
	llm_path = Path(__file__).resolve().parent / "trained" / "best_LLM.pth"
	test_dir = Path(__file__).resolve().parent / "data" / "beamsearch_test"
	ctc_sequences, texts = _load_sentence_named_ctc(test_dir)

	import LLM
	language_model = LLM.instantiate_model(
		llm_path,
		compile_model=True,
	)
	beam_searcher = BeamSearcher(language_model)
	for ctc, expected_text in zip(ctc_sequences, texts):
		greedy_out = greedy_decoder(ctc)
		pbs_out = prefix_beam_search(ctc, beam_searcher)
		# ref_pbs_out = ref_beamsearch.prefix_beam_search(ctc, beam_searcher.llm.get_word_probability)
		print()
		print(f"Reference:     {expected_text}")
		print(f"Greedy Decode: {greedy_out}")
		print(f"My PBS:        {pbs_out}")
		# print(f"Ref PBS:       {ref_pbs_out}")
	

if __name__ == "__main__":
	test()
	
# The final word is often wrong
# There's one example which is fucked (#8)
# I'd like if it was faster