import numpy as np
import re
from pathlib import Path
from llama_cpp import Llama, LogitsProcessorList
import torch

model_name = "gemma-4-e2b-q4_k_m"
model_path = Path(__file__).parent.parent.joinpath("downloaded", model_name + ".gguf")

def load():
	# Load or pre-calculate non-ASCII token IDs to keep inference fast
	cache_path = model_path.withname(model_name + "-non_ascii_token_ids.npy")

	if cache_path.exists():
		non_ascii_token_ids = np.load(cache_path)
		print(f"Loaded cached non-ASCII tokens from {cache_path} ({len(non_ascii_token_ids)} ids)")
	else:
		model = Llama(model_path=str(model_path), verbose=False)

		non_ascii_token_ids = []
		num_tokens = model.n_vocab()

		allowed = re.compile("^[A-Za-z_:;.,' -]*$")
		for token_id in range(num_tokens):
			try:
				# Convert token ID to string byte representation
				token_bytes = model.detokenize([token_id])
				token_str = token_bytes.decode("utf-8")
				
				# Check if the string contains non-ASCII characters
				if not allowed.search(token_str):
					non_ascii_token_ids.append(token_id)

			except UnicodeDecodeError:
				# Ban partial/invalid byte tokens that form multi-byte UTF-8 chars
				non_ascii_token_ids.append(token_id)

		# Convert to a NumPy array for fast indexing
		non_ascii_token_ids = np.array(non_ascii_token_ids)
		np.save(cache_path, non_ascii_token_ids)
		print(f"Saved non-ASCII token IDs to {cache_path}")

	# print(f"non ascii ids: {non_ascii_token_ids}")

	# Filter out non-ascii characters (and also html tags)
	def ascii_only_processor(_, scores):
		scores[non_ascii_token_ids] = -float("inf")
		return scores

	processors = LogitsProcessorList([ascii_only_processor])
	
	return Llama(model_path=str(model_path), verbose=False, processors=processors, logits_all=True)

model = load()

def get_word_probability(prefix, word):		# TODO: Test to make sure this works
	word_id = model.tokenize(word, False)

	model.eval(prefix)

	raw_logits = model.scores[-1]
	probailities = torch.nn.functional.softmax(raw_logits)
	return probailities[word_id]



# TODO: This is more complicated than what they use, I should look into a simpler model
# For our language model, we use a  model similar to the “small-two” model described in	 https://arxiv.org/pdf/1910.11450
# with 4.76M parameters. This language model was trained on using a window of 500 characters on text
# sampled from 2 million articles in the CC-News dataset 