import torch
import numpy as np
from llama_cpp import Llama, LogitsProcessorList
import re
import weakref

class LLM:
	def __init__(self, path):
		# Load or pre-calculate non-ASCII token IDs to keep inference fast
		cache_path = path.with_name(path.stem + "-non_ascii_token_ids.npy")

		if cache_path.exists():
			non_ascii_token_ids = np.load(cache_path)
			print(f"Loaded cached non-ASCII tokens from {cache_path} ({len(non_ascii_token_ids)} ids)")
		else:
			model = Llama(model_path=str(path), verbose=False)
			try:
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
			finally:
				model.close()

			# Convert to a NumPy array for fast indexing
			non_ascii_token_ids = np.array(non_ascii_token_ids)
			np.save(cache_path, non_ascii_token_ids)
			print(f"Saved non-ASCII token IDs to {cache_path}")

		# Filter out non-ascii characters (and also html tags)
		def ascii_only_processor(_, scores):
			scores[non_ascii_token_ids] = -float("inf")
			return scores

		processors = LogitsProcessorList([ascii_only_processor])
		
		self.model = Llama(model_path=str(path), verbose=False, processors=processors, logits_all=True)
		self._model_finalizer = weakref.finalize(self, self.model.close)

	def get_word_probability(self, text):
		prefix, word = text.rsplit(" ", 1)
		word_id = self.model.tokenize(word, False)

		self.model.eval(prefix)

		raw_logits = self.model.scores[-1]
		probabilities = torch.nn.functional.softmax(raw_logits)
		return probabilities.tolist()[word_id]

	def close(self):
		if self._model_finalizer.alive:
			self._model_finalizer()



# TODO: Gemma 4 is more complicated than what they use, I should look into a simpler model
# For our language model, we use a  model similar to the “small-two” model described in	 https://arxiv.org/pdf/1910.11450
# with 4.76M parameters. This language model was trained on using a window of 500 characters on text
# sampled from 2 million articles in the CC-News dataset 