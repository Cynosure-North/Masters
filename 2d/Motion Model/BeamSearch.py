# Modified version of this code
# https://zhaozeyu1995.github.io/CTC-Prefix-Beam-Search-Decoding-Algorithm-with-Language-Model/
# https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306

from collections import Counter, deque
import editdistance # pyright: ignore[reportMissingModuleSource]
import re
from data import chars


alphabet = chars + ['-', '>']	# Blank character and end character
num_features = len(chars) + 1	# Also the non-character (when no key is pressed)
convergence_delay=6

# Init
prev_Pb, prev_Pnb = Counter(), Counter()
prev_Pb[''] = 1		# Probability of blank character
prev_Pnb[''] = 0		# Probability of non-blank character
previous_prefixes = deque(maxlen=convergence_delay)
previous_prefixes.append([''])


def incremental_prefix_beam_search(
	ctc,
	lm=None,
	*,
	k=100,
	alpha=0.30,
	beta=5,		# TODO: Tune this
	prune=0.001,
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

	pruned_alphabet = [alphabet[i] for i in ctc if i > prune]
	Pb, Pnb = Counter(), Counter()
	blank_prob = ctc[-1]

	for prefix in previous_prefixes[-1]:
		
		# Once you hit the end character stick with that
		if prefix[-1] == '>' and len(prefix) > 0:
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
				if c == prefix[-1] and len(prefix) > 0:
					Pnb[extended] += char_prob * prev_Pb[prefix]
					Pnb[prefix] += char_prob * prev_Pnb[prefix]

				# Extending with space/end character - triggers LM likelihood check
				elif c in (' ', '>') and len(prefix.replace(' ', '')) > 0:
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


	current_prefixes = Pb + Pnb
	word_count = lambda l: re.findall(r'\w+[\s>]', l)
	scorer = lambda l: current_prefixes[l] * (len(word_count(l)) + 1) ** beta
		
	# Force convergence
	if len(previous_prefixes < convergence_delay):
		# Select most probable prefixes
		current_prefixes = sorted(current_prefixes, key=scorer, reverse=True)
		previous_prefixes.append(current_prefixes[:k])
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
	prev_Pb, prev_Pnb = Counter(), Counter()
	for k, v in Pb.items():
		if k.startswith(converged_prefix):
			prev_Pb[k] = v
	for k, v in Pnb.items():
		if k.startswith(converged_prefix):
			prev_Pnb[k] = v
	current_prefixes = list(filter(lambda l: l.startswith(converged_prefix), current_prefixes))

	current_prefixes = sorted(current_prefixes, key=scorer, reverse=True)
	previous_prefixes.append(current_prefixes[:k])

	return previous_prefixes[0][0].strip('>')

	
