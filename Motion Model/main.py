from LLM import model as llm
from TNN import model as tnn
from BeamSearch import prefix_beam_search
from dataset import test_data

# Run at 60fps
# If a new keypress has come in from the TNN update the next char
# Else use the off-frames to run the llm in batches (of 4?) to manage performance

# controls the balance between the LLM and the TNN, lower is biased towards the LLM
alpha =  0.5
# LLM compensation factor (usually 4-14)
beta = 10
# number of branches to keep alive, they use 100
beam_width = 100

# https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306

def levenshtein_distance(s1: str, s2: str) -> int:
	# Ensure s1 is the shorter string to optimize memory
	if len(s1) > len(s2):
		s1, s2 = s2, s1
		
	# Initialize the base distance row (transforming from empty string)
	previous_row = list(range(len(s1) + 1))
	
	# Iterate through characters of the longer string
	for i, c2 in enumerate(s2):
		current_row = [i + 1]
		
		# Compare with each character of the shorter string
		for j, c1 in enumerate(s1):
			insertions = previous_row[j + 1] + 1
			deletions = current_row[j] + 1
			substitutions = previous_row[j] + (0 if c1 == c2 else 1)
			
			current_row.append(min(insertions, deletions, substitutions))
			
		previous_row = current_row
		
	return previous_row[-1]

num_incorrect = 0
total_distance = 0
incorrect = []

# TODO: Evaluate
for phrase in test_data:
	tnn_prediction = tnn(test_data)
	pred = prefix_beam_search(tnn_prediction, llm, beam_width, alpha, beta)
	levenshtein = levenshtein_distance(pred, test_data[phrase])

	if not pred == test_data[phrase]: num_incorrect += 1
	total_distance += levenshtein
	if levenshtein == 0:
		print(f"CORRECT - {pred}")
	else:
		print(f"{levenshtein} - prediction: {pred}, correct answer {test_data[phrase]}")
		incorrect.append((pred, test_data[phrase]))

print(f"\n\nnum_incorrect: {num_incorrect}\ntotal_distance: {total_distance}\nincorrect: {'\n'.join(incorrect)}")

# TODO: Interactive mode, I'll need to figure out how to collapse old prefixes
# In this interactive setting we constrained our beam search decoder to force convergence for any
# predictions older than 6 frames (0.1s) causing all beams to have a common prefix. We only rendered
# text in the common prefix of all beams, effectively imposing a fixed 0.1s delay.

# TODO: Measure size/performance of my model
# Our motion model network is compact, containing just 180KB of weights, enabling efficient
# evaluation on a GPU. Our language model consists of 19MB of weights. We are able to run both
# models, in addition to decoding, at interactive rates (i.e. 120 Hz) on a PC with an Nvidia RTX
# 2080Ti graphics card. These models were not optimized for compute and can be further accelerated
# using modern distillation and quantization techniques for neural networks

# TODO: Figure out how to extend this to other surfaces