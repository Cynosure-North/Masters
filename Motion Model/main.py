from LLM import model as llm
from TNN import model as tnn

# Run at 60fps
# If a new keypress has come in from the TNN update the next char
# Else use the off-frames to run the llm in batches (of 4?) to manage performance

# controls the balance between the LLM and the TNN, lower is biased towards the LLM
alpha =  0.5
# LLM compensation factor (usually 4-14)
beta = 10


# For all of our results, we use B = 100 beams.

# https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306



# For phrase in data
#	run TNN on it, get the predicted presses for each frame
#	Take that, run it through beam search
#	profit?

# TODO: Interactive mode
# In this interactive setting we constrained our beam search decoder to force convergence for any
# predictions older than 6 frames (0.1s) causing all beams to have a common prefix. We only rendered
# text in the common prefix of all beams, effectively imposing a fixed 0.1s delay.

# TODO: Measure size/performance of my model
# Our motion model network is compact, containing just 180KB of weights, enabling efficient
# evaluation on a GPU. Our language model consists of 19MB of weights. We are able to run both
# models, in addition to decoding, at interactive rates (i.e. 120 Hz) on a PC with an Nvidia RTX
# 2080Ti graphics card. These models were not optimized for compute and can be further accelerated
# using modern distillation and quantization techniques for neural networks

# TODO: Evaluate
# After decoding, we compute the uncorrected error rate UER(W,Wˆ ) as the Levenstein edit distance
# between the decoded and prompted strings divided by the number of characters of the longer of the
# two strings

# TODO: Figure out how to extend this to other surfaces