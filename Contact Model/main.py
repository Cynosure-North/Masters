# https://www.ijcai.org/proceedings/2021/0242.pdf

# We propose a novel baseline approach for the IMK decod-
# ing task. Specifically, taking a sequence of coordinate val-
# ues as input, we aim to decode a phrase which users in-
# tended to type on an invisible layout. Let ˆC = { ˆc1, ..., ˆcn}
# denote a predicted character sequence by the decoding algo-
# rithm. The decoder aims to find the character sequence, ˆC,
# with the highest conditional probability given a typing input
# $$ T = \{t_1, t_2, ..., t_n\} $$, as follows (equation 2)

# $$ \mathbb{\hat{C}} = argmax(P(\hat{c_1}, ..., \hat{c_n}|t_1, ..., t_n)) $$

# Our baseline approach, Self-Attention Neural Character Decoder (SA-NCD), consists of two decoder
# modules: 1) a geometric decoder (G) with its parameters φG and 2) a semantic decoder (S) with its
# parameters φS . For an input sequence $$ X = \{x_1, x_2, ..., x_n\} $$, let $$ G(X; \phi G) $$ and $$ S(X; \phi S) $$ denote the
# output from the geometric decoder and the semantic decoder, respectively. The geometric decoder
# takes in a touch input sequence and converts it into a character sequence by using the touch
# locations in the invisible keyboard layout. Then, the semantic decoder corrects decoding errors in
# the character sequence estimated by the geometric decoder while considering semantic meanings. Here,
# the confidence masking process determines the input of the semantic decoder by finding locations of
# the errors produced by the geometric decoder. The errors could be simply a case of incorrect
# predictions by the geometric decoder, or a case of absolute typos made by a user. One thing to note
# is that SA-NCD decodes the entire sequence input up to that point (not merely the most recent touch
# input). Therefore, the characters decoded in the previous steps are not fixed but can be changed by
# reflecting the semantic dependence.

# Geometric Decoder
# We use Bidirectional GRU (BiGRU) for the geometric decoder. To overcome the vanishing gradient
# problem, we adopt GRU as a recurrent neural network. In our task, even if a user enters the same
# character index, the input (the touch points) is different every time. Since the inputs are not a
# definite character index, the recurrent network suffers from the instability accumulation if it only
# proceeds in uni-direction. Thus, we take the advantage of forward and backward passes of Bi-GRU to
# overcome this issue.

# Confidence Masking
# The length of $$ G(X; \phi_G) = \{o_1, o_2, ..., o_n\} $$ is the same as the input length, and each element has the
# dimension size of the vocabulary size (v). When considering the i-th input $$ (1 \le i \le n) $$ from a
# sequence, the probability (confidence) of the k-th character in the vocabulary, pk i , can be
# calculated by the softmax function as follows:
# $$ \text{Softmax}(G(X; \phi_G)) = \{(p_i^1 , p_i^2, ..., p_i^v)\}_{i=1}^n $$
# with

# $$ p_i^k = \frac{exp(o_i^k)}{\sigma_{j=1}^v exp(o_i^j)} $$

# where $$ o_j $$ is the value of j-th element of $$ o_i $$ in $$ $$$$ G(X; \phi G) $$ and v is the vocabulary size.
# At this step, the touch input detected at an ambiguous location that is difficult to decode with
# only geometric information results in low confidence. We replace the positions with a confidence
# lower than $$ \tau $$ with mask tokens to provide the masked character sequence as input to the semantic
# decoder (confidence masking process). The following CM and M ask refer to the confidence masking
# process and the individual masking function as follows:

# $$ CM ({x_1, ..., x_n}) = {Mask(x_1), ..., Mask(x_n)} $$
# with
# $$ Mask(x_i) = \begin{cases} Embed(argmax p_i^j) & \text{if } max(p_j^i) \ge \tau \\ 
# 							Embed(j_{[mask]}) & \text{otherwise} \end{cases} $$

# where $$ x_i = (p_i^1, p_i^2, ..., p_i^v) $$ and $$ j_{[mask]} $$ is the index of mask token. $$ \text{Embed} $$ is an embedding layer which
# encodes index to an embedded vector.

# Semantic Decoder
# We use the Transformer encoder architecture as a semantic decoder. The embedded hidden state that
# has passed the geometric decoder followed by confidence masking is processed by the semantic decoder
# with the self-attention mechanism of Transformer. Here, the semantic decoder acts as a character
# language model, replacing the masked characters with appropriate characters. Through the proposed
# network architecture, the decoded character sequence $$ \mathbb{\hat{C}} $$ in Equation (2) can be expressed as fol-
# lows: $$ \mathbb{\hat{C}} = \text{Softmax}(G(X; \phi_G))); \phi_S $$

# Training Scheme
# We pre-train both the geometric decoder and the semantic decoder to enhance their respective roles.
# In particular, we pre-process Billion Word Benchmark to train the semantic decoder as a masked
# character language model by following the same pre-training scheme of BERT Then, we fine-tune the
# two decoder modules together by first freezing $$ \phi G $$ and only training $$ S(X; \phi S ) $$, and then training
# $$ G(X; \phi G) $$ while freezing $$ \phi S $$ repeatedly.

# Implementation Details
# We optimized φG with SGD update, with the learning rate of 3.0 and gradient clipping at 0.5. For $$ \phi S $$,
# we used the Adam optimizer with learning rate 1e-4. Threshold $$ \tau $$ for confidence masking was 0.45 and
# the vocabulary size v was 31. We minimized Cross-entropy loss to optimize the parameters. In
# addition, to quickly propagate information within the network, we employed the auxiliary losses for
# the character language model. Moreover, we altered the position of layer normalization prior to
# self-attention to obtain the stability of the gradient and induce rapid convergence. We implemented
# We augmented our data by randomly shifting the x and y positions of each touch point by 3 or fewer
# pixels with a 50% probability. The validation accuracy was measured by using the validation set at
# every epoch, and if the maximum accuracy was not updated for more than 3 epochs, the training was
# terminated by early stopping.