from llama_cpp import Llama

# Load the GGUF model
llm = Llama(
	model_path="Downloaded/gemma-4-e2b-q4_k_m.gguf",
	n_ctx=2048,
	verbose=False,
)

inpt = ""
while True:
	inpt = input("What say you? ")
	if inpt == "exit": break
	output = llm(inpt, max_tokens=128)
	print("\n" + output["choices"][0]["text"] + "\n")

# gemma-4-e2b-q4_k_m is slow, but also not partcularly accurate :/