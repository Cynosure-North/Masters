import editdistance # pyright: ignore[reportMissingModuleSource]

from model import model
from data import test_data

num_incorrect = 0
total_distance = 0
incorrect = []

# TODO: Evaluate
for phrase in test_data:
	# TODO: Assign finger first - use contact detection
	pred = model(test_data)
	levenshtein = editdistance.eval(pred, test_data[phrase])

	if not pred == test_data[phrase]: num_incorrect += 1
	total_distance += levenshtein
	if levenshtein == 0:
		print(f"CORRECT - {pred}")
	else:
		print(f"{levenshtein} - prediction: {pred}, correct answer {test_data[phrase]}")
		incorrect.append((pred, test_data[phrase]))

print(f"\n\nnum_incorrect: {num_incorrect}\ntotal_distance: {total_distance}\nincorrect: {'\n'.join(incorrect)}")

# TODO: Interactive mode