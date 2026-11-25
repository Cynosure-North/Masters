import json
import editdistance
import throughput
from pathlib import Path

# M. R. Zhang, S. Zhai, and J. O. Wobbrock, “Text Entry Throughput: Towards Unifying Speed and Accuracy in a Single Performance Metric,” in Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, Glasgow Scotland Uk: ACM, May 2019, pp. 1–13. doi: 10.1145/3290605.3300866.

# I. S. MacKenzie and K. Tanaka-Ishii, Eds., “Text entry systems: mobility, accessibility, universality Chapter 3,” in Text entry systems: mobility, accessibility, universality, in The Morgan Kaufmann series in interactive technologies. , San Francisco: Morgan Kaufmann, 2007, pp. 47–74.

def parse(data_dir: Path):
	
	surface_values = {}

	for participant in data_dir.iterdir():
		for surface in participant.iterdir():
			trials = []
			for file in surface.glob("*.json"):
				data = json.load(file)
				quote = data['quote']
				typed = str([c for _, c in data['keyboard_log']])
				final_text = data['final_text']
				time_taken = data['time_taken']
				
				
				incorrect_not_fixed = editdistance.eval(quote, final_text)
				incorrect_fixed = typed.count('-')
				c = max(len(quote), len(final)) - incorrect_not_fixed

				wpm = (len(final_text) -1) / time_taken * 12
				uer = incorrect_not_fixed = (c + incorrect_fixed + incorrect_not_fixed)
				cer = incorrect_fixed = (c + incorrect_fixed + incorrect_not_fixed)
				
				trials.append(quote, final_text, time_taken, wpm, uer, cer)

			if surface.stem in surface_values.keys():
				surface_values[surface.stem]["throughput"].append(
					throughput.Throughput(trials).calThroughput())
				surface_values[surface.stem]["avg_wpm"].append(
						sum(x[3] for x in trials) / len(trials))
				surface_values[surface.stem]["avg_uer"].append(
						sum(x[4] for x in trials) / len(trials))
				surface_values[surface.stem]["avg_cer"].append(
						sum(x[5] for x in trials) / len(trials))

			else:
				surface_values[surface.stem] = {
					"throughput": [throughput.Throughput(trials).calThroughput()],
					"avg_wpm": [sum(x[3] for x in trials) / len(trials)],
					"avg_uer": [sum(x[4] for x in trials) / len(trials)],
					"avg_cer": [sum(x[5] for x in trials) / len(trials)] }
	
	final_surface_values = {}
	for surface in surface_values:
		final_surface_values[surface] = {
		"throughput": sum(surface_values[surface.stem]["throughput"]) / len(surface),
		"avg_wpm": sum(surface_values[surface.stem]["avg_wpm"]) / len(surface),
		"avg_uer": sum(surface_values[surface.stem]["avg_uer"]) / len(surface),
		"avg_cer": sum(surface_values[surface.stem]["avg_cer"]) / len(surface) }
		
	print(final_surface_values)