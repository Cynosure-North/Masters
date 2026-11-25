import json
import editdistance

# TODO: Write parsing code now
#  I recall they standardised on a word being 5 chars + space
#  Should wpm be based on the text typed or on the target text?
#
#



def parse(file):
	data = json.load(file)
	quote = data['quote']
	wpm = len(quote) / data['time_taken']

	typed_chars = [c for _, c in data['keyboard_log']]

	
	uer = editdistance.eval(quote, data['final_text'])
	cer = editdistance.eval(quote, typed_chars) 