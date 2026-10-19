# TODO: Determine single unique contact moment
# 	- probably the first frame? but I could give a few frames as a buffer to discard erroneous keystrokes
# 	- track contact events so future frames are ignored
# 	- Determine which finger contacted
# 	Not sure if its the best place to do it here rather than in the interactive mode functions
# TODO: Report which hand touched
# TODO: I'll want to store the point of last contact to ensure the keys appear in the right place
# TODO: I'll want to fit different surfaces for each hand
#	Not sure how that will work with the motion model
# TODO: Figure out how to integrate this with the AI
# 	- Convert to UVs?
#		- But that would loose orientation and topology
#	- Measure which shapes fit the most and use those?
#	Whatever I do, it probably makes sense to have a preprocessing layer so I don't have to train a bunch of different models