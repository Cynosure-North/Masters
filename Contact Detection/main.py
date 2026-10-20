import numpy as np
from fit_shape import fit_surface

# TODO: Report which hand touched
# TODO: I'll want to store the point of last contact to ensure the keys appear in the right place
# TODO: Figure out how to integrate this with the AI
# 	- Convert to UVs?
#		- But that would loose orientation and topology
#	- Measure which shapes fit the most and use those?
#	- Not sure how the motion model will handle different surfaces for each hand
#	Whatever I do, it probably makes sense to have a preprocessing layer so I don't have to train a bunch of different models


def calibration(frames):
	def compressibility(reference_points, subsequent_frames):
		averages = np.mean(subsequent_frames-reference_points, axis=-1)
		max_compression_index = np.argmax(averages)
		max_compression = averages[max_compression_index]
		end_of_compression = np.argmin(averages[max_compression_index:])
		
		if max_compression < 0.2:
			return 0, 20	# even on hard surfaces fingers have some squish of their own
		else:
			return max_compression, end_of_compression
	
	frames = frames[:, :5, :]		# discard wrist + knuckles

	reference = np.mean(frames[:60])

	compressibility, end_index = compressibility(reference, frames[70:190])

	shape_fit = fit_surface(frames[70+end_index+10:])
	
	return shape_fit, compressibility

class Surface():
	def __init__(self, shape, compressibility):
		self.shape = shape	# 2d = isinstance(surface, PlaneFit)
		self.compressibility = compressibility
	
class ContactModeller():
	
	def __init__(self, frames):
		frames = np.asarray(frames, dtype=np.float32).reshape(54, 3)
		
		# (5 fingertips + 4 wrist markers) * xyz
		self.left_hand, self.right_hand = Surface(*calibration(frames[:27, :])), Surface(*calibration(frames[27:, :]))
		
	def calculate_touch_points(frames):
		pass
		# TODO: Determine single unique contact moment
		# 	- probably the first frame? but I could give a few frames as a buffer to discard erroneous keystrokes
		# 	- if frames are negative skip them
		# 	- Determine which finger contacted, multiple fingers could be in contact at any one time
		# 	Not sure if its the best place to do it here rather than in the interactive mode functions



class InteractiveContactModeller():
	pass
	# TODO: interactive contact detection