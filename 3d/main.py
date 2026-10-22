import numpy as np
from fit_shape import fit_surface, signed_distance_to_surface


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
		# CONCERN: I'm unsure how the motion model will work with a split setup
		# CONCERN: I'm unsure how well it will work if the keyboard is in a traditional layout rather than split
		self.shape = shape	# 2d = isinstance(surface, PlaneFit)
		self.compressibility = compressibility
		
	
class ContactModeller():
	
	def __init__(self, frames):
		frames = np.asarray(frames, dtype=np.float32).reshape(54, 3)
		
		# (5 fingertips + 4 wrist markers) * xyz
		self.left_hand, self.right_hand = Surface(*calibration(frames[:27, :])), Surface(*calibration(frames[27:, :]))

		self.finger_idxs = {
			"l_little": (0, self.left_hand),
			"l_ring": (2, self.left_hand),
			"l_middle": (3, self.left_hand),
			"l_index": (3, self.left_hand),
			"l_thumb": (4, self.left_hand),
			"r_thumb": (9, self.right_hand),
			"r_index": (10, self.right_hand),
			"r_middle": (11, self.right_hand),
			"r_ring": (12, self.right_hand),
			"r_little": (13,  self.right_hand) }
		
	def calculate_touch_points(self, frames):
		""" Identify the first frame of each finger press """
		# CONCERN: I may need debouncing

		contacting = np.asarray([[]] * (len(frames) + 1))
		frames = np.asarray(frames).reshape(-1, 54, 3)
		for frame_idx, frame in enumerate(frames):
			for finger, (finger_idx, hand) in self.finger_idxs:
					if signed_distance_to_surface(frame[finger_idx], hand.surface) < 0:
						if not contacting[frame_idx].contains(finger):
							contacting[frame_idx+1].append(finger)

		return contacting[1:]

		# IDEA: I should try averaging the output of the pose and contact models (another model on top may better but gaf)
	
		# TODO: 3d conversion
		# Note that these don't give the model the surface topology, could be good future work
		
		# For contact and pose models, unwrap surface, and also report normal + curvature of surface at that point
		
		# For motion model, for each finger, in each frame report the (delta of the) signed distance from that finger to the surface of the corresponding model
		
		# Even for plane fits, it's still good to use the plane SDF, rather than just the 2d implementation so I can handle angled planes
		
		def SDF_adjustment(frame):


class InteractiveContactModeller():
	pass
	# TODO: interactive contact detection