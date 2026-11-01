import numpy as np
from fit_shape import (
	fit_surface,
	signed_distance_to_surface,
	surface_uv,
	surface_normal,
	principal_curvature_frame,
)


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
	# TODO: This is going to need a lot of work, Some participants are just squeezing for the cushion, and on the pipe and table it's much closer to sliding.
	# Perhaps a factor could be if there's a clear peak in compression rather than a continuous increase in distance from the reference point
	
finger_idxs = {
	"l_little": 2,
	"l_ring": 3,
	"l_middle": 4,
	"l_index": 4,
	"l_thumb": 5,
	"r_thumb": 8,
	"r_index": 9,
	"r_middle": 10,
	"r_ring": 11,
	"r_little": 12}

def _unwrap_adjustment(self, frame, finger, compressibility):
	""" return [uv, normal, principal curvatures]"""
	point = frame[finger_idxs[finger]]
	u, v = surface_uv(point, surface)
	normal, tangent_u, tangent_v, curvature_0, curvature_1 = principal_curvatures_frame(point, surface)
	return np.concatenate((u, v, normal, tangent_u, tangent_v, curvature_0, curvature_1, compressibility))

def _SDF_adjustment(frame, left_hand, right_hand):
	""" for each finger, in each frame report the displacement from that finger to the surface of the corresponding model """
	frame = np.asarray(frame, dtype=np.float32).reshape(54, 3)
	adjustments = np.zeros((54, 3), dtype=np.float32)
	for idx, point in enumerate(frame):
		hand = left_hand if idx < 27 else right_hand
		adjustments[idx] = [signed_distance_to_surface(point, hand.shape), hand.compressibility]
	return np.concatenate((frame,adjustments), axis=-1)
	
class Surface():
	def __init__(self, shape, compressibility):
		# CONCERN: I'm unsure how the motion model will work with a split setup
		# NOTE: Even for plane fits, it's still good to use the plane SDF rather than just the 2d
		# implementation, so I can handle angled planes. So the pipeline is the same regardless of
		# whether it's 2d or 3d
		self.shape = shape	# 2d = isinstance(surface, PlaneFit)
		self.compressibility = compressibility

class ContactModeller():
	
	def __init__(self, frames):
		frames = np.asarray(frames, dtype=np.float32).reshape(-1, 54, 3)
		
		# (5 fingertips + 4 wrist markers) * xyz
		self.left_hand, self.right_hand = Surface(*calibration(frames[:27, :])), Surface(*calibration(frames[27:, :]))

	
	def motion_transform(franes):
		func = lambda l: _SDF_adjustment(l, self.left_hand, self.right_hand)
		return np.vectorize(func)(self.frames)	
	
	def contact_transform(self):
		contacting = _first_touching_frames(self.frames)
		result = []

		for idx, val in enumerate(contacting):
			if val == []:
				continue
			elif len(val) > 1:
				raise ValueError("Only one contact point per frame")
			else:
				hand = self.left_hand if val[0][0] = 'l' else self.right_hand
				res.append(self._unwrap_adjustment(self.frames[idx], val[0], hand.compressibility))
			.
		return result

		
	def _first_touching_frames(self):
		""" Identify the first frame of each finger press """
		# CONCERN: I may need debouncing

		contacting = np.asarray([[]] * len(frames), dtype=object)
		frames = np.asarray(frames).reshape(-1, 54, 3)
		for frame_idx, frame in enumerate(frames):
				contacting[frame_idx] self.contacting_fingers(frame)
			for finger, finger_idx in self.finger_idxs.items():
				point = frame[finger_idx]

				hand = self.left_hand if idx < 8 else self.right_hand
				distance_vector = signed_distance_to_surface(point, hand.shape)
				if np.linalg.norm(distance_vector) <= 1e-6:
					contacting[frame_idx].append(finger)

		for i in range(len(frames)-1, 1, -1):
			for val in contacting[i].copy():
				if val in contacting[i-1]:
					contacting[i].remove(val)
					
		# If multiple keys are first pressed on the same frame, displace one
		# Yes it could lead to some typos but I doubt it would be that common
		for i in range(1, len(frames)-1):
			if len(contacting[i] > 1):
				if len(contacting[i-1]) == 0:
					contacting[i-1].append(contacting[i].pop())
				elif len(contacting[i+1]) == 0:
					contacting[i-1].append(contacting[i].pop())
				else:
					raise Error("Can't displace keypress")

		return contacting

		# IDEA: I should try averaging the output of the pose and contact models (another model on top may better but gaf)
	
		# These don't give the model the surface topology, could be good future work




currently_contacting = []
displaced_keys = []

class InteractiveContactModeller():
	def contact_transform(self, frame):

		contacting = []

		for finger, (finger_idx, hand) in self.finger_idxs.items():
			point = frame[finger_idx]
			hand = self.left_hand if idx < 8 else self.right_hand
			distance_vector = signed_distance_to_surface(point, hand.shape)
			if np.linalg.norm(distance_vector) <= 1e-6:
				contacting.append(finger)
			elif finger in currently_contacting:
				currently_contacting.remove(finger)

		if displaced_keys:
			if contacting:
				displaced_keys.append(contacting)
			return _unwrap_adjustment(frame, displaced_keys.pop(0))
		else:
			if len(contacting) > 1:
				displaced_keys.append(contacting[1:])
			if contacting
				return _unwrap_adjustment(frame, contacting[0])
		return None
	
	def motion_transform(self, frame):
		_SDF_adjustment(frame, self.left_hand, self.right_hand)