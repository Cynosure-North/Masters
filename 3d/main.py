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
	
	def motion_transform(self):
		return np.vectorize(self._SDF_adjustment)(self.frames)
	
	def contact_transform(self):
		contacting = _first_touching_frames(self.frames)
		result = []

		for idx, val in enumerate(contacting):
			if val == []:
				continue
			elif len(val) > 1:
				raise ValueError("Only one contact point per frame")
			else:
				res.append(self._unwrap_adjustment(self.frames[idx], val[0]))
				
		return result

	def _SDF_adjustment(self, frame):
		""" for each finger, in each frame report the displacement from that finger to the surface of the corresponding model """
		frame = np.asarray(frame, dtype=np.float32).reshape(54, 3)
		adjustments = np.zeros((54, 3), dtype=np.float32)
		for idx, point in enumerate(frame):
			hand = self.left_hand if idx < 27 else self.right_hand
			adjustments[idx] = signed_distance_to_surface(point, hand.shape)
		return np.concatenate((frame,adjustments), axis=-1)

	def _unwrap_adjustment(self, frame, finger):
		""" return [uv, normal, principal curvatures]"""
		point = frame[self.finger_idxs[finger][0]]
		u, v = surface_uv(point, surface)
		normal = surface_normal(point, surface)
		normal, tangent_u, tangent_v, curvature_0, curvature_1 = principal_curvatures_frame(point, surface)
		return np.concatenate((u, v, normal, tangent_u, tangent_v, curvature_0, curvature_1))
		
	def _first_touching_frames(self):
		""" Identify the first frame of each finger press """
		# CONCERN: I may need debouncing

		contacting = np.asarray([[]] * len(frames), dtype=object)
		frames = np.asarray(frames).reshape(-1, 54, 3)
		for frame_idx, frame in enumerate(frames):
				contacting[frame_idx] self.contacting_fingers(frame)
			for finger, (finger_idx, hand) in self.finger_idxs.items():
				point = frame[finger_idx]
				distance_vector = signed_distance_to_surface(point, hand.shape)
				if np.linalg.norm(distance_vector) <= 1e-6:
					contacting[frame_idx].append(finger)

		for i in range(len(frames)-1, 1, -1):
			for val in contacting[i].copy():
				if val in contacting[i-1]:
					contacting[i].remove(val)

		return contacting

		# IDEA: I should try averaging the output of the pose and contact models (another model on top may better but gaf)
	
		# These don't give the model the surface topology, could be good future work
		
		
		
		

			


class InteractiveContactModeller():
	pass
	# TODO: interactive contact detection