"""Extract a hand's resting position and squeeze amount from calibration data."""

import numpy as np
from dataclasses import dataclass
from fit_shape import fit_surface, surface_uv, vector_to_surface, SurfaceFit

@dataclass(frozen=True)
class Hand:
	# NOTE: Even for plane fits, it's still good to use the plane SDF rather than just the 2d
	# implementation, so it can handle angled planes. So the pipeline is the same regardless of
	# whether it's 2d or 3d
	shape: SurfaceFit
	compressibility: float
	rest_point: np.ndarray	# Average hand position in 3d space during calibration (where on the surface to put the keyboard)
	rest_uv: np.ndarray	# Average hand position on the surface surface during calibration


def _finger_spread(frames):
	"""Return a spread score for each frame."""
	wrist = frames[..., 2, :]
	thumb = frames[..., 2, :]
	index = frames[..., 1, :]
	little = frames[..., -1, :]
	return np.linalg.norm(thumb - index, axis=-1) + np.linalg.norm(wrist - little, axis=-1)

def _find_reference(frame_sequences):
	"""The reference point is when the fingers are spread out, resting on the surface."""
	flattened = np.concatenate(frame_sequences)
	scores = _finger_spread(flattened)
	count = min(100, len(flattened))
	reference_frames = flattened[np.argpartition(scores, -count)[-count:]]
	return np.mean(reference_frames, axis=0), reference_frames

def _ideal_pipeline(frame_sequences: list[np.ndarray]):
	"""Return the frames belonging to the final pinch pulse.

	The calibration motion is expected to start from a relatively open hand, then go into a shallow
	squeeze, then relax again, then run fingers over the surface in a pinch the starting posture.
	The useful region for the second pulse is the interval between the second local maximum and the
	subsequent local minimum. """
	
	surface_contour_frames = []
	squeeze_distances = [None for _ in range(len(frame_sequences))]
	successful = 0

	for idx, sequence in enumerate(frame_sequences):

		spread = _finger_spread(sequence)
		if spread.size < 270:	# At least 1.5 seconds
			continue

		# Smooth the signal to suppress tiny hand jitters that would otherwise be
		# mistaken for local extrema.
		kernel = np.ones(25, dtype=float) / 25
		smoothed = np.convolve(spread, kernel, mode="same")


		# Look for the canonical pattern: high -> low -> high -> lower low.
		maxima = np.where((smoothed[1:-1] >= smoothed[:-2]) & (smoothed[1:-1] >= smoothed[2:]))[0] + 1
		minima = np.where((smoothed[1:-1] <= smoothed[:-2]) & (smoothed[1:-1] <= smoothed[2:]))[0] + 1
		if maxima.size != 2 or minima.size != 2:
			continue

		# indexes of these events
		first_relax, second_relax = maxima[0], maxima[1]
		squeeze, pinch = minima[0], minima[1]
		
		# Enforce ordering
		if not (first_relax < squeeze and squeeze < second_relax):
			continue
		if not (second_relax < pinch):
			continue

		# Enforce finger distance ratios
		if not (smoothed[first_relax] > smoothed[squeeze] and smoothed[second_relax] > smoothed[pinch]):
			continue
		if not (smoothed[squeeze] > smoothed[pinch]):
			continue

		# Ensure at least 1/3 a second between extremes
		if first_relax - squeeze < 20 or second_relax - pinch < 20:
			continue

		surface_contour_frames.append(sequence[second_relax:pinch])
		squeeze_distances[idx] = spread[first_relax] - spread[squeeze]
		successful += 1

	# If 80% are bad I doubt these calibrations can be trusted
	# Although maybe this pipeline is better than the alternative
	if successful < 3:
		return None, None

	return np.concatenate(surface_contour_frames), sum(squeeze_distances) / successful


def _only_squeeze(frame_sequences):
	squeeze_distances = [None for _ in range(len(frame_sequences))]
	successful = 0

	for idx, sequence in enumerate(frame_sequences):

		spread = _finger_spread(sequence)
		if spread.size < 270:	# At least 1.5 seconds
			continue

		# It doesn't matter if they wave their hands around, because this only considers the finger
		# spread, and that's at its extremes when squeezing and stretching
		squeeze_distances[idx] = np.max(spread) - np.min(spread)
		successful += 1

	return sum(squeeze_distances) / successful


def _fit_floor(frames: np.ndarray, reference_hand_position: np.ndarray):
	"""Fit a surface to the lower envelope of hand points near the reference pose."""

	if reference_hand_position is not None:
		near_reference = np.linalg.norm(frames - reference_hand_position, axis=-1) <= 0.20
		if near_reference.sum() >= 3:
			frames = frames[near_reference]
		else:
			raise RuntimeError("No points close enough to reference to find a floor")

	frames = frames.reshape(-1, 3)

	# A local XY neighbourhood avoids selecting only the globally lowest part of an inclined or curved surface.
	# Spatial buckets find the neighbours without allocating an O(n²) distance matrix.
	neighbourhood_radius = 0.02
	bucket_coordinates = np.floor(frames[:, :2] / neighbourhood_radius).astype(np.int64)
	buckets: dict[tuple[int, int], list[int]] = {}
	for index, coordinate in enumerate(bucket_coordinates):
		buckets.setdefault((int(coordinate[0]), int(coordinate[1])), []).append(index)

	local_floor = np.empty(len(frames), dtype=frames.dtype)
	for (bucket_x, bucket_y), bucket_indices in buckets.items():
		candidate_indices = []
		for offset_x in (-1, 0, 1):
			for offset_y in (-1, 0, 1):
				candidate_indices.extend(buckets.get((bucket_x + offset_x, bucket_y + offset_y), ()))

		targets = np.asarray(bucket_indices, dtype=np.intp)
		candidates = np.asarray(candidate_indices, dtype=np.intp)
		candidate_xy = frames[candidates, :2]
		for start in range(0, len(targets), 1024):
			targets_chunk = targets[start:start + 1024]
			xy_deltas = candidate_xy[None, :, :] - frames[targets_chunk, None, :2]
			squared_distances = np.sum(xy_deltas * xy_deltas, axis=-1)
			local_floor[targets_chunk] = np.min(
				np.broadcast_to(frames[candidates, 2], squared_distances.shape),
				axis=1,
				where=squared_distances <= neighbourhood_radius**2,
				initial=np.inf )

	floor_points = frames[frames[:, 2] <= local_floor + 0.2]

	if len(floor_points) < 3:
		floor_points = frames[np.argsort(frames[:, 2])[:3]]

	return fit_surface(floor_points)

def _rest_values(reference_hand, shape):
	# midpoint between fingers and wrist
	avg_finger = np.mean(reference_hand[2:], axis=0)
	reference_point = np.mean((avg_finger, reference_hand[0]), axis=0)

	rest_point = reference_point + vector_to_surface(reference_point, shape)
	rest_uv = surface_uv(rest_point, shape)

	return rest_point, rest_uv

def calibrate_hand(calibration_sequences: list[np.ndarray], surface: str) -> Hand:
	reference_hand_position, reference_frames = _find_reference(calibration_sequences)
	calibration_sequences = [sequence[2:] for sequence in calibration_sequences] # Drop wrist points
	surface_contour_frames, squeeze_amount = _ideal_pipeline(calibration_sequences)

	if surface_contour_frames is not None:

		shape_fit = _fit_floor(surface_contour_frames, None)
		rest_point, rest_uv = _rest_values(reference_hand_position, shape_fit)

		return Hand(shape_fit, squeeze_amount, rest_point, rest_uv)


	# Participants messed up in consistent ways
	# On the cushions they would only squeeze (I don't recall that happening when typing on legs)
	# And of course they can't squeeze hard surfaces
	#
	# So if they're on a cushion the best we can do is try and figure out the rough shape from the
	# frames we know they weren't squeezing on
	# And if they're on a hard surface, the fingertips are going to be on/above the surface every frame
	
	elif surface == "cushion":

		squeeze_amount = _only_squeeze(calibration_sequences)
		shape_fit = _fit_floor(reference_frames[:, 2:], None)
		rest_point, rest_uv = _rest_values(reference_hand_position, shape_fit)

		return Hand(shape_fit, squeeze_amount, rest_point, rest_uv)

		# IDEA: You could also use the basic motion model (or similar) to trigger the pose model
		# unfortunately I messed up the data gathering and can't test that

	else:
		shape_fit = _fit_floor(np.concatenate(calibration_sequences), reference_hand_position)
		rest_point, rest_uv = _rest_values(reference_hand_position, shape_fit)
			
		return Hand(shape_fit, 0, rest_point, rest_uv)
