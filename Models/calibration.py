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
	thumb = frames[..., 0, :]
	index = frames[..., 1, :]
	little = frames[..., -1, :]
	return np.linalg.norm(thumb - index, axis=-1) + np.linalg.norm(thumb - little, axis=-1)

def _find_reference(frame_sequences):
	"""The reference point is when the fingers are spread out, resting on the surface."""
	flattened = frame_sequences.flatten()
	scores = _finger_spread(flattened)
	count = min(100, len(flattened))
	reference_frames = flattened[np.argpartition(scores, -count)[-count:]]
	return np.mean(reference_frames, axis=0), reference_frames

def _ideal_pipeline(frame_sequences: np.ndarray):
	"""Return the frames belonging to the final pinch pulse.

	The calibration motion is expected to start from a relatively open hand, then go into a shallow
	squeeze, then relax again, then run fingers over the surface in a pinch the starting posture.
	The useful region for the second pulse is the interval between the second local maximum and the
	subsequent local minimum. """
	
	surface_contour_frames = []
	squeeze_distances = [None * len(frame_sequences)]
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

	return surface_contour_frames, sum(squeeze_distances) / successful


def _only_squeeze(frame_sequences):
	squeeze_distances = [None * len(frame_sequences)]
	successful = 0

	for idx, sequence in enumerate(frame_sequences):

		spread = _finger_spread(sequence)
		if spread.size < 270:	# At least 1.5 seconds
			continue

		# It doesn't matter if they wave their hands around, because this only considers the finger
		# spread, and that's at its extremes when squeezing and stretching
		squeeze_distances[idx] = np.max(spread) - np.min(spread)

	return sum(squeeze_distances) / successful


def _fit_floor(frames, reference_hand_position):
	"""Fit a surface to the lower envelope of hand points near the reference pose."""

	if reference_hand_position is not None:
		near_reference = np.linalg.norm(frames - reference_hand_position, axis=-1) <= 0.20
		if near_reference.sum() >= 3:
			frames = frames[near_reference]
		else:
			raise RuntimeError("No points close enough to reference to find a floor")

	# A local XY neighbourhood avoids selecting only the globally lowest part of an inclined or curved surface.
	xy_distances = np.linalg.norm(frames[:, None, :2] - frames[None, :, :2], axis=-1)
	local_floor = np.min(
		np.where(xy_distances <= 0.02, frames[None, :, 2], np.inf), axis=-1)
	floor_points = frames[frames[:, 2] <= local_floor + 0.2]
	if len(floor_points) < 3:
		floor_points = frames[np.argsort(frames[:, 2])[:3]]

	return fit_surface(floor_points)

def _rest_values(reference_hand, shape):
	# midpoint between fingers and wrist
	avg_finger = np.mean(reference_hand[2:])
	reference_point = np.mean((avg_finger, reference_hand[0]))

	rest_point = reference_point + vector_to_surface(reference_point, shape)
	rest_uv = surface_uv(rest_point, shape)

	return rest_point, rest_uv

def calibrate_hand(calibration_sequences: np.ndarray, surface: str) -> Hand:
	calibration_sequences = calibration_sequences[:, :, 2:, :]	# Drop wrist points

	reference_hand_position, reference_frames = _find_reference(calibration_sequences)
	surface_contour_frames, squeeze_amount = _ideal_pipeline(calibration_sequences)

	if surface_contour_frames:

		shape_fit = _fit_floor(surface_contour_frames, None)
		rest_point, rest_uv = _rest_values(reference_hand_position, shape_fit)

		return Hand(shape_fit, squeeze_amount, rest_point, rest_uv)


	# Participants messed up in consistent ways
	# On the cushions they would only squeeze (I don't recall that happening when typing on legs)
	# And of course they can't squeeze hard surfaces
	
	elif surface == "cushion":

		squeeze_amount = _only_squeeze(calibration_sequences, reference_hand_position)
		shape_fit = _fit_floor(reference_frames)
		rest_point, rest_uv = _rest_values(reference_hand_position, shape_fit)

		return Hand(shape_fit, squeeze_amount, rest_point, rest_uv)

		# IDEA: You could also use the basic motion model (or similar) to trigger the pose model
		# unfortunately I messed up the data gathering and can't test that

	else:
		shape_fit = _fit_floor(calibration_sequences, reference_hand_position)
		rest_point, rest_uv = _rest_values(reference_hand_position, shape_fit)
			
		return Hand(shape_fit, 0, rest_point, rest_uv)
