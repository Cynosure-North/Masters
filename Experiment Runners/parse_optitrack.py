from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Minimal dataclasses that match the repr output in out.txt.
# The file is essentially a Python object repr, so we can safely evaluate it in
# a controlled namespace.
# ---------------------------------------------------------------------------

@dataclass
class MarkerSetDescription:
	name: str
	marker_names: Tuple[str, ...]


@dataclass
class RigidBodyMarkerDescription:
	name: str
	active_label: int
	pos: Tuple[float, float, float]


@dataclass
class RigidBodyDescription:
	name: str
	id_num: int
	parent_id: int
	pos: Tuple[float, float, float]
	quat: Any = None
	markers: Tuple[Any, ...] = ()


@dataclass
class SkeletonDescription:
	name: str
	id_num: int
	rigid_body_descriptions: Tuple[RigidBodyDescription, ...]


@dataclass
class DataDescriptions:
	marker_sets: Tuple[MarkerSetDescription, ...]
	rigid_bodies: Tuple[Any, ...] = ()
	skeletons: Tuple[SkeletonDescription, ...] = ()
	force_plates: Tuple[Any, ...] = ()
	devices: Tuple[Any, ...] = ()
	cameras: Tuple[Any, ...] = ()
	assets: Tuple[Any, ...] = ()


@dataclass
class MarkerSet:
	model_name: str
	marker_pos_list: Tuple[Tuple[float, float, float], ...]


@dataclass
class RigidBody:
	id_num: int
	pos: Tuple[float, float, float]
	rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
	markers: Any = None
	tracking_valid: bool = False
	marker_error: float = 0.0


@dataclass
class Skeleton:
	id_num: int
	rigid_bodies: Tuple[RigidBody, ...]


@dataclass
class LabeledMarker:
	id_num: int
	pos: Tuple[float, float, float]
	size: float
	param: int
	residual: float


@dataclass
class FramePrefix:
	frame_number: int


@dataclass
class FrameSuffix:
	timecode: int
	timecode_sub: int
	timestamp: float
	stamp_camera_mid_exposure: int
	stamp_data_received: int
	stamp_transmit: int
	param: int
	is_recording: bool
	tracked_models_changed: bool


@dataclass
class DataFrame:
	prefix: FramePrefix
	marker_sets: Tuple[MarkerSet, ...]
	unlabeled_marker_pos: Tuple[Tuple[float, float, float], ...]
	rigid_bodies: Tuple[Any, ...]
	skeletons: Tuple[Skeleton, ...]
	assets: Any = None
	labeled_markers: Tuple[LabeledMarker, ...] = ()
	force_plates: Tuple[Any, ...] = ()
	devices: Tuple[Any, ...] = ()
	suffix: FrameSuffix = None


# Some constructors in the file are used in the data descriptions. Keep them in the
# eval namespace even if they are not used directly in this script.
@dataclass
class DeviceDescription:
	id_num: int
	name: str
	serial_number: str
	device_type: int
	channel_data_type: int
	channel_names: Tuple[str, ...]


@dataclass
class Device:
	id_num: int
	channel_arrays: Tuple[Tuple[float, ...], ...]


@dataclass
class CameraDescription:
	name: str
	position: Tuple[float, float, float]
	orientation_quat: Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


def eval_capture(text: str):
	"""Evaluate the repr() dump from NatNet into Python objects."""
	namespace = {
		"DataDescriptions": DataDescriptions,
		"MarkerSetDescription": MarkerSetDescription,
		"RigidBodyMarkerDescription": RigidBodyMarkerDescription,
		"RigidBodyDescription": RigidBodyDescription,
		"SkeletonDescription": SkeletonDescription,
		"MarkerSet": MarkerSet,
		"RigidBody": RigidBody,
		"Skeleton": Skeleton,
		"LabeledMarker": LabeledMarker,
		"FramePrefix": FramePrefix,
		"FrameSuffix": FrameSuffix,
		"DataFrame": DataFrame,
		"DeviceDescription": DeviceDescription,
		"Device": Device,
		"CameraDescription": CameraDescription,
		"None": None,
	}
	obj = eval(text, {"__builtins__": {}}, namespace)
	return obj


def load_capture(path: str | Path):
	text = Path(path).read_text(encoding="utf-8")
	data = eval_capture(text.strip())
	if not isinstance(data, list) or len(data) < 2:
		raise ValueError("Expected a list whose first item is DataDescriptions and the rest are frames")
	description, *frames = data
	if not isinstance(description, DataDescriptions):
		raise TypeError("The first element of the capture list was not a DataDescriptions object")
	return description, frames


def extract_hand_markers(description: DataDescriptions, frame: DataFrame, hand_name: str):
	"""Map marker names to 3D positions for a single hand."""
	marker_desc = next(md for md in description.marker_sets if md.name == hand_name)
	marker_names = list(marker_desc.marker_names)
	marker_set = next(ms for ms in frame.marker_sets if ms.model_name == hand_name)
	pos_list = list(marker_set.marker_pos_list)
	if len(pos_list) != len(marker_names):
		raise ValueError(f"Marker count mismatch for {hand_name}: {len(pos_list)} vs {len(marker_names)}")
	return {name.rsplit("_", 1)[-1]: pos for name, pos in zip(marker_names, pos_list)}


def extract_wrist(description: DataDescriptions, frame: DataFrame, hand_name: str):
	markers = extract_hand_markers(description, frame, hand_name)
	wrist_out = markers.get("LWristOut") or markers.get("RWristOut")
	wrist_in = markers.get("LWristIn") or markers.get("RWristIn")
	if wrist_out is None or wrist_in is None:
		raise KeyError(f"Could not find wrist markers for {hand_name}")
	return {
		"wrist_out": wrist_out,
		"wrist_in": wrist_in,
		"wrist_center": tuple((x + y) / 2.0 for x, y in zip(wrist_out, wrist_in))	# midpoint
	}


def extract_wrist_rotation(description: DataDescriptions, frame: DataFrame, hand_name: str):
	skeleton_id = next(sd.id_num for sd in description.skeletons if sd.name == hand_name)
	skeleton = next(s for s in frame.skeletons if s.id_num == skeleton_id)
	# The root rigid body is the first rigid body in the skeleton; its rotation is in quaternion form.
	root_rb = skeleton.rigid_bodies[0]
	return {
		"root_rb_id": root_rb.id_num,
		"rotation_quaternion": root_rb.rot,
	}


def extract_finger_tips(description: DataDescriptions, frame: DataFrame, hand_name: str):
	# The live frame sometimes remaps rigid-body IDs; the stable way to match them is by ordering.
	desc_skeleton = next(sd for sd in description.skeletons if sd.name == hand_name)
	frame_skeleton = next(s for s in frame.skeletons if s.id_num == desc_skeleton.id_num)
	desc_names = [rb.name for rb in desc_skeleton.rigid_body_descriptions]
	rb_positions = frame_skeleton.rigid_bodies

	tips = {}
	finger_suffixes = ("Thumb3", "Index3", "Middle3", "Ring3", "Pinky3")
	for desc_name, rb in zip(desc_names, rb_positions):
		suffix = desc_name.rsplit("_", 1)[-1]
		if any(finger in suffix for finger in finger_suffixes):
			tips[suffix] = rb.pos
	return tips


def extract_one_frame(description: DataDescriptions, frame: DataFrame):
	result = {}
	for hand_name in [sd.name for sd in description.skeletons]:
		wrist = extract_wrist(description, frame, hand_name)
		rotation = extract_wrist_rotation(description, frame, hand_name)
		tips = extract_finger_tips(description, frame, hand_name)
		result[hand_name] = {
			"wrist_out": wrist["wrist_out"],
			"wrist_in": wrist["wrist_in"],
			"wrist_center": wrist["wrist_center"],
			"rotation_quaternion": rotation["rotation_quaternion"],
			"fingertips": tips,
		}
	return result


def finger_names_from_desc(description: DataDescriptions, hand_name: str):
	skeleton = next(sd for sd in description.skeletons if sd.name == hand_name)
	names = []
	for rb_desc in skeleton.rigid_body_descriptions:
		suffix = rb_desc.name.rsplit("_", 1)[-1]
		if suffix.endswith(("Thumb3", "Index3", "Middle3", "Ring3", "Pinky3")):
			names.append(suffix)
	return names


point_names = (
	"left_hand_LWristOut",
	"left_hand_LWristIn",
	"left_hand_LHandOut",
	"left_hand_LHandIn",
	"left_hand_Thumb3",
	"left_hand_Index3",
	"left_hand_Middle3",
	"left_hand_Ring3",
	"left_hand_Pinky3",
	"right_hand_RWristOut",
	"right_hand_RWristIn",
	"right_hand_RHandOut",
	"right_hand_RHandIn",
	"right_hand_Thumb3",
	"right_hand_Index3",
	"right_hand_Middle3",
	"right_hand_Ring3",
	"right_hand_Pinky3",
)


def frame_point_array(frame: DataFrame, description: DataDescriptions | None = None):
	"""Flatten a single frame into a (N, 3) NumPy array with a documented ordering.

	The canonical ordering of tracked points in the flattened frame array is:
	  0-3   left_hand wrist/hand markers: left_hand_LWristOut, left_hand_LWristIn, left_hand_LHandOut, left_hand_LHandIn
	  4-8   left_hand fingertips: LeftHandThumb3, LeftHandIndex3, LeftHandMiddle3, LeftHandRing3, LeftHandPinky3
	  9-12  right_hand wrist/hand markers: right_hand_RWristOut, right_hand_RWristIn, right_hand_RHandOut, right_hand_RHandIn
	  13-17 right_hand fingertips: RightHandThumb3, RightHandIndex3, RightHandMiddle3, RightHandRing3, RightHandPinky3
	"""
	if description is None:
		raise ValueError("A DataDescriptions object is required to resolve point names for a frame.")

	points = []

	for hand_name in ("left_hand", "right_hand"):
		try:
			hand_markers = extract_hand_markers(description, frame, hand_name)
			marker_order = [
				f"{hand_name}_LWristOut" if hand_name == "left_hand" else f"{hand_name}_RWristOut",
				f"{hand_name}_LWristIn" if hand_name == "left_hand" else f"{hand_name}_RWristIn",
				f"{hand_name}_LHandOut" if hand_name == "left_hand" else f"{hand_name}_RHandOut",
				f"{hand_name}_LHandIn" if hand_name == "left_hand" else f"{hand_name}_RHandIn",
			]
			for key in marker_order:
				if key in hand_markers:
					points.append(hand_markers[key])
				else:
					points.append(np.zeros(3, dtype=float))

			hand_tips = extract_finger_tips(description, frame, hand_name)
			for finger in ["Thumb3", "Index3", "Middle3", "Ring3", "Pinky3"]:
				candidate = next((name for name in hand_tips if name.endswith(finger)), None)
				if candidate is None:
					points.append(np.zeros(3, dtype=float))
				else:
					points.append(np.asarray(hand_tips[candidate], dtype=float))
		except Exception:
			continue

	arr = np.asarray(points, dtype=float)
	if arr.size == 0:
		return np.zeros((len(point_names), 3), dtype=float)
	if arr.shape[0] != len(point_names):
		# Some frames may not have every point available; pad the missing points with zeros.
		pad = np.zeros((len(point_names) - arr.shape[0], 3), dtype=float)
		arr = np.vstack((arr, pad))
	return arr


def _frame_has_valid_hand_tracking(frame: DataFrame, description: DataDescriptions, hand_name: str) -> bool:
	"""Return True only when the hand has enough valid marker positions"""
	try:
		marker_set = next(ms for ms in frame.marker_sets if ms.model_name == hand_name)
		valid_marker_count = sum(
			1
			for pos in marker_set.marker_pos_list
			if pos is not None and np.all(np.isfinite(np.asarray(pos, dtype=float)))
		)
		if valid_marker_count < 3:
			return False

		frame_data = extract_one_frame(description, frame)[hand_name]
		if frame_data is None:
			return False
		wrist_out = np.asarray(frame_data["wrist_out"], dtype=float)
		wrist_in = np.asarray(frame_data["wrist_in"], dtype=float)
		if not np.all(np.isfinite(wrist_out)) or not np.all(np.isfinite(wrist_in)):
			return False

		# If a skeleton exists and some rigid bodies explicitly report tracking validity,
		# only reject when the frame is actually missing enough tracked joints. If every
		# rigid body flag is False because the file omitted that metadata, do not discard
		# otherwise-valid wrist-marker data.
		skeleton_id = next((sd.id_num for sd in description.skeletons if sd.name == hand_name), None)
		if skeleton_id is not None:
			skeleton = next((s for s in frame.skeletons if s.id_num == skeleton_id), None)
			if skeleton is not None:
				tracked_rigid_bodies = [
					rb for rb in skeleton.rigid_bodies if getattr(rb, "tracking_valid", None) is True
				]
				if tracked_rigid_bodies and len(tracked_rigid_bodies) < 3:
					return False

		return True
	except Exception:
		return False


def calculate_origin(description: DataDescriptions, frames: list[DataFrame]):
	"""Use the midpoint of both the wrist markers from at least 10 valid frames as the zero-point."""

	means = []
	for current_hand in ["left_hand", "right_hand"]:
		valid_points = []

		for frame in frames:
			if not _frame_has_valid_hand_tracking(frame, description, current_hand):
				continue

			frame_data = extract_one_frame(description, frame)[current_hand]
			if frame_data is None:
				continue

			wrist_out = np.asarray(frame_data["wrist_out"], dtype=float)
			wrist_in = np.asarray(frame_data["wrist_in"], dtype=float)
			wrist_mid = (wrist_out + wrist_in) / 2.0

			valid_points.append(wrist_mid)

		if len(valid_points) < 10:
			continue

		wrist_mean = np.mean(np.stack(valid_points), axis=0)
		means.append(wrist_mean)

	if len(means) != 2:
		return None
	return np.mean(np.stack(means), axis=0)


def zero_coordinates(frame_data: dict, baseline_origin: tuple[float, float, float]):
	"""Shift the current frame so the midpoint of the wrists becomes the new origin"""

	origin = np.asarray(baseline_origin, dtype=float)
	zeroed = {}
	zeroed["wrist_center"] = tuple(np.asarray(frame_data["wrist_center"], dtype=float) - origin)
	zeroed["fingertips"] = {}
	for finger, pos in frame_data["fingertips"].items():
		zeroed["fingertips"][finger] = tuple(np.asarray(pos, dtype=float) - origin)
	return zeroed


def test(frame: DataFrame, origin, description: DataDescriptions | None = None, threshold: float = 0.0):
	"""Identify the highest finger on each hand."""
	if description is None:
		raise ValueError("A DataDescriptions object is required to resolve the hand names for a frame.")

	results = {}

	for hand_name in ["left_hand", "right_hand"]:
		frame_data = extract_one_frame(description, frame).get(hand_name)
		if frame_data is None:
			results[hand_name] = "None"
			continue

		zeroed = zero_coordinates(frame_data, origin)
		finger_amounts = {finger: float(pos[0]) for finger, pos in zeroed["fingertips"].items()}
		if not finger_amounts:
			results[hand_name] = "None"
			continue

		best_finger, best_amount = max(finger_amounts.items(), key=lambda item: item[1])
		if best_finger is None or best_amount < threshold:
			results[hand_name] = "None"
		else:
			results[hand_name] = {"finger": best_finger, "amount": best_amount}

	return results
