from __future__ import annotations
import pprint
from dataclasses import dataclass, asdict
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


def extract_wrist(description: DataDescriptions, frame: DataFrame, hand_name: str):
	"""Map marker names to 3D positions for a single hand. (markers are the retroreflective speres)"""
	marker_desc = next(md for md in description.marker_sets if md.name == hand_name)
	marker_names = list(marker_desc.marker_names)
	marker_set = next(ms for ms in frame.marker_sets if ms.model_name == hand_name)
	pos_list = list(marker_set.marker_pos_list)
	if len(pos_list) != len(marker_names):
		raise ValueError(f"Marker count mismatch for {hand_name}: {len(pos_list)} vs {len(marker_names)}")
	markers = {name.rsplit("_", 1)[-1][1:]: pos for name, pos in zip(marker_names, pos_list)}

	wrist_out = markers["WristOut"]
	wrist_in = markers["WristIn"]
	if wrist_out is None or wrist_in is None:
		raise KeyError(f"Could not find wrist markers for {hand_name}")

	skeleton_id = next(sd.id_num for sd in description.skeletons if sd.name == hand_name)
	skeleton = next(s for s in frame.skeletons if s.id_num == skeleton_id)
	# The root rigid body is the first rigid body in the skeleton; its rotation is in quaternion form.
	root_rb = skeleton.rigid_bodies[0]
	return {
		"wrist_center": tuple((x + y) / 2.0 for x, y in zip(wrist_out, wrist_in)),
		"rotation_quaternion": root_rb.rot,

	}
	# TODO: I'll probably want to convert it to euler angles

def extract_fingertips(description: DataDescriptions, frame: DataFrame, hand_name: str):
	# The live frame sometimes remaps rigid-body IDs; the stable way to match them is by ordering.
	desc_skeleton = next(sd for sd in description.skeletons if sd.name == hand_name)
	frame_skeleton = next(s for s in frame.skeletons if s.id_num == desc_skeleton.id_num)
	desc_names = [rb.name for rb in desc_skeleton.rigid_body_descriptions]
	rb_positions = frame_skeleton.rigid_bodies

	tips = {}
	fingers = ("Thumb", "IndeX", "Middle", "Ring", "Pinky")
	for desc_name, rb in zip(desc_names, rb_positions):
		suffix = desc_name.rsplit("_", 1)[-1]
		if any(finger in suffix for finger in fingers):
			tips[suffix] = rb.pos
	return tips


def frame_point_array(frame: DataFrame, description: DataDescriptions, origin: np.ndarray | None=None) -> np.ndarray:
	"""Flatten a single frame into a (14, 3) NumPy array

	The canonical ordering of tracked points in the flattened frame array is:
	  0		left hand wrist center
	  1		left hand wrist rotation
	  2-6   left_hand fingertips: LeftHandThumb3, LeftHandIndex3, LeftHandMiddle3, LeftHandRing3, LeftHandPinky3
	  7		right hand wrist center
	  8		right hand wrist rotation
	  9-13	right_hand fingertips: RightHandThumb3, RightHandIndex3, RightHandMiddle3, RightHandRing3, RightHandPinky3
	"""
	if description is None:
		raise ValueError("A DataDescriptions object is required to resolve point names for a frame.")

	if frame.skeletons:
		print("".join(list(map(lambda l: f"{l.pos[0]:< 5.5f}\t", frame.skeletons[0].rigid_bodies))), sep='')

	points = []

	for hand_name in ("left_hand", "right_hand"):
		try:			
			wrist_values = extract_wrist(description, frame, hand_name)
			fingertip_values = extract_fingertips(description, frame, hand_name)
			values = wrist_values | fingertip_values

			for value in [
				'wrist_center',
				'wrist_rotation',
				'thumb',
				'index',
				'middle',
				'ring',
				'little']:
				
				if value in values:
					points.append(values[value])
				else:
					points.append(np.zeros(3, dtype=float))

		except Exception as e:
			continue

	arr = np.asarray(points, dtype=float)
	
	# TODO: I don't know which points are missing
	# TODO: detect missing frames, this may do enough though


	if origin is not None:
		arr = arr - origin

	return arr

def calculate_origin(frames: np.ndarray):
	"""Use the midpoint of both the wrist markers from at least 10 valid frames as the zero-point."""

	if len(frames) < 10:
		print(f"not enough frames to calibrate origin: {len(frames)}/10")
		return None

	means = []
	for current_hand in ["left_hand", "right_hand"]:
		valid_points = []

		for frame in frames:

			valid_points.append(frame[0])
			valid_points.append(frame[7])

		if len(valid_points) < 10:
			# print(f"not enough valid points: {len(valid_points)}/{len(frames)}")
			continue

		wrist_mean = np.mean(np.stack(valid_points), axis=0)
		means.append(wrist_mean)

	if len(means) != 2:
		print("At least one hand not tracking correctly")
		return None
	return np.mean(np.stack(means), axis=0)


def parse_test(frame: np.ndarray, threshold: float = 0.0):
	"""Identify the highest finger on each hand."""
	
	results = {}


	for hand in ["left", "right"]:
		finger_amounts = frame[2:7, 1] if hand == "left" else frame [9:14, 1]
		
		best_finger = np.argmax(finger_amounts)
		if float(finger_amounts[best_finger]) < threshold:
			results[hand] = "None"
		else:
			results[hand] = {"finger": ['thumb', 'index', 'middle', 'ring', 'little'][best_finger],
			                 "amount": finger_amounts[best_finger]}

	return results
