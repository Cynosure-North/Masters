import sys
from threading import Lock
from natnet import DataFrame, NatNetClient
from phase_2 import handle_frame
from parse_optitrack import calculate_origin, frame_point_array

# TODO: I want proper optitrack recordings, not my jank

class OptiTrackRecorder:
	def __init__(self):
		self.data_description = None
		self.calibrated_origin = None

		self.streaming_client = NatNetClient(server_ip_address="127.0.0.1", local_ip_address="127.0.0.1", use_multicast=False)
		self.streaming_client.on_data_frame_received_event.handlers.append(self.receive_new_frame)
		self.streaming_client.on_data_description_received_event.handlers.append(self.receive_data_description)

	def start_server(self):
		# This object is passed to multiprocessing.Process on Windows. Create
		# thread-only state after the child process has been spawned.
		self._state_lock = Lock()

		with self.streaming_client:
			# Start the receiver before requesting model definitions. The response is
			# asynchronous and must not depend on a one-shot synchronous read.
			self.streaming_client.run_async()
			self.streaming_client.request_modeldef()		# TODO: May want to redo pipe to enable recalibration

			print("OptiTrack server started.")

		sys.exit()

	def receive_data_description(self, data_desc):
		self.data_description = data_desc

	def receive_new_frame(self, data_frame: DataFrame):
		if self.data_description is None:
			return

		frame = frame_point_array(data_frame, self.data_description, self.calibrated_origin)
		if frame is not None:

			# Keep retrying until it sticks
			if self.calibrated_origin is None:
				self.calibrated_origin = calculate_origin(self.frames)

			if self.calibrated_origin is not None:
				handle_frame(data_frame, frame)		# TODO: may not be able to do interprocess calls


if __name__ == "__main__":
	recorder = OptiTrackRecorder(None)
	recorder.test = True
	recorder.start_server()
	

# NOTE: The marker positions of rigid bodies are only transmitted correctly if "Y-up" is selected in the streaming pane. If "Z-up" is selected, the frame of the rigid bodies is rotated but the marker positions are not, resulting in wrong positions of the markers relative to the rigid body.