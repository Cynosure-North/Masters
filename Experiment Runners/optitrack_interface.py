import sys
import time
from pathlib import Path

import numpy as np
from natnet import DataFrame, NatNetClient

from parse_optitrack import calculate_origin, frame_point_array, test as parse_test


class OptiTrackRecorder:
	def __init__(self, recv_pipe):
		self.frames = []
		self.recording = False
		self.pipe = recv_pipe
		self.data_description = None
		self.calibrated_origin = None
		self.last_test_time = 0.0

		self.streaming_client = NatNetClient(server_ip_address="127.0.0.1", local_ip_address="127.0.0.1", use_multicast=False)
		self.streaming_client.on_data_frame_received_event.handlers.append(self.receive_new_frame)
		self.streaming_client.on_data_descriptions_received_event.handlers.append(self.receive_data_description)

	def start_server(self):
		with self.streaming_client:
			while True:
				time.sleep(1)

				# Poll optitrack
				self.streaming_client.update_sync()

				# receive commands from the experiment runner process
				try:
					if self.pipe is not None:
						message = self.pipe.get_nowait()
					else:
						message = ""
				except Exception:
					continue

				if message == "start":	# Start recording
					self.recording = True
				elif message == "stop":	# End session
					break
				elif message == "folder":	# recalibrate for new folder
					self.calibrated_origin == None
				else:	# End recording
					with open(Path(message), 'w') as file:
						file.write(str(self.frames))

					self.recording = False
					self.frames = []
		sys.exit()

	def receive_data_description(self, data_desc):
		self.data_description = data_desc

	def receive_new_frame(self, data_frame: DataFrame):
		if self.recording and self.data_description is not None:
			self.frames.append(frame_point_array(data_frame, self.data_description))

		if self.calibrated_origin is None:	# Keeps retrying until it sticks
			self.calibrated_origin = calculate_origin(self.data_description, self.frames)

	def test(self, frame):
		now = time.monotonic()
		if now - self.last_test_time < 1.0:
			return
		self.last_test_time = now

		if self.calibrated_origin is None and self.data_description is not None:
			self.calibrated_origin = calculate_origin(self.data_description, self.frames)
		if self.calibrated_origin is None or self.data_description is None:
			return

		print(parse_test(frame, self.calibrated_origin, self.data_description))


if __name__ == "__main__":
	recorder = OptiTrackRecorder(None)
	recorder.streaming_client.on_data_frame_received_event.handlers.append(recorder.test)

# NOTE: The marker positions of rigid bodies are only transmitted correctly if "Y-up" is selected in the streaming pane. If "Z-up" is selected, the frame of the rigid bodies is rotated but the marker positions are not, resulting in wrong positions of the markers relative to the rigid body.