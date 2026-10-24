import sys
import time
from pathlib import Path
from natnet import DataFrame, NatNetClient

from parse_optitrack import calculate_origin, frame_point_array, parse_test


class OptiTrackRecorder:
	def __init__(self, recv_pipe):
		self.frames = []
		self.pipe = recv_pipe
		self.data_description = None
		self.calibrated_origin = None
		self.last_test_time = 0.0

		self.streaming_client = NatNetClient(server_ip_address="127.0.0.1", local_ip_address="127.0.0.1", use_multicast=False)
		self.streaming_client.on_data_frame_received_event.handlers.append(self.receive_new_frame)
		self.streaming_client.on_data_description_received_event.handlers.append(self.receive_data_description)

	def start_server(self):
		with self.streaming_client:
			self.streaming_client.request_modeldef()
			self.streaming_client.update_sync()

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
					self.frames = []
				elif message == "stop":	# End session
					break
				elif message != "":		# End recording
					with open(Path(message), 'w') as file:		# TODO: Numpy
						file.write(str(self.frames))

		sys.exit()

	def receive_data_description(self, data_desc):
		self.data_description = data_desc

	def receive_new_frame(self, data_frame: DataFrame):
		if self.data_description is not None:
			self.frames.append(frame_point_array(data_frame, self.data_description, self.calibrated_origin))

						# Keeps retrying until it sticks
			if self.calibrated_origin is None:
				self.calibrated_origin = calculate_origin(self.frames)
				if self.calibrated_origin is not None:
					self.frames = list(map(lambda l: l - self.calibrated_origin, self.frames))
				


	def test(self, frame):
		now = time.monotonic()
		# if now - self.last_test_time < 2.0:
		# 	return
		self.last_test_time = now
		
		if self.calibrated_origin is None or self.data_description is None:
			return

		# print(parse_test(frame_point_array(frame, self.data_description, self.calibrated_origin)))


if __name__ == "__main__":
	recorder = OptiTrackRecorder(None)
	recorder.streaming_client.on_data_frame_received_event.handlers.append(recorder.test)
	recorder.start_server()
	

# NOTE: The marker positions of rigid bodies are only transmitted correctly if "Y-up" is selected in the streaming pane. If "Z-up" is selected, the frame of the rigid bodies is rotated but the marker positions are not, resulting in wrong positions of the markers relative to the rigid body.