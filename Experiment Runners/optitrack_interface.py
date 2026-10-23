import sys
import time
from natnet import DataFrame, NatNetClient
from pathlib import Path


def OptiTrackRecorder():
	def __init__(self, recv_pipe):
		self.frames = []
		self.recording = False
		self.pipe = recv_pipe

		self.streaming_client = NatNetClient(server_ip_address="127.0.0.1", local_ip_address="127.0.0.1", use_multicast=False)
		self.streaming_client.on_data_frame_received_event.handlers.append(receive_new_frame)

	def start_server(self):
		with self.streaming_client:
			while True:
				time.sleep(1)

				# Poll optitrack
				self.streaming_client.update_sync()

				# receive commands
				try:
					message = self.pipe.get_nowait()
				except:
					continue
				
				if message == "start":
					self.start_recording()
				if message == "stop":
					break
				else:
					self.end_recording(message)
		sys.exit()

	def receive_new_frame(self, data_frame: DataFrame):
		if self.recording:
			self.frames.append(data_frame)

	def start_recording(self):
		self.recording = True

	def end_recording(self, save_path: str):
		self.recording = False
		with open(Path(save_path), 'w') as file:
			file.write(str(self.frames))
		self.frames = []


# NOTE: The marker positions of rigid bodies are only transmitted correctly if "Y-up" is selected in the streaming pane. If "Z-up" is selected, the frame of the rigid bodies is rotated but the marker positions are not, resulting in wrong positions of the markers relative to the rigid body.