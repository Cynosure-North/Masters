import sys
import time
from threading import Lock
from pathlib import Path
from natnet import DataFrame, NatNetClient

from parse_optitrack import calculate_origin, frame_point_array, parse_test


class OptiTrackRecorder:
	def __init__(self, recv_pipe):
		self.pipe = recv_pipe

		self.frames = []
		self.received_frames = 0
		self.rejected_frames = 0
		self.data_description = None
		self.calibrated_origin = None

		self._state_lock = None

		self.test = False
		self.last_test_time = 0.0

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
			self.streaming_client.request_modeldef()

			print("OptiTrack server started.")
			while True:
				# receive commands from the experiment runner process
				if self.pipe is None:
					time.sleep(0.1)
					continue

				if not self.pipe.poll(0.1):
					continue

				message = self.pipe.recv()
				if message == "start":	# Start recording
					with self._state_lock:
						self.frames = []
						self.received_frames = 0
						self.rejected_frames = 0
				elif message == "stop":	# End session
					break
				elif message == "folder":	# recalibrate for a new surface
					self.calibrated_origin = None
				elif message != "":		# End recording
					with self._state_lock:
						frames = list(self.frames)
						received_frames = self.received_frames
						rejected_frames = self.rejected_frames
					print(
						f"Saving {len(frames)} frames to {message} "
						f"(received={received_frames}, rejected={rejected_frames})"
					)
					with open(Path(message), 'w') as file:
						file.write(str(frames))

		sys.exit()

	def receive_data_description(self, data_desc):
		with self._state_lock:
			self.data_description = data_desc

	def receive_new_frame(self, data_frame: DataFrame):
		# print(
		# 	data_frame.prefix.frame_number,
		# 	[
		# 		(
		# 			skeleton.id_num,
		# 			skeleton.rigid_bodies[0].pos,
		# 			skeleton.rigid_bodies[0].rot,
		# 			skeleton.rigid_bodies[0].tracking_valid,
		# 		)
		# 		for skeleton in data_frame.skeletons
		# 	],
		# )

		with self._state_lock:
			self.received_frames += 1
			if self.data_description is None:
				self.rejected_frames += 1
				return

			frame = frame_point_array(data_frame, self.data_description, self.calibrated_origin)
			if frame is not None:
				self.frames.append(frame)

				# Keeps retrying until it sticks
				if self.calibrated_origin is None:
					self.calibrated_origin = calculate_origin(self.frames)
					if self.calibrated_origin is not None:
						self.frames = list(map(lambda l: (l[0], l[1] - self.calibrated_origin), self.frames))

				if self.test:
					# now = time.monotonic()
					# if now - self.last_test_time < 2.0:
					# 	return
					# self.last_test_time = now
					print(parse_test(frame))
			else:
				self.rejected_frames += 1
				

if __name__ == "__main__":
	recorder = OptiTrackRecorder(None)
	recorder.test = True
	recorder.start_server()
	

# NOTE: The marker positions of rigid bodies are only transmitted correctly if "Y-up" is selected in the streaming pane. If "Z-up" is selected, the frame of the rigid bodies is rotated but the marker positions are not, resulting in wrong positions of the markers relative to the rigid body.