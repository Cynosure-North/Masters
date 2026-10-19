#!/usr/bin/env python3
"""Experiment runner for randomCized study conditions and NatNet-style capture."""

from __future__ import annotations
import queue
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable
import keyboard  # type: ignore
import tkinter as tk
from consent import consent_screen
from interview import record_interview

FOLDER_NAMES = ["hard-flat", "hard-curved", "soft-flat", "soft-curved"]


def make_session_directory(parent: Path | None = None) -> Path:
	"""Create a UUID-named root directory and the four required condition folders."""
	run_root = (parent or Path.cwd()) / f"session_{uuid.uuid4().hex[:8]}"
	run_root.mkdir(parents=True, exist_ok=False)

	for folder_name in FOLDER_NAMES:
		(run_root / folder_name).mkdir(parents=True, exist_ok=True)

	return run_root

class NatNetRecorder:
	"""Minimal NatNet-style recorder scaffold.

	The actual OptiTrack SDK can be injected here later, but the structure matches the
	desired workflow: connect, record a quote, and save it when completion/typo is
	reported.
	"""

	def __init__(self, save_root: Path):
		self.save_root = save_root
		self._lock = threading.Lock()
		self._connected = False
		self._recording = False
		self._buffer: list[str] = []
		self._current_file: Path | None = None

	def connect(self) -> bool:
		self._connected = True
		return self._connected

	def start_recording(self, quote: str) -> Path:
		if not self._connected:
			self.connect()

		stem = quote.strip().replace(" ", "_")
		candidate = self.save_root / f"{stem}.csv"
		counter = 1
		while candidate.exists():
			candidate = self.save_root / f"{stem}_{counter}.csv"
			counter += 1

		self._buffer.clear()
		self._current_file = candidate
		self._recording = True
		self._buffer.append("timestamp,frame_id,x,y,z\n")
		return candidate

	def receive_frame(self, frame: str | dict | None) -> None:
		if not self._recording:
			return
		if isinstance(frame, dict):
			payload = str(frame)
		elif frame is None:
			payload = ""
		else:
			payload = str(frame)
		self._buffer.append(payload)

	def stop_recording(self) -> Path | None:
		self._recording = False
		return self._current_file

	def save_recording(self, *, quote: str, is_typo: bool) -> Path:
		file_path = self._current_file or self.start_recording(quote)
		target_stem = f"typo_{quote.strip().replace(' ', '_')}" if is_typo else quote.strip().replace(" ", "_")
		final_path = self.save_root / f"{target_stem}.csv"
		suffix = 1
		while final_path.exists():
			final_path = self.save_root / f"{target_stem}_{suffix}.csv"
			suffix += 1

		with file_path.open("w", encoding="utf-8") as fh:
			fh.write("".join(self._buffer))

		if file_path != final_path:
			file_path.replace(final_path)

		self._current_file = final_path
		self._recording = False
		return final_path


class NatNetMessageServer:
	"""Simple queue-based server representing the OptiTrack command channel."""

	def __init__(self, controller: "ExperimentController"):
		self.controller = controller
		self.queue: queue.Queue[dict] = queue.Queue()
		self._stop_event = threading.Event()
		self._thread = threading.Thread(target=self._run, daemon=True)

	def start(self) -> None:
		self._thread.start()

	def stop(self) -> None:
		self._stop_event.set()	# TODO: This bit

	def _run(self) -> None:
		while not self._stop_event.is_set():
			try:
				message = self.queue.get(timeout=0.1)
			except queue.Empty:
				continue

			message_type = message.get("type")
			if message_type == "set_folder":
				self.controller.set_current_folder(message["folder_path"])
			elif message_type == "start_quote":
				self.controller.start_quote(message["quote"])
			elif message_type == "mark_typo":
				self.controller.finish_quote(is_typo=True)
			elif message_type == "mark_complete":
				self.controller.finish_quote(is_typo=False)
			elif message_type == "stop":
				self._stop_event.set()
				break


class ExperimentController:
	"""Coordinates current folder and quote state for each recording block."""

	def __init__(self, root_dir: Path | None = None):
		self.root_dir = root_dir or make_session_directory()
		self.current_folder: Path | None = None
		self.current_quote: str | None = None
		self._lock = threading.Lock()
		self.recorder = NatNetRecorder(self.root_dir)
		self.server = NatNetMessageServer(self)

	def connect(self) -> bool:
		return self.recorder.connect()

	def set_current_folder(self, folder_path: str | Path) -> Path:
		path = Path(folder_path)
		path.mkdir(parents=True, exist_ok=True)
		with self._lock:
			self.current_folder = path
		return path

	def start_quote(self, quote: str) -> Path:
		with self._lock:
			self.current_quote = quote
			if self.current_folder is None:
				raise RuntimeError("No active folder selected before starting a quote.")
			return self.recorder.start_recording(quote)

	def finish_quote(self, *, is_typo: bool) -> Path:
		if self.current_quote is None:
			raise RuntimeError("No quote active to finalize.")

		quote = self.current_quote
		with self._lock:
			saved_path = self.recorder.save_recording(quote=quote, is_typo=is_typo)
			self.current_quote = None
		return saved_path

	def run_message_server(self) -> None:
		self.server.start()


class ExperimentDriver:
	"""Runs the randomized trials across all condition folders."""

	def __init__(self, controller: ExperimentController, quote_bank: Iterable[str] | None = None, duration_minutes: int = 10):
		self.controller = controller
		self.quote_bank = quote_bank
		self.duration_seconds = max(1, duration_minutes * 60)

	def is_typo(self) -> str:
		"""Wait for either a typo interrupt or a completed quote signal."""

		while True:
			if keyboard.is_pressed("f13"):
				return True
			if keyboard.is_pressed("enter"):
				return False
			time.sleep(0.05)

	def run(self) -> None:
		folders = list(FOLDER_NAMES)
		random.shuffle(folders)

		self.controller.connect()
		self.controller.run_message_server()

		window = tk.Tk()
		window.attributes('-fullscreen', True)
		label = tk.Label(window, text="", font=("Helvetica", 64, "bold"), wraplength=1500)
		label.pack(expand=True)

		for folder_name in folders:
			folder_path = self.controller.root_dir / folder_name
			self.controller.set_current_folder(folder_path)
			print(f"\nFolder: {folder_name}")

			keyboard.wait('space')
			end_time = time.monotonic() + self.duration_seconds
			while time.monotonic() < end_time:
				quote = random.choice(self.quote_bank)
				self.controller.start_quote(quote)
				label.config(text=quote)

				if self.is_typo():
					saved = self.controller.finish_quote(is_typo=True)
					print(f"Saved typo trial: {saved.name}")
				else:
					saved = self.controller.finish_quote(is_typo=False)
					print(f"Saved completed trial: {saved.name}")

		print(f"\nExperiment completed. Files saved under: {self.controller.root_dir}")


def main() -> None:
	parent = Path(r"C:\Users\mno64\Thesis\Mocap Data\phase1")
	consent = consent_screen()
	if consent is None:
		print("Consent was not submitted. The experiment will not start.")
		return

	participant_dir = make_session_directory(parent)
	controller = ExperimentController(participant_dir)
	ok_with_dataset, ok_with_quotes = consent

	if not ok_with_dataset:
		(participant_dir / "no_dataset").touch()
	if not ok_with_quotes:
		(participant_dir / "no_quotes").touch()

	print("Press F13 for a typo. Press Enter to complete a quote.")
	ExperimentDriver(controller).run()
	

	record_interview(participant_dir)

if __name__ == "__main__":
	main()