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