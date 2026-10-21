import multiprocessing as mp
import sys
import random
import time
from pathlib import Path
import keyboard  # type: ignore
import tkinter as tk

FOLDER_NAMES = ["hard-flat", "hard-curved", "soft-flat", "soft-curved"]


class Phase1:
	"""Coordinates current folder and quote state for each recording block."""

	def __init__(self, pipe, root_dir: Path, _quote_bank: list[str], duration_minutes: int = 40):
		self.pipe = pipe
		self.root_dir = root_dir
		self.quote_bank = _quote_bank
		self.current_folder: Path | None = None
		self.current_quote: str | None = None
		self.duration_seconds = max(1, duration_minutes * 60)

	def _set_current_folder(self, folder_path: str | Path) -> Path:
		path = Path(folder_path)
		path.mkdir(parents=True, exist_ok=True)
		self.current_folder = path
		return path

	def _start_quote(self, quote: str) -> None:
		self.current_quote = quote
		if self.current_folder is None:
			raise RuntimeError("No active folder selected before starting a quote.")
		self.pipe.put("start")

	def _finish_quote(self, *, is_typo: bool) -> Path:
		if self.current_quote is None:
			raise RuntimeError("No quote active to finalize.")

		filepath = self.current_folder / self.current_quote + "_typo" if is_typo else ""
		saved_path = self.pipe.put(filepath)
		self.current_quote = None
		return saved_path

	def _is_typo(self) -> str:
		"""Wait for either a typo interrupt or a completed quote signal."""

		while True:
			if keyboard.is_pressed("f13"):
				return True
			if keyboard.is_pressed("enter"):
				return False
			time.sleep(0.5)

	def run(self) -> None:
		print("Press F13 for a typo. Press Enter to complete a quote.")

		random.shuffle(FOLDER_NAMES)
		with open(self.root_dir / "surface_order", 'w') as f:
			f.write(FOLDER_NAMES)

		window = tk.Tk()
		window.attributes('-fullscreen', True)
		label = tk.Label(window, text="", font=("Helvetica", 64, "bold"), wraplength=1500)
		label.pack(expand=True)

		for folder_name in FOLDER_NAMES:
			folder_path = self.root_dir / folder_name
			self._set_current_folder(folder_path)
			print(f"\nFolder: {folder_name}")

			keyboard.wait('space')
			end_time = time.monotonic() + self.duration_seconds
			while time.monotonic() < end_time:
				quote = random.choice(self.quote_bank)
				self._start_quote(quote)
				label.config(text=quote)

				if self._is_typo():
					saved = self._finish_quote(is_typo=True)
					print(f"Saved typo trial: {saved.name}")
				else:
					saved = self._finish_quote(is_typo=False)
					print(f"Saved completed trial: {saved.name}")

		print(f"\nExperiment completed. Files saved under: {self.root_dir}")
		self.pipe.put("stop")
		sys.exit()