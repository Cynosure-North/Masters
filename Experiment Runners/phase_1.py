import sys
import random
import datetime as dt
from pathlib import Path
import tkinter as tk

class Phase1:
	"""Coordinates current folder and quote state for each recording block."""

	def __init__(self, pipe, root_dir: Path, _quote_bank: list[str], duration_minutes: float = 8):
		self.pipe = pipe
		self.root_dir = root_dir
		self.quote_bank = _quote_bank
		self.current_folder = None
		self.duration_seconds = duration_minutes * 60

	def _set_current_folder(self, folder_path: str | Path) -> None:
		path = Path(folder_path)
		path.mkdir(parents=True, exist_ok=True)
		self.current_folder = path

	def _start_recording(self) -> None:
		if self.current_folder is None:
			raise RuntimeError("No active folder selected before starting a recording.")
		if self.pipe is not None:
			self.pipe.put("start")

	def _finish_recording(self, filename: str) -> None:
		filepath = self.current_folder / filename
		if self.pipe is not None:
			self.pipe.put(filepath)

	def run(self) -> None:

		def next_stage():
			nonlocal stage_index, step, stages

			stage_index += 1
			if stage_index >= len(stages):
				end_of_experiment()
				return
			else:
				folder, step = stages[stage_index]
				print(f"Folder: {folder}, Step: {step}")
				self._set_current_folder(self.root_dir / folder)

				if step == "calibration":
					label.config(text=f"Change surface to {folder}\npress Enter when you're ready\n({(stage_index+2)//2}/{len(stages)//2})",
					             highlightbackground="red", highlightthickness=20)
				else:
					label.config(text="Type the following quotes\npress Enter when you're ready",
					             highlightbackground="red", highlightthickness=20)


		def next_calibration():
			nonlocal calibration_count

			print(f"calib count: {calibration_count}")
			if calibration_count >= 10:
				calibration_count = 0
				next_stage()
			else:
				calibration_count += 1
				self._start_recording()
				label.config(text=f"Do the calibration motion\n({calibration_count}/10)", highlightthickness=0)

		def next_quote() -> None:
			nonlocal current_quote, end_time

			if dt.datetime.now() >= end_time:
				current_quote, end_time = None, None
				next_stage()
			else:
				current_quote = random.choice(self.quote_bank)
				self._start_recording()
				label.config(text=current_quote, highlightthickness=0)

		def end_of_experiment():
			window.bind("<space>", on_space)
			window.bind("<Return>", on_enter)
			window.bind("<F13>", on_f13)

			print(f"\nExperiment completed. Files saved under: {self.root_dir}")
			if self.pipe is not None:
				self.pipe.put("stop")

			label.config(text="Experiment completed\n\nThank you for participating",
			             highlightbackground="red", highlightthickness=20)
			window.after(15_000, window.destroy)

		def on_enter(_event: tk.Event) -> None:
			nonlocal stage_index, end_time, calibration_count
			print(f"step: {step}")
			if step == "calibration" and calibration_count == 0:
				if self.pipe:
					self.pipe.put("folder")	# trigger recalibration
				next_calibration()
			elif step == "typing" and end_time is None:
				end_time = dt.datetime.now() + dt.timedelta(seconds=self.duration_seconds)
				next_quote()

		def on_space(_event: tk.Event) -> None:
			nonlocal calibration_count, current_quote

			if step == "calibration" and calibration_count != 0:
				self._finish_recording(f"calibration_{calibration_count-1}")
				next_calibration()
			if step == "typing" and current_quote is not None:
				self._finish_recording(current_quote)
				next_quote()

		def on_f13(_event: tk.Event) -> None:
			nonlocal current_quote

			if step == "typing" and current_quote is not None:
				self._finish_recording(current_quote + "_typo")
				next_quote()

		folder_names = ["table", "pipe", "cushion", "legs"]
		random.shuffle(folder_names)
		with open(self.root_dir / "surface_order", 'w') as f:
			f.write(str(folder_names))

		stages = [(folder, step) for folder in folder_names for step in ("calibration", "typing")]

		stage_index = -1
		step = None
		calibration_count = 0
		current_quote = None
		end_time = None

		window = tk.Tk()
		window.attributes('-fullscreen', True)
		label = tk.Label(window, text="", font=("Helvetica", 64, "bold"), wraplength=1500)
		label.pack(expand=True)

		window.bind("<space>", on_space)
		window.bind("<Return>", on_enter)
		window.bind("<F13>", on_f13)

		next_stage()
		window.mainloop()

if __name__ == "__main__":
	Phase1(None, (Path(__file__).parent / "data" / "test"), 
			["To live is the rarest thing in the world. Most people exist, that is all",
			"Live in the sunshine, swim the sea, drink the wild air",
			"Success usually comes to those who are too busy to be looking for it",
			"You know you're in love when you can't fall asleep because reality is finally better than your dreams" ],
			0.01).run()