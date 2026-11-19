import datetime as dt
import random
import tkinter as tk
from pathlib import Path
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Models.Motion.main import init as motion_init
from Models.Contact.main import init as contact_init

# I can use one of the pedal buttons as backspace (probably the middle one)
phaser = None
def handle_frame(frame):
	phaser._handle_frame(frame)

class Phase2:
	"""Coordinates current folder and quote state for each recording block."""
	def __init__(self):
		self.model_name = None
		self.model = None
		self.transform = None
		self.text = ""
		self.dropped_chars = []

		global phaser
		phaser = self

	@property
	def display_string(self):
		if self.text:
			if self.model_name is not None and "contact" in self.model_name:
				return self.text
			else:
				return "".join([x for x,y in zip(self.text, self.dropped_chars) if not y])
		return ""

	def _handle_frame(self, frame):
		res = self.model(frame)

		if self.model_name is not None and "contact" in self.model_name:
			self.text += str(res)
		else:
			if res != self.text:
				self.text = res(res)
				self.dropped_chars.append(False)

	def _change_model(self, model_name):
		if "motion" in model_name:
			self.model = motion_init(model_name)
		else:
			self.model = contact_init(model_name)
		self.model_name = model_name
		self.text = ""


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
					label.config(text=f"Change surface to {folder}\npress the left pedal button when you're ready\n({(stage_index+2)//2}/{len(stages)//2})",
					             highlightbackground="red", highlightthickness=20)
				else:
					label.config(text="Type the following quotes\npress the left pedal button when you're ready",
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
			window.unbind("<F13>")
			window.unbind("<F15>")
			window.unbind("<F14>")

			print(f"\nExperiment completed. Files saved under: {self.root_dir}")
			if self.pipe is not None:
				self.pipe.send("stop")

			label.config(text="Experiment completed\n\nThank you for participating",
			             highlightbackground="red", highlightthickness=20)
			window.after(15_000, window.destroy)

		def on_left_pedal_button(_event: tk.Event) -> None:
			nonlocal stage_index, end_time, calibration_count
			print(f"step: {step}")
			if step == "calibration" and calibration_count == 0:
				if self.pipe:
					self.pipe.put("folder")	# trigger recalibration
				next_calibration()
			elif step == "typing" and end_time is None:
				end_time = dt.datetime.now() + dt.timedelta(seconds=self.duration_seconds)
				next_quote()

		def on_right_pedal_button(_event: tk.Event) -> None:
			nonlocal calibration_count, current_quote, step, middle_pedal_count

			# the middle pedal triggers twice per press
			middle_pedal_count += 1
			if middle_pedal_count % 2 == 1:
				return

			if step == "calibration" and calibration_count != 0:
				self._finish_recording(f"calibration_{calibration_count-1}")
				next_calibration()
			if step == "typing" and current_quote is not None:
				self._finish_recording(current_quote)
				next_quote()

		def on_middle_pedal_button(_event: tk.Event) -> None:
			if self.text:
				if self.model_name is not None and "contact" in self.model_name:
					self.text = self.text[:-1]
				else:
					self.dropped_chars[-1] = True

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
		middle_pedal_count = 0

		window = tk.Tk()
		window.attributes('-fullscreen', True)
		label = tk.Label(window, text="", font=("Helvetica", 64, "bold"), wraplength=1500)
		label.pack(expand=True)

		window.bind("<F13>", on_right_pedal_button)
		window.bind("<F15>", on_left_pedal_button)
		window.bind("<F14>", on_middle_pedal_button)

		next_stage()
		window.mainloop()

if __name__ == "__main__":
	Phase2(None, (Path(__file__).parent / "data" / "test"), 
			["To live is the rarest thing in the world. Most people exist, that is all",
			"Live in the sunshine, swim the sea, drink the wild air",
			"Success usually comes to those who are too busy to be looking for it",
			"You know you're in love when you can't fall asleep because reality is finally better than your dreams" ],
			0.01).run()