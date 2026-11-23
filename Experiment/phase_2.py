import datetime as dt
import random
import tkinter as tk
from pathlib import Path
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Models.Motion.main import init as motion_init
from Models.Contact.main import init as contact_init
import editdistance

# I can use one of the pedal buttons as backspace (probably the middle one)
phase = None
def handle_frame(frame):
	phase._handle_frame(frame)

class Phase2:
	"""Coordinates current folder and quote state for each recording block."""
	def __init__(self):
		self.model_name = None
		self.model = None
		self.transform = None

		self.raw_text = ""
		self.dropped_chars = []

		self.frames = []
		self.keystrokes[]

		global phase
		phase = self

	@property
	def display_text(self):
		if self.raw_text:
			if self.model_name is not None and "contact" in self.model_name:
				return self.raw_text
			else:
				return "".join([x for x,y in zip(self.raw_text, self.dropped_chars) if not y])
		return ""

	def _handle_frame(self, frame):
		res = self.model(frame)

		if self.model_name is not None:
			if "contact" in self.model_name:
				self.raw_text += str(res)
			else:
				self.raw_text = res

			self.dropped_chars.append(False)		# TODO: If the text changes too much the masking will get weird

		# TODO: also save frame (in its totality)
		# And if the system triggered a keystroke, what keystroke/what is the string so far
		# Also record backspaces

	def _change_model(self, model_name):
		if "motion" in model_name:
			self.model = motion_init(model_name)
			self.dropped_chars = []
		else:
			self.model = contact_init(model_name)
			self.dropped_chars = ""
		self.model_name = model_name
		self.raw_text = ""


	def save_quote(time_taken, wpm, corrected_errors, uncorrected_errors):
		pass
		{ "quote": self.quote,
		  "time_taken": time_taken,
		  "wpm": wpm,
		  "corrected_errors": corrected_errors,
		  "uncorrected_errors": uncorrected_errors,
		  "keystroke_log": self.keystroke_log
		  "frames": self.frames }


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

		def next_quote() -> None:
			nonlocal current_quote, start_time

			time_taken = (dt.datetime.now() - start_time).total_seconds()
			wpm = len(self.display_text)/(time_taken/60)	# TODO: I recall they standardised on a word being 5 chars + space

			total_errors = editdistance.eval(current_quote, self.raw_text)
			uncorrected_errors = editdistance.eval(current_quote, self.display_text)
			corrected_errors = total_errors - uncorrected_errors
			# save recording, frames, vals

			
			# Gen new quote

			self._start_recording()
			label.config(text=f"Do the calibration motion\n({calibration_count}/10)", highlightthickness=0)

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
			pass

		def on_right_pedal_button(_event: tk.Event) -> None:
			if current_quote is not None:	# TODO: and reasonably near end
				next_quote()

		def on_middle_pedal_button(_event: tk.Event) -> None:
			if self.raw_text:
				self.dropped_chars[-1] = True
				
				# TODO: And log

		stage_index = -1
		step = None
		calibration_count = 0
		current_quote = None
		start_time = None

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
