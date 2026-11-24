import datetime as dt
import math
import random
import tkinter as tk
from pathlib import Path
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Models.Motion.main import init_model

phase_runner = None
def handle_frame(raw_frame, frame):
	phase_runner._handle_frame(raw_frame, frame)

class Phase2:
	"""Coordinates current folder and quote state for each recording block."""
	def __init__(self, root_dir: Path, quote_bank_: list[str], duration_minutes: float = 8):
		self.root_dir = root_dir
		self.quote_bank = quote_bank_
		random.shuffle(self.quote_bank)
		self.duration_seconds = duration_minutes * 60
		self.stage = None

		self.model = init_model()
		# TODO: Depending on what turns out best I may need a calibration step

		self.tk_typed_text = tk.StringVar()
		self.raw_text = ""
		self.dropped_chars = []
		self.frames = []
		self.text_log = []
		self.keyboard_log = []

		global phase_runner
		phase_runner = self

	@property
	def display_text(self):
		if self.raw_text:
			return "".join([x for x,y in zip(self.raw_text, self.dropped_chars) if not y])
		return ""

	def _handle_frame(self, raw_frame, frame):
		res = self.model(frame)

		if res is not None:
			self.dropped_chars.append(False)		# NOTE: If the text changes too much the masking will get weird
			self.raw_text = res
			self.tk_typed_text.set(self.display_text)

		now = dt.datetime.now()
		self.frames.append((now, raw_frame[1]))
		self.text_log.append((now, self.raw_text, self.dropped_chars))

	def _set_current_folder(self, folder_path: str | Path) -> None:
		path = Path(folder_path)
		path.mkdir(parents=True, exist_ok=True)
		self.current_folder = path

	def _finish_quote(self, quote, time_taken):
		result = {
			"quote": quote,
			"time_taken": time_taken,
			"keystroke_log": self.keystroke_log,
			"frames": self.frames }

		with open(self.root_dir / self.stage / f"{quote}.json", 'w') as file:
			file.write(json.dumps(result))

		self.tk_typed_text.set("")
		self.raw_text = ""
		self.dropped_chars = []
		self.frames = []
		self.keystroke_log = []
		self.keyboard_log = []


	def run(self) -> None:

		def next_stage():
			nonlocal stages, stage_end_time
			stage_end_time = None

			if stages:
				self.stage = stages.pop()
				target_text.config(text=f"Change surface to {self.stage}\npress the left pedal button when you're ready\n({3-len(stages)}/3)",
								highlightbackground="red", highlightthickness=20)
			else:
				end_of_experiment()
				return

		def next_quote() -> None:
			nonlocal current_quote, start_time, stage_end_time

			now = dt.datetime.now()
			time_taken = (now - start_time).total_seconds()

			if now >= stage_end_time:
				next_stage()
			else:
				# TODO: Write parsing code now
				#  I recall they standardised on a word being 5 chars + space
				#  Should wpm be based on the text typed or on the target text?
				self._finish_quote(current_quote, time_taken)

				current_quote = self.quote_bank.pop().lower()

				start_time = now
				target_text.config(text=current_quote, highlightthickness=0)

		def end_of_experiment():
			window.unbind("<F13>")
			window.unbind("<F15>")
			window.unbind("<F14>")

			print(f"\nExperiment completed. Files saved under: {self.root_dir}")

			target_text.config(text="Experiment completed\n\nThank you for participating",
							highlightbackground="red", highlightthickness=20)

			window.after(15_000, window.destroy)

		def on_left_pedal_button(_event: tk.Event) -> None:
			nonlocal stage_end_time

			if stage_end_time is None:
				stage_end_time = dt.datetime.now() + dt.timedelta(seconds=self.duration_seconds)
				next_quote()

		def on_right_pedal_button(_event: tk.Event) -> None:
			nonlocal current_quote

			# check if they're reasonably close to the end (to somewhat prevent accidental presses)
			if current_quote is not None and \
			math.abs((len(self.display_text) - current_quote) / len(current_quote)) < 0.1:
				next_quote()

		def on_middle_pedal_button(_event: tk.Event) -> None:
			if self.raw_text:
				self.dropped_chars[-1] = True
				self.tk_typed_text.set(self.display_text)

		def on_keyboard(event: tk.Event) -> None:
			if event.char.isalpha() and self.stage == "keyboard":
				self.keyboard_log.append((dt.datetime.now, event.char))

				# TODO: It'd be so weird if the keyboard didn't actually do anything
				self.raw_text.append(event.char)
				self.tk_typed_text.set(self.display_text)
				

		stages = ["table", "legs", "keyboard"]
		random.shuffle(stages)
		current_quote = None
		start_time = None
		stage_end_time = None

		window = tk.Tk()
		window.attributes('-fullscreen', True)
		target_text = tk.Label(window, text="", font=("Helvetica", 64, "bold"), wraplength=1500)
		typed_text = tk.Label(window, textvariable=self.tk_typed_text, font=("Helvetica", 64, "bold"), wraplength=1500)
		target_text.pack(expand=True)
		typed_text.pack(expand=True)

		window.bind("<F13>", on_right_pedal_button)
		window.bind("<F15>", on_left_pedal_button)
		window.bind("<F14>", on_middle_pedal_button)
		window.bind("<Key>", on_keyboard)

		next_stage()
		window.mainloop()

if __name__ == "__main__":
	Phase2(None, (Path(__file__).parent / "data" / "test"), 
			["To live is the rarest thing in the world. Most people exist, that is all",
			"Live in the sunshine, swim the sea, drink the wild air",
			"Success usually comes to those who are too busy to be looking for it",
			"You know you're in love when you can't fall asleep because reality is finally better than your dreams"],
			0.01).run()


# TODO: I trained the LLM to also include comma and fullstop