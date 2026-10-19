import tkinter as tk
import sounddevice as sd
import time
import wave
from pathlib import Path
from array import array

def _audio_has_speech(frames: bytes) -> bool:
	"""Return True if the clip has a simple speech-like amplitude profile."""
	if not frames:
		return False

	values = array("h")
	values.frombytes(frames)
	if not values:
		return False

	mean_abs = sum(abs(value) for value in values) / len(values)
	return mean_abs > 80

def _save_wav(path: Path, frames: bytes, sample_rate: int, channels: int) -> None:
	with wave.open(str(path), "wb") as wav_file:
		wav_file.setnchannels(channels)
		wav_file.setsampwidth(2)
		wav_file.setframerate(sample_rate)
		wav_file.writeframes(frames)

def record_interview(save_dir: Path) -> None:
	"""Ask the participant to record an interview clip and save it as interview.wav."""
	root = tk.Tk()
	root.title("Interview Recording")
	root.geometry("420x170")
	root.resizable(False, False)

	state: dict[str, object] = {
		"active": False,
		"started_at": 0.0,
		"frames": bytearray(),
		"sample_rate": 44100,
		"channels": 1,
		"stream": None,
	}

	def save_current_recording() -> None:
		output_path = save_dir / "interview.wav"
		frames = bytes(state["frames"])
		if frames and not _audio_has_speech(frames, int(state["sample_rate"]), int(state["channels"])):
			if not tk.messagebox.askyesno(
				"Low audio detected",
				"This recording does not appear to contain speech. Do you want to keep it anyway?",
			):
				status.config(text="Recording discarded; continue speaking and click Start recording again.")
				state["frames"] = bytearray()
				button.config(text="Start recording")
				root.update_idletasks()
				return False
		if frames:
			_save_wav(output_path, frames, int(state["sample_rate"]), int(state["channels"]))
			status.config(text=f"Saved to {output_path.name}")
		return True

	def toggle_recording() -> None:
		if not state["active"]:
			state["active"] = True
			state["started_at"] = time.monotonic()
			state["frames"] = bytearray()
			stream = sd.InputStream(
				samplerate=int(state["sample_rate"]),
				channels=int(state["channels"]),
				dtype="int16",
				callback=lambda indata, frames, time_info, status: state["frames"].extend(indata.tobytes()),
			)
			state["stream"] = stream
			stream.start()
			button.config(text="Stop recording")
			status.config(text="Recording in progress...")
			return

		stream = state["stream"]
		if stream is not None:
			stream.stop()
		state["active"] = False
		state["stream"] = None
		button.config(text="Start recording")
		duration = time.monotonic() - float(state["started_at"])
		if duration < 30:
			if tk.messagebox.askyesno(
				"Short recording",
				"This recording is less than 30 seconds. Are you sure you're finished?",
			):
				if save_current_recording():
					stream.close()
					root.destroy()
			else:
				stream.start()
				return
		else:
			if save_current_recording():
				stream.close()
				root.destroy()
			return

		# state["frames"] = bytearray()
		# status.config(text="Continue speaking, then click Start recording again.")

	label = tk.Label(root, text="Click below to start or stop the interview recording.", font=("Segoe UI", 11))
	label.pack(pady=(18, 8))
	button = tk.Button(root, text="Start recording", font=("Segoe UI", 12, "bold"), command=toggle_recording)
	button.pack(pady=(0, 8))
	status = tk.Label(root, text="Waiting for recording.", font=("Segoe UI", 10))
	status.pack()
	root.protocol("WM_DELETE_WINDOW", root.destroy)
	root.mainloop()