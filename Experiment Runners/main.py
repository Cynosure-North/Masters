from pathlib import Path
import uuid
import multiprocessing as mp

from optitrack_interface import OptiTrackRecorder
from consent import main as consent_screen
from phase_1 import Phase1
from interview import main as record_interview

def make_session_directory(parent: Path) -> Path:
	"""Create a UUID-named root directory and the four required condition folders."""
	run_root = parent / f"session_{uuid.uuid4().hex[:8]}"
	run_root.mkdir(parents=True, exist_ok=False)

	return run_root

def consent(participant_dir):
	consent = consent_screen() 
	if consent is None:
		print("Consent was not submitted. The experiment will not start.")
		return

	ok_with_dataset, ok_with_quotes = consent

	if not ok_with_dataset:
		(participant_dir / "no_dataset").touch()
	if not ok_with_quotes:
		(participant_dir / "no_quotes").touch()

def main():
	phase = 1
	parent = Path(Path(__file__).parent.parent / "Mocap Data" / f"phase{phase}")
	participant_dir = make_session_directory(parent)
	with open(Path(__file__).parent / "data" / "phrases.txt") as f:
		quotes = f.read().splitlines()

	consent(participant_dir)

	if phase == 1:
		recv, send = mp.Pipe(duplex=False)
		recorder = OptiTrackRecorder(recv)
		phase1 = Phase1(send, participant_dir, quotes, 0.5)

		recorder_process = mp.Process(target=recorder.start_server)
		phase_process = mp.Process(target=phase1.run)
		recorder_process.start()
		phase_process.start()

		try:
			record_interview(participant_dir)
		finally:
			send.send("stop")
			recorder_process.join(2)
			if recorder_process.is_alive():
				recorder_process.terminate()
				recorder_process.join()

			if phase_process.is_alive():
				phase_process.terminate()
				phase_process.join()
		
if __name__ == "__main__":
	main()