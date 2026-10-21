from pathlib import Path
import uuid
import multiprocessing as mp

from optitrack_interface import OptiTrackRecorder
from consent import main as consent_screen
import tutorial
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
	parent = Path(r"C:\Users\mno64\Thesis\Mocap Data\phase" + str(phase))
	phase = 1
	participant_dir = make_session_directory(parent)
	with open(r"C:\Users\mno64\Thesis\Experiment Runners\data\phrases.txt") as f:
		quotes = f.read().splitlines()

	consent(participant_dir)

	if phase == 1:
		tutorial.phase_1()

		recv, send = mp.Pipe(duplex=False)
		recorder = OptiTrackRecorder(recv)
		phase1 = Phase1(participant_dir, send, quotes)
		
		mp.Process(target=recorder.start_server).run()
		mp.Process(target=phase1.run).run()

		record_interview()
		