import re
from pathlib import Path

import numpy as np
import torch


DEFAULT_DATASET_DIR = Path(r"C:\Users\mno64\Datasets\LibriSpeech ASR\dev-clean")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def _safe_sentence_stem(sentence):
	"""Make transcript text usable as a Windows filename while preserving its words."""
	stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", sentence).rstrip(" .")
	stem = stem[:180].rstrip(" .")
	return stem or "empty-transcript"


def generate_librispeech_ctc(dataset_dir=DEFAULT_DATASET_DIR, output_dir=DEFAULT_OUTPUT_DIR):
	"""Save one frame-by-label CTC probability matrix per LibriSpeech utterance.

	LibriSpeech transcript files are named ``*.trans.txt`` and contain lines of
	``utterance-id transcript``. Output arrays are float32 ``.npy`` files shaped
	``(time_frames, labels)``; columns follow ``bundle.get_labels()`` order.
	"""
	import torchaudio

	dataset_dir = Path(dataset_dir)
	output_dir = Path(output_dir)
	if not dataset_dir.is_dir():
		raise FileNotFoundError(f"LibriSpeech directory not found: {dataset_dir}")

	transcript_files = sorted(dataset_dir.rglob("*.trans.txt"))
	if not transcript_files:
		raise FileNotFoundError(f"No LibriSpeech *.trans.txt files found in: {dataset_dir}")

	bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	acoustic_model = bundle.get_model().to(device).eval()
	output_dir.mkdir(parents=True, exist_ok=True)
	name_counts = {}
	saved_paths = []

	with torch.inference_mode():
		for transcript_file in transcript_files:
			with transcript_file.open("r", encoding="utf-8") as transcript:
				for line_number, line in enumerate(transcript, start=1):
					line = line.strip()
					if not line:
						continue
					parts = line.split(maxsplit=1)
					if len(parts) != 2:
						raise ValueError(
							f"Malformed transcript at {transcript_file}:{line_number}: {line!r}"
						)
					utterance_id, sentence = parts
					audio_file = transcript_file.parent / f"{utterance_id}.flac"
					if not audio_file.is_file():
						raise FileNotFoundError(f"Audio file listed in transcript not found: {audio_file}")

					waveform, sample_rate = torchaudio.load(audio_file)
					if waveform.shape[0] > 1:
						waveform = waveform.mean(dim=0, keepdim=True)
					if sample_rate != bundle.sample_rate:
						waveform = torchaudio.functional.resample(
							waveform, sample_rate, bundle.sample_rate
						)
					waveform = waveform.to(device)

					emission, _ = acoustic_model(waveform)
					probabilities = torch.softmax(emission[0], dim=-1).cpu().numpy().astype(np.float32)

					stem = _safe_sentence_stem(sentence)
					name_counts[stem] = name_counts.get(stem, 0) + 1
					duplicate_suffix = "" if name_counts[stem] == 1 else f"_{name_counts[stem]}"
					output_path = output_dir / f"{stem}{duplicate_suffix}.npy"
					np.save(output_path, probabilities)
					saved_paths.append(output_path)

	print(f"Saved {len(saved_paths)} CTC arrays to {output_dir}")
	print(f"CTC label order: {bundle.get_labels()}")
	return saved_paths


if __name__ == "__main__":
	generate_librispeech_ctc()