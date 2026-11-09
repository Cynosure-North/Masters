"""Parse the raw data from the experiment, apply the relevant processing for each model, and save it to that model's data dir"""

from pathlib import Path
import numpy as np
from fit_shape import calibrate_hand
import Pose.data as pose
import Contact.data as contact
import Motion.data as motion

raw_data_folder = Path(r"H:\Research_Data\Morgana North(MHIT)\phase1")
models_root = Path(__file__).resolve().parent 
pose_data_dir = models_root / "Pose" / "data" / "training"
contact_data_dir = models_root / "contact" / "data" / "training"
contact_data_dir_simple = models_root / "contact" / "data" / "training-simple"
motion_data_dir = models_root / "Motion" / "data" / "training"
motion_data_dir_raw = models_root / "Motion" / "data" / "training-raw"

typo_ratios = {"table": [], "cushion": [], "pipe": [], "legs": []}

def file_generator(data_folder):
	for participant in data_folder.iterdir():
		for surface in participant.iterdir():

			if surface.is_file():
				continue

			# (pose_data_dir / surface.stem).mkdir(parents=True, exist_ok=True)
			# (contact_data_dir / surface.stem).mkdir(parents=True, exist_ok=True)
			# (contact_data_dir_simple / surface.stem).mkdir(parents=True, exist_ok=True)
			# (motion_data_dir / surface.stem).mkdir(parents=True, exist_ok=True)
			(motion_data_dir_raw / surface.stem).mkdir(parents=True, exist_ok=True)


			calibration_files = set(surface.glob("calibration_[0-9]*"))
			sentence_files = [
				path for path in surface.iterdir()
				if path.is_file() and path not in calibration_files ]


			typo_ratios[surface.stem].append(
				len(list(surface.glob("*_typo"))) / len(sentence_files))


			def parse(files, filename=True):
				out = []
				for f in files:
					record = eval(f.read_text(), {"__builtins__": {}, "array": np.array})
					frames = np.asarray([frame for _, frame in record])
					if frames.ndim < 2:
						# Catches 0-length records
						print(f"Error on file {f}")
						continue

					# I collected the frame timestamp, but I'm not using it at the moment so only grab the frames
					if filename:
						out.append((f.stem, frames[:, 1]))
					else:
						out.append(frames[:, 1])
				return np.add

			calibrations = parse(calibration_files, filename=False)
			sentences = parse(sentence_files)


			# filter outliers
			# There were incidents of people accidentally double triggering submit resulting in a <2 frame submission
			#  and also some situations in which people stopped to ask questions, chat, or take a break
			calibration_mean = sum([len(calibration) for calibration in calibrations])/len(calibrations)
			calibrations = [x for x in calibrations if (calibration_mean/5 < x and x < calibration_mean*3)]

			sentence_mean = sum([len(sentence[1]) for sentence in sentences])/len(sentences)
			sentences = [x for x in sentences if (sentence_mean/5 < len(x[1]) and len(x[1]) < sentence_mean*3)]

			yield calibrations, sentences, participant, surface

def preprocess():

	for calibrations, sentences, participant, surface in file_generator(raw_data_folder):

		calibrations = np.asarray(calibrations)

		# Each hand is calibrated based on the fingertip positions
		left_hand, right_hand = calibrate_hand(calibrations[:, 2:7]), calibrate_hand(calibrations[:, 9:14])

		for sentence, frames in sentences:
			filename = f"{participant.stem.split('_')[1]}_{sentence}"
			# pose.fit_scaler(frames)

			contact_frames = contact.transform(frames, left_hand, right_hand)
			np.save(contact_data_dir / surface.stem / filename, contact_frames)
			# Simplified version only has UV coordinates of contact points and hand rest position
			np.save(contact_data_dir_simple / surface.stem / filename, contact_frames[:, :3])

			np.save(motion_data_dir / surface.stem / filename, motion.transform(frames, left_hand, right_hand))
			np.save(motion_data_dir_raw / surface.stem / filename, np.diff(frames))

	# for _, sentences, _, _ in file_generator(raw_data_folder):
	# 	for _, frames in sentences:
	# 		pose.fit_pca(frames)

	# for _, sentences, participant, surface in file_generator(raw_data_folder):
	# 	for sentence, frames in sentences:
	# 		filename = f"{participant.stem.split('_')[1]}_{sentence}"
	# 		np.save(pose.transform(frames), pose_data_dir / surface / filename)
	# 		# Scaler and PCA need to be fit on the full training set before transforming the data

	# Save scaler and PCA values with the data
	# preprocess_checkpoint = pose.get_preprocessor_state()
	# np.save(preprocess_checkpoint, pose_data_dir / "preprocessor_params")


	message = "Surface error averages -- "
	for k, v in typo_ratios.items():
		message += f"{k}: {sum(v) / len(v):.2%} "
	print(message)


if __name__ == "__main__":
	preprocess()