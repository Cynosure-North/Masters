"""Parse the raw data from the experiment, apply the relevant processing for each model, and save it to that model's data dir"""

import hashlib
from hmac import digest_size
from pathlib import Path
import numpy as np
from calibration import calibrate_hand
import Pose.data as pose
import Contact.data as contact
import Motion.data as motion


raw_data_folder = Path(r"H:\Research_Data\Morgana North(MHIT)\phase1")
models_root = Path(__file__).resolve().parent 
# pose_data_dir = models_root / "Pose" / "data" / "training"
contact_data_dir = models_root / "contact" / "data" / "3d"
contact_data_dir_simple = models_root / "contact" / "data" / "3d-simple"
motion_data_dir = models_root / "Motion" / "data" / "3d"
motion_data_dir_raw = models_root / "Motion" / "data" / "3d-raw"
# data_dirs = [pose_data_dir, contact_data_dir, contact_data_dir_simple, motion_data_dir, motion_data_dir_raw]
data_dirs = [motion_data_dir, motion_data_dir_raw]
cache_dir = models_root / "_cache"
cache_dir.mkdir(exist_ok=True)

typo_ratios = {"table": [], "cushion": [], "pipe": [], "legs": []}
# Surface error averages -- table: 11.83% cushion: 13.38% pipe: 23.19% legs: 20.05% 

def file_generator(data_folder):
	for participant in sorted(data_folder.iterdir()):
		for surface in sorted(participant.iterdir()):

			if surface.is_file():
				continue

			for split in ["train", "validation", "test"]:
				for folder in data_dirs:
					(folder / split / surface.stem).mkdir(parents=True, exist_ok=True)


			calibration_files = set(surface.glob("calibration_[0-9]*"))
			sentence_files = [
				path for path in sorted(surface.iterdir())
				if path.is_file() and path not in calibration_files and "_typo" not in path.stem]


			typo_ratios[surface.stem].append(
				len(list(surface.glob("*_typo"))) / len(sentence_files))


			def parse(files, filename=True):
				out = []
				for f in files:
					record = eval(f.read_text(), {"__builtins__": {}, "array": np.array})
					# I collected the frame timestamp, but I'm not using it at the moment so only grab the frames
					frames = np.asarray([frame for _, frame in record])
					if frames.ndim < 2:
						# Catches 0-length records
						# print(f"Zero length record: {f}")
						continue

					if filename:
						out.append((f.stem, frames))
					else:
						out.append(frames)
				return out

			calibrations = parse(calibration_files, filename=False)
			sentences = parse(sentence_files)


			# filter outliers
			# There were incidents of people accidentally double triggering submit resulting in a <2 frame submission
			#  and also some situations in which people stopped to ask questions, chat, or take a break
			calibration_mean = sum([len(calibration) for calibration in calibrations])/len(calibrations)
			calibrations = [x for x in calibrations if (calibration_mean/5 < len(x) and len(x) < calibration_mean*3)]

			sentence_mean = sum([len(sentence[1]) for sentence in sentences])/len(sentences)
			sentences = [x for x in sentences if (sentence_mean/5 < len(x[1]) and len(x[1]) < sentence_mean*3)]

			yield calibrations, sentences, participant.stem, surface.stem

def preprocess():

	for calibrations, sentences, participant, surface in file_generator(raw_data_folder):

		calibration_cache_path = cache_dir / f"{participant}_{surface}.npy"
		if calibration_cache_path.exists():
			left_hand, right_hand = np.load(calibration_cache_path, allow_pickle=True)
			print(f"loaded saved_calibration {calibration_cache_path}")
		else:
			# Each hand is calibrated based on the fingertip positions
			try:
				left_hand_calibration, right_hand_calibration = \
					[x[:, :7] for x in calibrations], [x[:, 7:] for x in calibrations]
				left_hand, right_hand = \
					calibrate_hand(left_hand_calibration, surface), calibrate_hand(right_hand_calibration, surface)
				print(
					f"{participant}\t{surface} left_hand.shape: {type(left_hand.shape).__name__}"
					f"\t right_hand.shape: {type(right_hand.shape).__name__}")
				np.save(calibration_cache_path, [left_hand, right_hand])
			except:
				print(f"Error calibrating {participant}/{surface}")
				raise

		for sentence, frames in sentences:
			filename = f"{participant.split('_')[1]}_{sentence}"
			# pose.fit_scaler(frames)
			
			split = ["validation", "test", "validation", "test", "validation", "test",
			 "train", "train", "train", "train", "train"][
				 int(hashlib.sha256(filename.encode('utf-8')).hexdigest(), 16) % 10 ]
			output_file = Path(split, surface, f"{filename}.npy")

			if all([(folder / output_file).exists() for folder in data_dirs]):
				print(f"Skipping existing file: {output_file}")
				continue
				# It's fairly unlikely the code will get interrupted in between save statements
				# and if it does and one or two files aren't saved as training data it's not a huge
				# deal, so only checking one file is fine
			print(f"Processing {output_file}")

			contact_frames = contact.transform(frames, left_hand, right_hand)
			np.save(contact_data_dir / output_file, contact_frames)
			# Simplified version only has UV coordinates of contact points and hand rest position
			np.save(contact_data_dir_simple / output_file, contact_frames[:, :3])

			# np.save(motion_data_dir / output_file, motion.transform(frames, left_hand, right_hand))
			# np.save(motion_data_dir_raw / output_file, np.diff(frames, axis=0))

	# for _, sentences, _, _ in file_generator(raw_data_folder):
	# 	for _, frames in sentences:
	# 		pose.fit_pca(frames)

	# for _, sentences, participant, surface in file_generator(raw_data_folder):
	# 	for sentence, frames in sentences:
	#
	# 		filename = f"{participant.split('_')[1]}_{sentence}"
	#		split = ["validation", "test", "validation", "test", "validation", "test",
	#		 "train", "train", "train", "train", "train"][rand.randint(0,9)]
	#
	# 		np.save(pose.transform(frames), pose_data_dir / surface / split / filename)
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
