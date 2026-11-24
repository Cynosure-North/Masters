"""Parse the raw data from the experiment, apply the relevant processing for each model, and save it to that model's data dir"""

from collections import defaultdict
import hashlib
from hmac import digest_size
from pathlib import Path
import numpy as np
from calibration import calibrate_hand
import Pose.data as pose
import Contact.data as contact
import Motion.data as motion
from collections import defaultdict


raw_data_folder = Path(r"H:\Research_Data\Morgana North(MHIT)\phase1")
models_root = Path(__file__).resolve().parent 
# pose_data_dir = models_root / "Pose" / "data" / "training"
contact_data_dir = models_root / "contact" / "data" / "3d"
contact_data_dir_simple = models_root / "contact" / "data" / "3d-simple"
motion_data_dir = models_root / "Motion" / "data" / "3d"
motion_data_dir_raw = models_root / "Motion" / "data" / "3d-raw"
# data_dirs = [pose_data_dir, contact_data_dir, contact_data_dir_simple, motion_data_dir, motion_data_dir_raw]
data_dirs = [motion_data_dir_raw]
cache_dir = models_root / "_cache"
cache_dir.mkdir(exist_ok=True)

typo_ratios = defaultdict(lambda: defaultdict(list))
# Surface error averages -- table: 11.83% cushion: 13.38% pipe: 23.19% legs: 20.05% 

def file_generator(data_folder):
	for participant in sorted(data_folder.iterdir()):

		
		if participant.name not in ["session_62a08473", "session_79fedad0", "session_85ad194a", "session_582f5770", "session_874f6161", "session_2474e5e5", "session_bdd5e9a9", "session_f81263ce"]:
			continue

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


			typo_ratios[participant.stem][surface.stem].append(
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

					frames = np.delete(frames, [0, 1, 7, 8], axis=1)

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

			# contact_frames = contact.transform(frames, left_hand, right_hand)
			# np.save(contact_data_dir / output_file, contact_frames)
			# # Simplified version only has UV coordinates of contact points and hand rest position
			# np.save(contact_data_dir_simple / output_file, contact_frames[:, :4])

			# np.save(motion_data_dir / output_file, motion.transform(frames, left_hand, right_hand))
			np.save(motion_data_dir_raw / output_file, np.diff(frames, axis=0))

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

	# participant_avg = defaultdict(list)
	# surface_avg = defaultdict(list)

	# for participant, surface in typo_ratios.items():
	# 	for vals in surface.values():
	# 		surface_avg[surface].append(vals)
	# 		participant_avg[surface].append(vals)
	
	# message = "Surface error averages -- "
	# for k, v in surface_avg:
	# 	message += f"{k}: {sum(v) / len(v):.2%} "
	# print(message)

	# message = "Participant error averages -- "
	# for k, v in participant_avg:
	# 	message += f"{k}: {sum(v) / len(v):.2%} "
	# print(message)


if __name__ == "__main__":
	preprocess()

# Not proportions
# 'session_2474e5e5': 'cushion': [0.0666],	'legs': [0.0],		'pipe': [0.1428],	'table': [0.0714]	= 0.2808
# 'session_32d94b77': 'cushion': [0.1470],	'legs': [0.2121],	'pipe': [0.1129],	'table': [0.125]	= 0.5970
# 'session_4931439e': 'cushion': [0.0408],	'legs': [0.0222],	'pipe': [0.1764],	'table': [0.0769]	= 0.3163
# 'session_582f5770': 'cushion': [0.0],		'legs': [0.0909],	'pipe': [0.0],		'table': [0.0652]	= 0.1561
# 'session_62a08473': 'cushion': [0.0952],	'legs': [0.0270],	'pipe': [0.3428],	'table': [0.0]		= 0.4650
# 'session_70617e90': 'cushion': [0.1616],	'legs': [0.2048],	'pipe': [0.17],		'table': [0.1931]	= 0.7295
# 'session_75a19362': 'cushion': [0.0666],	'legs': [0.0178],	'pipe': [0.2203],	'table': [0.06]		= 0.3647
# 'session_79fedad0': 'cushion': [0.2261],	'legs': [0.1875],	'pipe': [0.4081],	'table': [0.1475]	= 0.9692
# 'session_85ad194a': 'cushion': [0.3720],	'legs': [1.3333],	'pipe': [0.26],		'table': [0.2549]	= 2.2202
# 'session_874f6161': 'cushion': [0.1363],	'legs': [0.1176],	'pipe': [0.1052],	'table': [0.0806]	= 0.4397
# 'session_a2598998': 'cushion': [0.7647],	'legs': [0.0740],	'pipe': [0.1333],	'table': [1.0344]	= 2.0064
# 'session_bdd5e9a9': 'cushion': [0.1176],	'legs': [0.0483],	'pipe': [0.0],		'table': [0.0612]	= 0.2271
# 'session_c562cfdc': 'cushion': [0.0],		'legs': [0.0937		'pipe': [0.0222],	'table': [0.0]		= 0.1159
# 'session_f81263ce': 'cushion': [0.125],	'legs': [0.05],		'pipe': [0.6],		'table': [0.2777]	= 1.0527
# 'session_fce7566b': 'cushion': [0.1707],	'legs': [0.0526],	'pipe': [0.5],		'table': [0.1489]	= 0.8722


# cushion	.1660
# 0.0666 + 0.1470 + 0.0408 + 0.0 + 0.0952 + 0.1616 + 0.0666 + 0.2261 + 0.3720 + 0.1363 + 0.7647 + 0.1176 + 0.0 + 0.125 + 0.1707

# legs	.1688
# 0.0 + 0.2121 + 0.0222 + 0.0909 + 0.0270 + 0.2048 + 0.0178 + 0.1875 + 1.3333 + 0.1176 + 0.0740 + 0.0483 + 0.0937 + 0.05 + 0.0526

# pipe	.2129
# 0.1428 + 0.1129 + 0.1764 + 0.0 + 0.3428 + 0.17 + 0.2203 + 0.4081 + 0.26 + 0.1052 + 0.1333 + 0.0 + 0.0222 + 0.6 + 0.5

# table	.1731
# 0.0714 + 0.125 + 0.0769 + 0.0652 + 0.0 + 0.1931 + 0.06 + 0.1475 + 0.2549 + 0.0806 + 1.0344 + 0.0612 + 0.0 + 0.2777 + 0.1489


# session_32d94b77 - allegedly the wrists are four metres from the fingertips
# session_62a08473   wrist vectors point quite steeply up
# session_75a19362 - allegedly the wrists are four metres from the fingertips
# session_79fedad0   wrist vectors point quite steeply up
# session_85ad194a   wrist vectors point quite steeply up
# session_582f5770   plane not flat
# session_874f6161   plane not flat, wrist vectors in plane
# session_2474e5e5   wrist vectors point akimbo
# session_70617e90 - allegedly the wrists are four metres from the fingertips
# session_4931439e - allegedly the wrists are four metres from the fingertips
# session_a2598998 - allegedly the wrists are four metres from the fingertips
# session_bdd5e9a9   no right wrist vector
# session_c562cfdc - allegedly the wrists are four metres from the fingertips
# session_f81263ce - wrist vectors point about 45deg off to the left
# session_fce7566b - wrists are facing in opposite directions and frequent wrist vector dropouts
#
# session_32d94b77 - all points in a line
# session_62a08473 
# session_75a19362 - all points in a line
# session_79fedad0 
# session_85ad194a 
# session_582f5770 
# session_874f6161 
# session_2474e5e5 
# session_70617e90 - all points in a line
# session_4931439e - all points in a line
# session_a2598998 - all points in a line
# session_bdd5e9a9
# session_c562cfdc - all points in a line
# session_f81263ce
# session_fce7566b - left hand flipped upside down
#