"""Parse the raw data from the experiment, apply the relevant processing for each model, and save it to that model's data dir"""

import ast
from pathlib import Path
import numpy as np
from fit_shape import calibrate_hand
import Pose.data as pose
import Contact.data as contact
import Motion.data as motion

def preprocess():
	raw_data_folder = Path(r"H:\Research_Data\Morgana North(MHIT)\phase1")
	models_root = Path(__file__).resolve().parent 
	pose_data_dir = models_root / "Pose" / "data" / "training"
	contact_data_dir = models_root / "contact" / "data" / "training"
	motion_data_dir = models_root / "Motion" / "data" / "training"

	typo_ratios = {"table": [], "cushion": [], "pipe": [], "legs": []}

	def file_generator(data_folder):
		for participant in data_folder.iterdir():
			for surface in participant.iterdir():

				if surface.is_file():
					continue

				(pose_data_dir / surface).mkdir(parents=True, exist_ok=True)
				(contact_data_dir / surface).mkdir(parents=True, exist_ok=True)
				(motion_data_dir / surface).mkdir(parents=True, exist_ok=True)


				calibration_files = surface.glob("calibration_[0-9].pth")
				sentence_files = list(set(surface.glob("*")) - set(calibration_files))


				typo_ratios[surface].append(len(list(surface.glob("*_typo.pth"))) / len(sentence_files))


				# parse data files
				calibrations = []
				for c in calibration_files:
					calibrations.append(ast.literal_eval(c.read_text()))	# Text files were probably not the best decision

				sentences = []
				for s in sentence_files:
					# I collected the frame timestamp, but I'm not using it at the moment so only grab the frames
					sentences.append((s.stem, ast.literal_eval(s.read_text()[1])))


				# filter outliers
				# There were incidents of people accidentally double triggering submit resulting in a <2 frame submission
				#  and also some situations in which people stopped to ask questions, chat, or take a break
				calibration_mean = sum([len(calibration) for calibration in calibrations])/len(calibrations)
				calibrations = [x for x in calibrations if (calibration_mean/5 < x and x < calibration_mean*3)]

				sentence_mean = sum([len(sentence[1]) for sentence in sentences])/len(sentences)
				sentences = [x for x in sentences if (sentence_mean/5 < x and x < sentence_mean*3)]

				yield calibrations, sentences, participant, surface

	for calibrations, sentences, participant, surface in file_generator(raw_data_folder):

		calibrations = np.asarray(calibrations)

		left_hand, right_hand = calibrate_hand(calibrations[:, 2:7]), calibrate_hand(calibrations[:, 9:14])

		for sentence, frames in sentences:
			filename = f"{participant.split('_')[1]}_{sentence}"
			pose.fit_scaler(frames)
			np.save(contact.transform(frames, left_hand, right_hand), contact_data_dir / surface / filename)
			np.save(motion.transform(frames, left_hand, right_hand), motion_data_dir / surface / filename)

	for _, sentences, _, _ in file_generator(raw_data_folder):
		for _, frames in sentences:
			pose.fit_pca(frames)

	for _, sentences, participant, surface in file_generator(raw_data_folder):
		for sentence, frames in sentences:
			filename = f"{participant.split('_')[1]}_{sentence}"
			np.save(pose.transform(frames), pose_data_dir / surface / filename)
			# Scaler and PCA need to be fit on the full training set before transforming the data

	# Save scaler and PCA values with the data
	preprocess_checkpoint = pose.get_preprocessor_state()
	np.save(preprocess_checkpoint, pose_data_dir / "preprocessor_params")


	for k, v in typo_ratios.items():
		print(f"Surface: {k}, ratio: {v}")


if __name__ == "__main__":
	preprocess()