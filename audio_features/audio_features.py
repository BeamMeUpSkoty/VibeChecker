import parselmouth
import librosa
import numpy as np
from collections import ChainMap
from typing import Dict

class AudioFeatures(object):
	""" Extract acoustic features from an audio file

	Praat Manual: https://www.fon.hum.uva.nl/praat/manual/Query_submenu.html
	"""
	def __init__(self, path: str, languageCode: str):
		self.path: str = path
		print(self.path)
		self.languageCode: str = languageCode
		self.source_type: str = 'audio'
		self.sound: parselmouth.Sound = parselmouth.Sound(self.path)


	def get_average_pitch(self, verbose: bool = False) -> Dict[str, float]:
		"""assumes that the WAV file is located in the same directory as the script and is named example.wav. 
		You may need to modify the script to match the location and name of your own WAV file. 
		Additionally, you will need to install Praat and the Parselmouth library to run this script.
		"""

		# Extract pitch contour using Praat's default settings
		pitch = self.sound.to_pitch()

		# Extract average pitch
		pitch_values = pitch.selected_array['frequency']
		# remove unvoiced frames
		pitch_values = pitch_values[pitch_values > 0]

		average_pitch = float(np.mean(pitch_values)) if len(pitch_values) else 0.0

		if verbose:
			print(f"The average pitch is {average_pitch:.2f} Hz.")
		return {'average_pitch': average_pitch}


	def get_f0_statistics(self, verbose: bool =False):
		"""
		"""
		# Load the audio file
		y, sr = librosa.load(self.path)

		# Extract the F0 contour using the Yin algorithm
		f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)

		if f0 is None or not voiced_flag.any():
			return {}

		# Filter out unvoiced frames
		f0 = f0[voiced_flag]

		stats = {
			'min_f0': float(np.min(f0)),
			'max_f0': float(np.max(f0)),
			'mean_f0': float(np.mean(f0)),
			'median_f0': float(np.median(f0)),
			'sd_f0': float(np.std(f0)),
			'f0_range': float(np.max(f0) - np.min(f0)),
		}
		# Calculate the 80th percentile of F0
		percentile_80 = np.percentile(f0, 80)
		# Filter F0 values above the 80th percentile
		high_f0 = f0[f0 > percentile_80]
		# Calculate the range of F0 values above the 80th percentile
		stats['80_percentile_range'] = float(np.max(high_f0) - np.min(high_f0)) if len(high_f0) > 0 else 0.0

		if verbose:
			for k, v in stats.items():
				print(f"{k}: {v:.2f} Hz")

		return stats

	def get_articulation_rate(self, verbose: bool = False) -> Dict[str, float]:
		"""
		"""
		# Extract textgrid object using Praat's default settings
		tg = self.sound.to_textgrid()

		# Extract the total duration of the sound file
		total_duration = self.sound.get_total_duration()

		# Initialize variables for syllable count and pause duration
		syllable_count = 0
		pause_duration = 0

		# Loop through intervals in the textgrid
		for i in range(1, tg.get_number_of_intervals()+1):
			interval = tg.get_interval(i)
			
			# Skip any intervals that are not labeled as "word"
			if interval[2] != "word":
				continue
			
			# Extract the start and end times of the interval
			start_time = interval[0]
			end_time = interval[1]
			
			# Calculate the duration of the interval
			duration = end_time - start_time
			
			# If the duration is less than or equal to 0, skip it
			if duration <= 0:
				continue
			
			# If there was a pause before this interval, add it to the pause duration
			if pause_duration > 0:
				total_duration -= pause_duration
				pause_duration = 0
			
			# Calculate the number of syllables in the interval and add it to the syllable count
			num_syllables = len(interval[3].split("-"))
			syllable_count += num_syllables
			
			# Subtract the duration of the interval from the total duration
			total_duration -= duration
			
			# If there is a pause after this interval, add it to the pause duration
			next_interval = tg.get_interval(i+1)
			if next_interval[0] - end_time > 0:
				pause_duration += next_interval[0] - end_time

		# Calculate the syllables per second
		syllables_per_second = syllable_count / total_duration

		if verbose:
			# Print the syllables per second
			print(f"Syllables per second: {syllables_per_second:.2f}")

		return {'syllables_per_second':syllables_per_second}

	def get_cumulative_speech_index(self, verbose: bool = False) -> Dict[str, float]:
		"""
        Computes the cumulative speech index (CSI) from the pitch contour.

        Returns
        -------
        Dict[str, float]
            Dictionary with key 'csi'.
        """
		pitch = self.sound.to_pitch()
		pitch_values = pitch.selected_array['frequency']
		pitch_values = pitch_values[pitch_values > 0]  # voiced frames only

		time_step = pitch.get_time_step()
		csi = 0.0

		for i in range(1, len(pitch_values)):
			slope = (pitch_values[i] - pitch_values[i - 1]) / time_step
			csi += abs(slope)

		if verbose:
			print(f"Cumulative Speech Index (CSI): {csi:.2f}")

		return {'csi': csi}

	def get_vocal_intensity_statistics(self, verbose: bool = False) -> Dict[str, float]:
		"""
        Extracts basic intensity statistics from the entire file using Praat’s Intensity object.
        """

		# 1. Turn the Sound into an Intensity object (default window & time step)
		intensity: parselmouth.Intensity = self.sound.to_intensity()

		# 2. “Get minimum” / “Get maximum” require an interpolation method, not “dB”.
		#    Here we use “Parabolic” (you can also choose “Cubic” or “None”).
		min_intensity = parselmouth.praat.call(
			intensity, "Get minimum", 0.0, 0.0, "Parabolic"
		)
		max_intensity = parselmouth.praat.call(
			intensity, "Get maximum", 0.0, 0.0, "Parabolic"
		)

		# 3. “Get mean” takes an averaging method; passing “dB” returns the mean in dB.
		mean_intensity = parselmouth.praat.call(
			intensity, "Get mean", 0.0, 0.0, "dB"
		)

		# 4. “Get standard deviation” only takes the time‐range (two floats).
		#    It always returns dB‐based standard deviation.
		sd_intensity = parselmouth.praat.call(
			intensity, "Get standard deviation", 0.0, 0.0
		)

		# 6. Cast each to float and build the stats dict
		stats = {
			"min_intensity": float(min_intensity),
			"max_intensity": float(max_intensity),
			"mean_intensity": float(mean_intensity),
			"sd_intensity": float(sd_intensity),
		}

		if verbose:
			for name, value in stats.items():
				print(f"{name}: {value:.2f} dB")

		return stats

	def get_all_features(self, verbose: bool = True) -> Dict[str, float]:
		"""
		Combines all extracted features into a single dictionary.
		"""
		feature_dicts = [
			self.get_average_pitch(verbose),
			self.get_f0_statistics(verbose),
			self.get_cumulative_speech_index(verbose),
			self.get_vocal_intensity_statistics(verbose),
		]
		merged = dict(ChainMap(*feature_dicts))
		return merged


