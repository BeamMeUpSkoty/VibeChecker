import parselmouth
import librosa
import numpy as np
from collections import ChainMap

class AudioFeatures(object):
	"""
	"""
	def __init__(self, path, languageCode):
		self.path = path
		self.languageCode = languageCode
		self.source_type = 'audio'
		self.sound = sound = parselmouth.Sound(self.path)


	def get_average_pitch(self, verbose=False):
		"""assumes that the WAV file is located in the same directory as the script and is named example.wav. 
		You may need to modify the script to match the location and name of your own WAV file. 
		Additionally, you will need to install Praat and the Parselmouth library to run this script.
		"""

		# Extract pitch contour using Praat's default settings
		pitch = self.sound.to_pitch()

		# Extract average pitch
		pitch_values = pitch.selected_array['frequency']
		average_pitch = sum(pitch_values) / len(pitch_values)

		# Print the average pitch
		if verbose:
			print(f"The average pitch is {average_pitch:.2f} Hz.")

		return {'average_pitch': average_pitch}


	def get_f0_statistics(self, verbose=False):
		"""
		"""
		print('=== HERE ===')
		# Load the audio file
		y, sr = librosa.load(self.path)

		# Extract the F0 contour using the Yin algorithm
		f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)

		# Filter out unvoiced frames
		f0 = f0[voiced_flag]

		# Extract the minimum F0
		min_f0 = f0.min()
		max_f0 = f0.max()
		mean_f0 = f0.mean()
		median_f0 = np.median(f0)
		sd_f0 = np.std(f0)
		range_f0 = max_f0 - min_f0

		# Calculate the 80th percentile of F0
		percentile_80_f0 = np.percentile(f0, 80)
		# Filter F0 values above the 80th percentile
		f0_above_percentile = f0[f0 > percentile_80_f0]
		# Calculate the range of F0 values above the 80th percentile
		range_80th_percentile_f0 = np.max(f0_above_percentile) - np.min(f0_above_percentile)

		# Print the F0 statistics
		if verbose:
			print(f"Minimum F0: {min_f0:.2f} Hz")
			print(f"Maximum F0: {max_f0:.2f} Hz")
			print(f"Mean F0: {mean_f0:.2f} Hz")
			print(f"Median F0: {median_f0:.2f} Hz")
			print(f"Standard deviation of F0: {sd_f0:.2f} Hz")
			print(f"80th percentile F0 range: {range_80th_percentile_f0:.2f} Hz")

		return {'min_f0':min_f0, 'max_f0':max_f0, 'mean_f0':mean_f0, 'median_f0':median_f0, 'sd_f0':sd_f0, 'f0_range':range_f0, '80_percentile_range':range_80th_percentile_f0}


	def get_articulation_rate(self, verbose=False):
		"""
		"""
		# Extract textgrid object using Praat's default settings
		tg = self.sound.to_textgrid()

		# Extract the total duration of the sound file
		total_duration = sound.get_total_duration()

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


	def get_cumulative_speech_index(self, verbose=False):
		"""
		"""
		# Extract pitch object using Praat's default settings
		pitch = self.sound.to_pitch()

		# Initialize variables for the cumulative slope index
		csi = 0
		previous_pitch = None

		# Loop through frames in the pitch object
		for frame in pitch:
			# Skip frames where the pitch is undefined
			if frame.x == 0 or frame.xmax == 0:
				continue
			
			# If this is not the first frame, calculate the slope and add it to the CSI
			if previous_pitch is not None:
				slope = (frame.frequency - previous_pitch.frequency) / (frame.xmax - previous_pitch.xmax)
				csi += abs(slope)
			
			# Save the current frame as the previous frame for the next iteration
			previous_pitch = frame

		if verbose:
			# Print the cumulative slope index
			print(f"Cumulative slope index: {csi:.2f}")

		return {'csi':csi}


	def get_vocal_intensity_statistics(self, verbose=False):
		"""
		"""

		# Extract intensity object using Praat's default settings
		intensity = self.sound.to_intensity()

		# Calculate various statistics of intensity
		intensity_min = parselmouth.praat.call(intensity, "Get minimum", 0, 0, "Hertz")
		intensity_max = parselmouth.praat.call(intensity, "Get maximum", 0, 0, "Hertz")
		intensity_mean = parselmouth.praat.call(intensity, "Get mean", 0, 0, "Hertz")
		intensity_stddev = parselmouth.praat.call(intensity, "Get standard deviation", 0, 0, "Hertz")
		intensity_quartiles = parselmouth.praat.call(intensity, "Get quantiles", 0, 0, "Hertz", 4)

		if verbose:
			# Print the statistics of intensity
			print(f"Minimum intensity: {intensity_min:.2f} dB")
			print(f"Maximum intensity: {intensity_max:.2f} dB")
			print(f"Mean intensity: {intensity_mean:.2f} dB")
			print(f"Standard deviation of intensity: {intensity_stddev:.2f} dB")
			print(f"25th percentile intensity: {intensity_quartiles[0]:.2f} dB")
			print(f"50th percentile intensity: {intensity_quartiles[1]:.2f} dB")
			print(f"75th percentile intensity: {intensity_quartiles[2]:.2f} dB")
			print(f"100th percentile intensity: {intensity_quartiles[3]:.2f} dB")

		return {'min_intensity':intensity_min, 'max_intensity':intensity_max, 'mean intensity':intensity_mean, 'sd_intensity':intensity_stddev, '25_intensity':intensity_quartiles[0], '50_intensity':intensity_quartiles[1], '75_intensity':intensity_quartiles[2], '100_intensity':intensity_quartiles[3]}


	def get_all_features(self, verbose=True):
		"""
		"""
		average_pitch = self.get_average_pitch(verbose=verbose)
		#f0_statistics = self.get_f0_statistics(verbose=verbose)
		#articulation_rate = self.get_articulation_rate(verbose=verbose)
		#csi = self.get_cumulative_speech_index(verbose=verbose)
		#vocal_intensity = self.get_vocal_intensity_statistics(verbose=verbose)

		return average_pitch


