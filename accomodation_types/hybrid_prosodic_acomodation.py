import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict

import matplotlib.pyplot as plt

from data_types.audio_file import AudioFile
from data_types.transcript_file import TranscriptFile

from audio_features.audio_features import AudioFeatures
from audio_features.my_voice_analysis_features import MySpeechAnalysisFeatures

class HYBRIDProsodicAcommodation(object):
	'''
	class containing all prosocdic accomodation features
	'''
	def __init__(self, audio_path:str, diarization_path:str, language_code:str, window_size:int=5, step_size:int=2):
		self.audio_file = AudioFile(audio_path, language_code) 
		self.transcript = TranscriptFile(diarization_path, language_code)
		self.diarization_file = self.transcript.transcript
		self.language_code = language_code
		#self.outpath = outpath
		self.window_size = window_size
		self.step_size = step_size

		self.speaker_1 = self.diarization_file[self.diarization_file["speaker"] == 'SPEAKER 1']
		self.speaker_2 = self.diarization_file[self.diarization_file["speaker"] == 'SPEAKER 2']
		
		#self.accomodation_features = self.get_accomodation()


	def get_features(self, d:dict, verbose:bool=True) -> dict:
		""" Extracts acoustic features for a diarized window.

        Parameters
        ----------
        d : dict
            Dictionary with 'start' and 'end' times.

        Returns
        -------
        dict
            Feature dictionary including start and end times.
		"""

		#create audio chunks for each utterance in the transcript.
		chunk_path = self.audio_file.make_audio_chunk_by_utterance(d['start'], d['end'])

		#extract features from audio chunk
		#AF = AudioFeatures(chunk_path, self.language_code)
		#features = AF.get_all_mysp_features(verbose=verbose)

		MSAF = MySpeechAnalysisFeatures(chunk_path, 'audio', self.language_code)
		#features = MSAF.get_my_voice_analysis_features()
		raw_features = MSAF.get_my_voice_analysis_features()
		features = {}

		#resolve ~the NAN~ issue
		for k, v in raw_features.items():
			try:
				features[k] = float(v)
			except:
				features[k] = np.nan

		#add start, end, and speaker to utternace features dictionaries
		features['start'] = d['start']
		features['end'] =  d['end']
		return features		


	def get_features_by_speaker(self) -> tuple[list[dict], list[dict]]:
		""" creates a sliding window with fixed window size and step size. Adjusts
		the window 
		"""

		#create tmp/ file to hold audio chunks
		self.audio_file.remove_tmp_file()
		self.audio_file.create_tmp_file()

		#list to hold features
		speaker1_hybrid_features = []
		speaker2_hybrid_features = []

		speaker1_diarize_friendly_windows = self.audio_file.make_diar_friendly_sliding_window(self.speaker_1, duration=self.window_size, step_size=self.step_size)
		speaker2_diarize_friendly_windows = self.audio_file.make_diar_friendly_sliding_window(self.speaker_2, duration=self.window_size, step_size=self.step_size)

		for d in speaker1_diarize_friendly_windows:
			#fix this
			if d['start'] != None:
				speaker1_hybrid_features.append(self.get_features(d))
			#print(d)
		for e in speaker2_diarize_friendly_windows:

			speaker2_hybrid_features.append(self.get_features(e))
			#print(e)

		#remove audio chunks and tmp/ file	
		self.audio_file.remove_tmp_file()

		return speaker1_hybrid_features, speaker2_hybrid_features


	def sliding_window_correlation(self, speaker1:list[dict], speaker2:list[dict], features:list[str],
								   window_size:int=50, step_size:int=10) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
		""" default parameters are based on (De Looze, 2014). 
			window_size=5
			step_size=1

		Parameters
		------------
		speaker1 : xxx
			xxx
		speaker2 : xxx
			xxx
		features : list of strings
			contains names of features that match columns in features dataframe
		window_size : int
			default 10
		step_size : int
			default 5

		Returns
		-----------
		speaker1_df : pandas dataframe
			xxx
		speaker2_df : pandas dataframe
			xxx
		correlations : list of floats
			list of correlation values for each window
		"""
		#TODO: check speaker length; if len(spk_1) == len(spk_2):
		speaker1_df = pd.DataFrame(speaker1)
		speaker2_df = pd.DataFrame(speaker2)

		valid_features = ['mean_F0', 'median_F0', 'sd_F0', 'mean_intensity', 'sd_intensity', 'speech_rate', 'jitter',
						  'shimmer']
		features = [f for f in features if
					f in valid_features and f in speaker1_df.columns and f in speaker2_df.columns]

		# Calculate the Pearson correlation for each window
		correlations = defaultdict(list)
		significance = defaultdict(list)

		#iterate through specified feataures
		for feature in features:

			#subset dataframe by feature
			speaker1_feature = speaker1_df[feature]
			speaker2_feature = speaker2_df[feature]

			#offset window size for visualizaion
			for i in range(0, len(speaker1_df) - window_size + 1, step_size):
				x_window = speaker1_feature.iloc[i:i + window_size]
				y_window = speaker2_feature.iloc[i:i + window_size]

				# Try converting and checking
				try:
					x_window = x_window.astype(float)
					y_window = y_window.astype(float)

					if x_window.isnull().any() or y_window.isnull().any():
						raise ValueError("NaNs present")
					if x_window.nunique() <= 1 or y_window.nunique() <= 1:
						raise ValueError("No variation")

					corr, p = pearsonr(x_window, y_window)
					correlations[feature].append(corr)
					significance[feature].append(p)

				except Exception as e:
					correlations[feature].append(np.nan)
					significance[feature].append(np.nan)

		# return list of correlation values for each window
		return speaker1_df, speaker2_df, correlations, significance
	

	def get_convergence(self):
		"""
		"""
		return

	def get_synchrony(self):
		"""
		"""
		return

	def get_visualization(self, data_spk1, data_spk2, accomodation):
		"""
		"""
		#keys = ['speech_rate', 'sd_F0', 'median_F0', 'median_intensity', 'sd_intensity']
		keys = ['mean_F0']
		fig, axs = plt.subplots(len(keys), sharex=True, figsize=(5, 5))
		if len(keys) == 1:
			axs = [axs]  # wrap single Axes object in a list

		for i,column in enumerate(keys):
			#X_male = [j[column] for j in data_spk1]
			#X_female = [j[column] for j in data_spk2]
			X_male = data_spk1[column].to_numpy()
			X_female = data_spk2[column].to_numpy()

			# accomodation = accomodation[column][1:]
			axs[i].plot(X_male, 'blue', label='spk1')
			axs[i].plot(X_female, 'orange', label='spk2')
			ax2 = axs[i].twinx()
			ax2.plot(accomodation[column], 'r--', label='accomodation')
			ax2.set_ylim([-1, 1])
			# break
			axs[i].set_title(column)
		plt.show()
