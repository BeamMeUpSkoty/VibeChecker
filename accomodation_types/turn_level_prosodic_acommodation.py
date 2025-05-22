from audio_features.audio_features import AudioFeatures
from data_types.audio_file import AudioFile
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os

class TurnLevelProsodicAcommodation(object):
	'''
	class containing all turn-level prosocdic accomodation features as described in 
	"Prosodic Accomodation in Face-to-Face and Telephone Dialogues" by Pavel Sturm, Radek Skarnitzl,
	Tomas Nechansky
	'''
	def __init__(self, audio_path, diarization_path, language_code='fr', outpath=''):
		"""
		Parameters
		-------------
		audio_path : string
			path to audio file
		diarization : string
			path to speaker diarization file
		langauge : string 
			two letter code indicating language
		outpath : string
			path where output will be saved
		"""

		self.audio_file = AudioFile(audio_path, language_code) 
		self.diarization = pd.read_csv(diarization_path)
		self.language_code = language_code
		self.outpath = outpath
		self.accomodation_features = self.get_accomodation()


	def get_audio_features(self):
		""" iterates through speech turns. Computes features for each speech turn.
		
		Returns
		---------
		speech_turn_features : list of dictionaries
			list of features that map to speech turns
		"""
		#create tmp/ file to hold audio chunks
		self.audio_file.remove_tmp_file()
		self.audio_file.create_tmp_file()

		#list to hold features
		speech_turn_features = []

		#iterate through dataframe of speech turns
		for speech_turn in self.diarization.iterrows():
			start_time = speech_turn[1]['start']
			end_time = speech_turn[1]['end']
			speaker = speech_turn[1]['speaker']

			#create audio chunks for each utterance in the transcript.
			chunk_path = self.audio_file.make_audio_chunk_by_utterance(start_time, end_time)

			#extract features from audio chunk
			AF = AudioFeatures(chunk_path, self.language_code)
			features = AF.get_all_features(verbose=False)

			#add start, end, and speaker to utternace features dictionaries
			features['start'] = start_time
			features['end'] = end_time
			features['speaker'] = speaker

			#append feature dictionary to list
			speech_turn_features.append(features)

		#remove audio chunks and tmp/ file	
		self.audio_file.remove_tmp_file()

		return speech_turn_features


	def get_speaker_distance(self, features):
		""" add difference features to accomodation features dataframe.
		
		parameters
		---------
		features : dict()
			dictionary of features made in get_audio_features method

		Returns
		---------
		df : pandas dataframe
			pandas dataframe with the 
		"""
		df = pd.DataFrame(features)
		df['pitch_distance'] = df['average_pitch'] - df['average_pitch'].shift(-1)
		return df


	def get_accomodation(self):
		""" extracts features from audio file. Calculates turn-level speaker distances.

		Returns
		---------
		df : pandas dataframe
			pandas dataframe with the 
		"""
		features = self.get_audio_features()
		#features['phrase_index'] = features.index
		distance_features = self.get_speaker_distance(features)
		distance_features = distance_features.dropna()
		return distance_features


	def get_convergence(self):
		""" get convergence. Pearson correlation be speaker distance and time(i.e. prosodic
		phrase index/order within the interaction). If speakers converge, the distances
		should be increasingly smaller with time. This can be captured in an LME model by 
		using phrase index (proxy for time) as an indicator. Statistical significance of 
		the predictor would be indicate convergence (if negative).

		lme model from Sturm paper: speaker_distance ~ phrase_index + condition + session + (1+condition|speaker_pair)
	
		condition (factor): F2F, video call
		session (factor): session 1, session 2,...
		phrase_index (continuous): 
		Preceding Phrase Value (Continuous): 
		Return
		----------
		convergence : xxx
			pearson correlation between speaker distance and phrase index
		"""
		correlation = pearsonr(np.array(self.accomodation_features['pitch_distance']), np.array(self.accomodation_features.index)) 
		statistic = correlation[0]
		pvalue = correlation[1]
		return statistic, pvalue


	def get_synchrony(self):
		"""
		lme model from Sturm paper: turn-initial value ~ preceding phrase value + condition + session + (1+condition|speaker_pair) + (1+condition|speaker)
		
		Returns
		---------
		correlation : pearsonRResult

		"""
		SPEAKER1 = self.accomodation_features.loc[self.accomodation_features['speaker'] == 'SPEAKER 1']
		SPEAKER2 = self.accomodation_features.loc[self.accomodation_features['speaker'] == 'SPEAKER 2']

		sp1 = np.array(SPEAKER2['average_pitch'])
		sp2 = np.array(SPEAKER1['average_pitch'])

		if len(sp1) != len(sp2):
			if len(sp1) > len(sp2):
				sp1 = sp1[:-1]
			else:
				sp2 = sp2[:-1]

		correlation = pearsonr(sp1, sp2)
		statistic = correlation[0]
		pvalue = correlation[1]
		return statistic, pvalue
