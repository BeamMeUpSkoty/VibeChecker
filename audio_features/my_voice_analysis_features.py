'''
Created on 2 Aug 2021

@author: hali02
'''

import os
import statistics

import audio_features.my_voice_analysis.my_voice_anaylsis as mysp
import parselmouth

class MySpeechAnalysisFeatures(object):
	'''
	class containing all audio features

	Parameters
	----------
	PATH: string
		the path to the source of the data we would like to open
	source type: string
		string denoting the type of file
		should be 'audio' = .wav
	language_code: string
		two-letter code to denote what language the file is in
	'''
	def __init__(self, PATH, source_type, language_code):
		
		self.languageCode = language_code
		self.sourceType = source_type
		self.directory_path, self.audio_file_title = os.path.split(PATH)

	def get_speechrate(self):
		'''
		output: rate_of_speech= 3 # syllables/sec original duration
		'''
		return mysp.myspsr(self.audio_file_title, self.directory_path)

	def get_gender_recognition(self):
		'''
		output: a female, mood of speech: Reading, p-value/sample size= :0.00 5
		'''
		return mysp.myspsr(self.audio_file_title, self.directory_path)

	def get_pppsp(self):
		'''
		output: Pronunciation_posteriori_probability_score_percentage= :85.00
		'''
		return mysp.mysppron(self.audio_file_title, self.directory_path)

	def get_syllable_count(self):
		'''
		output: number_ of_syllables= 154
		'''
		return mysp.myspsyl(self.audio_file_title, self.directory_path)

	def get_pause_count(self):
		'''
		output: number_of_pauses= 22
		'''
		return mysp.mysppaus(self.audio_file_title, self.directory_path)

	def get_rate_of_articulation(self):
		'''
		output: articulation_rate= 5 # syllables/sec speaking duration
		'''
		return mysp.myspatc(self.audio_file_title, self.directory_path)

	def get_speaking_duration(self):
		'''
		output: speaking_duration= 31.6 # sec only speaking duration without pauses
		'''
		return mysp.myspst(self.audio_file_title, self.directory_path)

	def get_total_duration(self):
		'''
		output: original_duration= 49.2 # sec total speaking duration with pauses
		'''
		return mysp.myspod(self.audio_file_title, self.directory_path)

	def get_ratio_speaking_total(self):
		'''
		output: balance= 0.6 # ratio (speaking duration)/(original duration)
		'''
		return mysp.myspbala(self.audio_file_title, self.directory_path)

	def get_mean_f0(self):
		'''
		output: f0_mean= 212.45 # Hz global mean of fundamental frequency distribution
		'''
		return mysp.myspf0mean(self.audio_file_title, self.directory_path)


	def get_sd_f0(self):
		'''
		output: f0_SD= 57.85 # Hz global standard deviation of fundamental frequency distribution
		'''
		return mysp.myspf0sd(self.audio_file_title, self.directory_path)

	def get_median_f0(self):
		'''
		output: f0_MD= 205.7 # Hz global median of fundamental frequency distribution
		'''
		return mysp.myspf0med(self.audio_file_title, self.directory_path)

	def get_min_f0(self):
		'''
		output: f0_MD= 205.7 # Hz global median of fundamental frequency distribution
		'''
		return mysp.myspf0min(self.audio_file_title, self.directory_path)

	def get_max_f0(self):
		'''
		output: f0_MD= 205.7 # Hz global median of fundamental frequency distribution
		'''
		return mysp.myspf0max(self.audio_file_title, self.directory_path)

	def get_upperQ_f0(self):
		'''
		output: f0_MD= 205.7 # Hz global median of fundamental frequency distribution
		'''
		return mysp.myspf0q75(self.audio_file_title, self.directory_path)

	def get_lowerQ_f0(self):
		'''
		output: f0_MD= 205.7 # Hz global median of fundamental frequency distribution
		'''
		return mysp.myspf0q25(self.audio_file_title, self.directory_path)

	def get_all_audio_features(self):
		'''
		output: f0_MD= 205.7 # Hz global median of fundamental frequency distribution
		'''
		return mysp.mysptotal(self.audio_file_title,self.directory_path)

	def get_median_intensity(self):
		'''
		'''
		# data, rate = sf.read(os.path.join(directory_path, audio_file_title))
		# meter = pyln.Meter(rate) #
		# loudness = meter.integrated_loudness(data)
		# print(loudness)
		snd = parselmouth.Sound(os.path.join(self.directory_path, self.audio_file_title))
		intensity = snd.to_intensity()
		median = intensity.get_average(averaging_method = 'MEDIAN')
		print(median)
		return median

	def get_sd_intensity(self):
		'''
		'''
		# data, rate = sf.read(os.path.join(directory_path, audio_file_title))
		# meter = pyln.Meter(rate) #
		# loudness = meter.integrated_loudness(data)
		# print(loudness)
		snd = parselmouth.Sound(os.path.join(self.directory_path, self.audio_file_title))
		intensity = snd.to_intensity()
		sd = statistics.stdev(intensity.values[0])
		print(sd)
		return sd
	
	def get_my_voice_analysis_features(self):
		'''
		'''
		features = {}
		
		features['speech_rate'] = self.get_speechrate()
		#features['gender_recognition'] = self.get_gender_recognition)
		'''
		features['pppsp'] = self.get_pppsp()
		features['syllable_count'] = self.get_syllable_count()
		features['pause_count'] = self.get_pause_count()
		features['rate_of_articulation'] = self.get_rate_of_articulation()
		features['median_intensity'] = self.get_median_intensity()
		features['sd_intensity'] = self.get_sd_intensity()
		features['speaking_duration'] = self.get_speaking_duration()
		features['total_duration'] = self.get_total_duration()
		features['ratio_speaking_total'] = self.get_ratio_speaking_total()
		'''
		features['mean_F0'] = self.get_mean_f0()
		'''
		features['sd_F0'] = self.get_sd_f0()
		features['median_F0'] = self.get_median_f0()
		features['min_F0'] = self.get_min_f0()
		features['max_F0'] = self.get_max_f0()
		features['upperQ_F0'] = self.get_upperQ_f0()
		features['lowerQ_F0'] = self.get_lowerQ_f0()
		'''
	
		return features