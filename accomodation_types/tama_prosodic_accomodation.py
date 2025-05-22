from data_types.transcript_file import TranscriptFile
from data_types.audio_file import AudioFile

class TAMAProsodicAcommodation(object):
	""" Time Aligned Moving Average (TAMA) analyses the audio in a 
	fixed window, averageing values over the duration of the window. 
	This method is based off extracting average prosodic values by for
	each speaker from a series of overlapping fixed length windows

	de Looze 2011; Fixed window = 20s, Time step = 10s, weighed average

	"""

	def __init__(self, PATH, OUTPATH, diarization_path, language_code):
		self.path = PATH 
		self.outpath = OUTPATH
		self.diarization_file = TranscriptFile(diarization_path, language_code)
		self.language_code = language_code


	def get_acommodation(self):
		"""
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

		return

	def get_convergence(self):
		"""
		"""
		return

	def get_synchrony(self):
		"""
		"""
		return
