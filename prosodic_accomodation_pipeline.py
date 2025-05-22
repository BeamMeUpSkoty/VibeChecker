'''
created on April 3, 2023

@author: hali02
'''
import fire 

from data_types.audio_file import AudioFile
from data_types.transcript_file import TranscriptFile

from accomodation_types.turn_level_prosodic_acommodation import TurnLevelProsodicAcommodation
from accomodation_types.hybrid_prosodic_acomodation import HYBRIDProsodicAcommodation

class ProsodicAccomodation(object):

	def __init__(self, audio_path, diarization_path, results_path, accomodation_type, language_code, visualize=True, verbose=False):
		""" 
		parameters
		------------
		audio_path : string
		diraization_path : string
		accomodation_type : string
		language_code : string
		visualize : boolean
			default True, 
		verbose : boolean
			defualt False

		Attributes
		------------
		self.audio_path
		self.diarization_path
		self.accomodation_type
		self.visualize
		self.verbose
		"""

		self.audio_path = audio_path
		self.diarization_path = diarization_path
		self.results_path = results_path
		self.accomodation_type = accomodation_type
		self.language_code = language_code

		self.visualize = visualize
		self.verbose = verbose

	def prosodic_accomodation_pipeline(self):
		"""
		"""

		###### TURN LEVEL PA ##########
		if self.accomodation_type == 'turn_level': 
			TL = TurnLevelProsodicAcommodation(self.audio_path, self.diarization_path)
			print("convergence:", TL.get_convergence())
			print("synchrony:", TL.get_synchrony())

		###### HYBRID PA ##########
		if self.accomodation_type == 'hybrid': 
			HPA = HYBRIDProsodicAcommodation(self.audio_path, self.diarization_path, self.language_code)	
			speaker1_features, speaker2_features = HPA.get_features_by_speaker()
			data_spk1, data_spk2, accomodation, significance = HPA.sliding_window_correlation(speaker1_features, speaker2_features, ['mean_F0'])
			print(accomodation, significance)
			HPA.get_visualization(data_spk1, data_spk2, accomodation)

		return


if __name__ == '__main__':
	"""
	creates command line interface
	"""
	fire.Fire(ProsodicAccomodation)


	#audio_path = 'data/audio/audio-2.wav'
	#diar_file = 'data/transcripts/transcript.csv'
	#diar_file = 'data/combine_speech_turns_df.csv'
	#OUTPATH = ''

	#articulation rate (AR) as the number of syllables/sec

	#mean - mean F0

	#median - Median F0

	#SD F0

	#range - 80th percentile range which is the difference between the 90th and 10th percentile

	#CSI - cumulative slope index (ST/syllable)
