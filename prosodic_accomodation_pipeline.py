'''
created on April 3, 2023

@author: hali02
'''
import fire

from accomodation_types.turn_level_prosodic_acommodation import TurnLevelProsodicAcommodation
from accomodation_types.hybrid_prosodic_acomodation import HYBRIDProsodicAcommodation

class ProsodicAccomodation(object):

	def __init__(self, audio_path:str, diarization_path:str, results_path:str, accomodation_type:str, language_code:str, visualize:bool=True, verbose:bool=False):
		"""
		Wrapper class to run different prosodic accommodation pipelines.
		"""
		self.audio_path:str = audio_path
		self.diarization_path:str = diarization_path
		self.results_path:str = results_path
		self.accomodation_type:str = accomodation_type
		self.language_code:str = language_code

		self.visualize:bool = visualize
		self.verbose:bool = verbose

	def prosodic_accomodation_pipeline(self) -> None:
		"""
		Run the selected prosodic accommodation pipeline.
		"""
		if self.accomodation_type == 'turn_level':
			print("Running Turn-Level Prosodic Accommodation...")
			TL = TurnLevelProsodicAcommodation(
				self.audio_path,
				self.diarization_path,
				self.language_code,
			)
			convergence = TL.get_convergence()
			synchrony = TL.get_synchrony()
			print("Convergence (r, p):", convergence)
			print("Synchrony (r, p):", synchrony)

			if self.visualize:
				TL.get_visualization()

		elif self.accomodation_type == 'hybrid':
			print("Running HYBRID Prosodic Accommodation...")
			HPA = HYBRIDProsodicAcommodation(
				self.audio_path,
				self.diarization_path,
				self.language_code
			)
			speaker1_features, speaker2_features = HPA.get_features_by_speaker()
			data_spk1, data_spk2, accommodation, significance = HPA.sliding_window_correlation(
				speaker1_features,
				speaker2_features,
				features=['mean_F0']
			)
			print("Accommodation metrics:", accommodation)
			print("Significance:", significance)

			if self.visualize:
				HPA.get_visualization(data_spk1, data_spk2, accommodation)

		else:
			raise ValueError(
				f"Unknown accommodation type: {self.accomodation_type}. Choose from ['turn_level', 'hybrid'].")
		return


if __name__ == '__main__':
	"""
	creates command line interface
	"""
	fire.Fire(ProsodicAccomodation)


	#audio_path = 'data/audio/example.wav'
	#diar_file = 'data/transcripts/transcript.csv'
	#diar_file = 'data/combine_speech_turns_df.csv'
	#OUTPATH = ''

	#articulation rate (AR) as the number of syllables/sec

	#mean - mean F0

	#median - Median F0

	#SD F0

	#range - 80th percentile range which is the difference between the 90th and 10th percentile

	#CSI - cumulative slope index (ST/syllable)
