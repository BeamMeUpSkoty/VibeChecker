import pandas as pd

class TranscriptFile(object):
	"""Audio file in wav format. 
	"""

	def __init__(self, path, language):
		"""
		Parameters
		-----------
		path : string
			path to audio file
		language : string
			two letter code indicating language of audio file
		"""
		self.path = path
		self.language = language
		self.source_type = 'transcript'
		self.transcript = pd.read_csv(path)


	@staticmethod
	def open_file(path, language, encoding='utf-8'):
		"""
		"""
		return TranscriptFile(path, language)


	def make_alternating_speaker_df(self):
		""" merges rows for speaker diarization so that rows alternate between speakers.

		Returns
		---------
		combine_speech_turns_df : pandas dataframe
			alternating speakes, contains start, end, duration, speaker label
		"""
		combine_speech_turns = []

		for i, g in self.transcript.groupby([(self.transcript.speaker != self.transcript.speaker.shift()).cumsum()]):
			first_last = g[0::len(g)-1 if len(g) > 1  else 1]

			if len(first_last) > 1:
				first = first_last.iloc[0]['start']
				last = first_last.iloc[1]['end']
				speaker = first_last.iloc[0]['speaker']
				combine_speech_turns.append([first, last, last-first, speaker])
			else:
				first = first_last.iloc[0]['start']
				last = first_last.iloc[0]['end']
				speaker = first_last.iloc[0]['speaker']
				combine_speech_turns.append([first, last, last-first, speaker])

		combine_speech_turns_df = pd.DataFrame(combine_speech_turns, columns=['start', 'end', 'duration','speaker'])
		combine_speech_turns_df.to_csv('data/combine_speech_turns_df.csv')

		return combine_speech_turns_df
