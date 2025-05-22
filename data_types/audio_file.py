#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Jul 27, 2020

@author: hali02
'''
import subprocess
import fleep
import os
from pydub import AudioSegment
from pyannote.core import SlidingWindow
import torchaudio
from math import ceil
import shutil

#print(os.environ['PATH'])

class AudioFile(object):
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
		self.source_type = 'audio'
		self.audio = AudioSegment.from_wav(self.path)

	@staticmethod
	def open_file(path, language, encoding='utf-8'):
		"""Loads the feature source from file.

		Parameters
		----------
		path : string
			Path to load the features from.
		language : string
			two letter code indicating language of audio file

		Returns
		--------
		AudioFile :  AudioFile object
		"""

		audio_type = AudioFile._get_audio_type(path, verbose=False)
		if audio_type[0] != 'wav':
			AudioFile.convert_to_wav(path)
			path = path[:-4] + '.wav'
				
		return AudioFile(path, language)

	@staticmethod
	def _get_audio_type(filename, verbose=True):
		"""
		Parameters
		------------
		filename : string
			name of audio file
		verbose : boolean, default True
			if True, then prints information about the audio file

		Return
		-------
		info.extension : xxx
			xxx
		"""
		with open(filename, "rb") as file:
			info = fleep.get(file.read(128))
		
		if verbose:
			print(info.type)
			print(info.extension)
			print(info.mime) 
		return info.extension
	
	
	@staticmethod
	def convert_to_wav(file):
		"""
		Parameters
		-----------
		file : string
			path to audio file

		Return
		----------
		new_file_name : string
			return name of newly converted wav file		
		"""

		# command = ['ffmpeg', '-i', file, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', file[:-4] + '.wav']
		# print('tmp/' + file[:-5] + '.wav')
		command = ['ffmpeg', '-n', '-hide_banner', '-loglevel', 'quiet', '-i', file,'-ac', '1', '-ar', 16000, file[:-4] + '.wav']
		print(command)
		#subprocess.call(command)
		#subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		return file[:-4] + '.wav'

	@staticmethod
	def crop_audio(filename, start, stop, path, sampling_rate=16000):
		"""
		Parameters
		----------
		filename : string
			name of audio file
		start : float
			start time of audio chunk
		stop : float
			end time of audio chunk
		path : string
			path to audio file
		sampling_rate : int, default 16,000
			sample raae of the audio

		Return
		-------
		name : string
			path to new cropped audio
		"""

		duration = stop-start
		name = os.path.join(path, f'{os.path.basename(filename)}')
		command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'quiet', '-ss', str(start), '-t', str(duration), '-i', filename,'-ac', '1', '-ar', '16000', name]
		subprocess.call(command)
		return name

	
	def get_duration(self):
		"""
		get duration 

		returns
		--------
		duration : float
			duration of audio in seconds
		"""
		return self.audio.duration_seconds


	def resample(self, ip, op, sr=16000):
		"""
		Resample to input sampling rate and save new file

		Parameters
		----------
		ip : xxx
			xxx
		op : xxx
			xxx
		sr : xxx
			xxx
		"""
		arr, org_sr = torchaudio.load(ip)
		if not org_sr == sr:
			name = os.path.join('tmp/', f'resample-{os.path.basename(ip)}')
			command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'quiet', '-i', ip,'-ac', '1', '-ar', '16000', name]
			subprocess.call(command)
			# arr = torchaudio.functional.resample(arr, orig_freq=self.audio.frame_rate, new_freq=sr)
			# torchaudio.save(op, arr, sr, encoding="PCM_S", bits_per_sample=16)
			return name
		else:
			return False
	
	def make_sliding_window(self, duration=3, overlap=0.5):
		"""generate Sliding window over the audio file

		Parameters
		----------
		duration : int, optional
			chunk size in seconds, by default 3 sec
		overlap : float, optional
			overlap between consecutive chunks in seconds, by default 0.5 sec

		Returns
		-------
		sliding_window : pyannote.core.SlidingWindow
			list of audio segments of pyannote.core.Segments 
			Example:    [ 00:00:00.000 -->  00:00:03.000]
						[ 00:00:00.500 -->  00:00:03.500]
						....
		"""        
		durationInSeconds = self.get_duration()
		return SlidingWindow(duration, overlap, 0, durationInSeconds-duration+overlap)
	
	def make_chunks(self, chunk_length):
		"""
		method adapted from pyddub package
		
		Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
		long.
		if chunk_length is 50 then you'll get a list of 50 millisecond long audio
		segments back (except the last one, which can be shorter)
		
		Parameters
		-----------
		chunk_length : xxx
			xxx
		returns
		--------
		list of tuples: [(audio_chunk, start_time, end_time)]
		
		"""
		number_of_chunks = ceil(len(self.audio) / float(chunk_length))
		print(number_of_chunks)
		return [(self.audio[i * chunk_length:(i + 1) * chunk_length], i * chunk_length, (i + 1) * chunk_length) for i in range(int(number_of_chunks))]
	
	
	def split_audio_by_window_size(self, tmp_path='tmp/', window_size=5000):
		"""
		split wav file into multiple wave files. 
		new wave files have length of window_size
		split files are saved in tmp/ with naming convention startTime_endTime.wav
		
		Parameters
		----------
		tmp_path : xxx
			xxx
		window_size: int
			window size in milliseconds.
			
		Returns
		----------
		tmp/ : directory
			returns a directory tmp/ of audio files of size window_size
			Feautres can then be computed from each of the split files. 
		"""
		#get duration of audio file in seconds
		duration = self.get_duration()
		
		#create a new tmp file to store newly spliced audio file
		AudioFile.remove_tmp_file(tmp_path)
		AudioFile.create_tmp_file(tmp_path)

		#set window length in milliseconds
		chunk_length_ms = window_size  
		#cut audio file into window-sized chunks
		chunks = self.make_chunks(chunk_length_ms)  # Make chunks of one sec
		
		#store chunks in tmp file with the naming convention startTime_endTime.wav (this is necessary to get the time alignment later
		for i, ch in enumerate(chunks):
			if i == len(chunks)-1:
				#save last chunk with the over time duration becuase last chunk maybe smaller than predefined window size.
				ch[0].export(tmp_path + str(ch[1]) + '_' + str(duration*1000) + '.wav', format='wav')
			else:
				ch[0].export(tmp_path + str(ch[1]) + '_' + str(ch[2]) + '.wav', format='wav')
		return
	

	def make_sliding_window_chunks(self, duration, step_size, tmp_path):
		"""Split audio based on the sliding window interval

		Parameters
		----------
		duration : xxx
			xxx
		step_size : xxx
			xxx
		tmp_path : xxx
			xxx
		"""        
		sliding_window = self.make_sliding_window(duration, step_size)
		for index, seg in enumerate(sliding_window):
			split = self.audio[seg.start*1000:seg.end*1000] #convert to millisecond
			filename = os.path.join(tmp_path,'chunk_{}.wav'.format(index))
			split.export(filename, format='wav')


	def make_audio_chunk_by_utterance(self, start, end, tmp_path='tmp/'):
		""" splits audio by given start and end time. Chunks are saved in tmp/ folder.
		Parameters
		----------
		start : float
			time (in seconds) where the audio chunk should start
		end : float
			time (in seconds) where the audio chunk should end
		tmp_path : str
			temporary directory path

		Returns
		----------
		filename : string
			path to file created in tmp\
		"""
		
		#create audio chunk based on start and end time in miliseconds
		split = self.audio[start*1000:end*1000] #convert to millisecond
		filename = os.path.join(tmp_path,'chunk_{}_{}.wav'.format(start, end))
		split.export(filename, format='wav')
		return filename

	@staticmethod
	def lookup_and_extend_window(result, start, end, duration=10):
		"""xxx

		Parameters
		----------
		result : pandas df
			pandas df with speaker turns, start and end time
		start : float
			start time of chunck in seconds
		end : float
			end time of chunk in seconds
		duration : int
			length of fixed window size in seconds

		Returns
		----------
		start_time: 
			xxx
		end_time: 
			xxx
		"""   
		difference_start = duration*1000
		start_time = None
		difference_end = duration*1000
		end_time = None

		#iterate through speaker diarization data frame
		for index, row in result.iterrows():
			#find the differnce start of sliding window and speaker segment start
			difference = float(start) - float(row['start'])
			#print(float(start), row['start'], difference)
			
			#if difference is positive and within the start of window 
			if difference >= 0 and difference < difference_start:
				#update start
				difference_start = difference
				#save start row value 
				start_time = row['start']
			#update end of window based on speaker turn
			difference = float(row['end']) - float(end)

			#if difference is positive and within the end of window
			if difference >= 0 and  difference < difference_end:
				#update end
				difference_end = difference
				#save end time row value
				end_time = float(row['end'])

			#check initial value
			if float(start) == 0:
				difference_start = 0
				start_time = float(start)
		#check end value
		if end_time == None:
			end_time = float(row['end'])

		return start_time, end_time


	def make_diar_friendly_sliding_window(self, result, duration=20, step_size=10, tmp_path='tmp/',):
		"""xxx

		Parameters
		----------
		result : pandas df
			pandas df with speaker turns, start time and end time
		duration : int, default 20
			time in seconds to chunk the audio file
		step_size : int, default 10
			step size in seconds to start next chunk
		tmp_path : 
			xxx

		Returns
		----------
		adjusted_sliding_windows
		"""
		#creates sliding windows
		sliding_window = self.make_sliding_window(duration, step_size)

		adjusted_sliding_windows = []
		for index, seg in enumerate(sliding_window):
			start = seg.start
			end = seg.end
			start, end = AudioFile.lookup_and_extend_window(result, start, end, duration)
			#split = self.audio[start*1000:end*1000] #convert to millisecond
			#filename = os.path.join(tmp_path,'chunk_{}.wav'.format(index))
			#split.export(filename, format='wav')
			adjusted_sliding_windows.append({'start':start, 'end':end})
		return adjusted_sliding_windows


	@staticmethod
	def remove_tmp_file(tmp_path='tmp/', verbose=False):
		"""
		Parameters
		----------
			
		Returns
		----------
		remove tmp file once finished calculating features.
		"""
		if os.path.isdir(tmp_path):
			if verbose:
				print('### Removing tmp file: ', tmp_path, ' ###')
			shutil.rmtree(tmp_path, ignore_errors=True)
		return
	
	
	@staticmethod
	def create_tmp_file(tmp_path='tmp/', verbose=True):
		"""
		Parameters
		----------
			
		Returns
		----------
		remove tmp file once finished calculating features.
		"""
		if os.path.isdir(tmp_path) == False:
			if verbose:
				print('### creating tmp file: ', tmp_path, ' ###')
			os.mkdir(tmp_path)
		return
		
