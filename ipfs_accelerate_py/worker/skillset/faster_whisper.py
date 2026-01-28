import base64
import os
import subprocess
import tiktoken
from faster_whisper import WhisperModel
import pysbd
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import tempfile
import numpy as np
import io 
from io import BytesIO
import datetime

try:
    from ..common.storage_wrapper import storage_wrapper
except (ImportError, ValueError):
    try:
        from common.storage_wrapper import storage_wrapper
    except ImportError:
        storage_wrapper = None

tmp_model_dir = "/storage/cloudkit-models/faster-whisper-large-v3@hf/"

class hf_faster_whisper():
	def __init__(self, resources, meta):
		if storage_wrapper:
			try:
				self.storage = storage_wrapper()
			except:
				self.storage = None
		else:
			self.storage = None
		self.model = WhisperModel(resources['checkpoint'], device="cuda", compute_type="float16")
		self.nlp = pysbd.Segmenter(language="en", clean=False)
		self.encoding = tiktoken.get_encoding("cl100k_base")
		self.chunks = []
		self.tokens = []
		self.sentences = []
		self.transcription = ""
		self.noiseThreshold = 2000
		self.faster_whisper = self.runWhisper
		header_path = os.path.join(resources['checkpoint'], "header.bin")
		try:
			if self.storage:
				self.header = self.storage.read_file(header_path, pin=True)
			else:
				with open(header_path, "rb") as f:
					self.header = f.read()
		except:
			with open(header_path, "rb") as f:
				self.header = f.read()

	def __call__(self, method, **kwargs):
		if method == 'transcribe':
			return self.transcribe(**kwargs)
		elif method == 'faster_whisper':
			return self.runWhisper(**kwargs)
		else:
			print(self)
			raise Exception('bad method in __call__: %s' % method)		

	def transcribe(self, audio, fragment=None,  **kwargs):
		processed_data = self.dataPreprocess(audio)
		self.chunks.append(processed_data)
		self.writeToFile(self.chunks)
		
		if(self.noiseFilter()):
			self.transcription = self.runWhisper(self.file_path, fragment=None)

		return self.transcription

	def stop(self):
		self.chunks.clear()
		self.transcription = ""
		try:
			if self.storage and hasattr(self, 'file_path'):
				self.storage.remove_file(self.file_path)
			elif hasattr(self, 'file_path') and os.path.exists(self.file_path):
				os.remove(self.file_path)
		except:
			if hasattr(self, 'file_path') and os.path.exists(self.file_path):
				os.remove(self.file_path)

	def crop_audio_after_silence(self, audio_file, min_silence_len=500, silence_thresh=-16):
	
		sound = AudioSegment.from_file(audio_file, format="ogg")
		nonsilent_parts = detect_nonsilent(sound, min_silence_len, silence_thresh)
		
		if nonsilent_parts and len(nonsilent_parts) > 1:
			end_of_first_silence = nonsilent_parts[1][0]
			cropped_audio  = sound[:end_of_first_silence]
			return cropped_audio
		else:
			return sound

	def dataPreprocess(self, data):
		data = data.partition(",")[2]
		data_decoded = base64.b64decode(data)
		## convert to bytes
		return data_decoded

	def runWhisper(self, audio, **kwargs):
		data = audio
		if os.path.isfile(data):
			segments, info = self.model.transcribe(data,  vad_filter=True)
		elif ("data:audio" in data):
			audio_bytes = self.dataPreprocess(data)
			with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
				audio_segment = AudioSegment.from_file(io.BytesIO(self.header + audio_bytes),format="webm").export(temp_audio.name, format="ogg")
				segments, info = self.model.transcribe(temp_audio.name,  vad_filter=True)
			pass

		i = 0
		sentence_count = 1
		if "fragment" in kwargs:
			fragment = kwargs["fragment"]
		else:
			fragment = None

		if "timestamp" in kwargs:
			timestamp = kwargs["timestamp"]
		else:
			timestamp = None

		if fragment != None:
			if type(fragment) == str:					
				encodeed_text = self.encoding.encode(fragment)
				for token in encodeed_text:
					self.process_token(token, self.tokens, self.sentences, self.add_sentence_to_list)
			elif type(fragment) == list:
				fragment = " ".join(fragment)
				encodeed_text = self.encoding.encode(fragment)
				for token in encodeed_text:
					self.process_token(token, self.tokens, self.sentences, self.add_sentence_to_list)
			else:
				raise Exception("Fragment must be a string or a list of strings")
			pass
		for segment in segments:
			#print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
			encodeed_text = self.encoding.encode(segment.text)
			# self.chunks issue here 
			#self.sentences.append(segment.text)
			for token in encodeed_text:
				this_result = self.process_token(token, self.tokens, self.sentences, self.add_sentence_to_list)
				this_sentence_count = len(self.sentences)
				if this_sentence_count > sentence_count:
					prev_sentence = self.sentences[this_sentence_count - 2]
					self.process_sentence(prev_sentence)
					sentence_count = this_sentence_count
					pass
			i = i + 1
		
		#self.process_sentence(self.sentences[-1])
		result_sentences = self.sentences
		self.sentences = []
		self.tokens = []
		return {
            'text': result_sentences,
			'timestamp': timestamp,
            'done': True
        }


	def noiseFilter(self):
		if(os.path.getsize(self.file_path) >= self.noiseThreshold):
			return True
		else:
			return False

	def process_sentence(self, sentence):
		print(sentence)
		return sentence

	def process_token(self, token, token_list, sentence_list, callback):
		# Add the token to the token list
		token_list.append(token)
		# Join the token list into a single string
		text = self.encoding.decode(token_list)
		# Split the token list into sentences
		sentences = self.nlp.segment(text)
		# Use the callback to add the sentences to the sentence list
		for sentence in sentences:
			if sentence not in sentence_list:
				found = False
				for this_sentence in sentence_list:
					if sentence in this_sentence and len(sentence) > 32 and len(this_sentence) > 32:
						found = True
						break
				if not found:
					callback(sentence_list, sentence)

	def add_sentence_to_list(self, sentence_list, sentence):
		if len(sentence_list) == 0:
			sentence_list.append(sentence)
		else:
			found = False
			for this_sentence in sentence_list:
				if this_sentence in sentence :
					found = True
					sentence_index = sentence_list.index(this_sentence)
					sentence_list[sentence_index] = sentence
					break
			if not found:
				sentence_list.append(sentence)

	# def test(self):
	# 	text = 'The U.S. Supreme Court has said that the purpose of allowing federal officers to move cases against them to federal court is to protect the federal government from operational interference that could occur if federal officials were arrested and tried in state court for actions that fall within the scope of their duties, Pryor wrote. “Shielding officers performing current duties effects the statute’s purpose of protecting the operations of federal government,” he wrote. “But limiting protections to current officers also respects the balance between state and federal interests” by preventing federal interference with state criminal proceedings. Pryor also rejected Meadows’ argument that moving his case to federal court would allow him to assert federal immunity defenses that may apply to former officers, writing that he “cites no authority suggesting that state courts are unequipped to evaluate federal immunities.” The conspiracy to overturn the election alleged in the indictment and the acts of “superintending state election procedures or electioneering on behalf of the Trump campaign” were not related to Meadows’ duties as chief of staff, Pryor wrote. “Simply put, whatever the precise contours of Meadows’s official authority, that authority did not extend to an alleged conspiracy to overturn valid election results,” Pryor wrote. '
	# 	text_token_list = self.encoding.encode(text) 
		
	# 	for token in text_token_list:
	# 		self.process_token(token, self.tokens, self.sentences, self.add_sentence_to_list)
			
	# 	return self.sentences
	
	# def test2(self):
	# 	audio_url = "https://upload.wikimedia.org/wikipedia/commons/f/f9/%22Let_Us_Continue%22_speech_audio_trimmed.ogg"
	# 	with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
	# 		subprocess.run(["wget", "-O", temp_audio.name, audio_url])
	# 		audio = AudioSegment.from_file(temp_audio.name, format="ogg")
	# 		trimmed_audio = self.crop_audio_after_silence(temp_audio.name)
	# 		trimmed_audio.export("trimmed_audio.ogg", format="ogg")
	# 		trimmed_audio = AudioSegment.from_file("trimmed_audio.ogg", format="ogg")
	# 		audio_length = audio.duration_seconds
	# 		trimmed_audio_length = trimmed_audio.duration_seconds
	# 		print("audio length:", audio_length)
	# 		print("trimmed audio length:", trimmed_audio_length)
	# 		print("difference:", audio_length - trimmed_audio_length)
	# 	return [audio_length, trimmed_audio_length, audio_length - trimmed_audio_length]
	
	def test3(self):
		audio_url = "https://upload.wikimedia.org/wikipedia/commons/f/f9/%22Let_Us_Continue%22_speech_audio_trimmed.ogg"
		with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
			subprocess.run(["wget", "-O", temp_audio.name, audio_url])
			trimmed_audio = self.crop_audio_after_silence(temp_audio.name)
			trimmed_audio.export("trimmed_audio.ogg", format="ogg")
			trimmed_audio = AudioSegment.from_file("trimmed_audio.ogg", format="ogg")
			self.runWhisper("trimmed_audio.ogg")

	def test4(self):
		audio_url = "https://upload.wikimedia.org/wikipedia/commons/f/f9/%22Let_Us_Continue%22_speech_audio_trimmed.ogg"
		fragment = "Dunkin' Donuts LLC,[1] doing business as Dunkin' since 2019, is an American multinational coffee and doughnut company, as well as a quick service restaurant. It was founded by Bill Rosenberg (1916–2002) in Quincy, Massachusetts, in 1950. The chain was acquired by Baskin-Robbins's holding company Allied Lyons in 1990; its acquisition of the Mister Donut chain and the conversion of that chain to Dunkin' Donuts facilitated the brand's growth in North America that year.[5] Dunkin' and Baskin-Robbins eventually became subsidiaries of Dunkin' Brands, headquartered in Canton, Massachusetts, in 2004, until being purchased by Inspire Brands on December 15, 2020. The chain began rebranding as a beverage-led company, and was renamed Dunkin', in January 2019; while stores in the U.S. began using the new name, the company intends to roll out the rebranding to all of its international stores eventually."
		fragment_split = fragment.split(" ")
		with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
			subprocess.run(["wget", "-O", temp_audio.name, audio_url])
			trimmed_audio = self.crop_audio_after_silence(temp_audio.name)
			trimmed_audio.export("trimmed_audio.ogg", format="ogg")
			trimmed_audio = AudioSegment.from_file("trimmed_audio.ogg", format="ogg")
			start_timestamp = datetime.datetime.now()
			self.runWhisper("trimmed_audio.ogg", fragment_split)
			end_timestamp = datetime.datetime.now()
			print("time taken:")
			print(end_timestamp - start_timestamp)

	def test5(self):
		this_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base64ogg.txt")
		try:
			if self.storage:
				base64 = self.storage.read_file(this_file, pin=False).decode('utf-8')
			else:
				with open(this_file, "r") as file:
					base64 = file.read()
		except:
			with open(this_file, "r") as file:
				base64 = file.read()
		fragment = " "
		#fragment = "Dunkin' Donuts LLC,[1] doing business as Dunkin' since 2019, is an American multinational coffee and doughnut company, as well as a quick service restaurant. It was founded by Bill Rosenberg (1916–2002) in Quincy, Massachusetts, in 1950. The chain was acquired by Baskin-Robbins's holding company Allied Lyons in 1990; its acquisition of the Mister Donut chain and the conversion of that chain to Dunkin' Donuts facilitated the brand's growth in North America that year.[5] Dunkin' and Baskin-Robbins eventually became subsidiaries of Dunkin' Brands, headquartered in Canton, Massachusetts, in 2004, until being purchased by Inspire Brands on December 15, 2020. The chain began rebranding as a beverage-led company, and was renamed Dunkin', in January 2019; while stores in the U.S. began using the new name, the company intends to roll out the rebranding to all of its international stores eventually."
		fragment_split = fragment.split(" ")
		results = self.runWhisper(base64, fragment_split)
		return results


if __name__ == '__main__':
	#this_transcription = Transcription()
	this_transcription = hf_faster_whisper(None, None)
	#resulst1 = this_transcription.test()
	#results2 = this_transcription.test2()
	#results3 = this_transcription.test3()
	results4 = this_transcription.test4()
	#results5 = this_transcription.test5()
	#print(resulst1)
	#print(results2)
	#print(results3)
	print(results4)
	#print(results5)
	pass
