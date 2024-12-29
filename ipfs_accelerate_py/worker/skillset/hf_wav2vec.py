import torch
import librosa
from datasets import Dataset, Audio
from transformers import pipeline
import os
import numpy as np
from pydub import AudioSegment
import tempfile
import io
from transformers import AutoModelForAudioClassification
from transformers import AutoFeatureExtractor

class hf_wav2vec:
	def __init__(self, resources, meta=None):
		if os.path.exists(resources['checkpoint']) and os.path.isfile(resources['checkpoint'] + "/config.json"):
			self.model = AutoModelForAudioClassification.from_pretrained(
                resources['checkpoint'],
                local_files_only=True
            ).eval()
			self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                resources['checkpoint'],
                local_files_only=True
            )
		else:
			self.classifier = pipeline("audio-classification", model=resources['checkpoint'])
		with open(os.path.join(resources['checkpoint'], "header.bin"), "rb") as f:
			self.header = f.read()
		
	def __call__(self, method, **kwargs):
		if method == 'wav2vec_classify':
			return self.wav2vec_classify(**kwargs)
		else:
			raise Exception('unknown method: %s' % method)
		
	def map_to_array(self, example):
		speech, _ = librosa.load(example["file"], sr=16000, mono=True)
		example["speech"] = speech
		return example

	def wav2vec_classify(self, audio, **kwargs):
		if os.path.exists(audio) and os.path.isfile(audio):
			audio_filename = audio
			audio_dataset = Dataset.from_dict({"audio": [audio_filename],"file":[audio_filename]}).cast_column("audio", Audio())
			audio_dataset = audio_dataset.map(self.map_to_array)
			speech = audio_dataset[:4]["speech"]
		else:
			with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
				if type(audio) == str:
					audio = audio.encode()
				else:
					pass
				AudioSegment.from_file(io.BytesIO(self.header + audio),format="webm").export(temp_audio.name, format="ogg")
				audio_filename = temp_audio.name
				audio_dataset = Dataset.from_dict({"audio": [audio_filename],"file":[audio_filename]}).cast_column("audio", Audio())
				audio_dataset = audio_dataset.map(self.map_to_array)
				speech = audio_dataset[:4]["speech"]
		
		if "classifier" in self.__dict__.keys():
			## audio file path
			results = self.classifier(audio, top_k=5)
			#results = json.dumps(results)
			
		else:
			if "sampling_rate" in kwargs.keys():
				sampling_rate = kwargs["sampling_rate"]
			else:
				sampling_rate = 16000
			inputs = self.feature_extractor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
            #convert audio from base64 to numpy array of doubles
			with torch.no_grad():
				logits = self.model(**inputs).logits

			predicted_class_ids = torch.argmax(logits).item()
			predicted_label = self.model.config.id2label[predicted_class_ids]
			results = predicted_label
		
		return {
            'text': results, 
            'done': True
        }
	
	def test(self, **kwargs):
		audio_filename = "/tmp/temp.ogg"
		return self.wav2vec_classify(audio_filename)
		
	def test2(self, **kwargs):
		audio_filename = "/tmp/base64ogg.txt"
		with open(audio_filename, "rb") as audio_file:
			audio = audio_file.read()
		return self.wav2vec_classify(audio)

#if __name__ == '__main__':
#	test_hf_wav2vec = hf_wav2vec({'checkpoint': '/storage/cloudkit-models/hubert-large-superb-er@hf/'})
#	print(test_hf_wav2vec.test2())
	#test_hf_wav2vec = hf_wav2vec({'checkpoint': '/storage/cloudkit-models/wav2vec-english-speech-emotion-recognition@hf/'})
	#print(test_hf_wav2vec.test2())
	#test_hf_wav2vec = hf_wav2vec({'checkpoint': 'superb/hubert-large-superb-er'})
	#print(test_hf_wav2vec.test2())