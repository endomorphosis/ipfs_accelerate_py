import base64
import os
import subprocess
import tiktoken
# from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoConfig, AutoProcessor, AutoModel
import pysbd
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import tempfile
import numpy as np
import io 
from io import BytesIO
import datetime
import torch
import asyncio

class hf_whisper:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_whisper_endpoint_handler = self.create_openvino_whisper_endpoint_handler
        self.create_cuda_whisper_endpoint_handler = self.create_cuda_whisper_endpoint_handler
        self.create_cpu_whisper_endpoint_handler = self.create_cpu_whisper_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init = self.init
        self.__test__ = self.__test__
        return None

    def init_cpu (self, model, device, cpu_label):
        return None
    
    
    def init_cuda(self, model, device, cuda_label):
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = AutoProcessor.from_pretrained(model)
        endpoint = None
        try:
            endpoint = AutoModel.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_llm_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
    
    def init_openvino(self, model, model_type, device, openvino_label, get_openvino_genai_pipeline, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type):
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        batch_size = 0                
        tokenizer =  AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        endpoint = get_openvino_model(model, model_type, openvino_label)
        endpoint_handler = self.create_openvino_llm_endpoint_handler(endpoint,tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size          
    
    def create_cuda_whisper_endpoint_handler(self, local_cuda_endpoint, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y=None, local_cuda_endpoint=local_cuda_endpoint, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
            # if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
            #       self.local_endpoints[endpoint_model][cuda_label].eval()
            if "eval" in dir(local_cuda_endpoint):
                local_cuda_endpoint.eval()
            else:
                pass
            with torch.no_grad():
                try:
                    torch.cuda.empty_cache()
                    config = AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    
                    if x is not None and type(x) == str:
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    elif type(x) == tuple:
                        conversation = x
                    elif type(x) == dict:
                        raise Exception("Invalid input to vlm endpoint handler")
                    elif type(x) == list:
                        # conversation = x
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    else:
                        raise Exception("Invalid input to vlm endpoint handler")
                  
                    prompt = local_cuda_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = local_cuda_processor(prompt, return_tensors="pt").to(cuda_label, torch.float16)
                    output = local_cuda_endpoint.generate(**inputs, max_new_tokens=30)
                    result = local_cuda_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # Run model inference
                    torch.cuda.empty_cache()
                    return result
                except Exception as e:
                    # Cleanup GPU memory in case of error
                    torch.cuda.empty_cache()
                    raise e
        return handler

    def create_cpu_whisper_endpoint_handler(self, local_cuda_endpoint, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y=None, local_cuda_endpoint=local_cuda_endpoint, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
            # if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
            #       self.local_endpoints[endpoint_model][cuda_label].eval()
            if "eval" in dir(local_cuda_endpoint):
                local_cuda_endpoint.eval()
            else:
                pass
            with torch.no_grad():
                try:
                    torch.cuda.empty_cache()
                    config = AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    
                    if x is not None and type(x) == str:
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    elif type(x) == tuple:
                        conversation = x
                    elif type(x) == dict:
                        raise Exception("Invalid input to vlm endpoint handler")
                    elif type(x) == list:
                        # conversation = x
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": x},
                                ],
                            },
                        ]
                    else:
                        raise Exception("Invalid input to vlm endpoint handler")
                  
                    prompt = local_cuda_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = local_cuda_processor(prompt, return_tensors="pt").to(cuda_label, torch.float16)
                    output = local_cuda_endpoint.generate(**inputs, max_new_tokens=30)
                    result = local_cuda_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # Run model inference
                    torch.cuda.empty_cache()
                    return result
                except Exception as e:
                    # Cleanup GPU memory in case of error
                    torch.cuda.empty_cache()
                    raise e
        return handler

    def create_openvino_whisper_endpoint_handler(self, openvino_endpoint_handler, openvino_tokenizer, endpoint_model, openvino_label):
        def handler(x, y=None, openvino_endpoint_handler=openvino_endpoint_handler, openvino_tokenizer=openvino_tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label):
            chat = None
            if y is not None and x is not None:
                chat = x
            elif x is not None:
                if type(x) == tuple:
                    chat, image_file = x
                elif type(x) == list:
                    chat = x[0]
                    image_file = x[1]
                elif type(x) == dict:
                    chat = x["chat"]
                    image_file = x["image"]
                elif type(x) == str:
                    chat = x
                else:
                    pass

            pipeline_config = { "MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 512 ,  "NPUW_CACHE_DIR": ".npucache" }
            results = openvino_endpoint_handler.generate(x, max_new_tokens=100, do_sample=False)
            # prompt = openvino_endpoint_handler.apply_chat_template(chat, add_generation_prompt=True)
            # inputs = openvino_endpoint_handler(text=prompt, return_tensors="pt")
            # streamer = TextStreamer(openvino_tokenizer, skip_prompt=True, skip_special_tokens=True)
            # output_ids = openvino_endpoint_handler.generate(
            #     **inputs,
            #     do_sample=False,
            #     max_new_tokens=50,
            #     streamer=streamer,
            return results
        return handler


    # 	# self.model = WhisperModel(resources['checkpoint'], device="cuda", compute_type="float16")
    # 	self.nlp = pysbd.Segmenter(language="en", clean=False)
    # 	self.encoding = tiktoken.get_encoding("cl100k_base")
    # 	self.chunks = []
    # 	self.tokens = []
    # 	self.sentences = []
    # 	self.transcription = ""
    # 	self.noiseThreshold = 2000
    # 	self.faster_whisper = self.runWhisper
    # 	with open(os.path.join(resources['checkpoint'], "header.bin"), "rb") as f:
    # 		self.header = f.read()

    # def __call__(self, method, **kwargs):
    # 	if method == 'transcribe':
    # 		return self.transcribe(**kwargs)
    # 	elif method == 'faster_whisper':
    # 		return self.runWhisper(**kwargs)
    # 	else:
    # 		print(self)
    # 		raise Exception('bad method in __call__: %s' % method)		

    # def transcribe(self, audio, fragment=None,  **kwargs):
    # 	processed_data = self.dataPreprocess(audio)
    # 	self.chunks.append(processed_data)
    # 	self.writeToFile(self.chunks)
        
    # 	if(self.noiseFilter()):
    # 		self.transcription = self.runWhisper(self.file_path, fragment=None)

    # 	return self.transcription

    # def stop(self):
    # 	self.chunks.clear()
    # 	self.transcription = ""
    # 	if(os.path.exists(self.file_path)):
    # 		os.remove(self.file_path)

    # def crop_audio_after_silence(self, audio_file, min_silence_len=500, silence_thresh=-16):
    
    # 	sound = AudioSegment.from_file(audio_file, format="ogg")
    # 	nonsilent_parts = detect_nonsilent(sound, min_silence_len, silence_thresh)
        
    # 	if nonsilent_parts and len(nonsilent_parts) > 1:
    # 		end_of_first_silence = nonsilent_parts[1][0]
    # 		cropped_audio  = sound[:end_of_first_silence]
    # 		return cropped_audio
    # 	else:
    # 		return sound

    # def dataPreprocess(self, data):
    # 	data = data.partition(",")[2]
    # 	data_decoded = base64.b64decode(data)
    # 	## convert to bytes
    # 	return data_decoded

    # def runWhisper(self, audio, **kwargs):
    # 	data = audio
    # 	if os.path.isfile(data):
    # 		segments, info = self.model.transcribe(data,  vad_filter=True)
    # 	elif ("data:audio" in data):
    # 		audio_bytes = self.dataPreprocess(data)
    # 		with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
    # 			audio_segment = AudioSegment.from_file(io.BytesIO(self.header + audio_bytes),format="webm").export(temp_audio.name, format="ogg")
    # 			segments, info = self.model.transcribe(temp_audio.name,  vad_filter=True)
    # 		pass

    # 	i = 0
    # 	sentence_count = 1
    # 	if "fragment" in kwargs:
    # 		fragment = kwargs["fragment"]
    # 	else:
    # 		fragment = None

    # 	if "timestamp" in kwargs:
    # 		timestamp = kwargs["timestamp"]
    # 	else:
    # 		timestamp = None

    # 	if fragment != None:
    # 		if type(fragment) == str:					
    # 			encodeed_text = self.encoding.encode(fragment)
    # 			for token in encodeed_text:
    # 				self.process_token(token, self.tokens, self.sentences, self.add_sentence_to_list)
    # 		elif type(fragment) == list:
    # 			fragment = " ".join(fragment)
    # 			encodeed_text = self.encoding.encode(fragment)
    # 			for token in encodeed_text:
    # 				self.process_token(token, self.tokens, self.sentences, self.add_sentence_to_list)
    # 		else:
    # 			raise Exception("Fragment must be a string or a list of strings")
    # 		pass
    # 	for segment in segments:
    # 		#print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    # 		encodeed_text = self.encoding.encode(segment.text)
    # 		# self.chunks issue here 
    # 		#self.sentences.append(segment.text)
    # 		for token in encodeed_text:
    # 			this_result = self.process_token(token, self.tokens, self.sentences, self.add_sentence_to_list)
    # 			this_sentence_count = len(self.sentences)
    # 			if this_sentence_count > sentence_count:
    # 				prev_sentence = self.sentences[this_sentence_count - 2]
    # 				self.process_sentence(prev_sentence)
    # 				sentence_count = this_sentence_count
    # 				pass
    # 		i = i + 1
        
    # 	#self.process_sentence(self.sentences[-1])
    # 	result_sentences = self.sentences
    # 	self.sentences = []
    # 	self.tokens = []
    # 	return {
    #         'text': result_sentences,
    # 		'timestamp': timestamp,
    #         'done': True
    #     }


    # def noiseFilter(self):
    # 	if(os.path.getsize(self.file_path) >= self.noiseThreshold):
    # 		return True
    # 	else:
    # 		return False

    # def process_sentence(self, sentence):
    # 	print(sentence)
    # 	return sentence

    # def process_token(self, token, token_list, sentence_list, callback):
    # 	# Add the token to the token list
    # 	token_list.append(token)
    # 	# Join the token list into a single string
    # 	text = self.encoding.decode(token_list)
    # 	# Split the token list into sentences
    # 	sentences = self.nlp.segment(text)
    # 	# Use the callback to add the sentences to the sentence list
    # 	for sentence in sentences:
    # 		if sentence not in sentence_list:
    # 			found = False
    # 			for this_sentence in sentence_list:
    # 				if sentence in this_sentence and len(sentence) > 32 and len(this_sentence) > 32:
    # 					found = True
    # 					break
    # 			if not found:
    # 				callback(sentence_list, sentence)

    # def add_sentence_to_list(self, sentence_list, sentence):
    # 	if len(sentence_list) == 0:
    # 		sentence_list.append(sentence)
    # 	else:
    # 		found = False
    # 		for this_sentence in sentence_list:
    # 			if this_sentence in sentence :
    # 				found = True
    # 				sentence_index = sentence_list.index(this_sentence)
    # 				sentence_list[sentence_index] = sentence
    # 				break
    # 		if not found:
    # 			sentence_list.append(sentence)

    # # def test(self):
    # # 	text = 'The U.S. Supreme Court has said that the purpose of allowing federal officers to move cases against them to federal court is to protect the federal government from operational interference that could occur if federal officials were arrested and tried in state court for actions that fall within the scope of their duties, Pryor wrote. “Shielding officers performing current duties effects the statute’s purpose of protecting the operations of federal government,” he wrote. “But limiting protections to current officers also respects the balance between state and federal interests” by preventing federal interference with state criminal proceedings. Pryor also rejected Meadows’ argument that moving his case to federal court would allow him to assert federal immunity defenses that may apply to former officers, writing that he “cites no authority suggesting that state courts are unequipped to evaluate federal immunities.” The conspiracy to overturn the election alleged in the indictment and the acts of “superintending state election procedures or electioneering on behalf of the Trump campaign” were not related to Meadows’ duties as chief of staff, Pryor wrote. “Simply put, whatever the precise contours of Meadows’s official authority, that authority did not extend to an alleged conspiracy to overturn valid election results,” Pryor wrote. '
    # # 	text_token_list = self.encoding.encode(text) 
        
    # # 	for token in text_token_list:
    # # 		self.process_token(token, self.tokens, self.sentences, self.add_sentence_to_list)
            
    # # 	return self.sentences
    
    # # def test2(self):
    # # 	audio_url = "https://upload.wikimedia.org/wikipedia/commons/f/f9/%22Let_Us_Continue%22_speech_audio_trimmed.ogg"
    # # 	with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
    # # 		subprocess.run(["wget", "-O", temp_audio.name, audio_url])
    # # 		audio = AudioSegment.from_file(temp_audio.name, format="ogg")
    # # 		trimmed_audio = self.crop_audio_after_silence(temp_audio.name)
    # # 		trimmed_audio.export("trimmed_audio.ogg", format="ogg")
    # # 		trimmed_audio = AudioSegment.from_file("trimmed_audio.ogg", format="ogg")
    # # 		audio_length = audio.duration_seconds
    # # 		trimmed_audio_length = trimmed_audio.duration_seconds
    # # 		print("audio length:", audio_length)
    # # 		print("trimmed audio length:", trimmed_audio_length)
    # # 		print("difference:", audio_length - trimmed_audio_length)
    # # 	return [audio_length, trimmed_audio_length, audio_length - trimmed_audio_length]
    
    # def test3(self):
    # 	audio_url = "https://upload.wikimedia.org/wikipedia/commons/f/f9/%22Let_Us_Continue%22_speech_audio_trimmed.ogg"
    # 	with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
    # 		subprocess.run(["wget", "-O", temp_audio.name, audio_url])
    # 		trimmed_audio = self.crop_audio_after_silence(temp_audio.name)
    # 		trimmed_audio.export("trimmed_audio.ogg", format="ogg")
    # 		trimmed_audio = AudioSegment.from_file("trimmed_audio.ogg", format="ogg")
    # 		self.runWhisper("trimmed_audio.ogg")

    # def test4(self):
    # 	audio_url = "https://upload.wikimedia.org/wikipedia/commons/f/f9/%22Let_Us_Continue%22_speech_audio_trimmed.ogg"
    # 	fragment = "Dunkin' Donuts LLC,[1] doing business as Dunkin' since 2019, is an American multinational coffee and doughnut company, as well as a quick service restaurant. It was founded by Bill Rosenberg (1916–2002) in Quincy, Massachusetts, in 1950. The chain was acquired by Baskin-Robbins's holding company Allied Lyons in 1990; its acquisition of the Mister Donut chain and the conversion of that chain to Dunkin' Donuts facilitated the brand's growth in North America that year.[5] Dunkin' and Baskin-Robbins eventually became subsidiaries of Dunkin' Brands, headquartered in Canton, Massachusetts, in 2004, until being purchased by Inspire Brands on December 15, 2020. The chain began rebranding as a beverage-led company, and was renamed Dunkin', in January 2019; while stores in the U.S. began using the new name, the company intends to roll out the rebranding to all of its international stores eventually."
    # 	fragment_split = fragment.split(" ")
    # 	with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
    # 		subprocess.run(["wget", "-O", temp_audio.name, audio_url])
    # 		trimmed_audio = self.crop_audio_after_silence(temp_audio.name)
    # 		trimmed_audio.export("trimmed_audio.ogg", format="ogg")
    # 		trimmed_audio = AudioSegment.from_file("trimmed_audio.ogg", format="ogg")
    # 		start_timestamp = datetime.datetime.now()
    # 		self.runWhisper("trimmed_audio.ogg", fragment_split)
    # 		end_timestamp = datetime.datetime.now()
    # 		print("time taken:")
    # 		print(end_timestamp - start_timestamp)

    # def test5(self):
    # 	this_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base64ogg.txt")
    # 	with open(this_file, "r") as file:
    # 		base64 = file.read()
    # 	fragment = " "
    # 	#fragment = "Dunkin' Donuts LLC,[1] doing business as Dunkin' since 2019, is an American multinational coffee and doughnut company, as well as a quick service restaurant. It was founded by Bill Rosenberg (1916–2002) in Quincy, Massachusetts, in 1950. The chain was acquired by Baskin-Robbins's holding company Allied Lyons in 1990; its acquisition of the Mister Donut chain and the conversion of that chain to Dunkin' Donuts facilitated the brand's growth in North America that year.[5] Dunkin' and Baskin-Robbins eventually became subsidiaries of Dunkin' Brands, headquartered in Canton, Massachusetts, in 2004, until being purchased by Inspire Brands on December 15, 2020. The chain began rebranding as a beverage-led company, and was renamed Dunkin', in January 2019; while stores in the U.S. began using the new name, the company intends to roll out the rebranding to all of its international stores eventually."
    # 	fragment_split = fragment.split(" ")
    # 	results = self.runWhisper(base64, fragment_split)
    # 	return results

