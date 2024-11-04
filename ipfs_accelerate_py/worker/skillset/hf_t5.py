import re
from torch import inference_mode, float16
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import TextStreamer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','skillset')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','worker')))
import worker 

from chat_format import chat_format
from transformers import T5Tokenizer, T5ForConditionalGeneration

class hf_t5:
	def __init__(self, resources, meta=None):
		self.tokenizer = T5Tokenizer.from_pretrained(
			resources['checkpoint'], 
			local_files_only=True,
			device_map='auto',
			legacy=False
		)
		self.model = T5ForConditionalGeneration.from_pretrained(
			resources['checkpoint'], 
			local_files_only=True, 
			low_cpu_mem_usage=True,
			device_map='auto',
			torch_dtype=float16,
		).eval()
		self.worker = worker
		self.TaskAbortion = self.worker.TaskAbortion
		self.should_abort = self.worker.should_abort

	def __call__(self, method, **kwargs):
		if method == 'instruct_t5':
			return self.instruct_t5(**kwargs)
		elif method == 'unmask_t5':
			return self.unmask_t5(**kwargs)
		else:
			raise Exception('unknown method: %s' % method)

	def instruct_t5(self, instruction, input, max_tokens,  **kwargs):

		input_ids = self.tokenizer(instruction + input , return_tensors="pt").input_ids.to(self.model.device)
		outputs = self.model.generate(input_ids, max_length=max_tokens)
		#print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

		return {
            'text': self.tokenizer.decode(outputs[0], skip_special_tokens=True), 
            'done': True
        }

	def unmask_t5(self, masked_words, input, max_tokens, **kwargs):
		if masked_words is None:
			masked_words = []
		if isinstance(masked_words, str):
			masked_words = [masked_words]
		if not isinstance(masked_words, list):
			raise Exception('masked_words must be a list of strings')
		else:
			masked_words_len = len(masked_words)
			for i in range(masked_words_len):
				input = input.replace(masked_words[i], "<extra_id_"+str(i)+">")
			pass
                
		input_ids = self.tokenizer(input, return_tensors="pt").input_ids.to(self.model.device)
		sequence_ids = self.model.generate(input_ids, max_length=max_tokens )
		sequence = self.tokenizer.decode(sequence_ids[0])
		print(sequence)
		sequences = []
		for i in range(len(sequence_ids)):
			sequences.append(self.tokenizer.decode(sequence_ids[i]))
		print(sequences)
		return {
            'text': sequences, 
            'done': True
        }
    
#if __name__ == '__main__':
#    test_t5 = HF_T5({'checkpoint': '/storage/cloudkit-models/flan-t5-small@hf/'})
#    print(test_t5.task("translate English to German: ", "How old are you?", 100))