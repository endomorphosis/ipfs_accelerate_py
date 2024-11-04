import re
from torch import inference_mode, float16
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from transformers.generation.streamers import TextStreamer
import os
import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'worker')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'worker', 'skillset')))
import worker.worker as worker

chat_templates = [
	{
		'models': [
			"wizard-vicuna-30b-uncensored-4bit"
			'vicuna-7b',
		],
		'system_msg': 'A chat between a curious user and an artificial intelligence assistant. ' + \
        'The assistant gives helpful, detailed, and polite answers to the user\'s questions.',
		'user_msg': 'USER: {text}',
		'user_sep': '</s>',
		'assistant_msg': 'ASSISTANT: {text}',
		'assistant_sep': '</s>',
	},
	{
		'models': ['llama-2-70b-chat-4bit','llama-2-70b-chat-uncensored-4bit'],
		'system_msg': '[INST] <<SYS>> \{A chat between a curious user and an artificial intelligence assistant. ' + \
        'The assistant gives helpful, detailed, and polite answers to the user\'s questions. <</SYS>> [/INST]',
		'user_msg': '[INST] \{model_reply_ ' + '{number}' + '\} {text}[/INST]',
		'user_sep': '</s>',
		'assistant_msg': '\{model_reply_' + '{number}' + '\} ' + '{text}',
		'assistant_sep': '</s>',
	},
	{
		'models': ['solar-0-70b-4bit@gguf','stablebeluga2-70b-4bit','stablebeluga-7b-4bit','stablebeluga-13b-4bit','stable-platapus-13b-4bit','platypus2-70b-4bit','platypus2-70b-instruct-4bit','alpaca'],
		'system_msg': '### System: \n A chat between a curious user and an artificial intelligence assistant. ' + \
        'The assistant gives helpful, detailed, and polite answers to the user\'s questions.',
		'user_msg': '### Instruction: \n{text}',
		'user_sep': '</s>',
		'assistant_msg': '### Response: \n{text}',
		'assistant_sep': '</s>',
	},
]

class hf_lm:
	def __init__(self, resources, metadata=None):
		self.tokenizer = AutoTokenizer.from_pretrained(
			resources['checkpoint'], 
			local_files_only=True,
			legacy=False
		)
		self.model = AutoModelForCausalLM.from_pretrained(
			resources['checkpoint'], 
			local_files_only=True, 
			low_cpu_mem_usage=True,
			device_map='auto',
			torch_dtype=float16,
		).eval()
		if "dispatch_result" in resources:
			self.dispatch_result = resources["dispatch_result"]
		else:
			self.dispatch_result = worker.dispatch_result

	def __call__(self, method, **kwargs):
		if method == 'text_complete':
			return self.text_complete(**kwargs)
		elif method == 'chat':
			return self.chat(**kwargs)
		else:
			raise Exception('unknown method: %s' % method)


	def chat(self, messages, system=None, **kwargs):
		template = chat_templates[0] # todo: pull by model id
		prompt = '%s\n\n%s' % (
			system if system else template['system_msg'],
			'\n'.join([
				'%s%s' % (
					template['%s_msg' % m['role']].format(text=m['text']),
					template['%s_sep' % m['role']]
					if m != messages[-1] else ''
				)
				for m in messages
			])
		)

		if messages[-1]['role'] == 'user':
			prompt += '\n%s' % template['assistant_msg'].format(text='').rstrip()

		return self.text_complete(prompt, **kwargs)


	def text_complete(self, prompt, max_tokens, temperature, stream, stopping_regex=None):
		if stopping_regex:
			try:
				stopping_regex = re.compile(stopping_regex)
			except Exception as e:
				raise Exception('bad "stopping_regex": %s' % str(e))

		inputs = self.tokenizer(
			prompt, 
			return_tensors='pt'
		)

		streamer = ResultStreamer(
			tokenizer=self.tokenizer,
			skip_special_tokens=True,
			skip_prompt=True,
			stopping_regex=stopping_regex,
			emit_chunk=lambda text: self.dispatch_result({
				'text': text,
				'done': False
			}) if stream else None
		)

		with inference_mode():
			self.model.generate(
				inputs=inputs.input_ids.cuda(), 
				max_new_tokens=max_tokens, 
				temperature=min(temperature / 100, 0.0001),
				stopping_criteria=streamer.stopping_criteria,
				streamer=streamer,
			)

		if not stream:
			return {
				'text': streamer.full_text,
				'done': True
			}
		else:
			return {
				'text': '',
				'done': True
			}


class ResultStreamer(TextStreamer):
	def __init__(self, stopping_regex, emit_chunk, **kwargs):
		super().__init__(**kwargs)
		self.full_text = ''
		self.emit_chunk = emit_chunk
		self.stop = False
		self.stopping_regex = stopping_regex
		self.stopping_criteria = StoppingCriteriaList([
			lambda input_ids, score, **kwargs: self.stop
		]) if stopping_regex else None


	def on_finalized_text(self, text, stream_end=False):
		if self.stop:
			return

		if text == '':
			return
		
		full_text = self.full_text + text
		
		if self.stopping_regex and self.stopping_regex.search(full_text):
			self.stop = True
			return
			
		self.full_text = full_text

		if self.emit_chunk:
			self.emit_chunk(text)