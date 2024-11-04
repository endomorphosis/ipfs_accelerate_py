import os
import sys
import re
import gc
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','skillset')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','worker')))
from chat_format import chat_format
import worker
import llama_cpp
from llama_cpp import Llama
# from worker import TaskAbortion, should_abort
class llama_cpp:
	def __init__(self, resources, meta=None):
		if meta is not None and type(meta) is dict:
			if "chat_template" in meta:
				self.chat_template = "openai"
				pass
		self.chat_format = chat_format(resources)
		self.text_complete = self.text_complete
		self.instruct = self.instruct
		self.llama_cpp_chat = self.llama_cpp_chat
		self.chat_logits = self.chat_logits
		self.chat = self.chat
		self.llm_complete = self.llm_complete
		n_ctx = None
		n_gqa = None
		self.worker = worker
		self.TaskAbortion = self.worker.TaskAbortion
		self.should_abort = self.worker.should_abort
		if meta is not None and type(meta) is dict:
			if "contextSize" in list(meta.keys()):
				n_ctx = meta['contextSize']
			else:
				n_ctx = 2048
			if "quantization" in list(meta.keys()):
				this_parameters = meta['quantization']
				if int(this_parameters.replace("Q","").replace("_","")) > 69 * 1024 * 1024 * 1024:
					n_gqa = 8
				else:
					n_gqa = None
					pass
			else:
				n_gqa = None
				pass
		if "checkpoint" in list(resources.keys()):
			if "Q8_0" in resources['checkpoint']:
				n_gqa = 8
		else:
			n_gqa = None

		if n_ctx is None:
			n_ctx = 2048
		if n_gqa is None:
			n_gqa = 8

		checkpoint_directory = resources['checkpoint'].replace('@gguf','')
		checkpoint_files = os.listdir(checkpoint_directory)
		checkpoint_files = [file for file in checkpoint_files if file.endswith(".gguf")]

		self.model_name = resources['checkpoint'].split('@')[0]
		self.model = Llama(
			model_path=os.path.join(resources['checkpoint'].replace("@gguf",""), checkpoint_files[0]),
			n_gpu_layers=-1,
			n_ctx=n_ctx,
			n_gqa=n_gqa,
		)

	def __call__(self, method, **kwargs):
		if method == 'llm_chat':
			return self.chat(**kwargs)
		elif method == 'llm_chat_logits':
			return self.chat_logits(**kwargs)
		elif method == 'llm_complete':
			return self.llm_complete(**kwargs)
		elif method == 'llm_logits':
			return self.logits(**kwargs)
		if method == 'text_complete':
			return self.text_complete(**kwargs)
		elif method == 'instruct':
			return self.instruct(**kwargs)
		elif method == 'llama_cpp_chat':
			return self.llama_cpp_chat(**kwargs)
		elif method == 'llama_cpp':
			return self.chat(**kwargs)
		else:
			raise Exception('unknown method: %s' % method)

	def chat(self, messages, system=None, **kwargs):
		messages = [{'role': m['role'], 'content': m['text']} for m in messages]

		if system and messages[0]['role'] != 'system':
			messages = [
				{'role': 'system', 'content': system},
				*messages
			]



		prompt, stop = self.chat_format(
			self.chat_template, 
			messages=messages
		)
		print("chat_format")
		print("Prompt: ")
		print(prompt)
		return self.llm_complete(prompt, stop=stop, **kwargs)
	

	def llama_cpp_chat(self, messages, system=None, **kwargs):
		#template = self.chat_template	
		template = {
				'system_msg': 'A chat between a curious user and an artificial intelligence assistant. '
				'The assistant gives helpful, detailed, and polite answers to the user\'s questions.',
				'user_msg': 'USER: {text}',
				'user_sep': '</s>',
				'assistant_msg': 'ASSISTANT: {text}',
				'assistant_sep': '</s>',
			}

		prompt = '%s\n\n%s' % (
			system if system else template['system_msg'],
			'\n'.join([
				'%s%s' % (
					template['%s_msg' % m['role']].format(text=m['content']),
					template['%s_sep' % m['role']]
					if m != messages[-1] else ''
				)
				for m in messages
			])
		)
		if messages[-1]['role'] == 'user':
			prompt += '\n%s' % template['assistant_msg'].format(text='').rstrip()
	
		print("Prompt: ")
		print(prompt)

		if "prompt" in list(kwargs.keys()):
			del kwargs['prompt']

		return self.text_complete(str(prompt), **kwargs)


	def chat_logits(self, messages, logits_for, chat_template=None, system=None, **kwargs):
		parsed_messages = []
		for m in messages:
			this_message = {}
			this_message['role'] = m['role']
			if('text' in list(m.keys())):
				this_message['content'] = m['text']
			elif('content' in list(m.keys())):
				this_message['content'] = m['content']
			parsed_messages.append(this_message)
		
		results = self.chat_format.format_chat_prompt(
			chat_template, 
			parsed_messages,
		)
		prompt = results[0]
		stop = results[1]

		logits = self.logits(prompt=prompt, logits_for=logits_for)
		return logits
	
	def llm_complete(self, prompt, max_tokens = 256, temperature = 0, stream = True, stopping_regex=None, logit_bias=None, stop=None):
		if stopping_regex:
			try:
				stopping_regex = re.compile(stopping_regex)
			except Exception as e:
				raise Exception('bad "stopping_regex": %s' % str(e))
			
		if logit_bias:
			logit_bias = {
				self.model.tokenize(s.encode('utf-8'), add_bos=False)[0]: v 
				for s, v in logit_bias.items()
			}

		result = ''


		if logit_bias is None:
			streamer = self.model(
				prompt,
				max_tokens=max_tokens,
				temperature=temperature,
				stop=stop,
				stream=True
			)
		else:
			streamer = self.model(
				prompt,
				max_tokens=max_tokens,
				temperature=temperature,
				stop=stop,
				logit_bias=logit_bias,
				stream=True
			)

		for chunk in streamer:
			if self.should_abort():
				raise self.TaskAbortion

			text = chunk['choices'][0]['text']

			if text == '':
				continue
			
			if stopping_regex and stopping_regex.search(result + text):
				break
				
			result += text

			if stream:
				yield {
					'text': text,
					'done': False
				}

		yield {
			'text': '' if stream else result,
			'done': True
		}

	def logits(self, prompt, logits_for):
		prompt_tokens = self.model.tokenize(
			prompt.encode('utf-8'),
			#special=True
		)

		self.model.reset()
		self.model.eval(prompt_tokens)

		return {
			'logits': {
				token: float(self.model._scores[-1, self.model.tokenize(token.encode(), False)[0]])
				for token in logits_for
			}
		}


	def instruct(self, context, instruction, **kwargs):

		models2 = [
			"Platypus2-70B-instruct-4bit",
			"platypus2-70b-4bit"
		]
		template2 = " \
		### Instruction: \n \
		<prompt> \n \
		### Response: \n \
		"

		models =[
			'airoboros-33b-4bit',
			'airoboros-33b-gpt4-1.4-superhot-8K-4bit',
			'airoboros-7b-gpt4-1.4-4bit',
			'airoboros-7b-gpt4-1-4-superhot-8K-4bit',
			'airoboros-l2-13b-gpt4-2.0-4bit',
			'airoboros-l2-7b-gpt4-2.0-4bit',
			'airoboros-l2-70b-gpt4-2.0-4bit'
		]
		if self.model_name in models:

			context_input = ""
			for item in context:
				this_context = item
				this_context_index = context.index(item)
				metadata = ""
				data = ""
				if "metadata" in this_context:
					metadata = this_context["metadata"]
				if "data" in this_context:
					data = this_context["data"]
				if metadata != "" and data != "":
					context_input = context_input + "\n \
					BEGININPUT \n \
					BEGINCONTEXT \n \
					{metadata} \n \
					ENDCONTEXT \n \
					{context} \n \
					ENDINPUT \n \
					"
			instruction_input = "\n \
				BEGININSTRUCTION \n \
				{instruction} \n \
				ENDINSTRUCTION \n \
			"
			prompt =	"\
				USER: \n \
				{context_input}	\
				{instruction_input}	\
				ASSISTANT: \n\
				"
			return self.text_complete(prompt, **kwargs)
		else:
			raise Exception('bad "model": %s' % self.model_name)

	def text_complete(self, prompt, max_tokens = 256, temperature = 0, stream=None, stopping_regex=None):
		if stopping_regex:
			try:
				stopping_regex = re.compile(stopping_regex)
			except Exception as e:
				raise Exception('bad "stopping_regex": %s' % str(e))

		result = ''
		streamer = self.model(
			prompt,
			max_tokens=max_tokens,
			temperature=temperature,
			stream=True
		)

		for chunk in streamer:
			if self.should_abort():
				raise self.TaskAbortion

			text = chunk['choices'][0]['text']

			if text == '':
				continue
			
			if stopping_regex and stopping_regex.search(result + text):
				break
				
			result += text

			if stream:
				yield {
					'text': text,
					'done': False
				}

		yield {
			'text': '' if stream else result,
			'done': True
		}

	def unload(self):
		del self.model
		gc.collect()


	def test(self, **kwargs):
		
		pass


#if __name__ == "__main__":
#    this_test = llama_cpp_kit({'checkpoint': '/storage/cloudkit-models/law-LLM-GGUF-Q2_K@gguf/'})
#    this_test_results = this_test.test()
#    print(this_test_results)