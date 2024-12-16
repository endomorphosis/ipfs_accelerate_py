import asyncio
import os
import torch
import torch.nn.functional as F
from torch import inference_mode, float16, Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, StoppingCriteriaList
from transformers.generation.streamers import TextStreamer
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
from FlagEmbedding import FlagModel
import json

class hf_embed:
	def __init__(self, resources=None, metadata=None):
		self.modelName = metadata['model_name']

	def init(self):
		return None

	def init_cuda(self, model, device, cuda_label):
		config = AutoConfig.from_pretrained(model, trust_remote_code=True)
		tokenizer = AutoTokenizer.from_pretrained(model, device=device, use_fast=True, trust_remote_code=True)
		try:
			endpoint = AutoModel.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code=True).to(device)
		except Exception as e:
			try:
				endpoint = AutoModel.from_pretrained(model, trust_remote_code=True, device=device)
			except Exception as e:
				print(e)
				pass
		endpoint_handler = self.create_endpoint_handler(endpoint, cuda_label)
		torch.cuda.empty_cache()
		batch_size = 0
		return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size

	def init_openvino(self, model, model_type, device, openvino_label, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type):
		ov_count = 0
		device = "openvino:" + str(ov_count)
		openvino_label = "openvino:" + str(ov_count)
		try:
			tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True,  trust_remote_code=True)
			model_type =  str(get_openvino_pipeline_type(model))
			endpoint = pipeline(model_type, model = get_optimum_openvino_model(model, model_type), tokenizer=tokenizer)
			endpoint_handler = self.create_openvino_endpoint_handler(model, openvino_label)
			batch_size = 0
		# elif self.hwtest["openvino"] == True:                            
		except Exception as e:
			try:
				tokenizer =  AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
				model = get_openvino_model(model, model_type, openvino_label)
				endpoint_handler = lambda x: model({**tokenizer(x, 0, return_tensors='pt')})
				batch_size = 0
			except Exception as e:
				print(e)
				pass
		ov_count = ov_count + 1  
		return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size          

	def create_openvino_endpoint_handler(self, endpoint_model, openvino_label, endpoint=None, ):
		def handler(x):
			if endpoint is None:
				return self.local_endpoints[endpoint_model][openvino_label](x)
			else:
				return endpoint(x)
		return handler

	def create_endpoint_handler(self, endpoint_model, cuda_label, endpoint=None, tokenizer=None):
		def handler(x):
			if "eval" in dir(endpoint):
				endpoint.eval()
			else:
				pass
			with torch.no_grad():
				try:
					torch.cuda.empty_cache()
					# Tokenize input with truncation and padding
					tokens = tokenizer[endpoint_model][cuda_label](
						x, 
						return_tensors='pt', 
						padding=True, 
						truncation=True,
						max_length=endpoint.config.max_position_embeddings
					)
					
					# Move tokens to the correct device
					input_ids = tokens['input_ids'].to(endpoint.device)
					attention_mask = tokens['attention_mask'].to(endpoint.device)
					
					# Run model inference
					outputs = endpoint(
						input_ids=input_ids,
						attention_mask=attention_mask,
						return_dict=True
					)
						
					# Process and prepare outputs
					if hasattr(outputs, 'last_hidden_state'):
						hidden_states = outputs.last_hidden_state.cpu().numpy()
						attention_mask_np = attention_mask.cpu().numpy()
						result = {
							'hidden_states': hidden_states,
							'attention_mask': attention_mask_np
						}
					else:
						result = outputs.to('cpu').detach().numpy()

					# Cleanup GPU memory
					del tokens, input_ids, attention_mask, outputs
					if 'hidden_states' in locals(): del hidden_states
					if 'attention_mask_np' in locals(): del attention_mask_np
					torch.cuda.empty_cache()
					return result
				except Exception as e:
					# Cleanup GPU memory in case of error
					if 'tokens' in locals(): del tokens
					if 'input_ids' in locals(): del input_ids
					if 'attention_mask' in locals(): del attention_mask
					if 'outputs' in locals(): del outputs
					if 'hidden_states' in locals(): del hidden_states
					if 'attention_mask_np' in locals(): del attention_mask_np
					torch.cuda.empty_cache()
					raise e
		return handler

	def embed_bak(self, instruction, text , **kwargs):
		self.input = text
		self.method = 'embed'
		embeddings = None
		if "instructor" in self.modelName:
			embeddings = self.model.encode([[instruction,self.input]])
			print(embeddings)
		if "gte" in self.modelName:
			embeddings = self.model.encode([self.input])
			print(embeddings)
		if "bge" in self.modelName:
			if self.model == None:
				self.model = FlagModel(
					'BAAI/'+self.modelName, query_instruction_for_retrieval=instruction,
					use_fp16=True
				)
			embeddings = self.model.encode(str(self.input))
			print(embeddings)

		if type(embeddings) != str:
			embeddings = json.dumps(embeddings.tolist())

		return {
			'text': embeddings, 
			'done': True
		}
		
	def average_pool_bak(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
		last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
		return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

export = hf_embed
