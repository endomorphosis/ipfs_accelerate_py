import os
import torch
import torch.nn.functional as F
from torch import inference_mode, float16, Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
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

	def init_cuda(self):
		return None

	def init_openvino(self):
		return None		
			
	def __call__(self, method, **kwargs):
		if method == 'hf_embed':
			return self.embed(**kwargs)
		elif method == 'instruct_embed':
			return self.embed(**kwargs)
		else:
			raise Exception('unknown method: %s' % method)

	def create_openvino_endpoint_handler(self, endpoint_model, openvino_label):
		def handler(x):
			return self.local_endpoints[endpoint_model][openvino_label](x)
		return handler

	def create_endpoint_handler(self, endpoint_model, cuda_label):
		def handler(x):
			if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
				self.local_endpoints[endpoint_model][cuda_label].eval()
			else:
				pass
			with torch.no_grad():
				try:
					torch.cuda.empty_cache()
					# Tokenize input with truncation and padding
					tokens = self.tokenizer[endpoint_model][cuda_label](
						x, 
						return_tensors='pt', 
						padding=True, 
						truncation=True,
						max_length=self.local_endpoints[endpoint_model][cuda_label].config.max_position_embeddings
					)
					
					# Move tokens to the correct device
					input_ids = tokens['input_ids'].to(self.local_endpoints[endpoint_model][cuda_label].device)
					attention_mask = tokens['attention_mask'].to(self.local_endpoints[endpoint_model][cuda_label].device)
					
					# Run model inference
					outputs = self.local_endpoints[endpoint_model][cuda_label](
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

# def test():
# 	cwd = os.getcwd()
# 	dir = os.path.dirname(__file__)
# 	grandparent = os.path.dirname(dir)
# 	models = os.path.join(grandparent, "models")
# 	checkpoint = 'bge-base-en-v1.5'
# 	resources = {}
# 	resources['checkpoint'] = models + "/" + checkpoint + "@hf"
	
# 	print(resources["checkpoint"])
# 	meta = {"modelName":"bge-base-en-v1.5"}
# 	text = "sample text to embed"
# 	model = "bge-base-en-v1.5"
# 	instruction = "Represent this sentence for searching relevant passages:"
# 	embed = hf_embed(resources, meta)
# 	results = embed.embed(instruction, text)
# 	print(results)
# 	return results

# if __name__ == '__main__':
# 	test()
# 	# pass