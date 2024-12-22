from torch import no_grad
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

class hf_clip:
	def __init__(self, resources, meta=None):
		self.model = CLIPModel.from_pretrained(resources['clip'], local_files_only=True).to('cuda')
		self.processor = CLIPProcessor.from_pretrained(resources['clip'], local_files_only=True)
		self.tokenizer  = AutoTokenizer.from_pretrained(resources['clip'], local_files_only=True)

	def __call__(self, method, text=None, image=None):
		if method == 'clip_text':
			inputs = self.tokenizer([text], return_tensors='pt').to('cuda')

			with no_grad():
				text_features = self.model.get_text_features(**inputs)

			return {
				'embedding': text_features[0].cpu().numpy().tolist()
			}
		
		elif method == 'clip_image':
			inputs = self.processor(images=image, return_tensors='pt').to('cuda')

			with no_grad():
				image_features  = self.model.get_image_features(**inputs)

			return {
				'embedding': image_features[0].cpu().numpy().tolist()
			}