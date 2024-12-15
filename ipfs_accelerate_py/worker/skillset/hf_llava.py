import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoConfig
from transformers.generation.streamers import TextStreamer
import torch 
import openvino as ov
from pathlib import Path
import numpy as np
import torch
import json

def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return image, ov.Tensor(image_data)

class hf_llava:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_vlm_endpoint_handler = self.create_openvino_vlm_endpoint_handler
        self.create_vlm_endpoint_handler = self.create_vlm_endpoint_handler
        return None
    
    def init(self):
    #     self.processor = AutoProcessor.from_pretrained(
    #         self.resources['checkpoint'], 
    #         local_files_only=True
    #     )
    #     self.config = AutoConfig.from_pretrained(
    #         self.resources['checkpoint'], 
    #         local_files_only=True
    #     )
    #     self.model = AutoModelForSequenceClassification.from_pretrained(
    #         self.resources['checkpoint'], 
    #         local_files_only=True
    #     )
        return None
    
    def init_cuda(self):
        return None
    
    def init_openvino(self):       
        return None
    
    # def __call__(self, method, **kwargs):
    #     if method == 'text_complete':
    #         return self.text_complete(**kwargs)
    #     elif method == 'chat':
    #         return self.chat(**kwargs)
    #     else:
    #         raise Exception('unknown method: %s' % method)
        
    
    def create_vlm_endpoint_handler(self, local_cuda_endpoint, local_cuda_tokenizer, endpoint_model, cuda_label):
        def handler(x):
            # if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
            #       self.local_endpoints[endpoint_model][cuda_label].eval()
            if "eval" in dir(local_cuda_endpoint):
                local_cuda_endpoint.eval()
            else:
                pass
            with torch.no_grad():
                try:
                    torch.cuda.empty_cache()
                    # Tokenize input with truncation and padding
                    # tokens = self.tokenizer[endpoint_model][cuda_label](
                    #     x, 
                    #     return_tensors='pt', 
                    #     padding=True, 
                    #     truncation=True,
                    #     max_length=self.local_endpoints[endpoint_model][cuda_label].config.max_position_embeddings
                    # )
                    config = AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    tokens = local_cuda_tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=512)
                    
                    # Move tokens to the correct device
                    input_ids = tokens['input_ids'].to(local_cuda_endpoint.device)
                    attention_mask = tokens['attention_mask'].to(local_cuda_endpoint.device)
                    
                    # Run model inference
                    outputs = local_cuda_endpoint(
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

    def create_openvino_vlm_endpoint_handler(self, openvino_endpoint_handler, openvino_tokenizer, endpoint_model, openvino_label):
        def handler(x, y=None):
            chat = None
            image_file = None
            if y is not None and x is not None:
                chat, image_file = x
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
                
            image = load_image(image_file)
            prompt = openvino_endpoint_handler.apply_chat_template(chat, add_generation_prompt=True)
            inputs = openvino_endpoint_handler(images=image, text=prompt, return_tensors="pt")
            streamer = TextStreamer(openvino_tokenizer, skip_prompt=True, skip_special_tokens=True)
            output_ids = openvino_endpoint_handler.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=50,
                streamer=streamer,
            )
        return handler

hf_llava = hf_llava()