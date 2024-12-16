import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoConfig, AutoTokenizer, AutoModelForImageTextToText, pipeline
from transformers.generation.streamers import TextStreamer
import torch
from torch import Tensor as T
from torchvision.transforms import InterpolationMode
import torch 
import asyncio
import openvino as ov
from pathlib import Path
import numpy as np
import torch
import json


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
    
def load_image_tensor(image_file):
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
        return None
    
    def init_cuda(self, model, device, cuda_label):
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = AutoProcessor.from_pretrained(model)
        endpoint = None
        try:
            endpoint = AutoModelForImageTextToText.from_pretrained(model,  torch_dtype=torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_vlm_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
    
    def init_openvino(self, model, model_type, device, openvino_label, get_openvino_model):
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        batch_size = 0                
        tokenizer =  AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        model = get_openvino_model(model, model_type, openvino_label)
        endpoint_handler = self.create_openvino_vlm_endpoint_handler(model, tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size          
    
    def create_vlm_endpoint_handler(self, local_cuda_endpoint, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y=None, local_cuda_endpoint=local_cuda_endpoint, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
            # if "eval" in dir(self.local_endpoints[endpoint_model][cuda_label]):
            #       self.local_endpoints[endpoint_model][cuda_label].eval()
            tmp_url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
            if "eval" in dir(local_cuda_endpoint):
                local_cuda_endpoint.eval()
            else:
                pass
            with torch.no_grad():
                try:
                    torch.cuda.empty_cache()
                    config = AutoConfig.from_pretrained(endpoint_model, trust_remote_code=True)
                    if y is not None and type(y) == str:
                        image = load_image(y)
                    elif type(y) == tuple:
                        image = load_image(y[1])
                    elif type(y) == dict:
                        image = load_image(y["image"])
                    elif type(y) == list:
                        image = load_image(y[1])
                    else:
                        image = Image.open(requests.get(tmp_url, stream=True).raw)
                    
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
                    inputs = local_cuda_processor(image, prompt, return_tensors="pt").to(cuda_label, torch.float16)
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