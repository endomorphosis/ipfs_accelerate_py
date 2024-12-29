import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoConfig, AutoTokenizer, AutoModelForImageTextToText, pipeline
# from transformers.generation.streamers import TextStreamer
from ipfs_transformers_py import AutoModel
import torch
from torch import Tensor as T
import torchvision 
from torchvision.transforms import InterpolationMode, Compose, Lambda, Resize, ToTensor, Normalize
import torch 
import asyncio
import openvino as ov
from pathlib import Path
import numpy as np
import torch
import json
import time
import os
import tempfile
import openvino_genai as ov_genai
from transformers import TextStreamer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def streamer(subword: str) -> bool:
    """

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    """
    print(subword, end="", flush=True)

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

def load_image_bak(image_file, input_size=448, max_num=12):
    if os.path.exists(image_file):
        image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    if os.path.exists(image_file):
        image = Image.open(image_file).convert('RGB')
    elif "http" in image_file:
        try:
            with tempfile.NamedTemporaryFile(delete=True) as f:
                f.write(requests.get(image_file).content)
                image = Image.open(f).convert('RGB')
        except Exception as e:
            print(e)
            raise ValueError("Invalid image file")
    else:
        raise ValueError("Invalid image file")
        
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return image, ov.Tensor(image_data)

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
        self.create_openvino_genai_vlm_endpoint_handler = self.create_openvino_genai_vlm_endpoint_handler
        self.create_optimum_vlm_endpoint_handler = self.create_optimum_vlm_endpoint_handler
        self.build_transform = build_transform
        self.load_image = load_image
        self.load_image_tensor = load_image_tensor
        self.dynamic_preprocess = dynamic_preprocess
        self.load_image_bak = load_image_bak
        self.find_closest_aspect_ratio = find_closest_aspect_ratio
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init = self.init
        self.__test__ = self.__test__
        return None
    
    def init(self):
        return None
    
    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        sentence_2 = "The quick brown fox jumps over the lazy dog"
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, image_1)
        except Exception as e:
            print(e)
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens = tokenizer[endpoint_label]()
        len_tokens = len(tokens["input_ids"])
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"tokens: {len_tokens}")
        print(f"tokens per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        if "openvino" not in endpoint_label:
            with torch.no_grad():
                if "cuda" in dir(torch):
                    torch.cuda.empty_cache()
        print("hf_llava test")
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

    def init_openvino(self, model , model_type, device, openvino_label, get_openvino_genai_pipeline, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type, openvino_cli_convert ):
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        homedir = os.path.expanduser("~")
        model_name_convert = model.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache/huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
        huggingface_cache_models_files_dirs = [os.path.join(huggingface_cache_models, file) for file in huggingface_cache_models_files if os.path.isdir(os.path.join(huggingface_cache_models, file))]
        huggingface_cache_models_files_dirs_models = [ x for x in huggingface_cache_models_files_dirs if "model" in x ]
        huggingface_cache_models_files_dirs_models_model_name = [ x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x ]
        model_src_path = os.path.join(huggingface_cache_models, huggingface_cache_models_files_dirs_models_model_name[0])
        model_dst_path = os.path.join(model_src_path, "openvino")
        config = AutoConfig.from_pretrained(model)
        task = get_openvino_pipeline_type(model, model_type)
        openvino_index = int(openvino_label.split(":")[1])
        weight_format = ""
        if openvino_index is not None:
            if openvino_index == 0:
                weight_format = "int8" ## CPU
            if openvino_index == 1:
                weight_format = "int4" ## gpu
            if openvino_index == 2:
                weight_format = "int4" ## npu
        model_dst_path = model_dst_path+"_"+weight_format
        if not os.path.exists(model_dst_path):
            os.makedirs(model_dst_path)
            openvino_cli_convert(model, model_dst_path=model_dst_path, task=task, weight_format=weight_format, ratio="1.0", group_size=128, sym=True )
        tokenizer =  AutoProcessor.from_pretrained(
            model_dst_path, patch_size=config.vision_config.patch_size, vision_feature_select_strategy=config.vision_feature_select_strategy
        )
        # genai_model = get_openvino_genai_pipeline(model, model_type, openvino_label)
        model = get_optimum_openvino_model(model, model_type)
        endpoint_handler = self.create_openvino_vlm_endpoint_handler(model, tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size          
    
    def create_optimum_vlm_endpoint_handler(self, cuda_endpoint_handler, local_cuda_processor, endpoint_model, cuda_label):
        def handler(x, y, cuda_endpoint_handler=cuda_endpoint_handler, local_cuda_processor=local_cuda_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
                try:
                    if y is not None and type(y) == str:
                        image = load_image(y)
                    elif type(y) == tuple:
                        image = load_image(y[1])
                    elif type(y) == dict:
                        image = load_image(y["image"])
                    elif type(y) == list:
                        image = load_image(y[1])
                    else:
                        image = Image.open(requests.get(y, stream=True).raw)
                    
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
                    result = None
                    # prompt = local_cuda_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    # inputs = local_cuda_processor(image, prompt, return_tensors="pt").to(cuda_label, torch.float16)
                    # output = cuda_endpoint_handler.generate(**inputs, max_new_tokens=30)
                    # result = local_cuda_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    return result
                except Exception as e:
                    raise e
        return handler

    def create_openvino_genai_vlm_endpoint_handler(self, openvino_endpoint_handler, openvino_processor, endpoint_model, openvino_label):
        def handler(x, y, openvino_endpoint_handler=openvino_endpoint_handler, openvino_processor=openvino_processor, endpoint_model=endpoint_model, openvino_label=openvino_label):
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = 100

            try:
                if y is not None and type(y) == str and "http" in y:
                    max_retries = 3
                    retry_delay = 1
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(y, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content)).convert('RGB')
                            image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
                            image_tensor = ov.Tensor(image_data)
                            break
                        except (requests.RequestException, Image.UnidentifiedImageError) as e:
                            if attempt == max_retries - 1:
                                raise ValueError(f"Failed to load image from URL after {max_retries} attempts: {y}. Error: {str(e)}")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                elif y is not None and type(y) == str:
                    image = load_image(y)
                elif type(y) == tuple:
                    image = load_image(y[1])
                elif type(y) == dict:
                    image = load_image(y["image"])
                elif type(y) == list:
                    image = load_image(y[1])
                else:
                    image = Image.open(requests.get(y, stream=True).raw)
                
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
                prompt = x
                output = openvino_endpoint_handler.generate(prompt, image=image_tensor, generation_config=config, streamer=streamer)
                # Run model inference
                return output
            except Exception as e:
                # Cleanup GPU memory in case of error
                raise e
        return handler

    
    def create_openvino_vlm_endpoint_handler(self, openvino_endpoint_handler, local_openvino_processor, endpoint_model, cuda_label):
        def handler(x, y, openvino_endpoint_handler=openvino_endpoint_handler, local_openvino_processor=local_openvino_processor, endpoint_model=endpoint_model, cuda_label=cuda_label):
                try:
                    if y.startswith("http") or y.startswith("https"):
                        response = requests.get(y)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(y).convert("RGB")
                    if x is not None and type(x) == str:
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": x},
                                    {"type": "image"}
                                ]
                            }
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
                                    {"type": "text", "text": x},
                                    {"type": "image"}
                                ]
                            }
                        ]
                        
                    else:
                        raise Exception("Invalid input to vlm endpoint handler")
                    result = None
                    streamer = TextStreamer(local_openvino_processor, skip_prompt=True, skip_special_tokens=True)
                    prompt = local_openvino_processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = local_openvino_processor(image, prompt, return_tensors="pt")

                    output_ids = endpoint_model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=50,
                        streamer=streamer,
                    )
                    outputs = local_openvino_processor.decode(output_ids[0], skip_special_tokens=True)
                    # result = local_openvino_processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    return outputs
                except Exception as e:
                    raise e
        return handler

    
hf_llava = hf_llava()