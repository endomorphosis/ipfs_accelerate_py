import requests
from PIL import Image
from io import BytesIO
import asyncio
from pathlib import Path
import json
import time
import os
import tempfile
import numpy as np
import torch
from torchvision.transforms import InterpolationMode, Compose, Lambda, Resize, ToTensor, Normalize

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def streamer(subword: str) -> bool:
    """
    Stream tokens as they are generated
    
    Args:
        subword: The subword/token to stream
        
    Returns:
        Boolean indicating whether to continue streaming
    """
    print(subword, end="", flush=True)
    return True

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Find the closest aspect ratio to the target to minimize distortion
    
    Args:
        aspect_ratio: Original aspect ratio
        target_ratios: List of target aspect ratios to choose from
        width: Original image width
        height: Original image height
        image_size: Target size for the image
        
    Returns:
        Tuple of (width, height) for the resized image
    """
    closest_ratio = min(target_ratios, key=lambda r: abs(r - aspect_ratio))
    
    if closest_ratio > 1:  # width > height
        new_width = image_size
        new_height = int(image_size / closest_ratio)
    else:  # height > width
        new_height = image_size
        new_width = int(image_size * closest_ratio)
        
    return new_width, new_height

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess image based on its properties
    
    Args:
        image: PIL Image to process
        min_num: Minimum number of image patches
        max_num: Maximum number of image patches
        image_size: Target image size
        use_thumbnail: Whether to use image thumbnail
        
    Returns:
        Processed image ready for model input
    """
    width, height = image.size
    aspect_ratio = width / height
    
    # Common target aspect ratios
    target_ratios = [0.5, 0.75, 1.0, 1.33, 1.5, 2.0]
    
    new_width, new_height = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    )
    
    # Resize the image
    if use_thumbnail:
        image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        image = image.resize((new_width, new_height), Image.BICUBIC)
    
    # Additional preprocessing for model input would go here
    transform = build_transform(image_size)
    tensor = transform(image)
    
    return tensor

def load_image(image_file):
    """
    Load image from file path or URL
    
    Args:
        image_file: Path or URL to image
        
    Returns:
        PIL Image object
    """
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        import requests
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_image_tensor(image_file):
    """
    Load image and convert directly to tensor
    
    Args:
        image_file: Path or URL to image
        
    Returns:
        Image as tensor ready for model input
    """
    image = load_image(image_file)
    transform = build_transform(448)  # Default size
    return transform(image).unsqueeze(0)  # Add batch dimension

class hf_llava_next:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_vlm_endpoint_handler = self.create_openvino_vlm_endpoint_handler
        self.create_openvino_genai_vlm_endpoint_handler = self.create_openvino_genai_vlm_endpoint_handler
        self.create_optimum_vlm_endpoint_handler = self.create_optimum_vlm_endpoint_handler
        self.create_cuda_vlm_endpoint_handler = self.create_cuda_vlm_endpoint_handler
        self.create_cpu_vlm_endpoint_handler = self.create_cpu_vlm_endpoint_handler
        self.build_transform = build_transform
        self.load_image = load_image
        self.load_image_tensor = load_image_tensor
        self.dynamic_preprocess = dynamic_preprocess
        self.find_closest_aspect_ratio = find_closest_aspect_ratio
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init = self.init
        self.__test__ = self.__test__
        self.init_cpu = self.init_cpu
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        self.create_cpu_llava_endpoint_handler = self.create_cpu_llava_endpoint_handler
        self.create_cuda_llava_endpoint_handler = self.create_cuda_llava_endpoint_handler
        self.create_openvino_llava_endpoint_handler = self.create_openvino_llava_endpoint_handler
        self.create_apple_llava_endpoint_handler = self.create_apple_llava_endpoint_handler
        self.create_qualcomm_llava_endpoint_handler = self.create_qualcomm_llava_endpoint_handler
        return None
    

    def init(self):
        
        if "torch" not in list(self.resources.keys()):
            import torch
            self.torch = torch
        else:
            self.torch = self.resources["torch"]

        if "transformers" not in list(self.resources.keys()):
            import transformers
            self.transformers = transformers
        else:
            self.transformers = self.resources["transformers"]
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]

        if "decord" not in list(self.resources.keys()):
            import decord
            self.decord = decord
        else:
            self.decord = self.resources["decord"]
        self.np.random.seed(0)
        return None
    
    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        sentence_2 = "The quick brown fox jumps over the lazy dog"
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, image_1)
            print(test_batch)
            print("hf_llava test passed")
        except Exception as e:
            print(e)
            print("hf_llava test failed")
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
            with self.torch.no_grad():
                if "cuda" in dir(self.torch):
                    self.torch.cuda.empty_cache()
        print("hf_llava test")
        return None
    
    def init_cuda(self, model, device, cuda_label):
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoProcessor.from_pretrained(model)
        endpoint = None
        try:
            endpoint = self.transformers.AutoModelForImageTextToText.from_pretrained(model,  torch_dtype=self.torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_vlm_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        self.torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0

    def init_openvino(self, model , model_type, device, openvino_label, get_openvino_genai_pipeline, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type, openvino_cli_convert ):
        self.init()
        if "openvino" not in list(self.resources.keys()):
            import openvino as ov
            self.ov = ov
        else:
            self.ov = self.resources["openvino"]
            
        if "ov_genai" not in list(self.resources.keys()):
            import openvino_genai as ov_genai
            self.ov_genai = ov_genai
        else:
            self.ov_genai = self.resources["ov_genai"]
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
        config = self.transformers.AutoConfig.from_pretrained(model)
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
        tokenizer =  self.transformers.AutoProcessor.from_pretrained(
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
            config = self.ov_genai.GenerationConfig()
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
                            image_data = self.np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(self.np.byte)
                            image_tensor = self.ov.Tensor(image_data)
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
                    streamer = self.transformers.TextStreamer(local_openvino_processor, skip_prompt=True, skip_special_tokens=True)
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

    def create_cpu_vlm_endpoint_handler(self, openvino_endpoint_handler, local_openvino_processor, endpoint_model, cuda_label):
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
                    streamer = self.transformers.TextStreamer(local_openvino_processor, skip_prompt=True, skip_special_tokens=True)
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
    
    
    def create_cuda_vlm_endpoint_handler(self, openvino_endpoint_handler, local_openvino_processor, endpoint_model, cuda_label):
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
                    streamer = self.transformers.TextStreamer(local_openvino_processor, skip_prompt=True, skip_special_tokens=True)
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
    
    def init_cpu(self, model, device, cpu_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)
        tokenizer = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
        try:
            endpoint = self.transformers.AutoModelForVision2Seq.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            print(e)
            try:
                endpoint = self.transformers.LlavaNextForConditionalGeneration.from_pretrained(model, trust_remote_code=True)
            except Exception as e:
                print(e)
                pass
        endpoint_handler = self.create_cpu_llava_endpoint_handler(endpoint, tokenizer, model, cpu_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size
    
    def init_apple(self, model, device, apple_label):
        self.init()
        try:
            import coremltools
        except ImportError:
            print("coremltools not installed. Can't initialize Apple backend.")
            return None, None, None, None, 0
            
        tokenizer = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
        endpoint = None  # In real implementation, load the CoreML model here
        endpoint_handler = self.create_apple_llava_endpoint_handler(endpoint, tokenizer, model, apple_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size
    
    def init_qualcomm(self, model, device, qualcomm_label):
        """
        Initialize LLaVA model for Qualcomm hardware
        
        Args:
            model: HuggingFace model name or path
            device: Device to run inference on
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            Tuple of model components for Qualcomm execution
        """
        self.init()
        
        # Import SNPE utilities
        try:
            from .qualcomm_snpe_utils import get_snpe_utils
            self.snpe_utils = get_snpe_utils()
        except ImportError:
            print("Failed to import Qualcomm SNPE utilities")
            return None, None, None, None, 0
            
        if not self.snpe_utils.is_available():
            print("Qualcomm SNPE is not available on this system")
            return None, None, None, None, 0
            
        try:
            # Load tokenizer directly from HuggingFace
            processor = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with SNPE
            model_name = model.replace("/", "--")
            dlc_path = f"~/snpe_models/{model_name}_llava.dlc"
            dlc_path = Path(dlc_path).expanduser()
            
            # Create directory if needed
            dlc_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert or load the model
            if not dlc_path.exists():
                print(f"Converting {model} to SNPE format...")
                self.snpe_utils.convert_model(model, "llava", str(dlc_path))
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Create endpoint handler
            endpoint_handler = self.create_qualcomm_llava_endpoint_handler(
                endpoint, processor, model, qualcomm_label
            )
            
            import asyncio
            return endpoint, processor, endpoint_handler, asyncio.Queue(16), 1
        except Exception as e:
            print(f"Error initializing Qualcomm LLaVA model: {e}")
            return None, None, None, None, 0
    
    def create_cpu_llava_endpoint_handler(self, endpoint, tokenizer, model, cpu_label):
        def handler(text, image=None, endpoint=endpoint, tokenizer=tokenizer, model=model, cpu_label=cpu_label):
            if image is not None:
                if isinstance(image, str):
                    image = load_image(image)
                
                inputs = tokenizer(text=text, images=image, return_tensors="pt")
                with self.torch.no_grad():
                    outputs = endpoint.generate(**inputs, max_new_tokens=256)
                    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                return result
            else:
                inputs = tokenizer(text=text, return_tensors="pt")
                with self.torch.no_grad():
                    outputs = endpoint.generate(**inputs, max_new_tokens=256)
                    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                return result
        return handler
    
    def create_cuda_llava_endpoint_handler(self, endpoint, tokenizer, model, cuda_label):
        def handler(text, image=None, endpoint=endpoint, tokenizer=tokenizer, model=model, cuda_label=cuda_label):
            try:
                if "eval" in dir(endpoint):
                    endpoint.eval()
                
                with self.torch.no_grad():
                    self.torch.cuda.empty_cache()
                    
                    if image is not None:
                        if isinstance(image, str):
                            image = load_image(image)
                        
                        inputs = tokenizer(text=text, images=image, return_tensors="pt").to(cuda_label)
                        outputs = endpoint.generate(**inputs, max_new_tokens=256)
                        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                    else:
                        inputs = tokenizer(text=text, return_tensors="pt").to(cuda_label)
                        outputs = endpoint.generate(**inputs, max_new_tokens=256)
                        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                    
                    self.torch.cuda.empty_cache()
                    return result
            except Exception as e:
                self.torch.cuda.empty_cache()
                raise e
        return handler
    
    def create_openvino_llava_endpoint_handler(self, endpoint, tokenizer, model, openvino_label):
        def handler(text, image=None, endpoint=endpoint, tokenizer=tokenizer, model=model, openvino_label=openvino_label):
            try:
                if image is not None:
                    if isinstance(image, str):
                        image = load_image(image)
                    
                    inputs = tokenizer(text=text, images=image, return_tensors="pt")
                    outputs = endpoint.generate(**inputs, max_new_tokens=256)
                    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                else:
                    inputs = tokenizer(text=text, return_tensors="pt")
                    outputs = endpoint.generate(**inputs, max_new_tokens=256)
                    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                
                return result
            except Exception as e:
                print(f"Error in OpenVINO LLaVA handler: {e}")
                raise e
        return handler
    
    def create_apple_llava_endpoint_handler(self, endpoint, tokenizer, model, apple_label):
        def handler(text, image=None, endpoint=endpoint, tokenizer=tokenizer, model=model, apple_label=apple_label):
            try:
                # Implementation for Apple silicon would go here
                # This would use the CoreML model loaded in init_apple
                result = "Apple Silicon implementation not available yet"
                return result
            except Exception as e:
                print(f"Error in Apple LLaVA handler: {e}")
                raise e
        return handler
    
    def create_qualcomm_llava_endpoint_handler(self, endpoint, processor, model_name, qualcomm_label):
        """
        Create endpoint handler for Qualcomm LLaVA model
        
        Args:
            endpoint: Loaded SNPE model
            processor: HuggingFace processor
            model_name: Name of the model
            qualcomm_label: Label for the endpoint
            
        Returns:
            Handler function for the endpoint
        """
        def handler(text_input, image_input=None, endpoint=endpoint, processor=processor, 
                   model_name=model_name, qualcomm_label=qualcomm_label):
            try:
                # Process image if provided
                if image_input is not None:
                    if isinstance(image_input, str):
                        image = load_image(image_input)
                        image_tensor = dynamic_preprocess(image)
                    else:
                        # Assume it's already a tensor or PIL image
                        if isinstance(image_input, Image.Image):
                            image_tensor = dynamic_preprocess(image_input)
                        else:
                            image_tensor = image_input
                    
                    # Convert to numpy for SNPE
                    image_np = image_tensor.numpy() if hasattr(image_tensor, 'numpy') else self.np.array(image_tensor)
                
                # Process text input
                if text_input is not None:
                    if isinstance(text_input, str):
                        text_inputs = processor(text=text_input, return_tensors="np")
                    else:
                        text_inputs = text_input
                
                # Prepare inputs for SNPE
                if image_input is not None and text_input is not None:
                    inputs = {
                        "input_ids": text_inputs["input_ids"],
                        "attention_mask": text_inputs["attention_mask"],
                        "pixel_values": image_np
                    }
                elif text_input is not None:
                    inputs = {
                        "input_ids": text_inputs["input_ids"],
                        "attention_mask": text_inputs["attention_mask"]
                    }
                elif image_input is not None:
                    inputs = {
                        "pixel_values": image_np
                    }
                
                # Run inference using SNPE
                results = self.snpe_utils.run_inference(endpoint, inputs)
                
                # Process and return results
                if "logits" in results:
                    # For text generation
                    return {
                        "logits": self.torch.tensor(results["logits"]),
                        "text": processor.decode(results["logits"].argmax(axis=-1)[0])
                    }
                elif "image_embeds" in results:
                    # For image embeddings
                    return {
                        "image_embeds": self.torch.tensor(results["image_embeds"])
                    }
                else:
                    # General case
                    return {k: self.torch.tensor(v) if isinstance(v, self.np.ndarray) else v 
                           for k, v in results.items()}
                
            except Exception as e:
                print(f"Error in Qualcomm LLaVA endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler