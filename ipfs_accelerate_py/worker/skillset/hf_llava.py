import requests
from PIL import Image
from io import BytesIO
import asyncio
from pathlib import Path
import json
import time
import os
import tempfile

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    import torch
    from torchvision.transforms import InterpolationMode, Compose, Lambda, Resize, ToTensor, Normalize
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

# def load_image_bak(image_file, input_size=448, max_num=12):
#     if os.path.exists(image_file):
#         image = Image.open(image_file).convert('RGB')
#     transform = build_transform(input_size=input_size)
#     if os.path.exists(image_file):
#         image = Image.open(image_file).convert('RGB')
#     elif "http" in image_file:
#         try:
#             with tempfile.NamedTemporaryFile(delete=True) as f:
#                 f.write(requests.get(image_file).content)
#                 image = Image.open(f).convert('RGB')
#         except Exception as e:
#             print(e)
#             raise ValueError("Invalid image file")
#     else:
#         raise ValueError("Invalid image file")
        
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values

def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_image_tensor(image_file):
    import openvino as ov
    import numpy as np
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
        self.create_cuda_vlm_endpoint_handler = self.create_cuda_vlm_endpoint_handler
        self.create_cpu_vlm_endpoint_handler = self.create_cpu_vlm_endpoint_handler
        self.create_apple_vlm_endpoint_handler = self.create_apple_vlm_endpoint_handler
        self.create_qualcomm_vlm_endpoint_handler = self.create_qualcomm_vlm_endpoint_handler
        self.build_transform = build_transform
        self.load_image = load_image
        self.load_image_tensor = load_image_tensor
        self.dynamic_preprocess = dynamic_preprocess
        # self.load_image_bak = load_image_bak
        self.find_closest_aspect_ratio = find_closest_aspect_ratio
        self.init_cpu = self.init_cpu
        self.init_qualcomm = self.init_qualcomm
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init = self.init
        self.__test__ = self.__test__
        self.snpe_utils = None
        self.coreml_utils = None
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
    
    def init_cpu(self, model, device, cpu_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoProcessor.from_pretrained(model)
        endpoint = self.transformers.AutoModelForImageTextToText.from_pretrained(model, trust_remote_code=True)
        endpoint_handler = self.create_optimum_vlm_endpoint_handler(endpoint, tokenizer, model, cpu_label)
        self.torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
    
    def init_qualcomm(self, model, device, qualcomm_label):
        """Initialize LLaVA model for Qualcomm hardware.
        
        Args:
            model: HuggingFace model name or path
            device: Device to run inference on
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
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
            # Initialize processor directly from HuggingFace
            processor = self.transformers.LlavaProcessor.from_pretrained(model)
            
            # Convert model path to be compatible with SNPE
            model_name = model.replace("/", "--")
            dlc_path = f"~/snpe_models/{model_name}_llava.dlc"
            dlc_path = os.path.expanduser(dlc_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(dlc_path):
                print(f"Converting {model} to SNPE format...")
                self.snpe_utils.convert_model(model, "llava", str(dlc_path))
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Optimize for the specific Qualcomm device if possible
            if ":" in qualcomm_label:
                device_type = qualcomm_label.split(":")[1]
                optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                if optimized_path != dlc_path:
                    endpoint = self.snpe_utils.load_model(optimized_path)
            
            # Create endpoint handler
            endpoint_handler = self.create_qualcomm_llava_endpoint_handler(processor, model, qualcomm_label, endpoint)
            
            return endpoint, processor, endpoint_handler, asyncio.Queue(16), 1
        except Exception as e:
            print(f"Error initializing Qualcomm LLaVA model: {e}")
            return None, None, None, None, 0
    
    def init_cuda(self, model, device, cuda_label):
        self.init()
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
        if "ov_genai" not in list(self.resources.keys()):
            import openvino_genai as ov_genai
            self.ov_genai = ov_genai
        else:
            self.ov_genai = self.resources["ov_genai"]
        
        if "openvino" not in list(self.resources.keys()):
            import openvino as ov
            self.ov = ov
        else:
            self.ov = self.resources["openvino"]

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
    
    def init_apple(self, model, device, apple_label):
        """Initialize LLaVA model for Apple Silicon hardware."""
        self.init()
        
        try:
            from .apple_coreml_utils import get_coreml_utils
            self.coreml_utils = get_coreml_utils()
        except ImportError:
            print("Failed to import CoreML utilities")
            return None, None, None, None, 0
            
        if not self.coreml_utils.is_available():
            print("CoreML is not available on this system")
            return None, None, None, None, 0
            
        try:
            # Load processor from HuggingFace
            processor = self.transformers.LlavaProcessor.from_pretrained(model)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_llava.mlpackage"
            mlmodel_path = os.path.expanduser(mlmodel_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(mlmodel_path):
                print(f"Converting {model} to CoreML format...")
                self.coreml_utils.convert_model(model, "vision_text_dual", str(mlmodel_path))
            
            # Load the CoreML model
            endpoint = self.coreml_utils.load_model(str(mlmodel_path))
            
            # Optimize for Apple Silicon if possible
            if ":" in apple_label:
                compute_units = apple_label.split(":")[1]
                optimized_path = self.coreml_utils.optimize_for_device(mlmodel_path, compute_units)
                if optimized_path != mlmodel_path:
                    endpoint = self.coreml_utils.load_model(optimized_path)
            
            endpoint_handler = self.create_apple_multimodal_endpoint_handler(endpoint, processor, model, apple_label)
            
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon LLaVA model: {e}")
            return None, None, None, None, 0
            
    def create_apple_multimodal_endpoint_handler(self, endpoint, processor, model_name, apple_label):
        """Creates an Apple Silicon optimized handler for LLaVA multimodal processing."""
        def handler(x, y=None, endpoint=endpoint, processor=processor, model_name=model_name, apple_label=apple_label):
            try:
                # Process inputs
                if isinstance(x, str) and y is not None:
                    # Handle image + text input
                    if isinstance(y, str):
                        # Load image
                        image = load_image(y)
                        inputs = processor(
                            text=x,
                            images=image,
                            return_tensors="np",
                            padding=True
                        )
                    elif isinstance(y, list):
                        # Handle multiple images
                        images = [load_image(img_path) for img_path in y]
                        inputs = processor(
                            text=[x] * len(images),
                            images=images,
                            return_tensors="np",
                            padding=True
                        )
                else:
                    inputs = x
                
                # Convert inputs to CoreML format
                input_dict = {}
                for key, value in inputs.items():
                    if hasattr(value, 'numpy'):
                        input_dict[key] = value.numpy()
                    else:
                        input_dict[key] = value
                
                # Run inference
                outputs = self.coreml_utils.run_inference(endpoint, input_dict)
                
                # Process outputs - LLaVA typically outputs text responses
                if 'logits' in outputs:
                    logits = self.torch.tensor(outputs['logits'])
                    generated_ids = self.torch.argmax(logits, dim=-1)
                    responses = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    return responses[0] if len(responses) == 1 else responses
                
                return None
                
            except Exception as e:
                print(f"Error in Apple Silicon LLaVA handler: {e}")
                return None
                
        return handler
    
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
        
    def create_apple_vlm_endpoint_handler(self, apple_endpoint_handler, local_apple_processor, endpoint_model, apple_label):
        """Creates an endpoint handler for Apple Silicon.
        
        Args:
            apple_endpoint_handler: The model endpoint
            local_apple_processor: The tokenizer or processor
            endpoint_model: The model name or path
            apple_label: Label to identify this endpoint
            
        Returns:
            A handler function for the Apple endpoint
        """
        def handler(x, y, apple_endpoint_handler=apple_endpoint_handler, local_apple_processor=local_apple_processor, endpoint_model=endpoint_model, apple_label=apple_label):
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
                streamer = self.transformers.TextStreamer(local_apple_processor, skip_prompt=True, skip_special_tokens=True)
                prompt = local_apple_processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = local_apple_processor(image, prompt, return_tensors="pt")
                
                if apple_endpoint_handler is not None:
                    # Move inputs to MPS device if available
                    if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                        for key in inputs:
                            if isinstance(inputs[key], self.torch.Tensor):
                                inputs[key] = inputs[key].to("mps")
                
                    output_ids = apple_endpoint_handler.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=50,
                        streamer=streamer,
                    )
                    outputs = local_apple_processor.decode(output_ids[0], skip_special_tokens=True)
                    return outputs
                    
                return "Model not loaded properly on Apple Silicon"
            except Exception as e:
                raise e
        return handler
        
    def create_qualcomm_vlm_endpoint_handler(self, qualcomm_endpoint_handler, local_qualcomm_processor, endpoint_model, qualcomm_label):
        """Creates an endpoint handler for Qualcomm hardware.
        
        Args:
            qualcomm_endpoint_handler: The model endpoint
            local_qualcomm_processor: The tokenizer or processor
            endpoint_model: The model name or path
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            A handler function for the Qualcomm endpoint
        """
        def handler(x, y, qualcomm_endpoint_handler=qualcomm_endpoint_handler, local_qualcomm_processor=local_qualcomm_processor, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label):
            try:
                # Process inputs
                if isinstance(image_input, str):
                    # Load image from URL or file
                    image = load_image(image_input)
                elif isinstance(image_input, Image.Image):
                    # If it's already a PIL Image
                    image = image_input
                else:
                    image = None
                
                # Create model inputs in numpy format
                if image is not None:
                    inputs = processor(text=text_input, images=image, return_tensors="np")
                else:
                    inputs = processor(text=text_input, return_tensors="np")
                
                # Run initial inference for image encoding and prompt processing
                results = self.snpe_utils.run_inference(endpoint, inputs)
                
                # For LLaVA models, we might need to do generation token by token
                generated_ids = []
                
                # Check if we have direct generation output
                if "generated_ids" in results:
                    generated_ids = results["generated_ids"][0]
                else:
                    # We need to do token-by-token generation
                    # First, get the processed inputs if available
                    if "input_ids" in results:
                        generated_ids = results["input_ids"][0].tolist()
                    else:
                        generated_ids = inputs["input_ids"][0].tolist()
                    
                    # Prepare for token-by-token generation
                    past_key_values = results.get("past_key_values", None)
                    max_new_tokens = 256
                    
                    # Generate tokens one by one
                    for _ in range(max_new_tokens):
                        # Prepare inputs for next token prediction
                        gen_inputs = {
                            "input_ids": self.np.array([generated_ids[-1:]]),
                            "attention_mask": self.np.array([[1]])
                        }
                        
                        # Add past key values if available
                        if past_key_values is not None:
                            for i, (k, v) in enumerate(past_key_values):
                                gen_inputs[f"past_key_values.{i}.key"] = k
                                gen_inputs[f"past_key_values.{i}.value"] = v
                        
                        # Get next token
                        token_results = self.snpe_utils.run_inference(endpoint, gen_inputs)
                        
                        # Get logits and past key values
                        if "logits" in token_results:
                            logits = self.np.array(token_results["logits"])
                            
                            # Update past key values
                            if "past_key_values" in token_results:
                                past_key_values = token_results["past_key_values"]
                            
                            # Basic greedy decoding
                            next_token_id = int(self.np.argmax(logits[0, -1, :]))
                            
                            # Add token to generated sequence
                            generated_ids.append(next_token_id)
                            
                            # Check for EOS token
                            if next_token_id == self.processor.tokenizer.eos_token_id:
                                break
                        else:
                            break
                
                # Decode the generated text
                generated_text = self.processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
                
                # Return result
                return {
                    "generated_text": generated_text,
                    "model": endpoint_model
                }
                
            except Exception as e:
                print(f"Error in Qualcomm LLaVA endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler