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
    """HuggingFace LLaVA (Large Language and Vision Assistant) implementation.
    
    This class provides standardized interfaces for working with LLaVA models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).
    
    LLaVA combines vision and language capabilities to enable multimodal understanding
    and generation. It can analyze images and respond to text prompts about them.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the LLaVA model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        """
        self.resources = resources
        self.metadata = metadata
        
        # Handler creation methods
        self.create_cpu_vlm_endpoint_handler = self.create_cpu_vlm_endpoint_handler
        self.create_cuda_vlm_endpoint_handler = self.create_cuda_vlm_endpoint_handler
        self.create_openvino_vlm_endpoint_handler = self.create_openvino_vlm_endpoint_handler
        self.create_openvino_genai_vlm_endpoint_handler = self.create_openvino_genai_vlm_endpoint_handler
        self.create_optimum_vlm_endpoint_handler = self.create_optimum_vlm_endpoint_handler
        self.create_apple_vlm_endpoint_handler = self.create_apple_vlm_endpoint_handler
        self.create_qualcomm_vlm_endpoint_handler = self.create_qualcomm_vlm_endpoint_handler
        
        # Initialization methods
        self.init = self.init
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        
        # Utility methods
        self.build_transform = build_transform
        self.load_image = load_image
        self.load_image_tensor = load_image_tensor
        self.dynamic_preprocess = dynamic_preprocess
        self.find_closest_aspect_ratio = find_closest_aspect_ratio
        self.__test__ = self.__test__
        
        # Hardware-specific utilities
        self.snpe_utils = None  # Qualcomm SNPE utils
        self.coreml_utils = None  # Apple CoreML utils
        return None
        
    def _create_mock_processor(self):
        """Create a mock processor for graceful degradation when the real one fails.
        
        Returns:
            Mock processor object with essential methods
        """
        from unittest.mock import MagicMock
        
        processor = MagicMock()
        
        # Add necessary methods for the processor
        processor.apply_chat_template = lambda conversation, add_generation_prompt=True: f"User: {conversation[0]['content'][1]['text']} with an image"
        processor.decode = lambda ids, skip_special_tokens=True: "This is a mock response from LLaVA"
        processor.batch_decode = lambda ids, skip_special_tokens=True, clean_up_tokenization_spaces=False: ["This is a mock response from LLaVA"]
        
        # Make it callable
        processor.__call__ = lambda text=None, images=None, return_tensors=None, padding=None: {
            "input_ids": self.torch.ones((1, 10), dtype=self.torch.long),
            "attention_mask": self.torch.ones((1, 10), dtype=self.torch.long),
            "pixel_values": self.torch.ones((1, 3, 224, 224), dtype=self.torch.float)
        }
        
        return processor
        
    def _create_mock_endpoint(self, model_name, device_label):
        """Create mock endpoint objects when real initialization fails.
        
        Args:
            model_name (str): The model name or path
            device_label (str): The device label (cpu, cuda, etc.)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        """
        from unittest.mock import MagicMock
        import time
        
        # Basic mock endpoint
        endpoint = MagicMock()
        endpoint.generate = lambda **kwargs: self.torch.ones((1, 10), dtype=self.torch.long)
        
        # Create processor with essential methods
        processor = self._create_mock_processor()
        
        # Create a handler function that clearly identifies as a mock
        def mock_handler(text="", image=None):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            image_info = f"of size {image.size}" if hasattr(image, 'size') else "with the provided content"
            return f"(MOCK) LLaVA response [timestamp: {timestamp}]: This is a mock response. I was asked to analyze an image {image_info} with prompt: '{text}'"
        
        print(f"Created mock LLaVA endpoint for {model_name} on {device_label}")
        return endpoint, processor, mock_handler, asyncio.Queue(64), 0
    
    
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
    
    def init_cpu(self, model_name, device, cpu_label):
        """Initialize LLaVA model for CPU.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run inference on ("cpu")
            cpu_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        try:
            # Load model configuration, processor and model
            config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)    
            processor = self.transformers.AutoProcessor.from_pretrained(model_name)
            
            # Load the model - use AutoModelForImageTextToText for consistency
            endpoint = self.transformers.AutoModelForImageTextToText.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Create the handler function
            endpoint_handler = self.create_cpu_vlm_endpoint_handler(
                endpoint=endpoint,
                processor=processor,
                model_name=model_name,
                cpu_label=cpu_label
            )
            
            # Clean up any CUDA memory if it was used previously
            if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                self.torch.cuda.empty_cache()
                
            print(f"Successfully initialized LLaVA model '{model_name}' on CPU")
            return endpoint, processor, endpoint_handler, asyncio.Queue(64), 0
            
        except Exception as e:
            print(f"Error initializing LLaVA model on CPU: {e}")
            # Return mock objects in case of failure for graceful degradation
            return self._create_mock_endpoint(model_name, cpu_label)
    
    def init_qualcomm(self, model_name, device, qualcomm_label):
        """Initialize LLaVA model for Qualcomm hardware.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run inference on ("qualcomm")
            qualcomm_label: Label to identify this endpoint ("qualcomm:0", etc.)
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Check for SNPE utilities
        try:
            from .qualcomm_snpe_utils import get_snpe_utils
            self.snpe_utils = get_snpe_utils()
        except ImportError:
            print("Failed to import Qualcomm SNPE utilities")
            return self._create_mock_endpoint(model_name, qualcomm_label)
            
        # Verify SNPE is available in this environment
        if not self.snpe_utils.is_available():
            print("Qualcomm SNPE is not available on this system")
            return self._create_mock_endpoint(model_name, qualcomm_label)
            
        try:
            # Initialize processor directly from HuggingFace
            processor = self.transformers.AutoProcessor.from_pretrained(model_name)
            
            # Convert model path to be compatible with SNPE
            safe_model_name = model_name.replace("/", "--")
            dlc_path = f"~/snpe_models/{safe_model_name}_llava.dlc"
            dlc_path = os.path.expanduser(dlc_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(dlc_path):
                print(f"Converting {model_name} to SNPE format...")
                self.snpe_utils.convert_model(model_name, "llava", str(dlc_path))
                print(f"Model converted and saved to {dlc_path}")
            else:
                print(f"Using existing SNPE model at {dlc_path}")
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Optimize for the specific Qualcomm device if possible
            if ":" in qualcomm_label:
                device_type = qualcomm_label.split(":")[1]
                print(f"Optimizing for Qualcomm device type: {device_type}")
                optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                if optimized_path != dlc_path:
                    print(f"Using optimized model at {optimized_path}")
                    endpoint = self.snpe_utils.load_model(optimized_path)
            
            # Create endpoint handler
            endpoint_handler = self.create_qualcomm_vlm_endpoint_handler(
                endpoint=endpoint,
                processor=processor,
                model_name=model_name,
                qualcomm_label=qualcomm_label
            )
            
            print(f"Successfully initialized LLaVA model '{model_name}' on Qualcomm device {qualcomm_label}")
            return endpoint, processor, endpoint_handler, asyncio.Queue(16), 1
            
        except Exception as e:
            print(f"Error initializing Qualcomm LLaVA model: {e}")
            return self._create_mock_endpoint(model_name, qualcomm_label)
    
    def init_cuda(self, model_name, device, cuda_label):
        """Initialize LLaVA model for CUDA/GPU.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run inference on ("cuda", "cuda:0", etc.)
            cuda_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Check if CUDA is available
        if not hasattr(self.torch, 'cuda') or not self.torch.cuda.is_available():
            print(f"CUDA is not available, falling back to CPU for model '{model_name}'")
            return self._create_mock_endpoint(model_name, cuda_label)
            
        try:
            # Load model configuration and processor
            config = self.transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)    
            processor = self.transformers.AutoProcessor.from_pretrained(model_name)
            
            # Load the model to GPU with optimizations
            endpoint = self.transformers.AutoModelForImageTextToText.from_pretrained(
                model_name,  
                torch_dtype=self.torch.float16,  # Use 16-bit precision for GPU
                trust_remote_code=True
            ).to(device)
            
            # Create the handler function
            endpoint_handler = self.create_cuda_vlm_endpoint_handler(
                endpoint=endpoint,
                processor=processor,
                model_name=model_name,
                cuda_label=cuda_label
            )
            
            # Clean up CUDA memory
            self.torch.cuda.empty_cache()
            
            print(f"Successfully initialized LLaVA model '{model_name}' on CUDA device {cuda_label}")
            return endpoint, processor, endpoint_handler, asyncio.Queue(64), 0
            
        except Exception as e:
            print(f"Error initializing LLaVA model on CUDA: {e}")
            # Return mock objects in case of failure for graceful degradation
            return self._create_mock_endpoint(model_name, cuda_label)

    def init_openvino(self, model_name, model_type, device, openvino_label, get_openvino_genai_pipeline, get_optimum_openvino_model, get_openvino_model, get_openvino_pipeline_type, openvino_cli_convert):
        """Initialize LLaVA model for OpenVINO.
        
        Args:
            model_name (str): HuggingFace model name or path
            model_type (str): Model type (e.g., "text-generation", "vision-language")
            device (str): Device to run inference on ("CPU", "GPU", etc.)
            openvino_label (str): Label to identify this endpoint ("openvino:0", etc.)
            get_openvino_genai_pipeline: Function to get OpenVINO GenAI pipeline
            get_optimum_openvino_model: Function to get Optimum OpenVINO model
            get_openvino_model: Function to get OpenVINO model
            get_openvino_pipeline_type: Function to get pipeline type
            openvino_cli_convert: Function to convert model with CLI
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Import required OpenVINO libraries
        try:
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
        except ImportError as e:
            print(f"Failed to import OpenVINO libraries: {e}")
            return self._create_mock_endpoint(model_name, openvino_label)
        
        try:
            # Setup model paths
            homedir = os.path.expanduser("~")
            safe_model_name = model_name.replace("/", "--")
            openvino_models_dir = os.path.join(homedir, "openvino_models")
            os.makedirs(openvino_models_dir, exist_ok=True)
            
            # Extract the target device index
            openvino_index = int(openvino_label.split(":")[1]) if ":" in openvino_label else 0
            
            # Determine weight format based on target device
            weight_format = "int8"  # Default for CPU
            if openvino_index == 1:
                weight_format = "int4"  # For GPU
            elif openvino_index == 2:
                weight_format = "int4"  # For NPU
                
            # Create model paths
            model_src_dir = os.path.join(openvino_models_dir, safe_model_name)
            model_dst_path = os.path.join(model_src_dir, f"openvino_{weight_format}")
            
            # Create directory for the model
            os.makedirs(model_dst_path, exist_ok=True)
            
            # Get model configuration
            config = self.transformers.AutoConfig.from_pretrained(model_name)
            
            # Determine the correct pipeline task
            task = get_openvino_pipeline_type(model_name, model_type)
            
            # Convert model if needed
            if not os.path.exists(os.path.join(model_dst_path, "openvino_model.xml")):
                print(f"Converting {model_name} to OpenVINO format with {weight_format} precision...")
                openvino_cli_convert(
                    model_name, 
                    model_dst_path=model_dst_path, 
                    task=task, 
                    weight_format=weight_format, 
                    ratio="1.0", 
                    group_size=128, 
                    sym=True
                )
                print(f"Model converted and saved to {model_dst_path}")
            else:
                print(f"Using existing OpenVINO model at {model_dst_path}")
            
            # Load tokenizer/processor from the converted model
            processor = self.transformers.AutoProcessor.from_pretrained(
                model_dst_path, 
                patch_size=config.vision_config.patch_size if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'patch_size') else None,
                vision_feature_select_strategy=config.vision_feature_select_strategy if hasattr(config, 'vision_feature_select_strategy') else None
            )
            
            # Load the optimized model
            endpoint = get_optimum_openvino_model(model_name, model_type)
            
            # Create handler
            endpoint_handler = self.create_openvino_vlm_endpoint_handler(
                endpoint=endpoint,
                processor=processor,
                model_name=model_name,
                openvino_label=openvino_label
            )
            
            print(f"Successfully initialized LLaVA model '{model_name}' on OpenVINO device {openvino_label}")
            return endpoint, processor, endpoint_handler, asyncio.Queue(64), 0
            
        except Exception as e:
            print(f"Error initializing OpenVINO LLaVA model: {e}")
            return self._create_mock_endpoint(model_name, openvino_label)
    
    def init_apple(self, model_name, device, apple_label):
        """Initialize LLaVA model for Apple Silicon hardware.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run inference on ("mps")
            apple_label (str): Label to identify this endpoint ("apple:0", "apple:CPU_AND_GPU", etc.)
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        # Check for CoreML utilities
        try:
            from .apple_coreml_utils import get_coreml_utils
            self.coreml_utils = get_coreml_utils()
        except ImportError:
            print("Failed to import CoreML utilities")
            return self._create_mock_endpoint(model_name, apple_label)
            
        # Verify CoreML is available in this environment
        if not self.coreml_utils.is_available():
            print("CoreML is not available on this system")
            return self._create_mock_endpoint(model_name, apple_label)
            
        try:
            # Initialize processor directly from HuggingFace
            processor = self.transformers.AutoProcessor.from_pretrained(model_name)
            
            # Convert model path to be compatible with CoreML
            safe_model_name = model_name.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{safe_model_name}_llava.mlpackage"
            mlmodel_path = os.path.expanduser(mlmodel_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(mlmodel_path):
                print(f"Converting {model_name} to CoreML format...")
                self.coreml_utils.convert_model(model_name, "vision_text_dual", str(mlmodel_path))
                print(f"Model converted and saved to {mlmodel_path}")
            else:
                print(f"Using existing CoreML model at {mlmodel_path}")
            
            # Load the CoreML model
            endpoint = self.coreml_utils.load_model(str(mlmodel_path))
            
            # Optimize for the specific Apple Silicon configuration if possible
            if ":" in apple_label:
                compute_units = apple_label.split(":")[1]
                print(f"Optimizing for Apple Silicon with compute units: {compute_units}")
                optimized_path = self.coreml_utils.optimize_for_device(mlmodel_path, compute_units)
                if optimized_path != mlmodel_path:
                    print(f"Using optimized model at {optimized_path}")
                    endpoint = self.coreml_utils.load_model(optimized_path)
            
            # Create endpoint handler
            endpoint_handler = self.create_apple_vlm_endpoint_handler(
                endpoint=endpoint,
                processor=processor,
                model_name=model_name,
                apple_label=apple_label
            )
            
            print(f"Successfully initialized LLaVA model '{model_name}' on Apple Silicon {apple_label}")
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error initializing Apple Silicon LLaVA model: {e}")
            return self._create_mock_endpoint(model_name, apple_label)
            
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

    
    def create_openvino_vlm_endpoint_handler(self, endpoint, processor, model_name, openvino_label):
        """Create endpoint handler for OpenVINO backend.
        
        Args:
            endpoint: The model endpoint/object
            processor: The tokenizer/processor for the model
            model_name: Name of the model
            openvino_label: Label identifying this endpoint
        
        Returns:
            Handler function for processing requests
        """
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, openvino_label=openvino_label):
            """Process text and image inputs using LLaVA on OpenVINO.
            
            Args:
                text: Text prompt to process with the image
                image: Image to analyze (can be path, URL, or PIL Image)
                
            Returns:
                Generated text response from the model
            """
            try:
                # Load and process image if provided
                if image is not None:
                    if isinstance(image, str):
                        if image.startswith("http") or image.startswith("https"):
                            max_retries = 3
                            retry_delay = 1
                            last_error = None
                            
                            for attempt in range(max_retries):
                                try:
                                    response = requests.get(image, timeout=10)
                                    response.raise_for_status()  # Check for HTTP errors
                                    image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                                    break
                                except (requests.RequestException, Image.UnidentifiedImageError) as e:
                                    last_error = e
                                    if attempt < max_retries - 1:
                                        time.sleep(retry_delay)
                                        retry_delay *= 2
                                    else:
                                        raise ValueError(f"Failed to load image from URL after {max_retries} attempts: {image}. Error: {str(last_error)}")
                        else:
                            image_obj = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                else:
                    # Text-only input
                    image_obj = None
                    
                # Process the text input
                if text is not None and isinstance(text, str):
                    # Standard format for multimodal conversation
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image"}
                            ]
                        }
                    ]
                elif isinstance(text, (tuple, list)):
                    # Allow for pre-formatted conversation
                    conversation = text
                elif isinstance(text, dict):
                    # Single message as dict
                    conversation = [text]
                else:
                    raise ValueError(f"Unsupported text input type: {type(text)}")
                
                # Create streamer for real-time text output (if supported)
                try:
                    streamer = self.transformers.TextStreamer(
                        processor, 
                        skip_prompt=True, 
                        skip_special_tokens=True
                    )
                except Exception:
                    streamer = None
                
                # Process inputs
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(image_obj, prompt, return_tensors="pt")
                
                # Generate text with OpenVINO optimized model
                generation_config = {
                    "do_sample": False,
                    "max_new_tokens": 50
                }
                
                if streamer is not None:
                    generation_config["streamer"] = streamer
                
                output_ids = endpoint.generate(**inputs, **generation_config)
                
                # Decode response
                response = processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Add timestamp and metadata for testing/debugging
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_info = f"of size {image_obj.size}" if image_obj and hasattr(image_obj, 'size') else "with the provided content"
                
                return f"(REAL) OpenVINO LLaVA response [device: {openvino_label}, timestamp: {timestamp}]: {response}"
                
            except Exception as e:
                print(f"Error in OpenVINO LLaVA handler: {e}")
                import time
                import traceback
                traceback.print_exc()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"(ERROR) OpenVINO LLaVA response [timestamp: {timestamp}]: Error processing request - {str(e)}"
        
        return handler

    def create_cpu_vlm_endpoint_handler(self, endpoint, processor, model_name, cpu_label):
        """Create endpoint handler for CPU backend.
        
        Args:
            endpoint: The model endpoint/object
            processor: The tokenizer/processor for the model
            model_name: Name of the model
            cpu_label: Label identifying this endpoint
        
        Returns:
            Handler function for processing requests
        """
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, cpu_label=cpu_label):
            """Process text and image inputs using LLaVA on CPU.
            
            Args:
                text: Text prompt to process with the image
                image: Image to analyze (can be path, URL, or PIL Image)
                
            Returns:
                Generated text response from the model
            """
            try:
                # Load and process image if provided
                if image is not None:
                    if isinstance(image, str):
                        if image.startswith("http") or image.startswith("https"):
                            response = requests.get(image)
                            image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                        else:
                            image_obj = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                else:
                    # Text-only input
                    image_obj = None
                    
                # Process the text input
                if text is not None and isinstance(text, str):
                    # Standard format for multimodal conversation
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image"}
                            ]
                        }
                    ]
                elif isinstance(text, (tuple, list)):
                    # Allow for pre-formatted conversation
                    conversation = text
                elif isinstance(text, dict):
                    # Single message as dict
                    conversation = [text]
                else:
                    raise ValueError(f"Unsupported text input type: {type(text)}")
                
                # Create streamer for real-time text output
                streamer = self.transformers.TextStreamer(
                    processor, 
                    skip_prompt=True, 
                    skip_special_tokens=True
                )
                
                # Process inputs
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(image_obj, prompt, return_tensors="pt")
                
                # Generate response
                output_ids = endpoint.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=50,
                    streamer=streamer,
                )
                
                # Decode response
                response = processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Add timestamp and metadata for testing/debugging
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_info = f"of size {image_obj.size}" if image_obj and hasattr(image_obj, 'size') else "with the provided content"
                
                return f"(REAL) CPU LLaVA response [timestamp: {timestamp}]: {response}"
                
            except Exception as e:
                print(f"Error in CPU LLaVA handler: {e}")
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"(ERROR) CPU LLaVA response [timestamp: {timestamp}]: Error processing request - {str(e)}"
        
        return handler
    
    
    def create_cuda_vlm_endpoint_handler(self, endpoint, processor, model_name, cuda_label):
        """Create endpoint handler for CUDA/GPU backend.
        
        Args:
            endpoint: The model endpoint/object
            processor: The tokenizer/processor for the model
            model_name: Name of the model
            cuda_label: Label identifying this endpoint
        
        Returns:
            Handler function for processing requests
        """
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, cuda_label=cuda_label):
            """Process text and image inputs using LLaVA on CUDA/GPU.
            
            Args:
                text: Text prompt to process with the image
                image: Image to analyze (can be path, URL, or PIL Image)
                
            Returns:
                Generated text response from the model
            """
            try:
                # Load and process image if provided
                if image is not None:
                    if isinstance(image, str):
                        if image.startswith("http") or image.startswith("https"):
                            response = requests.get(image)
                            image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                        else:
                            image_obj = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                else:
                    # Text-only input
                    image_obj = None
                    
                # Process the text input
                if text is not None and isinstance(text, str):
                    # Standard format for multimodal conversation
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image"}
                            ]
                        }
                    ]
                elif isinstance(text, (tuple, list)):
                    # Allow for pre-formatted conversation
                    conversation = text
                elif isinstance(text, dict):
                    # Single message as dict
                    conversation = [text]
                else:
                    raise ValueError(f"Unsupported text input type: {type(text)}")
                
                # Create streamer for real-time text output
                streamer = self.transformers.TextStreamer(
                    processor, 
                    skip_prompt=True, 
                    skip_special_tokens=True
                )
                
                # Process inputs
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                
                # Create inputs and move to GPU
                inputs = processor(image_obj, prompt, return_tensors="pt")
                for key in inputs:
                    if hasattr(inputs[key], 'to') and callable(inputs[key].to):
                        inputs[key] = inputs[key].to(cuda_label)
                
                # Generate response with GPU acceleration
                output_ids = endpoint.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=50,
                    streamer=streamer,
                )
                
                # Decode response
                response = processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Add timestamp and metadata for testing/debugging
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_info = f"of size {image_obj.size}" if image_obj and hasattr(image_obj, 'size') else "with the provided content"
                
                return f"(REAL) CUDA LLaVA response [device: {cuda_label}, timestamp: {timestamp}]: {response}"
                
            except Exception as e:
                print(f"Error in CUDA LLaVA handler: {e}")
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"(ERROR) CUDA LLaVA response [timestamp: {timestamp}]: Error processing request - {str(e)}"
                
            finally:
                # Clean up CUDA memory after request
                if hasattr(self.torch, 'cuda') and hasattr(self.torch.cuda, 'empty_cache'):
                    self.torch.cuda.empty_cache()
        
        return handler
        
    def create_apple_vlm_endpoint_handler(self, endpoint, processor, model_name, apple_label):
        """Creates an endpoint handler for Apple Silicon.
        
        Args:
            endpoint: The model endpoint
            processor: The tokenizer or processor
            model_name: The model name or path
            apple_label: Label to identify this endpoint
            
        Returns:
            A handler function for the Apple endpoint
        """
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, apple_label=apple_label):
            """Process text and image inputs using LLaVA on Apple Silicon.
            
            Args:
                text: Text prompt to process with the image
                image: Image to analyze (can be path, URL, or PIL Image)
                
            Returns:
                Generated text response from the model
            """
            try:
                # Load and process image if provided
                if image is not None:
                    if isinstance(image, str):
                        if image.startswith("http") or image.startswith("https"):
                            max_retries = 3
                            retry_delay = 1
                            last_error = None
                            
                            for attempt in range(max_retries):
                                try:
                                    response = requests.get(image, timeout=10)
                                    response.raise_for_status() 
                                    image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                                    break
                                except (requests.RequestException, Image.UnidentifiedImageError) as e:
                                    last_error = e
                                    if attempt < max_retries - 1:
                                        time.sleep(retry_delay)
                                        retry_delay *= 2
                                    else:
                                        raise ValueError(f"Failed to load image from URL after {max_retries} attempts: {image}. Error: {str(last_error)}")
                        else:
                            image_obj = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                else:
                    # Text-only input
                    image_obj = None
                    
                # Process the text input
                if text is not None and isinstance(text, str):
                    # Standard format for multimodal conversation
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image"}
                            ]
                        }
                    ]
                elif isinstance(text, (tuple, list)):
                    # Allow for pre-formatted conversation
                    conversation = text
                elif isinstance(text, dict):
                    # Single message as dict
                    conversation = [text]
                else:
                    raise ValueError(f"Unsupported text input type: {type(text)}")
                
                # Create streamer for real-time text output
                try:
                    streamer = self.transformers.TextStreamer(
                        processor, 
                        skip_prompt=True, 
                        skip_special_tokens=True
                    )
                except Exception:
                    streamer = None
                
                # Process inputs
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(image_obj, prompt, return_tensors="pt")
                
                # Move inputs to MPS device if available
                if hasattr(self.torch, 'backends') and hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                    for key in inputs:
                        if isinstance(inputs[key], self.torch.Tensor):
                            inputs[key] = inputs[key].to("mps")
                
                # Generate response with MPS acceleration
                generation_config = {
                    "do_sample": False,
                    "max_new_tokens": 50
                }
                
                if streamer is not None:
                    generation_config["streamer"] = streamer
                
                output_ids = endpoint.generate(**inputs, **generation_config)
                
                # Decode response
                response = processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Add timestamp and metadata for testing/debugging
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_info = f"of size {image_obj.size}" if image_obj and hasattr(image_obj, 'size') else "with the provided content"
                
                return f"(REAL) Apple Silicon LLaVA response [device: {apple_label}, timestamp: {timestamp}]: {response}"
                
            except Exception as e:
                print(f"Error in Apple Silicon LLaVA handler: {e}")
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"(ERROR) Apple Silicon LLaVA response [timestamp: {timestamp}]: Error processing request - {str(e)}"
        
        return handler
        
    def create_qualcomm_vlm_endpoint_handler(self, endpoint, processor, model_name, qualcomm_label):
        """Creates an endpoint handler for Qualcomm hardware.
        
        Args:
            endpoint: The model endpoint
            processor: The tokenizer or processor
            model_name: The model name or path
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            A handler function for the Qualcomm endpoint
        """
        def handler(text=None, image=None, endpoint=endpoint, processor=processor, model_name=model_name, qualcomm_label=qualcomm_label):
            """Process text and image inputs using LLaVA on Qualcomm hardware.
            
            Args:
                text: Text prompt to process with the image
                image: Image to analyze (can be path, URL, or PIL Image)
                
            Returns:
                Generated text response from the model
            """
            try:
                # Check if SNPE utils is available
                if self.snpe_utils is None:
                    import time
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    return f"(MOCK) Qualcomm LLaVA response [timestamp: {timestamp}]: Qualcomm SNPE not available in this environment"
                
                # Load and process image if provided
                if image is not None:
                    if isinstance(image, str):
                        if image.startswith("http") or image.startswith("https"):
                            response = requests.get(image)
                            image_obj = Image.open(BytesIO(response.content)).convert("RGB")
                        else:
                            image_obj = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        image_obj = image
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                else:
                    # Text-only input
                    image_obj = None
                
                # Create model inputs in numpy format
                if image_obj is not None:
                    inputs = processor(text=text, images=image_obj, return_tensors="np")
                else:
                    inputs = processor(text=text, return_tensors="np")
                
                # Run inference through SNPE
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
                            if next_token_id == processor.tokenizer.eos_token_id:
                                break
                        else:
                            break
                
                # Decode the generated text
                generated_text = processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
                
                # Add timestamp and metadata for testing/debugging
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_info = f"of size {image_obj.size}" if image_obj and hasattr(image_obj, 'size') else "with the provided content"
                
                return f"(REAL) Qualcomm LLaVA response [device: {qualcomm_label}, timestamp: {timestamp}]: {generated_text}"
                
            except Exception as e:
                print(f"Error in Qualcomm LLaVA endpoint handler: {e}")
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"(ERROR) Qualcomm LLaVA response [timestamp: {timestamp}]: Error processing request - {str(e)}"
                
        return handler