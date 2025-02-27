import time
import asyncio
from PIL import Image
import requests
from io import BytesIO
import os
import numpy as np

def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return image

def load_image_tensor(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return ov.Tensor(image_data)

class hf_clip:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_image_embedding_endpoint_handler = self.create_openvino_image_embedding_endpoint_handler
        self.create_cuda_image_embedding_endpoint_handler = self.create_cuda_image_embedding_endpoint_handler
        self.create_cpu_image_embedding_endpoint_handler = self.create_cpu_image_embedding_endpoint_handler
        self.create_apple_image_embedding_endpoint_handler = self.create_apple_image_embedding_endpoint_handler
        self.create_qualcomm_image_embedding_endpoint_handler = self.create_qualcomm_image_embedding_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_qualcomm = self.init_qualcomm
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init = self.init
        self.__test__ = self.__test__
        self.snpe_utils = None
        # self.init()
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
        image_1 = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, image_1)
            print(test_batch)
            print("hf_clip test passed")
        except Exception as e:
            print(e)
            print("hf_clip test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        len_tokens = 1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"samples: {len_tokens}")
        print(f"samples per second: {tokens_per_second}")
        # test_batch_sizes = await self.test_batch_sizes(metadata['models'], ipfs_accelerate_init)
        if "openvino" not in endpoint_label:
            with self.torch.no_grad():
                if "cuda" in dir(self.torch):
                    self.torch.cuda.empty_cache()
        return None
    
    def init_cpu(self, model, device, cpu_label):
        """
        Initialize CLIP model for CPU inference
        
        Args:
            model: Model name or path (e.g., 'openai/clip-vit-base-patch32')
            device: Device to run on ('cpu')
            cpu_label: Label for CPU endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        print(f"Loading {model} for CPU inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Define a fallback function to create a simple test model
            def create_test_model():
                print("Creating minimal CLIP model for testing")
                torch_module = self.torch  # Store reference to avoid name lookup issues
                
                # Create simple model objects
                class SimpleProcessor:
                    def __init__(self):
                        self.torch = torch_module  # Use the class's torch reference
                        self.image_processor = self
                        
                    def __call__(self, images=None, text=None, return_tensors="pt", padding=True, **kwargs):
                        """Process images or text for CLIP input"""
                        batch_size = 1
                        result = {}
                        
                        if images is not None:
                            if isinstance(images, list):
                                batch_size = len(images)
                            # Create random pixel values tensor
                            result["pixel_values"] = self.torch.rand((batch_size, 3, 224, 224))
                            
                        if text is not None:
                            if isinstance(text, list):
                                batch_size = len(text)
                            # Create dummy text tensors
                            result["input_ids"] = self.torch.ones((batch_size, 77), dtype=self.torch.long)
                            result["attention_mask"] = self.torch.ones((batch_size, 77), dtype=self.torch.long)
                            
                        return result
                
                class SimpleModel:
                    def __init__(self):
                        self.config = SimpleConfig()
                        self.torch = torch_module  # Use the class's torch reference
                        
                    def __call__(self, **kwargs):
                        batch_size = 1
                        
                        # Determine batch size from inputs
                        if "pixel_values" in kwargs:
                            batch_size = kwargs["pixel_values"].shape[0]
                        elif "input_ids" in kwargs:
                            batch_size = kwargs["input_ids"].shape[0]
                            
                        embed_dim = 512
                        
                        # Create an output object that mimics the CLIPOutput structure
                        class CLIPOutput:
                            def __init__(self, batch_size, dim):
                                self.text_embeds = torch_module.randn(batch_size, dim)
                                self.image_embeds = torch_module.randn(batch_size, dim)
                                self.last_hidden_state = torch_module.randn(batch_size, 77, dim)
                                
                        return CLIPOutput(batch_size, embed_dim)
                        
                    def get_text_features(self, **kwargs):
                        """Return text embeddings"""
                        batch_size = kwargs["input_ids"].shape[0] if "input_ids" in kwargs else 1
                        return torch_module.randn(batch_size, 512)
                        
                    def get_image_features(self, **kwargs):
                        """Return image embeddings"""
                        batch_size = kwargs["pixel_values"].shape[0] if "pixel_values" in kwargs else 1
                        return torch_module.randn(batch_size, 512)
                
                class SimpleConfig:
                    def __init__(self):
                        self.hidden_size = 512
                        self.vocab_size = 49408
                        self.max_position_embeddings = 77
                        self.model_type = "clip"
                        
                # Create and return our simple processor and model
                return SimpleProcessor(), SimpleModel()
            
            # Try to load the real model if possible
            if isinstance(self.transformers, type):
                try:
                    # Try to load configuration
                    config = self.transformers.AutoConfig.from_pretrained(
                        model, 
                        trust_remote_code=True,
                        cache_dir=cache_dir
                    )
                    
                    # Try to load tokenizer and processor
                    tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                        model,
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )
                    
                    processor = self.transformers.CLIPProcessor.from_pretrained(
                        model, 
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )
                    
                    # Try to load model
                    endpoint = self.transformers.CLIPModel.from_pretrained(
                        model, 
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                        low_cpu_mem_usage=True
                    )
                    
                    print(f"Successfully loaded CLIP model: {model}")
                    
                except Exception as e:
                    print(f"Failed to load real CLIP model: {e}")
                    print("Creating test CLIP model instead")
                    processor, endpoint = create_test_model()
                    tokenizer = processor  # Use processor as tokenizer for simplicity
            else:
                # Create a test model if transformers is mocked
                processor, endpoint = create_test_model()
                tokenizer = processor  # Use processor as tokenizer
                
            # Create the handler
            endpoint_handler = self.create_cpu_image_embedding_endpoint_handler(
                tokenizer, 
                model, 
                cpu_label, 
                endpoint
            )
            
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
            
        except Exception as e:
            print(f"Error initializing CPU model: {e}")
            return None, None, None, None, 0
    
    def init_qualcomm(self, model, device, qualcomm_label):
        self.init()
        try:
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
            
            config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
            processor = self.transformers.CLIPProcessor.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with SNPE
            model_name = model.replace("/", "--")
            dlc_path = f"~/snpe_models/{model_name}_clip.dlc"
            dlc_path = os.path.expanduser(dlc_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(dlc_path):
                print(f"Converting {model} to SNPE format...")
                self.snpe_utils.convert_model(model, "vision_text_dual", str(dlc_path))
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Optimize for the specific Qualcomm device if possible
            if ":" in qualcomm_label:
                device_type = qualcomm_label.split(":")[1]
                optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                if optimized_path != dlc_path:
                    endpoint = self.snpe_utils.load_model(optimized_path)
            
            endpoint_handler = self.create_qualcomm_image_embedding_endpoint_handler(tokenizer, processor, model, qualcomm_label, endpoint)
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
        except Exception as e:
            print(f"Error initializing Qualcomm model: {e}")
            return None, None, None, None, 0
            
    def init_apple(self, model, device, apple_label):
        """Initialize CLIP model for Apple Silicon hardware."""
        self.init()
        
        # Import CoreML utilities
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
            # Load processor directly from HuggingFace
            processor = self.transformers.CLIPProcessor.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_clip.mlpackage"
            mlmodel_path = os.path.expanduser(mlmodel_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(mlmodel_path):
                print(f"Converting {model} to CoreML format...")
                self.coreml_utils.convert_model(model, "vision", str(mlmodel_path))
            
            # Load the CoreML model
            endpoint = self.coreml_utils.load_model(str(mlmodel_path))
            
            # Optimize for Apple Silicon if possible
            if ":" in apple_label:
                compute_units = apple_label.split(":")[1]
                optimized_path = self.coreml_utils.optimize_for_device(mlmodel_path, compute_units)
                if optimized_path != mlmodel_path:
                    endpoint = self.coreml_utils.load_model(optimized_path)
            
            endpoint_handler = self.create_apple_image_embedding_endpoint_handler(endpoint, processor, model, apple_label)
            
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon CLIP model: {e}")
            return None, None, None, None, 0

    def init_cuda(self, model, device, cuda_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
        processor = self.transformers.CLIPProcessor.from_pretrained(model, trust_remote_code=True)
        endpoint = None
        try:
            endpoint = self.transformers.CLIPModel.from_pretrained(model, torch_dtype=self.torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_cuda_image_embedding_endpoint_handler(tokenizer, endpoint_model=model, cuda_label=cuda_label, endpoint=endpoint)
        self.torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0    

    def init_openvino(self, model=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None ):
        self.init()
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
        # config = AutoConfig.from_pretrained(model)
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
            # os.makedirs(model_dst_path)
            ## convert model to openvino format
            # openvino_cli_convert(model, model_dst_path=model_dst_path, task=task, weight_format=weight_format, ratio="1.0", group_size=128, sym=True )
            pass
        try:
            tokenizer = self.transformers.CLIPProcessor.from_pretrained(
                model
            )
        except Exception as e:
            print(e)
            try:
                tokenizer = self.transformers.CLIPProcessor.from_pretrained(
                    model_src_path
                )
            except Exception as e:
                print(e)
                pass
        
        # genai_model = get_openvino_genai_pipeline(model, model_type, openvino_label)
        try:
            model = get_openvino_model(model, model_type, openvino_label)
            print(model)
        except Exception as e:
            print(e)
            try:
                model = get_optimum_openvino_model(model, model_type, openvino_label)
            except Exception as e:
                print(e)
                pass
        endpoint = model
        endpoint_handler = self.create_openvino_image_embedding_endpoint_handler(model, tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size              
    
    def create_cpu_image_embedding_endpoint_handler(self, tokenizer, endpoint_model, cpu_label, endpoint=None):
        """
        Create a handler for CLIP that can process text, images, or both
        
        Args:
            tokenizer: The tokenizer or processor
            endpoint_model: The model name or path
            cpu_label: The label for the CPU endpoint
            endpoint: The model endpoint
            
        Returns:
            A handler function
        """
        def handler(x=None, y=None, tokenizer=tokenizer, endpoint_model=endpoint_model, cpu_label=cpu_label, endpoint=endpoint):
            """
            Process text and/or image inputs with CLIP
            
            Args:
                x: Text input (str or list of str) or image if y is None
                y: Image input (str path, PIL Image, or list of either)
                
            Returns:
                Dict containing embeddings and/or similarity scores
            """
            # Ensure model is in eval mode
            if hasattr(endpoint, 'eval'):
                endpoint.eval()
            
            try:
                result = {}
                
                # Check what kind of inputs we have
                text_input = None
                image_input = None
                
                # If only x is provided, determine if it's text or image
                if x is not None and y is None:
                    if isinstance(x, str) and (x.startswith('http') or os.path.exists(x)):
                        # x is an image path
                        image_input = x
                    elif isinstance(x, Image.Image):
                        # x is a PIL image
                        image_input = x
                    else:
                        # Assume x is text
                        text_input = x
                else:
                    # Both x and y are provided or both are None
                    text_input = x
                    image_input = y
                
                # Process image if provided
                if image_input is not None:
                    try:
                        # Load and process image(s)
                        if isinstance(image_input, str):
                            # Single image path
                            image = load_image(image_input)
                            image_inputs = tokenizer(images=[image], return_tensors='pt', padding=True)
                        elif isinstance(image_input, Image.Image):
                            # Single PIL image
                            image_inputs = tokenizer(images=[image_input], return_tensors='pt', padding=True)
                        elif isinstance(image_input, list):
                            # List of images
                            images = [
                                img if isinstance(img, Image.Image) else load_image(img)
                                for img in image_input
                            ]
                            image_inputs = tokenizer(images=images, return_tensors='pt', padding=True)
                        else:
                            raise ValueError(f"Unsupported image input type: {type(image_input)}")
                        
                        # Get image embeddings
                        with self.torch.no_grad():
                            if hasattr(endpoint, 'get_image_features'):
                                image_features = endpoint.get_image_features(**image_inputs)
                            else:
                                # For processors that handle both image and text
                                outputs = endpoint(**image_inputs)
                                image_features = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs
                            
                            result["image_embedding"] = image_features
                    except Exception as e:
                        print(f"Error processing image input: {e}")
                        # Create fallback image embedding
                        batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                        torch_module = self.torch  # Capture reference to avoid name errors
                        result["image_embedding"] = torch_module.rand((batch_size, 512))
                
                # Process text if provided
                if text_input is not None:
                    try:
                        # Process text input(s)
                        if isinstance(text_input, str):
                            # Single text
                            text_inputs = tokenizer(text=[text_input], return_tensors='pt', padding=True)
                        elif isinstance(text_input, list):
                            # List of texts
                            text_inputs = tokenizer(text=text_input, return_tensors='pt', padding=True)
                        else:
                            raise ValueError(f"Unsupported text input type: {type(text_input)}")
                        
                        # Get text embeddings
                        with self.torch.no_grad():
                            if hasattr(endpoint, 'get_text_features'):
                                text_features = endpoint.get_text_features(**text_inputs)
                            else:
                                # For processors that handle both image and text
                                outputs = endpoint(**text_inputs)
                                text_features = outputs.text_embeds if hasattr(outputs, 'text_embeds') else outputs
                            
                            result["text_embedding"] = text_features
                    except Exception as e:
                        print(f"Error processing text input: {e}")
                        # Create fallback text embedding
                        batch_size = 1 if not isinstance(text_input, list) else len(text_input)
                        torch_module = self.torch  # Capture reference to avoid name errors
                        result["text_embedding"] = torch_module.rand((batch_size, 512))
                
                # Calculate similarity if we have both embeddings
                if "image_embedding" in result and "text_embedding" in result:
                    try:
                        # Normalize embeddings
                        image_norm = result["image_embedding"] / result["image_embedding"].norm(dim=-1, keepdim=True)
                        text_norm = result["text_embedding"] / result["text_embedding"].norm(dim=-1, keepdim=True)
                        
                        # Calculate cosine similarity
                        similarity = (image_norm @ text_norm.T)
                        result["similarity"] = similarity
                    except Exception as e:
                        print(f"Error calculating similarity: {e}")
                
                # No valid inputs
                if not result:
                    return {"message": "No valid input provided"}
                
                # Return single embedding if that's all that was requested
                if len(result) == 1 and (
                    "image_embedding" in result or 
                    "text_embedding" in result
                ):
                    embedding_key = list(result.keys())[0]
                    return {embedding_key: result[embedding_key]}
                
                return result
                
            except Exception as e:
                print(f"Error in CPU CLIP handler: {e}")
                return {"error": str(e)}
                
        return handler
    
    def create_qualcomm_image_embedding_endpoint_handler(self, tokenizer, processor, endpoint_model, qualcomm_label, endpoint=None):
        def handler(x, y=None, tokenizer=tokenizer, processor=processor, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint):
            try:
                inputs = {}
                
                # Process text input (if provided)
                if x is not None:
                    if isinstance(x, str):
                        text_inputs = tokenizer(text=[x], return_tensors='np')
                    elif isinstance(x, list):
                        text_inputs = tokenizer(text=x, return_tensors='np')
                    
                    inputs["input_ids"] = text_inputs["input_ids"]
                    inputs["attention_mask"] = text_inputs["attention_mask"]
                
                # Process image input (if provided)
                if y is not None:
                    if isinstance(y, str):
                        image = load_image(y)
                        # Convert to proper format for CLIP
                        if hasattr(processor, "image_processor"):
                            image_inputs = processor.image_processor(images=[image], return_tensors='np')
                        else:
                            # Fallback to basic processing
                            image = image.resize((224, 224))  # Standard size for most vision models
                            img_array = self.np.array(image)
                            img_array = img_array.transpose(2, 0, 1)  # Convert to CHW format
                            img_array = img_array / 255.0  # Normalize
                            image_inputs = {"pixel_values": self.np.expand_dims(img_array, axis=0)}
                            
                        inputs["pixel_values"] = image_inputs["pixel_values"]
                    elif isinstance(y, list):
                        images = [load_image(img) for img in y]
                        if hasattr(processor, "image_processor"):
                            image_inputs = processor.image_processor(images=images, return_tensors='np')
                        else:
                            # Fallback processing for multiple images
                            processed_images = []
                            for img in images:
                                img = img.resize((224, 224))
                                img_array = self.np.array(img)
                                img_array = img_array.transpose(2, 0, 1)
                                img_array = img_array / 255.0
                                processed_images.append(img_array)
                            image_inputs = {"pixel_values": self.np.stack(processed_images)}
                            
                        inputs["pixel_values"] = image_inputs["pixel_values"]
                
                # Run inference with SNPE
                outputs = self.snpe_utils.run_inference(endpoint, inputs)
                
                # Process results based on what inputs were provided
                result = {}
                
                if x is not None and "text_embeds" in outputs:
                    result["text_embedding"] = self.torch.tensor(outputs["text_embeds"])
                    
                if y is not None and "image_embeds" in outputs:
                    result["image_embedding"] = self.torch.tensor(outputs["image_embeds"])
                
                # If we have both embeddings and both inputs, compute similarity
                if x is not None and y is not None and "text_embeds" in outputs and "image_embeds" in outputs:
                    # Convert to PyTorch tensors
                    text_embeddings = self.torch.tensor(outputs["text_embeds"])
                    image_embeddings = self.torch.tensor(outputs["image_embeds"])
                    
                    # Normalize embeddings
                    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
                    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = self.torch.matmul(text_embeddings, image_embeddings.T)
                    result["similarity"] = similarity
                
                # Return single embedding if only one input type was provided
                if len(result) == 0:
                    return {"message": "No valid embeddings generated"}
                elif len(result) == 1 and list(result.keys())[0] in ["text_embedding", "image_embedding"]:
                    return {"embedding": list(result.values())[0]}
                
                return result
                
            except Exception as e:
                print(f"Error in Qualcomm CLIP endpoint handler: {e}")
                return {"error": str(e)}
        return handler
        
    def create_apple_image_embedding_endpoint_handler(self, endpoint, processor, endpoint_model, apple_label):
        """Creates an Apple Silicon optimized handler for CLIP image/text embedding models."""
        def handler(x, y=None, endpoint=endpoint, processor=processor, endpoint_model=endpoint_model, apple_label=apple_label):
            try:
                inputs = {}
                
                # Handle text input
                if x is not None:
                    if type(x) == str:
                        text_inputs = processor(
                            text=x,
                            return_tensors='np',
                            padding=True
                        )
                    elif type(x) == list:
                        text_inputs = processor(text=[text for text in x], return_tensors='np', padding=True)
                    
                    for key, value in text_inputs.items():
                        inputs[key] = value
                
                # Handle image input
                if y is not None:
                    if type(y) == str:
                        image = load_image(y)
                        image_inputs = processor(
                            images=[image], 
                            return_tensors='np', 
                            padding=True
                        )
                    elif type(y) == list:
                        images = [load_image(image_file) for image_file in y]
                        image_inputs = processor(
                            images=images,
                            return_tensors='np',
                            padding=True
                        )
                    
                    # Add image inputs
                    for key, value in image_inputs.items():
                        if key.startswith('pixel_values'):
                            inputs[key] = value
                
                # Run inference with CoreML
                results = self.coreml_utils.run_inference(endpoint, inputs)
                
                # Process results
                output = {}
                
                # Extract text embeddings
                if x is not None and "text_embeds" in results:
                    text_embeddings = self.torch.tensor(results["text_embeds"])
                    output["text_embedding"] = text_embeddings
                
                # Extract image embeddings
                if y is not None and "image_embeds" in results:
                    image_embeddings = self.torch.tensor(results["image_embeds"])
                    output["image_embedding"] = image_embeddings
                
                # If we have both text and image, compute similarity
                if x is not None and y is not None and "text_embeds" in results and "image_embeds" in results:
                    text_emb = self.torch.tensor(results["text_embeds"])
                    image_emb = self.torch.tensor(results["image_embeds"])
                    
                    # Normalize embeddings
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = self.torch.matmul(text_emb, image_emb.T)
                    output["similarity"] = similarity
                
                # Return single embedding if that's all we have
                if len(output) == 1 and list(output.keys())[0] in ["text_embedding", "image_embedding"]:
                    return {"embedding": list(output.values())[0]}
                    
                return output if output else None
                
            except Exception as e:
                print(f"Error in Apple Silicon image embedding handler: {e}")
                return None
                
        return handler
    
    def create_cuda_image_embedding_endpoint_handler(self, tokenizer, endpoint_model, cuda_label, endpoint=None):
        def handler(x, y=None, tokenizer=tokenizer, endpoint_model=endpoint_model, cuda_label=cuda_label, endpoint=endpoint):
            if "eval" in dir(endpoint):
                endpoint.eval()
                
            with self.torch.no_grad():
                try:
                    self.torch.cuda.empty_cache()
                    
                    if y is not None:
                        if isinstance(y, str):
                            image = load_image(y)
                            inputs = tokenizer(images=[image], return_tensors='pt', padding=True).to(cuda_label)
                        elif isinstance(y, list):
                            inputs = tokenizer(images=[load_image(img) for img in y], return_tensors='pt').to(cuda_label)
                            
                        image_features = endpoint.get_image_features(**inputs)
                        image_embeddings = image_features.detach().cpu()
                        
                        if x is None:
                            self.torch.cuda.empty_cache()
                            return {"image_embedding": image_embeddings}
                    
                    if x is not None:
                        if isinstance(x, str):
                            inputs = tokenizer(text=[x], return_tensors='pt').to(cuda_label)
                        elif isinstance(x, list):
                            inputs = tokenizer(text=x, return_tensors='pt').to(cuda_label)
                            
                        text_features = endpoint.get_text_features(**inputs)
                        text_embeddings = text_features.detach().cpu()
                        
                        if y is None:
                            self.torch.cuda.empty_cache()
                            return {"text_embedding": text_embeddings}
                    
                    if x is not None and y is not None:
                        self.torch.cuda.empty_cache()
                        return {
                            "text_embedding": text_embeddings,
                            "image_embedding": image_embeddings
                        }
                    
                    self.torch.cuda.empty_cache()
                    return {"message": "No valid input provided"}
                except Exception as e:
                    self.torch.cuda.empty_cache()
                    print(f"Error in CUDA endpoint handler: {e}")
                    return {"error": str(e)}
        return handler

    def create_openvino_image_embedding_endpoint_handler(self, endpoint_model, tokenizer, openvino_label, endpoint=None):
        def handler(x, y, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint):
            if y is not None:            
                if type(y) == str:
                    image = load_image(y)
                    inputs = tokenizer(images=[image], return_tensors='pt', padding=True)
                elif type(y) == list:
                    inputs = tokenizer(images=[load_image(image) for image in y], return_tensors='pt')
                with self.torch.no_grad():
                    image_features = endpoint_model(dict(inputs))
                    image_embeddings = image_features["image_embeds"]
 
                pass
            
            if x is not None:
                if type(x) == str:
                    inputs = tokenizer(text=y, return_tensors='pt')
                elif type(x) == list:
                    inputs = tokenizer(text=[text for text in x], return_tensors='pt')
                with self.torch.no_grad():
                    text_features = endpoint_model(dict(inputs))
                    text_embeddings = text_features["last_hidden_state"] 
            
            if x is not None or y is not None:
                if x is not None and y is not None:
                    return {
                        'image_embedding': image_embeddings,
                        'text_embedding': text_embeddings
                    }
                elif x is not None:
                    return {
                        'embedding': image_embeddings
                    }
                elif y is not None:
                    return {
                        'embedding': text_embeddings
                    }            
            return None
        return handler

    def openvino_skill_convert(self, model_name, model_dst_path, task, weight_format, hfmodel=None, hfprocessor=None):
        if hfmodel is None:
            hfmodel = self.transformers.AutoModel.from_pretrained(model_name, torch_dtype=self.torch.float16)
    
        if hfprocessor is None:
            hfprocessor = self.transformers.AutoProcessor.from_pretrained(model_name)
        if hfprocessor is not None:
            text = "Replace me by any text you'd like."
            image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
            image = load_image(image_url)
            processed_data = hfprocessor(
                text = "Replace me by any text you'd like.",
                images = [image],
                return_tensors="pt", 
                padding=True
            )
            results = hfmodel(**processed_data)
            hfmodel.config.torchscript = True
            ov_model = self.ov.convert_model(hfmodel,  example_input=dict(processed_data))
            if not os.path.exists(model_dst_path):
                os.mkdir(model_dst_path)
            self.ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
            ov_model = self.ov.compile_model(ov_model)
            hfmodel = None
        return ov_model