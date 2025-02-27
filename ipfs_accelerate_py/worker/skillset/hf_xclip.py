import time
import asyncio
import os
from PIL import Image
import requests
from io import BytesIO
import os
import tempfile
import numpy as np

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_image_tensor(image_file):
    import openvino as ov
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
    return ov.Tensor(image_data)

class hf_xclip:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.openvino_skill_convert = self.openvino_skill_convert
        self.create_openvino_video_embedding_endpoint_handler = self.create_openvino_video_embedding_endpoint_handler
        self.create_cuda_video_embedding_endpoint_handler = self.create_cuda_video_embedding_endpoint_handler
        self.create_cpu_video_embedding_endpoint_handler = self.create_cpu_video_embedding_endpoint_handler
        self.create_apple_video_embedding_endpoint_handler = self.create_apple_video_embedding_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_qualcomm = self.init_qualcomm
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

        if "decord" not in list(self.resources.keys()):
            import decord
            self.decord = decord
        else:
            self.decord = self.resources["decord"]
        self.np.random.seed(0)
        return None
    
    def init_qualcomm(self, model, device, qualcomm_label):
        """Initialize XClip model for Qualcomm hardware.
        
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
            processor = self.transformers.AutoProcessor.from_pretrained(model)
            
            # Convert model path to be compatible with SNPE
            model_name = model.replace("/", "--")
            dlc_path = f"~/snpe_models/{model_name}_xclip.dlc"
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
            
            # Create endpoint handler
            endpoint_handler = self.create_qualcomm_xclip_endpoint_handler(processor, model, qualcomm_label, endpoint)
            
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Qualcomm XClip model: {e}")
            return None, None, None, None, 0

    def init_apple(self, model, device, apple_label):
        """Initialize XClip model for Apple Silicon hardware."""
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
            processor = self.transformers.XCLIPProcessor.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_xclip.mlpackage"
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
            print(f"Error initializing Apple Silicon XClip model: {e}")
            return None, None, None, None, 0

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, video_url)
            print(test_batch)
            print("hf_xclip test passed")
        except Exception as e:
            print(e)
            print("hf_xclip test failed")
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
        print("hf_xclip test")
        return None
    
    def init_cpu(self, model, device, cpu_label):
        self.init()
        try:
            config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
            tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
            processor = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
            
            endpoint = self.transformers.AutoModel.from_pretrained(model, trust_remote_code=True)
            
            endpoint_handler = self.create_cpu_video_embedding_endpoint_handler(tokenizer, model, cpu_label, endpoint)
            return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0
        except Exception as e:
            print(f"Error initializing CPU model: {e}")
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
        endpoint_handler = self.create_video_embedding_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        self.torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0    

    def init_openvino(self, model=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None):
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
            os.makedirs(model_dst_path, exist_ok=True)
            # Try using openvino_skill_convert if available
            try:
                convert = self.openvino_skill_convert(model, model_dst_path, task, weight_format)
            except Exception as e:
                print(f"Error using openvino_skill_convert: {e}")
                # Fall back to openvino_cli_convert
                try:
                    convert = openvino_cli_convert(
                        model, 
                        model_dst_path=model_dst_path, 
                        task=task, 
                        weight_format=weight_format, 
                        ratio="1.0", 
                        group_size=128, 
                        sym=True
                    )
                    print(f"Successfully converted model using OpenVINO CLI: {convert}")
                except Exception as e:
                    print(f"Error using openvino_cli_convert: {e}")
        try:
            tokenizer =  self.transformers.AutoProcessor.from_pretrained(
                model
            )
        except Exception as e:
            print(e)
            try:
                tokenizer =  self.transformers.AutoProcessor.from_pretrained(
                    model_src_path
                )
            except Exception as e:
                print(e)
                pass
            
        if not os.path.exists(model_dst_path):
            try:
                convert = self.openvino_skill_convert(model, model_dst_path, task, weight_format)
            except Exception as e:
                print(e)
                try: 
                    convert = openvino_cli_convert(model, model_dst_path=model_dst_path, task=task, weight_format=weight_format, ratio="1.0", group_size=128, sym=True )
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
        endpoint_handler = self.create_openvino_video_embedding_endpoint_handler(model, tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size              
    
    def create_cpu_video_embedding_endpoint_handler(self, tokenizer , endpoint_model, cpu_label, endpoint=None, ):
        def handler(x, y=None, tokenizer=tokenizer, endpoint_model=endpoint_model, cpu_label=cpu_label, endpoint=endpoint):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            
            # Implement CPU-based video embedding logic here
            # This is a placeholder implementation
            return {"message": "CPU video embedding not fully implemented"}
        return handler
    
    def create_qualcomm_video_embedding_endpoint_handler(self, tokenizer , endpoint_model, qualcomm_label, endpoint=None):
        def handler(x, y=None, tokenizer=tokenizer, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            
            # Implement Qualcomm-specific video embedding logic here
            # This is a placeholder implementation
            return {"message": "Qualcomm video embedding not fully implemented"}
        return handler
    
    def create_apple_video_embedding_endpoint_handler(self, tokenizer, endpoint_model, apple_label, endpoint=None):
        def handler(x, y=None, tokenizer=tokenizer, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            
            # Implement Apple-specific video embedding logic here
            # This is a placeholder implementation that would use CoreML
            return {"message": "Apple CoreML video embedding not fully implemented"}
        return handler
    
    def create_cuda_video_embedding_endpoint_handler(self, tokenizer, endpoint_model, cuda_label, endpoint=None):
        def handler(x, y=None, tokenizer=tokenizer, endpoint_model=endpoint_model, cuda_label=cuda_label, endpoint=endpoint):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            
            # Implement full CUDA video embedding handler here
            return {"message": "CUDA video embedding not fully implemented"}
        return handler

    def create_openvino_video_embedding_endpoint_handler(self, endpoint_model , tokenizer , openvino_label, endpoint=None ):
        def handler(x, y, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint):
            self.np.random.seed(0)                       
            videoreader = None
            if y is not None:            
                if type(y) == str:
                    if os.path.exists(y):
                        videoreader = self.decord.VideoReader(y, num_threads=1, ctx=self.decord.cpu(0))
                    elif "http" in y:
                        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                            f.write(requests.get(y).content)
                            f.flush()
                            videoreader = self.decord.VideoReader(f.name, num_threads=1, ctx=self.decord.cpu(0))
                if videoreader is not None:
                    videoreader.seek(0)
                    indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
                    video = videoreader.get_batch(indices).asnumpy()
                pass
            
            if x is not None:
                if type(x) == str:
                    text = x
                else:
                    text = ""
            else:
                text = ""
            
            processed_data = tokenizer(
                text=text,
                videos=list(video),
                return_tensors="pt",
                padding=True,
            )

            new_processed_data = {
                'input_ids': processed_data["input_ids"],
                'attention_mask': processed_data["attention_mask"],
                'pixel_values': processed_data["pixel_values"]
            }

            inference_results = endpoint_model(dict(new_processed_data))
            results_list = list(inference_results.values())
            
            text_embeddings = results_list[3]
            video_embeddings = results_list[5]
            if x is not None or y is not None:
                if x is not None and y is not None:
                    return {
                        'video_embedding': video_embeddings,
                        'text_embedding': text_embeddings
                    }
                elif x is not None:
                    return {
                        'embedding': video_embeddings
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
            ##xclip processor
            video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
            self.np.random.seed(0)
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(requests.get(video_url).content)
                f.flush()
                videoreader = self.decord.VideoReader(f.name, num_threads=1, ctx=self.decord.cpu(0))
                videoreader.seek(0)
                indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=len(videoreader))
                video = videoreader.get_batch(indices).asnumpy()
                processed_data = hfprocessor(
                    text=text,
                    videos=list(video),
                    return_tensors="pt",
                    padding=True,
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

    def create_qualcomm_xclip_endpoint_handler(self, processor, endpoint_model, qualcomm_label, endpoint):
        """Creates an endpoint handler for Qualcomm hardware.
        
        Args:
            processor: The processor for text and image inputs
            endpoint_model: The model name or path
            qualcomm_label: Label to identify this endpoint
            endpoint: The SNPE model endpoint
            
        Returns:
            A handler function for the Qualcomm endpoint
        """
        def handler(text_input=None, image_input=None, processor=processor, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint):
            try:
                inputs = {}
                
                # Process text input if provided
                if text_input is not None:
                    if isinstance(text_input, str):
                        text_inputs = processor(text=text_input, return_tensors="np")
                    else:
                        # Assume it's a batch of texts
                        text_inputs = processor(text=text_input, return_tensors="np", padding=True)
                        
                    for key, value in text_inputs.items():
                        inputs[key] = value
                
                # Process image input if provided
                if image_input is not None:
                    if isinstance(image_input, str):
                        # Load image from URL or file
                        image = load_image(image_input)
                        image_inputs = processor(images=image, return_tensors="np")
                    elif isinstance(image_input, list):
                        # Process a batch of images
                        images = [load_image(img) for img in image_input]
                        image_inputs = processor(images=images, return_tensors="np", padding=True)
                    else:
                        # Assume it's already a PIL Image
                        image_inputs = processor(images=image_input, return_tensors="np")
                    
                    for key, value in image_inputs.items():
                        inputs[key] = value
                
                # Run inference with SNPE
                results = self.snpe_utils.run_inference(endpoint, inputs)
                
                # Process results
                output = {}
                
                # Convert numpy arrays to torch tensors
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        output[key] = self.torch.tensor(value)
                    else:
                        output[key] = value
                
                # Calculate similarity if both text and image embeddings are available
                if "text_embeds" in results and "image_embeds" in results:
                    text_embeds = self.torch.tensor(results["text_embeds"])
                    image_embeds = self.torch.tensor(results["image_embeds"])
                    
                    # Normalize embeddings
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = self.torch.matmul(text_embeds, image_embeds.T)
                    output["similarity"] = similarity
                
                return output
                
            except Exception as e:
                print(f"Error in Qualcomm XClip endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler

    def create_apple_multimodal_endpoint_handler(self, endpoint, processor, model_name, apple_label):
        """Creates an Apple Silicon optimized handler for XClip multimodal processing."""
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
                
                # Process outputs
                result = {}
                
                # Extract text embeddings
                if 'text_embeds' in outputs:
                    text_embeddings = self.torch.tensor(outputs['text_embeds'])
                    result['text_embedding'] = text_embeddings
                    
                # Extract image embeddings
                if 'image_embeds' in outputs:
                    image_embeddings = self.torch.tensor(outputs['image_embeds'])
                    result['image_embedding'] = image_embeddings
                    
                # If we have both embeddings, compute similarity
                if 'text_embeds' in outputs and 'image_embeds' in outputs:
                    text_emb = self.torch.tensor(outputs['text_embeds'])
                    image_emb = self.torch.tensor(outputs['image_embeds'])
                    
                    # Normalize embeddings
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = self.torch.matmul(text_emb, image_emb.T)
                    result['similarity'] = similarity
                
                # Return single embedding if that's all we have
                if len(result) == 1 and list(result.keys())[0] in ['text_embedding', 'image_embedding']:
                    return {'embedding': list(result.values())[0]}
                    
                return result if result else None
                
            except Exception as e:
                print(f"Error in Apple Silicon XClip handler: {e}")
                return None
                
        return handler