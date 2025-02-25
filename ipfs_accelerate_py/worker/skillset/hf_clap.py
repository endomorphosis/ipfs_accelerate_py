import os
import time
import asyncio
import requests
import io
from pathlib import Path

def load_audio_16khz(audio_file):
    import librosa
    audio_data, samplerate = load_audio(audio_file)
    if samplerate != 16000:
        ## convert to 16khz
        audio_data = librosa.resample(y=audio_data, orig_sr=samplerate, target_sr=16000)
    return audio_data, 16000

def load_audio(audio_file):
    import soundfile as sf
    import numpy as np
        
    if isinstance(audio_file, str) and (audio_file.startswith("http") or audio_file.startswith("https")):
        response = requests.get(audio_file)
        audio_data, samplerate = sf.read(io.BytesIO(response.content))
    else:
        audio_data, samplerate = sf.read(audio_file)
    
    # Ensure audio is mono and convert to float32
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)
    
    return audio_data, samplerate

def load_audio_tensor(audio_file):
    import soundfile as sf
    import numpy as np
    import openvino as ov
    
    if isinstance(audio_file, str) and (audio_file.startswith("http") or audio_file.startswith("https")):
        response = requests.get(audio_file)
        audio_data, samplerate = sf.read(io.BytesIO(response.content))
    else:
        audio_data, samplerate = sf.read(audio_file)
    
    # Ensure audio is mono and convert to float32
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)
    
    return ov.Tensor(audio_data.reshape(1, -1))

def cleanup_torchscript_cache():
    import torch
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

class hf_clap:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata    
        self.create_openvino_audio_embedding_endpoint_handler = self.create_openvino_audio_embedding_endpoint_handler
        self.create_cuda_audio_embedding_endpoint_handler = self.create_cuda_audio_embedding_endpoint_handler
        self.create_cpu_audio_embedding_endpoint_handler = self.create_cpu_audio_embedding_endpoint_handler
        self.create_apple_audio_embedding_endpoint_handler = self.create_apple_audio_embedding_endpoint_handler
        self.create_qualcomm_audio_embedding_endpoint_handler = self.create_qualcomm_audio_embedding_endpoint_handler
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_qualcomm = self.init_qualcomm
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.transformers = None
        self.init = self.init
        self.__test__ = self.__test__
        self.snpe_utils = None
        return None

    def load_audio(self, audio_file):
            
        if isinstance(audio_file, str) and (audio_file.startswith("http") or audio_file.startswith("https")):
            response = requests.get(audio_file)
            audio_data, samplerate = self.sf.read(io.BytesIO(response.content))
        else:
            audio_data, samplerate = self.sf.read(audio_file)
        
        # Ensure audio is mono and convert to float32
        if len(audio_data.shape) > 1:
            audio_data = self.np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(self.np.float32)
        return audio_data, samplerate

    def load_audio_tensor(self, audio_file):
        if "ov" not in dir(self):
            if "openvino" not in list(self.resources.keys()):    
                import openvino as ov
                self.ov = ov
            else:
                self.ov = self.resources["openvino"]
        if isinstance(audio_file, str) and (audio_file.startswith("http") or audio_file.startswith("https")):
            response = requests.get(audio_file)
            audio_data, samplerate = self.sf.read(io.BytesIO(response.content))
        else:
            audio_data, samplerate = self.sf.read(audio_file)
        
        # Ensure audio is mono and convert to float32
        if len(audio_data.shape) > 1:
            audio_data = self.np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(self.np.float32)
        
        return self.ov.Tensor(audio_data.reshape(1, -1))

    def cleanup_torchscript_cache(self):
        self.init()
        self.torch._C._jit_clear_class_registry()
        self.torch.jit._recursive.concrete_type_store = self.torch.jit._recursive.ConcreteTypeStore()
        self.torch.jit._state._clear_class_state()

    def init(self):
        if "sf" not in list(self.resources.keys()):
            import soundfile as sf
            self.sf = sf
        else:
            self.sf = self.resources["sf"]
            
        if "torch" not in list(self.resources.keys()):
            import torch
            self.resources["torch"] = torch
            self.torch = self.resources["torch"]
        else:
            self.torch = self.resources["torch"]

        if "transformers" not in list(self.resources.keys()):
            import transformers
            self.resources["transformers"] = transformers
            self.transformers = self.resources["transformers"]
        else:
            self.transformers = self.resources["transformers"]
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]
        return None
    
    def init_qualcomm(self, model, device, qualcomm_label):
        """
        Initialize CLAP model for Qualcomm hardware
        
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
            # Load processor directly from HuggingFace
            processor = self.transformers.ClapProcessor.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with SNPE
            model_name = model.replace("/", "--")
            dlc_path = f"~/snpe_models/{model_name}_clap.dlc"
            dlc_path = os.path.expanduser(dlc_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(dlc_path):
                print(f"Converting {model} to SNPE format...")
                self.snpe_utils.convert_model(model, "audio", str(dlc_path))
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Optimize for the specific Qualcomm device if possible
            if ":" in qualcomm_label:
                device_type = qualcomm_label.split(":")[1]
                optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                if optimized_path != dlc_path:
                    endpoint = self.snpe_utils.load_model(optimized_path)
            
            # Create endpoint handler
            endpoint_handler = self.create_qualcomm_audio_embedding_endpoint_handler(endpoint, processor, model, qualcomm_label)
            
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Qualcomm CLAP model: {e}")
            return None, None, None, None, 0

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        self.init()
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        audio_1 = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(sentence_1, audio_1)
        except Exception as e:
            print(e)
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
        print("hf_clap test")
        return None
    
    def init_cpu(self, model, device, cpu_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        processor = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
        try:
            endpoint = self.transformers.AutoModel.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading CPU model: {e}")
            endpoint = None
            
        endpoint_handler = self.create_cpu_audio_embedding_endpoint_handler(endpoint, processor, model, cpu_label)
        return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0

    def init_cuda(self, model, device, cuda_label):
        self.init()
        config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)    
        tokenizer = self.transformers.AutoTokenizer.from_pretrained(model)
        processor = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
        endpoint = None
        try:
            endpoint = self.transformers.AutoModel.from_pretrained(model, torch_dtype=self.torch.float16, trust_remote_code=True).to(device)
        except Exception as e:
            print(e)
            pass
        endpoint_handler = self.create_cuda_audio_embedding_endpoint_handler(endpoint, tokenizer, model, cuda_label)
        self.torch.cuda.empty_cache()
        # batch_size = await self.max_batch_size(endpoint_model, cuda_label)
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), 0    

    def init_openvino(self, model=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None ):
        self.init()
        if "ov" not in dir(self):
            if "openvino" not in list(self.resources.keys()):    
                import openvino as ov
                self.ov = ov
            else:
                self.ov = self.resources["openvino"]
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        homedir = os.path.expanduser("~")
        homedir = os.path.abspath(homedir)
        model_name_convert = model.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache","huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models = os.path.abspath(huggingface_cache_models)
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
        model_dst_path = os.path.abspath(model_dst_path)
        # if not os.path.exists(model_dst_path):
        #     # os.makedirs(model_dst_path)
        #     # convert model to openvino format
        #     # openvino_cli_convert(model, model_dst_path=model_dst_path, task=task, weight_format=weight_format, ratio="1.0", group_size=128, sym=True )
        #     pass
        try:
            tokenizer =  self.transformers.ClapProcessor.from_pretrained(
                model
            )
        except Exception as e:
            print(e)
            try:
                tokenizer =  self.transformers.ClapProcessor.from_pretrained(
                    model_src_path
                )
            except Exception as e:
                print(e)
                pass
        
            models_base_folder = model_dst_path
        
            clap_text_encoder_ir_path = os.path.join(models_base_folder, "clap_text_encoder.xml")
            clap_text_encoder_ir_path = os.path.abspath(clap_text_encoder_ir_path)
            # if not clap_text_encoder_ir_path.exists():
            #     with torch.no_grad():
            #         ov_model = ov.convert_model(
            #             ClapEncoderWrapper(pipe.text_encoder),  # model instance
            #             example_input={
            #                 "input_ids": torch.ones((1, 512), dtype=torch.long),
            #                 "attention_mask": torch.ones((1, 512), dtype=torch.long),
            #             },  # inputs for model tracing
            #         )
            #     ov.save_model(ov_model, clap_text_encoder_ir_path)
            #     del ov_model
            #     cleanup_torchscript_cache()
            #     gc.collect()
            #     print("Text Encoder successfully converted to IR")
            # else:
            #     print(f"Text Encoder will be loaded from {clap_text_encoder_ir_path}")
        
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
        endpoint_handler = self.create_openvino_audio_embedding_endpoint_handler(model, tokenizer, model, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size              
    
    def init_apple(self, model, device, apple_label):
        """Initialize CLAP model for Apple Silicon (M1/M2/M3) hardware."""
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
            processor = self.transformers.ClapProcessor.from_pretrained(model, trust_remote_code=True)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_clap.mlpackage"
            mlmodel_path = os.path.expanduser(mlmodel_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(mlmodel_path):
                print(f"Converting {model} to CoreML format...")
                self.coreml_utils.convert_model(model, "audio", str(mlmodel_path))
            
            # Load the CoreML model
            endpoint = self.coreml_utils.load_model(str(mlmodel_path))
            
            # Optimize for Apple Silicon if possible
            if ":" in apple_label:
                compute_units = apple_label.split(":")[1]
                optimized_path = self.coreml_utils.optimize_for_device(mlmodel_path, compute_units)
                if optimized_path != mlmodel_path:
                    endpoint = self.coreml_utils.load_model(optimized_path)
            
            endpoint_handler = self.create_apple_audio_embedding_endpoint_handler(endpoint, processor, model, apple_label)
            
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon CLAP model: {e}")
            return None, None, None, None, 0

    def create_cpu_audio_embedding_endpoint_handler(self, endpoint, processor, endpoint_model, cpu_label):
        """Creates an endpoint handler for CPU.
        
        Args:
            endpoint: The model endpoint
            processor: The audio processor
            endpoint_model: The model name or path
            cpu_label: Label to identify this endpoint
            
        Returns:
            A handler function for the CPU endpoint
        """
        def handler(x, y=None, endpoint=endpoint, processor=processor, endpoint_model=endpoint_model, cpu_label=cpu_label):
            if "eval" in dir(endpoint):
                endpoint.eval()
                
            try:
                with self.torch.no_grad():
                    # Handle text input
                    if x is not None:
                        if type(x) == str:
                            text_inputs = processor(
                                text=x,
                                return_tensors='pt',
                                padding=True
                            )
                        elif type(x) == list:
                            text_inputs = processor(text=[text for text in x], return_tensors='pt', padding=True)
                        
                        processed_data = {**text_inputs}
                        text_features = endpoint(**processed_data)
                        text_embeddings = text_features.text_embeds
                    
                    # Handle audio input
                    if y is not None:
                        if type(y) == str:
                            audio = self.load_audio(y)
                            audio_inputs = processor(
                                audios=[audio[0]], 
                                return_tensors='pt', 
                                padding=True,
                                sampling_rate=audio[1]
                            )
                        elif type(y) == list:
                            audio_inputs = processor(audios=[self.load_audio(audio_file)[0] for audio_file in y], 
                                                    return_tensors='pt',
                                                    sampling_rate=self.load_audio(y[0])[1])
                        
                        processed_data = {**audio_inputs}
                        audio_features = endpoint(**processed_data)
                        audio_embeddings = audio_features.audio_embeds
                
                # Return results based on what inputs were provided
                if x is not None and y is not None:
                    return {
                        'audio_embedding': audio_embeddings,
                        'text_embedding': text_embeddings
                    }
                elif x is not None:
                    return {'embedding': text_embeddings}
                elif y is not None:
                    return {'embedding': audio_embeddings}
                    
                return None
            except Exception as e:
                print(f"Error in CPU audio embedding handler: {e}")
                return None
                
        return handler
    
    def create_apple_audio_embedding_endpoint_handler(self, endpoint, processor, endpoint_model, apple_label):
        """Creates an Apple Silicon optimized handler for CLAP audio embedding models."""
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
                
                # Handle audio input
                if y is not None:
                    if type(y) == str:
                        audio, samplerate = load_audio(y)
                        audio_inputs = processor(
                            audios=[audio], 
                            return_tensors='np', 
                            padding=True,
                            sampling_rate=samplerate
                        )
                    elif type(y) == list:
                        audios_with_rates = [load_audio(audio_file) for audio_file in y]
                        audios = [audio[0] for audio in audios_with_rates]
                        # Use the sample rate from the first audio file
                        audio_inputs = processor(
                            audios=audios, 
                            return_tensors='np',
                            padding=True,
                            sampling_rate=audios_with_rates[0][1]
                        )
                    
                    # Ensure input_features key exists for CoreML
                    if "input_features" in audio_inputs:
                        inputs["input_features"] = audio_inputs["input_features"]
                    
                    # Some models use different key names
                    if "input_values" in audio_inputs:
                        inputs["input_values"] = audio_inputs["input_values"]
                
                # Run inference with CoreML
                results = self.coreml_utils.run_inference(endpoint, inputs)
                
                # Process results
                output = {}
                
                # Extract text embeddings
                if x is not None and "text_embeds" in results:
                    text_embeddings = self.torch.tensor(results["text_embeds"])
                    output["text_embedding"] = text_embeddings
                
                # Extract audio embeddings
                if y is not None and "audio_embeds" in results:
                    audio_embeddings = self.torch.tensor(results["audio_embeds"])
                    output["audio_embedding"] = audio_embeddings
                
                # If we have both text and audio, compute similarity
                if x is not None and y is not None and "text_embeds" in results and "audio_embeds" in results:
                    text_emb = self.torch.tensor(results["text_embeds"])
                    audio_emb = self.torch.tensor(results["audio_embeds"])
                    
                    # Normalize embeddings
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = self.torch.matmul(text_emb, audio_emb.T)
                    output["similarity"] = similarity
                
                # Return single embedding if that's all we have
                if len(output) == 1 and list(output.keys())[0] in ["text_embedding", "audio_embedding"]:
                    return {"embedding": list(output.values())[0]}
                    
                return output if output else None
                
            except Exception as e:
                print(f"Error in Apple Silicon audio embedding handler: {e}")
                return None
                
        return handler

    def create_qualcomm_audio_embedding_endpoint_handler(self, endpoint, processor, endpoint_model, qualcomm_label):
        """
        Creates a Qualcomm-optimized endpoint handler for CLAP audio embedding models
        
        Args:
            endpoint: The SNPE model endpoint
            processor: The audio processor
            endpoint_model: The model name or path
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            A handler function for the Qualcomm endpoint
        """
        def handler(x, y=None, endpoint=endpoint, processor=processor, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label):
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
                
                # Handle audio input
                if y is not None:
                    if type(y) == str:
                        audio, samplerate = load_audio(y)
                        audio_inputs = processor(
                            audios=[audio], 
                            return_tensors='np', 
                            padding=True,
                            sampling_rate=samplerate
                        )
                    elif type(y) == list:
                        audios_with_rates = [load_audio(audio_file) for audio_file in y]
                        audios = [audio[0] for audio in audios_with_rates]
                        # Use the sample rate from the first audio file
                        audio_inputs = processor(
                            audios=audios, 
                            return_tensors='np',
                            padding=True,
                            sampling_rate=audios_with_rates[0][1]
                        )
                    
                    # Ensure input_features key exists for SNPE
                    if "input_features" in audio_inputs:
                        inputs["input_features"] = audio_inputs["input_features"]
                    
                    # Some models use different key names
                    if "input_values" in audio_inputs:
                        inputs["input_values"] = audio_inputs["input_values"]
                
                # Run inference with SNPE
                results = self.snpe_utils.run_inference(endpoint, inputs)
                
                # Process results
                output = {}
                
                # Extract text embeddings
                if x is not None and "text_embeds" in results:
                    text_embeddings = self.torch.tensor(results["text_embeds"])
                    output["text_embedding"] = text_embeddings
                
                # Extract audio embeddings
                if y is not None and "audio_embeds" in results:
                    audio_embeddings = self.torch.tensor(results["audio_embeds"])
                    output["audio_embedding"] = audio_embeddings
                
                # If we have both text and audio, compute similarity
                if x is not None and y is not None and "text_embeds" in results and "audio_embeds" in results:
                    text_emb = self.torch.tensor(results["text_embeds"])
                    audio_emb = self.torch.tensor(results["audio_embeds"])
                    
                    # Normalize embeddings
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = self.torch.matmul(text_emb, audio_emb.T)
                    output["similarity"] = similarity
                
                # Return single embedding if that's all we have
                if len(output) == 1 and list(output.keys())[0] in ["text_embedding", "audio_embedding"]:
                    return {"embedding": list(output.values())[0]}
                    
                return output if output else None
                
            except Exception as e:
                print(f"Error in Qualcomm audio embedding handler: {e}")
                return None
                
        return handler

    def create_cuda_audio_embedding_endpoint_handler(self, tokenizer, endpoint_model, cuda_label, endpoint=None):
        def handler(x, tokenizer, endpoint_model, openvino_label, endpoint=None):
            if "eval" in dir(endpoint):
                endpoint.eval()
            else:
                pass
            return None
        return handler
    
    def create_openvino_audio_embedding_endpoint_handler(self, endpoint_model, tokenizer, openvino_label, endpoint=None):
        self.init()
        def handler(x, y, tokenizer=tokenizer, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=None):
            # text = "Replace me by any text you'd like."
            # audio_url = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
            # audio = load_audio(audio_url)
            # text_inputs = hftokenizer(text, return_tensors="pt", padding=True)
            # audio_inputs = hfprocessor(
            #     audios=[audio[0]],  # Use first channel only
            #     return_tensors="pt", 
            #     padding=True
            # )
            # processed_data = {**audio_inputs}
            # results = hfmodel(**processed_data)
            # hfmodel.config.torchscript = True
            # ov_model = ov.convert_model(hfmodel, example_input=processed_data)
            # if not os.path.exists(model_dst_path):
            #     os.mkdir(model_dst_path)
            # ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
            # ov_model = ov.compile_model(ov_model)
            # hfmodel = None

            if y is not None:            
                if type(y) == str:
                    audio =  load_audio(y)
                    audio_inputs = tokenizer(
                        audios=[audio[0]], 
                        return_tensors='pt', 
                        padding=True
                    )
                elif type(y) == list:
                    audio_inputs = tokenizer(images=[load_audio(y)[0] for image in y], return_tensors='pt')
                with self.torch.no_grad():
                    processed_data = {**audio_inputs}
                    image_features = endpoint_model(dict(processed_data))
                    # image_features = endpoint_model(**processed_data)
                    image_embeddings = image_features["audio_embeds"]
                pass
            
            if x is not None:
                if type(x) == str:
                    text_inputs = tokenizer(
                        text=y,
                        return_tensors='pt',
                        padding=True
                    )
                elif type(x) == list:
                    text_inputs = tokenizer(text=[text for text in x], return_tensors='pt', padding=True)
                with self.torch.no_grad():
                    processed_data = {**text_inputs}
                    text_features = endpoint_model(dict(processed_data))                    
                    # text_features = endpoint_model(**processed_data)                    
                    text_embeddings = text_features["text_embeds"] 
            
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
        self.init()
        if self.transformers is None:
            import transformers
            self.transformers = transformers
        if self.torch is None:
            import torch
            self.torch = torch
        if "ov" not in dir(self):
            if "openvino" not in list(self.resources.keys()):    
                import openvino as ov
                self.ov = ov
            else:
                self.ov = self.resources["openvino"]
        
        hfmodel = self.transformers.ClapModel.from_pretrained(model_name, torch_dtype=self.torch.float16)

        hfprocessor = self.transformers.ClapProcessor.from_pretrained(model_name)
        
        hftokenizer = self.transformers.ClapProcessor.from_pretrained(model_name)

        if hfprocessor is not None:
            if hfprocessor is not None:
                text = "Replace me by any text you'd like."
                audio_url = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
                audio = load_audio(audio_url)
                text_inputs = hftokenizer(text, return_tensors="pt", padding=True)
                audio_inputs = hfprocessor(
                    audios=[audio[0]],  # Use first channel only
                    return_tensors="pt", 
                    padding=True,
                    sampling_rate=audio[1]
                )
                hfmodel_dtype = hfmodel.dtype
                for key in audio_inputs:
                    if type(audio_inputs[key]) == self.torch.Tensor:
                        if audio_inputs[key].dtype != hfmodel_dtype:
                            audio_inputs[key] = audio_inputs[key].to(hfmodel_dtype)
                audio_inputs["input_ids"] = audio_inputs["input_features"]
                results = hfmodel(**audio_inputs)
                print(results)  # Use the results variable
                hfmodel.config.torchscript = True
                ov_model = ov.convert_model(hfmodel, example_input=audio_inputs)
                if not os.path.exists(model_dst_path):
                    os.mkdir(model_dst_path)
                ov.save_model(ov_model, os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml"))
                ov_model = ov.compile_model(ov_model)
                hfmodel = None
        return ov_model