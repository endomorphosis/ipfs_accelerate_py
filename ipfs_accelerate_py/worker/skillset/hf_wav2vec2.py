import os
import tempfile
import io
import json
import time
import asyncio
import requests
import gc
from pydub import AudioSegment
# from datasets import Dataset, Audio

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

def load_audio_16khz(audio_file):
    import librosa
    audio_data, samplerate = load_audio(audio_file)
    if samplerate != 16000:
        ## convert to 16khz
        audio_data = librosa.resample(y=audio_data, orig_sr=samplerate, target_sr=16000)
    return audio_data, 16000

def load_audio_tensor(audio_file):
    from openvino import ov, Tensor
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
    
    return Tensor(audio_data.reshape(1, -1))

import asyncio
import time
import os
from pathlib import Path

class hf_wav2vec2:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.init = self.init
        self.coreml_utils = None

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

    def init_cpu(self, model, device, cpu_label):
        self.init()
        try:
            processor = self.transformers.AutoProcessor.from_pretrained(model)
            endpoint = self.transformers.AutoModelForSpeechSeq2Seq.from_pretrained(model)
            endpoint_handler = self.create_cpu_wav2vec2_endpoint_handler(processor, model, cpu_label, endpoint)
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(e)
            return None, None, None, None, 0
    
    def init_cuda(self, model, device, cuda_label):
        self.init()
        try:
            processor = self.transformers.AutoProcessor.from_pretrained(model)
            endpoint = self.transformers.AutoModelForSpeechSeq2Seq.from_pretrained(model, torch_dtype=self.torch.float16).to(device)
            endpoint_handler = self.create_cuda_wav2vec2_endpoint_handler(processor, model, cuda_label, endpoint)
            self.torch.cuda.empty_cache()
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(e)
            return None, None, None, None, 0
    
    def init_openvino(self, model_name=None, model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None):
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
        model_name_convert = model_name.replace("/", "--")
        huggingface_cache = os.path.join(homedir, ".cache/huggingface")
        huggingface_cache_models = os.path.join(huggingface_cache, "hub")
        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
        huggingface_cache_models_files_dirs = [os.path.join(huggingface_cache_models, file) for file in huggingface_cache_models_files if os.path.isdir(os.path.join(huggingface_cache_models, file))]
        huggingface_cache_models_files_dirs_models = [ x for x in huggingface_cache_models_files_dirs if "model" in x ]
        huggingface_cache_models_files_dirs_models_model_name = [ x for x in huggingface_cache_models_files_dirs_models if model_name_convert in x ]
        model_src_path = os.path.join(huggingface_cache_models, huggingface_cache_models_files_dirs_models_model_name[0])
        model_dst_path = os.path.join(model_src_path, "openvino")
        task = get_openvino_pipeline_type(model_name, model_type)
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
            pass

        try:
            tokenizer =  self.transformers.Wav2Vec2Processor.from_pretrained(
                model_name
            )
        except Exception as e:
            print(e)
            try:
                tokenizer =  self.transformers.Wav2Vec2Processor.from_pretrained(
                    model_src_path
                )
            except Exception as e:
                print(e)
                pass
        
        try:
            model = get_openvino_model(model_name, model_type, openvino_label)
            print(model)
        except Exception as e:
            print(e)
            try:
                model = get_optimum_openvino_model(model_name, model_type, openvino_label)
            except Exception as e:
                print(e)
                pass
        endpoint = model
        endpoint_handler = self.create_openvino_wav2vec2_endpoint_handler(model, tokenizer, openvino_label)
        batch_size = 0
        return endpoint, tokenizer, endpoint_handler, asyncio.Queue(64), batch_size

    def init_apple(self, model, device, apple_label):
        """Initialize Wav2Vec2 model for Apple Silicon hardware."""
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
            processor = self.transformers.Wav2Vec2Processor.from_pretrained(model)
            
            # Convert model path to be compatible with CoreML
            model_name = model.replace("/", "--")
            mlmodel_path = f"~/coreml_models/{model_name}_wav2vec2.mlpackage"
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
            
            endpoint_handler = self.create_apple_audio_recognition_endpoint_handler(endpoint, processor, model, apple_label)
            
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon Wav2Vec2 model: {e}")
            return None, None, None, None, 0
            
    def create_apple_audio_recognition_endpoint_handler(self, endpoint, processor, model_name, apple_label):
        """Creates an Apple Silicon optimized handler for Wav2Vec2 audio recognition."""
        def handler(x, endpoint=endpoint, processor=processor, model_name=model_name, apple_label=apple_label):
            try:
                # Load and process audio
                if isinstance(x, str):
                    # Load audio file
                    audio_data, sample_rate = load_audio(x)
                    # Process audio input
                    inputs = processor(
                        audio_data, 
                        sampling_rate=sample_rate,
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
                if 'logits' in outputs:
                    logits = self.torch.tensor(outputs['logits'])
                    predictions = self.torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predictions)
                    return transcription[0] if transcription else None
                
                return None
                
            except Exception as e:
                print(f"Error in Apple Silicon Wav2Vec2 handler: {e}")
                return None
                
        return handler

    def init_qualcomm(self, model, device, qualcomm_label):
        """Initialize Wav2Vec2 model for Qualcomm hardware.
        
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
            dlc_path = f"~/snpe_models/{model_name}_wav2vec2.dlc"
            dlc_path = os.path.expanduser(dlc_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(dlc_path), exist_ok=True)
            
            # Convert or load the model
            if not os.path.exists(dlc_path):
                print(f"Converting {model} to SNPE format...")
                self.snpe_utils.convert_model(model, "speech", str(dlc_path))
            
            # Load the SNPE model
            endpoint = self.snpe_utils.load_model(str(dlc_path))
            
            # Optimize for the specific Qualcomm device if possible
            if ":" in qualcomm_label:
                device_type = qualcomm_label.split(":")[1]
                optimized_path = self.snpe_utils.optimize_for_device(dlc_path, device_type)
                if optimized_path != dlc_path:
                    endpoint = self.snpe_utils.load_model(optimized_path)
            
            # Create endpoint handler
            endpoint_handler = self.create_qualcomm_wav2vec2_endpoint_handler(processor, model, qualcomm_label, endpoint)
            
            return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Qualcomm Wav2Vec2 model: {e}")
            return None, None, None, None, 0
    
    def create_cpu_wav2vec2_endpoint_handler(self, processor, endpoint_model, cpu_label, endpoint):
        def handler(x, processor=processor, endpoint_model=endpoint_model, cpu_label=cpu_label, endpoint=endpoint):
            if "eval" in dir(endpoint) and endpoint is not None:
                endpoint.eval()
            
            with self.torch.no_grad():
                try:
                    if type(x) == str:
                        audio_data, audio_sampling_rate = load_audio_16khz(x)
                        inputs = processor(
                            audio_data,
                            return_tensors="pt",
                            padding="longest",
                            sampling_rate=audio_sampling_rate,
                        )
                        
                        outputs = endpoint(**inputs)
                        embeddings = outputs.last_hidden_state
                        # Average pooling
                        embeddings = self.torch.mean(embeddings, dim=1)
                        
                        return {
                            'embedding': embeddings[0].detach().numpy().tolist()
                        }
                except Exception as e:
                    print(f"CPU audio embedding error: {e}")
                    return None
            return None
        return handler
    
    def create_cuda_wav2vec2_endpoint_handler(self, processor, endpoint_model, cuda_label, endpoint):
        def handler(x, processor=processor, endpoint_model=endpoint_model, cuda_label=cuda_label, endpoint=endpoint):
            if "eval" in dir(endpoint) and endpoint is not None:
                endpoint.eval()
            
            with self.torch.no_grad():
                try:
                    self.torch.cuda.empty_cache()
                    if type(x) == str:
                        audio_data, audio_sampling_rate = load_audio_16khz(x)
                        inputs = processor(
                            audio_data,
                            return_tensors="pt",
                            padding="longest",
                            sampling_rate=audio_sampling_rate,
                        )
                        
                        # Move inputs to the GPU
                        for key in inputs:
                            if isinstance(inputs[key], self.torch.Tensor):
                                inputs[key] = inputs[key].to(endpoint.device)
                        
                        outputs = endpoint(**inputs)
                        embeddings = outputs.last_hidden_state
                        # Average pooling
                        embeddings = self.torch.mean(embeddings, dim=1)
                        
                        result = embeddings[0].cpu().detach().numpy().tolist()
                        
                        # Clean up GPU memory
                        del inputs, outputs, embeddings
                        self.torch.cuda.empty_cache()
                        
                        return {
                            'embedding': result
                        }
                except Exception as e:
                    if 'inputs' in locals(): del inputs
                    if 'outputs' in locals(): del outputs
                    if 'embeddings' in locals(): del embeddings
                    self.torch.cuda.empty_cache()
                    print(f"CUDA audio embedding error: {e}")
                    return None
            return None
        return handler
    
    def create_openvino_wav2vec2_endpoint_handler(self, endpoint, processor, openvino_label, endpoint_model=None):
        def handler(x, processor=processor, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=None):
            results = []
            if x is not None:            
                if type(x) == str:
                    audio_data, audio_sampling_rate = load_audio_16khz(x)                    
                    preprocessed_signal = processor(
                        audio_data,
                        return_tensors="pt",
                        padding="longest",
                        sampling_rate=audio_sampling_rate,
                    )
                    audio_inputs = preprocessed_signal.input_values
                    MAX_SEQ_LENGTH = 30480
                    if audio_inputs.shape[1] > MAX_SEQ_LENGTH:
                        audio_inputs = audio_inputs[:, :MAX_SEQ_LENGTH]
                    image_features = endpoint_model({'input_values': audio_inputs})
                    image_embeddings = list(image_features.values())[0]
                    image_embeddings = self.torch.tensor(image_embeddings)
                    image_embeddings = self.torch.mean(image_embeddings, dim=(1,))
                    results.append(image_embeddings)
                elif type(x) == list:
                    inputs = processor(images=[load_audio_16khz(image) for image in x], return_tensors='pt')
                    image_features = endpoint_model({'input_values': audio_inputs})
                    image_embeddings = list(image_features.values())[0]
                    image_embeddings = self.torch.tensor(image_embeddings)
                    image_embeddings = self.torch.mean(image_embeddings, dim=1)
                    results.append(image_embeddings)
                pass            

                if results is not None:                                        
                    if x is not None:
                        return {
                            'embedding': image_embeddings[0]
                        }            
            return None
        return handler

    def create_apple_wav2vec2_endpoint_handler(self, processor, endpoint_model, apple_label, endpoint):
        """Creates an endpoint handler for Apple Silicon.
        
        Args:
            processor: The audio processor
            endpoint_model: The model name or path
            apple_label: Label to identify this endpoint
            endpoint: The model endpoint
            
        Returns:
            A handler function for the Apple endpoint
        """
        def handler(audio_input, processor=processor, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint):
            if "eval" in dir(endpoint):
                endpoint.eval()
                
            try:
                # Process audio input
                if isinstance(audio_input, str):
                    # Load audio file
                    from .hf_clap import load_audio
                    audio_data, sample_rate = load_audio(audio_input)
                    
                    # Get audio features
                    with self.torch.no_grad():
                        inputs = processor(
                            audio=audio_data, 
                            sampling_rate=sample_rate, 
                            return_tensors="pt"
                        )
                        
                        # Move inputs to MPS device if available
                        if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                            inputs = {k: v.to("mps") if hasattr(v, 'to') else v for k, v in inputs.items()}
                        
                        # Generate transcription
                        if hasattr(endpoint, "generate"):
                            # For models like Whisper that have generate method
                            generated_ids = endpoint.generate(
                                inputs["input_features"] if "input_features" in inputs else inputs["input_values"]
                            )
                            
                            # Move results back to CPU for processing
                            if hasattr(generated_ids, "cpu"):
                                generated_ids = generated_ids.cpu()
                            
                            # Decode transcription
                            transcription = processor.batch_decode(
                                generated_ids, 
                                skip_special_tokens=True
                            )[0]
                        else:
                            # For models that return logits directly
                            outputs = endpoint(**inputs)
                            logits = outputs.logits
                            
                            # Move results back to CPU for processing
                            if hasattr(logits, "cpu"):
                                logits = logits.cpu()
                                
                            # Get predicted ids
                            predicted_ids = self.torch.argmax(logits, dim=-1)
                            
                            # Decode transcription
                            transcription = processor.batch_decode(predicted_ids)[0]
                        
                        return {
                            "text": transcription,
                            "model": endpoint_model
                        }
                else:
                    # Assume it's already processed
                    return {"error": "Unsupported input format"}
            
            except Exception as e:
                print(f"Error in Apple Wav2Vec2 endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler
    
    def create_qualcomm_wav2vec2_endpoint_handler(self, processor, endpoint_model, qualcomm_label, endpoint):
        """Creates an endpoint handler for Qualcomm hardware.
        
        Args:
            processor: The audio processor
            endpoint_model: The model name or path
            qualcomm_label: Label to identify this endpoint
            endpoint: The SNPE model endpoint
            
        Returns:
            A handler function for the Qualcomm endpoint
        """
        def handler(audio_input, processor=processor, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint):
            try:
                # Process audio input
                if isinstance(audio_input, str):
                    # Load audio file
                    from .hf_clap import load_audio
                    audio_data, sample_rate = load_audio(audio_input)
                    
                    # Get audio features
                    inputs = processor(
                        audio=audio_data, 
                        sampling_rate=sample_rate, 
                        return_tensors="np"
                    )
                    
                    # Run inference with SNPE
                    results = self.snpe_utils.run_inference(endpoint, inputs)
                    
                    # Process results to get transcription
                    if "logits" in results:
                        # Convert logits to predicted ids
                        logits = self.torch.tensor(results["logits"])
                        predicted_ids = self.torch.argmax(logits, dim=-1)
                        
                        # Decode transcription
                        if hasattr(processor, "batch_decode"):
                            transcription = processor.batch_decode(predicted_ids)[0]
                        elif hasattr(processor, "decode"):
                            transcription = processor.decode(predicted_ids[0])
                        else:
                            # If no specific decode method exists, try with tokenizer
                            if hasattr(processor, "tokenizer"):
                                transcription = processor.tokenizer.decode(predicted_ids[0])
                            else:
                                transcription = "[Transcription unavailable - decode method not found]"
                        
                        return {
                            "text": transcription,
                            "model": endpoint_model
                        }
                    else:
                        return {
                            "error": "Unexpected output format from model"
                        }
                else:
                    # Assume it's already processed
                    return {"error": "Unsupported input format"}
            
            except Exception as e:
                print(f"Error in Qualcomm Wav2Vec2 endpoint handler: {e}")
                return {"error": str(e)}
                
        return handler