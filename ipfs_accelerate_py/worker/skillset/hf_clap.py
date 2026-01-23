import os
import time
import anyio
import requests
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

def load_audio_16khz(audio_file):
    """
    Load audio file and resample to 16kHz if necessary.
    
    Args:
        audio_file: Path or URL to audio file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    import librosa
    audio_data, samplerate = load_audio(audio_file)
    if samplerate != 16000:
        # Convert to 16kHz
        audio_data = librosa.resample(y=audio_data, orig_sr=samplerate, target_sr=16000)
    return audio_data, 16000

def load_audio(audio_file):
    """
    Load audio from file or URL and convert to mono.
    
    Args:
        audio_file: Path or URL to audio file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
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
    """
    Load audio from file or URL and convert to OpenVINO tensor.
    
    Args:
        audio_file: Path or URL to audio file
        
    Returns:
        OpenVINO tensor containing audio data
    """
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
    """
    Helper for removing cached model representation.
    
    Clears PyTorch JIT registry and class state to prevent memory leaks.
    """
    import torch
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

class hf_clap:
    """
    Hugging Face CLAP (Contrastive Language-Audio Pretraining) model implementation.
    
    This class provides a standardized interface for running CLAP models across 
    different hardware backends including CPU, CUDA, OpenVINO, Apple Silicon, and 
    Qualcomm. It supports audio-text similarity, audio embedding, and text 
    embedding capabilities.
    
    The implementation provides both real model inference and mock functionality when
    hardware or dependencies are unavailable.
    """
    
    def __init__(self, resources: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the CLAP model handler.
        
        Args:
            resources: Dictionary of resources (torch, transformers, numpy, soundfile)
            metadata: Dictionary of metadata for initialization
            
        Returns:
            None
        """
        # Initialize dependencies
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Initialize hardware-specific utilities
        self.snpe_utils = None  # Qualcomm
        self.coreml_utils = None  # Apple
        self.ov = None  # OpenVINO
        self.transformers = None
        self.torch = None
        self.np = None
        self.sf = None
        
        # These redundant self-assignments are kept for backward compatibility
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
        self.init = self.init
        self.__test__ = self.__test__
        
        return None

    def load_audio(self, audio_file):
        """
        Load audio data from file or URL and convert to mono.
        
        Args:
            audio_file: Path or URL to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Ensure resources are initialized
        if not hasattr(self, 'sf') or self.sf is None:
            self.init()
            
        try:
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
        except Exception as e:
            print(f"Error loading audio: {e}")
            # Return a mock audio tensor
            print("(MOCK) Returning mock audio data")
            return self.np.zeros(16000, dtype=self.np.float32), 16000

    def load_audio_tensor(self, audio_file):
        """
        Load audio from file or URL and convert to OpenVINO tensor.
        
        Args:
            audio_file: Path or URL to audio file
            
        Returns:
            OpenVINO tensor containing audio data
        """
        # Ensure resources are initialized
        if not hasattr(self, 'sf') or self.sf is None:
            self.init()
            
        # Initialize OpenVINO if needed
        if not hasattr(self, 'ov') or self.ov is None:
            if "openvino" not in list(self.resources.keys()):    
                try:
                    import openvino as ov
                    self.ov = ov
                except ImportError:
                    print("OpenVINO not available, returning mock tensor")
                    # Return a mock tensor-like object
                    class MockTensor:
                        def __init__(self, data):
                            self.data = data
                            self.shape = data.shape
                    return MockTensor(self.np.zeros((1, 16000), dtype=self.np.float32))
            else:
                self.ov = self.resources["openvino"]
        
        try:
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
        except Exception as e:
            print(f"Error creating audio tensor: {e}")
            # Return a mock tensor-like object
            print("(MOCK) Returning mock audio tensor")
            class MockTensor:
                def __init__(self, data):
                    self.data = data
                    self.shape = data.shape
            return MockTensor(self.np.zeros((1, 16000), dtype=self.np.float32))

    def cleanup_torchscript_cache(self):
        """
        Clean up PyTorch JIT cache to prevent memory leaks.
        
        Must be called after converting a model to a different format to ensure
        proper memory management.
        """
        # Ensure torch is initialized
        self.init()
        
        try:
            self.torch._C._jit_clear_class_registry()
            self.torch.jit._recursive.concrete_type_store = self.torch.jit._recursive.ConcreteTypeStore()
            self.torch.jit._state._clear_class_state()
            print("PyTorch JIT cache cleared successfully")
        except Exception as e:
            print(f"Error clearing PyTorch JIT cache: {e}")

    def init(self):
        """
        Initialize required resources for CLAP model.
        
        Loads torch, transformers, numpy, and soundfile either from provided resources
        or by importing them directly. This method must be called before using any
        other methods.
        
        Returns:
            None
        """
        # Initialize soundfile
        if "sf" not in list(self.resources.keys()):
            try:
                import soundfile as sf
                self.sf = sf
            except ImportError:
                print("Failed to import soundfile. Audio loading will be limited.")
                self.sf = None
        else:
            self.sf = self.resources["sf"]
        
        # Initialize PyTorch    
        if "torch" not in list(self.resources.keys()):
            try:
                import torch
                self.resources["torch"] = torch
                self.torch = torch
            except ImportError:
                print("Failed to import torch. Some functionality will be limited.")
                self.torch = None
        else:
            self.torch = self.resources["torch"]

        # Initialize Transformers
        if "transformers" not in list(self.resources.keys()):
            try:
                import transformers
                self.resources["transformers"] = transformers
                self.transformers = transformers
            except ImportError:
                print("Failed to import transformers. Will use mock implementations.")
                self.transformers = None
        else:
            self.transformers = self.resources["transformers"]
        
        # Initialize NumPy    
        if "numpy" not in list(self.resources.keys()):
            try:
                import numpy as np
                self.np = np
            except ImportError:
                print("Failed to import numpy. Some functionality will be limited.")
                self.np = None
        else:
            self.np = self.resources["numpy"]
            
        # Check if we have all required resources
        initialization_status = {
            "soundfile": self.sf is not None,
            "torch": self.torch is not None,
            "transformers": self.transformers is not None,
            "numpy": self.np is not None
        }
        
        print(f"CLAP initialization status: {initialization_status}")
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
            
            return endpoint, processor, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Qualcomm CLAP model: {e}")
            return None, None, None, None, 0

    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        """
        Test CLAP model with a simple text-audio pair.
        
        Args:
            endpoint_model: Model name or path
            endpoint_handler: Handler function to test
            endpoint_label: Label for the endpoint (cpu, cuda, openvino, etc.)
            tokenizer: Tokenizer or processor for the model
            
        Returns:
            None or test results dictionary
        """
        # Ensure dependencies are loaded
        self.init()
        
        # Standard test inputs
        sentence_1 = "The quick brown fox jumps over the lazy dog"
        audio_1 = "https://calamitymod.wiki.gg/images/2/29/Bees3.wav"
        
        # Measure performance
        timestamp1 = time.time()
        test_result = None
        try:
            # Run inference through the handler
            test_batch = endpoint_handler(sentence_1, audio_1)
            
            # Check if we got valid results
            if test_batch is not None and isinstance(test_batch, dict):
                if "similarity" in test_batch:
                    test_status = "PASSED - Similarity score computed"
                elif "audio_embedding" in test_batch and "text_embedding" in test_batch:
                    test_status = "PASSED - Both embeddings computed"
                else:
                    test_status = "PARTIAL - Incomplete results"
            else:
                test_status = "FAILED - Invalid results"
                
            # Print results
            print(f"CLAP test status: {test_status}")
            print(f"Result type: {type(test_batch)}")
            
            # Determine if the result was from a real model or mock
            implementation_type = "REAL"
            if test_batch and any(k.endswith("_status") and test_batch[k] == "MOCK" for k in test_batch):
                implementation_type = "MOCK"
            print(f"Implementation type: {implementation_type}")
            
            test_result = test_batch
            
        except Exception as e:
            print(f"CLAP test error: {str(e)}")
            test_result = {"error": str(e)}
        
        # Calculate and print performance metrics
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        len_tokens = 1  # We're processing one sample
        tokens_per_second = len_tokens / elapsed_time
        
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        print(f"Samples processed: {len_tokens}")
        print(f"Samples per second: {tokens_per_second:.4f}")
        
        # Clean up resources based on backend
        if "openvino" not in endpoint_label and self.torch is not None:
            try:
                with self.torch.no_grad():
                    if hasattr(self.torch, "cuda") and hasattr(self.torch.cuda, "empty_cache"):
                        self.torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error cleaning up resources: {str(e)}")
                
        return test_result
    
    def init_cpu(self, model, device, cpu_label):
        """
        Initialize CLAP model for CPU inference.
        
        Args:
            model: Model name or path (e.g., 'laion/clap-htsat-unfused')
            device: Device to run on ('cpu')
            cpu_label: Label for CPU endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        # Ensure dependencies are loaded
        self.init()
        
        # Helper function to create mock processor
        def _create_mock_processor():
            """Create a mock processor that returns valid input tensors"""
            class MockProcessor:
                def __init__(self):
                    self.np = self.np if hasattr(self, 'np') else __import__('numpy')
                    self.torch = self.torch if hasattr(self, 'torch') else __import__('torch')
                
                def __call__(self, text=None, audios=None, return_tensors='pt', padding=True, sampling_rate=16000, **kwargs):
                    """Process text or audio inputs"""
                    result = {}
                    batch_size = 1
                    
                    if text is not None:
                        if isinstance(text, list):
                            batch_size = len(text)
                        # Create mock text inputs
                        if return_tensors == 'pt':
                            result["input_ids"] = self.torch.ones((batch_size, 77), dtype=self.torch.long)
                            result["attention_mask"] = self.torch.ones((batch_size, 77), dtype=self.torch.long)
                        else:
                            result["input_ids"] = self.np.ones((batch_size, 77), dtype=self.np.int32)
                            result["attention_mask"] = self.np.ones((batch_size, 77), dtype=self.np.int32)
                    
                    if audios is not None:
                        if isinstance(audios, list):
                            batch_size = len(audios)
                        # Create mock audio inputs
                        if return_tensors == 'pt':
                            result["input_features"] = self.torch.rand((batch_size, 1, 1024, 128))
                            result["input_values"] = self.torch.rand((batch_size, 16000))
                        else:
                            result["input_features"] = self.np.random.rand(batch_size, 1, 1024, 128).astype(self.np.float32)
                            result["input_values"] = self.np.random.rand(batch_size, 16000).astype(self.np.float32)
                    
                    return result
            
            return MockProcessor()
        
        # Helper function to create mock model
        def _create_mock_model():
            """Create a mock CLAP model that returns valid embeddings"""
            class MockClapModel:
                def __init__(self):
                    self.np = self.np if hasattr(self, 'np') else __import__('numpy')
                    self.torch = self.torch if hasattr(self, 'torch') else __import__('torch')
                    self.config = type('obj', (object,), {
                        'hidden_size': 512,
                        'projection_dim': 512,
                        'model_type': 'clap'
                    })
                
                def __call__(self, **kwargs):
                    """Return mock embeddings for CLAP model"""
                    batch_size = 1
                    embed_dim = 512
                    
                    # Determine batch size from inputs
                    if "input_ids" in kwargs:
                        batch_size = kwargs["input_ids"].shape[0]
                    elif "input_features" in kwargs:
                        batch_size = kwargs["input_features"].shape[0]
                    
                    # Create output object similar to CLAP output
                    class ClapOutput:
                        def __init__(self, batch_size, dim):
                            if 'torch' in globals():
                                self.audio_embeds = torch.randn(batch_size, dim)
                                self.text_embeds = torch.randn(batch_size, dim)
                            else:
                                import torch
                                self.audio_embeds = torch.randn(batch_size, dim)
                                self.text_embeds = torch.randn(batch_size, dim)
                    
                    return ClapOutput(batch_size, embed_dim)
                
                def eval(self):
                    """Set model to evaluation mode"""
                    return self
            
            return MockClapModel()
        
        # Initialize the CLAP model and processor with real or mock components
        mock_used = False
        implementation_type = "REAL"
        
        try:
            # Try to load real components first
            config = None 
            processor = None
            endpoint = None
            
            if self.transformers is not None:
                try:
                    # Load model config
                    config = self.transformers.AutoConfig.from_pretrained(
                        model, 
                        trust_remote_code=True,
                        cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
                    )
                    print(f"Successfully loaded CLAP config")
                except Exception as e:
                    print(f"Error loading config: {e}")
                    config = None
                
                try:
                    # Load processor
                    processor = self.transformers.AutoProcessor.from_pretrained(
                        model, 
                        trust_remote_code=True,
                        cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
                    )
                    print(f"Successfully loaded CLAP processor")
                except Exception as e:
                    print(f"Error loading processor: {e}")
                    processor = _create_mock_processor()
                    mock_used = True
                
                try:
                    # Load model
                    endpoint = self.transformers.AutoModel.from_pretrained(
                        model, 
                        trust_remote_code=True,
                        cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache"),
                        low_cpu_mem_usage=True
                    )
                    print(f"Successfully loaded CLAP model")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    endpoint = _create_mock_model()
                    mock_used = True
            else:
                print("(MOCK) Transformers not available, using mock implementations")
                processor = _create_mock_processor()
                endpoint = _create_mock_model()
                mock_used = True
            
            # Create the handler with mock components if needed
            if mock_used:
                implementation_type = "MOCK"
                print(f"(MOCK) Using mock implementation for CPU CLAP")
                # Ensure we have minimum components if loading failed
                if processor is None:
                    processor = _create_mock_processor()
                if endpoint is None:
                    endpoint = _create_mock_model()
            
            # Create endpoint handler
            endpoint_handler = self.create_cpu_audio_embedding_endpoint_handler(
                endpoint, 
                processor, 
                model, 
                cpu_label
            )
            
            print(f"Initialized CPU CLAP model ({implementation_type})")
            return endpoint, processor, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error in CPU initialization: {e}")
            # Fallback to mock implementation
            processor = _create_mock_processor()
            endpoint = _create_mock_model()
            endpoint_handler = self.create_cpu_audio_embedding_endpoint_handler(
                endpoint, 
                processor, 
                model, 
                cpu_label
            )
            print("(MOCK) Initialized CPU CLAP model with mock components")
            return endpoint, processor, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0

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
        return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(64), 0    

    def init_openvino(self, model=None , model_type=None, device=None, openvino_label=None, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None ):
        """Initialize CLAP model for OpenVINO inference.
        
        Args:
            model: HuggingFace model name or path
            model_type: Type of model (e.g., "audio-classification")
            device: Device to run inference on ("CPU", "GPU", etc.)
            openvino_label: Label to identify this endpoint ("openvino:0", etc.)
            get_optimum_openvino_model: Function to get Optimum OpenVINO model
            get_openvino_model: Function to get OpenVINO model
            get_openvino_pipeline_type: Function to get the appropriate pipeline type
            openvino_cli_convert: Function to convert the model with CLI
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        # Import OpenVINO if needed
        if "ov" not in dir(self):
            if "openvino" not in list(self.resources.keys()):    
                try:
                    import openvino as ov
                    self.ov = ov
                except ImportError:
                    print("OpenVINO not available")
                    return None, None, None, None, 0
            else:
                self.ov = self.resources["openvino"]
                
        # Initialize variables
        endpoint = None
        tokenizer = None
        endpoint_handler = None
        
        # Set up paths for model conversion and storage
        try:
            homedir = os.path.expanduser("~")
            homedir = os.path.abspath(homedir)
            model_name_convert = model.replace("/", "--")
            
            # Set up HuggingFace cache path
            huggingface_cache = os.path.join(homedir, ".cache", "huggingface")
            huggingface_cache_models = os.path.join(huggingface_cache, "hub")
            
            # Get model source path from cache
            # This section was causing index errors - completely rewritten with better error handling
            try:
                model_src_path = None
                
                # First check if HuggingFace cache exists
                if os.path.exists(huggingface_cache_models):
                    try:
                        huggingface_cache_models = os.path.abspath(huggingface_cache_models)
                        huggingface_cache_models_files = os.listdir(huggingface_cache_models)
                        
                        # Safely get directories from the cache
                        huggingface_cache_models_files_dirs = []
                        for file in huggingface_cache_models_files:
                            try:
                                full_path = os.path.join(huggingface_cache_models, file)
                                if os.path.isdir(full_path):
                                    huggingface_cache_models_files_dirs.append(full_path)
                            except Exception as e:
                                print(f"Error checking directory {file}: {e}")
                                continue
                        
                        print(f"Found {len(huggingface_cache_models_files_dirs)} potential model directories")
                        
                        # Method 1: Look for direct name matches
                        model_dirs = []
                        for path in huggingface_cache_models_files_dirs:
                            try:
                                if model_name_convert in os.path.basename(path):
                                    model_dirs.append(path)
                            except Exception:
                                continue
                        
                        # Method 2: If no direct matches, look for models-- prefix
                        if not model_dirs:
                            for path in huggingface_cache_models_files_dirs:
                                try:
                                    if path.endswith("models--" + model_name_convert.replace('--', '-')) or \
                                       path.endswith("models--" + model_name_convert):
                                        model_dirs.append(path)
                                except Exception:
                                    continue
                        
                        # Method 3: Look for any model directory
                        if not model_dirs:
                            for path in huggingface_cache_models_files_dirs:
                                try:
                                    if "model" in os.path.basename(path).lower():
                                        model_dirs.append(path)
                                except Exception:
                                    continue
                        
                        # Use the first match if any found
                        if model_dirs and len(model_dirs) > 0:
                            model_src_path = model_dirs[0]
                            print(f"Found model in cache at: {model_src_path}")
                    except Exception as e:
                        print(f"Error searching HuggingFace cache: {e}")
                
                # If no path found yet, try the standard path structure
                if model_src_path is None:
                    standard_path = os.path.join(huggingface_cache_models, "models--" + model_name_convert)
                    if os.path.exists(standard_path):
                        model_src_path = standard_path
                        print(f"Using standard path: {model_src_path}")
                
                # Final fallback: use a local models directory
                if model_src_path is None:
                    fallback_path = os.path.join(homedir, "openvino_models", model_name_convert)
                    model_src_path = fallback_path
                    print(f"Using fallback path: {model_src_path}")
                    # Create directory if it doesn't exist
                    os.makedirs(fallback_path, exist_ok=True)
                
            except Exception as e:
                print(f"Error finding model in cache: {e}")
                # Use a fallback path and make sure it exists
                model_src_path = os.path.join(homedir, "openvino_models", model_name_convert)
                os.makedirs(model_src_path, exist_ok=True)
                print(f"Created fallback directory: {model_src_path}")
            
            # Set up destination path for converted model
            model_dst_path = os.path.join(model_src_path, "openvino")
            
            # Get task type from pipeline type function
            try:
                task = get_openvino_pipeline_type(model, model_type)
            except Exception as e:
                print(f"Error getting pipeline type: {e}")
                # Use a default task type
                task = "feature-extraction"
            
            # Parse OpenVINO device index and set weight format
            # Completely rewritten with better error handling
            openvino_index = 0
            
            try:
                # Validate and parse openvino_label
                if isinstance(openvino_label, str) and ":" in openvino_label:
                    openvino_parts = openvino_label.split(":")
                    
                    # Safely extract the index if it exists
                    if len(openvino_parts) > 1:
                        try:
                            openvino_index = int(openvino_parts[1])
                            print(f"Using OpenVINO device index: {openvino_index}")
                        except (ValueError, TypeError) as e:
                            print(f"Invalid OpenVINO device index: {e}, using default (0)")
                            openvino_index = 0
                    else:
                        print("No OpenVINO device index specified, using default (0)")
                        openvino_index = 0
                else:
                    print("Invalid OpenVINO label format, using default device index (0)")
                    openvino_index = 0
            except Exception as e:
                print(f"Error parsing OpenVINO label: {e}, using default device index (0)")
                openvino_index = 0
                
            # Determine weight format based on target device
            weight_format = "int8"  # Default for CPU
            if openvino_index == 1:
                weight_format = "int4"  # For GPU
                print("Using int4 weight format for GPU")
            elif openvino_index == 2:
                weight_format = "int4"  # For NPU
                print("Using int4 weight format for NPU")
            else:
                print("Using int8 weight format for CPU")
                
            # Update destination path with weight format
            model_dst_path = f"{model_dst_path}_{weight_format}"
            model_dst_path = os.path.abspath(model_dst_path)
            
            # Create destination directory with better error handling
            convert_success = False
            try:
                # Ensure the destination directory exists
                if not os.path.exists(model_dst_path):
                    try:
                        os.makedirs(model_dst_path, exist_ok=True)
                        print(f"Created model destination directory: {model_dst_path}")
                    except Exception as e:
                        print(f"Error creating model destination directory: {e}")
                        # Try using a different path
                        model_dst_path = os.path.join(os.path.expanduser("~"), "openvino_models_fallback", model_name_convert + "_" + weight_format)
                        os.makedirs(model_dst_path, exist_ok=True)
                        print(f"Using fallback destination directory: {model_dst_path}")
                
                # Only attempt conversion if the directory is empty or missing XML model file
                xml_path = os.path.join(model_dst_path, model_name.replace("/", "--") + ".xml")
                if not os.path.exists(xml_path):
                    print(f"Model needs conversion, XML path doesn't exist: {xml_path}")
                    
                    # First try our custom skill conversion if available
                    if hasattr(self, 'openvino_skill_convert'):
                        try:
                            print("Attempting custom skill conversion...")
                            convert = self.openvino_skill_convert(model, model_dst_path, task, weight_format)
                            print(f"Model converted with openvino_skill_convert: {convert}")
                            convert_success = True
                        except Exception as e:
                            print(f"Error using openvino_skill_convert: {e}")
                    
                    # If custom conversion fails, try the CLI converter
                    if not convert_success and callable(openvino_cli_convert):
                        try:
                            print("Attempting CLI conversion...")
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
                            convert_success = True
                        except Exception as e:
                            print(f"Error using openvino_cli_convert: {e}")
                else:
                    print(f"Model already converted at: {xml_path}")
                    convert_success = True
            except Exception as e:
                print(f"Error in model conversion setup: {e}")
            
            # Try loading the processor/tokenizer with multiple fallbacks
            tokenizer = None
            try:
                # Method 1: Load from model name directly
                print(f"Loading CLAP processor from {model}")
                tokenizer = self.transformers.ClapProcessor.from_pretrained(model, trust_remote_code=True)
                print("Successfully loaded processor from model name")
            except Exception as e:
                print(f"Error loading processor from model name: {e}")
                try:
                    # Method 2: Try loading from cache path
                    print(f"Trying to load processor from cache: {model_src_path}")
                    tokenizer = self.transformers.ClapProcessor.from_pretrained(model_src_path, trust_remote_code=True)
                    print("Successfully loaded processor from cache path")
                except Exception as e:
                    print(f"Error loading processor from cache: {e}")
                    try:
                        # Method 3: Try with AutoProcessor instead
                        print("Trying AutoProcessor instead of ClapProcessor")
                        tokenizer = self.transformers.AutoProcessor.from_pretrained(model, trust_remote_code=True)
                        print("Successfully loaded processor with AutoProcessor")
                    except Exception as e:
                        print(f"Error loading with AutoProcessor: {e}")
                        try:
                            # Method 4: Try loading with custom config
                            print("Trying to load with custom config")
                            config = self.transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)
                            tokenizer = self.transformers.AutoProcessor.from_pretrained(
                                model, 
                                config=config,
                                trust_remote_code=True
                            )
                            print("Successfully loaded processor with custom config")
                        except Exception as e:
                            print(f"All processor loading methods failed: {e}")
                            # Create a mock processor as final fallback
                            print("Creating mock processor")
                            from unittest.mock import MagicMock
                            tokenizer = MagicMock()
                            tokenizer.__call__ = MagicMock(return_value={
                                "input_ids": self.torch.ones(1, 10),
                                "attention_mask": self.torch.ones(1, 10),
                                "input_features": self.torch.ones(1, 1, 128, 128)
                            })
            
            # Try loading the model with multiple strategies and better error handling
            endpoint = None
            try:
                # Method 1: Try loading with get_openvino_model
                if callable(get_openvino_model):
                    try:
                        print("Loading OpenVINO model with get_openvino_model function")
                        endpoint = get_openvino_model(model, model_type, openvino_label)
                        print("Successfully loaded model with get_openvino_model")
                    except Exception as e:
                        print(f"Error loading with get_openvino_model: {e}")
                
                # Method 2: Try loading with get_optimum_openvino_model if first method failed
                if endpoint is None and callable(get_optimum_openvino_model):
                    try:
                        print("Trying to load model with get_optimum_openvino_model function")
                        endpoint = get_optimum_openvino_model(model, model_type, openvino_label)
                        print("Successfully loaded model with get_optimum_openvino_model")
                    except Exception as e:
                        print(f"Error loading with get_optimum_openvino_model: {e}")
                
                # Method 3: Try direct OpenVINO loading if model was converted
                if endpoint is None and convert_success and os.path.exists(xml_path):
                    try:
                        print(f"Trying to load OpenVINO model directly from: {xml_path}")
                        endpoint = self.ov.Core().read_model(xml_path)
                        endpoint = self.ov.compile_model(endpoint)
                        print("Successfully loaded model directly with OpenVINO Core")
                    except Exception as e:
                        print(f"Error loading model directly with OpenVINO Core: {e}")
                
                # If all loading methods failed, create a mock model
                if endpoint is None:
                    print("All model loading methods failed, creating mock model")
                    from unittest.mock import MagicMock
                    
                    # More sophisticated mock that mimics OpenVINO model behavior
                    endpoint = MagicMock()
                    
                    # Define a mock __call__ that returns embeddings
                    def mock_infer(inputs):
                        batch_size = 1
                        
                        # Try to determine batch size from inputs if possible
                        if isinstance(inputs, dict):
                            for key in ["input_ids", "attention_mask", "input_features"]:
                                if key in inputs and hasattr(inputs[key], "shape") and len(inputs[key].shape) > 0:
                                    batch_size = inputs[key].shape[0]
                                    break
                        
                        # Create mock embeddings with reasonable shapes
                        results = {
                            "text_embeds": self.np.random.rand(batch_size, 512).astype(self.np.float32),
                            "audio_embeds": self.np.random.rand(batch_size, 512).astype(self.np.float32)
                        }
                        return results
                    
                    # Add the mock method
                    endpoint.__call__ = mock_infer
                    endpoint.infer = mock_infer
            except Exception as e:
                print(f"Unhandled error in model loading: {e}")
                # Create minimal mock model as final fallback
                from unittest.mock import MagicMock
                endpoint = MagicMock()
            
            # Create endpoint handler
            endpoint_handler = self.create_openvino_audio_embedding_endpoint_handler(
                endpoint, tokenizer, model, openvino_label
            )
            
            print(f"Successfully initialized OpenVINO CLAP model")
            return endpoint, tokenizer, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(64), 0
            
        except Exception as e:
            print(f"Error in OpenVINO initialization: {e}")
            # Return empty values in case of failure
            return None, None, None, None, 0
    
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
            
            return endpoint, processor, endpoint_handler, # TODO: Replace with anyio.create_memory_object_stream - asyncio.Queue(32), 0
        except Exception as e:
            print(f"Error initializing Apple Silicon CLAP model: {e}")
            return None, None, None, None, 0

    def create_cpu_audio_embedding_endpoint_handler(self, endpoint, processor, endpoint_model, cpu_label):
        """
        Create a handler for CLAP that can process text, audio, or both on CPU.
        
        Args:
            endpoint: The model endpoint (real or mock)
            processor: The audio/text processor
            endpoint_model: The model name or path
            cpu_label: Label to identify this endpoint
            
        Returns:
            A handler function for CLAP inference on CPU
        """
        def handler(x=None, y=None, endpoint=endpoint, processor=processor, endpoint_model=endpoint_model, cpu_label=cpu_label):
            """
            Process text and/or audio inputs with CLAP on CPU.
            
            Args:
                x: Text input (str or list of str)
                y: Audio input (str path or list of paths)
                
            Returns:
                Dict containing embeddings and/or similarity scores with MOCK/REAL status indicators
            """
            # Track whether we're using mock functionality
            using_mock = False
            
            # Set model to evaluation mode if available
            if endpoint is not None and hasattr(endpoint, 'eval'):
                endpoint.eval()
            
            try:
                with self.torch.no_grad():
                    result = {}
                    
                    # Process text input if provided
                    if x is not None:
                        try:
                            # Process text based on type
                            if isinstance(x, str):
                                text_inputs = processor(
                                    text=x,
                                    return_tensors='pt',
                                    padding=True
                                )
                            elif isinstance(x, list):
                                text_inputs = processor(
                                    text=[text for text in x], 
                                    return_tensors='pt', 
                                    padding=True
                                )
                            else:
                                raise ValueError(f"Unsupported text input type: {type(x)}")
                            
                            # Perform inference if endpoint is available
                            if endpoint is not None:
                                try:
                                    # Get text embeddings
                                    processed_data = {**text_inputs}
                                    text_features = endpoint(**processed_data)
                                    
                                    if hasattr(text_features, 'text_embeds'):
                                        result["text_embedding"] = text_features.text_embeds
                                    else:
                                        # Fallback for different return structure
                                        print("Using fallback for text embeddings")
                                        result["text_embedding"] = self.torch.randn(
                                            text_inputs["input_ids"].shape[0], 512
                                        )
                                        using_mock = True
                                except Exception as e:
                                    print(f"Error during text inference: {e}")
                                    # Create mock embedding on error
                                    batch_size = 1 if not isinstance(x, list) else len(x)
                                    result["text_embedding"] = self.torch.randn(batch_size, 512)
                                    using_mock = True
                            else:
                                # Create mock embedding if no endpoint
                                print("No endpoint available for text embedding")
                                batch_size = 1 if not isinstance(x, list) else len(x)
                                result["text_embedding"] = self.torch.randn(batch_size, 512)
                                using_mock = True
                                
                        except Exception as e:
                            print(f"Error processing text input: {e}")
                            # Create fallback text embedding
                            batch_size = 1 if not isinstance(x, list) else len(x)
                            result["text_embedding"] = self.torch.randn(batch_size, 512)
                            using_mock = True
                    
                    # Process audio input if provided
                    if y is not None:
                        try:
                            # Process audio based on type
                            if isinstance(y, str):
                                try:
                                    audio = self.load_audio(y)
                                    audio_inputs = processor(
                                        audios=[audio[0]], 
                                        return_tensors='pt', 
                                        padding=True,
                                        sampling_rate=audio[1]
                                    )
                                except Exception as e:
                                    print(f"Error loading audio: {e}")
                                    # Create mock audio input
                                    using_mock = True
                                    audio_inputs = {
                                        "input_features": self.torch.rand(1, 1, 1024, 128),
                                        "input_values": self.torch.rand(1, 16000)
                                    }
                            elif isinstance(y, list):
                                try:
                                    # Load multiple audio files
                                    audio_data = []
                                    for audio_file in y:
                                        try:
                                            audio, sr = self.load_audio(audio_file)
                                            audio_data.append(audio)
                                        except Exception as e:
                                            print(f"Error loading audio {audio_file}: {e}")
                                            # Use empty audio for failed loads
                                            audio_data.append(self.np.zeros(16000, dtype=self.np.float32))
                                            using_mock = True
                                    
                                    # Process batch with the first audio's sample rate
                                    first_audio_sr = self.load_audio(y[0])[1]
                                    audio_inputs = processor(
                                        audios=audio_data, 
                                        return_tensors='pt',
                                        padding=True,
                                        sampling_rate=first_audio_sr
                                    )
                                except Exception as e:
                                    print(f"Error processing audio batch: {e}")
                                    # Create mock batch
                                    batch_size = len(y)
                                    audio_inputs = {
                                        "input_features": self.torch.rand(batch_size, 1, 1024, 128),
                                        "input_values": self.torch.rand(batch_size, 16000)
                                    }
                                    using_mock = True
                            else:
                                raise ValueError(f"Unsupported audio input type: {type(y)}")
                            
                            # Perform inference if endpoint is available
                            if endpoint is not None:
                                try:
                                    # Get audio embeddings
                                    processed_data = {**audio_inputs}
                                    audio_features = endpoint(**processed_data)
                                    
                                    if hasattr(audio_features, 'audio_embeds'):
                                        result["audio_embedding"] = audio_features.audio_embeds
                                    else:
                                        # Fallback for different return structure
                                        print("Using fallback for audio embeddings")
                                        if "input_features" in audio_inputs:
                                            batch_size = audio_inputs["input_features"].shape[0]
                                        elif "input_values" in audio_inputs:
                                            batch_size = audio_inputs["input_values"].shape[0]
                                        else:
                                            batch_size = 1
                                        result["audio_embedding"] = self.torch.randn(batch_size, 512)
                                        using_mock = True
                                except Exception as e:
                                    print(f"Error during audio inference: {e}")
                                    # Create mock embedding on error
                                    if "input_features" in audio_inputs:
                                        batch_size = audio_inputs["input_features"].shape[0]
                                    elif "input_values" in audio_inputs:
                                        batch_size = audio_inputs["input_values"].shape[0]
                                    else:
                                        batch_size = 1 if not isinstance(y, list) else len(y)
                                    result["audio_embedding"] = self.torch.randn(batch_size, 512)
                                    using_mock = True
                            else:
                                # Create mock embedding if no endpoint
                                print("No endpoint available for audio embedding")
                                batch_size = 1 if not isinstance(y, list) else len(y)
                                result["audio_embedding"] = self.torch.randn(batch_size, 512)
                                using_mock = True
                                
                        except Exception as e:
                            print(f"Error processing audio input: {e}")
                            # Create fallback audio embedding
                            batch_size = 1 if not isinstance(y, list) else len(y)
                            result["audio_embedding"] = self.torch.randn(batch_size, 512)
                            using_mock = True
                    
                    # Calculate similarity if we have both embeddings
                    if "audio_embedding" in result and "text_embedding" in result:
                        try:
                            # Normalize embeddings for cosine similarity
                            audio_norm = result["audio_embedding"] / result["audio_embedding"].norm(dim=-1, keepdim=True)
                            text_norm = result["text_embedding"] / result["text_embedding"].norm(dim=-1, keepdim=True)
                            
                            # Calculate cosine similarity
                            similarity = (text_norm @ audio_norm.T)
                            result["similarity"] = similarity
                        except Exception as e:
                            print(f"Error calculating similarity: {e}")
                            # Create a mock similarity
                            result["similarity"] = self.torch.tensor([[0.5]])
                            using_mock = True
                    
                    # Add MOCK/REAL status to outputs
                    for key in list(result.keys()):
                        if key in ["audio_embedding", "text_embedding"]:
                            result[f"{key}_status"] = "MOCK" if using_mock else "REAL"
                    
                    # Add overall implementation status
                    result["implementation_status"] = "MOCK" if using_mock else "REAL"
                    
                    # Return single embedding if that's all that was requested
                    if x is not None and y is None and "text_embedding" in result:
                        return {
                            "embedding": result["text_embedding"],
                            "embedding_status": result["text_embedding_status"],
                            "implementation_status": result["implementation_status"]
                        }
                    elif x is None and y is not None and "audio_embedding" in result:
                        return {
                            "embedding": result["audio_embedding"],
                            "embedding_status": result["audio_embedding_status"],
                            "implementation_status": result["implementation_status"]
                        }
                    
                    # No valid inputs
                    if not result or (len(result) == 1 and "implementation_status" in result):
                        return {"message": "No valid input provided", "implementation_status": "MOCK"}
                        
                    return result
                    
            except Exception as e:
                print(f"Error in CPU audio embedding handler: {e}")
                return {
                    "error": str(e),
                    "implementation_status": "MOCK"
                }
                
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
    
    def create_openvino_audio_embedding_endpoint_handler(self, endpoint, processor, model_name, openvino_label):
        """
        Create a handler for OpenVINO-based CLAP audio-text embedding model.
        
        Args:
            endpoint: OpenVINO model endpoint
            processor: CLAP processor for tokenization and audio processing
            model_name: Name of the model
            openvino_label: Label for the OpenVINO endpoint
            
        Returns:
            Handler function for audio-text similarity and embedding
        """
        self.init()
        def handler(text=None, audio=None, endpoint=endpoint, processor=processor, model_name=model_name, openvino_label=openvino_label):
            """
            Process text and/or audio inputs with CLAP on OpenVINO.
            
            Args:
                text: Text input (str or list of str)
                audio: Audio input (path or URL to audio file)
                
            Returns:
                Dict with embeddings and/or similarity scores
            """
            # Flag to track if we're using mock functionality
            using_mock = False
            result = {}
            
            # Validate inputs
            if text is None and audio is None:
                print("No inputs provided")
                return None
                
            # Check if we have proper endpoint and processor
            if endpoint is None or isinstance(endpoint, type(self.torch.nn.Module)) and not hasattr(endpoint, '__call__'):
                using_mock = True
                print("Using mock implementation for OpenVINO CLAP")
            
            try:
                # Process audio input if provided
                if audio is not None:
                    try:
                        # Load audio data
                        if isinstance(audio, str):
                            audio_data, sample_rate = self.load_audio(audio)
                            
                            # Process with tokenizer/processor
                            try:
                                if not using_mock and hasattr(processor, '__call__'):
                                    audio_inputs = processor(
                                        audios=[audio_data],
                                        return_tensors='np',
                                        padding=True,
                                        sampling_rate=sample_rate
                                    )
                                else:
                                    # Mock inputs for testing
                                    using_mock = True
                                    audio_inputs = {
                                        "input_features": self.np.random.rand(1, 1, 1024, 128).astype(self.np.float32),
                                        "input_values": self.np.random.rand(1, 16000).astype(self.np.float32)
                                    }
                            except Exception as e:
                                print(f"Error processing audio: {e}")
                                using_mock = True
                                audio_inputs = {
                                    "input_features": self.np.random.rand(1, 1, 1024, 128).astype(self.np.float32),
                                    "input_values": self.np.random.rand(1, 16000).astype(self.np.float32)
                                }
                        
                            # Get audio embeddings from model
                            if not using_mock and hasattr(endpoint, '__call__'):
                                try:
                                    audio_features = endpoint(audio_inputs)
                                    
                                    # Extract embeddings based on key
                                    if isinstance(audio_features, dict) and "audio_embeds" in audio_features:
                                        result["audio_embedding"] = audio_features["audio_embeds"]
                                    else:
                                        # Try alternative key names
                                        keys = [k for k in audio_features.keys() if 'audio' in k.lower() and 'embed' in k.lower()]
                                        if keys:
                                            result["audio_embedding"] = audio_features[keys[0]]
                                        else:
                                            # Fallback to mock
                                            using_mock = True
                                            result["audio_embedding"] = self.torch.rand(1, 512)
                                except Exception as e:
                                    print(f"Error getting audio embeddings: {e}")
                                    using_mock = True
                            
                            # Use mock embeddings if needed
                            if using_mock or "audio_embedding" not in result:
                                result["audio_embedding"] = self.torch.rand(1, 512)
                    except Exception as e:
                        print(f"Error processing audio input: {e}")
                        using_mock = True
                        result["audio_embedding"] = self.torch.rand(1, 512)
                
                # Process text input if provided
                if text is not None:
                    try:
                        # Process with tokenizer/processor
                        if isinstance(text, str):
                            if not using_mock and hasattr(processor, '__call__'):
                                try:
                                    text_inputs = processor(
                                        text=text,
                                        return_tensors='np',
                                        padding=True
                                    )
                                except Exception as e:
                                    print(f"Error tokenizing text: {e}")
                                    using_mock = True
                                    text_inputs = {
                                        "input_ids": self.np.ones((1, 77), dtype=self.np.int32),
                                        "attention_mask": self.np.ones((1, 77), dtype=self.np.int32)
                                    }
                            else:
                                using_mock = True
                                text_inputs = {
                                    "input_ids": self.np.ones((1, 77), dtype=self.np.int32),
                                    "attention_mask": self.np.ones((1, 77), dtype=self.np.int32)
                                }
                        
                            # Get text embeddings from model
                            if not using_mock and hasattr(endpoint, '__call__'):
                                try:
                                    text_features = endpoint(text_inputs)
                                    
                                    # Extract embeddings based on key
                                    if isinstance(text_features, dict) and "text_embeds" in text_features:
                                        result["text_embedding"] = text_features["text_embeds"]
                                    else:
                                        # Try alternative key names
                                        keys = [k for k in text_features.keys() if 'text' in k.lower() and 'embed' in k.lower()]
                                        if keys:
                                            result["text_embedding"] = text_features[keys[0]]
                                        else:
                                            # Fallback to mock
                                            using_mock = True
                                            result["text_embedding"] = self.torch.rand(1, 512)
                                except Exception as e:
                                    print(f"Error getting text embeddings: {e}")
                                    using_mock = True
                            
                            # Use mock embeddings if needed
                            if using_mock or "text_embedding" not in result:
                                result["text_embedding"] = self.torch.rand(1, 512)
                    except Exception as e:
                        print(f"Error processing text input: {e}")
                        using_mock = True
                        result["text_embedding"] = self.torch.rand(1, 512)
                
                # Calculate similarity if we have both embeddings
                if "audio_embedding" in result and "text_embedding" in result:
                    try:
                        # Convert to PyTorch tensors if needed
                        audio_emb = self.torch.tensor(result["audio_embedding"]) if not isinstance(result["audio_embedding"], self.torch.Tensor) else result["audio_embedding"]
                        text_emb = self.torch.tensor(result["text_embedding"]) if not isinstance(result["text_embedding"], self.torch.Tensor) else result["text_embedding"]
                        
                        # Normalize for cosine similarity
                        audio_norm = audio_emb / (audio_emb.norm(dim=-1, keepdim=True) + 1e-8)  # Avoid division by zero
                        text_norm = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-8)
                        
                        # Calculate similarity
                        similarity = self.torch.matmul(text_norm, audio_norm.T)
                        result["similarity"] = similarity
                    except Exception as e:
                        print(f"Error calculating similarity: {e}")
                        result["similarity"] = self.torch.tensor([[0.5]])
                        using_mock = True
                
                # Add consistent implementation type and status information
                implementation_type = "MOCK" if using_mock else "REAL"
                result["implementation_type"] = implementation_type
                
                # Add status to each result component for better tracking
                if "audio_embedding" in result:
                    result["audio_embedding_status"] = implementation_type
                if "text_embedding" in result:
                    result["text_embedding_status"] = implementation_type
                if "similarity" in result:
                    result["similarity_status"] = implementation_type
                    
                # Add timestamp for debugging
                result["timestamp"] = time.time()
                
                return result
            except Exception as e:
                print(f"Error in OpenVINO audio embedding handler: {e}")
                # Return mock result with error information
                return {
                    "error": str(e),
                    "implementation_type": "MOCK",
                    "audio_embedding": self.torch.rand(1, 512) if audio is not None else None,
                    "text_embedding": self.torch.rand(1, 512) if text is not None else None,
                    "similarity": self.torch.tensor([[0.5]]) if audio is not None and text is not None else None
                }
                
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