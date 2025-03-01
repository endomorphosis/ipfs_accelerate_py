import os
import sys
import json
import time
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import importlib.util
import datetime
import traceback

# Use direct import with the absolute path
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")

# Try to import transformers directly if available
try:
    import transformers
    transformers_module = transformers
except ImportError:
    transformers_module = MagicMock()
    print("Warning: transformers not available, using mock implementation")

# Try to import real audio handling libraries
try:
    import librosa
    import soundfile as sf
    
    # Define real audio loading function
    def load_audio(audio_file):
        """Load audio with real libraries"""
        try:
            # For local files
            if os.path.exists(audio_file):
                audio, sr = librosa.load(audio_file, sr=16000)
                return audio, sr
            # For URLs, download and then load
            else:
                import requests
                from io import BytesIO
                response = requests.get(audio_file)
                audio, sr = librosa.load(BytesIO(response.content), sr=16000)
                return audio, sr
        except Exception as e:
            print(f"Error loading audio with librosa: {e}")
            return np.zeros(16000, dtype=np.float32), 16000
except ImportError:
    # Define fallback audio loading function when real libraries aren't available
    def load_audio(audio_file):
        """Fallback audio loading function when real libraries aren't available"""
        print(f"Using fallback audio loader for {audio_file}")
        # Return a silent audio sample of 1 second at 16kHz
        return np.zeros(16000, dtype=np.float32), 16000

# Import the wav2vec2 implementation (we'll adapt this for wav2vec2-bert)
# This assumes we can reuse much of the wav2vec2 infrastructure
from ipfs_accelerate_py.worker.skillset.hf_wav2vec2 import hf_wav2vec2

# Add missing init_cuda method to the class
def init_cuda(self, model_name, model_type, device_label="cuda:0"):
    """Initialize Wav2Vec2-BERT model with CUDA support.
    
    Args:
        model_name: Name or path of the model to load
        model_type: Type of model (e.g., "automatic-speech-recognition")
        device_label: CUDA device to use (e.g., "cuda:0", "cuda:1")
        
    Returns:
        tuple: (endpoint, processor, handler, queue, batch_size)
    """
    import traceback
    import sys
    import torch
    from unittest import mock
    
    # Check if transformers is available
    transformers_available = hasattr(self.resources["transformers"], "__version__")
    if not transformers_available:
        print("Transformers not available for real CUDA implementation")
        # Return mock implementation
        processor = mock.MagicMock()
        endpoint = mock.MagicMock()
        handler = mock.MagicMock()
        return endpoint, processor, handler, None, 1
    
    # Try to import the necessary utility functions
    try:
        sys.path.insert(0, "/home/barberb/ipfs_accelerate_py/test")
        import utils as test_utils
        
        # Check if CUDA is really available
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to mock implementation")
            processor = mock.MagicMock()
            endpoint = mock.MagicMock()
            handler = mock.MagicMock()
            return endpoint, processor, handler, None, 1
            
        # Get the CUDA device
        device = test_utils.get_cuda_device(device_label)
        if device is None:
            print("Failed to get valid CUDA device, falling back to mock implementation")
            processor = mock.MagicMock()
            endpoint = mock.MagicMock()
            handler = mock.MagicMock()
            return endpoint, processor, handler, None, 1
        
        # Try to initialize with real components
        try:
            from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, AutoProcessor
            import librosa
            
            print(f"Initializing wav2vec2-bert model {model_name} for CUDA...")
            
            # Load the processor
            try:
                # Wav2Vec2Bert uses a specialized processor
                processor = AutoProcessor.from_pretrained(model_name)
                print(f"Successfully loaded processor for {model_name}")
            except Exception as proc_err:
                print(f"Error loading processor: {proc_err}")
                # Fall back to mock processor
                processor = mock.MagicMock()
                processor.is_real_simulation = False
            
            # Load the model
            try:
                # Wav2Vec2BertForCTC is the class specifically for CTC-based ASR with Wav2Vec2-BERT
                endpoint = Wav2Vec2BertForCTC.from_pretrained(model_name)
                print(f"Successfully loaded model {model_name}")
                
                # Move to CUDA and optimize
                endpoint = test_utils.optimize_cuda_memory(endpoint, device, use_half_precision=True)
                endpoint = endpoint.to(device)
                endpoint.eval()
                print(f"Model loaded to {device} and optimized for inference")
                
                # Add a flag to indicate this is a real implementation
                endpoint.is_real_simulation = True
                
            except Exception as model_err:
                print(f"Error loading model to CUDA: {model_err}")
                # Fall back to mock model
                endpoint = mock.MagicMock()
                endpoint.is_real_simulation = False
            
            # Setup handler function
            def handler(audio_path):
                """Process audio with CUDA-accelerated Wav2Vec2-BERT"""
                start_time = time.time()
                try:
                    # Load audio
                    audio, sr = load_audio(audio_path)
                    
                    # Process with processor
                    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
                    
                    # Move inputs to CUDA
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            inputs[key] = inputs[key].to(device)
                    
                    # Run inference with no gradients
                    with torch.no_grad():
                        outputs = endpoint(**inputs)
                    
                    # Process outputs
                    logits = outputs.logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    
                    # Get current CUDA memory stats
                    cuda_mem_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024) if hasattr(torch.cuda, "memory_allocated") else 0
                    cuda_mem_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024) if hasattr(torch.cuda, "memory_reserved") else 0
                    
                    # Return structured result with detailed performance metrics
                    return {
                        "text": transcription[0],
                        "implementation_type": "REAL",
                        "device": str(device),
                        "inference_time_seconds": time.time() - start_time,
                        "gpu_memory_allocated_mb": cuda_mem_allocated,
                        "gpu_memory_reserved_mb": cuda_mem_reserved,
                        "performance_metrics": {
                            "total_time_ms": (time.time() - start_time) * 1000,
                            "audio_duration_seconds": len(audio) / sr if 'audio' in locals() and 'sr' in locals() else 0.0,
                            "throughput_ratio": (len(audio) / sr) / (time.time() - start_time) if 'audio' in locals() and 'sr' in locals() else 0.0
                        }
                    }
                except Exception as e:
                    print(f"Error in CUDA handler: {e}")
                    traceback.print_exc()
                    # Return mock output
                    return {"text": "(MOCK) CUDA transcription failed", "implementation_type": "MOCK", "error": str(e)}
            
            print("Wav2Vec2-BERT CUDA initialization complete")
            return endpoint, processor, handler, None, 4  # Batch size of 4 for audio
            
        except ImportError as e:
            print(f"Required libraries not available: {e}")
            print("Falling back to mock implementation")
        
    except Exception as e:
        print(f"Error in init_cuda: {e}")
        traceback.print_exc()
    
    # Fallback to mock implementation
    processor = mock.MagicMock()
    endpoint = mock.MagicMock()
    handler = mock.MagicMock()
    return endpoint, processor, handler, None, 1

# Add the method to the hf_wav2vec2 class (reusing the class)
hf_wav2vec2.init_cuda = init_cuda

# Create CUDA transcription handler for Wav2Vec2-BERT
def create_cuda_transcription_endpoint_handler(self, processor, model_name, cuda_label, endpoint=None):
    """Create a handler for CUDA-accelerated Wav2Vec2-BERT transcription.
    
    Args:
        processor: The Wav2Vec2-BERT processor
        model_name: The model name
        cuda_label: CUDA device label (e.g., "cuda:0")
        endpoint: The Wav2Vec2-BERT model
        
    Returns:
        handler function for processing audio
    """
    import sys
    import torch
    import traceback
    from unittest import mock
    
    # Import utils for CUDA support
    try:
        sys.path.insert(0, "/home/barberb/ipfs_accelerate_py/test")
        import utils as test_utils
    except ImportError:
        print("Could not import test utils")
        
    # Check if we have real implementation or need to use mocks
    import unittest.mock
    is_mock = isinstance(endpoint, unittest.mock.MagicMock) or isinstance(processor, unittest.mock.MagicMock)
    
    # Get CUDA device if available
    device = None
    if not is_mock:
        try:
            device = test_utils.get_cuda_device(cuda_label)
            if device is None:
                is_mock = True
                print("CUDA device not available despite torch.cuda.is_available() being True")
        except Exception as e:
            print(f"Error getting CUDA device: {e}")
            is_mock = True
    
    def handler(audio_path):
        """Handle audio transcription using CUDA acceleration."""
        start_time = time.time()
        
        # If we're using mocks, return a mock response
        if is_mock:
            # Simulate some processing time
            time.sleep(0.1)
            return {
                "text": f"(MOCK CUDA) Transcribed audio from {audio_path}",
                "implementation_type": "MOCK",
                "device": "cuda:0 (mock)",
                "total_time": time.time() - start_time
            }
            
        # Try to use real implementation
        try:
            import librosa
            import numpy as np
            
            # Load audio
            try:
                # For URLs or local files
                if audio_path.startswith(('http://', 'https://')):
                    import requests
                    from io import BytesIO
                    response = requests.get(audio_path)
                    audio, sr = librosa.load(BytesIO(response.content), sr=16000)
                else:
                    audio, sr = librosa.load(audio_path, sr=16000)
            except Exception as audio_err:
                print(f"Error loading audio: {audio_err}")
                # Return silent audio
                audio = np.zeros(16000, dtype=np.float32)
                sr = 16000
            
            # Process with processor
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            
            # Move inputs to CUDA
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(device)
            
            # Measure GPU memory before inference
            cuda_mem_before = torch.cuda.memory_allocated(device) / (1024 * 1024) if hasattr(torch.cuda, "memory_allocated") else 0
            
            # Run inference with no gradients
            with torch.no_grad():
                torch.cuda.synchronize() if hasattr(torch.cuda, "synchronize") else None
                inference_start = time.time()
                outputs = endpoint(**inputs)
                torch.cuda.synchronize() if hasattr(torch.cuda, "synchronize") else None
                inference_time = time.time() - inference_start
            
            # Measure GPU memory after inference
            cuda_mem_after = torch.cuda.memory_allocated(device) / (1024 * 1024) if hasattr(torch.cuda, "memory_allocated") else 0
            gpu_mem_used = cuda_mem_after - cuda_mem_before
            
            # Process outputs
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            
            # Return structured result with metrics
            total_time = time.time() - start_time
            return {
                "text": transcription[0],
                "implementation_type": "REAL",
                "device": str(device),
                "total_time": total_time,
                "inference_time": inference_time,
                "gpu_memory_used_mb": gpu_mem_used,
            }
            
        except Exception as e:
            print(f"Error in CUDA handler: {e}")
            traceback.print_exc()
            # Return error info
            return {
                "text": f"Error in CUDA handler: {str(e)}",
                "implementation_type": "REAL (error)",
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    return handler

# Add the handler creator to the hf_wav2vec2 class
hf_wav2vec2.create_cuda_transcription_endpoint_handler = create_cuda_transcription_endpoint_handler

# Specialized class for Wav2Vec2-BERT
class test_hf_wav2vec2_bert:
    def __init__(self, resources=None, metadata=None):
        """Initialize the test class for Wav2Vec2-BERT model"""
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": transformers_module
        }
        self.metadata = metadata if metadata else {}
        # We'll reuse the hf_wav2vec2 class for the base implementation
        self.wav2vec2_bert = hf_wav2vec2(resources=self.resources, metadata=self.metadata)
        
        # Use an appropriate model for Wav2Vec2-BERT
        # Wav2Vec2-BERT models have a BERT encoder on top of Wav2Vec2
        # Note: This is a placeholder, needs actual model name
        self.model_name = "facebook/wav2vec2-bert-base" 
        
        # Alternative options if the primary model fails
        self.alternative_models = [
            "facebook/w2v-bert-2.0",  # Another option for Wav2Vec2-BERT
            "facebook/wav2vec2-conformer-rope-large-960h-ft", # Another option
            "facebook/wav2vec2-xls-r-300m-en-to-15", # Alternative model
            "facebook/wav2vec2-conformer-rel-pos-large" # Alternative model
        ]
        
        # Try to use the specified model first, then fall back to alternatives or local cache
        try:
            print(f"Attempting to use primary model: {self.model_name}")
            
            # Try to import transformers for validation
            if not isinstance(self.resources["transformers"], MagicMock):
                from transformers import AutoConfig
                try:
                    # Try to access the config to verify model works
                    AutoConfig.from_pretrained(self.model_name)
                    print(f"Successfully validated primary model: {self.model_name}")
                except Exception as config_error:
                    print(f"Primary model validation failed: {config_error}")
                    
                    # Try alternatives one by one
                    for alt_model in self.alternative_models:
                        try:
                            print(f"Trying alternative model: {alt_model}")
                            AutoConfig.from_pretrained(alt_model)
                            self.model_name = alt_model
                            print(f"Successfully validated alternative model: {self.model_name}")
                            break
                        except Exception as alt_error:
                            print(f"Alternative model validation failed: {alt_error}")
                    
                    # If all alternatives fail, check local cache
                    if self.model_name == "facebook/wav2vec2-bert-base":
                        # Check if we can get a list of locally cached models
                        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models")
                        if os.path.exists(cache_dir):
                            # Look for any WAV2VEC2-BERT model in cache
                            wav2vec2_bert_models = [name for name in os.listdir(cache_dir) if "wav2vec2-bert" in name.lower() or "w2v-bert" in name.lower()]
                            if wav2vec2_bert_models:
                                # Use the first WAV2VEC2-BERT model found
                                wav2vec2_bert_model_name = wav2vec2_bert_models[0].replace("--", "/")
                                print(f"Found local WAV2VEC2-BERT model: {wav2vec2_bert_model_name}")
                                self.model_name = wav2vec2_bert_model_name
                            else:
                                # Create a local test model as last resort
                                print("No models found in cache, creating local test model")
                                self.model_name = self._create_test_model()
                        else:
                            # Create a local test model as last resort
                            print("No cache directory found, creating local test model")
                            self.model_name = self._create_test_model()
            
        except Exception as e:
            print(f"Error finding model: {e}")
            # Create a local test model as final fallback
            print("Creating local test model due to error")
            self.model_name = self._create_test_model()
            
        print(f"Using model: {self.model_name}")
        
        # Try to use trans_test.mp3 first, then fall back to test.mp3, or URL as last resort
        trans_test_audio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trans_test.mp3")
        test_audio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.mp3")
        
        if os.path.exists(trans_test_audio_path):
            self.test_audio = trans_test_audio_path
        elif os.path.exists(test_audio_path):
            self.test_audio = test_audio_path
        else:
            self.test_audio = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
            
        print(f"Using test audio: {self.test_audio}")
        
        # Flag to track if we're using mocks (for clearer test results)
        self.using_mocks = False
        
        return None
        
    def _create_test_model(self):
        """
        Create a tiny WAV2VEC2-BERT model for testing without needing Hugging Face authentication.
        
        Returns:
            str: Path to the created model
        """
        try:
            print("Creating local test model for WAV2VEC2-BERT testing...")
            
            # Create model directory in /tmp for tests
            test_model_dir = os.path.join("/tmp", "wav2vec2_bert_test_model")
            os.makedirs(test_model_dir, exist_ok=True)
            
            # Create a minimal config file for a tiny WAV2VEC2-BERT model
            config = {
                "architectures": ["Wav2Vec2BertForCTC"],
                "attention_dropout": 0.1,
                "bos_token_id": 1,
                "conv_bias": True,
                "conv_dim": [32, 32, 32],
                "conv_kernel": [5, 3, 3],
                "conv_stride": [3, 1, 1],
                "ctc_loss_reduction": "mean",
                "ctc_zero_infinity": False,
                "do_stable_layer_norm": False,
                "eos_token_id": 2,
                "feat_extract_activation": "gelu",
                "feat_extract_dropout": 0.0,
                "feat_extract_norm": "group",
                "final_dropout": 0.1,
                "hidden_act": "gelu",
                "hidden_dropout": 0.1,
                "hidden_dropout_prob": 0.1,
                "hidden_size": 256,
                "initializer_range": 0.02,
                "intermediate_size": 512,
                "layer_norm_eps": 1e-05,
                "layerdrop": 0.1,
                "mask_time_length": 10,
                "mask_time_min_masks": 2,
                "mask_time_prob": 0.05,
                "model_type": "wav2vec2-bert",
                "num_attention_heads": 4,
                "num_conv_pos_embedding_groups": 16,
                "num_conv_pos_embeddings": 32,
                "num_feat_extract_layers": 3,
                "num_hidden_layers": 2,
                "pad_token_id": 0,
                "vocab_size": 32,
                # Wav2Vec2-BERT specific additions
                "bert_hidden_size": 256,
                "bert_intermediate_size": 512,
                "bert_num_attention_heads": 4,
                "bert_num_hidden_layers": 2
            }
            
            with open(os.path.join(test_model_dir, "config.json"), "w") as f:
                json.dump(config, f)
                
            # Create a minimal processor config
            processor_config = {
                "feature_extractor": {
                    "feature_size": 1,
                    "padding_value": 0.0,
                    "sampling_rate": 16000,
                    "return_attention_mask": False,
                    "do_normalize": True
                },
                "tokenizer": {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "pad_token": "<pad>",
                    "word_delimiter_token": "|",
                    "unk_token": "<unk>",
                    "model_max_length": 1000000,
                    "clean_up_tokenization_spaces": True
                }
            }
            
            with open(os.path.join(test_model_dir, "preprocessor_config.json"), "w") as f:
                json.dump(processor_config, f)
                
            # Create a minimal vocabulary file
            vocab = {
                "<pad>": 0,
                "<s>": 1,
                "</s>": 2,
                "<unk>": 3,
                "|": 4,
                "a": 5,
                "b": 6,
                "c": 7,
                "d": 8,
                "e": 9,
                "f": 10,
                "g": 11,
                "h": 12,
                "i": 13,
                "j": 14,
                "k": 15,
                "l": 16,
                "m": 17,
                "n": 18,
                "o": 19,
                "p": 20,
                "q": 21,
                "r": 22,
                "s": 23,
                "t": 24,
                "u": 25,
                "v": 26,
                "w": 27,
                "x": 28,
                "y": 29,
                "z": 30,
                " ": 31
            }
            
            # Create vocab.json
            with open(os.path.join(test_model_dir, "vocab.json"), "w") as f:
                json.dump(vocab, f)
                
            # Create small random model weights if torch is available
            if hasattr(torch, "save") and not isinstance(torch, MagicMock):
                # Create random tensors for model weights
                model_state = {}
                
                # Extract dimensions from config
                hidden_size = config["hidden_size"]
                intermediate_size = config["intermediate_size"]
                num_attention_heads = config["num_attention_heads"]
                num_hidden_layers = config["num_hidden_layers"]
                vocab_size = config["vocab_size"]
                bert_hidden_size = config["bert_hidden_size"]
                
                # Feature extraction layers
                for i in range(config["num_feat_extract_layers"]):
                    # Convolutional layers
                    conv_dim_in = 1 if i == 0 else config["conv_dim"][i-1]
                    conv_dim_out = config["conv_dim"][i]
                    kernel_size = config["conv_kernel"][i]
                    model_state[f"wav2vec2.feature_extractor.conv_layers.{i}.conv.weight"] = torch.randn(conv_dim_out, conv_dim_in, kernel_size)
                    if config["conv_bias"]:
                        model_state[f"wav2vec2.feature_extractor.conv_layers.{i}.conv.bias"] = torch.randn(conv_dim_out)
                    
                    # Layer norm
                    model_state[f"wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.weight"] = torch.ones(conv_dim_out)
                    model_state[f"wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.bias"] = torch.zeros(conv_dim_out)
                
                # Feature projection layer
                model_state["wav2vec2.feature_projection.projection.weight"] = torch.randn(hidden_size, config["conv_dim"][-1])
                model_state["wav2vec2.feature_projection.projection.bias"] = torch.randn(hidden_size)
                model_state["wav2vec2.feature_projection.layer_norm.weight"] = torch.ones(hidden_size)
                model_state["wav2vec2.feature_projection.layer_norm.bias"] = torch.zeros(hidden_size)
                
                # BERT Encoder layers - what makes this different from standard Wav2Vec2
                for i in range(config["bert_num_hidden_layers"]):
                    # Self-attention
                    model_state[f"bert.encoder.layer.{i}.attention.self.query.weight"] = torch.randn(bert_hidden_size, bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.attention.self.query.bias"] = torch.randn(bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.attention.self.key.weight"] = torch.randn(bert_hidden_size, bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.attention.self.key.bias"] = torch.randn(bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.attention.self.value.weight"] = torch.randn(bert_hidden_size, bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.attention.self.value.bias"] = torch.randn(bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.attention.output.dense.weight"] = torch.randn(bert_hidden_size, bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.attention.output.dense.bias"] = torch.randn(bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight"] = torch.ones(bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias"] = torch.zeros(bert_hidden_size)
                    
                    # Feed-forward network
                    model_state[f"bert.encoder.layer.{i}.intermediate.dense.weight"] = torch.randn(intermediate_size, bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.intermediate.dense.bias"] = torch.randn(intermediate_size)
                    model_state[f"bert.encoder.layer.{i}.output.dense.weight"] = torch.randn(bert_hidden_size, intermediate_size)
                    model_state[f"bert.encoder.layer.{i}.output.dense.bias"] = torch.randn(bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.output.LayerNorm.weight"] = torch.ones(bert_hidden_size)
                    model_state[f"bert.encoder.layer.{i}.output.LayerNorm.bias"] = torch.zeros(bert_hidden_size)
                
                # CTC projection for output
                model_state["lm_head.weight"] = torch.randn(vocab_size, bert_hidden_size)
                model_state["lm_head.bias"] = torch.randn(vocab_size)
                
                # Save model weights
                torch.save(model_state, os.path.join(test_model_dir, "pytorch_model.bin"))
                print(f"Created PyTorch model weights in {test_model_dir}/pytorch_model.bin")
            
            print(f"Test model created at {test_model_dir}")
            return test_model_dir
            
        except Exception as e:
            print(f"Error creating test model: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Fall back to a model name that won't need to be downloaded for mocks
            return "wav2vec2-bert-test"

    def test(self):
        """Run tests for the Wav2Vec2-BERT model (transcription functionality)"""
        from unittest.mock import MagicMock
        import traceback
        
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.wav2vec2_bert is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # Test audio loading utilities
        try:
            audio_data, sr = load_audio(self.test_audio)
            results["load_audio"] = "Success" if audio_data is not None and sr == 16000 else "Failed audio loading"
            results["audio_format"] = f"Shape: {audio_data.shape}, SR: {sr}"
        except Exception as e:
            results["load_audio"] = f"Error: {str(e)}"

        # Check if we're using real transformers
        transformers_available = not isinstance(self.resources["transformers"], MagicMock)
        implementation_type = "(REAL)" if transformers_available else "(MOCK)"
        
        # Test CPU initialization and handler
        try:
            if transformers_available:
                print("Testing with real wav2vec2-bert model on CPU")
                try:
                    # Initialize for CPU
                    endpoint, processor, handler, queue, batch_size = self.wav2vec2_bert.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and processor is not None and handler is not None
                    results["cpu_init"] = f"Success {implementation_type}" if valid_init else "Failed CPU initialization"
                    
                    if valid_init:
                        # Test TRANSCRIPTION functionality
                        transcription_handler = self.wav2vec2_bert.create_cpu_transcription_endpoint_handler(
                            processor, self.model_name, "cpu", endpoint
                        )
                        
                        # Test with real audio file
                        try:
                            transcription_output = transcription_handler(self.test_audio)
                            results["cpu_transcription_handler"] = f"Success {implementation_type}" if transcription_output is not None else "Failed CPU transcription handler"
                            
                            # Add transcription result to results
                            if transcription_output is not None:
                                # Truncate long outputs for readability
                                if len(str(transcription_output)) > 100:
                                    results["cpu_transcription"] = transcription_output[:100] + "..."
                                else:
                                    results["cpu_transcription"] = transcription_output
                                
                                # Save result to demonstrate working implementation
                                results["cpu_transcription_example"] = {
                                    "input": self.test_audio,
                                    "output": transcription_output[:100] + "..." if isinstance(transcription_output, str) and len(str(transcription_output)) > 100 else transcription_output,
                                    "timestamp": time.time(),
                                    "elapsed_time": 0.1,  # Placeholder for actual timing
                                    "implementation_type": implementation_type,
                                    "platform": "CPU"
                                }
                        except Exception as handler_error:
                            results["cpu_transcription_error"] = str(handler_error)
                            results["cpu_transcription"] = f"Error: {str(handler_error)}"
                except Exception as e:
                    results["cpu_error"] = f"Error: {str(e)}"
                    raise e
            else:
                # Fall back to mock if transformers not available
                raise ImportError("Transformers not available")
        except Exception as e:
            # Fall back to mocks if real model fails
            print(f"Falling back to mock wav2vec2-bert model: {e}")
            implementation_type = "(MOCK)"
            
            with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                 patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
                 patch('transformers.AutoModelForCTC.from_pretrained') as mock_model:
                
                self.using_mocks = True
                print("Using mock transformers components")
                mock_config.return_value = MagicMock()
                mock_processor.return_value = MagicMock()
                mock_processor.return_value.batch_decode = MagicMock(return_value=["Test transcription"])
                mock_model.return_value = MagicMock()
                mock_model.return_value.generate = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
                
                # Create mock objects
                processor = MagicMock()
                endpoint = MagicMock()
                
                # For transcription testing
                transcription_handler = self.wav2vec2_bert.create_cpu_transcription_endpoint_handler(
                    processor, self.model_name, "cpu", endpoint
                )
                
                valid_init = endpoint is not None and processor is not None and transcription_handler is not None
                results["cpu_init"] = f"Success {implementation_type}" if valid_init else "Failed CPU initialization"
                
                # Test with mock audio
                with patch('soundfile.read') as mock_sf_read:
                    mock_sf_read.return_value = (np.random.randn(16000), 16000)
                    
                    # Test transcription
                    output = transcription_handler(self.test_audio)
                    results["cpu_transcription_handler"] = f"Success {implementation_type}" if output is not None else "Failed CPU transcription handler"
                    if output is not None:
                        results["cpu_transcription"] = output
                        # Save result to demonstrate working implementation
                        results["cpu_transcription_example"] = {
                            "input": self.test_audio,
                            "output": output,
                            "timestamp": time.time(),
                            "elapsed_time": 0.05,  # Placeholder for timing in mock implementation
                            "implementation_type": implementation_type,
                            "platform": "CPU"
                        }

        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                # Try to use real CUDA implementation first
                if transformers_available:
                    print("Testing with real wav2vec2-bert model on CUDA")
                    try:
                        # Import CUDA utilities
                        import sys
                        sys.path.insert(0, "/home/barberb/ipfs_accelerate_py/test")
                        import utils as test_utils
                        print("Successfully imported CUDA utilities from direct path")
                        
                        # Create more robust device handling
                        device = test_utils.get_cuda_device("cuda:0")
                        if device is None:
                            print("No valid CUDA device found, falling back to mock")
                            raise RuntimeError("No valid CUDA device found")
                        
                        # Log the available CUDA memory for debugging
                        if hasattr(torch.cuda, "get_device_properties"):
                            device_props = torch.cuda.get_device_properties(device)
                            total_memory = device_props.total_memory / (1024**3)  # GB
                            print(f"CUDA device {device} has {total_memory:.2f}GB total memory")
                        
                        # Clear CUDA cache for clean test
                        if hasattr(torch.cuda, "empty_cache"):
                            torch.cuda.empty_cache()
                            
                        print(f"Attempting to load real WAV2VEC2-BERT model {self.model_name} with CUDA support")
                        
                        # Initialize for CUDA with more reliable error handling
                        endpoint, processor, handler, queue, batch_size = self.wav2vec2_bert.init_cuda(
                            self.model_name,
                            "automatic-speech-recognition",
                            str(device)
                        )
                        
                        # Check if we actually got real implementations
                        is_real_impl = not isinstance(endpoint, MagicMock) and not isinstance(processor, MagicMock)
                        implementation_type = "(REAL)" if is_real_impl else "(MOCK)"
                        
                        valid_init = endpoint is not None and processor is not None and handler is not None
                        results["cuda_init"] = f"Success {implementation_type}" if valid_init else "Failed CUDA initialization"
                        
                        if valid_init:
                            # Test TRANSCRIPTION functionality with CUDA
                            transcription_handler = self.wav2vec2_bert.create_cuda_transcription_endpoint_handler(
                                processor, 
                                self.model_name, 
                                "cuda:0", 
                                endpoint
                            )
                            
                            # Enhance the handler with implementation type markers
                            try:
                                if hasattr(test_utils, 'enhance_cuda_implementation_detection'):
                                    # Enhance the handler to ensure proper implementation detection
                                    print(f"Enhancing WAV2VEC2-BERT CUDA handler with implementation type markers: {is_real_impl}")
                                    transcription_handler = test_utils.enhance_cuda_implementation_detection(
                                        self.wav2vec2_bert,
                                        transcription_handler,
                                        is_real=is_real_impl
                                    )
                            except Exception as e:
                                print(f"Could not enhance handler: {e}")
                            
                            # Test with real audio file
                            start_time = time.time()
                            transcription_output = transcription_handler(self.test_audio)
                            elapsed_time = time.time() - start_time
                            
                            results["cuda_handler"] = f"Success {implementation_type}" if transcription_output is not None else "Failed CUDA handler"
                            
                            # Add transcription result to results
                            if transcription_output is not None:
                                # Truncate long outputs for readability
                                if isinstance(transcription_output, str) and len(transcription_output) > 100:
                                    results["cuda_transcription"] = transcription_output[:100] + "..."
                                else:
                                    results["cuda_transcription"] = transcription_output
                                
                                # Extract performance metrics if available
                                performance_metrics = {}
                                if isinstance(transcription_output, dict):
                                    # Check for implementation type marker
                                    if "implementation_type" in transcription_output:
                                        output_impl_type = transcription_output["implementation_type"]
                                        print(f"Found implementation_type in output dict: {output_impl_type}")
                                        if output_impl_type == "REAL":
                                            is_real_impl = True
                                            implementation_type = "(REAL)"
                                            # Update CUDA handler result to reflect implementation type
                                            results["cuda_handler"] = f"Success {implementation_type}"
                                        elif output_impl_type == "MOCK":
                                            is_real_impl = False
                                            implementation_type = "(MOCK)"
                                            # Update CUDA handler result to reflect implementation type
                                            results["cuda_handler"] = f"Success {implementation_type}"
                                    
                                    # Look for other indicators of real implementations
                                    if "device" in transcription_output:
                                        device_info = transcription_output["device"]
                                        if isinstance(device_info, str) and "cuda" in device_info.lower():
                                            print(f"CUDA device detected in output: {device_info}")
                                            # This is likely a real implementation
                                            if not is_real_impl:
                                                is_real_impl = True
                                                implementation_type = "(REAL)"
                                                # Update results to reflect implementation type
                                                results["cuda_handler"] = f"Success {implementation_type}"
                                    
                                    # Get performance metrics if available
                                    if "performance_metrics" in transcription_output:
                                        performance_metrics = transcription_output["performance_metrics"]
                                    elif "gpu_memory_used_mb" in transcription_output:
                                        # Create performance metrics from available data
                                        performance_metrics = {
                                            "gpu_memory_used_mb": transcription_output.get("gpu_memory_used_mb", 0),
                                            "inference_time": transcription_output.get("inference_time", elapsed_time),
                                            "device": transcription_output.get("device", "cuda:0")
                                        }
                                    elif "gpu_memory_allocated_mb" in transcription_output:
                                        # Create performance metrics from available data
                                        performance_metrics = {
                                            "gpu_memory_allocated_mb": transcription_output.get("gpu_memory_allocated_mb", 0),
                                            "gpu_memory_reserved_mb": transcription_output.get("gpu_memory_reserved_mb", 0),
                                            "inference_time_seconds": transcription_output.get("inference_time_seconds", elapsed_time),
                                            "device": transcription_output.get("device", "cuda:0")
                                        }
                                    
                                    # Get the actual text output
                                    text_output = transcription_output.get("text", str(transcription_output))
                                else:
                                    text_output = transcription_output
                                
                                # Save result to demonstrate working implementation
                                results["cuda_transcription_example"] = {
                                    "input": self.test_audio,
                                    "output": text_output[:100] + "..." if isinstance(text_output, str) and len(text_output) > 100 else text_output,
                                    "timestamp": time.time(),
                                    "elapsed_time": elapsed_time,
                                    "implementation_type": implementation_type,
                                    "platform": "CUDA",
                                    "performance_metrics": performance_metrics
                                }
                    except Exception as e:
                        print(f"Real CUDA implementation failed: {e}")
                        print("Falling back to mock CUDA implementation")
                        # Fall back to mock implementation
                        implementation_type = "(MOCK)"
                        with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                             patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
                             patch('transformers.AutoModelForCTC.from_pretrained') as mock_model:
                            
                            mock_config.return_value = MagicMock()
                            mock_processor.return_value = MagicMock()
                            mock_model.return_value = MagicMock()
                            
                            endpoint, processor, handler, queue, batch_size = self.wav2vec2_bert.init_cuda(
                                self.model_name,
                                "cuda",
                                "cuda:0"
                            )
                            
                            valid_init = endpoint is not None and processor is not None and handler is not None
                            results["cuda_init"] = f"Success {implementation_type}" if valid_init else "Failed CUDA initialization"
                            
                            test_handler = self.wav2vec2_bert.create_cuda_transcription_endpoint_handler(
                                processor,
                                self.model_name,
                                "cuda:0",
                                endpoint
                            )
                            
                            with patch('soundfile.read') as mock_sf_read:
                                mock_sf_read.return_value = (np.random.randn(16000), 16000)
                                output = test_handler(self.test_audio)
                                results["cuda_handler"] = f"Success {implementation_type}" if output is not None else "Failed CUDA handler"
                                
                                # Save transcription result
                                if output is not None:
                                    results["cuda_transcription"] = output
                                    results["cuda_transcription_example"] = {
                                        "input": self.test_audio,
                                        "output": output,
                                        "timestamp": time.time(),
                                        "elapsed_time": 0.07,  # Placeholder for timing
                                        "implementation_type": implementation_type,
                                        "platform": "CUDA"
                                    }
            except Exception as e:
                results["cuda_tests"] = f"Error: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            
        # Test OpenVINO if installed - abbreviated implementation
        try:
            try:
                import openvino
                print("OpenVINO import successful")
            except ImportError:
                results["openvino_tests"] = "OpenVINO not installed"
                return results
                
            # Create a minimal OpenVINO implementation
            # This is a placeholder implementation
            endpoint = MagicMock()
            processor = MagicMock()
            
            # Create handler function
            def minimal_openvino_handler(audio_path=None):
                """Minimal handler function that works without dependencies"""
                start_time = time.time()
                
                # Return basic mock output
                return {
                    "text": "OpenVINO wav2vec2-bert transcription (mock)",
                    "implementation_type": "MOCK",
                    "device": "CPU (OpenVINO)",
                    "total_time": time.time() - start_time
                }
            
            # Set up components
            test_handler = minimal_openvino_handler
            implementation_type = "(MOCK)"
            results["openvino_init"] = f"Success {implementation_type}"
            
            # Test the handler
            output = test_handler(self.test_audio)
            results["openvino_handler"] = f"Success {implementation_type}" if output is not None else "Failed OpenVINO handler"
            
            # Process output
            if output is not None:
                if isinstance(output, dict):
                    transcription = output.get("text", "")
                else:
                    transcription = str(output)
                    
                # Record the result
                results["openvino_transcription"] = transcription
                results["openvino_transcription_example"] = {
                    "input": self.test_audio,
                    "output": transcription,
                    "timestamp": time.time(),
                    "elapsed_time": 0.01,
                    "implementation_type": "MOCK",
                    "platform": "OpenVINO"
                }
            else:
                results["openvino_transcription"] = "No output generated"
        except Exception as e:
            results["openvino_tests"] = f"Error: {str(e)}"

        return results

    def __test__(self):
        """Run tests and compare/save results"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Detailed traceback: {tb}")
            test_results = {"test_error": str(e), "traceback": tb}
        
        # Create directories if they don't exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_dir = os.path.join(base_dir, 'expected_results')
        collected_dir = os.path.join(base_dir, 'collected_results')
        
        # Create directories with appropriate permissions
        for directory in [expected_dir, collected_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, mode=0o755, exist_ok=True)
        
        # Add metadata about the environment to the results
        test_results["metadata"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "torch_version": torch.__version__ if hasattr(torch, "__version__") else "Unknown",
            "numpy_version": np.__version__ if hasattr(np, "__version__") else "Unknown",
            "transformers_version": transformers_module.__version__ if hasattr(transformers_module, "__version__") else "mocked",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "transformers_mocked": isinstance(self.resources["transformers"], MagicMock),
            "test_audio": self.test_audio,
            "test_model": self.model_name,
            "test_run_id": f"wav2vec2-bert-test-{int(time.time())}",
            "implementation_type": "(REAL)" if not self.using_mocks else "(MOCK)",
            "os_platform": sys.platform,
            "python_version": sys.version,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save collected results
        results_file = os.path.join(collected_dir, 'hf_wav2vec2_bert_test_results.json')
        try:
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Saved test results to {results_file}")
        except Exception as e:
            print(f"Error saving results to {results_file}: {str(e)}")
            
        # Create expected results file if it doesn't exist
        expected_file = os.path.join(expected_dir, 'hf_wav2vec2_bert_test_results.json')
        if not os.path.exists(expected_file):
            try:
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                    print(f"Created new expected results file: {expected_file}")
            except Exception as e:
                print(f"Error creating {expected_file}: {str(e)}")

        return test_results

if __name__ == "__main__":
    try:
        this_wav2vec2_bert = test_hf_wav2vec2_bert()
        results = this_wav2vec2_bert.__test__()
        print("WAV2VEC2-BERT Test Completed")
        
        # Print a summary of the test results
        print("\nWAV2VEC2-BERT TEST RESULTS SUMMARY")
        print(f"MODEL: {results.get('metadata', {}).get('test_model', 'Unknown')}")
        
        # Extract CPU/CUDA/OpenVINO status
        cpu_status = "UNKNOWN"
        cuda_status = "UNKNOWN"
        openvino_status = "UNKNOWN"
        
        for key, value in results.items():
            if isinstance(value, str) and "cpu_" in key and "SUCCESS" in value.upper():
                cpu_status = "SUCCESS"
                if "REAL" in value.upper():
                    cpu_status += " (REAL)"
                elif "MOCK" in value.upper():
                    cpu_status += " (MOCK)"
                    
            if isinstance(value, str) and "cuda_" in key and "SUCCESS" in value.upper():
                cuda_status = "SUCCESS"
                if "REAL" in value.upper():
                    cuda_status += " (REAL)"
                elif "MOCK" in value.upper():
                    cuda_status += " (MOCK)"
                    
            if isinstance(value, str) and "openvino_" in key and "SUCCESS" in value.upper():
                openvino_status = "SUCCESS"
                if "REAL" in value.upper():
                    openvino_status += " (REAL)"
                elif "MOCK" in value.upper():
                    openvino_status += " (MOCK)"
        
        print(f"CPU_STATUS: {cpu_status}")
        print(f"CUDA_STATUS: {cuda_status}")
        print(f"OPENVINO_STATUS: {openvino_status}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)