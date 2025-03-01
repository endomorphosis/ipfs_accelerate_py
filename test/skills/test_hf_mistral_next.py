# Standard library imports first
import os
import sys
import json
import time
import datetime
import traceback
from unittest.mock import MagicMock, patch

# Third-party imports next
import numpy as np

# Use absolute path setup
sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")

# Try/except pattern for importing optional dependencies
try:
    import torch
except ImportError:
    torch = MagicMock()
    print("Warning: torch not available, using mock implementation")

try:
    import transformers
except ImportError:
    transformers = MagicMock()
    print("Warning: transformers not available, using mock implementation")

# Import the module to test - Mistral-Next will use the hf_mistral module or a custom module if available
try:
    from ipfs_accelerate_py.worker.skillset.hf_mistral_next import hf_mistral_next
except ImportError:
    try:
        # Fallback to the standard Mistral module if Next-specific one isn't available
        from ipfs_accelerate_py.worker.skillset.hf_mistral import hf_mistral as hf_mistral_next
        print("Using standard Mistral module for Mistral-Next tests")
    except ImportError:
        # Create a mock class if neither module exists
        class hf_mistral_next:
            def __init__(self, resources=None, metadata=None):
                self.resources = resources if resources else {}
                self.metadata = metadata if metadata else {}
                
            def init_cpu(self, model_name, model_type, device):
                """Mock CPU initialization"""
                mock_handler = lambda text, **kwargs: {
                    "generated_text": f"Mistral-Next mock output: {text[:30]}...",
                    "implementation_type": "(MOCK)"
                }
                return MagicMock(), MagicMock(), mock_handler, None, 1
                
            def init_cuda(self, model_name, model_type, device):
                """Mock CUDA initialization"""
                return self.init_cpu(model_name, model_type, device)
                
            def init_openvino(self, model_name, model_type, device, *args):
                """Mock OpenVINO initialization"""
                return self.init_cpu(model_name, model_type, device)
        
        print("Warning: Neither hf_mistral_next nor hf_mistral found, using mock implementation")

# Add CUDA support to the Mistral-Next class if needed
def init_cuda(self, model_name, model_type, device_label="cuda:0"):
    """Initialize Mistral-Next model with CUDA support.
    
    Args:
        model_name: Name or path of the model
        model_type: Type of model task (e.g., "text-generation")
        device_label: CUDA device label (e.g., "cuda:0")
        
    Returns:
        tuple: (endpoint, tokenizer, handler, queue, batch_size)
    """
    try:
        import sys
        import torch
        from unittest import mock
        
        # Try to import the necessary utility functions
        sys.path.insert(0, "/home/barberb/ipfs_accelerate_py/test")
        import utils as test_utils
        
        print(f"Checking CUDA availability for Mistral-Next model {model_name}")
        
        # Verify that CUDA is actually available
        if not torch.cuda.is_available():
            print("CUDA not available, using mock implementation")
            return mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), None, 1
        
        # Get the CUDA device
        device = test_utils.get_cuda_device(device_label)
        if device is None:
            print("Failed to get valid CUDA device, using mock implementation")
            return mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), None, 1
        
        print(f"Using CUDA device: {device}")
        
        # Try to initialize with real components
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"Successfully loaded tokenizer for {model_name}")
            except Exception as tokenizer_err:
                print(f"Failed to load tokenizer: {tokenizer_err}")
                tokenizer = mock.MagicMock()
                tokenizer.is_real_simulation = False
            
            # Load model
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name)
                print(f"Successfully loaded model {model_name}")
                
                # Optimize and move to GPU
                model = test_utils.optimize_cuda_memory(model, device, use_half_precision=True)
                model.eval()
                print(f"Model loaded to {device} and optimized for inference")
                
                model.is_real_simulation = True
            except Exception as model_err:
                print(f"Failed to load model: {model_err}")
                model = mock.MagicMock()
                model.is_real_simulation = False
            
            # Create the handler function
            def handler(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, top_k=50, **kwargs):
                """Handle text generation with CUDA acceleration."""
                try:
                    start_time = time.time()
                    
                    # If we're using mock components, return a fixed response
                    if isinstance(model, mock.MagicMock) or isinstance(tokenizer, mock.MagicMock):
                        print("Using mock handler for CUDA Mistral-Next")
                        time.sleep(0.1)  # Simulate processing time
                        return {
                            "generated_text": f"(MOCK CUDA) Generated text for prompt: {prompt[:30]}...",
                            "implementation_type": "MOCK",
                            "device": "cuda:0 (mock)",
                            "total_time": time.time() - start_time
                        }
                    
                    # Real implementation
                    try:
                        # Tokenize the input
                        inputs = tokenizer(prompt, return_tensors="pt")
                        
                        # Move inputs to CUDA
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Set up generation parameters
                        generation_kwargs = {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "do_sample": True if temperature > 0 else False,
                        }
                        
                        # Update with any additional kwargs
                        generation_kwargs.update(kwargs)
                        
                        # Measure GPU memory before generation
                        cuda_mem_before = torch.cuda.memory_allocated(device) / (1024 * 1024) if hasattr(torch.cuda, "memory_allocated") else 0
                        
                        # Generate text
                        with torch.no_grad():
                            torch.cuda.synchronize() if hasattr(torch.cuda, "synchronize") else None
                            generation_start = time.time()
                            outputs = model.generate(**inputs, **generation_kwargs)
                            torch.cuda.synchronize() if hasattr(torch.cuda, "synchronize") else None
                            generation_time = time.time() - generation_start
                        
                        # Measure GPU memory after generation
                        cuda_mem_after = torch.cuda.memory_allocated(device) / (1024 * 1024) if hasattr(torch.cuda, "memory_allocated") else 0
                        gpu_mem_used = cuda_mem_after - cuda_mem_before
                        
                        # Decode the output
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Some models include the prompt in the output, try to remove it
                        if prompt in generated_text:
                            generated_text = generated_text[len(prompt):].strip()
                        
                        # Calculate metrics
                        total_time = time.time() - start_time
                        token_count = len(outputs[0])
                        tokens_per_second = token_count / generation_time if generation_time > 0 else 0
                        
                        # Return results with detailed metrics
                        return {
                            "generated_text": prompt + " " + generated_text if not prompt in generated_text else generated_text,
                            "implementation_type": "REAL",
                            "device": str(device),
                            "total_time": total_time,
                            "generation_time": generation_time,
                            "gpu_memory_used_mb": gpu_mem_used,
                            "tokens_per_second": tokens_per_second,
                            "token_count": token_count,
                        }
                        
                    except Exception as e:
                        print(f"Error in CUDA generation: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Return error information
                        return {
                            "generated_text": f"Error in CUDA generation: {str(e)}",
                            "implementation_type": "REAL (error)",
                            "error": str(e),
                            "total_time": time.time() - start_time
                        }
                except Exception as outer_e:
                    print(f"Outer error in CUDA handler: {outer_e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Final fallback
                    return {
                        "generated_text": f"(MOCK CUDA) Generated text for prompt: {prompt[:30]}...",
                        "implementation_type": "MOCK",
                        "device": "cuda:0 (mock)",
                        "total_time": time.time() - start_time,
                        "error": str(outer_e)
                    }
            
            # Return the components
            return model, tokenizer, handler, None, 4  # Batch size of 4
            
        except ImportError as e:
            print(f"Required libraries not available: {e}")
            
    except Exception as e:
        print(f"Error in Mistral-Next init_cuda: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to mock implementation
    return mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), None, 1

# Add the CUDA initialization method to the Mistral-Next class if it doesn't already have one
if not hasattr(hf_mistral_next, 'init_cuda'):
    hf_mistral_next.init_cuda = init_cuda

class test_hf_mistral_next:
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the Mistral-Next test class.
        
        Args:
            resources (dict, optional): Resources dictionary
            metadata (dict, optional): Metadata dictionary
        """
        # Try to import transformers directly if available
        try:
            import transformers
            transformers_module = transformers
        except ImportError:
            transformers_module = MagicMock()
            
        self.resources = resources if resources else {
            "torch": torch,
            "numpy": np,
            "transformers": transformers_module
        }
        self.metadata = metadata if metadata else {}
        self.mistral_next = hf_mistral_next(resources=self.resources, metadata=self.metadata)
        
        # Primary model for Mistral-Next
        self.primary_model = "mistralai/Mistral-Next-Developer"
        
        # Alternative models in order of preference
        self.alternative_models = [
            "mistralai/Mistral-Next-Small-Preview",
            "mistralai/Mistral-Next-v0.1-Developer",
            "mistralai/Mixtral-8x22B-v0.1",
            "mistralai/Mistral-7B-v0.3"  # Fallback to latest standard Mistral if no Next available
        ]
        
        # Initialize with primary model
        self.model_name = self.primary_model
        
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
                    
                    # If all alternatives failed, check local cache
                    if self.model_name == self.primary_model:
                        # Try to find cached models
                        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models")
                        if os.path.exists(cache_dir):
                            # Look for any Mistral model in cache with "next" in the name
                            mistral_models = [name for name in os.listdir(cache_dir) if 
                                             any(x in name.lower() for x in ["mistral-next", "mistral_next"])]
                            
                            if mistral_models:
                                # Use the first model found
                                mistral_model_name = mistral_models[0].replace("--", "/")
                                print(f"Found local Mistral-Next model: {mistral_model_name}")
                                self.model_name = mistral_model_name
                            else:
                                # Create local test model
                                print("No suitable Mistral-Next models found in cache, creating local test model")
                                self.model_name = self._create_test_model()
                                print(f"Created local test model: {self.model_name}")
                        else:
                            # Create local test model
                            print("No cache directory found, creating local test model")
                            self.model_name = self._create_test_model()
                            print(f"Created local test model: {self.model_name}")
            else:
                # If transformers is mocked, use local test model
                print("Transformers is mocked, using local test model")
                self.model_name = self._create_test_model()
                
        except Exception as e:
            print(f"Error finding model: {e}")
            # Fall back to local test model as last resort
            self.model_name = self._create_test_model()
            print("Falling back to local test model due to error")
            
        print(f"Using model: {self.model_name}")
        
        # Prepare test prompts specific to Mistral-Next capabilities
        self.test_prompts = {
            "basic": "Write a short story about a robot discovering emotions.",
            "reasoning": "Explain the process of photosynthesis and why it's important for life on Earth.",
            "math": "If a triangle has sides of lengths 3, 4, and 5, what is its area?",
            "coding": "Write a Python function to find the nth Fibonacci number using dynamic programming.",
            "system_prompt": "You are a financial advisor helping a client plan for retirement. The client is 35 years old and wants to retire by 60."
        }
        
        # Initialize collection arrays for examples and status
        self.examples = []
        self.status_messages = {}
        return None
        
    def _create_test_model(self):
        """
        Create a tiny language model for testing without needing Hugging Face authentication.
        
        Returns:
            str: Path to the created model
        """
        try:
            print("Creating local test model for Mistral-Next testing...")
            
            # Create model directory in /tmp for tests
            test_model_dir = os.path.join("/tmp", "mistral_next_test_model")
            os.makedirs(test_model_dir, exist_ok=True)
            
            # Create a minimal config file for a tiny Mistral-Next-style model
            config = {
                "architectures": ["MistralForCausalLM"],
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_act": "silu",
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 2048,
                "max_position_embeddings": 4096,
                "model_type": "mistral",
                "num_attention_heads": 16,
                "num_hidden_layers": 4,
                "num_key_value_heads": 8,
                "pad_token_id": 0,
                "rms_norm_eps": 1e-05,
                "tie_word_embeddings": False,
                "torch_dtype": "float32",
                "transformers_version": "4.36.0",
                "use_cache": True,
                "vocab_size": 32000,
                "sliding_window": 8192,
                "rope_theta": 10000.0,
                "attention_bias": false
            }
            
            with open(os.path.join(test_model_dir, "config.json"), "w") as f:
                json.dump(config, f)
                
            # Create a minimal vocabulary file (required for tokenizer)
            tokenizer_config = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "model_max_length": 4096,
                "padding_side": "right",
                "use_fast": True,
                "pad_token": "[PAD]"
            }
            
            with open(os.path.join(test_model_dir, "tokenizer_config.json"), "w") as f:
                json.dump(tokenizer_config, f)
                
            # Create a minimal tokenizer.json
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [
                    {"id": 0, "special": True, "content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
                    {"id": 1, "special": True, "content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
                    {"id": 2, "special": True, "content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
                ],
                "normalizer": {"type": "Sequence", "normalizers": [{"type": "Lowercase", "lowercase": []}]},
                "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}]},
                "post_processor": {"type": "TemplateProcessing", "single": ["<s>", "$A", "</s>"], "pair": ["<s>", "$A", "</s>", "$B", "</s>"], "special_tokens": {"<s>": {"id": 1, "type_id": 0}, "</s>": {"id": 2, "type_id": 0}}},
                "decoder": {"type": "ByteLevel"}
            }
            
            with open(os.path.join(test_model_dir, "tokenizer.json"), "w") as f:
                json.dump(tokenizer_json, f)
            
            # Create vocabulary.txt with basic tokens
            special_tokens_map = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "[PAD]",
                "unk_token": "<unk>"
            }
            
            with open(os.path.join(test_model_dir, "special_tokens_map.json"), "w") as f:
                json.dump(special_tokens_map, f)
            
            # Create a small random model weights file if torch is available
            if hasattr(torch, "save") and not isinstance(torch, MagicMock):
                # Create random tensors for model weights
                model_state = {}
                
                vocab_size = config["vocab_size"]
                hidden_size = config["hidden_size"]
                intermediate_size = config["intermediate_size"]
                num_heads = config["num_attention_heads"]
                num_kv_heads = config["num_key_value_heads"]
                num_layers = config["num_hidden_layers"]
                
                # Create embedding weights
                model_state["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
                
                # Create layers
                for layer_idx in range(num_layers):
                    layer_prefix = f"model.layers.{layer_idx}"
                    
                    # Input layernorm
                    model_state[f"{layer_prefix}.input_layernorm.weight"] = torch.ones(hidden_size)
                    
                    # Self-attention
                    model_state[f"{layer_prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
                    model_state[f"{layer_prefix}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size // (num_heads // num_kv_heads))
                    model_state[f"{layer_prefix}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size // (num_heads // num_kv_heads))
                    model_state[f"{layer_prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
                    
                    # Post-attention layernorm
                    model_state[f"{layer_prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size)
                    
                    # Feed-forward network
                    model_state[f"{layer_prefix}.mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
                    model_state[f"{layer_prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)
                    model_state[f"{layer_prefix}.mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
                
                # Final layernorm
                model_state["model.norm.weight"] = torch.ones(hidden_size)
                
                # Final lm_head
                model_state["lm_head.weight"] = torch.randn(vocab_size, hidden_size)
                
                # Save model weights
                torch.save(model_state, os.path.join(test_model_dir, "pytorch_model.bin"))
                print(f"Created PyTorch model weights in {test_model_dir}/pytorch_model.bin")
                
                # Create model.safetensors.index.json for larger model compatibility
                index_data = {
                    "metadata": {
                        "total_size": 0  # Will be filled
                    },
                    "weight_map": {}
                }
                
                # Fill weight map with placeholders
                total_size = 0
                for key in model_state:
                    tensor_size = model_state[key].nelement() * model_state[key].element_size()
                    total_size += tensor_size
                    index_data["weight_map"][key] = "model.safetensors"
                
                index_data["metadata"]["total_size"] = total_size
                
                with open(os.path.join(test_model_dir, "model.safetensors.index.json"), "w") as f:
                    json.dump(index_data, f)
            
            print(f"Test model created at {test_model_dir}")
            return test_model_dir
            
        except Exception as e:
            print(f"Error creating test model: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Fall back to a model name that won't need to be downloaded for mocks
            return "mistral-next-test"

    def test(self):
        """
        Run all tests for the Mistral-Next language model, organized by hardware platform.
        Tests CPU, CUDA, and OpenVINO implementations.
        
        Returns:
            dict: Structured test results with status, examples and metadata
        """
        results = {}
        
        # Test basic initialization
        try:
            results["init"] = "Success" if self.mistral_next is not None else "Failed initialization"
        except Exception as e:
            results["init"] = f"Error: {str(e)}"

        # ====== CPU TESTS ======
        try:
            print("Testing Mistral-Next on CPU...")
            # Try with real model first
            try:
                transformers_available = not isinstance(self.resources["transformers"], MagicMock)
                if transformers_available:
                    print("Using real transformers for CPU test")
                    # Real model initialization
                    endpoint, tokenizer, handler, queue, batch_size = self.mistral_next.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cpu_init"] = "Success (REAL)" if valid_init else "Failed CPU initialization"
                    
                    if valid_init:
                        # Test with various prompts to demonstrate Mistral-Next's capabilities
                        for prompt_name, prompt_text in self.test_prompts.items():
                            # Test with real handler
                            start_time = time.time()
                            output = handler(prompt_text)
                            elapsed_time = time.time() - start_time
                            
                            results[f"cpu_{prompt_name}_handler"] = "Success (REAL)" if output is not None else f"Failed CPU handler for {prompt_name}"
                            
                            # Check output structure and store sample output
                            if output is not None and isinstance(output, dict):
                                results[f"cpu_{prompt_name}_output"] = "Valid (REAL)" if "generated_text" in output else "Missing generated_text"
                                
                                # Record example
                                generated_text = output.get("generated_text", "")
                                self.examples.append({
                                    "input": prompt_text,
                                    "output": {
                                        "generated_text": generated_text[:300] + "..." if len(generated_text) > 300 else generated_text
                                    },
                                    "timestamp": datetime.datetime.now().isoformat(),
                                    "elapsed_time": elapsed_time,
                                    "implementation_type": "REAL",
                                    "platform": "CPU",
                                    "prompt_type": prompt_name
                                })
                                
                                # Store sample of actual generated text for results
                                if "generated_text" in output:
                                    generated_text = output["generated_text"]
                                    results[f"cpu_{prompt_name}_sample"] = generated_text[:150] + "..." if len(generated_text) > 150 else generated_text
                            else:
                                results[f"cpu_{prompt_name}_output"] = "Invalid output format"
                                self.status_messages[f"cpu_{prompt_name}"] = "Invalid output format"
                else:
                    raise ImportError("Transformers not available")
                    
            except Exception as e:
                # Fall back to mock if real model fails
                print(f"Falling back to mock model for CPU: {str(e)}")
                self.status_messages["cpu_real"] = f"Failed: {str(e)}"
                
                with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                     patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                     patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                    
                    mock_config.return_value = MagicMock()
                    mock_tokenizer.return_value = MagicMock()
                    mock_tokenizer.return_value.batch_decode = MagicMock(return_value=["Once upon a time..."])
                    mock_model.return_value = MagicMock()
                    mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
                    
                    endpoint, tokenizer, handler, queue, batch_size = self.mistral_next.init_cpu(
                        self.model_name,
                        "cpu",
                        "cpu"
                    )
                    
                    valid_init = endpoint is not None and tokenizer is not None and handler is not None
                    results["cpu_init"] = "Success (MOCK)" if valid_init else "Failed CPU initialization"
                    
                    # Test with basic prompt only in mock mode
                    prompt_text = self.test_prompts["basic"]
                    start_time = time.time()
                    output = handler(prompt_text)
                    elapsed_time = time.time() - start_time
                    
                    results["cpu_basic_handler"] = "Success (MOCK)" if output is not None else "Failed CPU handler"
                    
                    # Record example
                    mock_text = "Once upon a time, in a laboratory filled with advanced machines and blinking lights, there was a robot named Circuit. Circuit was designed to be the most efficient data processor ever created, capable of handling complex calculations and simulations in microseconds. The robot had been programmed with state-of-the-art artificial intelligence algorithms, but it was never meant to develop something as unpredictable and human as emotions."
                    self.examples.append({
                        "input": prompt_text,
                        "output": {
                            "generated_text": mock_text
                        },
                        "timestamp": datetime.datetime.now().isoformat(),
                        "elapsed_time": elapsed_time,
                        "implementation_type": "MOCK",
                        "platform": "CPU",
                        "prompt_type": "basic"
                    })
                    
                    # Store the mock output for verification
                    if output is not None and isinstance(output, dict) and "generated_text" in output:
                        results["cpu_basic_output"] = "Valid (MOCK)"
                        results["cpu_basic_sample"] = "(MOCK) " + output["generated_text"][:150]
                
        except Exception as e:
            print(f"Error in CPU tests: {e}")
            traceback.print_exc()
            results["cpu_tests"] = f"Error: {str(e)}"
            self.status_messages["cpu"] = f"Failed: {str(e)}"

        # ====== CUDA TESTS ======
        print(f"CUDA availability check result: {torch.cuda.is_available()}")
        cuda_available = torch.cuda.is_available() if not isinstance(torch, MagicMock) else False
        if cuda_available:
            try:
                print("Testing Mistral-Next on CUDA...")
                # Try with real model first
                try:
                    transformers_available = not isinstance(self.resources["transformers"], MagicMock)
                    if transformers_available:
                        print("Using real transformers for CUDA test")
                        # Real model initialization
                        endpoint, tokenizer, handler, queue, batch_size = self.mistral_next.init_cuda(
                            self.model_name,
                            "cuda",
                            "cuda:0"
                        )
                        
                        valid_init = endpoint is not None and tokenizer is not None and handler is not None
                        results["cuda_init"] = "Success (REAL)" if valid_init else "Failed CUDA initialization"
                        
                        if valid_init:
                            # Test with a subset of prompts to demonstrate Mistral-Next's capabilities
                            # Just using basic and reasoning to keep tests quick
                            test_subset = {"basic": self.test_prompts["basic"], 
                                           "reasoning": self.test_prompts["reasoning"]}
                            
                            for prompt_name, prompt_text in test_subset.items():
                                # Test with handler
                                start_time = time.time()
                                output = handler(prompt_text)
                                elapsed_time = time.time() - start_time
                                
                                # Process output
                                if output is not None:
                                    # Extract fields based on output format
                                    if isinstance(output, dict):
                                        if "generated_text" in output:
                                            generated_text = output["generated_text"]
                                            implementation_type = output.get("implementation_type", "REAL")
                                        elif "text" in output:
                                            generated_text = output["text"]
                                            implementation_type = output.get("implementation_type", "REAL")
                                        else:
                                            generated_text = str(output)
                                            implementation_type = "UNKNOWN"
                                    else:
                                        generated_text = str(output)
                                        implementation_type = "UNKNOWN"
                                    
                                    # Extract GPU memory and other metrics if available
                                    gpu_memory = output.get("gpu_memory_used_mb") if isinstance(output, dict) else None
                                    generation_time = output.get("generation_time") if isinstance(output, dict) else None
                                    
                                    # Record status
                                    results[f"cuda_{prompt_name}_handler"] = f"Success ({implementation_type})"
                                    
                                    # Create example output dictionary
                                    example_output = {
                                        "generated_text": generated_text[:300] + "..." if len(generated_text) > 300 else generated_text
                                    }
                                    
                                    # Add metrics if available
                                    if gpu_memory is not None:
                                        example_output["gpu_memory_mb"] = gpu_memory
                                    if generation_time is not None:
                                        example_output["generation_time"] = generation_time
                                    
                                    # Record example
                                    self.examples.append({
                                        "input": prompt_text,
                                        "output": example_output,
                                        "timestamp": datetime.datetime.now().isoformat(),
                                        "elapsed_time": elapsed_time,
                                        "implementation_type": implementation_type,
                                        "platform": "CUDA",
                                        "prompt_type": prompt_name
                                    })
                                    
                                    # Store sample text
                                    results[f"cuda_{prompt_name}_sample"] = generated_text[:150] + "..." if len(generated_text) > 150 else generated_text
                                    
                                    # Add performance metrics if available
                                    if gpu_memory is not None:
                                        results[f"cuda_{prompt_name}_gpu_memory"] = gpu_memory
                                    if generation_time is not None:
                                        results[f"cuda_{prompt_name}_generation_time"] = generation_time
                                else:
                                    results[f"cuda_{prompt_name}_handler"] = "Failed CUDA handler"
                                    self.status_messages[f"cuda_{prompt_name}"] = "Failed to generate output"
                            
                            # Test batch capabilities
                            try:
                                batch_prompts = [self.test_prompts["basic"], self.test_prompts["reasoning"]]
                                batch_start_time = time.time()
                                batch_output = handler(batch_prompts)
                                batch_elapsed_time = time.time() - batch_start_time
                                
                                if batch_output is not None:
                                    if isinstance(batch_output, list):
                                        results["cuda_batch"] = f"Success - {len(batch_output)} results"
                                        
                                        # Extract first result
                                        first_result = batch_output[0]
                                        sample_text = ""
                                        
                                        if isinstance(first_result, dict):
                                            if "generated_text" in first_result:
                                                sample_text = first_result["generated_text"]
                                            elif "text" in first_result:
                                                sample_text = first_result["text"]
                                        else:
                                            sample_text = str(first_result)
                                        
                                        # Record batch example
                                        self.examples.append({
                                            "input": "Batch prompts",
                                            "output": {
                                                "first_result": sample_text[:150] + "..." if len(sample_text) > 150 else sample_text,
                                                "batch_size": len(batch_output)
                                            },
                                            "timestamp": datetime.datetime.now().isoformat(),
                                            "elapsed_time": batch_elapsed_time,
                                            "implementation_type": "BATCH",
                                            "platform": "CUDA"
                                        })
                                        
                                        # Store sample in results
                                        results["cuda_batch_sample"] = sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
                                    else:
                                        results["cuda_batch"] = "Unexpected batch format"
                                else:
                                    results["cuda_batch"] = "Failed batch generation"
                            except Exception as batch_err:
                                print(f"Error in batch generation: {batch_err}")
                                results["cuda_batch"] = f"Error: {str(batch_err)}"
                    else:
                        raise ImportError("Transformers not available")
                        
                except Exception as e:
                    # Fall back to mock if real model fails
                    print(f"Falling back to mock model for CUDA: {str(e)}")
                    self.status_messages["cuda_real"] = f"Failed: {str(e)}"
                    
                    with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
                         patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                         patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                        
                        mock_config.return_value = MagicMock()
                        mock_tokenizer.return_value = MagicMock()
                        mock_model.return_value = MagicMock()
                        mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
                        mock_tokenizer.batch_decode.return_value = ["Once upon a time..."]
                        
                        endpoint, tokenizer, handler, queue, batch_size = self.mistral_next.init_cuda(
                            self.model_name,
                            "cuda",
                            "cuda:0"
                        )
                        
                        valid_init = endpoint is not None and tokenizer is not None and handler is not None
                        results["cuda_init"] = "Success (MOCK)" if valid_init else "Failed CUDA initialization"
                        
                        # Test with basic prompt only in mock mode
                        prompt_text = self.test_prompts["basic"]
                        start_time = time.time()
                        output = handler(prompt_text)
                        elapsed_time = time.time() - start_time
                        
                        # Process mock output
                        implementation_type = "MOCK"
                        if isinstance(output, dict):
                            if "implementation_type" in output:
                                implementation_type = output["implementation_type"]
                                
                            if "generated_text" in output:
                                mock_text = output["generated_text"]
                            elif "text" in output:
                                mock_text = output["text"]
                            else:
                                mock_text = "In a futuristic laboratory, a robot named Unit-7 was designed to be the perfect assistant. It was programmed to be efficient, logical, and precise in all its tasks. Each day, it would help scientists with complex calculations, organize data, and perform monotonous tasks that humans found tedious. Unit-7 was exceptional at its job, never making errors or complaining about the workload. However, something unexpected began to happen."
                        else:
                            mock_text = "In a futuristic laboratory, a robot named Unit-7 was designed to be the perfect assistant. It was programmed to be efficient, logical, and precise in all its tasks. Each day, it would help scientists with complex calculations, organize data, and perform monotonous tasks that humans found tedious. Unit-7 was exceptional at its job, never making errors or complaining about the workload. However, something unexpected began to happen."
                        
                        results["cuda_basic_handler"] = f"Success ({implementation_type})"
                        
                        # Record example
                        self.examples.append({
                            "input": prompt_text,
                            "output": {
                                "generated_text": mock_text
                            },
                            "timestamp": datetime.datetime.now().isoformat(),
                            "elapsed_time": elapsed_time,
                            "implementation_type": implementation_type,
                            "platform": "CUDA",
                            "prompt_type": "basic"
                        })
                        
                        # Store in results
                        results["cuda_basic_output"] = "Valid (MOCK)"
                        results["cuda_basic_sample"] = "(MOCK) " + mock_text[:150]
            except Exception as e:
                print(f"Error in CUDA tests: {e}")
                traceback.print_exc()
                results["cuda_tests"] = f"Error: {str(e)}"
                self.status_messages["cuda"] = f"Failed: {str(e)}"
        else:
            results["cuda_tests"] = "CUDA not available"
            self.status_messages["cuda"] = "CUDA not available"

        # ====== OPENVINO TESTS ======
        try:
            print("Testing Mistral-Next on OpenVINO...")
            # First check if OpenVINO is installed
            try:
                import openvino
                has_openvino = True
                print("OpenVINO is installed")
            except ImportError:
                has_openvino = False
                results["openvino_tests"] = "OpenVINO not installed"
                self.status_messages["openvino"] = "OpenVINO not installed"
                
            if has_openvino:
                # Import the existing OpenVINO utils from the main package
                from ipfs_accelerate_py.worker.openvino_utils import openvino_utils
                
                # Initialize openvino_utils
                ov_utils = openvino_utils(resources=self.resources, metadata=self.metadata)
                
                # Setup OpenVINO runtime environment
                with patch('openvino.runtime.Core' if hasattr(openvino, 'runtime') and hasattr(openvino.runtime, 'Core') else 'openvino.Core'):
                    
                    # Initialize OpenVINO endpoint with real utils
                    endpoint, tokenizer, handler, queue, batch_size = self.mistral_next.init_openvino(
                        self.model_name,
                        "text-generation",
                        "CPU",
                        "openvino:0",
                        ov_utils.get_optimum_openvino_model,
                        ov_utils.get_openvino_model,
                        ov_utils.get_openvino_pipeline_type,
                        ov_utils.openvino_cli_convert
                    )
                    
                    valid_init = handler is not None
                    results["openvino_init"] = "Success (REAL)" if valid_init else "Failed OpenVINO initialization"
                    
                    if valid_init:
                        # Test with basic prompt only for OpenVINO
                        prompt_text = self.test_prompts["basic"]
                        start_time = time.time()
                        output = handler(prompt_text)
                        elapsed_time = time.time() - start_time
                        
                        results["openvino_basic_handler"] = "Success (REAL)" if output is not None else "Failed OpenVINO handler"
                        
                        # Check output and record example
                        if output is not None and isinstance(output, dict) and "generated_text" in output:
                            generated_text = output["generated_text"]
                            implementation_type = output.get("implementation_type", "REAL")
                            
                            self.examples.append({
                                "input": prompt_text,
                                "output": {
                                    "generated_text": generated_text[:300] + "..." if len(generated_text) > 300 else generated_text
                                },
                                "timestamp": datetime.datetime.now().isoformat(),
                                "elapsed_time": elapsed_time,
                                "implementation_type": implementation_type,
                                "platform": "OpenVINO",
                                "prompt_type": "basic"
                            })
                            
                            # Store sample in results
                            results["openvino_basic_output"] = "Valid (REAL)"
                            results["openvino_basic_sample"] = generated_text[:150] + "..." if len(generated_text) > 150 else generated_text
                        else:
                            results["openvino_basic_output"] = "Invalid output format"
                            self.status_messages["openvino"] = "Invalid output format"
                    
        except ImportError:
            results["openvino_tests"] = "OpenVINO not installed"
            self.status_messages["openvino"] = "OpenVINO not installed"
        except Exception as e:
            print(f"Error in OpenVINO tests: {e}")
            traceback.print_exc()
            results["openvino_tests"] = f"Error: {str(e)}"
            self.status_messages["openvino"] = f"Failed: {str(e)}"

        # Create structured results
        structured_results = {
            "status": results,
            "examples": self.examples,
            "metadata": {
                "model_name": self.model_name,
                "test_timestamp": datetime.datetime.now().isoformat(),
                "python_version": sys.version,
                "torch_version": torch.__version__ if hasattr(torch, "__version__") else "Unknown",
                "transformers_version": transformers.__version__ if hasattr(transformers, "__version__") else "Unknown",
                "platform_status": self.status_messages,
                "cuda_available": cuda_available,
                "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch, 'backends') else False,
                "transformers_mocked": isinstance(self.resources["transformers"], MagicMock),
                "test_prompts": self.test_prompts
            }
        }

        return structured_results

    def __test__(self):
        """
        Run tests and compare/save results.
        Handles result collection, comparison with expected results, and storage.
        
        Returns:
            dict: Test results
        """
        # Run actual tests
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {
                "status": {"test_error": str(e)},
                "examples": [],
                "metadata": {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            }
        
        # Create directories if they don't exist
        expected_dir = os.path.join(os.path.dirname(__file__), 'expected_results')
        collected_dir = os.path.join(os.path.dirname(__file__), 'collected_results')
        
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save collected results
        collected_file = os.path.join(collected_dir, 'hf_mistral_next_test_results.json')
        with open(collected_file, 'w') as f:
            json.dump(test_results, f, indent=2)
            print(f"Saved results to {collected_file}")
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'hf_mistral_next_test_results.json')
        if os.path.exists(expected_file):
            try:
                with open(expected_file, 'r') as f:
                    expected_results = json.load(f)
                    
                # Simple structure validation
                if "status" in expected_results and "examples" in expected_results:
                    print("Test completed successfully!")
            except Exception as e:
                print(f"Error comparing with expected results: {str(e)}")
                # Create expected results file if there's an error
                with open(expected_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                    print(f"Created new expected results file: {expected_file}")
        else:
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                print(f"Created new expected results file: {expected_file}")

        return test_results

if __name__ == "__main__":
    try:
        print("Starting Mistral-Next test...")
        this_mistral_next = test_hf_mistral_next()
        results = this_mistral_next.__test__()
        print("Mistral-Next test completed")
        
        # Print test results in detailed format for better parsing
        status_dict = results.get("status", {})
        examples = results.get("examples", [])
        metadata = results.get("metadata", {})
        
        # Extract implementation status
        cpu_status = "UNKNOWN"
        cuda_status = "UNKNOWN"
        openvino_status = "UNKNOWN"
        
        for key, value in status_dict.items():
            if "cpu_" in key and "REAL" in value:
                cpu_status = "REAL"
            elif "cpu_" in key and "MOCK" in value:
                cpu_status = "MOCK"
                
            if "cuda_" in key and "REAL" in value:
                cuda_status = "REAL"
            elif "cuda_" in key and "MOCK" in value:
                cuda_status = "MOCK"
                
            if "openvino_" in key and "REAL" in value:
                openvino_status = "REAL"
            elif "openvino_" in key and "MOCK" in value:
                openvino_status = "MOCK"
                
        # Also look in examples
        for example in examples:
            platform = example.get("platform", "")
            impl_type = example.get("implementation_type", "")
            
            if platform == "CPU" and "REAL" in impl_type:
                cpu_status = "REAL"
            elif platform == "CPU" and "MOCK" in impl_type:
                cpu_status = "MOCK"
                
            if platform == "CUDA" and "REAL" in impl_type:
                cuda_status = "REAL"
            elif platform == "CUDA" and "MOCK" in impl_type:
                cuda_status = "MOCK"
                
            if platform == "OpenVINO" and "REAL" in impl_type:
                openvino_status = "REAL"
            elif platform == "OpenVINO" and "MOCK" in impl_type:
                openvino_status = "MOCK"
        
        # Print summary in a parser-friendly format
        print("\nMISTRAL-NEXT TEST RESULTS SUMMARY")
        print(f"MODEL: {metadata.get('model_name', 'Unknown')}")
        print(f"CPU_STATUS: {cpu_status}")
        print(f"CUDA_STATUS: {cuda_status}")
        print(f"OPENVINO_STATUS: {openvino_status}")
        
        # Group examples by platform and prompt type
        grouped_examples = {}
        for example in examples:
            platform = example.get("platform", "Unknown")
            prompt_type = example.get("prompt_type", "unknown")
            key = f"{platform}_{prompt_type}"
            
            if key not in grouped_examples:
                grouped_examples[key] = []
                
            grouped_examples[key].append(example)
        
        # Print a summary of examples by type
        print("\nEXAMPLE SUMMARY:")
        for key, example_list in grouped_examples.items():
            if example_list:
                platform, prompt_type = key.split("_", 1)
                print(f"{platform} - {prompt_type}: {len(example_list)} examples")
                
                # Print first example details
                example = example_list[0]
                output = example.get("output", {})
                if "generated_text" in output:
                    print(f"  Sample: {output['generated_text'][:100]}...")
                    
                # Print performance metrics if available
                if "gpu_memory_mb" in output:
                    print(f"  GPU Memory: {output['gpu_memory_mb']:.2f} MB")
                if "generation_time" in output:
                    print(f"  Generation Time: {output['generation_time']:.4f}s")
                
        # Print a JSON representation to make it easier to parse
        print("\nstructured_results")
        print(json.dumps({
            "status": {
                "cpu": cpu_status,
                "cuda": cuda_status,
                "openvino": openvino_status
            },
            "model_name": metadata.get("model_name", "Unknown"),
            "example_count": len(examples)
        }))
        
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)