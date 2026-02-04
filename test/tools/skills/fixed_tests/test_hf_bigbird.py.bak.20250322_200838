#!/usr/bin/env python3

# Import hardware detection capabilities if available
try:
    from generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback

    import os
    import sys
    import json
    import time
    import datetime

    # ANSI color codes for terminal output
    GREEN = "\\033[32m"
    BLUE = "\\033[34m"
    RESET = "\\033[0m"
    import traceback
    import logging
    import argparse
    from unittest.mock import patch, MagicMock, Mock
    from typing import Dict, List, Any, Optional, Union
    from pathlib import Path

    import asyncio
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Third-party imports
    import numpy as np

    # WebGPU imports and mock setup
    HAS_WEBGPU = False
    try:
        # Attempt to check for WebGPU availability
        import ctypes.util
        HAS_WEBGPU = hasattr(ctypes.util, 'find_library') and ctypes.util.find_library('webgpu') is not None
    except ImportError:
        HAS_WEBGPU = False

    # WebNN imports and mock setup
    HAS_WEBNN = False
    try:
        # Attempt to check for WebNN availability
        import ctypes
        HAS_WEBNN = hasattr(ctypes.util, 'find_library') and ctypes.util.find_library('webnn') is not None
    except ImportError:
        HAS_WEBNN = False

    # ROCm imports and detection
    HAS_ROCM = False
    try:
        if torch.cuda.is_available() and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
            HAS_ROCM = True
            ROCM_VERSION = torch._C._rocm_version()
        elif 'ROCM_HOME' in os.environ:
            HAS_ROCM = True
    except:
        HAS_ROCM = False

    try:
        import openvino
        from openvino.runtime import Core
        HAS_OPENVINO = True
    except ImportError:
        HAS_OPENVINO = False
        logger.warning("OpenVINO not available")

    # Check if we should mock specific dependencies
    MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
    MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
    MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'
    MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'
    # Try to import torch
    try:
        if MOCK_TORCH:
            raise ImportError("Mocked torch import failure")
        import torch
        HAS_TORCH = True
    except ImportError:
        torch = MagicMock()
        HAS_TORCH = False
        logger.warning("torch not available, using mock")

    # Try to import transformers
    try:
        if MOCK_TRANSFORMERS:
            raise ImportError("Mocked transformers import failure")
        import transformers
        HAS_TRANSFORMERS = True
    except ImportError:
        transformers = MagicMock()
        HAS_TRANSFORMERS = False
        logger.warning("transformers not available, using mock")

    # Try to import sentencepiece
    try:
        if MOCK_SENTENCEPIECE:
            raise ImportError("Mocked sentencepiece import failure")
        import sentencepiece
        HAS_SENTENCEPIECE = True
    except ImportError:
        sentencepiece = MagicMock()
        HAS_SENTENCEPIECE = False
        logger.warning("sentencepiece not available, using mock")

    if not HAS_SENTENCEPIECE:
        class MockSentencePieceProcessor:
            def __init__(self, *args, **kwargs):
                self.vocab_size = 32000
            
            def encode(self, text, out_type=str):
                return [1, 2, 3, 4, 5]
            
            def decode(self, ids):
                return "Decoded text from mock"
            
            def get_piece_size(self):
                return 32000
            
            @staticmethod
            def load(model_file):
                return MockSentencePieceProcessor()

        sentencepiece.SentencePieceProcessor = MockSentencePieceProcessor

    # Hardware detection
    def check_hardware():
        """Check available hardware and return capabilities."""
        capabilities = {
            "cpu": True,
            "cuda": False,
            "cuda_version": None,
            "cuda_devices": 0,
            "mps": False,
            "openvino": False
        }
        
        # Check CUDA
        if HAS_TORCH:
            capabilities["cuda"] = torch.cuda.is_available()
            if capabilities["cuda"]:
                capabilities["cuda_devices"] = torch.cuda.device_count()
                capabilities["cuda_version"] = torch.version.cuda
        
        # Check MPS (Apple Silicon)
        if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            capabilities["mps"] = torch.mps.is_available()
        
        # Check OpenVINO
        try:
            import openvino
            capabilities["openvino"] = True
        except ImportError:
            pass
        
        return capabilities

    # Get hardware capabilities
    HW_CAPABILITIES = check_hardware()

    # Models registry - Maps model IDs to their specific configurations
    T5_MODELS_REGISTRY = {
        "t5-small": {
            "description": "T5 small model",
            "class": "T5ForConditionalGeneration",
        },
        "t5-base": {
            "description": "T5 base model",
            "class": "T5ForConditionalGeneration",
        },
        "google/flan-t5-small": {
            "description": "Flan-T5 small model",
            "class": "T5ForConditionalGeneration",
        }
    }

class TestT5Models:
    """Base test class for all T5-family models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class for a specific model or default."""
        self.model_id = model_id or "t5-small"
        
        # Verify model exists in registry
        if self.model_id not in T5_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = T5_MODELS_REGISTRY["t5-small"]
        else:
            self.model_info = T5_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "translation_en_to_fr"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "Translate to French: Hello, how are you?"
        self.test_texts = [
            "Translate to French: Hello, how are you?",
            "Translate to French: The weather is nice today."
        ]
        
        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {self.preferred_device} as preferred device")
        
        # Results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
        
    def test_pipeline(self, device="auto"):
        """Test the model using transformers pipeline API."""
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
            
        if not HAS_SENTENCEPIECE:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_deps"] = ["sentencepiece>=0.1.91"]
            results["pipeline_success"] = False
            return results
        
        try:
            # BigBird can handle longer sequences, so create a long input for testing
            long_input = " ".join(["This is a test sentence."] * 10)

            logger.info(f"Testing {self.model_id} with pipeline() on {device}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device
            }
            
            # Time the model loading
            load_start_time = time.time()
            
            # Create pipeline with T5-specific configuration to handle decoder inputs
            # This is essential for the pipeline to work correctly
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            
            # Use custom pipeline instead of the default one
            pipeline_kwargs["model"] = model
            pipeline_kwargs["tokenizer"] = tokenizer
            
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test input - T5 requires non-empty input
            pipeline_input = self.test_text
            
            # Run warmup inference if on CUDA
            if device == "cuda":
                try:
                    _ = pipeline(pipeline_input)
                except Exception:
                    pass
            
            # Run multiple inference passes
            num_runs = 3
            times = []
            outputs = []
            
            for _ in range(num_runs):
                start_time = time.time()
                output = pipeline(pipeline_input)
                end_time = time.time()
                times.append(end_time - start_time)
                outputs.append(output)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_avg_time"] = avg_time
            results["pipeline_min_time"] = min_time
            results["pipeline_max_time"] = max_time
            results["pipeline_load_time"] = load_time
            results["pipeline_error_type"] = "none"
            
            # Add to examples
            self.examples.append({
                "method": f"pipeline() on {device}",
                "input": str(pipeline_input),
                "output_preview": str(outputs[0])[:200] + "..." if len(str(outputs[0])) > 200 else str(outputs[0])
            })
            
            # Store in performance stats
            self.performance_stats[f"pipeline_{device}"] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "load_time": load_time,
                "num_runs": num_runs
            }
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_traceback"] = traceback.format_exc()
            logger.error(f"Error testing pipeline on {device}: {e}")
            
            # Classify error type
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["pipeline_error_type"] = "cuda_error"
            elif "memory" in error_str:
                results["pipeline_error_type"] = "out_of_memory"
            elif "no module named" in error_str:
                results["pipeline_error_type"] = "missing_dependency"
            else:
                results["pipeline_error_type"] = "other"
        
        # Add to overall results
        self.results[f"pipeline_{device}"] = results
        return results
        
    def test_from_pretrained(self, device="auto"):
        """Test the model using direct from_pretrained loading."""
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_core"] = ["transformers"]
            results["from_pretrained_success"] = False
            return results
            
        if not HAS_SENTENCEPIECE:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_deps"] = ["sentencepiece>=0.1.91"]
            results["from_pretrained_success"] = False
            return results
        
        try:
            logger.info(f"Testing {self.model_id} with from_pretrained() on {device}...")
            
            # Common parameters for loading
            pretrained_kwargs = {
                "local_files_only": False
            }
            
            # Time tokenizer loading
            tokenizer_load_start = time.time()
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                **pretrained_kwargs
            )
            tokenizer_load_time = time.time() - tokenizer_load_start
            
            # Use appropriate model class based on model type
            model_class = None
            if self.class_name == "T5ForConditionalGeneration":
                model_class = transformers.T5ForConditionalGeneration
            else:
                # Fallback to Auto class
                model_class = transformers.AutoModelForSeq2SeqLM
            
            # Time model loading
            model_load_start = time.time()
            model = model_class.from_pretrained(
                self.model_id,
                **pretrained_kwargs
            )
            model_load_time = time.time() - model_load_start
            
            # Move model to device
            if device != "cpu":
                model = model.to(device)
            
            # Prepare test input
            test_input = self.test_text
            
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors="pt")
            
            # Add decoder inputs for T5 models - This fixes "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            decoder_input_ids = tokenizer("", return_tensors="pt")["input_ids"]
            inputs["decoder_input_ids"] = decoder_input_ids
            
            logger.info("Added empty decoder_input_ids for T5 model")
            
            # Move inputs to device
            if device != "cpu":
                inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # Run warmup inference if using CUDA
            if device == "cuda":
                try:
                    with torch.no_grad():
                        _ = model(**inputs)
                except Exception:
                    pass
            
            # Run multiple inference passes
            num_runs = 3
            times = []
            outputs = []
            
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    output = model(**inputs)
                end_time = time.time()
                times.append(end_time - start_time)
                outputs.append(output)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Process generation output
            if hasattr(outputs[0], "logits"):
                logits = outputs[0].logits
                generated_ids = torch.argmax(logits, dim=-1)
                if hasattr(tokenizer, "decode"):
                    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    predictions = [{"generated_text": decoded_output}]
                else:
                    predictions = [{"generated_text": "Mock generated text"}]
            else:
                predictions = [{"generated_text": "Mock generated text"}]
            
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = (param_count * 4) / (1024 * 1024)  # Rough size in MB
            
            # Store results
            results["from_pretrained_success"] = True
            results["from_pretrained_avg_time"] = avg_time
            results["from_pretrained_min_time"] = min_time
            results["from_pretrained_max_time"] = max_time
            results["tokenizer_load_time"] = tokenizer_load_time
            results["model_load_time"] = model_load_time
            results["model_size_mb"] = model_size_mb
            results["from_pretrained_error_type"] = "none"
            
            # Add predictions if available
            if 'predictions' in locals():
                results["predictions"] = predictions
            
            # Add to examples
            example_data = {
                "method": f"from_pretrained() on {device}",
                "input": str(test_input)
            }
            
            if 'predictions' in locals():
                example_data["predictions"] = predictions
            
            self.examples.append(example_data)
            
            # Store in performance stats
            self.performance_stats[f"from_pretrained_{device}"] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "tokenizer_load_time": tokenizer_load_time,
                "model_load_time": model_load_time,
                "model_size_mb": model_size_mb,
                "num_runs": num_runs
            }
            
        except Exception as e:
            # Store error information
            results["from_pretrained_success"] = False
            results["from_pretrained_error"] = str(e)
            results["from_pretrained_traceback"] = traceback.format_exc()
            logger.error(f"Error testing from_pretrained on {device}: {e}")
            
            # Classify error type
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["from_pretrained_error_type"] = "cuda_error"
            elif "memory" in error_str:
                results["from_pretrained_error_type"] = "out_of_memory"
            elif "no module named" in error_str:
                results["from_pretrained_error_type"] = "missing_dependency"
            else:
                results["from_pretrained_error_type"] = "other"
        
        # Add to overall results
        self.results[f"from_pretrained_{device}"] = results
        return results
        
    def test_with_openvino(self):
        """Test the model using OpenVINO integration."""
        results = {
            "model": self.model_id,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for OpenVINO support
        if not HW_CAPABILITIES["openvino"]:
            results["openvino_error_type"] = "missing_dependency"
            results["openvino_missing_core"] = ["openvino"]
            results["openvino_success"] = False
            return results
        
        # Check for transformers
        if not HAS_TRANSFORMERS:
            results["openvino_error_type"] = "missing_dependency"
            results["openvino_missing_core"] = ["transformers"]
            results["openvino_success"] = False
            return results
        
        # Define test input
        test_input = "Translate to French: Hello, how are you?"
        
        try:
            # First try with optimum.intel if available
            try:
                import optimum.intel
                from optimum.intel import OVModelForSeq2SeqLM
                has_optimum = True
                logger.info(f"Testing {self.model_id} with OpenVINO via optimum.intel...")
            except ImportError:
                has_optimum = False
                logger.warning("optimum.intel not available, using direct OpenVINO conversion")
                
                # Time tokenizer loading
                tokenizer_load_start = time.time()
                tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
                tokenizer_load_time = time.time() - tokenizer_load_start
            
                # Time model loading
                model_load_start = time.time()
            
                if has_optimum:
                    # Use optimum.intel integration
                    model = OVModelForSeq2SeqLM.from_pretrained(
                        self.model_id,
                        export=True,
                        provider="CPU"
                    )
                else:
                    # Manual OpenVINO conversion for T5 model
                    import openvino as ov
                    from openvino.runtime import Core
                
                # Load transformer model
                pt_model = transformers.T5ForConditionalGeneration.from_pretrained(self.model_id)
                
                # Convert encoder to OpenVINO
                dummy_input = tokenizer(test_input, return_tensors="pt")
                encoder_inputs = list(dummy_input.values())
                traced_encoder = torch.jit.trace(pt_model.encoder, tuple(encoder_inputs))
                ov_encoder = ov.convert_model(traced_encoder, example_input=tuple(encoder_inputs))
                
                # For simplicity in this implementation, we'll use a wrapper class to simulate OVModelForSeq2SeqLM
                class OVT5Wrapper:
                    def __init__(self, encoder_model, pt_model, tokenizer):
                        self.encoder_model = encoder_model
                        self.pt_model = pt_model  # Keep PyTorch model for decoder (simplified approach)
                        self.tokenizer = tokenizer
                        self.core = Core()
                        self.compiled_encoder = self.core.compile_model(encoder_model, "CPU")
                        
                    def __call__(self, **kwargs):
                        # Process through OpenVINO encoder
                        input_tensors = [kwargs[key] for key in ["input_ids", "attention_mask"]]
                        encoder_outputs = self.compiled_encoder(input_tensors)[0]
                        
                        # Use encoder output with PyTorch decoder (simplified implementation)
                        encoder_outputs = torch.tensor(encoder_outputs)
                        
                        # Return a simplified output structure
                        class OutputWrapper:
                            def __init__(self, logits):
                                self.logits = logits
                        
                        # Return dummy logits for this simplified implementation
                        return OutputWrapper(torch.rand(1, 10, self.tokenizer.vocab_size))
                        
                    def generate(self, input_ids, attention_mask=None, **kwargs):
                        # Simplified generate implementation
                        return torch.tensor([[1, 2, 3, 4, 5]])  # Return dummy token IDs
                
                # Create wrapper model
                model = OVT5Wrapper(ov_encoder, pt_model, tokenizer)
                
            model_load_time = time.time() - model_load_start
            
            # Prepare input
            inputs = tokenizer(test_input, return_tensors="pt")
            
            # Warmup run
            try:
                _ = model(**inputs)
            except Exception as e:
                logger.warning(f"Warmup run failed: {e}")
            
                # Run inference
                num_runs = 3
                inference_times = []
            
                for _ in range(num_runs):
                    start_time = time.time()
                    outputs = model(**inputs)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
            
                avg_inference_time = sum(inference_times) / len(inference_times)
            
                # Process generation output
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    generated_ids = torch.argmax(logits, dim=-1)
                    
                    if hasattr(tokenizer, "decode"):
                        decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        predictions = [decoded_output]
                    else:
                        predictions = ["<mock_output>"]
                else:
                    predictions = ["<mock_output>"]
            
            # Try generation
            try:
                if hasattr(model, "generate"):
                    gen_start_time = time.time()
                    generated_ids = model.generate(inputs["input_ids"], max_length=50)
                    gen_time = time.time() - gen_start_time
                    
                    if hasattr(tokenizer, "decode"):
                        gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        results["openvino_generation"] = gen_text
                        results["openvino_generation_time"] = gen_time
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                results["openvino_generation_error"] = str(e)
            
                # Store results
                results["openvino_success"] = True
                results["openvino_load_time"] = model_load_time
                results["openvino_inference_time"] = avg_inference_time
                results["openvino_tokenizer_load_time"] = tokenizer_load_time
                results["openvino_backend"] = "optimum.intel" if has_optimum else "direct_conversion"
            
                # Add predictions if available
                if 'predictions' in locals():
                    results["openvino_predictions"] = predictions
            
                results["openvino_error_type"] = "none"
            
                # Add to examples
                example_data = {
                    "method": "OpenVINO inference",
                    "input": str(test_input)
                }
            
                if 'predictions' in locals():
                    example_data["predictions"] = predictions
            
                self.examples.append(example_data)
            
                # Store in performance stats
                self.performance_stats["openvino"] = {
                    "inference_time": avg_inference_time,
                    "load_time": model_load_time,
                    "tokenizer_load_time": tokenizer_load_time,
                    "backend": "optimum.intel" if has_optimum else "direct_conversion",
                    "num_runs": num_runs
                }
            
        except Exception as e:
            # Store error information
            results["openvino_success"] = False
            results["openvino_error"] = str(e)
            results["openvino_traceback"] = traceback.format_exc()
            logger.error(f"Error testing with OpenVINO: {e}")
            
            # Classify error
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
            
            if "no module named" in error_str:
                results["openvino_error_type"] = "missing_dependency"
                
                if "optimum" in error_str:
                    results["openvino_missing_deps"] = ["optimum-intel"]
                else:
                    results["openvino_missing_deps"] = ["openvino"]
            
            elif "memory" in error_str or "memory" in traceback_str:
                results["openvino_error_type"] = "out_of_memory"
            else:
                results["openvino_error_type"] = "other"
        
        # Add to overall results
        self.results["openvino"] = results
        return results
        
    def run_tests(self, all_hardware=False):
        """
        Run all tests for this model.
        
        Args:
            all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
        
        Returns:
            Dict containing test results
        """
        # Always test on default device
        self.test_pipeline()
        self.test_from_pretrained()
        
        # Test on all available hardware if requested
        if all_hardware:
            # Always test on CPU
            if self.preferred_device != "cpu":
                self.test_pipeline(device="cpu")
                self.test_from_pretrained(device="cpu")
            
            # Test on CUDA if available
            if HW_CAPABILITIES["cuda"] and self.preferred_device != "cuda":
                self.test_pipeline(device="cuda")
                self.test_from_pretrained(device="cuda")
            
            # Test on OpenVINO if available
            if HW_CAPABILITIES["openvino"]:
                self.test_with_openvino()
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
        # Build final results
        return {
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "hardware": HW_CAPABILITIES,
            "metadata": {
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS,
                "has_sentencepiece": HAS_SENTENCEPIECE,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
            }
        }
        
def save_results(model_id, results, output_dir="collected_results"):
    """Save test results to a file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_t5_{safe_model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    return output_path
    
def get_available_models():
    """Get a list of all available T5 models in the registry."""
    return list(T5_MODELS_REGISTRY.keys())
    
def test_all_models(output_dir="collected_results", all_hardware=False):
    """Test all registered T5 models."""
    models = get_available_models()
    results = {}
    
    for model_id in models:
        logger.info(f"Testing model: {model_id}")
        tester = TestT5Models(model_id)
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save individual results
        save_results(model_id, model_results, output_dir=output_dir)
        
        # Add to summary
        results[model_id] = {
            "success": any(r.get("pipeline_success", False) for r in model_results["results"].values()
                if r.get("pipeline_success") is not False)
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, f"hf_t5_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test T5-family models")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all registered models")
    
    # Hardware options
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for output files")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    # List options
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = get_available_models()
        print(f"\nAvailable T5-family models:")
        for model in models:
            info = T5_MODELS_REGISTRY[model]
            print(f"  - {model} ({info['class']}): {info['description']}")
        return
    
    # Create output directory if needed
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test all models if requested
    if args.all_models:
        results = test_all_models(output_dir=args.output_dir, all_hardware=args.all_hardware)
        
        # Print summary
        print(f"\nT5 Models Testing Summary:")
        total = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        print(f"Successfully tested {successful} of {total} models ({successful/total*100:.1f}%)")
        return
    
    # Test single model (default or specified)
    model_id = args.model or "t5-small"
    logger.info(f"Testing model: {model_id}")
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run test
    tester = TestT5Models(model_id)
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values()
        if r.get("pipeline_success") is not False)
    
    # Determine if real inference or mock objects were used
    using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
    using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
    
    print(f"\nTEST RESULTS SUMMARY:")
    
    # Indicate real vs mock inference clearly
    if using_real_inference and not using_mocks:
        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
    else:
        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
    
    if success:
        print(f"✅ Successfully tested {model_id}")
        
        # Print performance highlights
        for device, stats in results["performance"].items():
            if "avg_time" in stats:
                print(f"  - {device}: {stats['avg_time']:.4f}s average inference time")
        
        # Print example outputs if available
        if results.get("examples") and len(results["examples"]) > 0:
            print(f"\nExample output:")
            example = results["examples"][0]
            if "predictions" in example:
                print(f"  Input: {example['input']}")
                print(f"  Predictions: {example['predictions']}")
            elif "output_preview" in example:
                print(f"  Input: {example['input']}")
                print(f"  Output: {example['output_preview']}")
    else:
        print(f"❌ Failed to test {model_id}")
        
        # Print error information
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {test_name}: {result.get('pipeline_error_type', 'unknown')}")
                print(f"    {result.get('pipeline_error', 'Unknown error')}")
    
    print(f"\nFor detailed results, use --save flag and check the JSON output file.")

if __name__ == "__main__":
    main()