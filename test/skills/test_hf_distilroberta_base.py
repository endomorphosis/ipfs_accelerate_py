#!/usr/bin/env python3

# Import hardware detection capabilities if available:::
try:
    from generators.hardware.hardware_detection import ())))
    HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
    detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback
    """
    Comprehensive test file for distilroberta-base
    - Tests both pipeline())))) and from_pretrained())))) methods
    - Includes CPU, CUDA, and OpenVINO hardware support
    - Handles missing dependencies with sophisticated mocks
    - Supports benchmarking with multiple input sizes
    - Tracks hardware-specific performance metrics
    - Reports detailed dependency information
    """

    import os
    import sys
    import json
    import time
    import datetime
    import traceback
    import logging
    from unittest.mock import patch, MagicMock, Mock
    from typing import Dict, List, Any, Optional, Union

# Configure logging
    logging.basicConfig())))level=logging.INFO, format='%())))asctime)s - %())))levelname)s - %())))message)s')
    logger = logging.getLogger())))__name__)

# Add parent directory to path for imports
    sys.path.insert())))0, os.path.dirname())))os.path.dirname())))os.path.abspath())))__file__))))

# Third-party imports
    import numpy as np

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()))))
    HAS_TORCH = False
    print())))"Warning: torch not available, using mock")

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()))))
    HAS_TRANSFORMERS = False
    print())))"Warning: transformers not available, using mock")

# Additional imports based on model type
if "language" == "vision" or "language" == "multimodal":
    try:
        from PIL import Image
        HAS_PIL = True
    except ImportError:
        Image = MagicMock()))))
        HAS_PIL = False
        print())))"Warning: PIL not available, using mock")

if "language" == "audio":
    try:
        import librosa
        HAS_LIBROSA = True
    except ImportError:
        librosa = MagicMock()))))
        HAS_LIBROSA = False
        print())))"Warning: librosa not available, using mock")


# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()))))
    print())))f"Warning: tokenizers not available, using mock")

# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()))))
    print())))f"Warning: sentencepiece not available, using mock")





# Mock for tokenizers
class MockTokenizer:
    def __init__())))self, *args, **kwargs):
        self.vocab_size = 32000
        
    def encode())))self, text, **kwargs):
        return {}}}}}}}}}}}}}"ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
        ,
    def decode())))self, ids, **kwargs):
        return "Decoded text from mock"
        
        @staticmethod
    def from_file())))vocab_filename):
        return MockTokenizer()))))

if not HAS_TOKENIZERS:
    tokenizers.Tokenizer = MockTokenizer

# Mock for sentencepiece
class MockSentencePieceProcessor:
    def __init__())))self, *args, **kwargs):
        self.vocab_size = 32000
        
    def encode())))self, text, out_type=str):
        return [1, 2, 3, 4, 5]
        ,
    def decode())))self, ids):
        return "Decoded text from mock"
        
    def get_piece_size())))self):
        return 32000
        
        @staticmethod
    def load())))model_file):
        return MockSentencePieceProcessor()))))

if not HAS_SENTENCEPIECE:
    sentencepiece.SentencePieceProcessor = MockSentencePieceProcessor



# Hardware detection
def check_hardware())))):
    """Check available hardware and return capabilities."""
    capabilities = {}}}}}}}}}}}}}
    "cpu": True,
    "cuda": False,
    "cuda_version": None,
    "cuda_devices": 0,
    "mps": False,
    "openvino": False
    }
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available())))),
        if capabilities["cuda"]:,,,
        capabilities["cuda_devices"] = torch.cuda.device_count())))),
        capabilities["cuda_version"] = torch.version.cuda
        ,
    # Check MPS ())))Apple Silicon)
    if HAS_TORCH and hasattr())))torch, "mps") and hasattr())))torch.mps, "is_available"):
        capabilities["mps"] = torch.mps.is_available()))))
        ,
    # Check OpenVINO
    try:
        import openvino
        capabilities["openvino"] = True,
    except ImportError:
        pass
    
        return capabilities

# Get hardware capabilities
        HW_CAPABILITIES = check_hardware()))))


# Check for other required dependencies
        HAS_TOKENIZERS = False
        HAS_SENTENCEPIECE = False


class test_hf_distilroberta_base:
    def __init__())))self):
        # Use appropriate model for testing
        self.model_name = "distilroberta-base"
        
        # Test inputs appropriate for this model type
        
        # Text inputs
        self.test_text = "The quick brown fox jumps over the lazy dog"
        self.test_batch = ["The quick brown fox jumps over the lazy dog", "The five boxing wizards jump quickly"],
        self.test_prompt = "Complete this sentence: The quick brown fox"
        self.test_query = "What is the capital of France?"
        self.test_pairs = [())))"What is the capital of France?", "Paris"), ())))"Who wrote Hamlet?", "Shakespeare")],
        self.test_long_text = """This is a longer piece of text that spans multiple sentences.
        It can be used for summarization, translation, or other text2text tasks.
        The model should be able to process this multi-line input appropriately."""
        
        
        # Results storage
        self.examples = [],
        self.performance_stats = {}}}}}}}}}}}}}}
        
        # Hardware selection for testing ())))prioritize CUDA if available:::)
        if HW_CAPABILITIES["cuda"]:,,,
        self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:,
    self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
            
            logger.info())))f"Using {}}}}}}}}}}}}}self.preferred_device} as preferred device")
        
    def get_input_for_pipeline())))self):
        """Get appropriate input for pipeline testing based on model type."""
            return self.test_text.replace())))'lazy', '[MASK]')
            ,
    def test_pipeline())))self, device="auto"):
        """Test using the transformers pipeline())))) method."""
        results = {}}}}}}}}}}}}}}
        
        if device == "auto":
            device = self.preferred_device
        
            results["device"] = device
            ,,
        if not HAS_TRANSFORMERS:
            results["pipeline_test"] = "Transformers not available",
            results["pipeline_error_type"] = "missing_dependency",,,
            results["pipeline_missing_core"] = ["transformers"],
            return results
            
        # Check required dependencies for this model
            missing_deps = [],
        
        # Check each dependency
        
        if not HAS_TOKENIZERS:
            missing_deps.append())))"tokenizers>=0.11.0")
        
        if not HAS_SENTENCEPIECE:
            missing_deps.append())))"sentencepiece")
        
        
        if missing_deps:
            results["pipeline_missing_deps"] = missing_deps,
            results["pipeline_error_type"] = "missing_dependency",,,
            results["pipeline_test"] = f"Missing dependencies: {}}}}}}}}}}}}}', '.join())))missing_deps)}",
            return results
            
        try:
            logger.info())))f"Testing distilroberta_base with pipeline())))) on {}}}}}}}}}}}}}device}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {}}}}}}}}}}}}}
            "task": "fill-mask",
            "model": self.model_name,
            "trust_remote_code": false,
            "device": device
            }
            
            # Time the model loading separately
            load_start_time = time.time()))))
            pipeline = transformers.pipeline())))**pipeline_kwargs)
            load_time = time.time())))) - load_start_time
            results["pipeline_load_time"] = load_time
            ,
            # Get appropriate input
            pipeline_input = self.get_input_for_pipeline()))))
            
            # Run warmup inference if on CUDA:
            if device == "cuda":
                try:
                    _ = pipeline())))pipeline_input)
                except Exception:
                    pass
            
            # Run multiple inferences for better timing
                    num_runs = 3
                    times = [],
            
            for _ in range())))num_runs):
                start_time = time.time()))))
                output = pipeline())))pipeline_input)
                end_time = time.time()))))
                times.append())))end_time - start_time)
            
            # Calculate statistics
                avg_time = sum())))times) / len())))times)
                min_time = min())))times)
                max_time = max())))times)
            
            # Store results
                results["pipeline_success"] = True,
                results["pipeline_avg_time"] = avg_time,
                results["pipeline_min_time"] = min_time,
                results["pipeline_max_time"] = max_time,
                results["pipeline_times"] = times,
                results["pipeline_uses_remote_code"] = false
                ,
            # Add error type classification for detailed tracking
                results["pipeline_error_type"] = "none"
                ,
            # Store in performance stats
                self.performance_stats[f"pipeline_{}}}}}}}}}}}}}device}"] = {}}}}}}}}}}}}},
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "load_time": load_time,
                "num_runs": num_runs
                }
            
            # Add to examples
                self.examples.append()))){}}}}}}}}}}}}}
                "method": f"pipeline())))) on {}}}}}}}}}}}}}device}",
                "input": str())))pipeline_input),
                "output_type": str())))type())))output)),
                "output": str())))output)[:500] + ())))"..." if str())))output) and len())))str())))output)) > 500 else ""),
                })
            :
        except Exception as e:
            # Store basic error info
            results["pipeline_error"] = str())))e),
            results["pipeline_traceback"] = traceback.format_exc())))),
            logger.error())))f"Error testing pipeline on {}}}}}}}}}}}}}device}: {}}}}}}}}}}}}}e}")
            
            # Classify error type for better diagnostics
            error_str = str())))e).lower()))))
            traceback_str = traceback.format_exc())))).lower()))))
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["pipeline_error_type"] = "cuda_error",
            elif "memory" in error_str or "cuda out of memory" in traceback_str:
                results["pipeline_error_type"] = "out_of_memory",
            elif "trust_remote_code" in error_str:
                results["pipeline_error_type"] = "remote_code_required",
            elif "permission" in error_str or "access" in error_str:
                results["pipeline_error_type"] = "permission_error",
            elif "module" in error_str and "has no attribute" in error_str:
                results["pipeline_error_type"] = "missing_attribute",
            elif "no module named" in error_str.lower())))):
                results["pipeline_error_type"] = "missing_dependency",,,
                # Try to extract the missing module name
                import re
                match = re.search())))r"no module named '())))[^']+)'", error_str.lower()))))),,
                if match:
                    results["pipeline_missing_module"] = match.group())))1),
            else:
                results["pipeline_error_type"] = "other"
                ,
                    return results
        
    def test_from_pretrained())))self, device="auto"):
        """Test using from_pretrained())))) method."""
        results = {}}}}}}}}}}}}}}
        
        if device == "auto":
            device = self.preferred_device
        
            results["device"] = device
            ,,
        if not HAS_TRANSFORMERS:
            results["from_pretrained_test"] = "Transformers not available",
            results["from_pretrained_error_type"] = "missing_dependency",,,
            results["from_pretrained_missing_core"] = ["transformers"],
            return results
            
        # Check required dependencies for this model
            missing_deps = [],
        
        # Check each dependency
        
        if not HAS_TOKENIZERS:
            missing_deps.append())))"tokenizers>=0.11.0")
        
        if not HAS_SENTENCEPIECE:
            missing_deps.append())))"sentencepiece")
        
        
        if missing_deps:
            results["from_pretrained_missing_deps"] = missing_deps,
            results["from_pretrained_error_type"] = "missing_dependency",,,
            results["from_pretrained_test"] = f"Missing dependencies: {}}}}}}}}}}}}}', '.join())))missing_deps)}",
            return results
            
        try:
            logger.info())))f"Testing distilroberta_base with from_pretrained())))) on {}}}}}}}}}}}}}device}...")
            
            # Record remote code requirements
            results["requires_remote_code"] = false,
            if false:
                results["remote_code_reason"] = "Model requires custom code"
                ,
            # Common parameters for loading model components
                pretrained_kwargs = {}}}}}}}}}}}}}
                "trust_remote_code": false,
                "local_files_only": False
                }
            
            # Time tokenizer loading
                tokenizer_load_start = time.time()))))
                tokenizer = transformers.AutoTokenizer.from_pretrained())))
                self.model_name,
                **pretrained_kwargs
                )
                tokenizer_load_time = time.time())))) - tokenizer_load_start
            
            # Time model loading
                model_load_start = time.time()))))
                model = transformers.AutoModelForMaskedLM.from_pretrained())))
                self.model_name,
                **pretrained_kwargs
                )
                model_load_time = time.time())))) - model_load_start
            
            # Move model to device
            if device != "cpu":
                model = model.to())))device)
                
            # Get input based on model category
            if "language" == "language":
                # Tokenize input
                inputs = tokenizer())))self.test_text, return_tensors="pt")
                # Move inputs to device
                if device != "cpu":
                    inputs = {}}}}}}}}}}}}}key: val.to())))device) for key, val in inputs.items()))))}
                
            elif "language" == "vision":
                # Use image inputs
                if hasattr())))self, "test_image_tensor") and self.test_image_tensor is not None:
                    inputs = {}}}}}}}}}}}}}"pixel_values": self.test_image_tensor.unsqueeze())))0)}
                    if device != "cpu":
                        inputs = {}}}}}}}}}}}}}key: val.to())))device) for key, val in inputs.items()))))}
                else:
                    results["from_pretrained_test"] = "Image tensor not available",
                        return results
                    
            elif "language" == "audio":
                # Use audio inputs
                if hasattr())))self, "test_audio_tensor") and self.test_audio_tensor is not None:
                    inputs = {}}}}}}}}}}}}}"input_values": self.test_audio_tensor}
                    if device != "cpu":
                        inputs = {}}}}}}}}}}}}}key: val.to())))device) for key, val in inputs.items()))))}
                else:
                    results["from_pretrained_test"] = "Audio tensor not available",
                        return results
                    
            elif "language" == "multimodal":
                # Use combined inputs based on model
                results["from_pretrained_test"] = "Complex multimodal input not implemented for direct model testing",
                        return results
            else:
                # Default to text input
                inputs = tokenizer())))self.test_text, return_tensors="pt")
                if device != "cpu":
                    inputs = {}}}}}}}}}}}}}key: val.to())))device) for key, val in inputs.items()))))}
            
            # Run warmup inference if using CUDA:
            if device == "cuda":
                try:
                    with torch.no_grad())))):
                        _ = model())))**inputs)
                except Exception:
                        pass
            
            # Run multiple inference passes for better timing
                        num_runs = 3
                        times = [],
            
            for _ in range())))num_runs):
                start_time = time.time()))))
                with torch.no_grad())))):
                    outputs = model())))**inputs)
                    end_time = time.time()))))
                    times.append())))end_time - start_time)
            
            # Calculate statistics
                    avg_time = sum())))times) / len())))times)
                    min_time = min())))times)
                    max_time = max())))times)
            
            # Get model size if possible
            model_size_mb = None:
            try:
                model_size_params = sum())))p.numel())))) for p in model.parameters()))))):
                    model_size_mb = model_size_params * 4 / ())))1024 * 1024)  # Rough estimate in MB
            except Exception:
                    pass
            
            # Store results
                    results["from_pretrained_success"] = True,
                    results["from_pretrained_avg_time"] = avg_time,
                    results["from_pretrained_min_time"] = min_time,
                    results["from_pretrained_max_time"] = max_time,
                    results["from_pretrained_times"] = times,
                    results["tokenizer_load_time"] = tokenizer_load_time,
                    results["model_load_time"] = model_load_time,
                    results["model_size_mb"] = model_size_mb,
                    results["from_pretrained_uses_remote_code"] = false
                    ,
            # Store in performance stats
                    self.performance_stats[f"from_pretrained_{}}}}}}}}}}}}}device}"] = {}}}}}}}}}}}}},
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "tokenizer_load_time": tokenizer_load_time,
                    "model_load_time": model_load_time,
                    "model_size_mb": model_size_mb,
                    "num_runs": num_runs
                    }
            
            # Add to examples
                    self.examples.append()))){}}}}}}}}}}}}}
                    "method": f"from_pretrained())))) on {}}}}}}}}}}}}}device}",
                    "input_keys": str())))list())))inputs.keys())))))),
                    "output_type": str())))type())))outputs)),
                "output_keys": str())))outputs._fields if hasattr())))outputs, "_fields") else list())))outputs.keys()))))) if hasattr())))outputs, "keys") else "N/A"),:
                    "has_logits": hasattr())))outputs, "logits")
                    })
            
        except Exception as e:
            # Store basic error info
            results["from_pretrained_error"] = str())))e),
            results["from_pretrained_traceback"] = traceback.format_exc())))),
            logger.error())))f"Error testing from_pretrained on {}}}}}}}}}}}}}device}: {}}}}}}}}}}}}}e}")
            
            # Classify error type for better diagnostics
            error_str = str())))e).lower()))))
            traceback_str = traceback.format_exc())))).lower()))))
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["from_pretrained_error_type"] = "cuda_error",
            elif "memory" in error_str or "cuda out of memory" in traceback_str:
                results["from_pretrained_error_type"] = "out_of_memory",
            elif "trust_remote_code" in error_str:
                results["from_pretrained_error_type"] = "remote_code_required",
            elif "permission" in error_str or "access" in error_str:
                results["from_pretrained_error_type"] = "permission_error",
            elif "module" in error_str and "has no attribute" in error_str:
                results["from_pretrained_error_type"] = "missing_attribute",
            elif "no module named" in error_str.lower())))):
                results["from_pretrained_error_type"] = "missing_dependency",,,
                # Try to extract the missing module name
                import re
                match = re.search())))r"no module named '())))[^']+)'", error_str.lower()))))),,
                if match:
                    results["from_pretrained_missing_module"] = match.group())))1),
            elif "could not find model" in error_str or "404" in error_str:
                results["from_pretrained_error_type"] = "model_not_found",
            else:
                results["from_pretrained_error_type"] = "other"
                ,
                return results
        
    def test_with_openvino())))self):
        """Test model with OpenVINO if available:::."""
        results = {}}}}}}}}}}}}}}
        
        if not HW_CAPABILITIES["openvino"]:,,
        results["openvino_test"] = "OpenVINO not available",
                return results
            
        try:
            from optimum.intel import OVModelForSequenceClassification, OVModelForCausalLM
            
            # Load the model with OpenVINO
            logger.info())))f"Testing distilroberta_base with OpenVINO...")
            
            # Determine which OV model class to use based on task
            if "fill-mask" == "text-generation":
                ov_model_class = OVModelForCausalLM
            else:
                ov_model_class = OVModelForSequenceClassification
            
            # Load tokenizer
                tokenizer = transformers.AutoTokenizer.from_pretrained())))self.model_name)
            
            # Load model with OpenVINO
                load_start_time = time.time()))))
                model = ov_model_class.from_pretrained())))
                self.model_name,
                export=True,
                trust_remote_code=false
                )
                load_time = time.time())))) - load_start_time
            
            # Tokenize input
                inputs = tokenizer())))self.test_text, return_tensors="pt")
            
            # Run inference
                start_time = time.time()))))
                outputs = model())))**inputs)
                inference_time = time.time())))) - start_time
            
            # Store results
                results["openvino_success"] = True,
                results["openvino_load_time"] = load_time,
                results["openvino_inference_time"] = inference_time
                ,
            # Store in performance stats
                self.performance_stats["openvino"] = {}}}}}}}}}}}}},
                "load_time": load_time,
                "inference_time": inference_time
                }
            
            # Add to examples
                self.examples.append()))){}}}}}}}}}}}}}
                "method": "OpenVINO inference",
                "input": self.test_text,
                "output_type": str())))type())))outputs)),
                "has_logits": hasattr())))outputs, "logits")
                })
            
        except Exception as e:
            results["openvino_error"] = str())))e),
            results["openvino_traceback"] = traceback.format_exc())))),
            logger.error())))f"Error testing with OpenVINO: {}}}}}}}}}}}}}e}")
            
                return results
        
    def run_all_hardware_tests())))self):
        """Run tests on all available hardware."""
        all_results = {}}}}}}}}}}}}}}
        
        # Always run CPU tests
        cpu_pipeline_results = self.test_pipeline())))device="cpu")
        all_results["cpu_pipeline"] = cpu_pipeline_results
        ,
        cpu_pretrained_results = self.test_from_pretrained())))device="cpu")
        all_results["cpu_pretrained"] = cpu_pretrained_results
        ,
        # Run CUDA tests if available:::
        if HW_CAPABILITIES["cuda"]:,,,
        cuda_pipeline_results = self.test_pipeline())))device="cuda")
        all_results["cuda_pipeline"] = cuda_pipeline_results
        ,
        cuda_pretrained_results = self.test_from_pretrained())))device="cuda")
        all_results["cuda_pretrained"] = cuda_pretrained_results
        ,
        # Run OpenVINO tests if available:::
        if HW_CAPABILITIES["openvino"]:,,
        openvino_results = self.test_with_openvino()))))
        all_results["openvino"] = openvino_results
        ,
                return all_results
        
    def run_tests())))self):
        """Run all tests and return results."""
        # Collect hardware capabilities
        hw_info = {}}}}}}}}}}}}}
        "capabilities": HW_CAPABILITIES,
        "preferred_device": self.preferred_device
        }
        
        # Run tests on preferred device
        pipeline_results = self.test_pipeline()))))
        pretrained_results = self.test_from_pretrained()))))
        
        # Build dependency information
        dependency_status = {}}}}}}}}}}}}}}
        
        # Check each dependency
        
        dependency_status["tokenizers>=0.11.0"] = HAS_TOKENIZERS
        ,
        dependency_status["sentencepiece"] = HAS_SENTENCEPIECE
        
        ,
        # Run all hardware tests if --all-hardware flag is provided
        all_hardware_results = None:
        if "--all-hardware" in sys.argv:
            all_hardware_results = self.run_all_hardware_tests()))))
        
            return {}}}}}}}}}}}}}
            "results": {}}}}}}}}}}}}}
            "pipeline": pipeline_results,
            "from_pretrained": pretrained_results,
            "all_hardware": all_hardware_results
            },
            "examples": self.examples,
            "performance": self.performance_stats,
            "hardware": hw_info,
            "metadata": {}}}}}}}}}}}}}
            "model": self.model_name,
            "category": "language",
            "task": "fill-mask",
            "timestamp": datetime.datetime.now())))).isoformat())))),
            "generation_timestamp": "2025-03-01 16:47:46",
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH,
            "dependencies": dependency_status,
            "uses_remote_code": False
            }
            }
        
if __name__ == "__main__":
    logger.info())))f"Running tests for distilroberta_base...")
    tester = test_hf_distilroberta_base()))))
    test_results = tester.run_tests()))))
    
    # Save results to file if --save flag is provided:
    if "--save" in sys.argv:
        output_dir = "collected_results"
        os.makedirs())))output_dir, exist_ok=True)
        output_file = os.path.join())))output_dir, f"hf_distilroberta_base_test_results.json")
        with open())))output_file, "w") as f:
            json.dump())))test_results, f, indent=2)
            logger.info())))f"Saved results to {}}}}}}}}}}}}}output_file}")
    
    # Print summary results
            print())))"\nTEST RESULTS SUMMARY:")
            if test_results["results"]["pipeline"].get())))"pipeline_success", False):,
            pipeline_time = test_results["results"]["pipeline"].get())))"pipeline_avg_time", 0),
            print())))f"✅ Pipeline test successful ()))){}}}}}}}}}}}}}pipeline_time:.4f}s)")
    else:
        error = test_results["results"]["pipeline"].get())))"pipeline_error", "Unknown error"),
        print())))f"❌ Pipeline test failed: {}}}}}}}}}}}}}error}")
        
        if test_results["results"]["from_pretrained"].get())))"from_pretrained_success", False):,
        model_time = test_results["results"]["from_pretrained"].get())))"from_pretrained_avg_time", 0),
        print())))f"✅ from_pretrained test successful ()))){}}}}}}}}}}}}}model_time:.4f}s)")
    else:
        error = test_results["results"]["from_pretrained"].get())))"from_pretrained_error", "Unknown error"),
        print())))f"❌ from_pretrained test failed: {}}}}}}}}}}}}}error}")
        
    # Show top 3 examples
        if test_results["examples"]:,
        print())))"\nEXAMPLES:")
        for i, example in enumerate())))test_results["examples"][:2]):,
        print())))f"Example {}}}}}}}}}}}}}i+1}: {}}}}}}}}}}}}}example['method']}"),
            if "input" in example:
                print())))f"  Input: {}}}}}}}}}}}}}example['input']}"),
            if "output_type" in example:
                print())))f"  Output type: {}}}}}}}}}}}}}example['output_type']}")
                ,
                print())))"\nFor detailed results, use --save flag and check the JSON output file.")