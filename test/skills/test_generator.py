#!/usr/bin/env python3
"""
Hugging Face Test Generator

This module provides a comprehensive framework for generating test files that cover
all Hugging Face model architectures, with support for:
- Multiple hardware backends (CPU, CUDA, OpenVINO)
- Both from_pretrained() and pipeline() API approaches
- Consistent performance benchmarking and result collection

Usage:
  python test_generator.py --all-families
  python test_generator.py --family bert
  python test_generator.py --generate-template t5
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = CURRENT_DIR.parent
RESULTS_DIR = CURRENT_DIR / "collected_results"
EXPECTED_DIR = CURRENT_DIR / "expected_results"
TEMPLATES_DIR = CURRENT_DIR / "templates"

# Model Registry - Maps model families to their configurations
MODEL_REGISTRY = {
    "bert": {
        "family_name": "BERT",
        "description": "BERT-family masked language models",
        "default_model": "bert-base-uncased",
        "class": "BertForMaskedLM",
        "test_class": "TestBertModels",
        "module_name": "test_hf_bert",
        "tasks": ["fill-mask"],
        "inputs": {
            "text": "The quick brown fox jumps over the [MASK] dog."
        },
        "dependencies": ["transformers", "tokenizers", "sentencepiece"],
        "task_specific_args": {
            "fill-mask": {"top_k": 5}
        },
        "models": {
            "bert-base-uncased": {
                "description": "BERT base model (uncased)",
                "class": "BertForMaskedLM",
                "vocab_size": 30522
            },
            "distilbert-base-uncased": {
                "description": "DistilBERT base model (uncased)",
                "class": "DistilBertForMaskedLM",
                "vocab_size": 30522
            },
            "roberta-base": {
                "description": "RoBERTa base model",
                "class": "RobertaForMaskedLM",
                "vocab_size": 50265
            }
        }
    },
    "gpt2": {
        "family_name": "GPT-2",
        "description": "GPT-2 causal language models",
        "default_model": "gpt2",
        "class": "GPT2LMHeadModel",
        "test_class": "TestGpt2Models",
        "module_name": "test_hf_gpt2",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "Once upon a time"
        },
        "dependencies": ["transformers", "tokenizers"],
        "task_specific_args": {
            "text-generation": {"max_length": 50, "min_length": 20}
        },
        "models": {
            "gpt2": {
                "description": "GPT-2 small model",
                "class": "GPT2LMHeadModel"
            },
            "gpt2-medium": {
                "description": "GPT-2 medium model",
                "class": "GPT2LMHeadModel"
            },
            "distilgpt2": {
                "description": "DistilGPT-2 model",
                "class": "GPT2LMHeadModel"
            }
        }
    },
    "t5": {
        "family_name": "T5",
        "description": "T5 encoder-decoder models",
        "default_model": "t5-small",
        "class": "T5ForConditionalGeneration",
        "test_class": "TestT5Models",
        "module_name": "test_hf_t5",
        "tasks": ["translation_en_to_fr", "summarization"],
        "inputs": {
            "translation_en_to_fr": "My name is Sarah and I live in London",
            "summarization": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. It was the first structure to reach a height of 300 metres."
        },
        "dependencies": ["transformers", "sentencepiece"],
        "task_specific_args": {
            "translation_en_to_fr": {"max_length": 40},
            "summarization": {"max_length": 100, "min_length": 30}
        },
        "models": {
            "t5-small": {
                "description": "T5 small model",
                "class": "T5ForConditionalGeneration"
            },
            "t5-base": {
                "description": "T5 base model",
                "class": "T5ForConditionalGeneration"
            },
            "google/flan-t5-small": {
                "description": "Flan-T5 small model",
                "class": "T5ForConditionalGeneration"
            }
        }
    },
    "clip": {
        "family_name": "CLIP",
        "description": "CLIP vision-language models",
        "default_model": "openai/clip-vit-base-patch32",
        "class": "CLIPModel",
        "test_class": "TestClipModels",
        "module_name": "test_hf_clip",
        "tasks": ["zero-shot-image-classification"],
        "inputs": {
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "candidate_labels": ["a photo of a cat", "a photo of a dog"]
        },
        "dependencies": ["transformers", "pillow", "requests"],
        "task_specific_args": {
            "zero-shot-image-classification": {}
        },
        "models": {
            "openai/clip-vit-base-patch32": {
                "description": "CLIP ViT-Base-Patch32 model",
                "class": "CLIPModel"
            },
            "openai/clip-vit-base-patch16": {
                "description": "CLIP ViT-Base-Patch16 model",
                "class": "CLIPModel"
            }
        }
    },
    "llama": {
        "family_name": "LLaMA",
        "description": "LLaMA causal language models",
        "default_model": "meta-llama/Llama-2-7b-hf",
        "class": "LlamaForCausalLM",
        "test_class": "TestLlamaModels",
        "module_name": "test_hf_llama",
        "tasks": ["text-generation"],
        "inputs": {
            "text": "In this paper, we propose"
        },
        "dependencies": ["transformers", "tokenizers", "accelerate"],
        "task_specific_args": {
            "text-generation": {"max_length": 50, "min_length": 20}
        },
        "models": {
            "meta-llama/Llama-2-7b-hf": {
                "description": "LLaMA 2 7B model",
                "class": "LlamaForCausalLM"
            },
            "meta-llama/Llama-2-7b-chat-hf": {
                "description": "LLaMA 2 7B chat model",
                "class": "LlamaForCausalLM"
            }
        }
    },
    "whisper": {
        "family_name": "Whisper",
        "description": "Whisper speech recognition models",
        "default_model": "openai/whisper-tiny",
        "class": "WhisperForConditionalGeneration",
        "test_class": "TestWhisperModels",
        "module_name": "test_hf_whisper",
        "tasks": ["automatic-speech-recognition"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "automatic-speech-recognition": {"max_length": 448}
        },
        "models": {
            "openai/whisper-tiny": {
                "description": "Whisper tiny model",
                "class": "WhisperForConditionalGeneration"
            },
            "openai/whisper-base": {
                "description": "Whisper base model",
                "class": "WhisperForConditionalGeneration"
            }
        }
    },
    "wav2vec2": {
        "family_name": "Wav2Vec2",
        "description": "Wav2Vec2 speech models",
        "default_model": "facebook/wav2vec2-base",
        "class": "Wav2Vec2ForCTC",
        "test_class": "TestWav2Vec2Models",
        "module_name": "test_hf_wav2vec2",
        "tasks": ["automatic-speech-recognition"],
        "inputs": {
            "audio_file": "audio_sample.mp3"
        },
        "dependencies": ["transformers", "librosa", "soundfile"],
        "task_specific_args": {
            "automatic-speech-recognition": {}
        },
        "models": {
            "facebook/wav2vec2-base": {
                "description": "Wav2Vec2 base model",
                "class": "Wav2Vec2ForCTC"
            },
            "facebook/wav2vec2-large": {
                "description": "Wav2Vec2 large model",
                "class": "Wav2Vec2ForCTC"
            }
        }
    }
}

# Base template strings
BASE_TEST_FILE_TEMPLATE = """#!/usr/bin/env python3
\"\"\"
Class-based test file for all {family_name}-family models.
This file provides a unified testing interface for:
{model_class_comments}
\"\"\"

import os
import sys
import json
import time
import datetime
import traceback
import logging
import argparse
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

{dependency_imports}

# Hardware detection
def check_hardware():
    \"\"\"Check available hardware and return capabilities.\"\"\"
    capabilities = {{
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False
    }}
    
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
{models_registry}

class {test_class_name}:
    \"\"\"Base test class for all {family_name}-family models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class for a specific model or default.\"\"\"
        self.model_id = model_id or "{default_model}"
        
        # Verify model exists in registry
        if self.model_id not in {registry_name}:
            logger.warning(f"Model {{self.model_id}} not in registry, using default configuration")
            self.model_info = {registry_name}["{default_model}"]
        else:
            self.model_info = {registry_name}[self.model_id]
        
        # Define model parameters
        self.task = "{default_task}"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        {test_inputs}
        
        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}
    
    {test_pipeline_method}
    
    {test_from_pretrained_method}
    
    {test_openvino_method}
    
    def run_tests(self, all_hardware=False):
        \"\"\"
        Run all tests for this model.
        
        Args:
            all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
        
        Returns:
            Dict containing test results
        \"\"\"
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
        
        # Build final results
        return {{
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "hardware": HW_CAPABILITIES,
            "metadata": {{
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                {has_dependencies}
            }}
        }}

def save_results(model_id, results, output_dir="collected_results"):
    \"\"\"Save test results to a file.\"\"\"
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_{family_lower}_{{safe_model_id}}_{{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {{output_path}}")
    return output_path

def get_available_models():
    \"\"\"Get a list of all available {family_name} models in the registry.\"\"\"
    return list({registry_name}.keys())

def test_all_models(output_dir="collected_results", all_hardware=False):
    \"\"\"Test all registered {family_name} models.\"\"\"
    models = get_available_models()
    results = {{}}
    
    for model_id in models:
        logger.info(f"Testing model: {{model_id}}")
        tester = {test_class_name}(model_id)
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save individual results
        save_results(model_id, model_results, output_dir=output_dir)
        
        # Add to summary
        results[model_id] = {{
            "success": any(r.get("pipeline_success", False) for r in model_results["results"].values() 
                          if r.get("pipeline_success") is not False)
        }}
    
    # Save summary
    summary_path = os.path.join(output_dir, f"hf_{family_lower}_summary_{{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {{summary_path}}")
    return results

def main():
    \"\"\"Command-line entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test {family_name}-family models")
    
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
        print("\\nAvailable {family_name}-family models:")
        for model in models:
            info = {registry_name}[model]
            print(f"  - {{model}} ({{info['class']}}): {{info['description']}}")
        return
    
    # Create output directory if needed
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test all models if requested
    if args.all_models:
        results = test_all_models(output_dir=args.output_dir, all_hardware=args.all_hardware)
        
        # Print summary
        print("\\n{family_name} Models Testing Summary:")
        total = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        print(f"Successfully tested {{successful}} of {{total}} models ({{successful/total*100:.1f}}%)")
        return
    
    # Test single model (default or specified)
    model_id = args.model or "{default_model}"
    logger.info(f"Testing model: {{model_id}}")
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run test
    tester = {test_class_name}(model_id)
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values()
                  if r.get("pipeline_success") is not False)
    
    print("\\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {{model_id}}")
        
        # Print performance highlights
        for device, stats in results["performance"].items():
            if "avg_time" in stats:
                print(f"  - {{device}}: {{stats['avg_time']:.4f}}s average inference time")
        
        # Print example outputs if available
        if results.get("examples") and len(results["examples"]) > 0:
            print("\\nExample output:")
            example = results["examples"][0]
            if "predictions" in example:
                print(f"  Input: {{example['input']}}")
                print(f"  Predictions: {{example['predictions']}}")
            elif "output_preview" in example:
                print(f"  Input: {{example['input']}}")
                print(f"  Output: {{example['output_preview']}}")
    else:
        print(f"❌ Failed to test {{model_id}}")
        
        # Print error information
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {{test_name}}: {{result.get('pipeline_error_type', 'unknown')}}")
                print(f"    {{result.get('pipeline_error', 'Unknown error')}}")
    
    print("\\nFor detailed results, use --save flag and check the JSON output file.")

if __name__ == "__main__":
    main()
"""

# Template for pipeline test method
PIPELINE_TEST_TEMPLATE = """
def test_pipeline(self, device="auto"):
    \"\"\"Test the model using transformers pipeline API.\"\"\"
    if device == "auto":
        device = self.preferred_device
    
    results = {{
        "model": self.model_id,
        "device": device,
        "task": self.task,
        "class": self.class_name
    }}
    
    # Check for dependencies
    if not HAS_TRANSFORMERS:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_core"] = ["transformers"]
        results["pipeline_success"] = False
        return results
        
    {pipeline_dependency_checks}
    
    try:
        logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
        
        # Create pipeline with appropriate parameters
        pipeline_kwargs = {{
            "task": self.task,
            "model": self.model_id,
            "device": device
        }}
        
        # Time the model loading
        load_start_time = time.time()
        pipeline = transformers.pipeline(**pipeline_kwargs)
        load_time = time.time() - load_start_time
        
        # Prepare test input
        {pipeline_input_preparation}
        
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
        self.examples.append({{
            "method": f"pipeline() on {{device}}",
            "input": str(pipeline_input),
            "output_preview": str(outputs[0])[:200] + "..." if len(str(outputs[0])) > 200 else str(outputs[0])
        }})
        
        # Store in performance stats
        self.performance_stats[f"pipeline_{{device}}"] = {{
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "load_time": load_time,
            "num_runs": num_runs
        }}
        
    except Exception as e:
        # Store error information
        results["pipeline_success"] = False
        results["pipeline_error"] = str(e)
        results["pipeline_traceback"] = traceback.format_exc()
        logger.error(f"Error testing pipeline on {{device}}: {{e}}")
        
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
    self.results[f"pipeline_{{device}}"] = results
    return results
"""

# Template for from_pretrained test method
FROM_PRETRAINED_TEMPLATE = """
def test_from_pretrained(self, device="auto"):
    \"\"\"Test the model using direct from_pretrained loading.\"\"\"
    if device == "auto":
        device = self.preferred_device
    
    results = {{
        "model": self.model_id,
        "device": device,
        "task": self.task,
        "class": self.class_name
    }}
    
    # Check for dependencies
    if not HAS_TRANSFORMERS:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_core"] = ["transformers"]
        results["from_pretrained_success"] = False
        return results
        
    {from_pretrained_dependency_checks}
    
    try:
        logger.info(f"Testing {{self.model_id}} with from_pretrained() on {{device}}...")
        
        # Common parameters for loading
        pretrained_kwargs = {{
            "local_files_only": False
        }}
        
        # Time tokenizer loading
        tokenizer_load_start = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            **pretrained_kwargs
        )
        tokenizer_load_time = time.time() - tokenizer_load_start
        
        # Use appropriate model class based on model type
        model_class = None
        if self.class_name == "{class_name}":
            model_class = transformers.{class_name}
        else:
            # Fallback to Auto class
            model_class = transformers.{auto_model_class}
        
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
        
        {from_pretrained_input_preparation}
        
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
        
        {from_pretrained_output_processing}
        
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
        example_data = {{
            "method": f"from_pretrained() on {{device}}",
            "input": str(test_input)
        }}
        
        if 'predictions' in locals():
            example_data["predictions"] = predictions
        
        self.examples.append(example_data)
        
        # Store in performance stats
        self.performance_stats[f"from_pretrained_{{device}}"] = {{
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "tokenizer_load_time": tokenizer_load_time,
            "model_load_time": model_load_time,
            "model_size_mb": model_size_mb,
            "num_runs": num_runs
        }}
        
    except Exception as e:
        # Store error information
        results["from_pretrained_success"] = False
        results["from_pretrained_error"] = str(e)
        results["from_pretrained_traceback"] = traceback.format_exc()
        logger.error(f"Error testing from_pretrained on {{device}}: {{e}}")
        
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
    self.results[f"from_pretrained_{{device}}"] = results
    return results
"""

# Template for OpenVINO test method
OPENVINO_TEST_TEMPLATE = """
def test_with_openvino(self):
    \"\"\"Test the model using OpenVINO integration.\"\"\"
    results = {{
        "model": self.model_id,
        "task": self.task,
        "class": self.class_name
    }}
    
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
    
    try:
        from optimum.intel import {openvino_model_class}
        logger.info(f"Testing {{self.model_id}} with OpenVINO...")
        
        # Time tokenizer loading
        tokenizer_load_start = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        tokenizer_load_time = time.time() - tokenizer_load_start
        
        # Time model loading
        model_load_start = time.time()
        model = {openvino_model_class}.from_pretrained(
            self.model_id,
            export=True,
            provider="CPU"
        )
        model_load_time = time.time() - model_load_start
        
        {openvino_input_preparation}
        
        # Run inference
        start_time = time.time()
        outputs = model(**inputs)
        inference_time = time.time() - start_time
        
        {openvino_output_processing}
        
        # Store results
        results["openvino_success"] = True
        results["openvino_load_time"] = model_load_time
        results["openvino_inference_time"] = inference_time
        results["openvino_tokenizer_load_time"] = tokenizer_load_time
        
        # Add predictions if available
        if 'predictions' in locals():
            results["openvino_predictions"] = predictions
        
        results["openvino_error_type"] = "none"
        
        # Add to examples
        example_data = {{
            "method": "OpenVINO inference",
            "input": str(test_input)
        }}
        
        if 'predictions' in locals():
            example_data["predictions"] = predictions
        
        self.examples.append(example_data)
        
        # Store in performance stats
        self.performance_stats["openvino"] = {{
            "inference_time": inference_time,
            "load_time": model_load_time,
            "tokenizer_load_time": tokenizer_load_time
        }}
        
    except Exception as e:
        # Store error information
        results["openvino_success"] = False
        results["openvino_error"] = str(e)
        results["openvino_traceback"] = traceback.format_exc()
        logger.error(f"Error testing with OpenVINO: {{e}}")
        
        # Classify error
        error_str = str(e).lower()
        if "no module named" in error_str:
            results["openvino_error_type"] = "missing_dependency"
        else:
            results["openvino_error_type"] = "other"
    
    # Add to overall results
    self.results["openvino"] = results
    return results
"""

def generate_model_class_comments(family_info):
    """Generate comments for model classes in this family."""
    models = family_info.get("models", {})
    classes = set()
    for model_info in models.values():
        if "class" in model_info:
            classes.add(model_info["class"])
    
    result = []
    for cls in classes:
        result.append(f"- {cls}")
    
    # Add default if empty
    if not result:
        result.append(f"- {family_info['class']}")
    
    return "\n".join(result)

def generate_dependency_imports(family_info):
    """Generate import statements for family-specific dependencies."""
    dependencies = family_info.get("dependencies", [])
    result = []
    
    # Add standard imports for common dependencies
    for dep in dependencies:
        if dep == "tokenizers":
            result.append("""
# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")
""")
        elif dep == "sentencepiece":
            result.append("""
# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")
""")
        elif dep == "pillow":
            result.append("""
# Try to import PIL
try:
    from PIL import Image
    import requests
    from io import BytesIO
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    requests = MagicMock()
    BytesIO = MagicMock()
    HAS_PIL = False
    logger.warning("PIL or requests not available, using mock")
""")
        elif dep == "librosa":
            result.append("""
# Try to import audio processing libraries
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    librosa = MagicMock()
    sf = MagicMock()
    HAS_AUDIO = False
    logger.warning("librosa or soundfile not available, using mock")
""")
        elif dep == "accelerate":
            result.append("""
# Try to import accelerate
try:
    import accelerate
    HAS_ACCELERATE = True
except ImportError:
    accelerate = MagicMock()
    HAS_ACCELERATE = False
    logger.warning("accelerate not available, using mock")
""")
    
    # Add mock implementations for dependencies
    if "tokenizers" in dependencies:
        result.append("""
# Mock implementations for missing dependencies
if not HAS_TOKENIZERS:
    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            self.vocab_size = 32000
            
        def encode(self, text, **kwargs):
            return {"ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
            
        def decode(self, ids, **kwargs):
            return "Decoded text from mock"
            
        @staticmethod
        def from_file(vocab_filename):
            return MockTokenizer()

    tokenizers.Tokenizer = MockTokenizer
""")
    
    if "sentencepiece" in dependencies:
        result.append("""
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
""")
    
    if "pillow" in dependencies:
        result.append("""
if not HAS_PIL:
    class MockImage:
        @staticmethod
        def open(file):
            class MockImg:
                def __init__(self):
                    self.size = (224, 224)
                def convert(self, mode):
                    return self
                def resize(self, size):
                    return self
            return MockImg()
            
    class MockRequests:
        @staticmethod
        def get(url):
            class MockResponse:
                def __init__(self):
                    self.content = b"mock image data"
                def raise_for_status(self):
                    pass
            return MockResponse()

    Image.open = MockImage.open
    requests.get = MockRequests.get
""")
    
    if "librosa" in dependencies:
        result.append("""
if not HAS_AUDIO:
    def mock_load(file_path, sr=None, mono=True):
        return (np.zeros(16000), 16000)
        
    class MockSoundFile:
        @staticmethod
        def write(file, data, samplerate):
            pass
    
    if isinstance(librosa, MagicMock):
        librosa.load = mock_load
    
    if isinstance(sf, MagicMock):
        sf.write = MockSoundFile.write
""")
    
    return "\n".join(result)

def generate_models_registry(family_info):
    """Generate the models registry dictionary for a family."""
    family_name = family_info["family_name"].upper()
    models = family_info.get("models", {})
    
    registry_lines = [f"{family_name}_MODELS_REGISTRY = {{"]
    
    for model_id, model_info in models.items():
        registry_lines.append(f'    "{model_id}": {{')
        for key, value in model_info.items():
            if isinstance(value, str):
                registry_lines.append(f'        "{key}": "{value}",')
            else:
                registry_lines.append(f'        "{key}": {value},')
        registry_lines.append('    },')
    
    registry_lines.append("}")
    
    return "\n".join(registry_lines)

def generate_test_inputs(family_info):
    """Generate test input initialization."""
    task = family_info.get("tasks", ["text-generation"])[0]
    inputs = family_info.get("inputs", {})
    
    lines = []
    
    if "text" in inputs:
        lines.append(f'self.test_text = "{inputs["text"]}"')
        lines.append('self.test_texts = [')
        lines.append(f'    "{inputs["text"]}",')
        lines.append(f'    "{inputs["text"]} (alternative)"')
        lines.append(']')
    
    if "image_url" in inputs:
        lines.append(f'self.test_image_url = "{inputs["image_url"]}"')
        if "candidate_labels" in inputs:
            labels = inputs["candidate_labels"]
            lines.append('self.candidate_labels = [')
            for label in labels:
                lines.append(f'    "{label}",')
            lines.append(']')
    
    if "audio_file" in inputs:
        lines.append(f'self.test_audio = "{inputs["audio_file"]}"')
    
    return "\n        ".join(lines)

def generate_pipeline_dependency_checks(family_info):
    """Generate dependency checks for pipeline test."""
    dependencies = family_info.get("dependencies", [])
    checks = []
    
    for dep in dependencies:
        if dep == "tokenizers":
            checks.append("""if not HAS_TOKENIZERS:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["tokenizers>=0.11.0"]
        results["pipeline_success"] = False
        return results""")
        elif dep == "sentencepiece":
            checks.append("""if not HAS_SENTENCEPIECE:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["sentencepiece>=0.1.91"]
        results["pipeline_success"] = False
        return results""")
        elif dep == "pillow":
            checks.append("""if not HAS_PIL:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
        results["pipeline_success"] = False
        return results""")
        elif dep == "librosa":
            checks.append("""if not HAS_AUDIO:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["librosa>=0.8.0", "soundfile>=0.10.0"]
        results["pipeline_success"] = False
        return results""")
        elif dep == "accelerate":
            checks.append("""if not HAS_ACCELERATE:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_deps"] = ["accelerate>=0.12.0"]
        results["pipeline_success"] = False
        return results""")
    
    return "\n    ".join(checks)

def generate_from_pretrained_dependency_checks(family_info):
    """Generate dependency checks for from_pretrained test."""
    dependencies = family_info.get("dependencies", [])
    checks = []
    
    for dep in dependencies:
        if dep == "tokenizers":
            checks.append("""if not HAS_TOKENIZERS:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["tokenizers>=0.11.0"]
        results["from_pretrained_success"] = False
        return results""")
        elif dep == "sentencepiece":
            checks.append("""if not HAS_SENTENCEPIECE:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["sentencepiece>=0.1.91"]
        results["from_pretrained_success"] = False
        return results""")
        elif dep == "pillow":
            checks.append("""if not HAS_PIL:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
        results["from_pretrained_success"] = False
        return results""")
        elif dep == "librosa":
            checks.append("""if not HAS_AUDIO:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["librosa>=0.8.0", "soundfile>=0.10.0"]
        results["from_pretrained_success"] = False
        return results""")
        elif dep == "accelerate":
            checks.append("""if not HAS_ACCELERATE:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_deps"] = ["accelerate>=0.12.0"]
        results["from_pretrained_success"] = False
        return results""")
    
    return "\n    ".join(checks)

def generate_pipeline_input_preparation(family_info):
    """Generate code to prepare input for pipeline."""
    task = family_info.get("tasks", ["text-generation"])[0]
    inputs = family_info.get("inputs", {})
    
    if "text" in inputs:
        return "pipeline_input = self.test_text"
    elif "image_url" in inputs:
        return """if HAS_PIL:
            pipeline_input = requests.get(self.test_image_url).content
        else:
            pipeline_input = self.test_image_url"""
    elif "audio_file" in inputs:
        return """if os.path.exists(self.test_audio):
            pipeline_input = self.test_audio
        else:
            # Use a sample array if file not found
            pipeline_input = np.zeros(16000)"""
    else:
        return "pipeline_input = None  # Default empty input"

def generate_from_pretrained_input_preparation(family_info):
    """Generate code to prepare input for from_pretrained."""
    task = family_info.get("tasks", ["text-generation"])[0]
    inputs = family_info.get("inputs", {})
    
    if task in ["text-generation", "fill-mask", "translation_en_to_fr", "summarization"]:
        return """# Prepare test input
        test_input = self.test_text
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # Move inputs to device
        if device != "cpu":
            inputs = {key: val.to(device) for key, val in inputs.items()}"""
    elif task in ["zero-shot-image-classification"]:
        return """# Prepare test input
        test_input = self.test_image_url
        
        # Get image
        if HAS_PIL:
            response = requests.get(test_input)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Mock image
            image = None
            
        # Get text features
        inputs = tokenizer(self.candidate_labels, padding=True, return_tensors="pt")
        
        if HAS_PIL:
            # Get image features
            processor = transformers.AutoProcessor.from_pretrained(self.model_id)
            image_inputs = processor(images=image, return_tensors="pt")
            inputs.update(image_inputs)
        
        # Move inputs to device
        if device != "cpu":
            inputs = {key: val.to(device) for key, val in inputs.items()}"""
    elif task in ["automatic-speech-recognition"]:
        return """# Prepare test input
        test_input = self.test_audio
        
        # Load audio
        if HAS_AUDIO and os.path.exists(test_input):
            waveform, sample_rate = librosa.load(test_input, sr=16000)
            inputs = {"input_values": torch.tensor([waveform]).float()}
        else:
            # Mock audio input
            inputs = {"input_values": torch.zeros(1, 16000).float()}
            
        # Move inputs to device
        if device != "cpu":
            inputs = {key: val.to(device) for key, val in inputs.items()}"""
    else:
        # Generic fallback
        return """# Prepare test input
        test_input = "Generic input for testing"
        
        # Create generic inputs
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        
        # Move inputs to device
        if device != "cpu":
            inputs = {key: val.to(device) for key, val in inputs.items()}"""

def generate_from_pretrained_output_processing(family_info):
    """Generate code to process output for from_pretrained."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task == "fill-mask":
        return """# Get top predictions for masked position
        if hasattr(tokenizer, "mask_token_id"):
            mask_token_id = tokenizer.mask_token_id
            mask_positions = (inputs["input_ids"] == mask_token_id).nonzero()
            
            if len(mask_positions) > 0:
                mask_index = mask_positions[0][-1].item()
                logits = outputs[0].logits[0, mask_index]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                top_k = torch.topk(probs, 5)
                
                predictions = []
                for i, (prob, idx) in enumerate(zip(top_k.values, top_k.indices)):
                    if hasattr(tokenizer, "convert_ids_to_tokens"):
                        token = tokenizer.convert_ids_to_tokens(idx.item())
                    else:
                        token = f"token_{idx.item()}"
                    predictions.append({
                        "token": token,
                        "probability": prob.item()
                    })
            else:
                predictions = []
        else:
            predictions = []"""
    elif task == "text-generation":
        return """# Process generation output
        predictions = outputs[0]
        if hasattr(tokenizer, "decode"):
            if hasattr(outputs[0], "logits"):
                logits = outputs[0].logits
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                next_token = tokenizer.decode([next_token_id])
                predictions = [{"token": next_token, "score": 1.0}]
            else:
                predictions = [{"generated_text": "Mock generated text"}]"""
    elif task in ["translation_en_to_fr", "summarization"]:
        return """# Process generation output
        if hasattr(outputs[0], "logits"):
            logits = outputs[0].logits
            generated_ids = torch.argmax(logits, dim=-1)
            if hasattr(tokenizer, "decode"):
                decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                predictions = [{"generated_text": decoded_output}]
            else:
                predictions = [{"generated_text": "Mock generated text"}]
        else:
            predictions = [{"generated_text": "Mock generated text"}]"""
    elif task == "zero-shot-image-classification":
        return """# Process classification output
        if hasattr(outputs, "logits_per_image"):
            logits = outputs.logits_per_image[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = []
            for i, (label, prob) in enumerate(zip(self.candidate_labels, probs)):
                predictions.append({
                    "label": label,
                    "score": prob.item()
                })
        else:
            predictions = [{"label": "Mock label", "score": 0.95}]"""
    elif task == "automatic-speech-recognition":
        return """# Process speech recognition output
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            if hasattr(tokenizer, "decode"):
                transcription = tokenizer.decode(predicted_ids[0])
                predictions = [{"text": transcription}]
            else:
                predictions = [{"text": "Mock transcription"}]
        else:
            predictions = [{"text": "Mock transcription"}]"""
    else:
        # Generic fallback
        return """# Generic output processing
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            predictions = [{"output": "Processed model output"}]
        else:
            predictions = [{"output": "Mock output"}]"""

def generate_openvino_input_preparation(family_info):
    """Generate code to prepare input for OpenVINO."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task in ["fill-mask", "text-generation", "translation_en_to_fr", "summarization"]:
        return """# Prepare input
        if hasattr(tokenizer, "mask_token") and "[MASK]" in self.test_text:
            mask_token = tokenizer.mask_token
            test_input = self.test_text.replace("[MASK]", mask_token)
        else:
            test_input = self.test_text
            
        inputs = tokenizer(test_input, return_tensors="pt")"""
    elif task == "zero-shot-image-classification":
        return """# Prepare input
        test_input = self.test_image_url
        
        # Process image
        if HAS_PIL:
            response = requests.get(test_input)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Get text features
            inputs = tokenizer(self.candidate_labels, padding=True, return_tensors="pt")
            
            # Get image features
            processor = transformers.AutoProcessor.from_pretrained(self.model_id)
            image_inputs = processor(images=image, return_tensors="pt")
            inputs.update(image_inputs)
        else:
            # Mock inputs
            inputs = {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                "pixel_values": torch.zeros(1, 3, 224, 224)
            }"""
    elif task == "automatic-speech-recognition":
        return """# Prepare input
        test_input = self.test_audio
        
        # Load audio
        if HAS_AUDIO and os.path.exists(test_input):
            waveform, sample_rate = librosa.load(test_input, sr=16000)
            inputs = {"input_values": torch.tensor([waveform]).float()}
        else:
            # Mock audio input
            inputs = {"input_values": torch.zeros(1, 16000).float()}"""
    else:
        # Generic fallback
        return """# Prepare generic input
        test_input = "Generic input for testing"
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}"""

def generate_openvino_output_processing(family_info):
    """Generate code to process output for OpenVINO."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task == "fill-mask":
        return """# Get predictions
        if hasattr(tokenizer, "mask_token_id"):
            mask_token_id = tokenizer.mask_token_id
            mask_positions = (inputs["input_ids"] == mask_token_id).nonzero()
            
            if len(mask_positions) > 0:
                mask_index = mask_positions[0][-1].item()
                logits = outputs.logits[0, mask_index]
                top_k_indices = torch.topk(logits, 5).indices.tolist()
                
                predictions = []
                for idx in top_k_indices:
                    if hasattr(tokenizer, "convert_ids_to_tokens"):
                        token = tokenizer.convert_ids_to_tokens(idx)
                    else:
                        token = f"token_{idx}"
                    predictions.append(token)
            else:
                predictions = []
        else:
            predictions = []"""
    elif task == "text-generation":
        return """# Process generation output
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            if hasattr(tokenizer, "decode"):
                next_token = tokenizer.decode([next_token_id])
                predictions = [next_token]
            else:
                predictions = ["<mock_token>"]
        else:
            predictions = ["<mock_output>"]"""
    elif task in ["translation_en_to_fr", "summarization"]:
        return """# Process generation output
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            generated_ids = torch.argmax(logits, dim=-1)
            
            if hasattr(tokenizer, "decode"):
                decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                predictions = [decoded_output]
            else:
                predictions = ["<mock_output>"]
        else:
            predictions = ["<mock_output>"]"""
    elif task == "zero-shot-image-classification":
        return """# Process classification output
        if hasattr(outputs, "logits_per_image"):
            logits = outputs.logits_per_image[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            predictions = []
            for i, (label, prob) in enumerate(zip(self.candidate_labels, probs)):
                predictions.append({
                    "label": label,
                    "score": float(prob)
                })
        else:
            predictions = [{"label": "Mock label", "score": 0.95}]"""
    elif task == "automatic-speech-recognition":
        return """# Process speech recognition output
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            
            if hasattr(tokenizer, "decode"):
                transcription = tokenizer.decode(predicted_ids[0])
                predictions = [transcription]
            else:
                predictions = ["<mock_transcription>"]
        else:
            predictions = ["<mock_transcription>"]"""
    else:
        # Generic fallback
        return """# Generic output processing
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            predictions = ["Processed OpenVINO output"]
        else:
            predictions = ["<mock_output>"]"""

def generate_dependency_flags(family_info):
    """Generate dependency flag list for metadata."""
    dependencies = family_info.get("dependencies", [])
    
    flags = []
    for dep in dependencies:
        if dep == "tokenizers":
            flags.append('"has_tokenizers": HAS_TOKENIZERS')
        elif dep == "sentencepiece":
            flags.append('"has_sentencepiece": HAS_SENTENCEPIECE')
        elif dep == "pillow":
            flags.append('"has_pil": HAS_PIL')
        elif dep == "librosa":
            flags.append('"has_audio": HAS_AUDIO')
        elif dep == "accelerate":
            flags.append('"has_accelerate": HAS_ACCELERATE')
    
    return ",\n                ".join(flags)

def generate_auto_model_class(family_info):
    """Generate the appropriate Auto model class."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task == "fill-mask":
        return "AutoModelForMaskedLM"
    elif task == "text-generation":
        return "AutoModelForCausalLM"
    elif task in ["translation_en_to_fr", "summarization"]:
        return "AutoModelForSeq2SeqLM"
    elif task == "zero-shot-image-classification":
        return "AutoModel"
    elif task == "automatic-speech-recognition":
        return "AutoModelForSpeechSeq2Seq"
    else:
        # Generic fallback
        return "AutoModel"

def generate_openvino_model_class(family_info):
    """Generate the appropriate OpenVINO model class."""
    task = family_info.get("tasks", ["text-generation"])[0]
    
    if task == "fill-mask":
        return "OVModelForMaskedLM"
    elif task == "text-generation":
        return "OVModelForCausalLM"
    elif task in ["translation_en_to_fr", "summarization"]:
        return "OVModelForSeq2SeqLM"
    elif task == "zero-shot-image-classification":
        return "OVModelForVision"
    elif task == "automatic-speech-recognition":
        return "OVModelForSpeechSeq2Seq"
    else:
        # Generic fallback
        return "OVModel"

def generate_test_file(family_id):
    """Generate a complete test file for a model family."""
    family_info = MODEL_REGISTRY.get(family_id)
    if not family_info:
        logger.error(f"Unknown model family: {family_id}")
        return None
    
    # Prepare template variables
    variables = {
        "family_name": family_info["family_name"],
        "family_lower": family_id.lower(),
        "default_model": family_info["default_model"],
        "class_name": family_info["class"],
        "test_class_name": family_info["test_class"],
        "registry_name": f"{family_info['family_name'].upper()}_MODELS_REGISTRY",
        "default_task": family_info["tasks"][0] if family_info.get("tasks") else "text-generation",
        "model_class_comments": generate_model_class_comments(family_info),
        "dependency_imports": generate_dependency_imports(family_info),
        "models_registry": generate_models_registry(family_info),
        "test_inputs": generate_test_inputs(family_info),
        "has_dependencies": generate_dependency_flags(family_info),
        "auto_model_class": generate_auto_model_class(family_info),
        "openvino_model_class": generate_openvino_model_class(family_info)
    }
    
    # Generate pipeline test method
    pipeline_test_variables = {
        "pipeline_dependency_checks": generate_pipeline_dependency_checks(family_info),
        "pipeline_input_preparation": generate_pipeline_input_preparation(family_info)
    }
    pipeline_method = PIPELINE_TEST_TEMPLATE.format(**pipeline_test_variables)
    variables["test_pipeline_method"] = pipeline_method
    
    # Generate from_pretrained test method
    from_pretrained_variables = {
        "from_pretrained_dependency_checks": generate_from_pretrained_dependency_checks(family_info),
        "from_pretrained_input_preparation": generate_from_pretrained_input_preparation(family_info),
        "from_pretrained_output_processing": generate_from_pretrained_output_processing(family_info),
        "class_name": family_info["class"],
        "auto_model_class": generate_auto_model_class(family_info)
    }
    from_pretrained_method = FROM_PRETRAINED_TEMPLATE.format(**from_pretrained_variables)
    variables["test_from_pretrained_method"] = from_pretrained_method
    
    # Generate OpenVINO test method
    openvino_variables = {
        "openvino_model_class": generate_openvino_model_class(family_info),
        "openvino_input_preparation": generate_openvino_input_preparation(family_info),
        "openvino_output_processing": generate_openvino_output_processing(family_info)
    }
    openvino_method = OPENVINO_TEST_TEMPLATE.format(**openvino_variables)
    variables["test_openvino_method"] = openvino_method
    
    # Generate complete file
    test_file_content = BASE_TEST_FILE_TEMPLATE.format(**variables)
    
    return test_file_content

def create_test_file(family_id):
    """Create a test file for a model family and save it."""
    content = generate_test_file(family_id)
    if not content:
        return False
    
    family_info = MODEL_REGISTRY.get(family_id)
    module_name = family_info["module_name"]
    
    # Write to file
    file_path = CURRENT_DIR / f"{module_name}.py"
    with open(file_path, "w") as f:
        f.write(content)
    
    logger.info(f"Created test file: {file_path}")
    return True

def generate_all_test_files():
    """Generate test files for all model families."""
    successful = []
    failed = []
    
    for family_id in MODEL_REGISTRY:
        logger.info(f"Generating test file for {family_id}...")
        if create_test_file(family_id):
            successful.append(family_id)
        else:
            failed.append(family_id)
    
    logger.info(f"Successfully generated {len(successful)} test files")
    if failed:
        logger.warning(f"Failed to generate {len(failed)} test files: {', '.join(failed)}")
    
    return successful, failed

def update_test_all_models():
    """Update the test_all_models.py file with all model families."""
    model_families = {}
    
    for family_id, family_info in MODEL_REGISTRY.items():
        model_families[family_id] = {
            "module": family_info["module_name"],
            "description": family_info["description"],
            "default_model": family_info["default_model"],
            "class": family_info["test_class"],
            "status": "complete"
        }
    
    # Update file
    file_path = CURRENT_DIR / "test_all_models.py"
    
    # Check if file exists
    if not file_path.exists():
        logger.warning(f"test_all_models.py not found at {file_path}")
        return False
    
    # Read current content
    with open(file_path, "r") as f:
        content = f.read()
    
    # Find MODEL_FAMILIES definition
    import re
    model_families_match = re.search(r"MODEL_FAMILIES\s*=\s*\{([^}]+)\}", content, re.DOTALL)
    if not model_families_match:
        logger.warning("MODEL_FAMILIES definition not found in test_all_models.py")
        return False
    
    # Generate new MODEL_FAMILIES definition
    model_families_str = "MODEL_FAMILIES = {\n"
    for family_id, family_info in model_families.items():
        model_families_str += f'    "{family_id}": {{\n'
        model_families_str += f'        "module": "{family_info["module"]}",\n'
        model_families_str += f'        "description": "{family_info["description"]}",\n'
        model_families_str += f'        "default_model": "{family_info["default_model"]}",\n'
        model_families_str += f'        "class": "{family_info["class"]}",\n'
        model_families_str += f'        "status": "{family_info["status"]}"\n'
        model_families_str += f'    }},\n'
    model_families_str += "}"
    
    # Replace MODEL_FAMILIES definition
    new_content = re.sub(r"MODEL_FAMILIES\s*=\s*\{[^}]+\}", model_families_str, content, flags=re.DOTALL)
    
    # Write new content
    with open(file_path, "w") as f:
        f.write(new_content)
    
    logger.info(f"Updated test_all_models.py with {len(model_families)} model families")
    return True

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate test files for Hugging Face models")
    
    # Generation options
    parser.add_argument("--generate", type=str, help="Generate test file for a specific model family")
    parser.add_argument("--all", action="store_true", help="Generate test files for all model families")
    parser.add_argument("--update-all-models", action="store_true", help="Update test_all_models.py with all model families")
    
    # List options
    parser.add_argument("--list-families", action="store_true", help="List all model families in the registry")
    
    args = parser.parse_args()
    
    # List model families if requested
    if args.list_families:
        print("\nAvailable Model Families in Registry:")
        for family_id, family_info in MODEL_REGISTRY.items():
            print(f"  - {family_id}: {family_info['description']} ({family_info['default_model']})")
        return
    
    # Generate a specific test file
    if args.generate:
        family_id = args.generate
        if family_id not in MODEL_REGISTRY:
            print(f"Unknown model family: {family_id}")
            return
        
        create_test_file(family_id)
    
    # Generate all test files
    if args.all:
        generate_all_test_files()
    
    # Update test_all_models.py
    if args.update_all_models:
        update_test_all_models()
    
    # Default: print help
    if not (args.generate or args.all or args.update_all_models or args.list_families):
        parser.print_help()

if __name__ == "__main__":
    main()