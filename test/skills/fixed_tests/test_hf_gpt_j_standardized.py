#!/usr/bin/env python3

"""
Standardized test for the GptJ model type from HuggingFace Transformers.
This file implements a consistent testing approach with model-specific adaptations.
"""

import os
import sys
import json
import time
import logging
import traceback
import argparse
from unittest.mock import MagicMock
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if dependencies are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")

# Model registry
GPT_J_MODELS_REGISTRY = {
    "EleutherAI/gpt-j-6b": {
        "full_name": "GptJ Base",
        "architecture": "decoder-only",
        "description": "GptJ model for text-generation",
        "model_type": "gpt-j",
        "parameters": "110M",
        "default_task": "text-generation"
    }
}

def select_device():
    """Detect the best available device."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    elif HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
        return "mps"
    else:
        return "cpu"

class TestGptJModels:
    """Test class for GptJ models."""
    
    def __init__(self, model_id=None, device=None):
        """
        Initialize the test class for a specific model.
        
        Args:
            model_id: Model identifier (default: "EleutherAI/gpt-j-6b")
            device: Device to run on (default: auto-detect)
        """
        self.model_id = model_id or "EleutherAI/gpt-j-6b"
        
        # Verify model exists in registry
        if self.model_id not in GPT_J_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_config = GPT_J_MODELS_REGISTRY["EleutherAI/gpt-j-6b"]
        else:
            self.model_config = GPT_J_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = self.model_config.get("default_task", "text-generation")
        self.class_name = "GPTJForCausalLM"
        self.architecture = self.model_config.get("architecture", "decoder-only")
        
        # Test inputs - model type specific
        self.test_text = "GPT-J is a transformer model that"
        
        # Hardware configuration
        self.preferred_device = device or select_device()
        
        # Results and examples storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
    
    def get_model_class(self):
        """Get the appropriate model class based on model type."""
        if hasattr(transformers, self.class_name):
            return getattr(transformers, self.class_name)
            
        # Fallback based on task
        if self.task == "text-generation":
            return transformers.AutoModelForCausalLM
        elif self.task == "fill-mask":
            return transformers.AutoModelForMaskedLM
        elif self.task == "text2text-generation":
            return transformers.AutoModelForSeq2SeqLM
        elif self.task == "image-classification":
            return transformers.AutoModelForImageClassification
        elif self.task == "automatic-speech-recognition":
            return transformers.AutoModelForSpeechSeq2Seq
        else:
            # Default fallback
            return transformers.AutoModel
    
    def prepare_test_input(self):
        """Prepare appropriate test input for the model type."""
        return self.test_text
    
    def process_model_output(self, output, tokenizer):
        """Process model output based on model type."""
        try:
            # For language models with logits
            if hasattr(output, "logits"):
                logits = output.logits
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                next_token = tokenizer.decode([next_token_id])
                return [{"token": next_token, "score": 1.0}]
            # For masked language models
            elif hasattr(output, "logits") and self.task == "fill-mask":
                logits = output.logits
                mask_token_index = torch.where(tokenizer.mask_token_id == inputs.input_ids)[1]
                mask_logits = logits[0, mask_token_index, :]
                top_tokens = torch.topk(mask_logits, k=5)
                tokens = [tokenizer.decode(token_id.item()) for token_id in top_tokens.indices[0]]
                return [{"predictions": tokens}]
            # For other model types
            elif hasattr(output, "last_hidden_state"):
                return [{"hidden_state_shape": list(output.last_hidden_state.shape)}]
            else:
                return [{"output_processed": True}]
        except Exception as e:
            logger.warning(f"Error processing model output: {e}")
            return [{"error": "Unable to process model output"}]
    
    def test_pipeline(self, device="auto"):
        """Test the model using the pipeline API."""
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
            
        if not HAS_TOKENIZERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_deps"] = ["tokenizers>=0.11.0"]
            results["pipeline_success"] = False
            return results
        
        try:
            logger.info(f"Testing {self.model_id} with pipeline() on {device}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device if device != "cpu" else -1
            }
            
            # Time the model loading
            load_start_time = time.time()
            
            # First load the tokenizer to fix any model-specific issues
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            
            # Create pipeline with the tokenizer
            pipeline_kwargs["tokenizer"] = tokenizer
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test input
            pipeline_input = self.prepare_test_input()
            
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
            
        if not HAS_TOKENIZERS:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_deps"] = ["tokenizers>=0.11.0"]
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
            
            # Fix padding token if needed (common for decoder-only models)
            if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token for {self.model_id} tokenizer")
                
            tokenizer_load_time = time.time() - tokenizer_load_start
            
            # Get appropriate model class
            model_class = self.get_model_class()
            
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
            test_input = self.prepare_test_input()
            
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors="pt")
            
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
            
            # Process model output based on model type
            predictions = self.process_model_output(outputs[0], tokenizer)
            
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
            if predictions:
                results["predictions"] = predictions
            
            # Add to examples
            example_data = {
                "method": f"from_pretrained() on {device}",
                "input": str(test_input)
            }
            
            if predictions:
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
    
    def run_tests(self, all_hardware=False):
        """
        Run all tests for this model.
        
        Args:
            all_hardware: If True, tests on all available hardware
        
        Returns:
            Dict containing test results
        """
        # Run pipeline test
        self.test_pipeline()
        
        # Run from_pretrained test
        self.test_from_pretrained()
        
        # Test on all hardware if requested
        if all_hardware:
            if self.preferred_device != "cpu":
                self.test_pipeline(device="cpu")
                self.test_from_pretrained(device="cpu")
            
            if HAS_TORCH and torch.cuda.is_available() and self.preferred_device != "cuda":
                self.test_pipeline(device="cuda")
                self.test_from_pretrained(device="cuda")
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
        # Build final results
        return {
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "metadata": {
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "architecture": self.architecture,
                "timestamp": datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS,
                "has_sentencepiece": HAS_SENTENCEPIECE,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
            }
        }

def save_results(model_id, results, model_family, output_dir="collected_results"):
    """Save test results to a file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    model_family_lower = model_family.lower()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"hf_gpt-j_gpt-j_2025-03-22T23:33:33.766357.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    return output_path

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test GptJ HuggingFace models")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-j-6b", help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to run tests on (cuda, cpu)")
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for output files")
    
    args = parser.parse_args()
    
    # Initialize the test class
    gpt_j_tester = TestGptJModels(model_id=args.model, device=args.device)
    
    # Run the tests
    results = gpt_j_tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(args.model, results, model_family, output_dir=args.output_dir)
    
    # Print a summary
    pipeline_success = any(r.get("pipeline_success", False) for r in results["results"].values() if "pipeline" in r)
    pretrained_success = any(r.get("from_pretrained_success", False) for r in results["results"].values() if "from_pretrained" in r)
    success = pipeline_success or pretrained_success
    
    # Determine if real inference or mock objects were used
    using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
    using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
    
    print("\nTEST RESULTS SUMMARY:")
    
    # Indicate real vs mock inference clearly
    if using_real_inference and not using_mocks:
        print(f"🚀 Using REAL INFERENCE with actual models")
    else:
        print(f"🔷 Using MOCK OBJECTS for CI/CD testing only")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
    
    if success:
        print(f"✅ Successfully tested {args.model}")
        print(f"  - Device: {gpt_j_tester.preferred_device}")
        
        # Print pipeline performance stats if available
        if pipeline_success and f"pipeline_{gpt_j_tester.preferred_device}" in results["performance"]:
            stats = results["performance"][f"pipeline_{gpt_j_tester.preferred_device}"]
            print(f"  - Pipeline: {stats.get('avg_time', 0):.4f}s avg inference")
            
        # Print from_pretrained performance stats if available
        if pretrained_success and f"from_pretrained_{gpt_j_tester.preferred_device}" in results["performance"]:
            stats = results["performance"][f"from_pretrained_{gpt_j_tester.preferred_device}"]
            print(f"  - from_pretrained: {stats.get('avg_time', 0):.4f}s avg inference")
            
    else:
        print(f"❌ Failed to test {args.model}")
        
        # Print error information
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {test_name}: {result.get('pipeline_error_type', 'unknown')}")
                print(f"    {result.get('pipeline_error', 'Unknown error')}")
            elif "from_pretrained_error" in result:
                print(f"  - Error in {test_name}: {result.get('from_pretrained_error_type', 'unknown')}")
                print(f"    {result.get('from_pretrained_error', 'Unknown error')}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
