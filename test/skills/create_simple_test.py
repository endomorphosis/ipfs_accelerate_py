#!/usr/bin/env python3
"""
Create a simple test file that implements the standardized from_pretrained() method.
This script focuses on creating one model test at a time with correct syntax.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_TYPES = {
    "bert": {
        "registry_name": "BERT_MODELS_REGISTRY",
        "class_name": "BertForMaskedLM",
        "test_class_name": "TestBertModels",
        "task": "fill-mask",
        "test_input": "The capital of France is [MASK]."
    },
    "gpt2": {
        "registry_name": "GPT2_MODELS_REGISTRY",
        "class_name": "GPT2LMHeadModel",
        "test_class_name": "TestGpt2Models",
        "task": "text-generation",
        "test_input": "GPT-2 is a language model that"
    },
    "t5": {
        "registry_name": "T5_MODELS_REGISTRY",
        "class_name": "T5ForConditionalGeneration",
        "test_class_name": "TestT5Models",
        "task": "text2text-generation",
        "test_input": "translate English to German: The house is wonderful."
    },
    "vit": {
        "registry_name": "VIT_MODELS_REGISTRY",
        "class_name": "ViTForImageClassification",
        "test_class_name": "TestVitModels",
        "task": "image-classification",
        "test_input": "[IMAGE_PLACEHOLDER]"
    }
}

def create_model_test(model_type, output_dir="fixed_tests"):
    """Create a standardized test file for a given model type."""
    if model_type not in MODEL_TYPES:
        logger.error(f"Model type {model_type} not found in supported types")
        logger.info(f"Supported types: {', '.join(MODEL_TYPES.keys())}")
        return False
    
    # Get model info
    model_info = MODEL_TYPES[model_type]
    
    # Create file path
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"test_hf_{model_type}.py")
    
    logger.info(f"Creating test file for {model_type} at {file_path}")
    
    # Create file content with correct indentation 
    content = f'''#!/usr/bin/env python3

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

# Check for mock environment variables
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'

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

# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

# Model registry
{model_info["registry_name"]} = {{
    "{model_type}": {{
        "description": "{model_type.upper()} model",
        "class": "{model_info["class_name"]}",
    }},
    "{model_type}-base": {{
        "description": "{model_type.upper()} base model",
        "class": "{model_info["class_name"]}",
    }}
}}

class {model_info["test_class_name"]}:
    """Test class for {model_type.upper()} models."""
    
    def __init__(self, model_id=None):
        """Initialize with the model ID to test."""
        self.model_id = model_id or "{model_type}"
        
        # Define model parameters
        self.task = "{model_info["task"]}"
        self.class_name = "{model_info["class_name"]}"
        self.description = "Test for {model_type.upper()} models"
        
        # Define test inputs
        self.test_text = "{model_info["test_input"]}"
        
        # Configure hardware preference
        if HAS_TORCH and torch.cuda.is_available():
            self.preferred_device = "cuda"
        elif HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {{self.preferred_device}} as preferred device")
        
        # Results storage
        self.results = {{}}
        self.examples = []
        self.performance_stats = {{}}
    
    def get_model_class(self):
        """Get the appropriate model class based on model type."""
        if self.class_name == "{model_info["class_name"]}":
            return transformers.{model_info["class_name"]}
        else:
            # Fallback to Auto class based on task
            if self.task == "text-generation":
                return transformers.AutoModelForCausalLM
            elif self.task == "fill-mask":
                return transformers.AutoModelForMaskedLM
            elif self.task == "text2text-generation":
                return transformers.AutoModelForSeq2SeqLM
            elif self.task == "image-classification":
                return transformers.AutoModelForImageClassification
            else:
                return transformers.AutoModel
    
    def prepare_test_input(self):
        """Prepare appropriate test input for the model type."""
        return self.test_text
    
    def process_model_output(self, output, tokenizer):
        """Process model output based on model type."""
        if not hasattr(output, "logits"):
            return [{{"output": "Model output without logits"}}]
            
        logits = output.logits
        
        # Task-specific processing
        if self.task == "text-generation":
            # Extract next token prediction
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            if hasattr(tokenizer, "decode"):
                next_token = tokenizer.decode([next_token_id])
                return [{{"token": next_token, "score": float(next_token_logits.max())}}]
        
        elif self.task == "fill-mask":
            # Get prediction for masked token
            mask_token_index = torch.where(torch.tensor(tokenizer.encode(self.test_text)) == tokenizer.mask_token_id)[0]
            if len(mask_token_index) > 0:
                mask_index = mask_token_index[0].item()
                mask_logits = logits[0, mask_index, :]
                top_tokens = torch.topk(mask_logits, 5)
                return [{{
                    "token": tokenizer.decode([token_id.item()]),
                    "score": float(score.item())
                }} for token_id, score in zip(top_tokens.indices, top_tokens.values)]
        
        elif self.task == "text2text-generation":
            # For seq2seq models
            return [{{"generated_text": tokenizer.decode(output.logits.argmax(dim=-1)[0], skip_special_tokens=True)}}]
            
        elif self.task == "image-classification":
            # For image classification
            class_id = torch.argmax(logits, dim=1).item()
            return [{{"class_id": class_id, "score": float(logits[0, class_id])}}]
        
        # Default fallback
        return [{{"prediction": "Generic model output"}}]
    
    def test_from_pretrained(self, device="auto"):
        """Test the model using direct from_pretrained loading."""
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
            
        if not HAS_TOKENIZERS:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_deps"] = ["tokenizers>=0.11.0"]
            results["from_pretrained_success"] = False
            return results
        
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
            
            # Apply model-specific tokenizer fixes
            if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
                if hasattr(tokenizer, 'eos_token'):
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token for {{self.model_id}} tokenizer")
                
            tokenizer_load_time = time.time() - tokenizer_load_start
            
            # Use appropriate model class based on model type
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
            
            # Prepare appropriate test input for model type
            test_input = self.prepare_test_input()
            
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors="pt")
            
            # Move inputs to device
            if device != "cpu":
                inputs = {{key: val.to(device) for key, val in inputs.items()}}
            
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
            example_data = {{
                "method": f"from_pretrained() on {{device}}",
                "input": str(test_input)
            }}
            
            if predictions:
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
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        # Always test on default device
        self.test_from_pretrained()
        
        # Build final results
        return {{
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "metadata": {{
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS
            }}
        }}

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test {model_type.upper()} models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    args = parser.parse_args()
    
    # Test single model (default or specified)
    model_id = args.model or "{model_type}"
    logger.info(f"Testing model: {{model_id}}")
    
    # Run test
    tester = {model_info["test_class_name"]}(model_id)
    results = tester.run_tests()
    
    # Print summary
    success = any(r.get("from_pretrained_success", False) for r in results["results"].values()
        if r.get("from_pretrained_success") is not False)
    
    print("\\nTEST RESULTS SUMMARY:")
    
    if success:
        print(f"✅ Successfully tested {{model_id}}")
    else:
        print(f"❌ Failed to test {{model_id}}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
'''

    # Write to file 
    with open(file_path, 'w') as f:
        f.write(content)
    
    # Verify syntax
    try:
        with open(file_path, 'r') as f:
            test_content = f.read()
        compile(test_content, file_path, 'exec')
        logger.info(f"Successfully created syntactically correct test file for {model_type}")
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in generated file {file_path} at line {e.lineno}: {e.msg}")
        if hasattr(e, 'text'):
            logger.error(f"Line content: {e.text}")
        return False

def main():
    """Main function to create standardized test files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create standardized test files")
    parser.add_argument("--model-type", required=True, choices=MODEL_TYPES.keys(),
                        help="Model type to generate test for")
    parser.add_argument("--output-dir", default="fixed_tests",
                        help="Output directory for test files")
    
    args = parser.parse_args()
    
    # Create test file
    success = create_model_test(args.model_type, args.output_dir)
    
    if success:
        logger.info(f"Successfully created test file for {args.model_type}")
        return 0
    else:
        logger.error(f"Failed to create test file for {args.model_type}")
        return 1

if __name__ == "__main__":
    sys.exit(main())