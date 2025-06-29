#!/usr/bin/env python3
"""
Script to standardize the test_from_pretrained method in test files
to ensure consistent implementation across model types.

Usage:
    python standardize_test_from_pretrained.py [file_path]
"""

import os
import sys
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Standardized test_from_pretrained implementation
STANDARD_IMPLEMENTATION = """
    def test_from_pretrained(self, device="auto"):
        \"\"\"Test the model using direct from_pretrained loading.\"\"\""""
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
            
            # Use helper method to get appropriate model class
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
            
            # Prepare test input using helper method
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
            
            # Process model output using helper method
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
"""

# Standard helper methods to add
HELPER_METHODS = """
    def get_model_class(self):
        \"\"\"Get the appropriate model class based on model type.\"\"\""""
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
        \"\"\"Prepare appropriate test input for the model type.\"\"\""""
        return self.test_text
    
    def process_model_output(self, output, tokenizer):
        \"\"\"Process model output based on model type.\"\"\""""
        try:
            # For language models with logits
            if hasattr(output, "logits"):
                logits = output.logits
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                next_token = tokenizer.decode([next_token_id])
                return [{"token": next_token, "score": 1.0}]
            # For other model types
            elif hasattr(output, "last_hidden_state"):
                return [{"hidden_state_shape": list(output.last_hidden_state.shape)}]
            else:
                return [{"output_processed": True}]
        except Exception as e:
            logger.warning(f"Error processing model output: {e}")
            return [{"error": "Unable to process model output"}]
"""

def add_or_replace_method(content, method_name, new_implementation):
    """
    Add a new method or replace an existing one in a class definition.
    
    Args:
        content: The file content
        method_name: The method name to find/replace
        new_implementation: The new method implementation
        
    Returns:
        Updated content
    """
    # Find existing implementation
    pattern = rf'def {method_name}\([^)]*\).*?(?=\n    def |\n\n|$)'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Replace existing implementation
        return content.replace(match.group(0), new_implementation.strip())
    else:
        # Find a good place to add the method - before run_tests or at the end of the class
        run_tests_pattern = r'def run_tests'
        run_tests_match = re.search(run_tests_pattern, content)
        
        if run_tests_match:
            # Add before run_tests
            position = run_tests_match.start()
            return content[:position] + new_implementation + "\n    " + content[position:]
        else:
            # Find end of class definition
            class_pattern = r'class\s+\w+.*?(\n\ndef|\Z)'
            class_match = re.search(class_pattern, content, re.DOTALL)
            
            if class_match:
                # Add at end of class
                end_pos = class_match.end(0) - len(class_match.group(1))
                return content[:end_pos] + new_implementation + "\n" + content[end_pos:]
            else:
                # Couldn't find a good place, append to file
                logger.warning("Could not find a good location to add the method")
                return content + "\n" + new_implementation
                
def fix_file(file_path):
    """
    Add or replace test_from_pretrained method and helper methods in a test file.
    
    Args:
        file_path: Path to the test file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Create backup
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup at {backup_path}")
        
        # Check if the file has a class definition
        class_match = re.search(r'class\s+\w+', content)
        if not class_match:
            logger.error(f"No class definition found in {file_path}")
            return False
        
        # Add helper methods
        content = add_or_replace_method(content, "get_model_class", HELPER_METHODS)
        
        # Add or replace test_from_pretrained method
        content = add_or_replace_method(content, "test_from_pretrained", STANDARD_IMPLEMENTATION)
        
        # Write changes
        with open(file_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Successfully updated {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python standardize_test_from_pretrained.py <file_path>")
        return 1
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return 1
        
    success = fix_file(file_path)
    
    if success:
        print(f"✅ Successfully standardized {file_path}")
    else:
        print(f"❌ Failed to standardize {file_path}")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())