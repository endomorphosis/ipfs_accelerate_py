#!/usr/bin/env python3
"""
Fix the test_generator_fixed.py file to address issues with:
1. CUDA block indentation in test_from_pretrained
2. Registry duplication for hyphenated models
3. Add standardized helper methods

Usage:
    python fix_test_generator.py
"""

import os
import sys
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"test_generator_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Path to the test generator
TEST_GENERATOR_PATH = "/home/barberb/ipfs_accelerate_py/test/skills/test_generator_fixed.py"
BACKUP_PATH = f"{TEST_GENERATOR_PATH}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_PATH = f"{TEST_GENERATOR_PATH}.new"

def fix_cuda_indentation(content):
    """
    Fix indentation issues in the CUDA testing block of the template.
    These issues affect the test_from_pretrained method.
    """
    # Pattern for CUDA conditional block with indentation issues
    pattern = r'(if device == "cuda":\s+try:.*?with torch\.no_grad\(\):.*?_ = model\(\*\*inputs\).*?except Exception:.*?)(\s+pass\s+)(\s+# Run multiple)'
    
    def reindent_block(match):
        """Fix indentation for the matched CUDA block."""
        cuda_block = match.group(1)
        pass_stmt = match.group(2)
        next_section = match.group(3)
        
        # Calculate proper indentation
        lines = cuda_block.split('\n')
        try_line = [line for line in lines if "try:" in line][0]
        indent_level = len(try_line) - len(try_line.lstrip())
        indentation = " " * indent_level
        
        # Fix indentation and rebuild block
        return f"{cuda_block}{pass_stmt}\n{indentation}\n{indentation}{next_section.strip()}"
    
    try:
        # Apply the fix with regex substitution
        fixed_content = re.sub(pattern, reindent_block, content, flags=re.DOTALL)
        if fixed_content == content:
            logger.info("No CUDA block indentation issues found or pattern didn't match")
        else:
            logger.info("Successfully fixed CUDA block indentation")
        return fixed_content
    except Exception as e:
        logger.error(f"Error fixing CUDA indentation: {e}")
        return content

def fix_registry_duplication(content):
    """
    Fix registry name duplication for hyphenated models.
    This prevents issues like GPT_GPT_GPT_GPT_J_MODELS_REGISTRY.
    """
    # Function to replace duplicated registry patterns
    def replace_duplicate_registry(match):
        model_name = match.group(1)
        model_name_upper = model_name.upper()
        # Replace any duplicate segments like GPT_GPT_GPT with just GPT
        cleaned_name = re.sub(r'([A-Z]+)(?:_+\1)+', r'\1', model_name_upper)
        return f"{cleaned_name}_MODELS_REGISTRY"
    
    # Pattern to find duplicated registry names like GPT_GPT_GPT_GPT_J_MODELS_REGISTRY
    pattern = r'([A-Za-z0-9_]+)(?:_+[A-Za-z0-9_]+)*_MODELS_REGISTRY'
    
    try:
        # Apply the fix with regex substitution
        fixed_content = re.sub(pattern, replace_duplicate_registry, content)
        if fixed_content == content:
            logger.info("No registry duplication issues found or pattern didn't match")
        else:
            logger.info("Successfully fixed registry duplication")
        return fixed_content
    except Exception as e:
        logger.error(f"Error fixing registry duplication: {e}")
        return content

def add_helper_methods(content):
    """
    Add standardized helper methods to the template that's used for tests.
    This ensures consistent implementation across all model types.
    """
    # Helper methods to add to the templates
    helper_methods = """    
    def get_model_class(self):
        \"\"\"Get the appropriate model class based on model type.\"\"\"
        if self.class_name in globals() and callable(globals()[self.class_name]):
            return globals()[self.class_name]
        
        # Try direct model class from transformers
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
    
    # Find where to add helper methods - before run_tests method
    pattern = r'(def run_tests.*?\(self)'
    
    try:
        # Add the helper methods before run_tests 
        replacement = f"{helper_methods}\n    \g<1>"
        fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        if fixed_content == content:
            logger.warning("Could not add helper methods - pattern didn't match")
        else:
            logger.info("Successfully added helper methods")
        
        return fixed_content
    except Exception as e:
        logger.error(f"Error adding helper methods: {e}")
        return content

def fix_test_from_pretrained_method(content):
    """
    Replace the test_from_pretrained method with a standardized implementation.
    This ensures consistent behavior across all model types.
    """
    # Standardized test_from_pretrained implementation
    standard_method = """
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
"""
    
    # Pattern to match existing test_from_pretrained method
    pattern = r'def test_from_pretrained.*?return results\s+'
    
    try:
        # Replace the method with the standardized version
        fixed_content = re.sub(pattern, standard_method, content, flags=re.DOTALL)
        if fixed_content == content:
            logger.warning("Could not replace test_from_pretrained method - pattern didn't match")
        else:
            logger.info("Successfully replaced test_from_pretrained method")
        return fixed_content
    except Exception as e:
        logger.error(f"Error replacing test_from_pretrained method: {e}")
        return content

def fix_valid_identifier_function(content):
    """
    Enhance the to_valid_identifier function to better handle hyphenated model names.
    """
    improved_function = """
def to_valid_identifier(name):
    \"\"\"
    Convert a model name to a valid Python identifier.
    Specifically handles hyphenated model names to avoid duplication issues.
    
    Args:
        name: The model name to convert
        
    Returns:
        A valid Python identifier
    \"\"\"
    # Replace hyphens with underscores
    valid_name = name.replace("-", "_")
    
    # Ensure the name doesn't start with a number
    if valid_name and valid_name[0].isdigit():
        valid_name = f"m{valid_name}"
    
    # Replace any invalid characters with underscores
    valid_name = re.sub(r'[^a-zA-Z0-9_]', '_', valid_name)
    
    # Deduplicate consecutive underscores
    valid_name = re.sub(r'_+', '_', valid_name)
    
    return valid_name
"""
    
    # Pattern to match existing to_valid_identifier function
    pattern = r'def to_valid_identifier.*?\):.*?return.*?\n'
    
    try:
        # Replace the function with the improved version
        fixed_content = re.sub(pattern, improved_function, content, flags=re.DOTALL)
        if fixed_content == content:
            logger.warning("Could not replace to_valid_identifier function - pattern didn't match")
        else:
            logger.info("Successfully replaced to_valid_identifier function")
        return fixed_content
    except Exception as e:
        logger.error(f"Error replacing to_valid_identifier function: {e}")
        return content

def fix_test_generator():
    """Apply all fixes to the test generator."""
    try:
        # Create backup of original file
        with open(TEST_GENERATOR_PATH, 'r') as f:
            original_content = f.read()
        
        with open(BACKUP_PATH, 'w') as f:
            f.write(original_content)
        logger.info(f"Created backup at {BACKUP_PATH}")
        
        # Apply all fixes
        content = original_content
        content = fix_cuda_indentation(content)
        content = fix_registry_duplication(content)
        content = add_helper_methods(content)
        content = fix_test_from_pretrained_method(content)
        content = fix_valid_identifier_function(content)
        
        # First save to a new file for safety
        with open(OUTPUT_PATH, 'w') as f:
            f.write(content)
        logger.info(f"Saved fixed generator to {OUTPUT_PATH}")
        
        # Replace the original file
        with open(TEST_GENERATOR_PATH, 'w') as f:
            f.write(content)
        logger.info(f"Successfully updated {TEST_GENERATOR_PATH}")
        
        return True
    except Exception as e:
        logger.error(f"Error fixing test generator: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting test generator fix...")
    success = fix_test_generator()
    
    if success:
        print(f"✅ Successfully fixed test generator at {TEST_GENERATOR_PATH}")
        print(f"  Original file backed up to {BACKUP_PATH}")
    else:
        print(f"❌ Failed to fix test generator. See log for details.")
    
    sys.exit(0 if success else 1)