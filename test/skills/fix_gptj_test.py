#!/usr/bin/env python3
"""
Fix the GPT-J test file by applying standardized implementations.

This script fixes the issues in the test_hf_gpt-j.py file, including:
1. Fixing indentation problems in the CUDA section
2. Standardizing the from_pretrained method
3. Fixing the model registry duplication issue
4. Adding proper helper methods

Usage:
    python fix_gptj_test.py
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_indentation(content):
    """Fix indentation issues in the CUDA block of test_from_pretrained."""
    # Find the CUDA block
    pattern = r'(if device == "cuda":\s+try:.*?with torch\.no_grad\(\):.*?_ = model\(\*\*inputs\).*?except Exception:.*?)(\s+pass\s+)(\s+# Run multiple)'
    
    def reindent_block(match):
        cuda_block = match.group(1)
        pass_stmt = match.group(2)
        next_section = match.group(3)
        
        # Calculate indentation
        lines = cuda_block.split('\n')
        try_line = [line for line in lines if "try:" in line][0]
        indent_level = len(try_line) - len(try_line.lstrip())
        
        # Fix indentation
        indentation = " " * indent_level
        return f"{cuda_block}{pass_stmt}\n{indentation}\n{indentation}{next_section.strip()}"
    
    try:
        # Apply the fix
        fixed_content = re.sub(pattern, reindent_block, content, flags=re.DOTALL)
        return fixed_content
    except Exception as e:
        logger.error(f"Error fixing indentation: {e}")
        return content

def fix_registry_name(content):
    """Fix the duplicated registry name."""
    # Replace the duplicated registry name
    pattern = r'GPT_GPT_GPT_GPT_J_MODELS_REGISTRY'
    replacement = r'GPT_J_MODELS_REGISTRY'
    
    fixed_content = content.replace(pattern, replacement)
    
    # Fix other registry references
    fixed_content = fixed_content.replace("gpt_j_j_j_j_j_j_j_", "gpt_j_")
    
    return fixed_content

def add_helper_methods(content):
    """Add the standardized helper methods."""
    # Find the location to insert helper methods
    pattern = r'(def test_from_pretrained.*?return results\s+)(def run_tests)'
    
    helper_methods = """    def get_model_class(self):
        \"\"\"Get the appropriate model class based on model type.\"\"\"
        if self.class_name == "GptJLMHeadModel":
            return transformers.GPTJForCausalLM
        
        # Try direct model class from transformers
        if hasattr(transformers, self.class_name):
            return getattr(transformers, self.class_name)
            
        # Fallback based on task
        if self.task == "text-generation":
            return transformers.AutoModelForCausalLM
        else:
            # Default fallback
            return transformers.AutoModel
    
    def prepare_test_input(self):
        \"\"\"Prepare appropriate test input for the model type.\"\"\"
        return self.test_text
    
    def process_model_output(self, output, tokenizer):
        \"\"\"Process model output based on model type.\"\"\"
        try:
            # For text generation models with logits
            if hasattr(output, "logits"):
                logits = output.logits
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                next_token = tokenizer.decode([next_token_id])
                return [{"token": next_token, "score": 1.0}]
            else:
                return [{"generated_text": "Mock generated text"}]
        except Exception as e:
            logger.warning(f"Error processing model output: {e}")
            return [{"error": "Unable to process model output"}]
    
    """
    
    # Insert helper methods before run_tests
    replacement = f"\\1{helper_methods}\\2"
    return re.sub(pattern, replacement, content, flags=re.DOTALL)

def fix_test_file(input_path, output_path):
    """Apply all fixes to the test file."""
    try:
        # Read the file content
        with open(input_path, 'r') as f:
            content = f.read()
        
        # Apply fixes
        fixed_content = fix_indentation(content)
        fixed_content = fix_registry_name(fixed_content)
        fixed_content = add_helper_methods(fixed_content)
        
        # Write the fixed content
        with open(output_path, 'w') as f:
            f.write(fixed_content)
        
        logger.info(f"Successfully fixed and saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error fixing test file: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    # Define paths
    input_path = "/home/barberb/ipfs_accelerate_py/test/skills/fixed_tests/test_hf_gpt-j.py.issues.bak"
    output_path = "/home/barberb/ipfs_accelerate_py/test/skills/fixed_tests/test_hf_gptj.py"
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fix the test file
    success = fix_test_file(input_path, output_path)
    
    if success:
        print(f"✅ Successfully fixed GPT-J test file at {output_path}")
        return 0
    else:
        print("❌ Failed to fix GPT-J test file")
        return 1

if __name__ == "__main__":
    main()