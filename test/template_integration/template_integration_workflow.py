#!/usr/bin/env python3
"""
Template integration workflow for generating refactored tests.

This script provides a complete workflow for generating refactored test files
from templates and validating them. It fixes indentation issues with special
handling code and ensures proper class inheritance.
"""

import os
import sys
import argparse
import logging
import tempfile
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"integration_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import template utilities
try:
    from fix_template_issues import customize_template, verify_test_file
    from generate_refactored_test import (
        determine_architecture, sanitize_model_name, get_template_path,
        generate_output_path, MODEL_ARCHITECTURE_MAPPING
    )
except ImportError as e:
    logger.error(f"Could not import required modules: {e}")
    sys.exit(1)

def fix_indentation(content, model_name, architecture):
    """Fix indentation issues in the template content."""
    lines = content.split('\n')
    fixed_lines = []
    
    # Track indentation state
    in_method = False
    current_method = None
    method_indent = ""
    block_indent = ""
    special_block_end = -1  # Initialize to handle scope issues
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip lines we've already processed
        if i <= special_block_end:
            i += 1
            continue
            
        # Detect method declarations
        if line.strip().startswith('def ') and line.strip().endswith(':'):
            in_method = True
            current_method = line.strip().split('(')[0].replace('def ', '')
            # Capture the method indentation
            method_indent = line[:line.find('def')]
            # Calculate the block indentation (method indent + 4 spaces)
            block_indent = method_indent + '    '
            fixed_lines.append(line)
            i += 1
            continue
        
        # Look for special handling code blocks that are improperly indented
        if (in_method and 
            (line.strip().startswith('# Create dummy image') or 
             line.strip().startswith('# Create dummy audio') or
             line.strip().startswith('if not os.path.exists("test.jpg")') or
             line.strip().startswith('if not os.path.exists("test.wav")') or
             line.strip().startswith('if not os.path.exists(self.test_audio_path)') or
             line.strip().startswith('# Generate a simple sine wave') or
             line.strip().startswith('if "whisper" in self.model_id.lower():') or
             line.strip().startswith('if self.model_type == "clip":') or
             line.strip().startswith('elif "wav2vec2" in self.model_id.lower():') or
             line.strip().startswith('elif self.model_type == "blip":') or
             line.strip().startswith('try:') or
             line.strip().startswith('import openvino') or
             line.strip().startswith('from optimum.intel import') or
             line.strip().startswith('except ImportError:'))):
            
            # Mark the start of a special block
            special_block_start = i
            special_block_end = i
            
            # Find the end of the special block (blank line)
            for j in range(i+1, len(lines)):
                if not lines[j].strip():
                    special_block_end = j
                    break
                elif j == len(lines) - 1:  # End of file
                    special_block_end = j
                    break
            
            # Extract the special block
            special_block = lines[special_block_start:special_block_end+1]
            
            # Fix indentation for the block
            fixed_block = []
            for block_line in special_block:
                # Ensure proper indentation (add block_indent)
                if block_line.strip():
                    # Remove any existing indentation
                    stripped = block_line.strip()
                    # Add correct indentation
                    fixed_block.append(f"{block_indent}{stripped}")
                else:
                    fixed_block.append("")
            
            # Add the fixed block to the result
            fixed_lines.extend(fixed_block)
            
            # Skip ahead to after the block
            i = special_block_end + 1
            continue
        
        # Add regular lines normally
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def generate_test_file(model_name, architecture=None, debug=False):
    """Generate a test file with fixed indentation for the given model."""
    # Determine architecture if not provided
    if architecture is None:
        architecture = determine_architecture(model_name)
    
    logger.info(f"Generating refactored test file for {model_name} with architecture {architecture}")
    
    # First create a temp file to analyze and fix
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Get template path
        template_path = get_template_path(architecture, refactored=True)
        logger.info(f"Using template: {template_path}")
        
        # Get sanitized model name for class names to avoid syntax errors
        sanitized_model_name = sanitize_model_name(model_name)
        
        # Customize the template to the temp file
        model_params = {
            "model_name": model_name,
            "sanitized_model_name": sanitized_model_name,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "architecture": architecture,
            "base_class": "ModelTest"
        }
        
        if not customize_template(template_path, temp_path, model_params):
            logger.error(f"Failed to customize template for {model_name}")
            return False
        
        # Now read the content, fix indentation issues, and write back
        with open(temp_path, 'r') as f:
            content = f.read()
        
        # Fix indentation
        fixed_content = fix_indentation(content, model_name, architecture)
        
        # Fix class name to prevent double capitalization
        fixed_content = fixed_content.replace(f"Test{sanitized_model_name.capitalize()}{architecture.capitalize()}", 
                                            f"Test{sanitized_model_name.capitalize()}")
        
        # Fix template placeholders that didn't get replaced
        fixed_content = fixed_content.replace(f"{{sanitized_model_name}}", sanitized_model_name)
        fixed_content = fixed_content.replace(f"{{model_name}}", model_name)
        
        # Ensure model_id is properly set
        fixed_content = fixed_content.replace('self.model_id = "MODEL_ID"', f'self.model_id = "{model_name}"')
        
        # Write back to temp file
        with open(temp_path, 'w') as f:
            f.write(fixed_content)
        
        # Verify syntax
        verify_result = verify_test_file(temp_path)
        if not verify_result["valid"]:
            logger.error(f"Syntax verification failed: {verify_result['error']}")
            if debug:
                logger.info(f"Debug content of the problematic file:")
                with open(temp_path, 'r') as f:
                    content = f.read()
                logger.info(f"\n{content}")
            return False
        
        # Generate final output path
        output_path = generate_output_path(model_name, architecture, refactored=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Copy from temp to final location
        with open(temp_path, 'r') as src, open(output_path, 'w') as dst:
            dst.write(src.read())
        
        logger.info(f"Successfully generated test file: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error generating test file: {e}")
        return False
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate refactored tests with fixed indentation")
    parser.add_argument("--model", type=str, help="Model name/ID to generate test for")
    parser.add_argument("--architecture", type=str, help="Model architecture (vision, text, speech, etc.)")
    parser.add_argument("--list-models", action="store_true", help="List available model architectures")
    parser.add_argument("--debug", action="store_true", help="Print debug information for failed tests")
    
    args = parser.parse_args()
    
    # List model types if requested
    if args.list_models:
        print("Available model architectures:")
        for model_type, arch in sorted(MODEL_ARCHITECTURE_MAPPING.items()):
            print(f"  - {model_type} -> {arch}")
        return 0
    
    # Ensure model is provided
    if not args.model:
        parser.error("--model is required unless --list-models is specified")
    
    # Generate the test file
    success = generate_test_file(args.model, args.architecture, debug=args.debug)
    
    # Print result
    if success:
        print(f"✅ Successfully generated test file for {args.model}")
        return 0
    else:
        print(f"❌ Failed to generate test file for {args.model}")
        return 1

if __name__ == "__main__":
    sys.exit(main())