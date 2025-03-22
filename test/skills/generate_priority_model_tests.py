#!/usr/bin/env python3
"""
Generate Priority Model Tests

This script generates test files for missing high-priority HuggingFace models
based on the output from the missing model report.

It uses two approaches:
1. For hyphenated models, it uses simplified_fix_hyphenated.py
2. For regular models, it uses test_generator_fixed.py

Usage:
    python generate_priority_model_tests.py [--priority PRIORITY] [--directory TESTS_DIR]
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
import importlib
import importlib.util
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_missing_models(tests_dir, priority_level="critical"):
    """Get list of missing models by running the missing model analyzer."""
    # Create a temporary file for the report
    with tempfile.NamedTemporaryFile(suffix=".md") as temp_file:
        # Run the missing model analyzer
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(__file__), "generate_missing_model_report.py"),
            "--directory", tests_dir,
            "--report", temp_file.name
        ]
        subprocess.run(cmd, check=True)
        
        # Parse the report to extract missing models
        # This is a simplified approach; in practice, you might want to
        # extract this information more directly from the analyzer
        missing_models = []
        
        # Read the report
        with open(temp_file.name, 'r') as f:
            report_lines = f.readlines()
        
        # Find the section for the requested priority level
        for i, line in enumerate(report_lines):
            if line.strip() == f"### {priority_level.capitalize()} Priority Models":
                # Skip ahead to the model list
                j = i + 4  # Skip the header and the description
                while j < len(report_lines) and report_lines[j].strip() and report_lines[j].strip().startswith('- `'):
                    # Extract model name and architecture from the line, e.g., "- `model_name` (architecture)"
                    line = report_lines[j].strip()
                    parts = line.split('`')
                    if len(parts) > 1:
                        model_name = parts[1]
                        arch_part = parts[2].strip()
                        architecture = arch_part.strip('() ')
                        missing_models.append({
                            "name": model_name,
                            "architecture": architecture
                        })
                    j += 1
                break
    
    return missing_models

def generate_hyphenated_model_test(model_name, output_dir):
    """Generate a test file for a hyphenated model name."""
    logger.info(f"Generating hyphenated model test for {model_name}")
    
    # Get the path to the simplified_fix_hyphenated.py script
    script_path = os.path.join(os.path.dirname(__file__), "simplified_fix_hyphenated.py")
    
    # If the script exists, use it to generate the test file
    if os.path.exists(script_path):
        # Try to load the script as a module
        try:
            hyphenated_module = import_from_file("simplified_fix_hyphenated", script_path)
            
            # Check if the script has the create_hyphenated_test_file function
            if hasattr(hyphenated_module, "create_hyphenated_test_file"):
                # Call the function to create the test file
                success, error = hyphenated_module.create_hyphenated_test_file(model_name, output_dir)
                return success, error
            else:
                # If the function is not found, use subprocess to run the script
                return run_hyphenated_script(model_name, output_dir)
        except Exception as e:
            logger.error(f"Error importing hyphenated model script: {e}")
            # Fall back to subprocess
            return run_hyphenated_script(model_name, output_dir)
    else:
        logger.error(f"Hyphenated model script not found at {script_path}")
        return False, f"Script not found: {script_path}"

def run_hyphenated_script(model_name, output_dir):
    """Run the simplified_fix_hyphenated.py script as a subprocess."""
    script_path = os.path.join(os.path.dirname(__file__), "simplified_fix_hyphenated.py")
    
    try:
        # Run the script with subprocess
        result = subprocess.run(
            [sys.executable, script_path, "--model", model_name, "--output-dir", output_dir],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if the output file was created
        expected_file = os.path.join(output_dir, f"test_hf_{model_name.replace('-', '_')}.py")
        if os.path.exists(expected_file):
            return True, None
        else:
            return False, f"Expected file not created: {expected_file}"
    except subprocess.CalledProcessError as e:
        return False, f"Script execution failed: {e.stderr}"
    except Exception as e:
        return False, f"Error running script: {str(e)}"

def generate_regular_model_test(model_name, architecture, output_dir):
    """Generate a test file for a regular (non-hyphenated) model name using test_generator_fixed.py."""
    logger.info(f"Generating regular model test for {model_name} ({architecture})")
    
    # Get the path to the test_generator_fixed.py script
    script_path = os.path.join(os.path.dirname(__file__), "test_generator_fixed.py")
    
    # If the script exists, use it to generate the test file
    if os.path.exists(script_path):
        try:
            # Run the script with subprocess
            result = subprocess.run(
                [
                    sys.executable, 
                    script_path, 
                    "--model-type", model_name,
                    "--output-dir", output_dir,
                    "--architecture", architecture
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check if the output file was created
            expected_file = os.path.join(output_dir, f"test_hf_{model_name}.py")
            if os.path.exists(expected_file):
                return True, None
            else:
                return False, f"Expected file not created: {expected_file}"
        except subprocess.CalledProcessError as e:
            return False, f"Script execution failed: {e.stderr}"
        except Exception as e:
            return False, f"Error running script: {str(e)}"
    else:
        logger.error(f"Test generator script not found at {script_path}")
        return False, f"Script not found: {script_path}"

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate test files for missing high-priority HuggingFace models")
    parser.add_argument("--priority", type=str, default="critical", choices=["critical", "high", "medium"],
                        help="Priority level of models to generate (default: critical)")
    parser.add_argument("--directory", "-d", type=str, default="fixed_tests",
                        help="Directory to save generated test files (default: fixed_tests)")
    parser.add_argument("--models", type=str, nargs="+",
                        help="Specific models to generate (optional, overrides priority)")
    
    args = parser.parse_args()
    
    # Resolve output directory path
    output_dir = args.directory
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of models to generate
    models_to_generate = []
    
    if args.models:
        # If specific models are provided, use those
        for model_name in args.models:
            # Try to determine architecture
            if "-" in model_name:
                # For hyphenated models, try to determine architecture
                if "bert" in model_name or "roberta" in model_name:
                    architecture = "encoder-only"
                elif "gpt" in model_name or "llama" in model_name:
                    architecture = "decoder-only"
                elif "t5" in model_name or "bart" in model_name:
                    architecture = "encoder-decoder"
                elif "vit" in model_name or "swin" in model_name:
                    architecture = "vision"
                elif "clip" in model_name or "blip" in model_name:
                    architecture = "vision-text"
                elif "wav2vec" in model_name or "whisper" in model_name:
                    architecture = "speech"
                else:
                    architecture = "unknown"
            else:
                # For regular models, use a default architecture
                # This is a simplification; in practice, you would want to determine this more accurately
                architecture = "unknown"
            
            models_to_generate.append({
                "name": model_name,
                "architecture": architecture
            })
    else:
        # Otherwise, get missing models by priority
        models_to_generate = get_missing_models(output_dir, args.priority)
    
    logger.info(f"Found {len(models_to_generate)} models to generate")
    
    # Initialize counters
    success_count = 0
    failure_count = 0
    
    # Generate test files for each model
    for model in models_to_generate:
        model_name = model["name"]
        architecture = model["architecture"]
        
        # Determine if this is a hyphenated model
        if "-" in model_name:
            # Generate hyphenated model test
            success, error = generate_hyphenated_model_test(model_name, output_dir)
        else:
            # Generate regular model test
            success, error = generate_regular_model_test(model_name, architecture, output_dir)
        
        # Update counters
        if success:
            logger.info(f"✅ Successfully generated test for {model_name}")
            success_count += 1
        else:
            logger.error(f"❌ Failed to generate test for {model_name}: {error}")
            failure_count += 1
    
    # Print summary
    logger.info(f"Generated {success_count} test files successfully, {failure_count} failed")
    
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())