#!/usr/bin/env python3
"""
Script to generate tests for the critical priority models.
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"critical_models_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SKILLS_DIR, "fixed_tests")
MODELS_JSON = os.path.join(SKILLS_DIR, "critical_models.json")
GENERATOR_SCRIPT = os.path.join(SKILLS_DIR, "test_generator_fixed.py")

def syntax_check(file_path):
    """Check Python syntax of a file."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", file_path],
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stderr

def generate_model_test(model_name, architecture, template=None, default_model=None, task=None, original_name=None):
    """Generate a test file for a specific model."""
    logger.info(f"Generating test for {model_name}...")
    
    # Build command
    command = [
        sys.executable,
        GENERATOR_SCRIPT,
        "--generate", model_name,
        "--output-dir", OUTPUT_DIR
    ]
    
    if template:
        command.extend(["--template", template])
    
    if default_model:
        command.extend(["--model", default_model])
    
    if task:
        command.extend(["--task", task])
    
    # Handle hyphenated names
    if original_name and original_name != model_name:
        command.extend(["--original-name", original_name])
    
    # Run generator
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Generation failed for {model_name}: {result.stderr}")
            return False
        
        # Verify the file exists
        output_path = os.path.join(OUTPUT_DIR, f"test_hf_{model_name}.py")
        if not os.path.exists(output_path):
            logger.error(f"Generated file not found: {output_path}")
            return False
        
        # Check syntax
        syntax_valid, syntax_error = syntax_check(output_path)
        if not syntax_valid:
            logger.error(f"Syntax check failed for {model_name}: {syntax_error}")
            return False
        
        logger.info(f"✅ Successfully generated test for {model_name}: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Exception generating {model_name}: {e}")
        return False

def main():
    """Main function to generate critical model tests."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model definitions
    if not os.path.exists(MODELS_JSON):
        logger.error(f"Models JSON file not found: {MODELS_JSON}")
        return 1
    
    try:
        with open(MODELS_JSON, 'r') as f:
            model_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading models JSON: {e}")
        return 1
    
    # Generate tests for each model
    success_count = 0
    total_count = 0
    
    for architecture, models in model_data.items():
        for model in models:
            total_count += 1
            success = generate_model_test(
                model_name=model["name"],
                architecture=model["architecture"],
                template=model.get("template"),
                default_model=model.get("default_model"),
                task=model.get("task"),
                original_name=model.get("original_name")
            )
            if success:
                success_count += 1
    
    # Report results
    logger.info(f"\nGeneration complete: {success_count}/{total_count} models successfully generated")
    
    if success_count == total_count:
        logger.info("✅ All critical models were successfully generated!")
        return 0
    else:
        logger.warning(f"⚠️ {total_count - success_count} models failed to generate")
        return 1

if __name__ == "__main__":
    sys.exit(main())
