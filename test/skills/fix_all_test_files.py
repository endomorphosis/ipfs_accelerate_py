#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test(model_type, output_dir):
    """Generate a test for a model type."""
    try:
        cmd = [
            "python", 
            "test_generator_fixed.py", 
            "--generate", 
            model_type, 
            "--output-dir", 
            output_dir
        ]
        
        logger.info(f"Generating test for {model_type}...")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        if proc.returncode != 0:
            logger.error(f"Error generating test for {model_type}: {proc.stderr}")
            return False
        
        logger.info(f"Successfully generated test for {model_type}")
        return True
    except Exception as e:
        logger.error(f"Exception generating test for {model_type}: {e}")
        return False

def fix_test_file(file_path):
    """Fix a test file using the template fixer."""
    try:
        cmd = [
            "python", 
            "fix_template_imports.py", 
            "--file", 
            file_path
        ]
        
        logger.info(f"Fixing test file {file_path}...")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        if proc.returncode != 0:
            logger.error(f"Error fixing {file_path}: {proc.stderr}")
            return False
        
        logger.info(f"Successfully fixed {file_path}")
        return True
    except Exception as e:
        logger.error(f"Exception fixing {file_path}: {e}")
        return False

def verify_syntax(file_path):
    """Verify the syntax of a test file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile the file to check syntax
        compile(content, file_path, 'exec')
        logger.info(f"✅ {file_path}: Syntax is valid")
        return True
    except SyntaxError as e:
        logger.error(f"❌ {file_path}: Syntax error: {e}")
        logger.error(f"At line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        logger.error(f"Error verifying syntax of {file_path}: {e}")
        return False

def process_model(model_type, output_dir):
    """Process a single model: generate, fix, and verify."""
    # Generate the test
    if not generate_test(model_type, output_dir):
        return (model_type, False, "Generation failed")
    
    # Determine the output file path
    file_path = os.path.join(output_dir, f"test_hf_{model_type.replace('-', '_')}.py")
    
    # Fix the test file
    if not fix_test_file(file_path):
        return (model_type, False, "Fixing failed")
    
    # Verify the syntax
    if not verify_syntax(file_path):
        return (model_type, False, "Syntax verification failed")
    
    return (model_type, True, "Success")

def main():
    parser = argparse.ArgumentParser(description="Generate and fix all model tests")
    parser.add_argument("--output-dir", type=str, default="fixed_tests", help="Output directory for test files")
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to process")
    parser.add_argument("--hyphenated-only", action="store_true", help="Process only hyphenated model names")
    parser.add_argument("--all", action="store_true", help="Process all models")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    
    args = parser.parse_args()
    
    if not (args.models or args.hyphenated_only or args.all):
        parser.error("At least one of --models, --hyphenated-only, or --all must be specified")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the list of models to process
    if args.models:
        models_to_process = args.models
    else:
        # Run a separate process to get the list of models
        try:
            import_cmd = ["python", "-c", 
                "from test_generator_fixed import list_model_families, get_hyphenated_model_families; "
                "import json; "
                "print(json.dumps(list_model_families() if True else [])) if not True else "
                f"print(json.dumps(get_hyphenated_model_families() if {args.hyphenated_only} else list_model_families()))"
            ]
            
            proc = subprocess.run(import_cmd, capture_output=True, text=True)
            
            if proc.returncode != 0:
                logger.error(f"Error getting model list: {proc.stderr}")
                return 1
            
            import json
            models_to_process = json.loads(proc.stdout)
        except Exception as e:
            logger.error(f"Error importing model families: {e}")
            return 1
    
    logger.info(f"Processing {len(models_to_process)} models")
    
    # Process the models
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_model, model_type, args.output_dir)
            for model_type in models_to_process
        ]
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Summarize the results
    successful = [r[0] for r in results if r[1]]
    failed = [(r[0], r[2]) for r in results if not r[1]]
    
    logger.info(f"Process summary: {len(successful)} models successful, {len(failed)} models failed")
    
    if failed:
        logger.info("Failed models:")
        for model, reason in failed:
            logger.info(f"  - {model}: {reason}")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())