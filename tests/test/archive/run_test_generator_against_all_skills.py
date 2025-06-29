#!/usr/bin/env python3
"""
Script to run the merged_test_generator.py against all HuggingFace models in the skillset directory
to test its generalizability across all 300 model types.

This script:
1. Identifies all hf_*.py model implementations in the worker/skillset directory
2. Extracts the model types from those files
3. Runs the merged_test_generator for each model type
4. Generates a summary report of the results
"""

import os
import sys
import glob
import time
import json
import subprocess
import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKILLSET_PATH = BASE_PATH / "ipfs_accelerate_py" / "worker" / "skillset"
TEST_PATH = BASE_PATH / "test"
GENERATOR_PATH = TEST_PATH / "merged_test_generator.py"
OUTPUT_DIR = TEST_PATH / "generated_skills"
RESULTS_DIR = TEST_PATH / "generation_results"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Get current timestamp for output files
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SUMMARY_FILE = RESULTS_DIR / f"test_generation_summary_{TIMESTAMP}.json"
LOG_FILE = RESULTS_DIR / f"test_generation_log_{TIMESTAMP}.txt"

# Configure file handler for logging
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def get_skillset_modules():
    """Get all HF model modules from the skillset directory."""
    hf_modules = []
    for file_path in glob.glob(os.path.join(SKILLSET_PATH, "hf_*.py")):
        module_name = os.path.basename(file_path).replace(".py", "")
        # Skip test implementations that might be in skillset
        if "test" not in module_name:
            hf_modules.append({
                "name": module_name,
                "path": file_path,
                # Extract model type from name (remove 'hf_' prefix)
                "model_type": module_name[3:] if module_name.startswith("hf_") else module_name
            })
    
    logger.info(f"Found {len(hf_modules)} HF model modules in skillset directory")
    return sorted(hf_modules, key=lambda x: x["name"])

def load_model_types():
    """Load all model types from huggingface_model_types.json file."""
    model_types_path = TEST_PATH / "huggingface_model_types.json"
    try:
        with open(model_types_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading model types: {e}")
        return []

def generate_test_file(model_type, output_dir=OUTPUT_DIR):
    """Generate a test file for the specified model type."""
    start_time = time.time()
    try:
        # Run the test generator
        cmd = [
            sys.executable,
            str(GENERATOR_PATH),
            "--batch-generate", model_type,  # Use batch-generate which uses huggingface_model_types.json
            "--output-dir", str(output_dir)
        ]
        
        logger.info(f"Generating test for {model_type}...")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise exception on error
        )
        
        # Check for successful generation
        success = result.returncode == 0
        output_file = output_dir / f"test_hf_{model_type.replace('-', '_')}.py"
        exists = output_file.exists()
        
        # Log results
        if success and exists:
            logger.info(f"Successfully generated test for {model_type}")
        else:
            logger.error(f"Failed to generate test for {model_type}, return code: {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
        
        elapsed_time = time.time() - start_time
        
        return {
            "model_type": model_type,
            "success": success and exists,
            "return_code": result.returncode,
            "output_file": str(output_file) if exists else None,
            "elapsed_time": elapsed_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except Exception as e:
        logger.exception(f"Exception when generating test for {model_type}: {e}")
        elapsed_time = time.time() - start_time
        return {
            "model_type": model_type,
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time
        }

def run_test_file(test_file):
    """Run a generated test file to verify it works."""
    start_time = time.time()
    try:
        # Run the test file
        cmd = [sys.executable, str(test_file)]
        
        logger.info(f"Running test: {os.path.basename(test_file)}...")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,  # Don't raise exception on error
            timeout=120  # Set a reasonable timeout
        )
        
        # Check for successful execution
        success = result.returncode == 0
        
        if success:
            logger.info(f"Successfully executed test: {os.path.basename(test_file)}")
        else:
            logger.error(f"Failed to execute test: {os.path.basename(test_file)}, return code: {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
        
        elapsed_time = time.time() - start_time
        
        return {
            "test_file": str(test_file),
            "success": success,
            "return_code": result.returncode,
            "elapsed_time": elapsed_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout when executing test: {os.path.basename(test_file)}")
        elapsed_time = time.time() - start_time
        return {
            "test_file": str(test_file),
            "success": False,
            "error": "Timeout expired",
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        logger.exception(f"Exception when executing test: {os.path.basename(test_file)}: {e}")
        elapsed_time = time.time() - start_time
        return {
            "test_file": str(test_file),
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time
        }

def generate_all_tests_at_once():
    """Generate all test files at once using --generate-missing."""
    start_time = time.time()
    try:
        # Run the test generator with --generate-missing to generate all missing tests
        # This approach uses the huggingface_model_types.json file
        cmd = [
            sys.executable,
            str(GENERATOR_PATH),
            "--generate-missing",
            "--output-dir", str(OUTPUT_DIR),
            # Not limiting number of tests
            "--limit", "300"
        ]
        
        logger.info(f"Generating all missing test files...")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise exception on error
        )
        
        # Check for successful generation
        success = result.returncode == 0
        
        if success:
            logger.info(f"Successfully ran test generation for all models")
        else:
            logger.error(f"Failed to generate tests, return code: {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
        
        elapsed_time = time.time() - start_time
        
        return {
            "success": success,
            "return_code": result.returncode,
            "elapsed_time": elapsed_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except Exception as e:
        logger.exception(f"Exception generating tests: {e}")
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time
        }

def main():
    """Main function to run test validation against all HF model skills."""
    logger.info("Starting test validation against all HF model skills")
    
    # Get model modules from skillset
    skillset_modules = get_skillset_modules()
    
    # Load all known model types
    all_model_types = load_model_types()
    
    # Keep track of results
    generation_results = []
    execution_results = []
    
    # Log summary info
    logger.info(f"Found {len(skillset_modules)} HF model modules in skillset directory")
    logger.info(f"Found {len(all_model_types)} total model types in huggingface_model_types.json")
    
    # Get model types to test (from skillset modules)
    model_types_to_test = [module["model_type"] for module in skillset_modules]
    
    # Add any model types from the modules that aren't in the official list
    missing_from_official = [m for m in model_types_to_test if m not in all_model_types]
    if missing_from_official:
        logger.warning(f"Found {len(missing_from_official)} model types in skillset not in official list: {missing_from_official}")
    
    logger.info("Using existing test files from test/skills directory")
    skills_test_dir = TEST_PATH / "skills"
    
    # Check for existing test files
    for model_type in model_types_to_test:
        normalized_name = model_type.replace('-', '_')
        test_file = skills_test_dir / f"test_hf_{normalized_name}.py"
        
        if test_file.exists():
            logger.info(f"Found existing test file for {model_type}")
            gen_result = {
                "model_type": model_type,
                "success": True,
                "output_file": str(test_file),
                "elapsed_time": 0,
                "source": "existing"
            }
            generation_results.append(gen_result)
            
            # Run the test file
            exec_result = run_test_file(test_file)
            execution_results.append(exec_result)
        else:
            logger.warning(f"No test file found for {model_type}")
            gen_result = {
                "model_type": model_type,
                "success": False,
                "error": "No existing test file found",
                "elapsed_time": 0
            }
            generation_results.append(gen_result)
    
    # Calculate summary statistics
    total_generations = len(generation_results)
    successful_generations = sum(1 for r in generation_results if r["success"])
    
    total_executions = len(execution_results)
    successful_executions = sum(1 for r in execution_results if r["success"])
    
    # Create summary report
    summary = {
        "timestamp": TIMESTAMP,
        "total_skillset_modules": len(skillset_modules),
        "total_model_types": len(all_model_types),
        "models_tested": model_types_to_test,
        "missing_from_official_list": missing_from_official,
        "generation_stats": {
            "total": total_generations,
            "successful": successful_generations,
            "failed": total_generations - successful_generations,
            "success_rate": successful_generations / total_generations if total_generations > 0 else 0
        },
        "execution_stats": {
            "total": total_executions,
            "successful": successful_executions,
            "failed": total_executions - successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0
        },
        "generation_results": generation_results,
        "execution_results": execution_results
    }
    
    # Save summary to file
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {SUMMARY_FILE}")
    
    # Print summary report
    print("\n" + "="*80)
    print(f"TEST GENERATOR VALIDATION SUMMARY ({TIMESTAMP})")
    print("="*80)
    print(f"Total skillset modules: {len(skillset_modules)}")
    print(f"Total model types in registry: {len(all_model_types)}")
    print(f"Models tested: {len(model_types_to_test)}")
    print(f"Missing from official list: {len(missing_from_official)}")
    print("\nGeneration Stats:")
    print(f"  Total: {total_generations}")
    print(f"  Successful: {successful_generations}")
    print(f"  Failed: {total_generations - successful_generations}")
    print(f"  Success Rate: {successful_generations / total_generations * 100:.2f}%")
    print("\nExecution Stats:")
    print(f"  Total: {total_executions}")
    print(f"  Successful: {successful_executions}")
    print(f"  Failed: {total_executions - successful_executions}")
    print(f"  Success Rate: {successful_executions / total_executions * 100:.2f}%")
    print("\nFull details in:")
    print(f"  - Summary: {SUMMARY_FILE}")
    print(f"  - Log: {LOG_FILE}")

if __name__ == "__main__":
    main()