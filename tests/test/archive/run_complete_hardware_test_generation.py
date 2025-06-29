#!/usr/bin/env python3
"""
Complete Hardware Test Generation Script

This script improves all test generators, template handlers, and hardware detection,
and then runs a sequence of commands to generate tests for all 13 key model types
with proper hardware support for all platforms.

Usage:
  python run_complete_hardware_test_generation.py [--all] [--force]
"""

import os
import sys
import logging
import subprocess
import argparse
import time
import concurrent.futures
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hardware_test_coverage.log')
    ]
)

logger = logging.getLogger("complete_hardware_test_generation")

def run_command(command, description):
    """Run a shell command and log the result."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Command succeeded in {elapsed_time:.2f} seconds")
        logger.info(f"Output: {result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False, e.stderr

def find_key_model_templates():
    """Find all key model templates in the hardware_test_templates directory."""
    template_dir = Path("hardware_test_templates")
    if not template_dir.exists() or not template_dir.is_dir():
        logger.warning(f"Template directory not found: {template_dir}")
        return []
    
    templates = []
    for template_file in template_dir.glob("template_*.py"):
        if template_file.name != "template_database.py" and template_file.name != "__init__.py":
            model_name = template_file.name.replace("template_", "").replace(".py", "")
            templates.append(model_name)
    
    return templates

def run_hardware_enhancement_steps(force=False):
    """Run all the hardware enhancement steps in sequence."""
    steps = [
        {
            "command": "./fix_generator_hardware_support.py" + (" --force" if force else ""),
            "description": "Fix hardware support in generators"
        },
        {
            "command": "./update_test_generator_with_hardware_templates.py",
            "description": "Update generators with hardware templates"
        },
        {
            "command": "chmod +x fix_hardware_integration.py && ./fix_hardware_integration.py --all-key-models --analyze-only --output-json hardware_analysis.json",
            "description": "Analyze hardware integration for all key models"
        }
    ]
    
    success_count = 0
    for step in steps:
        success, output = run_command(step["command"], step["description"])
        if success:
            success_count += 1
        else:
            logger.error(f"Step failed: {step['description']}")
    
    return success_count == len(steps)

def generate_key_model_tests(all_models=False):
    """Generate tests for key models with hardware support."""
    # Find key model templates
    key_models = find_key_model_templates()
    if not key_models:
        logger.warning("No key model templates found.")
        key_models = [
            "bert", "t5", "llama", "vit", "clip", "detr", 
            "clap", "wav2vec2", "whisper", "llava", "llava_next", 
            "xclip", "qwen2"
        ]
    
    logger.info(f"Generating tests for key models: {', '.join(key_models)}")
    
    # Generate tests for each key model
    success_count = 0
    for model in key_models:
        command = f"python merged_test_generator.py --generate {model} --platform all"
        success, output = run_command(command, f"Generate test for {model}")
        if success:
            success_count += 1
    
    # Generate tests for additional models if requested
    if all_models:
        command = "python merged_test_generator.py --generate-missing --high-priority-only --limit 20"
        success, output = run_command(command, "Generate tests for high priority models")
        if success:
            success_count += 1
    
    return success_count == len(key_models) + (1 if all_models else 0)

def run_parallel_integrations(models, template=None):
    """Run integration steps in parallel for multiple models."""
    results = []
    
    def process_model(model):
        if template:
            # Use integrated_skillset_generator with a template
            command = f"python integrated_skillset_generator.py --model {model} --cross-platform --hardware all"
        else:
            # Use the merged_test_generator
            command = f"python merged_test_generator.py --generate {model} --platform all"
        
        description = f"Generate {model} test with {'integrated_skillset_generator' if template else 'merged_test_generator'}"
        success, output = run_command(command, description)
        return {
            "model": model,
            "success": success,
            "output": output
        }
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_model, model) for model in models]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if result["success"]:
                    logger.info(f"Successfully generated test for {result['model']}")
                else:
                    logger.error(f"Failed to generate test for {result['model']}")
            except Exception as e:
                logger.error(f"Error processing model: {e}")
    
    return results

def generate_benchmark_db_integration():
    """Generate and run benchmarks with database integration."""
    steps = [
        {
            "command": "python benchmark_db_maintenance.py --check-integrity",
            "description": "Check benchmark database integrity"
        },
        {
            "command": "python benchmark_hardware_models.py --key-models-only --hardware cpu,cuda --db-path ./benchmark_db.duckdb",
            "description": "Benchmark key models with DB integration"
        },
        {
            "command": "python benchmark_hardware_performance.py --output-dir ./benchmark_results --model-type text,vision --db-path ./benchmark_db.duckdb",
            "description": "Run hardware performance benchmark with DB integration"
        }
    ]
    
    success_count = 0
    for step in steps:
        success, output = run_command(step["command"], step["description"])
        if success:
            success_count += 1
        else:
            logger.error(f"Benchmark step failed: {step['description']}")
    
    return success_count == len(steps)

def main():
    """Main function to run the complete hardware test generation."""
    parser = argparse.ArgumentParser(description="Complete Hardware Test Generation")
    parser.add_argument("--all", action="store_true", help="Generate tests for all models, not just key models")
    parser.add_argument("--force", action="store_true", help="Force updates even if hardware support is already present")
    parser.add_argument("--skip-enhancement", action="store_true", help="Skip the hardware enhancement steps")
    parser.add_argument("--skip-generation", action="store_true", help="Skip the test generation steps")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip the benchmark integration steps")
    parser.add_argument("--parallel", action="store_true", help="Run test generation in parallel")
    args = parser.parse_args()
    
    # Print header
    logger.info("=" * 80)
    logger.info("COMPLETE HARDWARE TEST GENERATION")
    logger.info("=" * 80)
    logger.info(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Options: all={args.all}, force={args.force}, skip_enhancement={args.skip_enhancement}, skip_generation={args.skip_generation}, skip_benchmarks={args.skip_benchmarks}")
    
    # Step 1: Run hardware enhancement steps
    if not args.skip_enhancement:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: HARDWARE ENHANCEMENT")
        logger.info("=" * 40)
        
        enhancement_success = run_hardware_enhancement_steps(force=args.force)
        if enhancement_success:
            logger.info("Hardware enhancement steps completed successfully.")
        else:
            logger.error("Hardware enhancement steps failed. Continuing with generation...")
    
    # Step 2: Generate tests for key models
    if not args.skip_generation:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: TEST GENERATION")
        logger.info("=" * 40)
        
        # Define key models
        key_models = [
            "bert", "t5", "llama", "vit", "clip", "detr", 
            "clap", "wav2vec2", "whisper", "llava", "llava_next", 
            "xclip", "qwen2"
        ]
        
        if args.parallel:
            # Run in parallel
            logger.info("Running test generation in parallel...")
            results = run_parallel_integrations(key_models)
            success_count = sum(1 for result in results if result["success"])
            logger.info(f"Successfully generated {success_count} of {len(key_models)} key model tests in parallel.")
        else:
            # Run sequentially
            generation_success = generate_key_model_tests(all_models=args.all)
            if generation_success:
                logger.info("Test generation completed successfully.")
            else:
                logger.error("Test generation failed for some models.")
    
    # Step 3: Run benchmarks with database integration
    if not args.skip_benchmarks:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: BENCHMARK DB INTEGRATION")
        logger.info("=" * 40)
        
        benchmark_success = generate_benchmark_db_integration()
        if benchmark_success:
            logger.info("Benchmark database integration completed successfully.")
        else:
            logger.error("Benchmark database integration failed for some steps.")
    
    # Print summary
    logger.info("\n" + "=" * 40)
    logger.info("COMPLETION SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Hardware test generation process completed.")
    logger.info("Check 'hardware_test_coverage.log' for detailed logs.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())