#!/usr/bin/env python3
"""
Script to generate test files for all models in the worker/skillset directory
using the template_test_generator.py.

This serves as a demonstration of how the test generator can work across
all model types in the worker/skillset directory.
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
WORKER_SKILLSET = PROJECT_ROOT / "ipfs_accelerate_py" / "worker" / "skillset"
OUTPUT_DIR = TEST_DIR / "generated_worker_tests"

def get_skillset_models():
    """Get all HF model types from the worker/skillset directory."""
    models = []
    
    for file_path in glob.glob(str(WORKER_SKILLSET / "hf_*.py")):
        filename = os.path.basename(file_path)
        if "test" not in filename:  # Skip test files
            model_type = filename[3:-3]  # Remove 'hf_' prefix and '.py' suffix
            models.append(model_type)
    
    return sorted(models)

def generate_test(model_type, output_dir=OUTPUT_DIR, force=False):
    """Generate a test file for the specified model type."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the command
    cmd = [
        sys.executable,
        str(TEST_DIR / "template_test_generator.py"),
        "--model", model_type,
        "--output-dir", str(output_dir)
    ]
    
    if force:
        cmd.append("--force")
    
    # Run the generator
    print(f"Generating test for {model_type}...")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    
    # Check if generation was successful
    if result.returncode == 0:
        print(f"Successfully generated test for {model_type}")
        return True
    else:
        print(f"Failed to generate test for {model_type}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

def run_test(model_type, output_dir=OUTPUT_DIR):
    """Run the generated test file for the specified model type."""
    test_file = output_dir / f"test_hf_{model_type}.py"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return False
    
    # Run the test
    print(f"Running test for {model_type}...")
    result = subprocess.run(
        [sys.executable, str(test_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30  # Set a 30-second timeout
    )
    
    # Check if test was successful
    if result.returncode == 0:
        print(f"Successfully ran test for {model_type}")
        return True
    else:
        print(f"Test failed for {model_type}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

def main():
    """Main function to generate and run tests for all skillset models."""
    # Get all model types from worker/skillset
    models = get_skillset_models()
    print(f"Found {len(models)} models in worker/skillset: {', '.join(models)}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating tests in: {OUTPUT_DIR}")
    
    # Track results
    generation_results = {}
    test_results = {}
    
    # Generate and run tests for each model
    for model_type in models:
        # Generate test
        gen_success = generate_test(model_type, OUTPUT_DIR, force=True)
        generation_results[model_type] = gen_success
        
        # Run test if generation was successful
        if gen_success:
            test_success = run_test(model_type, OUTPUT_DIR)
            test_results[model_type] = test_success
        else:
            test_results[model_type] = False
    
    # Print summary
    print("\n" + "="*80)
    print("TEST GENERATION SUMMARY")
    print("="*80)
    
    print("\nGeneration Results:")
    for model, success in generation_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  - {model}: {status}")
    
    gen_success_count = sum(1 for success in generation_results.values() if success)
    print(f"\nGenerated {gen_success_count} of {len(models)} test files successfully ({gen_success_count/len(models)*100:.1f}%)")
    
    print("\nTest Execution Results:")
    for model, success in test_results.items():
        if model in generation_results and generation_results[model]:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  - {model}: {status}")
    
    test_success_count = sum(1 for success in test_results.values() if success)
    gen_success_count = sum(1 for success in generation_results.values() if success)
    if gen_success_count > 0:
        print(f"\nExecuted {test_success_count} of {gen_success_count} tests successfully ({test_success_count/gen_success_count*100:.1f}%)")
    
    # Print overall success rate
    overall_success = test_success_count / len(models) if len(models) > 0 else 0
    print(f"\nOverall Success Rate: {overall_success*100:.1f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())