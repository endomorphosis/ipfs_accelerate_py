#!/usr/bin/env python3
"""
Script to generate sample test files using the template_test_generator.py.
This tests a representative subset of models from different categories.
"""

import os
import sys
import subprocess
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
OUTPUT_DIR = TEST_DIR / "sample_tests"

# Sample models from different categories
SAMPLE_MODELS = [
    "bert",        # Language model
    "vit",         # Vision model
    "whisper",     # Audio model
    "llava",       # Multimodal model
]

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
        print(f"Output: {result.stdout}")
        return True
    else:
        print(f"Test failed for {model_type}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

def main():
    """Main function to generate and run tests for sample models."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating tests in: {OUTPUT_DIR}")
    
    # Track results
    generation_results = {}
    test_results = {}
    
    # Generate and run tests for each model
    for model_type in SAMPLE_MODELS:
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
    print("SAMPLE TEST GENERATION SUMMARY")
    print("="*80)
    
    print("\nGeneration Results:")
    for model, success in generation_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  - {model}: {status}")
    
    gen_success_count = sum(1 for success in generation_results.values() if success)
    print(f"\nGenerated {gen_success_count} of {len(SAMPLE_MODELS)} test files successfully ({gen_success_count/len(SAMPLE_MODELS)*100:.1f}%)")
    
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
    overall_success = test_success_count / len(SAMPLE_MODELS) if len(SAMPLE_MODELS) > 0 else 0
    print(f"\nOverall Success Rate: {overall_success*100:.1f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())