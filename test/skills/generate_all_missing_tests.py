#!/usr/bin/env python3
"""
Batch Generator for All Missing Hugging Face Tests

This script will generate test files for all missing Hugging Face model types
in batches to ensure full coverage of all 300 model types in 
huggingface_model_types.json.

The script will:
1. Identify all model types without test files
2. Generate test files in batches of specified size
3. Create a summary of test coverage

Usage:
  python generate_all_missing_tests.py --batch-size 10  # Generate tests in batches of 10
  python generate_all_missing_tests.py --all            # Generate all missing tests
  python generate_all_missing_tests.py --verify         # Verify test coverage after generation
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
GENERATOR_SCRIPT = CURRENT_DIR / "generate_missing_hf_tests.py"

def generate_batch(batch_size=10, total_count=None, start_index=0):
    """Generate a batch of test files."""
    cmd = ["python", str(GENERATOR_SCRIPT), "--batch", str(batch_size)]
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error generating batch: {process.stderr}")
        return False
    
    print(process.stdout)
    
    # Extract number of models generated from output
    generated = 0
    for line in process.stdout.splitlines():
        if "Generated" in line and "test files" in line:
            parts = line.split("Generated ")[1].split(" test files")[0]
            try:
                generated = int(parts)
            except ValueError:
                generated = 0
    
    # Return whether we successfully generated tests
    return generated > 0

def generate_all_missing():
    """Generate all missing test files in batches."""
    print("Generating all missing test files in batches of 10...")
    
    # Initial report to see how many we need to generate
    cmd = ["python", str(GENERATOR_SCRIPT), "--report"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error getting initial report: {process.stderr}")
        return
    
    # Parse output for missing tests count
    missing_count = 0
    for line in process.stdout.splitlines():
        if "Missing Tests:" in line:
            parts = line.split("Missing Tests:")[1].strip()
            try:
                missing_count = int(parts)
            except ValueError:
                missing_count = 0
    
    print(f"Found {missing_count} missing test files to generate")
    
    # Generate in batches until all are done
    batch_size = 10
    total_generated = 0
    
    while total_generated < missing_count:
        print(f"\nGenerating batch {total_generated//batch_size + 1} of {(missing_count+batch_size-1)//batch_size}...")
        
        start_time = time.time()
        success = generate_batch(batch_size)
        elapsed_time = time.time() - start_time
        
        if success:
            total_generated += batch_size
            print(f"Generated {min(total_generated, missing_count)}/{missing_count} test files")
            print(f"Batch took {elapsed_time:.1f} seconds")
            
            # Sleep briefly to avoid overwhelming the system
            time.sleep(1)
        else:
            print("Batch generation failed, trying again...")
            time.sleep(2)
            
        # If we've generated all or more, break
        if total_generated >= missing_count:
            break
    
    # Final report
    print("\nFinal coverage report:")
    cmd = ["python", str(GENERATOR_SCRIPT), "--report"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    print(process.stdout)
    
    print("\nAll missing test files have been generated!")

def verify_coverage():
    """Verify current test coverage."""
    cmd = ["python", str(GENERATOR_SCRIPT), "--report"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error verifying coverage: {process.stderr}")
        return
    
    print(process.stdout)

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate all missing Hugging Face test files")
    
    # Options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch-size", type=int, help="Generate tests in batches of specified size")
    group.add_argument("--all", action="store_true", help="Generate all missing tests")
    group.add_argument("--verify", action="store_true", help="Verify test coverage")
    
    args = parser.parse_args()
    
    # Verify generator script exists
    if not os.path.exists(GENERATOR_SCRIPT):
        print(f"Error: Generator script not found at {GENERATOR_SCRIPT}")
        return 1
    
    # Generate a batch
    if args.batch_size:
        print(f"Generating a batch of {args.batch_size} test files...")
        generate_batch(args.batch_size)
    
    # Generate all missing tests
    elif args.all:
        generate_all_missing()
    
    # Verify coverage
    elif args.verify:
        verify_coverage()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())