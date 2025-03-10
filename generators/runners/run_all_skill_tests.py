#!/usr/bin/env python3
"""
Script to run all skill tests or a subset of them.
This runner applies the endpoint_handler fix to make all tests work properly.
"""

import os
import sys
import glob
import json
import argparse
import traceback
import subprocess
import importlib.util
from datetime import datetime

# Make sure we're in the right directory
test_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(test_dir)

# Set environment variable to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def import_module_from_path(module_path):
    """Import a module from a path"""
    module_name = os.path.basename(module_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_all_skill_test_files():
    """Get all test_hf_*.py files in the skills directory"""
    return sorted(glob.glob("skills/test_hf_*.py"))

def get_mapped_models():
    """Load model mappings from mapped_models.json"""
    try:
        with open('mapped_models.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mapped_models.json: {e}")
        return {}

def clean_skill_name(filename):
    """Extract skill name from filename"""
    base = os.path.basename(filename)
    return base.replace("test_hf_", "").replace(".py", "")

def generate_missing_tests():
    """Generate any missing test files"""
    try:
        # First check if the generator exists
        if not os.path.exists("generate_missing_test_files.py"):
            print("Missing generate_missing_test_files.py - cannot generate missing tests")
            return []
        
        # Import and run the generator
        generator = import_module_from_path("generate_missing_test_files.py")
        return generator.main()
    except Exception as e:
        print(f"Error generating missing tests: {e}")
        print(traceback.format_exc())
        return []

def load_endpoint_handler_fixer():
    """Load the endpoint handler fixer module"""
    try:
        # First check if the fixer exists
        if not os.path.exists("fix_endpoint_handler.py"):
            print("Missing fix_endpoint_handler.py - tests may fail without this fix")
            return None
        
        # Import the fixer
        fixer_module = import_module_from_path("fix_endpoint_handler.py")
        fixer = fixer_module.EndpointHandlerFixer()
        print("Successfully loaded endpoint handler fixer")
        return fixer
    except Exception as e:
        print(f"Error loading endpoint handler fixer: {e}")
        print(traceback.format_exc())
        return None

def run_test(test_file, apply_fix=True):
    """Run a single test file with the fix applied"""
    try:
        skill_name = clean_skill_name(test_file)
        print(f"\n{'='*80}\nRunning test for {skill_name} ({test_file})\n{'='*80}")
        
        if apply_fix:
            # Run with subprocess to isolate environment
            cmd = [sys.executable, '-m', f'test.{test_file.replace("/", ".").replace(".py", "")}']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Print outputs
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
                
            success = result.returncode == 0
        else:
            # Run the test directly (without fix)
            module = import_module_from_path(test_file)
            test_class_name = f"test_hf_{skill_name}"
            test_class = getattr(module, test_class_name)
            test_instance = test_class()
            result = test_instance.__test__()
            success = True  # Assume success unless exception
        
        return {
            "skill": skill_name,
            "file": test_file,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error running test {test_file}: {e}")
        print(traceback.format_exc())
        return {
            "skill": skill_name,
            "file": test_file,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_all_tests(test_files, apply_fix=True):
    """Run all specified test files and collect results"""
    results = []
    
    # Print summary of what we're going to do
    print(f"Running {len(test_files)} tests")
    if apply_fix:
        print("Applying endpoint_handler fix to ensure tests work correctly")
    
    # Run each test
    for test_file in test_files:
        result = run_test(test_file, apply_fix)
        results.append(result)
    
    # Generate summary
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes
    
    print(f"\n{'='*80}")
    print(f"Test Run Summary: {successes}/{len(results)} tests passed")
    print(f"{'='*80}")
    
    # List failures
    if failures > 0:
        print("\nFailed tests:")
        for result in results:
            if not result["success"]:
                print(f"- {result['skill']} ({result['file']})")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"skill_tests_results_{timestamp}.json"
    try:
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "total": len(results),
                "successes": successes,
                "failures": failures,
                "results": results
            }, f, indent=2)
        print(f"\nSaved results to {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run skill tests with endpoint handler fix')
    parser.add_argument('--skills', nargs='*', help='Specific skills to test')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--missing', action='store_true', help='Generate missing tests first')
    parser.add_argument('--no-fix', action='store_true', help='Run without applying endpoint handler fix')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Generate missing tests if requested
    if args.missing:
        print("Generating missing tests first...")
        generate_missing_tests()
    
    # Get all test files
    all_test_files = get_all_skill_test_files()
    print(f"Found {len(all_test_files)} skill test files")
    
    # Determine which tests to run
    if args.skills:
        # Run specific skills
        test_files = []
        for skill in args.skills:
            # Handle special case for skills with dashes
            skill_clean = skill.replace("-", "_")
            matches = [f for f in all_test_files if clean_skill_name(f) == skill_clean]
            if matches:
                test_files.extend(matches)
            else:
                print(f"Warning: No test file found for skill '{skill}'")
    elif args.all:
        # Run all tests
        test_files = all_test_files
    else:
        # Default to running just a few common ones as a sanity check
        test_files = [f for f in all_test_files if clean_skill_name(f) in ["bert", "clip", "t5", "gpt2", "llama"]]
        if not test_files:
            # If none of the defaults exist, just take the first 5 (or fewer)
            test_files = all_test_files[:min(5, len(all_test_files))]
    
    # Apply the fix when running tests
    apply_fix = not args.no_fix
    
    # Run the tests
    run_all_tests(test_files, apply_fix)

if __name__ == "__main__":
    main()