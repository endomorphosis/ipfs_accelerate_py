#!/usr/bin/env python3
"""
Test script for model inference validation.

This script tests the inference validation functionality with a specific model.
It can be used to verify that the validation logic works correctly.

Usage:
    python test_inference_validation.py MODEL_NAME
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FIXED_TESTS_DIR = CURRENT_DIR / "fixed_tests"
TEST_RESULTS_DIR = CURRENT_DIR / "test_results"

# Create test results directory
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

def to_valid_identifier(text):
    """Convert hyphenated model names to valid Python identifiers."""
    return text.replace("-", "_")

def run_inference_test(model_name, use_small=True):
    """Run inference validation test for a specific model."""
    try:
        # Import the validation modules directly
        sys.path.insert(0, str(CURRENT_DIR))
        from validate_model_inference import validate_model_inference
        
        # Construct file path for the model's test file
        model_id = to_valid_identifier(model_name)
        test_file = FIXED_TESTS_DIR / f"test_hf_{model_id}.py"
        
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            return False, f"Test file not found: {test_file}"
        
        # Run the inference validation
        logger.info(f"Running inference validation for {model_name} with test file {test_file}")
        validation_results = validate_model_inference(model_name, str(test_file), use_small)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = TEST_RESULTS_DIR / f"inference_test_{model_id}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
            
        logger.info(f"Results saved to {results_file}")
        
        # Check results
        success = validation_results.get("success", False)
        message = "Inference validation passed" if success else "Inference validation failed"
        
        if not success:
            issues = validation_results.get("issues", [])
            message += f": {'; '.join(issues)}" if issues else ""
        
        return success, message
        
    except ImportError as e:
        logger.error(f"Error importing validation modules: {str(e)}")
        return False, f"Error importing validation modules: {str(e)}"
    except Exception as e:
        logger.error(f"Error running inference test: {str(e)}")
        return False, f"Error running inference test: {str(e)}"

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test inference validation functionality")
    parser.add_argument("model", type=str, help="Model name to test")
    parser.add_argument("--use-small", action="store_true", help="Use small model variant for testing")
    
    args = parser.parse_args()
    
    # Run the inference test
    success, message = run_inference_test(args.model, args.use_small)
    
    # Print result
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\nInference validation test for {args.model}: {status}")
    print(f"Message: {message}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())