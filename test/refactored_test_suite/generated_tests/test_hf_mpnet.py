#!/usr/bin/env python3
"""
Test file for MPNET models (encoder-only architecture).

This test verifies the functionality of the mpnet model.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model test base
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refactored_test_suite.model_test_base import EncoderOnlyModelTest


class TestMpnetModel(EncoderOnlyModelTest):
    """Test class for mpnet model."""
    
    def __init__(self, model_id=None, device=None):
        """Initialize the test."""
        # Set model type explicitly
        self.model_type = "mpnet"
        self.task = "fill-mask"
        self.architecture_type = "encoder-only"
        
        # Call parent initializer
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        """Get the default model ID for this model type."""
        return "mpnet"
    
    def run_all_tests(self):
        """Run all tests for this model."""
        # Run basic tests through parent class
        results = self.run_tests()
        
        # Optionally run additional model-specific tests
        # For example:
        # model_data = self.load_model()
        # custom_input = "This is a custom [MASK] for testing."
        # custom_verification = self.verify_model_output(model_data, custom_input)
        # results["custom_verification"] = custom_verification
        
        return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test mpnet model")
    parser.add_argument("--model-id", type=str, help="Specific model ID to test")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Device to test on")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for results")
    
    args = parser.parse_args()
    
    # Create and run the test
    test = TestMpnetModel(args.model_id, args.device)
    results = test.run_all_tests()
    
    # Print a summary
    success = results.get("model_loading", {}).get("success", False)
    model_id = results.get("metadata", {}).get("model", test.model_id)
    device = results.get("metadata", {}).get("device", test.device)
    
    if success:
        print(f"✅ Successfully tested {model_id} on {device}")
    else:
        print(f"❌ Failed to test {model_id} on {device}")
        error = results.get("model_loading", {}).get("error", "Unknown error")
        print(f"Error: {error}")
    
    # Save results if requested
    if args.save:
        output_path = test.save_results(args.output_dir)
        if output_path:
            print(f"Results saved to {output_path}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())