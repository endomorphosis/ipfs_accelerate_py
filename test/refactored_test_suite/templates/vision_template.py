#!/usr/bin/env python3
"""
Test file for {model_type_upper} models (vision architecture).

This test verifies the functionality of the {model_type} model.
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
try:
    from refactored_test_suite.model_test_base import VisionModelTest
except ModuleNotFoundError:
    # Try direct import if not in a package
    from model_test_base import VisionModelTest


class {test_class_name}(VisionModelTest):
    """Test class for {model_type} model."""
    
    def __init__(self, model_id=None, device=None):
        """Initialize the test."""
        # Set model type explicitly
        self.model_type = "{model_type}"
        self.task = "image-classification"
        self.architecture_type = "vision"
        
        # Call parent initializer
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        """Get the default model ID for this model type."""
        return "{default_model_id}"
    
    def run_all_tests(self):
        """Run all tests for this model."""
        # Run basic tests through parent class
        results = self.run_tests()
        
        # Optionally run additional model-specific tests
        # For example, test with a specific image:
        # from PIL import Image
        # model_data = self.load_model()
        # try:
        #     # Try to load a test image if available
        #     test_image = Image.open("test.jpg")
        #     custom_verification = self.verify_model_output(model_data, test_image)
        #     results["custom_image_test"] = custom_verification
        # except Exception as e:
        #     logger.warning(f"Skipping custom image test: {e}")
        
        return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test {model_type} model")
    parser.add_argument("--model-id", type=str, help="Specific model ID to test")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn"], help="Device to test on")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for results")
    parser.add_argument("--image", type=str, help="Path to image file for testing")
    
    args = parser.parse_args()
    
    # Create and run the test
    test = {test_class_name}(args.model_id, args.device)
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