#!/usr/bin/env python3
"""
Test file for Stable Diffusion models (diffusion architecture).

This test verifies the functionality of Stable Diffusion and other diffusion models.
Example model: latent_diffusion (latent-diffusion)
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
    from refactored_test_suite.model_test_base import DiffusionModelTest
except ModuleNotFoundError:
    # Try direct import if not in a package
    from model_test_base import DiffusionModelTest


class TestStableDiffusionModel(DiffusionModelTest):
    """Test class for Stable Diffusion and other diffusion models."""
    
    def __init__(self, model_id=None, device=None):
        """Initialize the test."""
        # Set model type explicitly
        self.model_type = "diffusion"
        self.task = "text-to-image"
        self.architecture_type = "diffusion"
        
        # Call parent initializer
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        """Get the default model ID for this model type."""
        return "latent-diffusion"
    
    def run_all_tests(self):
        """Run all tests for diffusion models."""
        # Run basic tests through parent class
        results = self.run_tests()
        
        # Optionally run additional diffusion-specific tests
        # For example:
        # model_data = self.load_model()
        # custom_input = "A photo of a cat"
        # custom_verification = self.verify_model_output(model_data, custom_input)
        # results["custom_verification"] = custom_verification
        
        return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test diffusion models")
    parser.add_argument("--model-id", type=str, help="Specific model ID to test")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn"], help="Device to test on")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for results")
    
    args = parser.parse_args()
    
    # Create and run the test
    test = TestStableDiffusionModel(args.model_id, args.device)
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