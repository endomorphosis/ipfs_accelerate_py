#!/usr/bin/env python3
"""
Test file for IMAGEBIND models (multimodal architecture).

This test verifies the functionality of the imagebind model for multimodal tasks.
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
from model_test_base import MultimodalModelTest


class TestImagebindModel(MultimodalModelTest):
    """Test class for imagebind model."""
    
    def __init__(self, model_id=None, device=None):
        """Initialize the test."""
        # Set model type explicitly
        self.model_type = "imagebind"
        self.task = "multimodal"
        self.architecture_type = "multimodal"
        
        # Call parent initializer
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        """Get the default model ID for this model type."""
        return "imagebind"
    
    def run_all_tests(self):
        """Run all tests for this model."""
        # Run basic tests through parent class
        results = self.run_tests()
        
        # Optionally run additional model-specific tests
        # For example:
        # model_data = self.load_model()
        # 
        # if self.model_type == "llava":
        #     # Test LLaVA's image understanding capabilities
        #     text_input = "What's in this image? Describe it in detail."
        #     custom_verification = self.verify_model_output(model_data, text_input=text_input)
        #     results["custom_verification"] = custom_verification
        # elif self.model_type == "flava":
        #     # Test FLAVA's multimodal embedding capabilities
        #     text_input = "A photo of a beautiful sunset"
        #     custom_verification = self.verify_model_output(model_data, text_input=text_input)
        #     results["custom_verification"] = custom_verification
        
        return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test imagebind model")
    parser.add_argument("--model-id", type=str, help="Specific model ID to test")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Device to test on")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for results")
    
    args = parser.parse_args()
    
    # Create and run the test
    test = TestImagebindModel(args.model_id, args.device)
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