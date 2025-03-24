#!/usr/bin/env python3
"""
Test file for ENCODEC models (speech architecture).

This test verifies the functionality of the encodec model for speech recognition or audio tasks.
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
from model_test_base import SpeechModelTest


class TestEncodecModel(SpeechModelTest):
    """Test class for encodec model."""
    
    def __init__(self, model_id=None, device=None):
        """Initialize the test."""
        # Set model type explicitly
        self.model_type = "encodec"
        self.task = "automatic-speech-recognition"
        self.architecture_type = "speech"
        
        # Set model-specific parameters
        self.sampling_rate = 24000  # Default sampling rate for this model
        
        # Call parent initializer
        super().__init__(model_id, device)
    
    def get_default_model_id(self) -> str:
        """Get the default model ID for this model type."""
        return "encodec"
    
    def run_all_tests(self):
        """Run all tests for this model."""
        # Run basic tests through parent class
        results = self.run_tests()
        
        # Optionally run additional model-specific tests
        # For example:
        # model_data = self.load_model()
        # custom_audio = self.create_dummy_audio(duration=10)
        # custom_verification = self.verify_model_output(model_data, custom_audio)
        # results["custom_verification"] = custom_verification
        
        return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test encodec model")
    parser.add_argument("--model-id", type=str, help="Specific model ID to test")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Device to test on")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for results")
    
    args = parser.parse_args()
    
    # Create and run the test
    test = TestEncodecModel(args.model_id, args.device)
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