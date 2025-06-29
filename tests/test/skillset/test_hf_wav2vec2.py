#!/usr/bin/env python3
"""
Test for the wav2vec2 skillset implementation.

This test verifies the functionality of the wav2vec2 model implementation
in the skillset directory.
"""

import os
import sys
import logging
import argparse
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path to access the skillset module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestSkillsetWav2Vec2:
    """Test class for the wav2vec2 skillset implementation."""
    
    def __init__(self, device=None):
        """Initialize the test.
        
        Args:
            device: The device to run the test on.
        """
        self.device = device or "cpu"
        self.model_type = "wav2vec2"
        # Safe version of model_type with hyphens replaced by underscores
        self.model_type_safe = "wav2vec2"
        self.results = {
            "metadata": {
                "model": self.model_type,
                "device": self.device,
            },
            "success": False,
            "error": None,
            "tests": {}
        }
        
    def import_skillset(self):
        """Import the skillset module.
        
        Returns:
            The imported skillset module or None if import failed.
        """
        try:
            # Try to import the skillset module from the correct directory
            # Get the path to the test directory by starting from the current file
            test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            skillset_path = os.path.join(test_dir, "ipfs_accelerate_py", "worker", "skillset", f"hf_{self.model_type}.py")
            
            if not os.path.exists(skillset_path):
                raise ImportError(f"Skillset file not found: {skillset_path}")
                
            spec = importlib.util.spec_from_file_location(f"hf_{self.model_type}", skillset_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.results["tests"]["import"] = {
                "success": True,
                "message": f"Successfully imported skillset module: {skillset_path}"
            }
            return module
        except Exception as e:
            error_message = f"Error importing skillset module: {e}"
            logger.error(error_message)
            self.results["tests"]["import"] = {
                "success": False,
                "error": error_message
            }
            return None
            
    def check_class_existence(self, module):
        """Check if the skillset class exists in the module.
        
        Args:
            module: The imported skillset module.
            
        Returns:
            The skillset class or None if not found.
        """
        try:
            # Expected class name format is hf_modelname with hyphens converted to underscores
            class_name = f"hf_{self.model_type_safe}"
            
            if not hasattr(module, class_name):
                raise AttributeError(f"Class '{class_name}' not found in module")
                
            skillset_class = getattr(module, class_name)
            
            self.results["tests"]["class_existence"] = {
                "success": True,
                "message": f"Class '{class_name}' found in module"
            }
            return skillset_class
        except Exception as e:
            error_message = f"Error finding skillset class: {e}"
            logger.error(error_message)
            self.results["tests"]["class_existence"] = {
                "success": False,
                "error": error_message
            }
            return None
            
    def instantiate_class(self, skillset_class):
        """Instantiate the skillset class.
        
        Args:
            skillset_class: The skillset class to instantiate.
            
        Returns:
            An instance of the skillset class or None if instantiation failed.
        """
        try:
            # Try to instantiate the class
            instance = skillset_class(resources={"device": self.device})
            
            self.results["tests"]["instantiation"] = {
                "success": True,
                "message": "Successfully instantiated skillset class"
            }
            return instance
        except Exception as e:
            error_message = f"Error instantiating skillset class: {e}"
            logger.error(error_message)
            self.results["tests"]["instantiation"] = {
                "success": False,
                "error": error_message
            }
            return None
            
    def check_methods(self, instance):
        """Check if the skillset instance has required methods.
        
        Args:
            instance: An instance of the skillset class.
            
        Returns:
            True if all required methods exist, False otherwise.
        """
        try:
            # Define required methods based on the actual skillset implementation
            required_methods = [
                "init",
                "init_cpu",
                "init_cuda",
                "init_rocm",
                "init_openvino",
                "init_apple",
                "init_qualcomm",
                "__test__"
            ]
            
            # Check each method
            missing_methods = []
            for method in required_methods:
                if not hasattr(instance, method):
                    missing_methods.append(method)
                    
            if missing_methods:
                raise AttributeError(f"Missing required methods: {', '.join(missing_methods)}")
                
            self.results["tests"]["methods"] = {
                "success": True,
                "message": "All required methods exist"
            }
            return True
        except Exception as e:
            error_message = f"Error checking required methods: {e}"
            logger.error(error_message)
            self.results["tests"]["methods"] = {
                "success": False,
                "error": error_message,
                "missing_methods": missing_methods if 'missing_methods' in locals() else []
            }
            return False
    
    def run_tests(self):
        """Run all tests for this skillset.
        
        Returns:
            A dictionary with test results.
        """
        # Import skillset module
        module = self.import_skillset()
        if not module:
            self.results["success"] = False
            self.results["error"] = "Failed to import skillset module"
            return self.results
            
        # Check class existence
        skillset_class = self.check_class_existence(module)
        if not skillset_class:
            self.results["success"] = False
            self.results["error"] = "Skillset class not found in module"
            return self.results
            
        # Instantiate class
        instance = self.instantiate_class(skillset_class)
        if not instance:
            self.results["success"] = False
            self.results["error"] = "Failed to instantiate skillset class"
            return self.results
            
        # Check methods
        if not self.check_methods(instance):
            self.results["success"] = False
            self.results["error"] = "Missing required methods"
            return self.results
            
        # All tests passed
        self.results["success"] = True
        return self.results
        
    def save_results(self, output_dir="test_results"):
        """Save test results to a file.
        
        Args:
            output_dir: Directory to save results in.
            
        Returns:
            The path to the saved results file or None if saving failed.
        """
        try:
            import json
            import time
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create the output file path
            timestamp = int(time.time())
            output_file = os.path.join(output_dir, f"skillset_test_{self.model_type}_{timestamp}.json")
            
            # Write results to file
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            return output_file
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test wav2vec2 skillset implementation")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn"], default="cpu",
                       help="Device to test on")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="skillset_test_results", help="Directory for results")
    
    args = parser.parse_args()
    
    # Create and run the test
    test = TestSkillsetWav2Vec2(args.device)
    results = test.run_tests()
    
    # Print a summary
    if results["success"]:
        print(f"✅ Successfully tested {test.model_type} skillset implementation on {args.device}")
    else:
        print(f"❌ Failed to test {test.model_type} skillset implementation on {args.device}")
        print(f"Error: {results['error']}")
    
    # Save results if requested
    if args.save:
        output_path = test.save_results(args.output_dir)
        if output_path:
            print(f"Results saved to {output_path}")
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())