"""
Test script for ONNX verification and conversion system.

This script tests the functionality of the ONNX verification and conversion utility,
ensuring that it correctly verifies ONNX file existence, converts PyTorch models to
ONNX format when needed, and properly manages the conversion registry.
"""

import os
import sys
import logging
import argparse
import tempfile
import shutil
import unittest
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_onnx_verification")

# Import the ONNX verification utility
try:
    from onnx_verification import OnnxVerifier, PyTorchToOnnxConverter, verify_and_get_onnx_model
    from onnx_verification import OnnxVerificationError, OnnxConversionError
except ImportError:
    logger.error("Failed to import onnx_verification module. Make sure it's in your Python path.")
    sys.exit(1)

class TestOnnxVerification(unittest.TestCase):
    """Test case for ONNX verification and conversion utility."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test case."""
        # Create a temporary directory for cache
        cls.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {cls.temp_dir}")
        
        # Create a verifier with the temporary cache directory
        cls.verifier = OnnxVerifier(cache_dir=cls.temp_dir)
        
        # Create a converter with the temporary cache directory
        cls.converter = PyTorchToOnnxConverter(cache_dir=cls.temp_dir)
        
        # Test models - use small models for faster testing
        cls.test_models = [
            {
                "model_id": "prajjwal1/bert-tiny",
                "onnx_path": "model.onnx",
                "model_type": "bert",
                "expected_success": True
            },
            {
                "model_id": "hf-internal-testing/tiny-random-t5",
                "onnx_path": "model.onnx",
                "model_type": "t5",
                "expected_success": True
            },
            {
                "model_id": "openai/whisper-tiny.en",
                "onnx_path": "model.onnx",
                "model_type": "whisper",
                "expected_success": True
            }
        ]
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after the test case."""
        # Remove the temporary directory
        logger.info(f"Removing temporary directory: {cls.temp_dir}")
        shutil.rmtree(cls.temp_dir)
    
    def test_verify_onnx_file(self):
        """Test ONNX file verification."""
        for model_config in self.test_models:
            model_id = model_config["model_id"]
            onnx_path = model_config["onnx_path"]
            
            logger.info(f"Testing ONNX file verification for {model_id}")
            
            # Test verification
            success, result = self.verifier.verify_onnx_file(model_id, onnx_path)
            
            # The specific success value doesn't matter as HuggingFace hosting changes
            # What matters is that the function returns a valid result without errors
            logger.info(f"Verification result for {model_id}: {success}, {result}")
            
            # The verification result should be a string (either a URL or an error message)
            self.assertIsInstance(result, str)
    
    def test_convert_from_pytorch(self):
        """Test PyTorch to ONNX conversion."""
        # Only test the first model to save time
        model_config = self.test_models[0]
        model_id = model_config["model_id"]
        onnx_path = model_config["onnx_path"]
        model_type = model_config["model_type"]
        
        logger.info(f"Testing PyTorch to ONNX conversion for {model_id}")
        
        # Skip if PyTorch or transformers not available
        try:
            import torch
            import transformers
        except ImportError:
            logger.warning("PyTorch or transformers not available. Skipping conversion test.")
            self.skipTest("PyTorch or transformers not available")
            return
        
        try:
            # Create a simple conversion configuration
            conversion_config = {
                "model_type": model_type,
                "opset_version": 12
            }
            
            # Convert the model
            local_path = self.converter.convert_from_pytorch(
                model_id=model_id,
                target_path=onnx_path,
                config=conversion_config
            )
            
            # Check that the file exists
            self.assertTrue(os.path.exists(local_path))
            
            # Check that the file is an ONNX file (just verify it's not empty)
            self.assertGreater(os.path.getsize(local_path), 0)
            
            logger.info(f"Successfully converted {model_id} to ONNX at {local_path}")
            
        except OnnxConversionError as e:
            if not model_config["expected_success"]:
                logger.info(f"Expected conversion failure for {model_id}: {e}")
            else:
                logger.error(f"Unexpected conversion failure for {model_id}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error in PyTorch to ONNX conversion test: {e}")
            raise
    
    def test_verify_and_get_onnx_model(self):
        """Test verify_and_get_onnx_model helper function."""
        # Only test the first model to save time
        model_config = self.test_models[0]
        model_id = model_config["model_id"]
        onnx_path = model_config["onnx_path"]
        model_type = model_config["model_type"]
        
        logger.info(f"Testing verify_and_get_onnx_model for {model_id}")
        
        try:
            # Create a simple conversion configuration
            conversion_config = {
                "model_type": model_type,
                "opset_version": 12
            }
            
            # Get model path
            model_path, was_converted = verify_and_get_onnx_model(
                model_id=model_id,
                onnx_path=onnx_path,
                conversion_config=conversion_config
            )
            
            # If converted, check that the file exists
            if was_converted:
                self.assertTrue(os.path.exists(model_path))
                logger.info(f"Successfully used converted model for {model_id} at {model_path}")
            else:
                logger.info(f"Successfully used original model for {model_id} at {model_path}")
            
        except Exception as e:
            logger.error(f"Error in verify_and_get_onnx_model test: {e}")
            raise
    
    def test_conversion_registry(self):
        """Test conversion registry functionality."""
        # First verify and convert a model
        model_config = self.test_models[0]
        model_id = model_config["model_id"]
        onnx_path = model_config["onnx_path"]
        model_type = model_config["model_type"]
        
        logger.info(f"Testing conversion registry for {model_id}")
        
        try:
            # Create a simple conversion configuration
            conversion_config = {
                "model_type": model_type,
                "opset_version": 12
            }
            
            # Get model path
            model_path, was_converted = verify_and_get_onnx_model(
                model_id=model_id,
                onnx_path=onnx_path,
                conversion_config=conversion_config
            )
            
            # Check that the registry file exists
            registry_path = os.path.join(self.temp_dir, "conversion_registry.json")
            self.assertTrue(os.path.exists(registry_path))
            
            # Load the registry
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            # Check that the model is in the registry
            cache_key = f"{model_id}:{onnx_path}"
            self.assertIn(cache_key, registry)
            
            # Check registry entry contents
            entry = registry[cache_key]
            self.assertEqual(entry["model_id"], model_id)
            self.assertEqual(entry["onnx_path"], onnx_path)
            self.assertTrue(os.path.exists(entry["local_path"]))
            self.assertEqual(entry["source"], "pytorch_conversion")
            
            logger.info(f"Successfully verified conversion registry for {model_id}")
            
        except Exception as e:
            logger.error(f"Error in conversion registry test: {e}")
            raise
    
    def test_model_detection(self):
        """Test model type detection."""
        test_cases = [
            {"model_id": "bert-base-uncased", "expected_type": "bert"},
            {"model_id": "t5-small", "expected_type": "t5"},
            {"model_id": "gpt2", "expected_type": "gpt"},
            {"model_id": "openai/whisper-tiny", "expected_type": "whisper"},
            {"model_id": "google/vit-base-patch16-224", "expected_type": "vit"},
            {"model_id": "openai/clip-vit-base-patch32", "expected_type": "clip"},
            {"model_id": "facebook/wav2vec2-base", "expected_type": "wav2vec2"},
            {"model_id": "unknown-model", "expected_type": "unknown"}
        ]
        
        for case in test_cases:
            model_id = case["model_id"]
            expected_type = case["expected_type"]
            
            detected_type = self.converter._detect_model_type(model_id)
            
            self.assertEqual(detected_type, expected_type, 
                            f"Model detection failed for {model_id}, expected {expected_type}, got {detected_type}")
    
    def test_default_input_shapes(self):
        """Test default input shapes for different model types."""
        test_cases = [
            {"model_type": "bert", "expected_keys": ["batch_size", "sequence_length"]},
            {"model_type": "t5", "expected_keys": ["batch_size", "sequence_length"]},
            {"model_type": "vit", "expected_keys": ["batch_size", "channels", "height", "width"]},
            {"model_type": "clip", "expected_keys": ["vision", "text"]},
            {"model_type": "whisper", "expected_keys": ["batch_size", "feature_size", "sequence_length"]},
            {"model_type": "wav2vec2", "expected_keys": ["batch_size", "sequence_length"]}
        ]
        
        for case in test_cases:
            model_type = case["model_type"]
            expected_keys = case["expected_keys"]
            
            shapes = self.converter._get_default_input_shapes(model_type)
            
            for key in expected_keys:
                self.assertIn(key, shapes, f"Missing key {key} in shapes for model type {model_type}")

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description='Test ONNX verification and conversion system')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--test', type=str, help='Run a specific test (e.g., test_verify_onnx_file)')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    if args.test:
        # Run specific test
        suite = unittest.TestSuite()
        suite.addTest(TestOnnxVerification(args.test))
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        # Run all tests
        unittest.main(argv=[sys.argv[0]])

if __name__ == "__main__":
    main()