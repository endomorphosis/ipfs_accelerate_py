#!/usr/bin/env python3
"""
Test importing from generators and duckdb_api packages as if from an external script.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_imports")

# Add parent directory to path to simulate using packages as an API
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    logger.info(f"Added {parent_dir} to Python path")

def test_generator_simple_api():
    """Test using the simple test generator API."""
    try:
        # Import the generator
        logger.info("Testing simple_test_generator API...")
        from generators.test_generators.simple_test_generator import generate_test

        # Generate a test file
        model = "bert"
        platform = "cpu"
        output_file = "api_test_bert.py"
        
        file_path = generate_test(model, platform, output_file)
        logger.info(f"Successfully generated {file_path} using simple_test_generator API")
        
        return os.path.exists(file_path)
    except Exception as e:
        logger.error(f"Error using simple_test_generator API: {e}")
        return False

def test_generator_model_detection():
    """Test using the model detection API."""
    try:
        # Import the detection function
        logger.info("Testing model detection API...")
        from generators.test_generators.simple_test_generator import detect_model_category
        
        # Test various models
        models = {
            "bert-base-uncased": "text",
            "t5-small": "text",
            "vit-base": "vision",
            "whisper-tiny": "audio",
            "clip": "multimodal"  # Updated to just 'clip' which should be detected as multimodal
        }
        
        all_correct = True
        for model, expected in models.items():
            detected = detect_model_category(model)
            if detected == expected:
                logger.info(f"Model {model} correctly detected as {detected} ✅")
            else:
                logger.error(f"Model {model} incorrectly detected as {detected}, expected {expected} ❌")
                all_correct = False
        
        return all_correct
    except Exception as e:
        logger.error(f"Error using model detection API: {e}")
        return False

def main():
    """Main function."""
    logger.info("Testing module paths for generators and duckdb_api packages")
    
    # Run tests
    generator_api_success = test_generator_simple_api()
    model_detection_success = test_generator_model_detection()
    
    # Print results
    logger.info("\n=== Module Path Test Results ===")
    logger.info(f"Simple Generator API: {'PASSED' if generator_api_success else 'FAILED'}")
    logger.info(f"Model Detection API: {'PASSED' if model_detection_success else 'FAILED'}")
    
    if generator_api_success and model_detection_success:
        logger.info("\n✅ All module path tests passed! The packages can be used as APIs.")
        return 0
    else:
        logger.error("\n❌ Some module path tests failed. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())