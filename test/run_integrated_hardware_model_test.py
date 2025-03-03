#!/usr/bin/env python
"""
Script to test the integration of ResourcePool, hardware_detection, and model_family_classifier.
This script ensures that all components work together correctly.
"""

import os
import sys
import time
import logging
import argparse
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file_exists(file_path):
    """Check if a file exists and log the result"""
    exists = os.path.exists(file_path)
    if exists:
        logger.info(f"✅ File exists: {file_path}")
    else:
        logger.warning(f"⚠️ File not found: {file_path}")
    return exists

def get_missing_files():
    """Check for all required files and return a list of missing ones"""
    required_files = [
        "resource_pool.py",
        "hardware_detection.py",
        "model_family_classifier.py",
        "test_resource_pool.py",
        "test_comprehensive_hardware.py",
        "test_generator_with_resource_pool.py",
        "RESOURCE_POOL_GUIDE.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if not check_file_exists(full_path):
            missing_files.append(file_path)
    
    return missing_files

def run_comprehensive_tests():
    """Run comprehensive tests for all components"""
    logger.info("Starting comprehensive integration tests...")
    
    # First check if all files exist
    missing_files = get_missing_files()
    if missing_files:
        logger.error(f"Cannot run tests - missing files: {', '.join(missing_files)}")
        return False
    
    # Import required modules
    try:
        from resource_pool import get_global_resource_pool
        from hardware_detection import detect_hardware_with_comprehensive_checks
        from model_family_classifier import classify_model
        
        logger.info("✅ All required modules imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        return False
    
    # Test 1: Get resource pool instance
    try:
        pool = get_global_resource_pool()
        logger.info("✅ ResourcePool instance created successfully")
    except Exception as e:
        logger.error(f"Failed to create ResourcePool instance: {e}")
        return False
    
    # Test 2: Detect hardware
    try:
        hardware_info = detect_hardware_with_comprehensive_checks()
        logger.info(f"✅ Hardware detection completed successfully")
        
        # Show best available hardware
        best_device = None
        if hardware_info.get("cuda", False):
            best_device = "cuda"
        elif hardware_info.get("mps", False):
            best_device = "mps"
        elif hardware_info.get("rocm", False):
            best_device = "rocm"
        elif hardware_info.get("openvino", False):
            best_device = "openvino"
        else:
            best_device = "cpu"
            
        logger.info(f"Best available hardware: {best_device}")
        torch_device = hardware_info.get("torch_device", best_device)
        logger.info(f"PyTorch device: {torch_device}")
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        return False
    
    # Test 3: Classify a test model
    try:
        # Test with a simple model name
        test_model = "bert-base-uncased"
        classification = classify_model(test_model)
        logger.info(f"✅ Model classification completed successfully")
        logger.info(f"Classification for {test_model}: {classification.get('family')} (confidence: {classification.get('confidence', 0):.2f})")
    except Exception as e:
        logger.error(f"Model classification failed: {e}")
        return False
    
    # Test 4: Integrate all components
    try:
        pool = get_global_resource_pool()
        
        # Load PyTorch through resource pool
        torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
        if torch is None:
            logger.warning("PyTorch not available, skipping final integration test")
        else:
            logger.info("✅ PyTorch loaded successfully through ResourcePool")
            
            # Get hardware info
            hardware_info = detect_hardware_with_comprehensive_checks()
            
            # Create hardware preferences for a model
            model_name = "bert-base-uncased"
            classification = classify_model(model_name)
            
            # Determine device
            best_device = hardware_info.get("torch_device", "cpu")
            
            # Define a dummy model constructor for testing
            def mock_model_constructor():
                logger.info(f"Creating model using mock constructor (device: {best_device})")
                # Just return a simple tensor to simulate a model
                return torch.zeros((1, 10))
            
            # Load the model through resource pool
            logger.info(f"Loading model with hardware awareness")
            mock_model = pool.get_model(
                model_type=classification.get("family", "default"),
                model_name=model_name,
                constructor=mock_model_constructor,
                hardware_preferences={"device": best_device}
            )
            
            if mock_model is not None:
                logger.info("✅ Model loaded successfully with hardware integration")
            else:
                logger.error("Mock model could not be loaded")
                return False
    except Exception as e:
        logger.error(f"Final integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False
    
    # All tests passed
    logger.info("All integration tests passed successfully!")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test integration of ResourcePool, hardware_detection, and model_family_classifier")
    parser.add_argument("--check-only", action="store_true", help="Only check if files exist")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    try:
        # Print information about the test
        print("\n=== ResourcePool Integration Test ===")
        print("This test verifies that all components work together correctly:")
        print("- ResourcePool (resource management)")
        print("- hardware_detection (hardware-aware resource allocation)")
        print("- model_family_classifier (model type detection)")
        print("- test_generator_with_resource_pool (test file generation)\n")
        
        # Check if all required files exist
        missing_files = get_missing_files()
        
        if missing_files:
            logger.error(f"⚠️ Missing files: {', '.join(missing_files)}")
            print("You need to create the missing files before running this test.")
            return 1
        
        if args.check_only:
            logger.info("Only checking files - all required files exist.")
            return 0
        
        # Run the integration tests
        success = run_comprehensive_tests()
        
        if success:
            print("\n✅ Integration test completed successfully!")
            return 0
        else:
            print("\n❌ Integration test failed. See errors above.")
            return 1
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())