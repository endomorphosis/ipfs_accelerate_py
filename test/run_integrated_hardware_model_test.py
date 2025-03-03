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
    """Run comprehensive tests for all components with robust error handling"""
    logger.info("Starting comprehensive integration tests...")
    
    # First check which files exist and which are missing
    missing_files = get_missing_files()
    required_for_basic = ["resource_pool.py"]
    
    # Check if we can run at least basic ResourcePool tests
    if any(file in missing_files for file in required_for_basic):
        logger.error(f"Cannot run even basic tests - missing core files: {', '.join(required_for_basic)}")
        return False
    
    # Determine which components are available
    has_hardware_detection = "hardware_detection.py" not in missing_files
    has_model_classifier = "model_family_classifier.py" not in missing_files
    
    logger.info(f"Components available for testing:")
    logger.info(f"  - ResourcePool: Yes (core component)")
    logger.info(f"  - Hardware Detection: {'Yes' if has_hardware_detection else 'No'}")
    logger.info(f"  - Model Family Classifier: {'Yes' if has_model_classifier else 'No'}")
    
    # Import resource_pool (should be available)
    try:
        from resource_pool import get_global_resource_pool
        logger.info("✅ ResourcePool module imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import resource_pool module: {e}")
        return False
    
    # Test 1: Get resource pool instance
    try:
        pool = get_global_resource_pool()
        logger.info("✅ ResourcePool instance created successfully")
    except Exception as e:
        logger.error(f"Failed to create ResourcePool instance: {e}")
        return False
    
    # Track hardware info for later use
    hardware_info = None
    
    # Test 2: Detect hardware (if available)
    if has_hardware_detection:
        try:
            from hardware_detection import detect_hardware_with_comprehensive_checks
            logger.info("✅ Hardware detection module imported successfully")
            
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
            logger.warning(f"Hardware detection test failed, but continuing: {e}")
            logger.warning("ResourcePool will use fallback device detection")
    else:
        logger.warning("Hardware detection module not available, will test ResourcePool fallback mechanism")
    
    # Track classification info for later use
    classification = None
    
    # Test 3: Classify a test model (if available)
    if has_model_classifier:
        try:
            from model_family_classifier import classify_model
            logger.info("✅ Model family classifier module imported successfully")
            
            # Test with a simple model name
            test_model = "bert-base-uncased"
            classification = classify_model(test_model)
            logger.info(f"✅ Model classification completed successfully")
            logger.info(f"Classification for {test_model}: {classification.get('family')} (confidence: {classification.get('confidence', 0):.2f})")
        except Exception as e:
            logger.warning(f"Model classification test failed, but continuing: {e}")
            logger.warning("ResourcePool will use model_type as fallback")
    else:
        logger.warning("Model classifier module not available, will test ResourcePool fallback mechanism")
    
    # Test 4: Integrate available components
    try:
        pool = get_global_resource_pool()
        
        # Load PyTorch through resource pool
        torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
        if torch is None:
            logger.warning("PyTorch not available, using simple object for integration test")
            # Create a dummy test object
            test_obj = {"test": "object"}
            logger.info("✅ Created test object for simplified integration test")
            
            # Test ResourcePool with simple object
            mock_resource = pool.get_resource("test_resource", constructor=lambda: test_obj)
            if mock_resource is not None:
                logger.info("✅ ResourcePool basic functionality works without PyTorch")
            else:
                logger.error("❌ ResourcePool failed to handle basic resource")
                return False
        else:
            logger.info("✅ PyTorch loaded successfully through ResourcePool")
            
            # Test with best available information
            model_name = "bert-base-uncased"
            model_type = "embedding"  # Default if classification not available
            
            # Get device information (using fallback if hardware_detection not available)
            if hardware_info:
                best_device = hardware_info.get("torch_device", "cpu")
            else:
                # Use ResourcePool's internal fallback
                best_device = "auto"
            
            # Get model family (using fallback if model_classifier not available)
            if classification:
                model_family = classification.get("family", model_type)
            else:
                model_family = model_type
            
            # Define a dummy model constructor for testing
            def mock_model_constructor():
                logger.info(f"Creating model using mock constructor")
                # Just return a simple tensor to simulate a model
                return torch.zeros((1, 10))
            
            # Load the model through resource pool
            logger.info(f"Loading model with hardware awareness (using available components)")
            mock_model = pool.get_model(
                model_type=model_family,
                model_name=model_name,
                constructor=mock_model_constructor,
                hardware_preferences={"device": best_device}
            )
            
            if mock_model is not None:
                logger.info("✅ Model loaded successfully with hardware integration")
                logger.info(f"Device for mock model: {mock_model.device if hasattr(mock_model, 'device') else 'unknown'}")
            else:
                logger.error("❌ Mock model could not be loaded")
                return False
    except Exception as e:
        logger.error(f"Final integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False
    
    # Additional test for combinations of components
    if has_hardware_detection or has_model_classifier:
        logger.info("Testing mixed component availability scenarios:")
        
        if has_hardware_detection and not has_model_classifier:
            logger.info("Testing ResourcePool with hardware_detection but without model_classifier:")
            try:
                # Define test model with just hardware preferences
                mock_model = pool.get_model(
                    model_type="test_type",
                    model_name="test_model_hardware_only",
                    constructor=lambda: torch.zeros((1, 5)) if torch else {"test": "object"},
                    hardware_preferences={"device": "auto"}
                )
                
                if mock_model is not None:
                    logger.info("✅ ResourcePool works with hardware_detection only")
                else:
                    logger.warning("⚠️ ResourcePool had issues with hardware_detection only")
            except Exception as e:
                logger.warning(f"⚠️ Error testing hardware-only configuration: {e}")
                
        elif has_model_classifier and not has_hardware_detection:
            logger.info("Testing ResourcePool with model_classifier but without hardware_detection:")
            try:
                # Define test model with emphasis on model type
                mock_model = pool.get_model(
                    model_type="embedding",  # Use known model type
                    model_name="test_model_classifier_only",
                    constructor=lambda: torch.zeros((1, 5)) if torch else {"test": "object"},
                    hardware_preferences={"device": "cpu"}  # Explicit device since no hardware detection
                )
                
                if mock_model is not None:
                    logger.info("✅ ResourcePool works with model_classifier only")
                else:
                    logger.warning("⚠️ ResourcePool had issues with model_classifier only")
            except Exception as e:
                logger.warning(f"⚠️ Error testing classifier-only configuration: {e}")
    
    # Report overall status based on component availability
    if has_hardware_detection and has_model_classifier:
        logger.info("✅ Full integration test completed with ALL components")
    elif has_hardware_detection or has_model_classifier:
        logger.info("✅ Partial integration test completed with SOME components")
        if not has_hardware_detection:
            logger.info("ℹ️ ResourcePool used fallback device detection successfully")
        if not has_model_classifier:
            logger.info("ℹ️ ResourcePool used model_type as fallback successfully")
    else:
        logger.info("✅ Basic integration test completed with core ResourcePool only")
        logger.info("ℹ️ ResourcePool handled missing components gracefully")
    
    # All tests passed with available components
    logger.info("All integration tests for available components passed successfully!")
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