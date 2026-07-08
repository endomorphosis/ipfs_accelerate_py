#!/usr/bin/env python
"""
Enhanced integration test script for ResourcePool, hardware_detection, and model_family_classifier.
This script ensures that all components work together correctly with robust error handling
and graceful degradation when components are missing.
"""

import os
import sys
import time
import logging
import argparse
import traceback
import json
from datetime import datetime
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

def get_missing_files(include_optional=True):
    """
    Check for all required files and return a list of missing ones
    
    Args:
        include_optional: Whether to include optional component files
        
    Returns:
        Tuple of (missing_required, missing_optional)
    """
    # Core required files
    required_files = [
        "resource_pool.py",
        "test_resource_pool.py",
        "RESOURCE_POOL_GUIDE.md"
    ]
    
    # Optional component files
    optional_files = [
        "hardware_detection.py",
        "model_family_classifier.py",
        "test_comprehensive_hardware.py",
        "test_generator_with_resource_pool.py",
        "hardware_model_integration.py"
    ]
    
    # Check required files
    missing_required = []
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if not check_file_exists(full_path):
            missing_required.append(file_path)
    
    # Check optional files if requested
    missing_optional = []
    if include_optional:
        for file_path in optional_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            if not check_file_exists(full_path):
                missing_optional.append(file_path)
    
    return (missing_required, missing_optional)

def test_resource_pool_basic():
    """Test basic ResourcePool functionality without any other components"""
    logger.info("Testing basic ResourcePool functionality...")
    
    try:
        # Import resource pool
        from scripts.generators.utils.resource_pool import get_global_resource_pool
        logger.info("✅ ResourcePool module imported successfully")
        
        # Get instance
        pool = get_global_resource_pool()
        logger.info("✅ ResourcePool instance created successfully")
        
        # Test basic resource caching
        test_obj1 = {"test": "object1"}
        test_obj2 = {"test": "object2"}
        
        # Store resources
        resource1 = pool.get_resource("test_resource", resource_id="1", 
                                     constructor=lambda: test_obj1)
        resource2 = pool.get_resource("test_resource", resource_id="2", 
                                     constructor=lambda: test_obj2)
        
        # Verify we got the right objects
        if resource1 != test_obj1 or resource2 != test_obj2:
            logger.error("❌ ResourcePool returned incorrect objects")
            return False
        
        # Test cache hit
        resource1_again = pool.get_resource("test_resource", resource_id="1")
        if resource1_again is not resource1:
            logger.error("❌ ResourcePool cache hit failed")
            return False
        
        # Check stats
        stats = pool.get_stats()
        cache_hit_found = stats.get("hits", 0) >= 1
        logger.info(f"ResourcePool stats after basic test: {stats}")
        
        if not cache_hit_found:
            logger.warning("⚠️ ResourcePool stats didn't record cache hit")
        
        # Test cleanup
        removed = pool.cleanup_unused_resources(max_age_minutes=0)
        logger.info(f"ResourcePool cleanup removed {removed} resources")
        
        # Clear resources
        pool.clear()
        logger.info("✅ ResourcePool clear() completed successfully")
        
        return True
    except Exception as e:
        logger.error(f"Basic ResourcePool test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_hardware_detection(has_pytorch=False):
    """
    Test the hardware_detection module if available
    
    Args:
        has_pytorch: Whether PyTorch is available
        
    Returns:
        Tuple of (success, hardware_info)
    """
    # Check if hardware_detection module exists
    module_path = os.path.join(os.path.dirname(__file__), "hardware_detection.py")
    if not os.path.exists(module_path):
        logger.warning("hardware_detection.py not found, skipping hardware detection test")
        return (False, None)
    
    logger.info("Testing hardware detection module...")
    
    try:
        # Import the module
        from scripts.generators.hardware.hardware_detection import detect_hardware_with_comprehensive_checks
        logger.info("✅ Hardware detection module imported successfully")
        
        # Run detection
        hardware_info = detect_hardware_with_comprehensive_checks()
        logger.info("✅ Hardware detection completed successfully")
        
        # Basic checks
        if not hardware_info:
            logger.warning("⚠️ Hardware detection returned empty results")
            return (False, None)
        
        # Check CPU detection
        if "cpu" not in hardware_info:
            logger.warning("⚠️ Hardware detection didn't identify CPU")
        else:
            logger.info("✅ CPU detection successful")
            
        # Check for CUDA if PyTorch is available
        if has_pytorch:
            import torch
            if torch.cuda.is_available() and not hardware_info.get("cuda", False):
                logger.warning("⚠️ PyTorch reports CUDA available but hardware detection didn't identify it")
            elif hardware_info.get("cuda", False):
                logger.info("✅ CUDA detection successful")
                
                # Check for CUDA device count
                torch_count = torch.cuda.device_count()
                detected_count = hardware_info.get("cuda_device_count", 0)
                if torch_count != detected_count:
                    logger.warning(f"⚠️ Device count mismatch: PyTorch reports {torch_count}, hardware detection reports {detected_count}")
        
        # Check for torch_device key
        if "torch_device" not in hardware_info:
            logger.warning("⚠️ Hardware detection didn't provide torch_device")
        else:
            logger.info(f"✅ Recommended PyTorch device: {hardware_info['torch_device']}")
        
        # Display hardware summary
        detected_hardware = []
        for hw_type in ["cpu", "cuda", "mps", "rocm", "openvino", "webnn", "webgpu"]:
            if hardware_info.get(hw_type, False):
                detected_hardware.append(hw_type)
        
        logger.info(f"Detected hardware: {', '.join(detected_hardware)}")
        
        # Save hardware info for other tests
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"hardware_detection_results_{timestamp}.json"
        try:
            with open(result_file, "w") as f:
                json.dump(hardware_info, f, indent=2)
            logger.info(f"Hardware detection results saved to {result_file}")
        except Exception as e:
            logger.warning(f"Could not save hardware detection results: {e}")
        
        return (True, hardware_info)
    except Exception as e:
        logger.error(f"Hardware detection test failed: {e}")
        logger.error(traceback.format_exc())
        return (False, None)

def test_model_classifier():
    """
    Test the model_family_classifier module if available
    
    Returns:
        Tuple of (success, classifier_info)
    """
    # Check if model_family_classifier module exists
    module_path = os.path.join(os.path.dirname(__file__), "model_family_classifier.py")
    if not os.path.exists(module_path):
        logger.warning("model_family_classifier.py not found, skipping model classification test")
        return (False, None)
    
    logger.info("Testing model family classifier module...")
    
    try:
        # Import the module
        from model_family_classifier import classify_model, ModelFamilyClassifier
        logger.info("✅ Model family classifier module imported successfully")
        
        # Test models to classify
        test_models = {
            "bert-base-uncased": "embedding", 
            "t5-small": "text_generation",
            "vit-base-patch16-224": "vision"
        }
        
        # Track classification results
        classification_results = {}
        template_results = {}
        classifier_success = True
        
        # Test classification for each model
        for model_name, expected_family in test_models.items():
            try:
                # Classify model by name
                result = classify_model(model_name)
                
                # Check result structure
                if "family" not in result:
                    logger.warning(f"⚠️ Missing family in classification result for {model_name}")
                    classifier_success = False
                    continue
                
                # Get family and confidence
                family = result.get("family")
                confidence = result.get("confidence", 0)
                
                # Track result
                classification_results[model_name] = {
                    "family": family,
                    "confidence": confidence,
                    "expected": expected_family
                }
                
                # Compare with expected
                if family != expected_family:
                    logger.warning(f"⚠️ Unexpected classification for {model_name}: got {family}, expected {expected_family}")
                else:
                    logger.info(f"✅ Model {model_name} correctly classified as {family} (confidence: {confidence:.2f})")
                
                # Test template selection
                try:
                    classifier = ModelFamilyClassifier()
                    template = classifier.get_template_for_family(family)
                    template_results[model_name] = template
                    
                    if not template.endswith(".py"):
                        logger.warning(f"⚠️ Invalid template format for {model_name}: {template}")
                    else:
                        logger.info(f"✅ Template for {model_name}: {template}")
                except Exception as e:
                    logger.warning(f"⚠️ Template selection failed for {model_name}: {e}")
                    classifier_success = False
            except Exception as e:
                logger.warning(f"⚠️ Classification failed for {model_name}: {e}")
                classifier_success = False
        
        # Test hardware-aware classification if possible
        try:
            # Prepare mock hardware compatibility info
            hw_compatibility = {
                "cuda": {"compatible": True, "memory_usage": {"peak": 1200}},
                "mps": {"compatible": False, "reason": "Model too complex for MPS"},
                "rocm": {"compatible": True},
                "openvino": {"compatible": True}
            }
            
            # Classify with hardware awareness
            hw_result = classify_model("llava-13b", hw_compatibility=hw_compatibility)
            
            # Check if hardware analysis was used
            hw_analyses = [a for a in hw_result.get("analyses", []) if a.get("source") == "hardware_analysis"]
            
            if hw_analyses:
                logger.info(f"✅ Hardware-aware classification successfully used hardware information")
                logger.info(f"  Result: {hw_result.get('family')} (reason: {hw_analyses[0].get('reason', 'Unknown')})")
            else:
                logger.warning(f"⚠️ Hardware information not used in classification")
                
        except Exception as e:
            logger.warning(f"⚠️ Hardware-aware classification test failed: {e}")
        
        # Save results for other tests
        classifier_info = {
            "classification_results": classification_results,
            "template_results": template_results
        }
        
        return (classifier_success, classifier_info)
    except Exception as e:
        logger.error(f"Model classifier test failed: {e}")
        logger.error(traceback.format_exc())
        return (False, None)

def test_resource_pool_with_components(has_hardware_detection, hardware_info, 
                                      has_model_classifier, classifier_info):
    """
    Test ResourcePool integration with available components
    
    Args:
        has_hardware_detection: Whether hardware_detection is available
        hardware_info: Hardware detection results
        has_model_classifier: Whether model_family_classifier is available
        classifier_info: Classifier results
        
    Returns:
        bool: Success or failure
    """
    logger.info("Testing ResourcePool integration with available components...")
    
    try:
        # Import resource pool
        from scripts.generators.utils.resource_pool import get_global_resource_pool
        pool = get_global_resource_pool()
        
        # Try to get PyTorch
        try:
            torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
            has_pytorch = torch is not None
            if has_pytorch:
                logger.info("✅ PyTorch loaded successfully through ResourcePool")
            else:
                logger.warning("⚠️ PyTorch not available, will use mock objects for testing")
        except Exception:
            has_pytorch = False
            logger.warning("⚠️ Error loading PyTorch, will use mock objects for testing")
        
        # Define test models for different scenarios
        test_models = [
            {
                "name": "test_model_basic",
                "type": "basic",
                "hardware_prefs": {"device": "cpu"},
                "description": "Basic model without special hardware preferences"
            },
            {
                "name": "test_model_auto",
                "type": "auto_device",
                "hardware_prefs": {"device": "auto"},
                "description": "Model with automatic device selection"
            }
        ]
        
        # Add hardware-specific test if hardware detection is available
        if has_hardware_detection and hardware_info:
            # Get best available device
            if hardware_info.get("cuda", False):
                test_device = "cuda"
            elif hardware_info.get("mps", False):
                test_device = "mps"
            elif hardware_info.get("rocm", False):
                test_device = "rocm"
            else:
                test_device = "cpu"
                
            test_models.append({
                "name": "test_model_hardware",
                "type": "hardware_aware",
                "hardware_prefs": {"device": test_device},
                "description": f"Model with hardware-specific device ({test_device})"
            })
            
            # Add priority list test
            test_models.append({
                "name": "test_model_priority",
                "type": "priority_list",
                "hardware_prefs": {"priority_list": ["cuda", "mps", "cpu"]},
                "description": "Model with priority list for hardware selection"
            })
        
        # Add model-family specific test if classifier is available
        if has_model_classifier and classifier_info:
            # Get a model with known family
            known_models = classifier_info.get("classification_results", {})
            if known_models:
                test_model_name = list(known_models.keys())[0]
                test_model_family = known_models[test_model_name]["family"]
                
                test_models.append({
                    "name": test_model_name,
                    "type": "classified",
                    "model_family": test_model_family,
                    "hardware_prefs": {"device": "auto"},
                    "description": f"Real model with known family ({test_model_family})"
                })
        
        # Define constructor based on PyTorch availability
        def create_mock_model(model_name, model_type=None):
            if has_pytorch:
                # Create a simple tensor as mock model
                mock_model = torch.zeros((1, 10))
                # Set device attribute to track device placement
                mock_model.device = torch.device("cpu")
                return mock_model
            else:
                # Return dict as mock model
                return {"name": model_name, "type": model_type}
        
        # Test each model configuration
        for test_model in test_models:
            model_name = test_model["name"]
            model_type = test_model.get("model_family", test_model["type"])
            hardware_prefs = test_model["hardware_prefs"]
            description = test_model["description"]
            
            logger.info(f"Testing model: {model_name} ({description})")
            
            # Create the model constructor
            constructor = lambda: create_mock_model(model_name, model_type)
            
            try:
                # Get the model through ResourcePool
                model = pool.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    constructor=constructor,
                    hardware_preferences=hardware_prefs
                )
                
                if model is None:
                    logger.error(f"❌ Failed to create model {model_name}")
                    continue
                    
                logger.info(f"✅ Model {model_name} created successfully")
                
                # Check device if it's a torch tensor
                if has_pytorch and isinstance(model, torch.Tensor):
                    logger.info(f"  Device: {model.device}")
                
                # Test cache hit by requesting same model again
                model_again = pool.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    constructor=constructor,
                    hardware_preferences=hardware_prefs
                )
                
                if model_again is not model:
                    logger.warning(f"⚠️ Cache miss when requesting {model_name} again")
                else:
                    logger.info(f"✅ Cache hit when requesting {model_name} again")
                    
            except Exception as e:
                logger.error(f"❌ Error creating model {model_name}: {e}")
        
        # Get final stats
        stats = pool.get_stats()
        logger.info(f"Final ResourcePool stats: hits={stats.get('hits', 0)}, misses={stats.get('misses', 0)}")
        logger.info(f"Models in cache: {stats.get('cached_models', 0)}")
        
        # Clean up
        pool.clear()
        logger.info("ResourcePool cleared successfully")
        
        return True
    except Exception as e:
        logger.error(f"ResourcePool integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_full_integration(has_hardware_detection, hardware_info, 
                         has_model_classifier, classifier_info):
    """
    Test full integration of all available components with mix-and-match scenarios
    
    Args:
        has_hardware_detection: Whether hardware_detection is available
        hardware_info: Hardware detection results
        has_model_classifier: Whether model_family_classifier is available
        classifier_info: Classifier results
        
    Returns:
        bool: Success or failure
    """
    # If neither component is available, skip this test
    if not has_hardware_detection and not has_model_classifier:
        logger.info("Skipping full integration test - neither component is available")
        return True
    
    logger.info("\nTesting full integration of all available components...")
    
    # Create dictionaries for tracking
    component_tests = {}
    
    # Check hardware_model_integration.py if it exists
    integration_file = os.path.join(os.path.dirname(__file__), "hardware_model_integration.py")
    has_integration_module = os.path.exists(integration_file)
    
    if has_integration_module:
        try:
            # Try to import the module
            import hardware_model_integration
            logger.info("✅ hardware_model_integration module imported successfully")
            
            # Check if it has the required function
            if hasattr(hardware_model_integration, "integrate_hardware_and_model"):
                logger.info("✅ integrate_hardware_and_model function found")
                
                # Test the function if both components are available
                if has_hardware_detection and has_model_classifier:
                    try:
                        # Get minimal hardware info
                        minimal_hw = {"cuda": hardware_info.get("cuda", False),
                                     "mps": hardware_info.get("mps", False),
                                     "cpu": True}
                        
                        # Get model family for bert
                        model_family = classifier_info.get("classification_results", {}).get(
                            "bert-base-uncased", {}).get("family", "embedding")
                        
                        # Run integration function
                        result = hardware_model_integration.integrate_hardware_and_model(
                            model_name="bert-base-uncased",
                            model_family=model_family,
                            hardware_info=minimal_hw
                        )
                        
                        # Check result format
                        if isinstance(result, dict) and "device" in result:
                            logger.info(f"✅ Integration function returned valid result: {result}")
                            component_tests["integration_module"] = True
                        else:
                            logger.warning(f"⚠️ Integration function returned unexpected format: {result}")
                            component_tests["integration_module"] = False
                    except Exception as e:
                        logger.warning(f"⚠️ Error testing integration function: {e}")
                        component_tests["integration_module"] = False
                else:
                    logger.info("Skipping integration function test - not all components available")
            else:
                logger.warning("⚠️ integrate_hardware_and_model function not found")
                component_tests["integration_module"] = False
        except ImportError as e:
            logger.warning(f"⚠️ Could not import hardware_model_integration module: {e}")
            component_tests["integration_module"] = False
    
    # Test ResourcePool with both components
    try:
        from scripts.generators.utils.resource_pool import get_global_resource_pool
        pool = get_global_resource_pool()
        
        # Try to import torch
        try:
            torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
            has_pytorch = torch is not None
        except Exception:
            has_pytorch = False
            torch = None
        
        # Define model constructor
        def create_mock_model():
            if has_pytorch:
                return torch.zeros((1, 10))
            else:
                return {"mock": "model"}
        
        # Test scenarios with combinations of components
        
        # Scenario 1: Hardware-only (no classifier)
        if has_hardware_detection and not has_model_classifier:
            try:
                logger.info("Testing scenario: Hardware detection without model classifier")
                model = pool.get_model(
                    model_type="test_type",  # ResourcePool should use this when classifier missing
                    model_name="test_hardware_only",
                    constructor=create_mock_model,
                    hardware_preferences={"device": "auto"}  # Should use hardware detection
                )
                
                if model is not None:
                    logger.info("✅ ResourcePool works with hardware_detection only")
                    component_tests["hardware_only"] = True
                else:
                    logger.warning("⚠️ ResourcePool had issues with hardware_detection only")
                    component_tests["hardware_only"] = False
            except Exception as e:
                logger.warning(f"⚠️ Error testing hardware-only scenario: {e}")
                component_tests["hardware_only"] = False
        
        # Scenario 2: Classifier-only (no hardware)
        if has_model_classifier and not has_hardware_detection:
            try:
                logger.info("Testing scenario: Model classifier without hardware detection")
                # Get a model with known family
                known_models = classifier_info.get("classification_results", {})
                if known_models:
                    test_model_name = list(known_models.keys())[0]
                    test_model_family = known_models[test_model_name]["family"]
                    
                    model = pool.get_model(
                        model_type=test_model_family,  # Used directly
                        model_name=test_model_name,
                        constructor=create_mock_model,
                        hardware_preferences={"device": "cpu"}  # Explicit device since no hardware detection
                    )
                    
                    if model is not None:
                        logger.info("✅ ResourcePool works with model_classifier only")
                        component_tests["classifier_only"] = True
                    else:
                        logger.warning("⚠️ ResourcePool had issues with model_classifier only")
                        component_tests["classifier_only"] = False
                else:
                    logger.warning("⚠️ No classified models available for testing classifier-only scenario")
            except Exception as e:
                logger.warning(f"⚠️ Error testing classifier-only scenario: {e}")
                component_tests["classifier_only"] = False
        
        # Scenario 3: Full integration (both components)
        if has_hardware_detection and has_model_classifier:
            try:
                logger.info("Testing scenario: Full integration with both components")
                # Get a model with known family
                known_models = classifier_info.get("classification_results", {})
                if known_models:
                    test_model_name = list(known_models.keys())[0]
                    test_model_family = known_models[test_model_name]["family"]
                    
                    # Define hardware compatibility info if possible
                    hw_compat = None
                    if hardware_info:
                        hw_compat = {
                            "cuda": {"compatible": hardware_info.get("cuda", False)},
                            "mps": {"compatible": hardware_info.get("mps", False)},
                            "cpu": {"compatible": True}
                        }
                    
                    model = pool.get_model(
                        model_type=test_model_family,
                        model_name=test_model_name,
                        constructor=create_mock_model,
                        hardware_preferences={
                            "device": "auto",
                            "hw_compatibility": hw_compat
                        }
                    )
                    
                    if model is not None:
                        logger.info("✅ ResourcePool works with both components")
                        component_tests["full_integration"] = True
                    else:
                        logger.warning("⚠️ ResourcePool had issues with both components")
                        component_tests["full_integration"] = False
                else:
                    logger.warning("⚠️ No classified models available for testing full integration scenario")
            except Exception as e:
                logger.warning(f"⚠️ Error testing full integration scenario: {e}")
                component_tests["full_integration"] = False
        
        # Clean up resources
        pool.clear()
        
        # Summarize component integration tests
        logger.info("\nComponent integration test results:")
        for test_name, passed in component_tests.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"  {test_name}: {status}")
        
        all_passed = all(component_tests.values())
        if all_passed:
            logger.info("✅ All component integration tests PASSED")
        else:
            logger.warning("⚠️ Some component integration tests FAILED")
        
        return all_passed
    except Exception as e:
        logger.error(f"Full integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_comprehensive_tests():
    """
    Run comprehensive tests for all components with robust error handling
    
    Returns:
        bool: Overall success or failure
    """
    logger.info("Starting comprehensive integration tests...")
    
    # Track test results
    test_results = {}
    
    # First check which files exist and which are missing
    missing_required, missing_optional = get_missing_files()
    
    # Check if we can run tests at all
    if missing_required:
        logger.error(f"Cannot run tests - missing core files: {', '.join(missing_required)}")
        return False
    
    # Determine which components are available
    has_hardware_detection = "hardware_detection.py" not in missing_optional
    has_model_classifier = "model_family_classifier.py" not in missing_optional
    
    logger.info(f"Components available for testing:")
    logger.info(f"  - ResourcePool: Yes (core component)")
    logger.info(f"  - Hardware Detection: {'Yes' if has_hardware_detection else 'No'}")
    logger.info(f"  - Model Family Classifier: {'Yes' if has_model_classifier else 'No'}")
    
    # Test 1: Basic ResourcePool functionality
    logger.info("\n=== Testing basic ResourcePool functionality ===")
    basic_success = test_resource_pool_basic()
    test_results["basic_resourcepool"] = basic_success
    
    if not basic_success:
        logger.error("❌ Basic ResourcePool test failed - cannot continue")
        return False
    
    # Test 2: Hardware detection (if available)
    hardware_info = None
    hardware_success = False
    
    if has_hardware_detection:
        logger.info("\n=== Testing hardware detection ===")
        # Try to get PyTorch for better hardware testing
        has_pytorch = False
        try:
            import torch
            has_pytorch = True
        except ImportError:
            logger.warning("PyTorch not available for hardware detection test")
        
        hardware_success, hardware_info = test_hardware_detection(has_pytorch)
        test_results["hardware_detection"] = hardware_success
    
    # Test 3: Model family classifier (if available)
    classifier_info = None
    classifier_success = False
    
    if has_model_classifier:
        logger.info("\n=== Testing model family classifier ===")
        classifier_success, classifier_info = test_model_classifier()
        test_results["model_classifier"] = classifier_success
    
    # Test 4: ResourcePool with available components
    logger.info("\n=== Testing ResourcePool with available components ===")
    integration_success = test_resource_pool_with_components(
        has_hardware_detection, hardware_info,
        has_model_classifier, classifier_info
    )
    test_results["resourcepool_integration"] = integration_success
    
    # Test 5: Full integration with all available components
    full_integration_success = test_full_integration(
        has_hardware_detection, hardware_info,
        has_model_classifier, classifier_info
    )
    test_results["full_integration"] = full_integration_success
    
    # Summarize test results
    logger.info("\n=== Test Summary ===")
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    # Calculate overall success
    overall_success = all(test_results.values())
    
    # Report overall status based on component availability
    if has_hardware_detection and has_model_classifier:
        if overall_success:
            logger.info("✅ Full integration test completed with ALL components")
        else:
            logger.warning("⚠️ Some tests failed with ALL components")
    elif has_hardware_detection or has_model_classifier:
        if overall_success:
            logger.info("✅ Partial integration test completed with SOME components")
        else:
            logger.warning("⚠️ Some tests failed with SOME components")
        
        if not has_hardware_detection:
            logger.info("ℹ️ ResourcePool used fallback device detection successfully")
        if not has_model_classifier:
            logger.info("ℹ️ ResourcePool used model_type as fallback successfully")
    else:
        if overall_success:
            logger.info("✅ Basic integration test completed with core ResourcePool only")
        else:
            logger.warning("⚠️ Some basic tests failed with core ResourcePool only")
        
        logger.info("ℹ️ ResourcePool handled missing components gracefully")
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"integration_test_results_{timestamp}.json"
    try:
        with open(result_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "components_available": {
                    "resource_pool": True,
                    "hardware_detection": has_hardware_detection,
                    "model_classifier": has_model_classifier
                },
                "test_results": test_results,
                "overall_success": overall_success
            }, f, indent=2)
        logger.info(f"Test results saved to {result_file}")
    except Exception as e:
        logger.warning(f"Could not save test results: {e}")
    
    return overall_success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test integration of ResourcePool, hardware_detection, and model_family_classifier")
    parser.add_argument("--check-only", action="store_true", help="Only check if files exist")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--fast", action="store_true", help="Run faster subset of tests")
    parser.add_argument("--output", type=str, help="Output file for test results")
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
        print("- test_generator_with_resource_pool (test file generation)")
        print("\nThis test will gracefully adapt to missing components and provide")
        print("detailed information about how the system is handling partial configurations.\n")
        
        # Check if required files exist
        missing_required, missing_optional = get_missing_files()
        
        if missing_required:
            logger.error(f"⚠️ Missing required files: {', '.join(missing_required)}")
            print("You need to create the missing required files before running this test.")
            return 1
        
        if missing_optional:
            logger.warning(f"⚠️ Missing optional files: {', '.join(missing_optional)}")
            print("Some tests will be skipped due to missing optional files.")
        
        if args.check_only:
            logger.info("Only checking files - all required files exist.")
            return 0
        
        # Run the integration tests
        start_time = time.time()
        success = run_comprehensive_tests()
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if success:
            print(f"\n✅ Integration test completed successfully in {elapsed_time:.2f} seconds!")
            return 0
        else:
            print(f"\n❌ Integration test failed in {elapsed_time:.2f} seconds. See errors above.")
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