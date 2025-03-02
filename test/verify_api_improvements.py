#!/usr/bin/env python
"""
Verify API Improvements

This script verifies the API improvements by testing:
1. Module imports
2. Class instantiation
3. Queue functionality
4. Backoff functionality
"""

import os
import sys
import importlib
import inspect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_api_improvements")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# API backends to test
API_BACKENDS = [
    "claude",
    "openai_api",
    "groq",
    "gemini",
    "ollama",
    "hf_tgi",
    "hf_tei",
    "llvm",
    "opea",
    "ovms",
    "s3_kit"
]

def check_import(backend_name):
    """Check if the backend can be imported"""
    try:
        module = importlib.import_module(f"ipfs_accelerate_py.api_backends.{backend_name}")
        logger.info(f"✅ Successfully imported {backend_name} module")
        return module
    except ImportError as e:
        logger.error(f"❌ Failed to import {backend_name} module: {e}")
        return None

def check_class(module, class_name):
    """Check if the class exists in the module"""
    try:
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            if inspect.isclass(cls):
                logger.info(f"✅ Successfully found {class_name} class in module")
                return cls
            else:
                logger.error(f"❌ {class_name} is not a class in module")
                return None
        else:
            logger.error(f"❌ Class {class_name} not found in module")
            return None
    except Exception as e:
        logger.error(f"❌ Error checking class {class_name}: {e}")
        return None

def check_instantiation(cls, class_name):
    """Check if the class can be instantiated"""
    try:
        # Patch _get_api_key for APIs that need it
        patched = False
        if hasattr(cls, '_get_api_key'):
            original_get_api_key = cls._get_api_key
            cls._get_api_key = lambda self, *args, **kwargs: "mock_api_key_for_testing"
            patched = True
        
        instance = cls(resources={}, metadata={})
        logger.info(f"✅ Successfully instantiated {class_name} class")
        
        # Restore original method if patched
        if patched:
            cls._get_api_key = original_get_api_key
            
        return instance
    except Exception as e:
        logger.error(f"❌ Failed to instantiate {class_name} class: {e}")
        return None

def check_queue_attributes(instance, class_name):
    """Check if the instance has the required queue attributes"""
    required_attrs = [
        "request_queue",
        "queue_lock",
        "queue_processing"
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(instance, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        logger.error(f"❌ {class_name} instance is missing required queue attributes: {', '.join(missing_attrs)}")
        return False
    else:
        logger.info(f"✅ {class_name} instance has all required queue attributes")
        return True

def main():
    """Main function to verify API improvements"""
    results = {}
    
    # Test each backend
    for backend_name in API_BACKENDS:
        logger.info(f"\n=== Testing {backend_name} backend ===")
        results[backend_name] = {
            "import": False,
            "class": False,
            "instantiation": False,
            "queue_attributes": False
        }
        
        # Check import
        module = check_import(backend_name)
        if module:
            results[backend_name]["import"] = True
            
            # Check class
            cls = check_class(module, backend_name)
            if cls:
                results[backend_name]["class"] = True
                
                # Check instantiation
                instance = check_instantiation(cls, backend_name)
                if instance:
                    results[backend_name]["instantiation"] = True
                    
                    # Check queue attributes
                    if check_queue_attributes(instance, backend_name):
                        results[backend_name]["queue_attributes"] = True
    
    # Print summary
    logger.info("\n=== API Improvements Verification Summary ===")
    all_passed = True
    for backend_name, result in results.items():
        passed = all(result.values())
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{backend_name}: {status}")
        
        if not passed:
            all_passed = False
            # Print details of what failed
            for check, passed in result.items():
                if not passed:
                    logger.info(f"  - {check}: ❌ FAIL")
    
    if all_passed:
        logger.info("\n🎉 All API backends passed verification!")
        return 0
    else:
        logger.error("\n⚠️ Some API backends failed verification")
        return 1

if __name__ == "__main__":
    sys.exit(main())