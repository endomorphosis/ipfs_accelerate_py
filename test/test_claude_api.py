#!/usr/bin/env python
"""
Test Claude API implementation
"""

import os
import sys
import importlib
import inspect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_claude_api")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_claude_api():
    """Test the Claude API implementation"""
    # Import the module
    try:
        module = importlib.import_module("ipfs_accelerate_py.api_backends.claude")
        logger.info("✅ Successfully imported claude module")
    except ImportError as e:
        logger.error(f"❌ Failed to import claude module: {e}")
        return False
    
    # Look for the class
    try:
        if hasattr(module, 'claude'):
            cls = getattr(module, 'claude')
            if inspect.isclass(cls):
                logger.info("✅ Found claude class in module")
            else:
                logger.error("❌ 'claude' is not a class in module")
                return False
        else:
            logger.error("❌ Could not find 'claude' class in module")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking for claude class: {e}")
        return False
    
    # Try to instantiate the class with minimal arguments
    try:
        # Patch _get_api_key to avoid any dependency on external resources
        original_get_api_key = cls._get_api_key
        cls._get_api_key = lambda self, metadata: "mock_claude_api_key_for_testing"
        
        instance = cls(resources={}, metadata={})
        logger.info("✅ Successfully instantiated claude class")
        
        # Restore original method
        cls._get_api_key = original_get_api_key
    except Exception as e:
        logger.error(f"❌ Failed to instantiate claude class: {e}")
        # Print the trace
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if test_claude_api():
        logger.info("✅ All Claude API tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Claude API tests failed")
        sys.exit(1)