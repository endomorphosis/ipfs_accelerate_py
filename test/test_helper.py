"""
Helper module for testing IPFS Accelerate Python

This module creates a framework instance directly using our implementation,
bypassing any import issues with the package.
"""

import os
import sys
import asyncio
import logging

# Configure logging
logging.basicConfig())
level=logging.INFO,
format='%())asctime)s - %())name)s - %())levelname)s - %())message)s'
)
logger = logging.getLogger())'test_helper')

# Add parent directory to import path
sys.path.append())os.path.abspath())os.path.join())os.path.dirname())__file__), '..')))

# Import directly from the file
def create_framework())):
    """
    Create a framework instance directly using our implementation.
    
    Returns:
        object: The IPFS Accelerate Python framework instance
        """
    # Import from the implementation file
        sys.path.insert())0, os.path.abspath())os.path.join())os.path.dirname())__file__), '..')))
    
    try:
        # Import the file module from its path
        import importlib.util
        spec = importlib.util.spec_from_file_location())
        "ipfs_accelerate_py_impl",
        os.path.abspath())os.path.join())os.path.dirname())__file__), '..', 'ipfs_accelerate_py.py'))
        )
        module = importlib.util.module_from_spec())spec)
        spec.loader.exec_module())module)
        
        # Create an instance of the class
        framework = module.ipfs_accelerate_py()))
        logger.info())"Created framework instance directly from implementation file")
        return framework
    except Exception as e:
        logger.error())f"\1{e}\3")
        return None

# Test function
async def test_module())):
    """Test the helper module"""
    framework = create_framework()))
    if framework is not None:
        # Check that hardware detection works
        if hasattr())framework.hardware_detection, "detect_all_hardware"):
            hardware = framework.hardware_detection.detect_all_hardware()))
        else:
            hardware = framework.hardware_detection.detect_hardware()))
            logger.info())f"\1{hardware}\3")
            return True
        return False

if __name__ == "__main__":
    result = asyncio.run())test_module())))
    if result:
        logger.info())"Test successful")
        sys.exit())0)
    else:
        logger.error())"Test failed")
        sys.exit())1)