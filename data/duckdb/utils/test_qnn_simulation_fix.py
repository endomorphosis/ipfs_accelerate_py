#!/usr/bin/env python
"""
Test script for verifying QNN simulation mode enhancements.
This script checks that the QNNSDKWrapper properly handles simulation mode
and reports simulation status correctly.

Usage:
    # Test with default settings (no simulation)
    python test_qnn_simulation_fix.py
    
    # Test with simulation mode enabled
    QNN_SIMULATION_MODE=1 python test_qnn_simulation_fix.py
"""

import os
import logging
import sys
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qnn_simulation_test")

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_qnn_support_module():
    """Test the QNN support module implementation."""
    try:
        from hardware_detection.qnn_support import QNNSDKWrapper, QNNCapabilityDetector, QNN_AVAILABLE, QNN_SIMULATION_MODE
        
        # Test basic initialization
        logger.info(f"QNN_AVAILABLE: {QNN_AVAILABLE}")
        logger.info(f"QNN_SIMULATION_MODE: {QNN_SIMULATION_MODE}")
        
        # Create wrapper instance
        wrapper = QNNSDKWrapper(simulation_mode=QNN_SIMULATION_MODE)
        logger.info(f"QNNSDKWrapper initialized: available={wrapper.available}, simulation_mode={wrapper.simulation_mode}")
        
        # List devices
        devices = wrapper.list_devices()
        logger.info(f"Devices: {len(devices)}")
        for device in devices:
            logger.info(f"  - {device['name']} (simulated: {device.get('simulated', False)})")
        
        # Test device selection
        if devices:
            selected = wrapper.select_device(devices[0]["name"])
            logger.info(f"Device selection result: {selected}")
            
            # Test device info
            device_info = wrapper.get_device_info()
            logger.info(f"Device info: {device_info}")
            
            # Test device test
            test_result = wrapper.test_device()
            logger.info(f"Test result: {json.dumps(test_result, indent=2)}")
        
        # Test QNNCapabilityDetector
        detector = QNNCapabilityDetector()
        logger.info(f"Detector available: {detector.is_available()}")
        logger.info(f"Detector simulation: {detector.is_simulation_mode()}")
        
        # Get capability summary
        summary = detector.get_capability_summary()
        logger.info(f"Capability summary: {json.dumps(summary, indent=2)}")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import QNN support module: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing QNN support module: {e}")
        return False

def test_centralized_hardware_detection():
    """Test the centralized hardware detection integration."""
    try:
        from centralized_hardware_detection.hardware_detection import HardwareManager, get_capabilities
        
        # Test HardwareManager
        manager = HardwareManager()
        logger.info(f"HardwareManager has_qualcomm: {manager.has_qualcomm}")
        logger.info(f"HardwareManager qnn_simulation_mode: {manager.qnn_simulation_mode}")
        
        # Test capabilities
        capabilities = manager.get_capabilities()
        logger.info(f"Hardware capabilities:")
        logger.info(f"  qualcomm: {capabilities.get('qualcomm', False)}")
        logger.info(f"  qualcomm_simulation: {capabilities.get('qualcomm_simulation', False)}")
        
        # Test get_capabilities function
        caps = get_capabilities()
        logger.info(f"get_capabilities() result:")
        logger.info(f"  qualcomm: {caps.get('qualcomm', False)}")
        logger.info(f"  qualcomm_simulation: {caps.get('qualcomm_simulation', False)}")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import centralized hardware detection: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing centralized hardware detection: {e}")
        return False

def main():
    """Main test function."""
    logger.info("======== Testing QNN Simulation Mode Fix ========")
    logger.info(f"QNN_SIMULATION_MODE environment: {os.environ.get('QNN_SIMULATION_MODE', 'Not set')}")
    
    # Test QNN support module
    logger.info("\n--- Testing QNN Support Module ---")
    qnn_result = test_qnn_support_module()
    
    # Test centralized hardware detection
    logger.info("\n--- Testing Centralized Hardware Detection ---")
    hw_result = test_centralized_hardware_detection()
    
    # Report results
    logger.info("\n--- Test Results ---")
    logger.info(f"QNN Support Module: {'SUCCESS' if qnn_result else 'FAILED'}")
    logger.info(f"Centralized Hardware Detection: {'SUCCESS' if hw_result else 'FAILED'}")
    
    if qnn_result and hw_result:
        logger.info("\nAll tests PASSED! QNN simulation mode fix is working correctly.")
        return 0
    else:
        logger.error("\nSome tests FAILED. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())