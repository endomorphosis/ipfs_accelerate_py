#!/usr/bin/env python3
"""
QNN Simulation Helper

This script provides explicit control for QNN simulation mode.
It ensures QNN hardware simulation is only enabled when explicitly requested
and clearly marks all results as simulated.

April 2025 Update: Part of the benchmark system improvements from NEXT_STEPS.md
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def enable_qnn_simulation():
    """
    Enable QNN simulation mode by setting environment variable.
    This should only be done for testing or demonstrations.
    """
    os.environ["QNN_SIMULATION_MODE"] = "1"
    logger.warning("SIMULATION MODE ENABLED: QNN hardware will be simulated")
    logger.warning("Results will NOT reflect real hardware performance")
    logger.warning("All results will be clearly marked as SIMULATED")
    
    # Import QNN support module and call simulation setup
    try:
        if os.path.exists('/home/barberb/ipfs_accelerate_py/test/hardware_detection/qnn_support_fixed.py'):
            # Use fixed implementation if available
            sys.path.append('/home/barberb/ipfs_accelerate_py/test/hardware_detection')
            from qnn_support_fixed import setup_qnn_simulation
        else:
            # Fall back to original implementation if fixed version not available
            from hardware_detection.qnn_support import setup_qnn_simulation
        
        # Set up simulation with sample devices
        setup_qnn_simulation()
        logger.info("QNN simulation setup completed")
        return True
    except ImportError as e:
        logger.error(f"Error importing QNN support module: {e}")
        return False
    except Exception as e:
        logger.error(f"Error setting up QNN simulation: {e}")
        return False

def disable_qnn_simulation():
    """
    Disable QNN simulation mode by removing environment variable.
    """
    if "QNN_SIMULATION_MODE" in os.environ:
        del os.environ["QNN_SIMULATION_MODE"]
        logger.info("QNN simulation mode disabled")
    else:
        logger.info("QNN simulation was not enabled")
    return True

def check_qnn_simulation_status():
    """
    Check if QNN simulation mode is enabled.
    """
    simulation_enabled = os.environ.get("QNN_SIMULATION_MODE", "0").lower() in ("1", "true", "yes")
    
    if simulation_enabled:
        logger.warning("QNN SIMULATION MODE IS CURRENTLY ENABLED")
        logger.warning("Results will NOT reflect real hardware performance")
        logger.warning("All results will be marked as SIMULATED")
    else:
        logger.info("QNN SIMULATION MODE IS NOT ENABLED")
        
    return simulation_enabled

def main():
    """Main entry point for controlling QNN simulation mode"""
    parser = argparse.ArgumentParser(description="QNN Simulation Helper")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enable", action="store_true",
                       help="Enable QNN simulation mode (for testing or demonstrations only)")
    group.add_argument("--disable", action="store_true",
                      help="Disable QNN simulation mode")
    group.add_argument("--check", action="store_true",
                      help="Check if QNN simulation mode is enabled")
    
    args = parser.parse_args()
    
    if args.enable:
        if enable_qnn_simulation():
            return 0
        else:
            return 1
    elif args.disable:
        if disable_qnn_simulation():
            return 0
        else:
            return 1
    elif args.check:
        simulation_enabled = check_qnn_simulation_status()
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())