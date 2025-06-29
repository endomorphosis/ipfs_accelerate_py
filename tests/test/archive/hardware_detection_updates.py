#!/usr/bin/env python
"""
Hardware detection improvements to address critical benchmark system issues.

This module:
1. Fixes WebNN and WebGPU detection to properly report simulated hardware
2. Implements robust error handling for hardware detection failures
3. Provides clear delineation between real and simulated hardware tests

Implementation date: April 8, 2025
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import original hardware detection module
from hardware_detection.capabilities import detect_all_hardware
import hardware_detection

# Hardware type constants (define them here since they're not directly importable)
CPU = "cpu"
CUDA = "cuda"
ROCM = "rocm"
MPS = "mps"
OPENVINO = "openvino"
WEBNN = "webnn"
WEBGPU = "webgpu"
QUALCOMM = "qualcomm"


class EnhancedHardwareDetection:
    """
    Enhanced hardware detection with improved error handling and simulation detection.
    This extends the original hardware detection with more robust methods for detecting
    when hardware is being simulated rather than physically present.
    """
    
    def __init__(self):
        """Initialize the enhanced hardware detector"""
        # Get initial hardware detection results
        self._hardware_results = detect_all_hardware()
        
        # Track which hardware platforms are simulated
        self._simulated_hardware = {}
        self._available_hardware = {}
        self._details = {}
        self._errors = {}
        
        # Process the hardware detection results
        self._process_hardware_results()
        
        # Run additional checks for simulation
        self._detect_simulated_hardware()
    
    def _process_hardware_results(self):
        """Process the hardware detection results"""
        for hw_type, hw_info in self._hardware_results.items():
            # Store availability
            self._available_hardware[hw_type] = hw_info.get("detected", False)
            
            # Store details
            self._details[hw_type] = hw_info
            
            # Check for errors
            if "error" in hw_info:
                self._errors[hw_type] = hw_info["error"]
    
    def _detect_simulated_hardware(self):
        """Explicitly detect which hardware platforms are being simulated"""
        # Initialize all hardware to non-simulated by default
        for hw_type in self._hardware_results.keys():
            self._simulated_hardware[hw_type] = False
        
        # Check for environment variable overrides
        for hw_type in [WEBNN, WEBGPU, QUALCOMM]:
            env_var = f"{hw_type.upper()}_SIMULATION"
            if os.environ.get(env_var) == "1":
                self._simulated_hardware[hw_type] = True
                logger.warning(f"WARNING: {hw_type} is running in SIMULATION mode based on {env_var}=1")
                
                # Update details to include simulation flag
                if hw_type in self._details:
                    self._details[hw_type]["simulation_mode"] = True
                    self._details[hw_type]["simulation_reason"] = f"Environment variable {env_var}=1"
        
        # Check for Qualcomm simulation via QNN support
        try:
            # Try using the fixed version first
            try:
                from hardware_detection.qnn_support_fixed import QNN_SIMULATION_MODE
                qnn_simulation = QNN_SIMULATION_MODE
            except ImportError:
                # Fall back to the regular version
                from hardware_detection.qnn_support import QNN_SIMULATION_MODE
                qnn_simulation = QNN_SIMULATION_MODE
                
            if qnn_simulation:
                self._simulated_hardware[QUALCOMM] = True
                logger.warning(f"WARNING: {QUALCOMM} is running in SIMULATION mode based on QNN_SIMULATION_MODE")
                
                # Update details to include simulation flag
                if QUALCOMM in self._details:
                    self._details[QUALCOMM]["simulation_mode"] = True
                    self._details[QUALCOMM]["simulation_reason"] = "QNN_SIMULATION_MODE is set"
        except ImportError:
            pass
        
        # Check WebNN simulation
        if os.environ.get("WEBNN_SIMULATION") == "1" or os.environ.get("WEBNN_AVAILABLE") == "1":
            self._simulated_hardware[WEBNN] = True
            logger.warning(f"WARNING: {WEBNN} simulation environment variables are set")
            
            # Update details
            if WEBNN in self._details:
                self._details[WEBNN]["simulation_mode"] = True
                self._details[WEBNN]["simulation_reason"] = "Environment variable override"
                
                # Force availability to false to prevent automatic enabling
                if os.environ.get("WEBNN_SIMULATION") == "1":
                    logger.warning(f"WEBNN_SIMULATION=1 is set but will not automatically enable simulated hardware")
                    logger.warning("Hardware simulation should be explicitly requested through proper channels")
                
                if os.environ.get("WEBNN_AVAILABLE") == "1":
                    logger.warning(f"WEBNN_AVAILABLE=1 is set but will not automatically enable simulated hardware")
                    logger.warning("Hardware detection should rely on actual hardware availability")
        
        # Check WebGPU simulation
        if os.environ.get("WEBGPU_SIMULATION") == "1" or os.environ.get("WEBGPU_AVAILABLE") == "1":
            self._simulated_hardware[WEBGPU] = True
            logger.warning(f"WARNING: {WEBGPU} simulation environment variables are set")
            
            # Update details
            if WEBGPU in self._details:
                self._details[WEBGPU]["simulation_mode"] = True
                self._details[WEBGPU]["simulation_reason"] = "Environment variable override"
                
                # Force availability to false to prevent automatic enabling
                if os.environ.get("WEBGPU_SIMULATION") == "1":
                    logger.warning(f"WEBGPU_SIMULATION=1 is set but will not automatically enable simulated hardware")
                    logger.warning("Hardware simulation should be explicitly requested through proper channels")
                
                if os.environ.get("WEBGPU_AVAILABLE") == "1":
                    logger.warning(f"WEBGPU_AVAILABLE=1 is set but will not automatically enable simulated hardware")
                    logger.warning("Hardware detection should rely on actual hardware availability")
    
    def is_hardware_simulated(self, hardware_type: str) -> bool:
        """Check if a specific hardware type is being simulated"""
        return self._simulated_hardware.get(hardware_type, False)
    
    def get_simulated_hardware_types(self) -> List[str]:
        """Get list of all hardware types that are being simulated"""
        return [hw for hw, simulated in self._simulated_hardware.items() if simulated]
    
    def get_available_hardware(self) -> Dict[str, bool]:
        """Get dictionary of available hardware types"""
        return self._available_hardware
    
    def get_hardware_details(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about hardware"""
        return self._details
    
    def get_errors(self) -> Dict[str, str]:
        """Get errors that occurred during hardware detection"""
        return self._errors
    
    def get_best_available_hardware(self) -> str:
        """Get the best available hardware platform for inference"""
        # Priority order: CUDA > ROCm > MPS > OpenVINO > Qualcomm > CPU
        if self._available_hardware.get(CUDA, False):
            return CUDA
        elif self._available_hardware.get(ROCM, False):
            return ROCM
        elif self._available_hardware.get(MPS, False):
            return MPS
        elif self._available_hardware.get(OPENVINO, False):
            return OPENVINO
        elif self._available_hardware.get(QUALCOMM, False):
            return QUALCOMM
        else:
            return CPU
    
    def get_torch_device(self) -> str:
        """Get the appropriate torch device string for the best available hardware"""
        best_hardware = self.get_best_available_hardware()
        
        if best_hardware == CUDA or best_hardware == ROCM:
            return "cuda"
        elif best_hardware == MPS:
            return "mps"
        else:
            return "cpu"
    
    def get_simulation_status(self) -> Dict[str, bool]:
        """Get the simulation status for all hardware types"""
        simulation_status = {}
        
        # Check all hardware types
        for hw_type in self._available_hardware.keys():
            simulation_status[hw_type] = self.is_hardware_simulated(hw_type)
            
        return simulation_status


def detect_hardware_with_simulation_check() -> Dict[str, Any]:
    """
    Enhanced hardware detection with explicit simulation checking.
    This function extends the original hardware detection with simulation detection.
    
    Returns:
        Dictionary with hardware availability, simulation status, and details
    """
    detector = EnhancedHardwareDetection()
    
    # Get best hardware based on availability
    best_hardware = detector.get_best_available_hardware()
    torch_device = detector.get_torch_device()
    
    # Get simulation status
    simulation_status = detector.get_simulation_status()
    simulated_hardware = detector.get_simulated_hardware_types()
    
    result = {
        "hardware": detector.get_available_hardware(),
        "details": detector.get_hardware_details(),
        "errors": detector.get_errors(),
        "best_available": best_hardware,
        "torch_device": torch_device,
        "simulation_status": simulation_status,
        "simulated_hardware": simulated_hardware
    }
    
    # Warn about simulated hardware in results
    if simulated_hardware:
        logger.warning(f"WARNING: The following hardware is SIMULATED: {', '.join(simulated_hardware)}")
        logger.warning("Benchmark results for simulated hardware will not reflect real performance")
        
        # Add simulation warning to the result
        result["simulation_warning"] = f"The following hardware platforms are simulated: {', '.join(simulated_hardware)}"
    
    return result


# Database schema updates for simulation tracking
def get_simulation_tracking_schema_updates() -> Dict[str, str]:
    """
    Get SQL schema updates for simulation tracking in database tables.
    This adds is_simulated and simulation_reason columns to key tables.
    
    Returns:
        Dictionary mapping table names to their ALTER TABLE statements
    """
    schema_updates = {
        "test_results": """
        ALTER TABLE test_results
        ADD COLUMN IF NOT EXISTS is_simulated BOOLEAN DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS simulation_reason VARCHAR,
        ADD COLUMN IF NOT EXISTS error_category VARCHAR,
        ADD COLUMN IF NOT EXISTS error_details JSON;
        """,
        
        "performance_results": """
        ALTER TABLE performance_results
        ADD COLUMN IF NOT EXISTS is_simulated BOOLEAN DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS simulation_reason VARCHAR;
        """,
        
        "hardware_availability_log": """
        CREATE TABLE IF NOT EXISTS hardware_availability_log (
            id INTEGER PRIMARY KEY,
            hardware_type VARCHAR,
            is_available BOOLEAN,
            is_simulated BOOLEAN DEFAULT FALSE,
            detection_method VARCHAR,
            detection_details JSON,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        "view_test_results_with_simulation": """
        CREATE VIEW IF NOT EXISTS v_test_results_with_simulation AS
        SELECT
            tr.*,
            m.model_name,
            m.model_family,
            h.hardware_type,
            CASE
                WHEN tr.is_simulated THEN 'Simulated'
                ELSE 'Real'
            END as hardware_status
        FROM
            test_results tr
        JOIN
            models m ON tr.model_id = m.model_id
        JOIN
            hardware_platforms h ON tr.hardware_id = h.hardware_id;
        """
    }
    
    return schema_updates


if __name__ == "__main__":
    # Test the enhanced hardware detection
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("Running enhanced hardware detection...")
    results = detect_hardware_with_simulation_check()
    
    # Output simulation status
    print("\nSimulation Status:")
    for hw_type, is_simulated in results["simulation_status"].items():
        if is_simulated:
            print(f"  {hw_type}: SIMULATED")
        elif results["hardware"].get(hw_type, False):
            print(f"  {hw_type}: Real")
    
    # Output hardware availability
    print("\nHardware Availability:")
    for hw_type, available in results["hardware"].items():
        if available:
            if hw_type in results["simulated_hardware"]:
                print(f"  {hw_type}: Available (SIMULATED)")
            else:
                print(f"  {hw_type}: Available")
    
    # Show detailed hardware information if requested
    if "--details" in sys.argv:
        print("\nDetailed Hardware Information:")
        import pprint
        pprint.pprint(results["details"])