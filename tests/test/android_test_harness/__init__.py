"""
Android Test Harness for IPFS Accelerate Python Framework

This package provides tools for testing, benchmarking, and analyzing machine learning
models on Android devices, with support for real model execution, hardware acceleration,
thermal monitoring, and performance metrics collection.

Components:
    - AndroidDevice: Manages Android device connections
    - AndroidModelRunner: Handles model deployment and execution
    - AndroidModelExecutor: Implements real model execution with hardware acceleration
    - AndroidThermalMonitor: Monitors thermal conditions during execution
    - AndroidTestHarness: Main class orchestrating the testing process

Date: April 2025
Status: Phase 2 (Alpha) Implementation
"""

from .android_test_harness import AndroidDevice, AndroidModelRunner, AndroidTestHarness

# Only import these if available
try:
    from .android_model_executor import AndroidModelExecutor, ModelFormat, AcceleratorType
except ImportError:
    pass

try:
    from .android_thermal_monitor import AndroidThermalMonitor
except ImportError:
    pass

__all__ = [
    'AndroidDevice',
    'AndroidModelRunner',
    'AndroidTestHarness',
]

# Add optional components to __all__ if available
try:
    __all__.extend(['AndroidModelExecutor', 'ModelFormat', 'AcceleratorType'])
except NameError:
    pass

try:
    __all__.extend(['AndroidThermalMonitor'])
except NameError:
    pass