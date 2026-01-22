"""
iOS Test Harness for IPFS Accelerate Python Framework

This package provides tools for testing, benchmarking, and analyzing machine learning
models on iOS devices, with support for Core ML models, thermal monitoring, and
performance metrics collection.

Components:
    - IOSDevice: Manages iOS device connections
    - IOSModelRunner: Handles model deployment and execution
    - IOSTestHarness: Main class orchestrating the testing process

Date: April 2025
Status: Phase 2 (Alpha) Implementation
"""

from .ios_test_harness import IOSDevice, IOSModelRunner, IOSTestHarness

__all__ = [
    'IOSDevice',
    'IOSModelRunner', 
    'IOSTestHarness'
]