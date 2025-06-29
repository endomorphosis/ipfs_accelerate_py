"""
Refactored Test Suite for IPFS Accelerate Python Framework.

This package contains the refactored test structure using standardized
base classes and organization patterns.
"""

__version__ = "0.1.0"

# Import base classes for easier access
from .base_test import BaseTest
from .model_test import ModelTest
from .hardware_test import HardwareTest
from .browser_test import BrowserTest
from .api_test import APITest