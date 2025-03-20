"""IPFS Accelerate Testing Framework

A comprehensive testing framework for the IPFS Accelerate Python library.
This package provides utilities, fixtures, and templates for testing
IPFS Accelerate across various hardware platforms and model types.

Structure:
- models/: Tests for specific model types (BERT, T5, ViT, etc.)
- hardware/: Tests for hardware platforms (WebGPU, WebNN, CUDA, etc.)
- api/: Tests for API endpoints and clients
- integration/: Tests for cross-system integration
- common/: Shared utilities and fixtures
- template_system/: Templates for generating tests
- docs/: Documentation for the testing framework

Key components:
- run.py: Unified test runner
- conftest.py: Global pytest configuration
- pytest.ini: Test session configuration
- common/hardware_detection.py: Hardware detection utilities
- common/model_helpers.py: Model utilities
- common/fixtures.py: Common test fixtures
"""

__version__ = "0.1.0"