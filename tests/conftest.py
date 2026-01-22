#!/usr/bin/env python3
"""
Pytest configuration for tests directory.
"""

import pytest

# Register custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to skip certain tests that are standalone scripts.
    """
    skip_standalone = pytest.mark.skip(reason="Standalone script, not a pytest test")
    
    for item in items:
        # Skip tests that are actually standalone scripts
        if item.nodeid.startswith("tests/test_single_import.py"):
            item.add_marker(skip_standalone)
        elif item.nodeid.startswith("tests/test_comprehensive_validation.py"):
            item.add_marker(skip_standalone)
        elif item.nodeid.startswith("tests/test_hf_api_integration.py"):
            item.add_marker(skip_standalone)
