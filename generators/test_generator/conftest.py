
import pytest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def test_dir():
    """Return the directory containing test data."""
    return os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
