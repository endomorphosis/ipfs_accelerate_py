import unittest
import logging
import json
import os

# Optional pytest import - only if available
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

class BaseTest(unittest.TestCase):
    """Base class for all tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up resources for the entire test class."""
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.setLevel(logging.INFO)
        
        # Ensure we have a handler to prevent "no handlers" warnings
        if not cls.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            cls.logger.addHandler(handler)
    
    def setUp(self):
        """Set up resources for each test method."""
        self.logger.info(f"Setting up test: {self._testMethodName}")
    
    def tearDown(self):
        """Clean up resources after each test method."""
        self.logger.info(f"Tearing down test: {self._testMethodName}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources after the entire test class."""
        pass
    
    def load_json_data(self, filepath):
        """Load JSON data from a file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def assertDictContainsSubset(self, subset, dictionary, msg=None):
        """Assert that dictionary contains subset."""
        for key, value in subset.items():
            if key not in dictionary:
                self.fail(f"{msg or 'Missing key'}: {key}")
            if value != dictionary[key]:
                self.fail(f"{msg or 'Value mismatch for key'}: {key}")