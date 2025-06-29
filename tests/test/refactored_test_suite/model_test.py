
from .base_test import BaseTest
import os
import tempfile

class ModelTest(BaseTest):
    """Base class for model tests."""
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = self.temp_dir.name
        
    def tearDown(self):
        self.temp_dir.cleanup()
        super().tearDown()
    
    def load_model(self, model_name):
        """Load a model for testing."""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def verify_model_output(self, model, input_data, expected_output):
        """Verify that model produces expected output."""
        output = model(input_data)
        self.assertEqual(expected_output, output)
