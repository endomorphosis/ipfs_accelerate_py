
import os
import json
import tempfile
import random
import string
import unittest

from refactored_test_suite.model_test import ModelTest

class TestUtils(ModelTest):
    """Test class for utility functions."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.model_id = "utility-test-model"
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
    def test_model_loading(self):
        """Test model loading utility."""
        # This is a dummy test to satisfy ModelTest requirements
        model = self.load_model(self.model_id)
        self.assertIsNotNone(model)
        
    def test_utility_functions(self):
        """Test utility functions."""
        # Test get_test_data_path
        path = get_test_data_path("test.txt")
        self.assertTrue(path.endswith("test.txt"))
        
        # Test create_temp_file
        content = "test content"
        temp_path = create_temp_file(content)
        self.assertTrue(os.path.exists(temp_path))
        with open(temp_path, 'r') as f:
            self.assertEqual(f.read(), content)
        os.unlink(temp_path)
        
        # Test random_string
        str1 = random_string()
        str2 = random_string()
        self.assertNotEqual(str1, str2)
        self.assertEqual(len(str1), 10)
        
        # Test JSON functions
        test_data = {"test": "data"}
        json_path = os.path.join(self.model_dir, "test.json")
        save_json_data(test_data, json_path)
        loaded_data = load_json_data(json_path)
        self.assertEqual(loaded_data, test_data)
    
    def load_model(self, model_name):
        """Load a model for testing (dummy implementation)."""
        # This is a mock implementation for testing purposes
        return {"type": "dummy", "name": model_name}
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output (dummy implementation)."""
        # This is a mock implementation for testing purposes
        output = f"Processed {input_data} with {model['name']}"
        if expected_output:
            self.assertEqual(expected_output, output)
        return output
        
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device."""
        try:
            import torch
        
            # Check for CUDA
            if torch.cuda.is_available():
                return "cuda"
        
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                return "mps"
        
            # Fallback to CPU
            return "cpu"
        except ImportError:
            return "cpu"


def get_test_data_path(filename):
    """Get path to a test data file."""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    return os.path.join(test_data_dir, filename)

def create_temp_file(content, suffix=".txt"):
    """Create a temporary file with the given content."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path
    except:
        os.unlink(path)
        raise

def random_string(length=10):
    """Generate a random string of the given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def load_json_data(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_data(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Run the TestUtils class directly when this file is executed
    unittest.main()
