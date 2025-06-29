#!/usr/bin/env python3
"""
Simple test file with corrected indentation.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestModel:
    """Class with proper indentation."""
    
    def __init__(self, model_id=None):
        """Initialize with consistent spacing."""
        self.model_id = model_id or "test-model"
        
    def test_method(self, param1=None):
        """Method with proper indentation."""
        if param1 is None:
            param1 = "default"
            
        # Process input
        result = self.process_data(param1)
        return result
    
    def process_data(self, data):
        """Process the input data."""
        processed = data.upper()
        return {
            "model": self.model_id,
            "input": data,
            "output": processed
        }

def main():
    model = TestModel("test-model")
    result = model.test_method("sample")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()