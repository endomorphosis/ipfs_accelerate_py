#!/usr/bin/env python3
"""
Unit tests for the template system enhancements.
"""

import os
import sys
import unittest
import tempfile
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import duckdb
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.error("DuckDB not available. These tests require DuckDB.")
    sys.exit(1)

# Import functions to test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_templates.template_system_enhancement import (
    extract_placeholders,
    validate_template_syntax,
    validate_hardware_support,
    validate_template
)

class TestTemplateEnhancements(unittest.TestCase):
    """Test the template system enhancements."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_template_db.duckdb")
        
        # Create a test database
        try:
            conn = duckdb.connect(self.db_path)
            conn.execute("""
            CREATE TABLE templates (
                model_type VARCHAR,
                template_type VARCHAR,
                template TEXT,
                hardware_platform VARCHAR
            )
            """)
            
            # Add a test template
            conn.execute("""
            INSERT INTO templates 
            (model_type, template_type, template, hardware_platform)
            VALUES (?, ?, ?, NULL)
            """, ["test_model", "test", "Template for {model_name}", None])
            
            conn.close()
        except Exception as e:
            logger.error(f"Error creating test database: {e}")
            self.fail(f"Could not create test database: {e}")

    def tearDown(self):
        """Clean up test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_extract_placeholders(self):
        """Test extracting placeholders from a template."""
        # Test a simple template
        template = "Template for {model_name} with {normalized_name} and {generated_at}"
        placeholders = extract_placeholders(template)
        self.assertEqual(placeholders, {"model_name", "normalized_name", "generated_at"})
        
        # Test a template with no placeholders
        template = "Template with no placeholders"
        placeholders = extract_placeholders(template)
        self.assertEqual(placeholders, set())
        
        # Test a template with duplicate placeholders
        template = "Template with {model_name} and {model_name} and {normalized_name}"
        placeholders = extract_placeholders(template)
        self.assertEqual(placeholders, {"model_name", "normalized_name"})

    def test_validate_template_syntax(self):
        """Test validating template syntax."""
        # Test a valid template
        template = """#!/usr/bin/env python3
\"\"\"
Test template for {model_name}
\"\"\"

import os

def test_function():
    \"\"\"Test function.\"\"\"
    print("Testing {model_name}")
"""
        success, errors = validate_template_syntax(template)
        self.assertTrue(success)
        self.assertEqual(errors, [])
        
        # Test a template with unbalanced braces
        template = """#!/usr/bin/env python3
\"\"\"
Test template for {model_name
\"\"\"

import os

def test_function():
    \"\"\"Test function.\"\"\"
    print("Testing {model_name}")
"""
        success, errors = validate_template_syntax(template)
        self.assertFalse(success)
        self.assertTrue(any("Unbalanced braces" in error for error in errors))
        
        # Test a template with Python syntax errors
        template = """#!/usr/bin/env python3
\"\"\"
Test template for {model_name}
\"\"\"

import os

def test_function()
    \"\"\"Test function.\"\"\"
    print("Testing {model_name}")
"""
        success, errors = validate_template_syntax(template)
        self.assertFalse(success)
        self.assertTrue(any("Python syntax error" in error for error in errors))
        
        # Test a template with double braces
        template = """#!/usr/bin/env python3
\"\"\"
Test template for {{model_name}}
\"\"\"
"""
        success, errors = validate_template_syntax(template)
        self.assertFalse(success)
        self.assertTrue(any("Double braces" in error for error in errors))

    def test_validate_hardware_support(self):
        """Test validating hardware support."""
        # Test a template with CUDA support
        template = """#!/usr/bin/env python3
\"\"\"
Test template with CUDA support
\"\"\"

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
"""
        success, hardware_support = validate_hardware_support(template)
        self.assertTrue(success)
        self.assertTrue(hardware_support["cpu"])
        self.assertTrue(hardware_support["cuda"])
        
        # Test a template with MPS support
        template = """#!/usr/bin/env python3
\"\"\"
Test template with MPS support
\"\"\"

import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
"""
        success, hardware_support = validate_hardware_support(template)
        self.assertTrue(success)
        self.assertTrue(hardware_support["cpu"])
        self.assertTrue(hardware_support["mps"])
        
        # Test a template with OpenVINO support
        template = """#!/usr/bin/env python3
\"\"\"
Test template with OpenVINO support
\"\"\"

import openvino
"""
        success, hardware_support = validate_hardware_support(template)
        self.assertTrue(success)
        self.assertTrue(hardware_support["cpu"])
        self.assertTrue(hardware_support["openvino"])
        
        # Test validating support for a specific hardware platform
        template = """#!/usr/bin/env python3
\"\"\"
Test template with CUDA support
\"\"\"

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
"""
        success, hardware_support = validate_hardware_support(template, "cuda")
        self.assertTrue(success)
        
        template = """#!/usr/bin/env python3
\"\"\"
Test template with no specific hardware support
\"\"\"

import os
"""
        success, hardware_support = validate_hardware_support(template, "cuda")
        self.assertFalse(success)

    def test_validate_template(self):
        """Test validating a complete template."""
        # Test a valid template
        template = """#!/usr/bin/env python3
\"\"\"
Test template for {model_name}
Generated on {generated_at}
\"\"\"

import os
import torch

class Test{normalized_name}:
    \"\"\"Test class for {model_name}.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize test.\"\"\"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def run(self):
        \"\"\"Run test.\"\"\"
        print(f"Testing {model_name} on {self.device}")
"""
        success, results = validate_template(template, "test", "bert")
        self.assertTrue(success)
        self.assertTrue(results["syntax"]["success"])
        self.assertTrue(results["hardware"]["success"])
        self.assertTrue(results["placeholders"]["success"])
        
        # Test a template with missing placeholders
        template = """#!/usr/bin/env python3
\"\"\"
Test template
\"\"\"

import os

def test_function():
    \"\"\"Test function.\"\"\"
    print("Testing")
"""
        success, results = validate_template(template, "test", "bert")
        self.assertFalse(success)
        self.assertTrue(results["syntax"]["success"])
        self.assertTrue(results["hardware"]["success"])
        self.assertFalse(results["placeholders"]["success"])
        self.assertTrue(len(results["placeholders"]["missing"]) > 0)

if __name__ == "__main__":
    unittest.main()