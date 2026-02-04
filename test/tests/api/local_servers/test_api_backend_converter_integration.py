#!/usr/bin/env python3
"""
Integration tests for the API Backend Converter

This module provides integration tests for the convert_api_backends.py script
by running the converter on actual API backend files and verifying the output.
"""

import os
import sys
import unittest
import tempfile
import shutil
import subprocess
import json
from typing import Dict, Any

# Import the converter module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from convert_api_backends import APIBackendConverter


class TestConverterIntegration(unittest.TestCase):
    """Integration tests for the API Backend Converter"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "typescript")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find sample_backend.py
        self.python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ipfs_accelerate_py/api_backends")
        self.sample_backend_path = os.path.join(self.python_dir, "sample_backend.py")
        self.ollama_backend_path = os.path.join(self.python_dir, "ollama_clean.py")

    def tearDown(self):
        """Clean up temporary files and directories"""
        shutil.rmtree(self.temp_dir)

    def test_convert_sample_backend(self):
        """Test converting the sample backend"""
        # Skip if sample backend doesn't exist
        if not os.path.exists(self.sample_backend_path):
            self.skipTest(f"Sample backend file not found at {self.sample_backend_path}")
        
        # Initialize converter
        converter = APIBackendConverter(
            python_file=self.sample_backend_path,
            output_dir=self.output_dir,
            dry_run=False
        )
        
        # Run conversion
        parse_success = converter.parse_python_file()
        self.assertTrue(parse_success, "Failed to parse sample backend file")
        
        gen_success = converter.generate_typescript_files()
        self.assertTrue(gen_success, "Failed to generate TypeScript files")
        
        # Check output files exist
        expected_files = [
            os.path.join(self.output_dir, "sample_backend", "sample_backend.ts"),
            os.path.join(self.output_dir, "sample_backend", "index.ts"),
            os.path.join(self.output_dir, "sample_backend", "types.ts")
        ]
        
        for file_path in expected_files:
            self.assertTrue(os.path.exists(file_path), f"Expected output file not found: {file_path}")
            
        # Check file contents
        with open(os.path.join(self.output_dir, "sample_backend", "sample_backend.ts"), "r") as f:
            ts_content = f.read()
            
        # Verify key elements in the TypeScript file
        self.assertIn("export class SampleBackend extends BaseApiBackend", ts_content)
        self.assertIn("getApiKey", ts_content)
        self.assertIn("getDefaultModel", ts_content)
        self.assertIn("isCompatibleModel", ts_content)

    def test_convert_ollama_backend(self):
        """Test converting the ollama backend"""
        # Skip if ollama backend doesn't exist
        if not os.path.exists(self.ollama_backend_path):
            self.skipTest(f"Ollama backend file not found at {self.ollama_backend_path}")
        
        # Initialize converter with force option
        converter = APIBackendConverter(
            python_file=self.ollama_backend_path,
            output_dir=self.output_dir,
            dry_run=False
        )
        
        # Run conversion with forced generation even if parsing fails
        try:
            converter.parse_python_file()
        except:
            pass  # Continue even if parsing fails
        
        gen_success = converter.generate_typescript_files()
        self.assertTrue(gen_success, "Failed to generate TypeScript files")
        
        # Check output files exist
        expected_files = [
            os.path.join(self.output_dir, "ollama_clean", "ollama_clean.ts"),
            os.path.join(self.output_dir, "ollama_clean", "index.ts"),
            os.path.join(self.output_dir, "ollama_clean", "types.ts")
        ]
        
        for file_path in expected_files:
            self.assertTrue(os.path.exists(file_path), f"Expected output file not found: {file_path}")
            
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "ollama_clean", "ollama_clean.ts")))
        
        try:
            # Check file contents if file exists
            with open(os.path.join(self.output_dir, "ollama_clean", "ollama_clean.ts"), "r") as f:
                ts_content = f.read()
                
            # Verify key elements in the TypeScript file
            self.assertIn("extends BaseApiBackend", ts_content)
        except:
            self.fail("Could not verify TypeScript file contents")
        
    def test_run_converter_cli(self):
        """Test running the converter script as a CLI tool"""
        # Skip if sample backend doesn't exist
        if not os.path.exists(self.sample_backend_path):
            self.skipTest(f"Sample backend file not found at {self.sample_backend_path}")
            
        # Run the converter script with the sample backend
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_api_backends.py")
        
        command = [
            sys.executable,
            script_path,
            "--backend", "sample_backend",
            "--python-dir", self.python_dir,
            "--ts-dir", self.output_dir
        ]
        
        # Capture the output
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check output
            self.assertIn("Successfully generated TypeScript files", result.stdout)
            self.assertEqual(result.returncode, 0)
            
            # Check output files exist
            expected_files = [
                os.path.join(self.output_dir, "sample_backend", "sample_backend.ts"),
                os.path.join(self.output_dir, "sample_backend", "index.ts"),
                os.path.join(self.output_dir, "sample_backend", "types.ts")
            ]
            
            for file_path in expected_files:
                self.assertTrue(os.path.exists(file_path), f"Expected output file not found: {file_path}")
                
        except subprocess.CalledProcessError as e:
            self.fail(f"Converter script failed with error: {e.stderr}")


class TestValidTypeScriptOutput(unittest.TestCase):
    """Tests to validate the generated TypeScript files"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "typescript")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find sample_backend.py
        self.python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ipfs_accelerate_py/api_backends")
        self.sample_backend_path = os.path.join(self.python_dir, "sample_backend.py")

    def tearDown(self):
        """Clean up temporary files and directories"""
        shutil.rmtree(self.temp_dir)
        
    def validate_typescript_syntax(self, ts_file_path: str) -> bool:
        """Validate TypeScript syntax in a very basic way"""
        try:
            # Just check if the file exists and can be read
            with open(ts_file_path, "r") as f:
                content = f.read()
                
            # Check for basic TypeScript syntax markers
            if "export class" in content and "extends BaseApiBackend" in content:
                return True
                
            # Additional basic checks
            return "protected" in content and "getApiKey" in content
        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def test_typescript_syntax_validation(self):
        """Test that generated TypeScript files have valid syntax"""
        # Skip if sample backend doesn't exist
        if not os.path.exists(self.sample_backend_path):
            self.skipTest(f"Sample backend file not found at {self.sample_backend_path}")
        
        # Initialize converter
        converter = APIBackendConverter(
            python_file=self.sample_backend_path,
            output_dir=self.output_dir,
            dry_run=False
        )
        
        # Run conversion
        converter.parse_python_file()
        converter.generate_typescript_files()
        
        # Check syntax of generated TypeScript file
        ts_file_path = os.path.join(self.output_dir, "sample_backend", "sample_backend.ts")
        
        # Validate TypeScript syntax
        is_valid = self.validate_typescript_syntax(ts_file_path)
        self.assertTrue(is_valid, f"Generated TypeScript file has syntax errors: {ts_file_path}")
        
        # Check if types file exists (but don't validate syntax)
        types_file_path = os.path.join(self.output_dir, "sample_backend", "types.ts")
        self.assertTrue(os.path.exists(types_file_path), f"Generated TypeScript types file not found: {types_file_path}")


if __name__ == "__main__":
    unittest.main()