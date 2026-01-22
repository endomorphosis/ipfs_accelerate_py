#!/usr/bin/env python3
"""
Test suite for the HuggingFace test generator.
This test suite validates that the test generator produces syntactically correct
and functional test files for different model architectures.
"""

import os
import sys
import unittest
import subprocess
import tempfile
import importlib.util
from pathlib import Path

# Model families to test
MODEL_FAMILIES = ["bert", "gpt2", "t5", "vit"]

class TestGeneratorTestCase(unittest.TestCase):
    """Test case for the HuggingFace test generator."""
    
    def setUp(self):
        """Set up the test case."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Path to the test generator script
        self.generator_path = os.path.abspath("../test_generator.py")
        
        # Check if the generator exists
        if not os.path.exists(self.generator_path):
            raise FileNotFoundError(f"Generator script not found at {self.generator_path}")
    
    def tearDown(self):
        """Clean up after the test case."""
        self.temp_dir.cleanup()
    
    def test_generator_imports(self):
        """Test that the generator can be imported without errors."""
        spec = importlib.util.spec_from_file_location("test_generator", self.generator_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            self.assertTrue(hasattr(module, "generate_test_file"))
        except Exception as e:
            self.fail(f"Failed to import test generator: {e}")
    
    def test_file_generation(self):
        """Test that the generator can generate files for all model families."""
        for family in MODEL_FAMILIES:
            output_path = os.path.join(self.output_dir, f"test_hf_{family}.py")
            
            # Generate test file
            cmd = [
                sys.executable,
                self.generator_path,
                "--family", family,
                "--output", self.output_dir
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, 
                                f"Generator failed for {family}: {result.stderr}")
                
                # Check that the file exists
                self.assertTrue(os.path.exists(output_path), 
                               f"Generated file does not exist: {output_path}")
                
                # Check file size
                file_size = os.path.getsize(output_path)
                self.assertGreater(file_size, 1000, 
                                  f"Generated file is too small: {file_size} bytes")
                
                # Check syntax
                syntax_check = subprocess.run(
                    [sys.executable, "-m", "py_compile", output_path],
                    capture_output=True,
                    text=True
                )
                self.assertEqual(syntax_check.returncode, 0, 
                                f"Syntax check failed: {syntax_check.stderr}")
            except Exception as e:
                self.fail(f"Test generation failed for {family}: {e}")
    
    def test_architecture_specifics(self):
        """Test that the generator includes architecture-specific code."""
        for family in MODEL_FAMILIES:
            output_path = os.path.join(self.output_dir, f"test_hf_{family}.py")
            
            # Generate test file
            cmd = [
                sys.executable,
                self.generator_path,
                "--family", family,
                "--output", self.output_dir
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Read generated file
                with open(output_path, 'r') as f:
                    content = f.read()
                
                # Check architecture-specific patterns
                if family == "bert":
                    self.assertIn("BERT_MODELS_REGISTRY", content)
                    self.assertIn("fill-mask", content)
                elif family == "gpt2":
                    self.assertIn("GPT2_MODELS_REGISTRY", content)
                    self.assertIn("text-generation", content)
                    # Check for decoder-only specific handling
                    self.assertIn("tokenizer.pad_token", content)
                elif family == "t5":
                    self.assertIn("T5_MODELS_REGISTRY", content)
                    self.assertIn("translation", content)
                    # Check for encoder-decoder specific handling
                    self.assertIn("encoder-decoder", content)
                elif family == "vit":
                    self.assertIn("VIT_MODELS_REGISTRY", content)
                    self.assertIn("image-classification", content)
                    # Check for vision-specific handling
                    self.assertIn("image", content)
            except Exception as e:
                self.fail(f"Architecture test failed for {family}: {e}")
    
    def test_hardware_detection(self):
        """Test that the generator includes hardware detection code."""
        # Generate test file for any family
        output_path = os.path.join(self.output_dir, "test_hf_bert.py")
        
        cmd = [
            sys.executable,
            self.generator_path,
            "--family", "bert",
            "--output", self.output_dir
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Read generated file
            with open(output_path, 'r') as f:
                content = f.read()
            
            # Check hardware detection code
            self.assertIn("check_hardware", content)
            self.assertIn("cuda", content)
            self.assertIn("mps", content)
            self.assertIn("openvino", content)
        except Exception as e:
            self.fail(f"Hardware detection test failed: {e}")
    
    def test_mock_imports(self):
        """Test that the generator includes mock imports for missing packages."""
        output_path = os.path.join(self.output_dir, "test_hf_bert.py")
        
        cmd = [
            sys.executable,
            self.generator_path,
            "--family", "bert",
            "--output", self.output_dir
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Read generated file
            with open(output_path, 'r') as f:
                content = f.read()
            
            # Check mock imports
            self.assertIn("try:", content)
            self.assertIn("import torch", content)
            self.assertIn("except ImportError:", content)
            self.assertIn("MagicMock", content)
        except Exception as e:
            self.fail(f"Mock imports test failed: {e}")

def run_tests():
    """Run all tests."""
    unittest.main()

if __name__ == "__main__":
    run_tests()