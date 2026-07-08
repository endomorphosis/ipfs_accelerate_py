#!/usr/bin/env python3
"""
Test script for the mock detection visualization system.
This script generates sample data and runs the visualizations to verify they work.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from typing import List, Dict, Any

# Import visualization modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mock_detection_visualization import (
    scan_test_files_for_mock_status,
    create_dependency_heatmap,
    create_mock_detection_summary,
    create_model_family_analysis,
    create_test_result_analysis,
    create_interactive_dashboard,
    generate_report
)

class TestMockDetectionVisualization(unittest.TestCase):
    """Test cases for mock detection visualization."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample scan results
        self.scan_results = [
            {
                "file": "test_hf_bert.py",
                "model": "hf_bert",
                "has_mock_detection": True,
                "has_real_inference_check": True,
                "has_emoji_indicators": True,
                "dependency_checks": {
                    "torch": True,
                    "transformers": True,
                    "tokenizers": True,
                    "sentencepiece": True
                }
            },
            {
                "file": "test_hf_gpt2.py",
                "model": "hf_gpt2",
                "has_mock_detection": True,
                "has_real_inference_check": True,
                "has_emoji_indicators": False,
                "dependency_checks": {
                    "torch": True,
                    "transformers": True,
                    "tokenizers": False,
                    "sentencepiece": False
                }
            },
            {
                "file": "test_hf_t5.py",
                "model": "hf_t5",
                "has_mock_detection": False,
                "has_real_inference_check": False,
                "has_emoji_indicators": False,
                "dependency_checks": {
                    "torch": False,
                    "transformers": False,
                    "tokenizers": False,
                    "sentencepiece": False
                }
            }
        ]
        
        # Sample mock test results
        self.mock_results = [
            {
                "timestamp": "2025-07-21 10:00:00",
                "environment": {
                    "python_version": "3.9.5",
                    "platform": "linux",
                    "cpu_only": True
                },
                "test_results": [
                    {
                        "test_name": "test_bert_base_uncased",
                        "model_id": "bert-base-uncased",
                        "success": True,
                        "using_mocks": True,
                        "duration_ms": 250,
                        "error": None,
                        "dependencies": {
                            "transformers": False,
                            "torch": True,
                            "tokenizers": True,
                            "sentencepiece": True
                        }
                    },
                    {
                        "test_name": "test_gpt2",
                        "model_id": "gpt2",
                        "success": True,
                        "using_mocks": True,
                        "duration_ms": 200,
                        "error": None,
                        "dependencies": {
                            "transformers": False,
                            "torch": True,
                            "tokenizers": True,
                            "sentencepiece": True
                        }
                    }
                ]
            },
            {
                "timestamp": "2025-07-21 11:00:00",
                "environment": {
                    "python_version": "3.9.5",
                    "platform": "linux",
                    "cpu_only": False
                },
                "test_results": [
                    {
                        "test_name": "test_bert_base_uncased",
                        "model_id": "bert-base-uncased",
                        "success": True,
                        "using_mocks": False,
                        "duration_ms": 1200,
                        "error": None,
                        "dependencies": {
                            "transformers": True,
                            "torch": True,
                            "tokenizers": True,
                            "sentencepiece": True
                        }
                    },
                    {
                        "test_name": "test_gpt2",
                        "model_id": "gpt2",
                        "success": False,
                        "using_mocks": False,
                        "duration_ms": 1500,
                        "error": "CUDA out of memory",
                        "dependencies": {
                            "transformers": True,
                            "torch": True,
                            "tokenizers": True,
                            "sentencepiece": True
                        }
                    }
                ]
            }
        ]
        
        # Save sample mock results to temp directory
        self.mock_results_file = os.path.join(self.temp_dir, "mock_detection_sample.json")
        with open(self.mock_results_file, 'w') as f:
            json.dump(self.mock_results[0], f)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_scan_test_files(self):
        """Test scanning test files for mock detection."""
        # Create sample test files in temp directory
        sample_file_content = """
        # Define flags for dependency detection
        HAS_TORCH = True
        HAS_TRANSFORMERS = True
        
        # Check for mock status
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference
        
        print("🚀 Using REAL INFERENCE with actual models")
        """
        
        sample_file_path = os.path.join(self.temp_dir, "test_sample.py")
        with open(sample_file_path, 'w') as f:
            f.write(sample_file_content)
        
        # Scan the temp directory
        results = scan_test_files_for_mock_status(self.temp_dir)
        
        # Verify results
        self.assertGreaterEqual(len(results), 1, "Should find at least one test file")
        self.assertEqual(results[0]["file"], "test_sample.py")
        self.assertTrue(results[0]["has_mock_detection"])
        self.assertTrue(results[0]["has_real_inference_check"])
        self.assertTrue(results[0]["has_emoji_indicators"])
    
    def test_create_dependency_heatmap(self):
        """Test creating dependency heatmap."""
        output_file = os.path.join(self.temp_dir, "heatmap.html")
        fig = create_dependency_heatmap(self.scan_results, output_file)
        
        # Verify output file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)  # Should be a non-empty HTML file
    
    def test_create_mock_detection_summary(self):
        """Test creating mock detection summary."""
        output_file = os.path.join(self.temp_dir, "summary.html")
        fig = create_mock_detection_summary(self.scan_results, output_file)
        
        # Verify output file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)  # Should be a non-empty HTML file
    
    def test_create_model_family_analysis(self):
        """Test creating model family analysis."""
        output_file = os.path.join(self.temp_dir, "family.html")
        fig = create_model_family_analysis(self.scan_results, output_file)
        
        # Verify output file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)  # Should be a non-empty HTML file
    
    def test_create_test_result_analysis(self):
        """Test creating test result analysis."""
        output_file = os.path.join(self.temp_dir, "test_results.html")
        fig = create_test_result_analysis(self.mock_results, output_file)
        
        # Verify output file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)  # Should be a non-empty HTML file
    
    def test_create_interactive_dashboard(self):
        """Test creating interactive dashboard."""
        output_file = os.path.join(self.temp_dir, "dashboard.html")
        fig = create_interactive_dashboard(self.scan_results, self.mock_results, output_file)
        
        # Verify output file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)  # Should be a non-empty HTML file
    
    def test_generate_report(self):
        """Test generating report."""
        output_file = os.path.join(self.temp_dir, "report.md")
        report = generate_report(self.scan_results, self.mock_results, output_file)
        
        # Verify output file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 100)  # Should be a non-empty markdown file
        
        # Verify report content
        self.assertIn("# Mock Detection Implementation Report", report)
        self.assertIn("## Summary", report)
        self.assertIn("## Implementation by Model Family", report)

if __name__ == "__main__":
    unittest.main()