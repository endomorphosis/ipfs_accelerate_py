#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for configuration file loading.

This script tests the loading of benchmark configurations from YAML and JSON files.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
import logging

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.benchmark_config import load_config_from_file, save_config_to_file, create_benchmark_configs_from_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TestConfigLoading(unittest.TestCase):
    """Test cases for configuration file loading."""
    
    def test_yaml_config_loading(self):
        """Test loading a YAML configuration file."""
        # Create a temporary YAML config file
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+", delete=False) as f:
            f.write("""
# Test configuration
models:
  - id: bert-base-uncased
    task: fill-mask
    batch_sizes: [1, 2]
  
  - id: gpt2
    task: text-generation
    batch_sizes: [1]

hardware:
  - cpu
  - cuda

metrics:
  - latency
  - memory
            """)
            yaml_path = f.name
        
        try:
            # Load configuration
            config = load_config_from_file(yaml_path)
            
            # Verify configuration
            self.assertIn("models", config)
            self.assertEqual(len(config["models"]), 2)
            self.assertEqual(config["models"][0]["id"], "bert-base-uncased")
            self.assertEqual(config["models"][1]["id"], "gpt2")
            self.assertIn("hardware", config)
            self.assertIn("metrics", config)
            
            # Create benchmark configs
            benchmark_configs = create_benchmark_configs_from_file(yaml_path)
            
            # Verify benchmark configs
            self.assertEqual(len(benchmark_configs), 2)
            self.assertEqual(benchmark_configs[0]["model_id"], "bert-base-uncased")
            self.assertEqual(benchmark_configs[1]["model_id"], "gpt2")
            
            logger.info("YAML config loading test passed")
            
        finally:
            # Clean up
            os.unlink(yaml_path)
    
    def test_json_config_loading(self):
        """Test loading a JSON configuration file."""
        # Create a temporary JSON config file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as f:
            f.write("""
{
    "models": [
        {
            "id": "bert-base-uncased",
            "task": "fill-mask",
            "batch_sizes": [1, 2]
        },
        {
            "id": "gpt2",
            "task": "text-generation",
            "batch_sizes": [1]
        }
    ],
    "hardware": ["cpu", "cuda"],
    "metrics": ["latency", "memory"]
}
            """)
            json_path = f.name
        
        try:
            # Load configuration
            config = load_config_from_file(json_path)
            
            # Verify configuration
            self.assertIn("models", config)
            self.assertEqual(len(config["models"]), 2)
            self.assertEqual(config["models"][0]["id"], "bert-base-uncased")
            self.assertEqual(config["models"][1]["id"], "gpt2")
            self.assertIn("hardware", config)
            self.assertIn("metrics", config)
            
            # Create benchmark configs
            benchmark_configs = create_benchmark_configs_from_file(json_path)
            
            # Verify benchmark configs
            self.assertEqual(len(benchmark_configs), 2)
            self.assertEqual(benchmark_configs[0]["model_id"], "bert-base-uncased")
            self.assertEqual(benchmark_configs[1]["model_id"], "gpt2")
            
            logger.info("JSON config loading test passed")
            
        finally:
            # Clean up
            os.unlink(json_path)
    
    def test_env_var_processing(self):
        """Test environment variable processing in configuration files."""
        # Set environment variable
        os.environ["TEST_TOKEN"] = "test_token_value"
        
        # Create a temporary YAML config file with environment variable
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+", delete=False) as f:
            f.write("""
# Test configuration with environment variable
model_id: bert-base-uncased
token: ${TEST_TOKEN}
            """)
            yaml_path = f.name
        
        try:
            # Load configuration
            config = load_config_from_file(yaml_path)
            
            # Verify configuration
            self.assertEqual(config["model_id"], "bert-base-uncased")
            self.assertEqual(config["token"], "test_token_value")
            
            logger.info("Environment variable processing test passed")
            
        finally:
            # Clean up
            os.unlink(yaml_path)
            # Remove environment variable
            del os.environ["TEST_TOKEN"]
    
    def test_save_and_load_config(self):
        """Test saving and loading a configuration file."""
        # Create a test configuration
        test_config = {
            "model_id": "bert-base-uncased",
            "batch_sizes": [1, 2, 4],
            "hardware": ["cpu", "cuda"],
            "metrics": ["latency", "memory"]
        }
        
        # Create temporary file paths
        yaml_path = tempfile.mktemp(suffix=".yaml")
        json_path = tempfile.mktemp(suffix=".json")
        
        try:
            # Save to YAML
            success = save_config_to_file(test_config, yaml_path)
            self.assertTrue(success)
            
            # Save to JSON
            success = save_config_to_file(test_config, json_path)
            self.assertTrue(success)
            
            # Load from YAML
            yaml_config = load_config_from_file(yaml_path)
            self.assertEqual(yaml_config["model_id"], test_config["model_id"])
            self.assertEqual(yaml_config["batch_sizes"], test_config["batch_sizes"])
            
            # Load from JSON
            json_config = load_config_from_file(json_path)
            self.assertEqual(json_config["model_id"], test_config["model_id"])
            self.assertEqual(json_config["batch_sizes"], test_config["batch_sizes"])
            
            logger.info("Config save and load test passed")
            
        finally:
            # Clean up
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

if __name__ == "__main__":
    unittest.main()