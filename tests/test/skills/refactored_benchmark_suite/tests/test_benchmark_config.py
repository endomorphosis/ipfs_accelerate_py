#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for benchmark configuration.

This script tests the BenchmarkConfig class functionality.
"""

import os
import sys
import unittest
from pathlib import Path
import logging

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from benchmark import BenchmarkConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TestBenchmarkConfig(unittest.TestCase):
    """Test cases for BenchmarkConfig."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        # Create a simple configuration
        config = BenchmarkConfig(
            model_id="bert-base-uncased",
            batch_sizes=[1, 2, 4],
            sequence_lengths=[16, 32],
            hardware=["cpu"],
            metrics=["latency", "throughput"]
        )
        
        # Verify configuration
        self.assertEqual(config.model_id, "bert-base-uncased")
        self.assertEqual(config.batch_sizes, [1, 2, 4])
        self.assertEqual(config.sequence_lengths, [16, 32])
        self.assertEqual(config.hardware, ["cpu"])
        self.assertEqual(config.metrics, ["latency", "throughput"])
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with invalid hardware
        config = BenchmarkConfig(
            model_id="bert-base-uncased",
            hardware=["invalid_hardware"]
        )
        
        # Should fallback to CPU
        self.assertEqual(config.hardware, ["cpu"])
        
        # Test with invalid metrics
        config = BenchmarkConfig(
            model_id="bert-base-uncased",
            metrics=["invalid_metric"]
        )
        
        # Should fallback to latency
        self.assertEqual(config.metrics, ["latency"])
    
    def test_config_dictionary_conversion(self):
        """Test configuration to dictionary conversion."""
        # Create a configuration
        config = BenchmarkConfig(
            model_id="bert-base-uncased",
            batch_sizes=[1, 2],
            sequence_lengths=[16],
            hardware=["cpu"],
            metrics=["latency"]
        )
        
        # Convert to dictionary
        config_dict = config.to_dict()
        
        # Verify dictionary
        self.assertEqual(config_dict["model_id"], "bert-base-uncased")
        self.assertEqual(config_dict["batch_sizes"], [1, 2])
        self.assertEqual(config_dict["sequence_lengths"], [16])
        self.assertEqual(config_dict["hardware"], ["cpu"])
        self.assertEqual(config_dict["metrics"], ["latency"])
    
    def test_config_from_dictionary(self):
        """Test configuration from dictionary."""
        # Create a dictionary
        config_dict = {
            "model_id": "bert-base-uncased",
            "batch_sizes": [1, 2],
            "sequence_lengths": [16],
            "hardware": ["cpu"],
            "metrics": ["latency"]
        }
        
        # Create configuration from dictionary
        config = BenchmarkConfig.from_dict(config_dict)
        
        # Verify configuration
        self.assertEqual(config.model_id, "bert-base-uncased")
        self.assertEqual(config.batch_sizes, [1, 2])
        self.assertEqual(config.sequence_lengths, [16])
        self.assertEqual(config.hardware, ["cpu"])
        self.assertEqual(config.metrics, ["latency"])
    
    def test_config_json_serialization(self):
        """Test configuration JSON serialization."""
        # Create a configuration
        config = BenchmarkConfig(
            model_id="bert-base-uncased",
            batch_sizes=[1, 2],
            sequence_lengths=[16],
            hardware=["cpu"],
            metrics=["latency"]
        )
        
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        
        try:
            # Save to JSON
            config.to_json(json_path)
            
            # Load from JSON
            loaded_config = BenchmarkConfig.from_json(json_path)
            
            # Verify configuration
            self.assertEqual(loaded_config.model_id, config.model_id)
            self.assertEqual(loaded_config.batch_sizes, config.batch_sizes)
            self.assertEqual(loaded_config.sequence_lengths, config.sequence_lengths)
            self.assertEqual(loaded_config.hardware, config.hardware)
            self.assertEqual(loaded_config.metrics, config.metrics)
            
        finally:
            # Clean up
            os.unlink(json_path)

if __name__ == "__main__":
    unittest.main()