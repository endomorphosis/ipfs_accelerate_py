#!/usr/bin/env python3
"""
Direct Test for the Advanced Visualization Module.

This script tests the visualization_minimal.py file directly without going through the package import.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define path to the visualization_minimal module
vis_module_path = os.path.join(script_dir, "predictive_performance", "visualization_minimal.py")

# Create namespace for imports
namespace = {}

# Execute the visualization_minimal.py file directly
with open(vis_module_path, 'r') as f:
    exec(f.read(), namespace)

# Extract classes from namespace
AdvancedVisualization = namespace['AdvancedVisualization']
create_visualization_report = namespace['create_visualization_report']

class TestAdvancedVisualization(unittest.TestCase):
    """Test cases for the AdvancedVisualization class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.test_dir.name)
        
        # Create visualization instance
        self.vis = AdvancedVisualization(
            output_dir=str(self.output_dir),
            interactive=False  # Use static visualizations for testing
        )
        
        # Generate sample data
        self.sample_data = self._generate_sample_data()
    
    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()
    
    def _generate_sample_data(self):
        """Generate sample data for testing visualizations."""
        # Create test data
        data = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define model and hardware options
        models = ["bert-base", "t5-small", "vit-base"]
        model_categories = ["text_embedding", "text_generation", "vision"]
        hardware = ["cpu", "cuda", "webgpu"]
        batch_sizes = [1, 8, 32]
        
        # Generate data points
        for model_name, model_category in zip(models, model_categories):
            for hw in hardware:
                for batch_size in batch_sizes:
                    # Generate metrics
                    throughput = 100.0 * (1.0 + np.random.random())
                    latency = 10.0 * (1.0 + np.random.random())
                    memory = 1000.0 * (1.0 + np.random.random())
                    power = 50.0 * (1.0 + np.random.random())
                    
                    # Add data point
                    data.append({
                        "model_name": model_name,
                        "model_category": model_category,
                        "hardware": hw,
                        "batch_size": batch_size,
                        "throughput": throughput,
                        "latency_mean": latency,
                        "memory_usage": memory,
                        "power_consumption": power,
                        "timestamp": datetime.now().isoformat(),
                        "confidence": 0.9
                    })
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test initialization of visualization object."""
        self.assertEqual(str(self.vis.output_dir), str(self.output_dir))
        self.assertEqual(self.vis.interactive, False)
    
    def test_prepare_data(self):
        """Test data preparation functionality."""
        # Test with DataFrame
        df = self.vis._prepare_data(self.sample_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.sample_data))
    
    def test_3d_visualization(self):
        """Test 3D visualization creation."""
        output_file = self.vis.create_3d_visualization(
            self.sample_data,
            x_metric="batch_size",
            y_metric="throughput",
            z_metric="memory_usage",
            color_metric="hardware",
            title="Test 3D Visualization"
        )
        
        # Check that output file exists
        self.assertTrue(os.path.exists(output_file))
    
    def test_batch_visualizations(self):
        """Test batch visualization creation."""
        visualization_files = self.vis.create_batch_visualizations(
            self.sample_data,
            metrics=["throughput", "latency_mean"],
            groupby=["model_category", "hardware"],
            include_3d=True,
            include_time_series=True,
            include_power_efficiency=True,
            include_dimension_reduction=True,
            include_confidence=True
        )
        
        # Check that visualization files dict is not empty
        self.assertIsInstance(visualization_files, dict)
        self.assertTrue(len(visualization_files) > 0)
        
        # Check that files exist
        for file_type, files in visualization_files.items():
            for file_path in files:
                self.assertTrue(os.path.exists(file_path))

if __name__ == "__main__":
    print("Running direct visualization tests...")
    unittest.main()