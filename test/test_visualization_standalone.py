#!/usr/bin/env python3
"""
Standalone Test for the Advanced Visualization Module.

This script tests the visualization.py file directly without relying on the module imports.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Add the directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the visualization module directly
from predictive_performance.visualization_minimal import AdvancedVisualization, create_visualization_report

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
    
    def test_performance_dashboard(self):
        """Test performance dashboard creation."""
        output_file = self.vis.create_performance_dashboard(
            self.sample_data,
            metrics=["throughput", "latency_mean"],
            groupby=["model_category", "hardware"],
            title="Test Performance Dashboard"
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
    
    def test_visualization_report(self):
        """Test visualization report creation."""
        # Generate visualizations
        visualization_files = self.vis.create_batch_visualizations(
            self.sample_data,
            metrics=["throughput"],
            groupby=["model_category"],
            include_3d=True
        )
        
        # Create report
        report_path = create_visualization_report(
            visualization_files=visualization_files,
            title="Test Visualization Report",
            output_file="test_report.html",
            output_dir=str(self.output_dir)
        )
        
        # Check that report exists
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith(".html"))

if __name__ == "__main__":
    print("Running standalone visualization tests...")
    unittest.main()