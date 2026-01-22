#!/usr/bin/env python3
"""
Unit Tests for the Advanced Visualization Module.

This module tests the functionality of the advanced visualization capabilities
in the Predictive Performance System, including 3D visualizations, interactive 
dashboards, and time-series performance tracking.
"""

import os
import sys
import json
import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Import visualization module
from predictive_performance.visualization import AdvancedVisualization, create_visualization_report

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
        
        # Generate timestamps for time-series data (past 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        timestamps = [start_date + timedelta(days=i) for i in range(8)]
        
        # Generate data points
        for model_name, model_category in zip(models, model_categories):
            for hw in hardware:
                for batch_size in batch_sizes:
                    base_throughput = 100.0 * (0.5 + np.random.random())
                    base_latency = 10.0 * (0.5 + np.random.random())
                    base_memory = 1000.0 * (0.5 + np.random.random())
                    base_power = 50.0 * (0.5 + np.random.random())
                    
                    # Hardware factors
                    if hw == "cuda":
                        hw_factor = 5.0
                        power_factor = 3.0
                    elif hw == "webgpu":
                        hw_factor = 3.0
                        power_factor = 2.0
                    else:
                        hw_factor = 1.0
                        power_factor = 1.0
                    
                    # Batch size factors
                    batch_factor = np.sqrt(batch_size)
                    
                    # Calculate metrics
                    throughput = base_throughput * hw_factor * batch_factor * (1.0 + np.random.normal(0, 0.1))
                    latency = base_latency / hw_factor * (1.0 + 0.1 * batch_size) * (1.0 + np.random.normal(0, 0.1))
                    memory = base_memory * (1.0 + 0.2 * (batch_size - 1)) * (1.0 + np.random.normal(0, 0.05))
                    power = base_power * power_factor * (1.0 + 0.1 * batch_size) * (1.0 + np.random.normal(0, 0.1))
                    
                    # Add confidence and bounds
                    confidence = 0.85 + np.random.random() * 0.15
                    
                    # Calculate bounds
                    throughput_lower = throughput * (1.0 - (1.0 - confidence) * 2)
                    throughput_upper = throughput * (1.0 + (1.0 - confidence) * 2)
                    
                    # For time-series data
                    for timestamp in timestamps:
                        # Add time trend
                        time_position = timestamps.index(timestamp) / len(timestamps)
                        time_factor = 1.0 + 0.2 * np.sin(time_position * 2 * np.pi)
                        
                        data.append({
                            "model_name": model_name,
                            "model_category": model_category,
                            "hardware": hw,
                            "batch_size": batch_size,
                            "throughput": throughput * time_factor,
                            "latency_mean": latency * time_factor,
                            "memory_usage": memory * time_factor,
                            "power_consumption": power * time_factor,
                            "timestamp": timestamp.isoformat(),
                            "confidence": confidence,
                            "throughput_lower_bound": throughput_lower * time_factor,
                            "throughput_upper_bound": throughput_upper * time_factor,
                            "latency_lower_bound": latency / time_factor * 0.9,
                            "latency_upper_bound": latency / time_factor * 1.1,
                            "memory_lower_bound": memory * time_factor * 0.9,
                            "memory_upper_bound": memory * time_factor * 1.1
                        })
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test initialization of visualization object."""
        self.assertEqual(self.vis.output_dir, str(self.output_dir))
        self.assertEqual(self.vis.interactive, False)
        self.assertEqual(self.vis.output_format, "png")
    
    def test_prepare_data(self):
        """Test data preparation functionality."""
        # Test with DataFrame
        df = self.vis._prepare_data(self.sample_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.sample_data))
        
        # Test with dict
        data_dict = self.sample_data.to_dict('records')
        df_from_dict = self.vis._prepare_data(data_dict)
        self.assertIsInstance(df_from_dict, pd.DataFrame)
        
        # Test with JSON file
        json_path = self.output_dir / "test_data.json"
        with open(json_path, 'w') as f:
            json.dump(data_dict, f)
        df_from_json = self.vis._prepare_data(str(json_path))
        self.assertIsInstance(df_from_json, pd.DataFrame)
        
        # Test with CSV file
        csv_path = self.output_dir / "test_data.csv"
        self.sample_data.to_csv(csv_path, index=False)
        df_from_csv = self.vis._prepare_data(str(csv_path))
        self.assertIsInstance(df_from_csv, pd.DataFrame)
    
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
        self.assertTrue(output_file.endswith(".png"))
    
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
        self.assertTrue(output_file.endswith(".png"))
    
    def test_time_series_visualization(self):
        """Test time series visualization creation."""
        output_file = self.vis.create_time_series_visualization(
            self.sample_data,
            time_column="timestamp",
            metric="throughput",
            groupby=["model_name", "hardware"],
            title="Test Time Series"
        )
        
        # Check that output file exists
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(output_file.endswith(".png"))
    
    def test_power_efficiency_visualization(self):
        """Test power efficiency visualization creation."""
        output_file = self.vis.create_power_efficiency_visualization(
            self.sample_data,
            performance_metric="throughput",
            power_metric="power_consumption",
            groupby=["model_category"],
            title="Test Power Efficiency"
        )
        
        # Check that output file exists
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(output_file.endswith(".png"))
    
    def test_dimension_reduction_visualization(self):
        """Test dimension reduction visualization creation."""
        output_file = self.vis.create_dimension_reduction_visualization(
            self.sample_data,
            features=["batch_size", "memory_usage", "power_consumption", "latency_mean"],
            target="throughput",
            method="pca",
            n_components=2,
            groupby="model_category",
            title="Test Dimension Reduction"
        )
        
        # Check that output file exists
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(output_file.endswith(".png"))
    
    def test_prediction_confidence_visualization(self):
        """Test prediction confidence visualization creation."""
        output_file = self.vis.create_prediction_confidence_visualization(
            self.sample_data,
            metric="throughput",
            confidence_column="confidence",
            groupby=["model_category", "hardware"],
            title="Test Prediction Confidence"
        )
        
        # Check that output file exists
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(output_file.endswith(".png"))
    
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
            include_3d=True,
            include_time_series=True,
            include_power_efficiency=True,
            include_dimension_reduction=True,
            include_confidence=True
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
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn("Test Visualization Report", content)
            self.assertIn("visualization-grid", content)

if __name__ == "__main__":
    unittest.main()