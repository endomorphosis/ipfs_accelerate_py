#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the visualization components of the benchmark suite.

This module contains tests that validate the visualization capabilities
of the benchmark suite, particularly the dashboard and plot generation.
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import ModelBenchmark, BenchmarkConfig, BenchmarkResult, BenchmarkResults
from visualizers.plots import plot_latency_comparison, plot_throughput_scaling, plot_memory_usage, plot_flops_comparison
from visualizers.dashboard import generate_dashboard

class MockHardwareAwareMetricsResult:
    """Mock class to simulate benchmark results with hardware-aware metrics."""
    
    def __init__(self, model_id="bert-base-uncased", hardware="cpu", batch_size=1, sequence_length=16):
        self.config = BenchmarkConfig(model_id=model_id, output_dir="./test_output")
        self.results = []
        
        # Create results for different hardware platforms
        hw_platforms = [hardware] if hardware else ["cpu", "cuda"]
        
        for hw in hw_platforms:
            for bs in [batch_size] if batch_size else [1, 2, 4, 8]:
                # Create a result with hardware-aware metrics
                metrics = {
                    "latency_ms": 10.5 * bs,
                    "latency_p90_ms": 12.3 * bs,
                    "latency_p95_ms": 14.1 * bs,
                    "latency_p99_ms": 18.7 * bs,
                    "throughput_items_per_sec": 95.0 / bs,
                    "memory_usage_mb": 1250 + (50 * bs),
                    "memory_peak_mb": 1500 + (75 * bs),
                    "memory_allocated_end_mb": 1200 + (40 * bs),
                    "memory_reserved_end_mb": 1400 + (60 * bs),
                    "cpu_memory_end_mb": 500 + (20 * bs),
                    "flops": 12500000000 * bs,
                    "gflops": 12.5 * bs,
                    "detailed_flops": {
                        "attention": 5500000000 * bs,
                        "feedforward": 6000000000 * bs,
                        "other": 1000000000 * bs
                    }
                }
                
                result = BenchmarkResult(
                    hardware=hw,
                    batch_size=bs,
                    sequence_length=sequence_length,
                    metrics=metrics
                )
                
                self.results.append(result)

    def export_to_json(self, output_path=None):
        """Mock export to JSON."""
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, 
                                       f"benchmark_{self.config.model_id.replace('/', '__')}.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create exportable data
        data = {
            "model_id": self.config.model_id,
            "results": []
        }
        
        for result in self.results:
            result_data = {
                "hardware": result.hardware,
                "batch_size": result.batch_size,
                "sequence_length": result.sequence_length,
                "metrics": result.metrics
            }
            data["results"].append(result_data)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return output_path


class TestVisualizers(unittest.TestCase):
    """Tests for the visualization components."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a temp directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_plot_latency_comparison(self):
        """Test latency comparison plot generation."""
        # Create mock results
        mock_results = MockHardwareAwareMetricsResult(hardware=None, batch_size=None)
        
        # Generate plot
        plot_path = plot_latency_comparison(mock_results, output_path=os.path.join(self.output_dir, "latency_test.png"))
        
        # Verify plot was created
        self.assertIsNotNone(plot_path)
        self.assertTrue(os.path.exists(plot_path))
    
    def test_plot_throughput_scaling(self):
        """Test throughput scaling plot generation."""
        # Create mock results
        mock_results = MockHardwareAwareMetricsResult(hardware=None, batch_size=None)
        
        # Generate plot
        plot_path = plot_throughput_scaling(mock_results, output_path=os.path.join(self.output_dir, "throughput_test.png"))
        
        # Verify plot was created
        self.assertIsNotNone(plot_path)
        self.assertTrue(os.path.exists(plot_path))
    
    def test_plot_memory_usage(self):
        """Test memory usage plot generation with detailed breakdown."""
        # Create mock results
        mock_results = MockHardwareAwareMetricsResult(hardware=None, batch_size=None)
        
        # Generate plot
        plot_path = plot_memory_usage(mock_results, output_path=os.path.join(self.output_dir, "memory_test.png"), detailed=True)
        
        # Verify plot was created
        self.assertIsNotNone(plot_path)
        self.assertTrue(os.path.exists(plot_path))
    
    def test_plot_flops_comparison(self):
        """Test FLOPs comparison plot generation with detailed breakdown."""
        # Create mock results
        mock_results = MockHardwareAwareMetricsResult(hardware=None, batch_size=None)
        
        # Generate plot
        plot_path = plot_flops_comparison(mock_results, output_path=os.path.join(self.output_dir, "flops_test.png"), detailed=True)
        
        # Verify plot was created
        self.assertIsNotNone(plot_path)
        self.assertTrue(os.path.exists(plot_path))
    
    def test_dashboard_generation(self):
        """Test dashboard generation with hardware-aware metrics."""
        # Skip if required packages are not available
        try:
            import dash
            import plotly
            import pandas as pd
        except ImportError:
            self.skipTest("Dashboard dependencies not available")
        
        # Create mock results
        mock_results = [MockHardwareAwareMetricsResult(hardware=None, batch_size=None)]
        
        # Create JSON files for mock results
        for result in mock_results:
            result.export_to_json(os.path.join(self.output_dir, f"benchmark_{result.config.model_id}.json"))
        
        # Generate dashboard
        dashboard_path = generate_dashboard(mock_results, self.output_dir)
        
        # Verify dashboard was created
        self.assertIsNotNone(dashboard_path)
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Verify dashboard HTML contains hardware-aware metrics
        with open(dashboard_path, 'r') as f:
            dashboard_html = f.read()
            
            # Check for latency percentiles
            self.assertIn('latency_p90_ms', dashboard_html)
            self.assertIn('latency_p95_ms', dashboard_html)
            self.assertIn('latency_p99_ms', dashboard_html)
            
            # Check for memory breakdown
            self.assertIn('memory_peak_mb', dashboard_html)
            self.assertIn('memory_allocated_end_mb', dashboard_html)
            
            # Check for GFLOPs
            self.assertIn('gflops', dashboard_html)


if __name__ == "__main__":
    unittest.main()