#!/usr/bin/env python3
"""
Test script for the Visualization components of the Simulation Accuracy and Validation Framework.

This script demonstrates how to use the ValidationVisualizer class to create
various visualizations of simulation validation results, including the enhanced
features like multi-format export, animated time series, and 3D visualizations.
"""

import os
import sys
import logging
import datetime
import json
import tempfile
import unittest
from typing import Dict, List, Any, Optional, Union, Tuple
from unittest.mock import patch, MagicMock
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_visualization")

# Import base classes and framework
# Fix relative imports when running script directly
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Use relative imports since we're inside the package
from core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)
from simulation_validation_framework import (
    SimulationValidationFramework,
    get_framework_instance
)
from visualization.validation_visualizer import ValidationVisualizer

def create_sample_validation_results():
    """Create a set of sample validation results for testing."""
    # Define hardware and model IDs
    hardware_ids = ["gpu_rtx3080", "gpu_a100", "cpu_intel_xeon", "webgpu_chrome"]
    model_ids = ["bert-base-uncased", "vit-base-patch16-224", "llama-7b", "whisper-base"]
    
    # Create validation results
    validation_results = []
    
    for hw_id in hardware_ids:
        for model_id in model_ids:
            # Create simulation result
            sim_result = SimulationResult(
                model_id=model_id,
                hardware_id=hw_id,
                metrics={
                    "throughput_items_per_second": 100.0 + (hash(hw_id + model_id) % 50),
                    "average_latency_ms": 10.0 + (hash(hw_id + model_id) % 20),
                    "memory_peak_mb": 1000.0 + (hash(hw_id + model_id) % 500),
                    "power_consumption_w": 50.0 + (hash(hw_id + model_id) % 30)
                },
                batch_size=16,
                precision="fp16",
                simulation_version="v1.0"
            )
            
            # Create hardware result with slightly different values
            hw_result = HardwareResult(
                model_id=model_id,
                hardware_id=hw_id,
                metrics={
                    "throughput_items_per_second": 100.0 + (hash(hw_id + model_id) % 50) * 1.1,
                    "average_latency_ms": 10.0 + (hash(hw_id + model_id) % 20) * 0.9,
                    "memory_peak_mb": 1000.0 + (hash(hw_id + model_id) % 500) * 1.05,
                    "power_consumption_w": 50.0 + (hash(hw_id + model_id) % 30) * 0.95
                },
                batch_size=16,
                precision="fp16",
                hardware_details={"type": hw_id.split("_")[0]},
                test_environment={"os": "linux", "driver_version": "123.45"}
            )
            
            # Create validation result with metrics comparison
            metrics_comparison = {}
            for metric, sim_value in sim_result.metrics.items():
                hw_value = hw_result.metrics[metric]
                abs_error = abs(sim_value - hw_value)
                rel_error = abs_error / hw_value
                mape = rel_error * 100.0
                
                metrics_comparison[metric] = {
                    "simulation_value": sim_value,
                    "hardware_value": hw_value,
                    "absolute_error": abs_error,
                    "relative_error": rel_error,
                    "mape": mape
                }
            
            validation_result = ValidationResult(
                simulation_result=sim_result,
                hardware_result=hw_result,
                metrics_comparison=metrics_comparison,
                validation_version="v1.0"
            )
            
            validation_results.append(validation_result)
    
    # Create some historical data for time series
    base_timestamp = datetime.datetime.now() - datetime.timedelta(days=30)
    for day in range(0, 30, 5):  # Every 5 days
        timestamp = base_timestamp + datetime.timedelta(days=day)
        
        # Just create for one hardware/model combo
        hw_id = "gpu_rtx3080"
        model_id = "bert-base-uncased"
        
        # Create with slightly different values over time
        factor = 1.0 + (day / 100)
        
        sim_result = SimulationResult(
            model_id=model_id,
            hardware_id=hw_id,
            metrics={
                "throughput_items_per_second": 100.0 * factor,
                "average_latency_ms": 10.0 * (2-factor),
                "memory_peak_mb": 1000.0 * factor,
                "power_consumption_w": 50.0 * factor
            },
            batch_size=16,
            precision="fp16",
            timestamp=timestamp.isoformat(),
            simulation_version="v1.0"
        )
        
        hw_result = HardwareResult(
            model_id=model_id,
            hardware_id=hw_id,
            metrics={
                "throughput_items_per_second": 110.0,
                "average_latency_ms": 9.0,
                "memory_peak_mb": 1050.0,
                "power_consumption_w": 48.0
            },
            batch_size=16,
            precision="fp16",
            timestamp=timestamp.isoformat(),
            hardware_details={"type": "gpu"},
            test_environment={"os": "linux", "driver_version": "123.45"}
        )
        
        metrics_comparison = {}
        for metric, sim_value in sim_result.metrics.items():
            hw_value = hw_result.metrics[metric]
            abs_error = abs(sim_value - hw_value)
            rel_error = abs_error / hw_value
            mape = rel_error * 100.0
            
            metrics_comparison[metric] = {
                "simulation_value": sim_value,
                "hardware_value": hw_value,
                "absolute_error": abs_error,
                "relative_error": rel_error,
                "mape": mape
            }
        
        validation_result = ValidationResult(
            simulation_result=sim_result,
            hardware_result=hw_result,
            metrics_comparison=metrics_comparison,
            validation_timestamp=timestamp.isoformat(),
            validation_version="v1.0"
        )
        
        validation_results.append(validation_result)
    
    return validation_results

def create_sample_drift_results():
    """Create sample drift detection results for testing."""
    return {
        "status": "success",
        "is_significant": True,
        "hardware_type": "gpu_rtx3080",
        "model_type": "bert-base-uncased",
        "historical_window_start": (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat(),
        "historical_window_end": (datetime.datetime.now() - datetime.timedelta(days=15)).isoformat(),
        "new_window_start": (datetime.datetime.now() - datetime.timedelta(days=15)).isoformat(),
        "new_window_end": datetime.datetime.now().isoformat(),
        "drift_metrics": {
            "throughput_items_per_second": {
                "p_value": 0.023,
                "drift_detected": True,
                "mean_change_pct": 12.5,
                "distribution_change": "significant"
            },
            "average_latency_ms": {
                "p_value": 0.102,
                "drift_detected": False,
                "mean_change_pct": 5.2,
                "distribution_change": "minor"
            },
            "memory_peak_mb": {
                "p_value": 0.008,
                "drift_detected": True,
                "mean_change_pct": 8.7,
                "distribution_change": "significant"
            },
            "power_consumption_w": {
                "p_value": 0.67,
                "drift_detected": False,
                "mean_change_pct": 2.1,
                "distribution_change": "negligible"
            }
        },
        "correlation_changes": {
            "throughput_latency": {
                "historical_correlation": -0.78,
                "new_correlation": -0.65,
                "is_significant": True
            },
            "throughput_memory": {
                "historical_correlation": 0.45,
                "new_correlation": 0.52,
                "is_significant": False
            }
        },
        "thresholds_used": {
            "p_value": 0.05,
            "mean_change_pct": 10.0,
            "significant_correlation_change": 0.1
        }
    }

def test_visualizer_directly():
    """Test the ValidationVisualizer class directly."""
    logger.info("Running direct visualizer tests")
    
    # Create sample data
    validation_results = create_sample_validation_results()
    drift_results = create_sample_drift_results()
    
    # Create output directory
    output_dir = Path("./output/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a visualizer
    visualizer = ValidationVisualizer()
    
    # Test MAPE comparison chart
    logger.info("Creating MAPE comparison chart")
    mape_chart_path = visualizer.create_mape_comparison_chart(
        validation_results,
        metric_name="all",
        output_path=str(output_dir / "mape_comparison.html"),
        interactive=True
    )
    logger.info(f"MAPE comparison chart saved to: {mape_chart_path}")
    
    # Test hardware comparison heatmap
    logger.info("Creating hardware comparison heatmap")
    heatmap_path = visualizer.create_hardware_comparison_heatmap(
        validation_results,
        metric_name="throughput_items_per_second",
        output_path=str(output_dir / "hardware_heatmap.html"),
        interactive=True
    )
    logger.info(f"Hardware comparison heatmap saved to: {heatmap_path}")
    
    # Test drift detection visualization
    logger.info("Creating drift detection visualization")
    drift_viz_path = visualizer.create_drift_detection_visualization(
        drift_results,
        output_path=str(output_dir / "drift_detection.html"),
        interactive=True
    )
    logger.info(f"Drift detection visualization saved to: {drift_viz_path}")
    
    # Test time series chart
    logger.info("Creating time series chart")
    time_series_path = visualizer.create_time_series_chart(
        validation_results,
        metric_name="throughput_items_per_second",
        hardware_id="gpu_rtx3080",
        model_id="bert-base-uncased",
        show_trend=True,
        output_path=str(output_dir / "time_series.html"),
        interactive=True
    )
    logger.info(f"Time series chart saved to: {time_series_path}")
    
    # Test comprehensive dashboard
    logger.info("Creating comprehensive dashboard")
    dashboard_path = visualizer.create_comprehensive_dashboard(
        validation_results,
        output_path=str(output_dir / "dashboard.html")
    )
    logger.info(f"Comprehensive dashboard saved to: {dashboard_path}")
    
    logger.info("Direct visualizer tests completed")

def test_framework_visualization():
    """Test the visualization functionality through the framework."""
    logger.info("Running framework visualization tests")
    
    # Create sample data
    validation_results = create_sample_validation_results()
    drift_results = create_sample_drift_results()
    
    # Create output directory
    output_dir = Path("./output/framework_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get framework instance
    framework = get_framework_instance()
    
    # Test MAPE comparison chart
    logger.info("Creating MAPE comparison chart through framework")
    mape_chart_path = framework.visualize_mape_comparison(
        validation_results,
        metric_name="all",
        output_path=str(output_dir / "framework_mape_comparison.html"),
        interactive=True
    )
    logger.info(f"Framework MAPE comparison chart saved to: {mape_chart_path}")
    
    # Test hardware comparison heatmap
    logger.info("Creating hardware comparison heatmap through framework")
    heatmap_path = framework.visualize_hardware_comparison_heatmap(
        validation_results,
        metric_name="throughput_items_per_second",
        output_path=str(output_dir / "framework_hardware_heatmap.html"),
        interactive=True
    )
    logger.info(f"Framework hardware comparison heatmap saved to: {heatmap_path}")
    
    # Test drift detection visualization
    logger.info("Creating drift detection visualization through framework")
    drift_viz_path = framework.visualize_drift_detection(
        drift_results,
        output_path=str(output_dir / "framework_drift_detection.html"),
        interactive=True
    )
    logger.info(f"Framework drift detection visualization saved to: {drift_viz_path}")
    
    # Test comprehensive dashboard
    logger.info("Creating comprehensive dashboard through framework")
    dashboard_path = framework.create_comprehensive_dashboard(
        validation_results,
        output_path=str(output_dir / "framework_dashboard.html")
    )
    logger.info(f"Framework comprehensive dashboard saved to: {dashboard_path}")
    
    logger.info("Framework visualization tests completed")

def main():
    """Run the visualization tests."""
    try:
        test_visualizer_directly()
        test_framework_visualization()
        logger.info("All visualization tests completed successfully")
    except Exception as e:
        logger.error(f"Error running visualization tests: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    
class TestVisualizationEnhancements(unittest.TestCase):
    """Test case for enhanced visualization features."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create visualizer with custom config
        self.visualizer = ValidationVisualizer(config={
            "output_directory": self.temp_dir.name,
            "export_formats": ["html", "png"],
            "animated_transitions": True,
            "animation_duration": 300
        })
        
        # Create sample validation results for testing
        self.validation_results = []
        
        # Create results for multiple timestamps
        for i in range(10):
            # Create timestamp with 1-day increments
            timestamp = (datetime.datetime(2025, 3, 1) + datetime.timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S")
            
            # Create simulation result
            sim_result = SimulationResult(
                model_id="bert-base-uncased",
                hardware_id="gpu_rtx3080",
                batch_size=32,
                precision="fp16",
                timestamp=timestamp,
                simulation_version="sim_v1.0",
                metrics={
                    "throughput_items_per_second": 90.0 + i * 2,
                    "average_latency_ms": 17.0 - i * 0.2,
                    "peak_memory_mb": 2000 + i * 10
                },
                simulation_params={
                    "model_params": {"hidden_size": 768, "num_layers": 12},
                    "hardware_params": {"gpu_compute_capability": "8.6", "gpu_memory": 10240}
                }
            )
            
            # Create hardware result
            hw_result = HardwareResult(
                model_id="bert-base-uncased",
                hardware_id="gpu_rtx3080",
                batch_size=32,
                precision="fp16",
                timestamp=timestamp,
                metrics={
                    "throughput_items_per_second": 95.0 + i * 1.5,
                    "average_latency_ms": 16.0 - i * 0.15,
                    "peak_memory_mb": 2200 + i * 8
                },
                hardware_details={
                    "name": "NVIDIA RTX 3080",
                    "compute_capability": "8.6",
                    "vram_gb": 10
                },
                test_environment={
                    "os": "Linux",
                    "cuda_version": "11.4",
                    "driver_version": "470.82.01"
                }
            )
            
            # Create validation result
            metrics_comparison = {
                "throughput_items_per_second": {
                    "simulation_value": sim_result.metrics["throughput_items_per_second"],
                    "hardware_value": hw_result.metrics["throughput_items_per_second"],
                    "error": hw_result.metrics["throughput_items_per_second"] - sim_result.metrics["throughput_items_per_second"],
                    "mape": abs(hw_result.metrics["throughput_items_per_second"] - sim_result.metrics["throughput_items_per_second"]) / hw_result.metrics["throughput_items_per_second"] * 100
                },
                "average_latency_ms": {
                    "simulation_value": sim_result.metrics["average_latency_ms"],
                    "hardware_value": hw_result.metrics["average_latency_ms"],
                    "error": hw_result.metrics["average_latency_ms"] - sim_result.metrics["average_latency_ms"],
                    "mape": abs(hw_result.metrics["average_latency_ms"] - sim_result.metrics["average_latency_ms"]) / hw_result.metrics["average_latency_ms"] * 100
                },
                "peak_memory_mb": {
                    "simulation_value": sim_result.metrics["peak_memory_mb"],
                    "hardware_value": hw_result.metrics["peak_memory_mb"],
                    "error": hw_result.metrics["peak_memory_mb"] - sim_result.metrics["peak_memory_mb"],
                    "mape": abs(hw_result.metrics["peak_memory_mb"] - sim_result.metrics["peak_memory_mb"]) / hw_result.metrics["peak_memory_mb"] * 100
                }
            }
            
            validation_result = ValidationResult(
                simulation_result=sim_result,
                hardware_result=hw_result,
                metrics_comparison=metrics_comparison,
                validation_timestamp=timestamp,
                validation_version="v1.0",
                overall_accuracy_score=5.0 - i * 0.2  # Improving accuracy over time
            )
            
            self.validation_results.append(validation_result)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('plotly.graph_objects.Figure.write_html')
    @patch('plotly.graph_objects.Figure.write_image')
    def test_export_visualization(self, mock_write_image, mock_write_html):
        """Test exporting a visualization to multiple formats."""
        # Mock plotly figure
        mock_fig = MagicMock()
        mock_fig.write_html = mock_write_html
        mock_fig.write_image = mock_write_image
        
        # Make the figure pass the hasattr check for write_html
        type(mock_fig).write_html = mock_write_html
        
        # Test exporting to multiple formats
        output_path = os.path.join(self.temp_dir.name, "test_export")
        formats = ["html", "png", "pdf"]
        
        result = self.visualizer.export_visualization(mock_fig, output_path, formats)
        
        # Verify the export calls
        self.assertEqual(mock_write_html.call_count, 1)
        self.assertEqual(mock_write_image.call_count, 2)  # Once for PNG, once for PDF
        
        # Verify the result
        self.assertIn("html", result)
        self.assertIn("png", result)
        self.assertIn("pdf", result)
        self.assertEqual(result["html"], f"{output_path}.html")
        self.assertEqual(result["png"], f"{output_path}.png")
        self.assertEqual(result["pdf"], f"{output_path}.pdf")
    
    @patch('plotly.graph_objects.Figure')
    @patch('pandas.DataFrame')
    def test_create_animated_time_series(self, mock_df, mock_figure):
        """Test creating an animated time series visualization."""
        # Setup mock for pandas DataFrame
        mock_df_instance = MagicMock()
        mock_df.return_value = mock_df_instance
        mock_df_instance.iloc = MagicMock()
        
        # Setup mock for plotly figure
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        # Setup export visualization mock
        self.visualizer.export_visualization = MagicMock(return_value={"html": "test.html"})
        
        # Test creating animated time series
        result = self.visualizer.create_animated_time_series(
            validation_results=self.validation_results[:5],  # Use first 5 results
            metric_name="throughput_items_per_second",
            hardware_id="gpu_rtx3080",
            model_id="bert-base-uncased",
            output_path=os.path.join(self.temp_dir.name, "animated_time_series")
        )
        
        # Verify the export was called
        self.visualizer.export_visualization.assert_called_once()
        
        # Verify the result
        self.assertEqual(result, {"html": "test.html"})
    
    @patch('plotly.graph_objects.Figure')
    @patch('pandas.DataFrame')
    def test_create_3d_error_visualization(self, mock_df, mock_figure):
        """Test creating a 3D error visualization."""
        # Setup mock for pandas DataFrame
        mock_df_instance = MagicMock()
        mock_df.return_value = mock_df_instance
        
        # Setup properties for DataFrame mock
        mock_df_instance.__getitem__.return_value = MagicMock()
        mock_df_instance.iloc = MagicMock()
        
        # Setup mock for plotly figure
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        # Setup export visualization mock
        self.visualizer.export_visualization = MagicMock(return_value={"html": "test.html"})
        
        # Test creating 3D visualization
        result = self.visualizer.create_3d_error_visualization(
            validation_results=self.validation_results,
            metrics=["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"],
            hardware_ids=["gpu_rtx3080"],
            model_ids=["bert-base-uncased"],
            output_path=os.path.join(self.temp_dir.name, "3d_visualization")
        )
        
        # Verify the export was called
        self.visualizer.export_visualization.assert_called_once()
        
        # Verify the result
        self.assertEqual(result, {"html": "test.html"})


# Run the unit tests when script is executed directly
def run_unit_tests():
    unittest.main()