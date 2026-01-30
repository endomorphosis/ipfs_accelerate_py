#!/usr/bin/env python3
"""
Test script for the ValidationVisualizerDBConnector in the Simulation Accuracy and Validation Framework.

This module tests the connection between the database integration and visualization components,
validating that visualizations can be generated directly from database queries.
"""

import os
import sys
import unittest
import json
import datetime
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the connector
from data.duckdb.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector

# Import the database integration
from data.duckdb.simulation_validation.db_integration import SimulationValidationDBIntegration

# Import the visualizer
from data.duckdb.simulation_validation.visualization.validation_visualizer import ValidationVisualizer

# Import base classes
from data.duckdb.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)


class TestValidationVisualizerDBConnector(unittest.TestCase):
    """Test cases for ValidationVisualizerDBConnector."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create mock objects for dependencies
        self.mock_db_integration = MagicMock(spec=SimulationValidationDBIntegration)
        self.mock_visualizer = MagicMock(spec=ValidationVisualizer)
        
        # Initialize the connector with mock dependencies
        self.connector = ValidationVisualizerDBConnector(
            db_integration=self.mock_db_integration,
            visualizer=self.mock_visualizer
        )
        
        # Create sample validation results data
        self.sample_db_results = [
            {
                "validation_id": "val_1",
                "simulation_id": "sim_1",
                "hardware_id": "hw_1",
                "validation_timestamp": "2025-03-01T12:00:00",
                "validation_version": "v1.0",
                "metrics_comparison": json.dumps({
                    "throughput_items_per_second": {
                        "simulation_value": 100.0,
                        "hardware_value": 90.0,
                        "absolute_error": 10.0,
                        "relative_error": 0.111,
                        "mape": 11.1
                    },
                    "average_latency_ms": {
                        "simulation_value": 15.0,
                        "hardware_value": 17.0,
                        "absolute_error": 2.0,
                        "relative_error": 0.118,
                        "mape": 11.8
                    }
                }),
                "additional_metrics": "{}",
                "overall_accuracy_score": 11.45,
                "throughput_mape": 11.1,
                "latency_mape": 11.8,
                "memory_mape": None,
                "power_mape": None,
                "model_id": "bert-base-uncased",
                "hardware_type": "gpu_rtx3080",
                "batch_size": 32,
                "precision": "fp16",
                "simulation_version": "sim_v1.0",
                "hardware_details": json.dumps({"name": "NVIDIA RTX 3080", "compute_capability": "8.6"}),
                "test_environment": json.dumps({"os": "Linux", "cuda_version": "11.4"})
            },
            {
                "validation_id": "val_2",
                "simulation_id": "sim_2",
                "hardware_id": "hw_2",
                "validation_timestamp": "2025-03-02T12:00:00",
                "validation_version": "v1.0",
                "metrics_comparison": json.dumps({
                    "throughput_items_per_second": {
                        "simulation_value": 50.0,
                        "hardware_value": 45.0,
                        "absolute_error": 5.0,
                        "relative_error": 0.111,
                        "mape": 11.1
                    },
                    "average_latency_ms": {
                        "simulation_value": 25.0,
                        "hardware_value": 28.0,
                        "absolute_error": 3.0,
                        "relative_error": 0.107,
                        "mape": 10.7
                    }
                }),
                "additional_metrics": "{}",
                "overall_accuracy_score": 10.9,
                "throughput_mape": 11.1,
                "latency_mape": 10.7,
                "memory_mape": None,
                "power_mape": None,
                "model_id": "bert-base-uncased",
                "hardware_type": "cpu_intel_xeon",
                "batch_size": 32,
                "precision": "fp32",
                "simulation_version": "sim_v1.0",
                "hardware_details": json.dumps({"name": "Intel Xeon", "cores": 24}),
                "test_environment": json.dumps({"os": "Linux"})
            }
        ]
        
        # Sample drift detection results
        self.sample_drift_results = [
            {
                "id": "drift_1",
                "timestamp": "2025-03-10T12:00:00",
                "hardware_type": "gpu_rtx3080",
                "model_type": "bert-base-uncased",
                "drift_metrics": {
                    "throughput_items_per_second": {
                        "p_value": 0.03,
                        "drift_detected": True,
                        "mean_change_pct": 15.5
                    },
                    "average_latency_ms": {
                        "p_value": 0.07,
                        "drift_detected": False,
                        "mean_change_pct": 5.2
                    }
                },
                "is_significant": True,
                "historical_window_start": "2025-02-01T00:00:00",
                "historical_window_end": "2025-02-28T23:59:59",
                "new_window_start": "2025-03-01T00:00:00",
                "new_window_end": "2025-03-10T12:00:00",
                "thresholds_used": {
                    "p_value": 0.05,
                    "mean_change_pct": 10.0
                }
            }
        ]
        
        # Sample calibration history
        self.sample_calibration_history = [
            {
                "id": "cal_1",
                "timestamp": "2025-03-15T12:00:00",
                "hardware_type": "gpu_rtx3080",
                "model_type": "bert-base-uncased",
                "previous_parameters": {
                    "correction_factors": {
                        "throughput_items_per_second": 1.0,
                        "average_latency_ms": 1.0
                    }
                },
                "updated_parameters": {
                    "correction_factors": {
                        "throughput_items_per_second": 1.1,
                        "average_latency_ms": 0.95
                    }
                },
                "validation_results_before": [
                    {
                        "simulation_result": {
                            "model_id": "bert-base-uncased",
                            "hardware_id": "gpu_rtx3080",
                            "batch_size": 32,
                            "precision": "fp16",
                            "timestamp": "2025-03-15T10:00:00",
                            "simulation_version": "sim_v1.0",
                            "metrics": {
                                "throughput_items_per_second": 100.0,
                                "average_latency_ms": 15.0
                            }
                        },
                        "hardware_result": {
                            "model_id": "bert-base-uncased",
                            "hardware_id": "gpu_rtx3080",
                            "batch_size": 32,
                            "precision": "fp16",
                            "timestamp": "2025-03-15T10:00:00",
                            "metrics": {
                                "throughput_items_per_second": 90.0,
                                "average_latency_ms": 17.0
                            }
                        },
                        "metrics_comparison": {
                            "throughput_items_per_second": {
                                "simulation_value": 100.0,
                                "hardware_value": 90.0,
                                "mape": 11.1
                            },
                            "average_latency_ms": {
                                "simulation_value": 15.0,
                                "hardware_value": 17.0,
                                "mape": 11.8
                            }
                        },
                        "validation_timestamp": "2025-03-15T10:00:00",
                        "validation_version": "v1.0"
                    }
                ],
                "validation_results_after": [
                    {
                        "simulation_result": {
                            "model_id": "bert-base-uncased",
                            "hardware_id": "gpu_rtx3080",
                            "batch_size": 32,
                            "precision": "fp16",
                            "timestamp": "2025-03-15T11:00:00",
                            "simulation_version": "sim_v1.0",
                            "metrics": {
                                "throughput_items_per_second": 99.0,
                                "average_latency_ms": 16.0
                            }
                        },
                        "hardware_result": {
                            "model_id": "bert-base-uncased",
                            "hardware_id": "gpu_rtx3080",
                            "batch_size": 32,
                            "precision": "fp16",
                            "timestamp": "2025-03-15T11:00:00",
                            "metrics": {
                                "throughput_items_per_second": 90.0,
                                "average_latency_ms": 17.0
                            }
                        },
                        "metrics_comparison": {
                            "throughput_items_per_second": {
                                "simulation_value": 99.0,
                                "hardware_value": 90.0,
                                "mape": 10.0
                            },
                            "average_latency_ms": {
                                "simulation_value": 16.0,
                                "hardware_value": 17.0,
                                "mape": 5.9
                            }
                        },
                        "validation_timestamp": "2025-03-15T11:00:00",
                        "validation_version": "v1.0"
                    }
                ],
                "improvement_metrics": {
                    "overall": {
                        "before_mape": 11.45,
                        "after_mape": 7.95,
                        "absolute_improvement": 3.5,
                        "relative_improvement_pct": 30.5
                    },
                    "throughput_items_per_second": {
                        "before_mape": 11.1,
                        "after_mape": 10.0,
                        "absolute_improvement": 1.1,
                        "relative_improvement_pct": 9.9
                    },
                    "average_latency_ms": {
                        "before_mape": 11.8,
                        "after_mape": 5.9,
                        "absolute_improvement": 5.9,
                        "relative_improvement_pct": 50.0
                    }
                },
                "calibration_version": "v1.0"
            }
        ]
        
        # Sample time series data
        self.sample_time_series_data = [
            {
                "time_period": "2025-03-01T00:00:00",
                "num_validations": 3,
                "avg_metric": 11.5,
                "min_metric": 10.2,
                "max_metric": 12.8,
                "stddev_metric": 1.3
            },
            {
                "time_period": "2025-03-02T00:00:00",
                "num_validations": 5,
                "avg_metric": 10.9,
                "min_metric": 9.5,
                "max_metric": 12.3,
                "stddev_metric": 1.1
            },
            {
                "time_period": "2025-03-03T00:00:00",
                "num_validations": 4,
                "avg_metric": 10.1,
                "min_metric": 8.7,
                "max_metric": 11.5,
                "stddev_metric": 1.2
            }
        ]
        
        # Sample simulation vs hardware data
        self.sample_sim_vs_hw_data = [
            {
                "model_id": "bert-base-uncased",
                "hardware_type": "gpu_rtx3080",
                "batch_size": 32,
                "precision": "fp16",
                "validation_timestamp": "2025-03-01T12:00:00",
                "simulation_value": 100.0,
                "hardware_value": 90.0,
                "mape": 11.1
            },
            {
                "model_id": "bert-base-uncased",
                "hardware_type": "cpu_intel_xeon",
                "batch_size": 32,
                "precision": "fp32",
                "validation_timestamp": "2025-03-02T12:00:00",
                "simulation_value": 50.0,
                "hardware_value": 45.0,
                "mape": 11.1
            }
        ]
        
        # Sample calibration effectiveness analysis
        self.sample_calibration_analysis = {
            "status": "success",
            "overall": {
                "mean_relative_improvement": 30.5,
                "mean_mape_before": 11.45,
                "mean_mape_after": 7.95
            },
            "by_hardware_model": {
                "gpu_rtx3080-bert-base-uncased": {
                    "count": 3,
                    "mean_relative_improvement": 30.5,
                    "mean_mape_before": 11.45,
                    "mean_mape_after": 7.95
                }
            },
            "calibration_events": 3,
            "hardware_model_combinations": 1
        }
        
        # Set up output directory for test files
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_init(self):
        """Test initialization of connector."""
        # Test with explicit dependencies
        connector = ValidationVisualizerDBConnector(
            db_integration=self.mock_db_integration,
            visualizer=self.mock_visualizer
        )
        self.assertEqual(connector.db_integration, self.mock_db_integration)
        self.assertEqual(connector.visualizer, self.mock_visualizer)
        
        # Test with default initialization
        with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.SimulationValidationDBIntegration') as mock_db_class:
            with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.ValidationVisualizer') as mock_vis_class:
                mock_db_instance = MagicMock()
                mock_vis_instance = MagicMock()
                mock_db_class.return_value = mock_db_instance
                mock_vis_class.return_value = mock_vis_instance
                
                connector = ValidationVisualizerDBConnector()
                
                mock_db_class.assert_called_once_with(db_path="./benchmark_db.duckdb")
                mock_vis_class.assert_called_once_with(config=None)
                self.assertEqual(connector.db_integration, mock_db_instance)
                self.assertEqual(connector.visualizer, mock_vis_instance)
    
    def test_create_mape_comparison_chart_from_db(self):
        """Test creation of MAPE comparison chart from database."""
        # Set up mock return values
        self.mock_db_integration.get_validation_results.return_value = self.sample_db_results
        self.mock_visualizer.create_mape_comparison_chart.return_value = "/path/to/chart.html"
        
        # Call the method
        result = self.connector.create_mape_comparison_chart_from_db(
            hardware_ids=["gpu_rtx3080"],
            model_ids=["bert-base-uncased"],
            metric_name="throughput_items_per_second",
            output_path="/path/to/output.html"
        )
        
        # Check that the database method was called with correct parameters
        self.mock_db_integration.get_validation_results.assert_called_once_with(
            hardware_id="gpu_rtx3080",
            model_id="bert-base-uncased",
            start_date=None,
            end_date=None
        )
        
        # Check that the visualizer method was called with correct parameters
        # Note: We don't check the exact validation results since they're converted
        self.mock_visualizer.create_mape_comparison_chart.assert_called_once()
        args, kwargs = self.mock_visualizer.create_mape_comparison_chart.call_args
        self.assertEqual(kwargs["metric_name"], "throughput_items_per_second")
        self.assertEqual(kwargs["hardware_ids"], ["gpu_rtx3080"])
        self.assertEqual(kwargs["model_ids"], ["bert-base-uncased"])
        self.assertEqual(kwargs["output_path"], "/path/to/output.html")
        
        # Check the result
        self.assertEqual(result, "/path/to/chart.html")
    
    def test_create_hardware_comparison_heatmap_from_db(self):
        """Test creation of hardware comparison heatmap from database."""
        # Set up mock return values
        self.mock_db_integration.get_validation_results.return_value = self.sample_db_results
        self.mock_visualizer.create_hardware_comparison_heatmap.return_value = "/path/to/heatmap.html"
        
        # Call the method
        result = self.connector.create_hardware_comparison_heatmap_from_db(
            metric_name="average_latency_ms",
            model_ids=["bert-base-uncased"],
            output_path="/path/to/output.html"
        )
        
        # Check that the database method was called with correct parameters
        self.mock_db_integration.get_validation_results.assert_called_once_with(
            model_id="bert-base-uncased",
            start_date=None,
            end_date=None
        )
        
        # Check that the visualizer method was called with correct parameters
        self.mock_visualizer.create_hardware_comparison_heatmap.assert_called_once()
        args, kwargs = self.mock_visualizer.create_hardware_comparison_heatmap.call_args
        self.assertEqual(kwargs["metric_name"], "average_latency_ms")
        self.assertEqual(kwargs["model_ids"], ["bert-base-uncased"])
        self.assertEqual(kwargs["output_path"], "/path/to/output.html")
        
        # Check the result
        self.assertEqual(result, "/path/to/heatmap.html")
    
    def test_create_time_series_chart_from_db(self):
        """Test creation of time series chart from database."""
        # Set up mock return values
        self.mock_db_integration.get_validation_results.return_value = self.sample_db_results
        self.mock_visualizer.create_time_series_chart.return_value = "/path/to/timeseries.html"
        
        # Call the method
        result = self.connector.create_time_series_chart_from_db(
            metric_name="throughput_items_per_second",
            hardware_id="gpu_rtx3080",
            model_id="bert-base-uncased",
            output_path="/path/to/output.html"
        )
        
        # Check that the database method was called with correct parameters
        self.mock_db_integration.get_validation_results.assert_called_once_with(
            hardware_id="gpu_rtx3080",
            model_id="bert-base-uncased",
            start_date=None,
            end_date=None
        )
        
        # Check that the visualizer method was called with correct parameters
        self.mock_visualizer.create_time_series_chart.assert_called_once()
        args, kwargs = self.mock_visualizer.create_time_series_chart.call_args
        self.assertEqual(kwargs["metric_name"], "throughput_items_per_second")
        self.assertEqual(kwargs["hardware_id"], "gpu_rtx3080")
        self.assertEqual(kwargs["model_id"], "bert-base-uncased")
        self.assertEqual(kwargs["output_path"], "/path/to/output.html")
        
        # Check the result
        self.assertEqual(result, "/path/to/timeseries.html")
    
    def test_create_drift_visualization_from_db(self):
        """Test creation of drift visualization from database."""
        # Set up mock return values
        self.mock_db_integration.get_drift_detection_results.return_value = self.sample_drift_results
        self.mock_visualizer.create_drift_detection_visualization.return_value = "/path/to/drift.html"
        
        # Call the method
        result = self.connector.create_drift_visualization_from_db(
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            output_path="/path/to/output.html"
        )
        
        # Check that the database method was called with correct parameters
        self.mock_db_integration.get_drift_detection_results.assert_called_once_with(
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            is_significant=True,
            limit=1
        )
        
        # Check that the visualizer method was called with correct parameters
        self.mock_visualizer.create_drift_detection_visualization.assert_called_once_with(
            drift_results=self.sample_drift_results[0],
            interactive=None,
            output_path="/path/to/output.html",
            title=None
        )
        
        # Check the result
        self.assertEqual(result, "/path/to/drift.html")
    
    def test_create_calibration_improvement_chart_from_db(self):
        """Test creation of calibration improvement chart from database."""
        # Set up mock return values
        self.mock_db_integration.get_calibration_history.return_value = self.sample_calibration_history
        self.mock_visualizer.create_calibration_improvement_chart.return_value = "/path/to/calibration.html"
        
        # Call the method
        result = self.connector.create_calibration_improvement_chart_from_db(
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            output_path="/path/to/output.html"
        )
        
        # Check that the database method was called with correct parameters
        self.mock_db_integration.get_calibration_history.assert_called_once_with(
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            calibration_version=None,
            limit=1
        )
        
        # Check that the visualizer method was called
        self.mock_visualizer.create_calibration_improvement_chart.assert_called_once()
        
        # Check the result
        self.assertEqual(result, "/path/to/calibration.html")
    
    def test_create_comprehensive_dashboard_from_db(self):
        """Test creation of comprehensive dashboard from database."""
        # Set up mock return values
        self.mock_db_integration.get_validation_results.return_value = self.sample_db_results
        self.mock_visualizer.create_comprehensive_dashboard.return_value = "/path/to/dashboard.html"
        
        # Call the method
        result = self.connector.create_comprehensive_dashboard_from_db(
            hardware_id="gpu_rtx3080",
            model_id="bert-base-uncased",
            output_path="/path/to/output.html"
        )
        
        # Check that the database method was called with correct parameters
        self.mock_db_integration.get_validation_results.assert_called_once_with(
            hardware_id="gpu_rtx3080",
            model_id="bert-base-uncased",
            start_date=None,
            end_date=None,
            limit=500
        )
        
        # Check that the visualizer method was called
        self.mock_visualizer.create_comprehensive_dashboard.assert_called_once()
        
        # Check the result
        self.assertEqual(result, "/path/to/dashboard.html")
        
    def test_dashboard_integration(self):
        """Test the dashboard integration functionality."""
        # Create a connector with dashboard integration enabled
        with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.SimulationValidationDBIntegration') as mock_db_class:
            with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.ValidationVisualizer') as mock_vis_class:
                with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.requests') as mock_requests:
                    mock_db_instance = MagicMock()
                    mock_vis_instance = MagicMock()
                    mock_db_class.return_value = mock_db_instance
                    mock_vis_class.return_value = mock_vis_instance
                    
                    connector = ValidationVisualizerDBConnector(
                        dashboard_integration=True,
                        dashboard_url="http://localhost:8080/dashboard",
                        dashboard_api_key="test_api_key"
                    )
                    
                    # Check that the dashboard integration is enabled
                    self.assertTrue(connector.dashboard_integration)
                    self.assertEqual(connector.dashboard_url, "http://localhost:8080/dashboard")
                    self.assertEqual(connector.dashboard_api_key, "test_api_key")
                    self.assertTrue(connector.dashboard_connected)
        
    def test_upload_visualization_to_dashboard(self):
        """Test uploading a visualization to the dashboard."""
        # Create a connector with dashboard integration enabled
        with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.SimulationValidationDBIntegration') as mock_db_class:
            with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.ValidationVisualizer') as mock_vis_class:
                with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.requests') as mock_requests:
                    with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.time') as mock_time:
                        mock_db_instance = MagicMock()
                        mock_vis_instance = MagicMock()
                        mock_db_class.return_value = mock_db_instance
                        mock_vis_class.return_value = mock_vis_instance
                        mock_time.time.return_value = 12345
                        
                        connector = ValidationVisualizerDBConnector(
                            dashboard_integration=True,
                            dashboard_url="http://localhost:8080/dashboard",
                            dashboard_api_key="test_api_key"
                        )
                        
                        # Create a dummy visualization data
                        visualization_data = {
                            "type": "mape_comparison",
                            "data": {
                                "hardware_types": ["gpu_rtx3080", "cpu_intel_xeon"],
                                "models": ["bert-base-uncased", "vit-base-patch16-224"],
                                "values": [
                                    {"hardware": "gpu_rtx3080", "model": "bert-base-uncased", "mape": 5.2},
                                    {"hardware": "gpu_rtx3080", "model": "vit-base-patch16-224", "mape": 7.1},
                                    {"hardware": "cpu_intel_xeon", "model": "bert-base-uncased", "mape": 12.5},
                                    {"hardware": "cpu_intel_xeon", "model": "vit-base-patch16-224", "mape": 9.3}
                                ]
                            }
                        }
                        
                        # Test uploading to the dashboard
                        result = connector.upload_visualization_to_dashboard(
                            visualization_type="mape_comparison",
                            visualization_data=visualization_data,
                            dashboard_id="test_dashboard",
                            refresh_interval=60
                        )
                        
                        # Check result
                        self.assertEqual(result["status"], "success")
                        self.assertEqual(result["visualization_id"], "vis_mape_comparison_12345")
                        self.assertEqual(result["panel_id"], "panel_12345")
                        self.assertEqual(result["dashboard_id"], "test_dashboard")
        
    def test_create_dashboard_panel(self):
        """Test creating a dashboard panel."""
        # Create a connector with dashboard integration enabled
        with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.SimulationValidationDBIntegration') as mock_db_class:
            with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.ValidationVisualizer') as mock_vis_class:
                with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.requests') as mock_requests:
                    with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.time') as mock_time:
                        mock_db_instance = MagicMock()
                        mock_vis_instance = MagicMock()
                        mock_db_class.return_value = mock_db_instance
                        mock_vis_class.return_value = mock_vis_instance
                        mock_time.time.return_value = 12345
                        
                        # Set up mock return values for _get_data_for_mape_comparison
                        mape_data = {
                            "status": "success",
                            "data": {
                                "type": "mape_comparison",
                                "metric": "throughput_items_per_second",
                                "hardware_ids": ["gpu_rtx3080"],
                                "model_ids": ["bert-base-uncased"],
                                "mape_data": [],
                                "summary_data": []
                            }
                        }
                        
                        connector = ValidationVisualizerDBConnector(
                            dashboard_integration=True,
                            dashboard_url="http://localhost:8080/dashboard",
                            dashboard_api_key="test_api_key"
                        )
                        
                        # Mock the _get_data_for_mape_comparison method
                        connector._get_data_for_mape_comparison = MagicMock(return_value=mape_data)
                        connector.upload_visualization_to_dashboard = MagicMock(return_value={
                            "status": "success",
                            "visualization_id": "vis_mape_comparison_12345",
                            "panel_id": "panel_12345",
                            "dashboard_id": "test_dashboard"
                        })
                        
                        # Test creating a dashboard panel
                        result = connector.create_dashboard_panel_from_db(
                            panel_type="mape_comparison",
                            hardware_type="gpu_rtx3080",
                            model_type="bert-base-uncased",
                            metric="throughput_items_per_second",
                            dashboard_id="test_dashboard",
                            panel_title="MAPE Comparison Panel",
                            refresh_interval=60
                        )
                        
                        # Check result
                        self.assertEqual(result["status"], "success")
                        self.assertEqual(result["title"], "MAPE Comparison Panel")
        
    def test_create_comprehensive_monitoring_dashboard(self):
        """Test creating a comprehensive monitoring dashboard."""
        # Create a connector with dashboard integration enabled
        with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.SimulationValidationDBIntegration') as mock_db_class:
            with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.ValidationVisualizer') as mock_vis_class:
                with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.requests') as mock_requests:
                    with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.time') as mock_time:
                        mock_db_instance = MagicMock()
                        mock_vis_instance = MagicMock()
                        mock_db_class.return_value = mock_db_instance
                        mock_vis_class.return_value = mock_vis_instance
                        mock_time.time.return_value = 12345
                        
                        connector = ValidationVisualizerDBConnector(
                            dashboard_integration=True,
                            dashboard_url="http://localhost:8080/dashboard",
                            dashboard_api_key="test_api_key"
                        )
                        
                        # Mock the create_dashboard_panel_from_db method
                        connector.create_dashboard_panel_from_db = MagicMock(return_value={
                            "status": "success",
                            "panel_id": "panel_12345",
                            "dashboard_id": "dashboard_12345",
                            "title": "Test Panel"
                        })
                        
                        # Test creating a comprehensive monitoring dashboard
                        result = connector.create_comprehensive_monitoring_dashboard(
                            hardware_type="gpu_rtx3080",
                            model_type="bert-base-uncased",
                            dashboard_title="BERT GPU Monitoring Dashboard",
                            refresh_interval=60,
                            include_panels=["mape_comparison", "time_series", "drift_detection"]
                        )
                        
                        # Check result
                        self.assertEqual(result["status"], "success")
                        self.assertEqual(result["dashboard_id"], "dashboard_12345")
                        self.assertEqual(result["dashboard_title"], "BERT GPU Monitoring Dashboard")
                        self.assertEqual(result["panel_count"], 3)
                        
                        # Check that create_dashboard_panel_from_db was called for each panel
                        self.assertEqual(connector.create_dashboard_panel_from_db.call_count, 3)
        
    def test_set_up_real_time_monitoring(self):
        """Test setting up real-time monitoring."""
        # Create a connector with dashboard integration enabled
        with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.SimulationValidationDBIntegration') as mock_db_class:
            with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.ValidationVisualizer') as mock_vis_class:
                with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.requests') as mock_requests:
                    with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.time') as mock_time:
                        with patch('duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector.datetime') as mock_datetime:
                            mock_db_instance = MagicMock()
                            mock_vis_instance = MagicMock()
                            mock_db_class.return_value = mock_db_instance
                            mock_vis_class.return_value = mock_vis_instance
                            mock_time.time.return_value = 12345
                            
                            # Create a datetime object for the current time
                            now = datetime.datetime(2025, 3, 15, 12, 0, 0)
                            # Create a timedelta for 5 minutes
                            delta = datetime.timedelta(seconds=300)
                            # Calculate the next check time
                            next_check = now + delta
                            
                            mock_datetime.datetime.now.return_value = now
                            mock_datetime.timedelta.return_value = delta
                            
                            connector = ValidationVisualizerDBConnector(
                                dashboard_integration=True,
                                dashboard_url="http://localhost:8080/dashboard",
                                dashboard_api_key="test_api_key"
                            )
                            
                            # Mock the create_dashboard_panel_from_db method
                            connector.create_dashboard_panel_from_db = MagicMock(return_value={
                                "status": "success",
                                "panel_id": "panel_12345",
                                "dashboard_id": "test_dashboard",
                                "title": "Real-time Monitoring Panel"
                            })
                            
                            # Test setting up real-time monitoring
                            result = connector.set_up_real_time_monitoring(
                                hardware_type="gpu_rtx3080",
                                model_type="bert-base-uncased",
                                metrics=["throughput_mape", "latency_mape"],
                                monitoring_interval=300,
                                alert_thresholds={
                                    "throughput_mape": 10.0,
                                    "latency_mape": 12.0
                                },
                                dashboard_id="test_dashboard"
                            )
                            
                            # Check result
                            self.assertEqual(result["status"], "success")
                            self.assertEqual(result["monitoring_job_id"], "monitor_12345")
                            self.assertEqual(result["monitoring_config"]["hardware_type"], "gpu_rtx3080")
                            self.assertEqual(result["monitoring_config"]["model_type"], "bert-base-uncased")
                            self.assertEqual(result["monitoring_config"]["metrics"], ["throughput_mape", "latency_mape"])
                            self.assertEqual(result["monitoring_config"]["monitoring_interval"], 300)
                            self.assertEqual(result["monitoring_config"]["alert_thresholds"]["throughput_mape"], 10.0)
                            self.assertEqual(result["monitoring_config"]["alert_thresholds"]["latency_mape"], 12.0)
                            self.assertEqual(result["created_panels"], 2)
                            self.assertEqual(result["next_check"], next_check)
    
    def test_create_simulation_vs_hardware_chart_from_db(self):
        """Test creation of simulation vs hardware chart from database."""
        # Set up mock return values
        self.mock_db_integration.get_simulation_vs_hardware_values.return_value = self.sample_sim_vs_hw_data
        
        # Since we're implementing the visualization directly in the connector,
        # we need to patch pandas, matplotlib and/or plotly
        
        # First test with interactive visualization
        with patch('pandas.DataFrame') as mock_df:
            with patch('plotly.express.scatter') as mock_scatter:
                with patch('plotly.io.to_html') as mock_to_html:
                    mock_fig = MagicMock()
                    mock_scatter.return_value = mock_fig
                    mock_to_html.return_value = "<html>Interactive chart</html>"
                    
                    # Call the method
                    result = self.connector.create_simulation_vs_hardware_chart_from_db(
                        metric_name="throughput_items_per_second",
                        hardware_id="gpu_rtx3080",
                        model_id="bert-base-uncased",
                        interactive=True
                    )
                    
                    # Check that the database method was called with correct parameters
                    self.mock_db_integration.get_simulation_vs_hardware_values.assert_called_once_with(
                        model_id="bert-base-uncased",
                        hardware_id="gpu_rtx3080",
                        metric="throughput_items_per_second",
                        limit=100
                    )
                    
                    # Check that plotly was used for visualization
                    mock_scatter.assert_called_once()
                    mock_to_html.assert_called_once()
                    
                    # Check the result
                    self.assertEqual(result, "<html>Interactive chart</html>")
        
        # Reset mock
        self.mock_db_integration.get_simulation_vs_hardware_values.reset_mock()
        
        # Test with static visualization
        with patch('pandas.DataFrame') as mock_df:
            with patch('matplotlib.pyplot.figure') as mock_figure:
                with patch('matplotlib.pyplot.scatter') as mock_scatter:
                    with patch('matplotlib.pyplot.plot') as mock_plot:
                        with patch('matplotlib.pyplot.savefig') as mock_savefig:
                            with patch('base64.b64encode') as mock_b64encode:
                                mock_b64encode.return_value = b"base64_encoded_image"
                                
                                # Call the method
                                result = self.connector.create_simulation_vs_hardware_chart_from_db(
                                    metric_name="throughput_items_per_second",
                                    hardware_id="gpu_rtx3080",
                                    model_id="bert-base-uncased",
                                    interactive=False
                                )
                                
                                # Check that the database method was called with correct parameters
                                self.mock_db_integration.get_simulation_vs_hardware_values.assert_called_once_with(
                                    model_id="bert-base-uncased",
                                    hardware_id="gpu_rtx3080",
                                    metric="throughput_items_per_second",
                                    limit=100
                                )
                                
                                # Check that matplotlib was used for visualization
                                mock_figure.assert_called_once()
                                
                                # Result should contain an image tag with base64 data
                                self.assertTrue(result.startswith('<img src="data:image/png;base64,'))
    
    def test_create_metrics_over_time_chart_from_db(self):
        """Test creation of metrics over time chart from database."""
        # Set up mock return values
        self.mock_db_integration.get_validation_metrics_over_time.return_value = self.sample_time_series_data
        
        # Test with interactive visualization
        with patch('pandas.DataFrame') as mock_df:
            with patch('plotly.express.line') as mock_line:
                with patch('plotly.io.to_html') as mock_to_html:
                    mock_fig = MagicMock()
                    mock_line.return_value = mock_fig
                    mock_to_html.return_value = "<html>Interactive chart</html>"
                    
                    # Call the method
                    result = self.connector.create_metrics_over_time_chart_from_db(
                        metric="throughput_mape",
                        hardware_type="gpu_rtx3080",
                        model_id="bert-base-uncased",
                        interactive=True
                    )
                    
                    # Check that the database method was called with correct parameters
                    self.mock_db_integration.get_validation_metrics_over_time.assert_called_once_with(
                        hardware_type="gpu_rtx3080",
                        model_id="bert-base-uncased",
                        metric="throughput_mape",
                        time_bucket="day",
                        start_date=None,
                        end_date=None
                    )
                    
                    # Check that plotly was used for visualization
                    mock_line.assert_called_once()
                    mock_to_html.assert_called_once()
                    
                    # Check the result
                    self.assertEqual(result, "<html>Interactive chart</html>")
    
    def test_export_visualization_data_from_db(self):
        """Test exporting visualization data from database."""
        # Set up mock return values
        self.mock_db_integration.export_data_for_visualization.return_value = {
            "status": "success",
            "query_type": "sim_vs_hw",
            "timestamp": "2025-03-15T12:00:00",
            "data": self.sample_sim_vs_hw_data
        }
        
        # Define output path
        output_path = self.output_dir / "test_export.json"
        
        # Call the method with mocked file operations
        with patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            with patch('json.dump') as mock_json_dump:
                result = self.connector.export_visualization_data_from_db(
                    query_type="sim_vs_hw",
                    export_path=str(output_path),
                    hardware_type="gpu_rtx3080",
                    model_id="bert-base-uncased",
                    metric="throughput_items_per_second"
                )
                
                # Check that the database method was called with correct parameters
                self.mock_db_integration.export_data_for_visualization.assert_called_once_with(
                    query_type="sim_vs_hw",
                    hardware_type="gpu_rtx3080",
                    model_id="bert-base-uncased",
                    metric="throughput_items_per_second",
                    start_date=None,
                    end_date=None
                )
                
                # Check that file operations were performed
                mock_open.assert_called_once_with(str(output_path), 'w')
                mock_json_dump.assert_called_once()
                
                # Check the result
                self.assertTrue(result)
    
    def test_visualize_calibration_effectiveness_from_db(self):
        """Test visualization of calibration effectiveness from database."""
        # Set up mock return values
        self.mock_db_integration.analyze_calibration_effectiveness.return_value = self.sample_calibration_analysis
        
        # Test with interactive visualization
        with patch('pandas.DataFrame') as mock_df:
            with patch('plotly.graph_objects.Bar') as mock_bar:
                with patch('plotly.graph_objects.Scatter') as mock_scatter:
                    with patch('plotly.io.to_html') as mock_to_html:
                        with patch('plotly.subplots.make_subplots') as mock_make_subplots:
                            mock_fig = MagicMock()
                            mock_make_subplots.return_value = mock_fig
                            mock_to_html.return_value = "<html>Interactive chart</html>"
                            
                            # Call the method
                            result = self.connector.visualize_calibration_effectiveness_from_db(
                                hardware_type="gpu_rtx3080",
                                model_type="bert-base-uncased",
                                interactive=True
                            )
                            
                            # Check that the database method was called with correct parameters
                            self.mock_db_integration.analyze_calibration_effectiveness.assert_called_once_with(
                                hardware_type="gpu_rtx3080",
                                model_type="bert-base-uncased",
                                start_date=None,
                                end_date=None
                            )
                            
                            # Check that plotly was used for visualization
                            mock_make_subplots.assert_called_once()
                            mock_to_html.assert_called_once()
                            
                            # Check the result
                            self.assertEqual(result, "<html>Interactive chart</html>")
    
    def test_convert_db_results_to_validation_results(self):
        """Test conversion of database results to ValidationResult objects."""
        # Call the method
        results = self.connector._convert_db_results_to_validation_results(self.sample_db_results)
        
        # Check that we got the expected number of results
        self.assertEqual(len(results), 2)
        
        # Check that each result is a ValidationResult object
        for result in results:
            self.assertIsInstance(result, ValidationResult)
            self.assertIsInstance(result.simulation_result, SimulationResult)
            self.assertIsInstance(result.hardware_result, HardwareResult)
    
    def test_convert_json_to_validation_results(self):
        """Test conversion of JSON data to ValidationResult objects."""
        # Call the method with data from calibration history
        json_data = self.sample_calibration_history[0]["validation_results_before"]
        results = self.connector._convert_json_to_validation_results(json_data)
        
        # Check that we got the expected number of results
        self.assertEqual(len(results), 1)
        
        # Check that each result is a ValidationResult object
        for result in results:
            self.assertIsInstance(result, ValidationResult)
            self.assertIsInstance(result.simulation_result, SimulationResult)
            self.assertIsInstance(result.hardware_result, HardwareResult)


if __name__ == "__main__":
    unittest.main()