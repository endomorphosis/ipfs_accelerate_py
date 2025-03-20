#!/usr/bin/env python3
"""
End-to-end test for the database visualization integration in the Simulation Accuracy and Validation Framework.

This test validates the entire flow from database operations to visualization generation,
ensuring that the components work together correctly in real-world scenarios.
"""

import os
import sys
import unittest
import json
import tempfile
import datetime
from pathlib import Path
import shutil

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required modules
from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration
from duckdb_api.simulation_validation.visualization.validation_visualizer import ValidationVisualizer
from duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult,
    CalibrationRecord,
    DriftDetectionResult
)


class TestE2EVisualizationDBIntegration(unittest.TestCase):
    """End-to-end tests for visualization database integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Create temporary directory for outputs
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_e2e_db.duckdb")
        cls.output_dir = os.path.join(cls.temp_dir, "visualizations")
        
        # Ensure output directory exists
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Create actual instances (not mocks)
        cls.db_integration = SimulationValidationDBIntegration(db_path=cls.db_path)
        cls.visualizer = ValidationVisualizer()
        cls.connector = ValidationVisualizerDBConnector(
            db_integration=cls.db_integration,
            visualizer=cls.visualizer
        )
        
        # Initialize database
        cls.db_integration.initialize_database()
        
        # Generate test data and populate database
        cls._populate_test_database()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are completed."""
        # Close database connection
        cls.db_integration.close()
        
        # Remove temporary directory and all contents
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _populate_test_database(cls):
        """Populate the database with test data for visualization testing."""
        # Create test hardware results
        hardware_results = []
        
        # GPU results
        for i in range(10):
            timestamp = datetime.datetime(2025, 3, 1) + datetime.timedelta(days=i)
            
            # Create a hardware result
            hw_result = HardwareResult(
                model_id="bert-base-uncased",
                hardware_id=f"gpu_rtx3080",
                batch_size=32,
                precision="fp16",
                timestamp=timestamp.isoformat(),
                metrics={
                    "throughput_items_per_second": 90.0 + (i * 0.5),
                    "average_latency_ms": 17.0 - (i * 0.1),
                    "peak_memory_mb": 2200 + (i * 10)
                },
                hardware_details={
                    "name": "NVIDIA RTX 3080",
                    "compute_capability": "8.6",
                    "vram_gb": 10
                },
                test_environment={
                    "os": "Linux",
                    "cuda_version": "11.4"
                }
            )
            hardware_results.append(hw_result)
        
        # CPU results
        for i in range(10):
            timestamp = datetime.datetime(2025, 3, 1) + datetime.timedelta(days=i)
            
            # Create a hardware result
            hw_result = HardwareResult(
                model_id="bert-base-uncased",
                hardware_id=f"cpu_intel_xeon",
                batch_size=16,
                precision="fp32",
                timestamp=timestamp.isoformat(),
                metrics={
                    "throughput_items_per_second": 40.0 + (i * 0.3),
                    "average_latency_ms": 35.0 - (i * 0.2),
                    "peak_memory_mb": 1800 + (i * 5)
                },
                hardware_details={
                    "name": "Intel Xeon",
                    "cores": 24,
                    "threads": 48
                },
                test_environment={
                    "os": "Linux",
                    "memory_gb": 64
                }
            )
            hardware_results.append(hw_result)
        
        # Create simulation results
        simulation_results = []
        
        # GPU simulation
        for i in range(10):
            timestamp = datetime.datetime(2025, 3, 1) + datetime.timedelta(days=i)
            
            # Create a simulation result
            sim_result = SimulationResult(
                model_id="bert-base-uncased",
                hardware_id=f"gpu_rtx3080",
                batch_size=32,
                precision="fp16",
                timestamp=timestamp.isoformat(),
                simulation_version="sim_v1.0",
                metrics={
                    "throughput_items_per_second": 95.0 + (i * 0.7),
                    "average_latency_ms": 16.0 - (i * 0.08),
                    "peak_memory_mb": 2000 + (i * 15)
                },
                simulation_params={
                    "model_params": {"hidden_size": 768, "num_layers": 12},
                    "hardware_params": {"gpu_compute_capability": "8.6", "gpu_memory": 10240}
                }
            )
            simulation_results.append(sim_result)
        
        # CPU simulation
        for i in range(10):
            timestamp = datetime.datetime(2025, 3, 1) + datetime.timedelta(days=i)
            
            # Create a simulation result
            sim_result = SimulationResult(
                model_id="bert-base-uncased",
                hardware_id=f"cpu_intel_xeon",
                batch_size=16,
                precision="fp32",
                timestamp=timestamp.isoformat(),
                simulation_version="sim_v1.0",
                metrics={
                    "throughput_items_per_second": 43.0 + (i * 0.4),
                    "average_latency_ms": 33.0 - (i * 0.15),
                    "peak_memory_mb": 1700 + (i * 8)
                },
                simulation_params={
                    "model_params": {"hidden_size": 768, "num_layers": 12},
                    "hardware_params": {"cpu_cores": 24, "cpu_threads": 48}
                }
            )
            simulation_results.append(sim_result)
        
        # Create validation results
        validation_results = []
        
        # Match hardware and simulation results to create validation results
        for i in range(20):
            hw_result = hardware_results[i]
            sim_result = simulation_results[i]
            
            # Create metrics comparison
            metrics_comparison = {}
            for metric in ["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"]:
                if metric in hw_result.metrics and metric in sim_result.metrics:
                    hw_value = hw_result.metrics[metric]
                    sim_value = sim_result.metrics[metric]
                    abs_error = abs(sim_value - hw_value)
                    rel_error = abs_error / hw_value if hw_value != 0 else 0
                    mape = rel_error * 100
                    
                    metrics_comparison[metric] = {
                        "simulation_value": sim_value,
                        "hardware_value": hw_value,
                        "absolute_error": abs_error,
                        "relative_error": rel_error,
                        "mape": mape
                    }
            
            # Calculate overall MAPE
            mape_values = [comparison["mape"] for comparison in metrics_comparison.values()]
            overall_mape = sum(mape_values) / len(mape_values) if mape_values else 0
            
            # Create a validation result
            val_result = ValidationResult(
                simulation_result=sim_result,
                hardware_result=hw_result,
                metrics_comparison=metrics_comparison,
                validation_timestamp=hw_result.timestamp,
                validation_version="v1.0",
                overall_accuracy_score=overall_mape
            )
            validation_results.append(val_result)
        
        # Store all results in the database
        for sim_result in simulation_results:
            cls.db_integration.store_simulation_result(sim_result)
        
        for hw_result in hardware_results:
            cls.db_integration.store_hardware_result(hw_result)
        
        for val_result in validation_results:
            cls.db_integration.store_validation_result(val_result)
        
        # Create and store calibration records
        calibration_records = []
        
        # GPU calibration
        calibration_record = CalibrationRecord(
            id="cal_1",
            timestamp=datetime.datetime(2025, 3, 15).isoformat(),
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            previous_parameters={
                "correction_factors": {
                    "throughput_items_per_second": 1.0,
                    "average_latency_ms": 1.0,
                    "peak_memory_mb": 1.0
                }
            },
            updated_parameters={
                "correction_factors": {
                    "throughput_items_per_second": 0.95,
                    "average_latency_ms": 1.05,
                    "peak_memory_mb": 0.9
                }
            },
            validation_results_before=validation_results[0:2],
            validation_results_after=[
                ValidationResult(
                    simulation_result=SimulationResult(
                        model_id="bert-base-uncased",
                        hardware_id="gpu_rtx3080",
                        batch_size=32,
                        precision="fp16",
                        timestamp=datetime.datetime(2025, 3, 15).isoformat(),
                        simulation_version="sim_v1.1",
                        metrics={
                            "throughput_items_per_second": 91.0,
                            "average_latency_ms": 17.5,
                            "peak_memory_mb": 2100
                        }
                    ),
                    hardware_result=HardwareResult(
                        model_id="bert-base-uncased",
                        hardware_id="gpu_rtx3080",
                        batch_size=32,
                        precision="fp16",
                        timestamp=datetime.datetime(2025, 3, 15).isoformat(),
                        metrics={
                            "throughput_items_per_second": 92.0,
                            "average_latency_ms": 17.0,
                            "peak_memory_mb": 2200
                        }
                    ),
                    metrics_comparison={
                        "throughput_items_per_second": {
                            "simulation_value": 91.0,
                            "hardware_value": 92.0,
                            "absolute_error": 1.0,
                            "relative_error": 0.011,
                            "mape": 1.1
                        },
                        "average_latency_ms": {
                            "simulation_value": 17.5,
                            "hardware_value": 17.0,
                            "absolute_error": 0.5,
                            "relative_error": 0.029,
                            "mape": 2.9
                        },
                        "peak_memory_mb": {
                            "simulation_value": 2100,
                            "hardware_value": 2200,
                            "absolute_error": 100,
                            "relative_error": 0.045,
                            "mape": 4.5
                        }
                    },
                    validation_timestamp=datetime.datetime(2025, 3, 15).isoformat(),
                    validation_version="v1.0"
                )
            ],
            improvement_metrics={
                "overall": {
                    "before_mape": 9.83,
                    "after_mape": 2.83,
                    "absolute_improvement": 7.0,
                    "relative_improvement_pct": 71.21
                },
                "throughput_items_per_second": {
                    "before_mape": 8.61,
                    "after_mape": 1.1,
                    "absolute_improvement": 7.51,
                    "relative_improvement_pct": 87.22
                },
                "average_latency_ms": {
                    "before_mape": 6.47,
                    "after_mape": 2.9,
                    "absolute_improvement": 3.57,
                    "relative_improvement_pct": 55.18
                },
                "peak_memory_mb": {
                    "before_mape": 14.41,
                    "after_mape": 4.5,
                    "absolute_improvement": 9.91,
                    "relative_improvement_pct": 68.77
                }
            },
            calibration_version="v1.0"
        )
        calibration_records.append(calibration_record)
        
        # CPU calibration
        calibration_record = CalibrationRecord(
            id="cal_2",
            timestamp=datetime.datetime(2025, 3, 16).isoformat(),
            hardware_type="cpu_intel_xeon",
            model_type="bert-base-uncased",
            previous_parameters={
                "correction_factors": {
                    "throughput_items_per_second": 1.0,
                    "average_latency_ms": 1.0,
                    "peak_memory_mb": 1.0
                }
            },
            updated_parameters={
                "correction_factors": {
                    "throughput_items_per_second": 0.92,
                    "average_latency_ms": 1.08,
                    "peak_memory_mb": 0.95
                }
            },
            validation_results_before=validation_results[10:12],
            validation_results_after=[
                ValidationResult(
                    simulation_result=SimulationResult(
                        model_id="bert-base-uncased",
                        hardware_id="cpu_intel_xeon",
                        batch_size=16,
                        precision="fp32",
                        timestamp=datetime.datetime(2025, 3, 16).isoformat(),
                        simulation_version="sim_v1.1",
                        metrics={
                            "throughput_items_per_second": 39.6,
                            "average_latency_ms": 37.1,
                            "peak_memory_mb": 1785
                        }
                    ),
                    hardware_result=HardwareResult(
                        model_id="bert-base-uncased",
                        hardware_id="cpu_intel_xeon",
                        batch_size=16,
                        precision="fp32",
                        timestamp=datetime.datetime(2025, 3, 16).isoformat(),
                        metrics={
                            "throughput_items_per_second": 40.3,
                            "average_latency_ms": 34.8,
                            "peak_memory_mb": 1805
                        }
                    ),
                    metrics_comparison={
                        "throughput_items_per_second": {
                            "simulation_value": 39.6,
                            "hardware_value": 40.3,
                            "absolute_error": 0.7,
                            "relative_error": 0.017,
                            "mape": 1.7
                        },
                        "average_latency_ms": {
                            "simulation_value": 37.1,
                            "hardware_value": 34.8,
                            "absolute_error": 2.3,
                            "relative_error": 0.066,
                            "mape": 6.6
                        },
                        "peak_memory_mb": {
                            "simulation_value": 1785,
                            "hardware_value": 1805,
                            "absolute_error": 20,
                            "relative_error": 0.011,
                            "mape": 1.1
                        }
                    },
                    validation_timestamp=datetime.datetime(2025, 3, 16).isoformat(),
                    validation_version="v1.0"
                )
            ],
            improvement_metrics={
                "overall": {
                    "before_mape": 11.87,
                    "after_mape": 3.13,
                    "absolute_improvement": 8.74,
                    "relative_improvement_pct": 73.63
                },
                "throughput_items_per_second": {
                    "before_mape": 10.54,
                    "after_mape": 1.7,
                    "absolute_improvement": 8.84,
                    "relative_improvement_pct": 83.87
                },
                "average_latency_ms": {
                    "before_mape": 9.25,
                    "after_mape": 6.6,
                    "absolute_improvement": 2.65,
                    "relative_improvement_pct": 28.65
                },
                "peak_memory_mb": {
                    "before_mape": 15.82,
                    "after_mape": 1.1,
                    "absolute_improvement": 14.72,
                    "relative_improvement_pct": 93.05
                }
            },
            calibration_version="v1.0"
        )
        calibration_records.append(calibration_record)
        
        # Store calibration records
        for cal_record in calibration_records:
            cls.db_integration.store_calibration_record(cal_record)
        
        # Create and store drift detection results
        drift_results = []
        
        # GPU drift
        drift_result = DriftDetectionResult(
            id="drift_1",
            timestamp=datetime.datetime(2025, 3, 20).isoformat(),
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            drift_metrics={
                "throughput_items_per_second": {
                    "p_value": 0.03,
                    "drift_detected": True,
                    "mean_change_pct": 15.5
                },
                "average_latency_ms": {
                    "p_value": 0.07,
                    "drift_detected": False,
                    "mean_change_pct": 5.2
                },
                "peak_memory_mb": {
                    "p_value": 0.01,
                    "drift_detected": True,
                    "mean_change_pct": 18.3
                }
            },
            is_significant=True,
            historical_window_start=datetime.datetime(2025, 3, 1).isoformat(),
            historical_window_end=datetime.datetime(2025, 3, 10).isoformat(),
            new_window_start=datetime.datetime(2025, 3, 11).isoformat(),
            new_window_end=datetime.datetime(2025, 3, 20).isoformat(),
            thresholds_used={
                "p_value": 0.05,
                "mean_change_pct": 10.0
            }
        )
        drift_results.append(drift_result)
        
        # CPU drift
        drift_result = DriftDetectionResult(
            id="drift_2",
            timestamp=datetime.datetime(2025, 3, 21).isoformat(),
            hardware_type="cpu_intel_xeon",
            model_type="bert-base-uncased",
            drift_metrics={
                "throughput_items_per_second": {
                    "p_value": 0.12,
                    "drift_detected": False,
                    "mean_change_pct": 5.5
                },
                "average_latency_ms": {
                    "p_value": 0.02,
                    "drift_detected": True,
                    "mean_change_pct": 12.7
                },
                "peak_memory_mb": {
                    "p_value": 0.09,
                    "drift_detected": False,
                    "mean_change_pct": 7.3
                }
            },
            is_significant=True,
            historical_window_start=datetime.datetime(2025, 3, 1).isoformat(),
            historical_window_end=datetime.datetime(2025, 3, 10).isoformat(),
            new_window_start=datetime.datetime(2025, 3, 11).isoformat(),
            new_window_end=datetime.datetime(2025, 3, 21).isoformat(),
            thresholds_used={
                "p_value": 0.05,
                "mean_change_pct": 10.0
            }
        )
        drift_results.append(drift_result)
        
        # Store drift detection results
        for drift_result in drift_results:
            cls.db_integration.store_drift_detection_result(drift_result)
    
    def test_e2e_mape_comparison_chart(self):
        """Test end-to-end creation of MAPE comparison chart from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_mape_comparison.html")
        
        # Create the chart
        result = self.connector.create_mape_comparison_chart_from_db(
            hardware_ids=["gpu_rtx3080", "cpu_intel_xeon"],
            model_ids=["bert-base-uncased"],
            metric_name="throughput_items_per_second",
            output_path=output_path
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)
        
        # Check file content (simple validation)
        with open(output_path, 'r') as f:
            content = f.read()
            # Verify it contains expected HTML elements
            self.assertIn("<html", content)
            self.assertIn("MAPE Comparison", content)
    
    def test_e2e_hardware_comparison_heatmap(self):
        """Test end-to-end creation of hardware comparison heatmap from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_hardware_heatmap.html")
        
        # Create the chart
        result = self.connector.create_hardware_comparison_heatmap_from_db(
            metric_name="average_latency_ms",
            model_ids=["bert-base-uncased"],
            output_path=output_path
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)
        
        # Check file content (simple validation)
        with open(output_path, 'r') as f:
            content = f.read()
            # Verify it contains expected HTML elements
            self.assertIn("<html", content)
            self.assertIn("Hardware Comparison", content)
    
    def test_e2e_time_series_chart(self):
        """Test end-to-end creation of time series chart from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_time_series.html")
        
        # Create the chart
        result = self.connector.create_time_series_chart_from_db(
            metric_name="throughput_items_per_second",
            hardware_id="gpu_rtx3080",
            model_id="bert-base-uncased",
            output_path=output_path
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)
        
        # Check file content (simple validation)
        with open(output_path, 'r') as f:
            content = f.read()
            # Verify it contains expected HTML elements
            self.assertIn("<html", content)
            self.assertIn("Time Series", content)
    
    def test_e2e_drift_visualization(self):
        """Test end-to-end creation of drift visualization from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_drift.html")
        
        # Create the chart
        result = self.connector.create_drift_visualization_from_db(
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            output_path=output_path
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)
        
        # Check file content (simple validation)
        with open(output_path, 'r') as f:
            content = f.read()
            # Verify it contains expected HTML elements
            self.assertIn("<html", content)
            self.assertIn("Drift Detection", content)
    
    def test_e2e_calibration_improvement_chart(self):
        """Test end-to-end creation of calibration improvement chart from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_calibration.html")
        
        # Create the chart
        result = self.connector.create_calibration_improvement_chart_from_db(
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            output_path=output_path
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)
        
        # Check file content (simple validation)
        with open(output_path, 'r') as f:
            content = f.read()
            # Verify it contains expected HTML elements
            self.assertIn("<html", content)
            self.assertIn("Calibration Improvement", content)
    
    def test_e2e_simulation_vs_hardware_chart(self):
        """Test end-to-end creation of simulation vs hardware chart from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_sim_vs_hw.html")
        
        # Create the chart
        result = self.connector.create_simulation_vs_hardware_chart_from_db(
            metric_name="throughput_items_per_second",
            hardware_id="gpu_rtx3080",
            model_id="bert-base-uncased",
            interactive=True,
            output_path=output_path
        )
        
        # Check that the file was created with proper content
        if os.path.exists(output_path):
            # If file is directly saved
            with open(output_path, 'r') as f:
                content = f.read()
                # Verify it contains expected HTML elements
                self.assertIn("<html", content)
                self.assertIn("Simulation vs Hardware", content)
        else:
            # If result is HTML content
            self.assertIn("<html", result)
            self.assertIn("Simulation vs Hardware", result)
            
            # Save it for inspection
            with open(output_path, 'w') as f:
                f.write(result)
    
    def test_e2e_comprehensive_dashboard(self):
        """Test end-to-end creation of comprehensive dashboard from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_dashboard.html")
        
        # Create the dashboard
        result = self.connector.create_comprehensive_dashboard_from_db(
            hardware_id="gpu_rtx3080",
            model_id="bert-base-uncased",
            output_path=output_path
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)
        
        # Check file content (simple validation)
        with open(output_path, 'r') as f:
            content = f.read()
            # Verify it contains expected HTML elements
            self.assertIn("<html", content)
            self.assertIn("Dashboard", content)
    
    def test_e2e_export_visualization_data(self):
        """Test end-to-end export of visualization data from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_export.json")
        
        # Export the data
        result = self.connector.export_visualization_data_from_db(
            query_type="sim_vs_hw",
            export_path=output_path,
            hardware_type="gpu_rtx3080",
            model_id="bert-base-uncased",
            metric="throughput_items_per_second"
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(result)
        
        # Check file content (validate JSON structure)
        with open(output_path, 'r') as f:
            data = json.load(f)
            # Verify expected keys
            self.assertIn("status", data)
            self.assertIn("query_type", data)
            self.assertIn("data", data)
            # Verify status
            self.assertEqual(data["status"], "success")
            # Verify query type
            self.assertEqual(data["query_type"], "sim_vs_hw")
            # Verify data is a list
            self.assertIsInstance(data["data"], list)
            # Verify data contains expected fields
            if data["data"]:
                item = data["data"][0]
                self.assertIn("model_id", item)
                self.assertIn("hardware_type", item)
                self.assertIn("simulation_value", item)
                self.assertIn("hardware_value", item)
    
    def test_e2e_visualize_calibration_effectiveness(self):
        """Test end-to-end visualization of calibration effectiveness from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_cal_effectiveness.html")
        
        # Create the visualization
        result = self.connector.visualize_calibration_effectiveness_from_db(
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            interactive=True,
            output_path=output_path
        )
        
        # Check that the file was created or result contains HTML
        if os.path.exists(output_path):
            # If file is directly saved
            with open(output_path, 'r') as f:
                content = f.read()
                # Verify it contains expected HTML elements
                self.assertIn("<html", content)
                self.assertIn("Calibration Effectiveness", content)
        else:
            # If result is HTML content
            self.assertIn("<html", result)
            self.assertIn("Calibration Effectiveness", result)
            
            # Save it for inspection
            with open(output_path, 'w') as f:
                f.write(result)
    
    def test_e2e_metrics_over_time_chart(self):
        """Test end-to-end creation of metrics over time chart from database."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_metrics_over_time.html")
        
        # Create the chart
        result = self.connector.create_metrics_over_time_chart_from_db(
            metric="throughput_mape",
            hardware_type="gpu_rtx3080",
            model_id="bert-base-uncased",
            interactive=True,
            output_path=output_path
        )
        
        # Check that the file was created or result contains HTML
        if os.path.exists(output_path):
            # If file is directly saved
            with open(output_path, 'r') as f:
                content = f.read()
                # Verify it contains expected HTML elements
                self.assertIn("<html", content)
                self.assertIn("Metrics Over Time", content)
        else:
            # If result is HTML content
            self.assertIn("<html", result)
            self.assertIn("Metrics Over Time", result)
            
            # Save it for inspection
            with open(output_path, 'w') as f:
                f.write(result)
    
    def test_e2e_multiple_hardware_comparison(self):
        """Test end-to-end comparison of multiple hardware types."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_multi_hardware.html")
        
        # Create the chart comparing GPU and CPU
        result = self.connector.create_mape_comparison_chart_from_db(
            hardware_ids=["gpu_rtx3080", "cpu_intel_xeon"],
            model_ids=["bert-base-uncased"],
            metric_name="throughput_items_per_second",
            output_path=output_path
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check file content for both hardware types
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn("gpu_rtx3080", content)
            self.assertIn("cpu_intel_xeon", content)
    
    def test_e2e_edge_case_empty_dataset(self):
        """Test handling of edge case with empty dataset."""
        # Define output path
        output_path = os.path.join(self.output_dir, "e2e_empty_dataset.html")
        
        # Create the chart with non-existent hardware
        result = self.connector.create_mape_comparison_chart_from_db(
            hardware_ids=["non_existent_hardware"],
            model_ids=["bert-base-uncased"],
            metric_name="throughput_items_per_second",
            output_path=output_path
        )
        
        # Check that the file was created with appropriate message
        self.assertTrue(os.path.exists(output_path))
        
        # Check file content for empty dataset message
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn("<html", content)
            # Should contain some indication of empty or no data
            self.assertTrue(
                any(phrase in content for phrase in ["No data", "Empty dataset", "No results"])
            )


if __name__ == "__main__":
    unittest.main()