#!/usr/bin/env python3
"""
Integration test for the Dashboard System with the Result Aggregator Service.

This tests the integration between the dashboard components and the result aggregator
to ensure they work together correctly for visualization and reporting.
"""

import os
import sys
import unittest
import tempfile
import anyio
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Import dashboard components
    from duckdb_api.distributed_testing.dashboard.dashboard_generator import DashboardGenerator
    from duckdb_api.distributed_testing.dashboard.dashboard_server import DashboardServer
    from duckdb_api.distributed_testing.dashboard.visualization import VisualizationEngine
    
    # Import result aggregator
    from duckdb_api.distributed_testing.result_aggregator.service import ResultAggregatorService
    
    # Import database manager
    from duckdb_api.core.db_manager import BenchmarkDBManager
    
    # Check if test should be skipped due to missing dependencies
    SKIP_TEST = False
except ImportError as e:
    print(f"Import error: {e}")
    SKIP_TEST = True


class MockDBManager:
    """Mock database manager for testing."""
    
    def __init__(self):
        """Initialize with mock data."""
        self.performance_results = self._generate_performance_results()
        self.compatibility_results = []
        self.integration_results = []
        self.web_platform_results = []
        
    def get_performance_results(self, aggregation_level=None, filter_params=None, time_range=None):
        """Return mock performance results."""
        return self.performance_results
        
    def get_compatibility_results(self, aggregation_level=None, filter_params=None, time_range=None):
        """Return mock compatibility results."""
        return self.compatibility_results
        
    def get_integration_test_results(self, aggregation_level=None, filter_params=None, time_range=None):
        """Return mock integration test results."""
        return self.integration_results
        
    def get_web_platform_results(self, aggregation_level=None, filter_params=None, time_range=None):
        """Return mock web platform results."""
        return self.web_platform_results
        
    def get_hardware_info(self, hardware_id):
        """Return mock hardware information."""
        hardware_info = {
            "gpu1": {
                "device_name": "NVIDIA A100",
                "hardware_type": "gpu",
                "platform": "cuda",
                "memory_gb": 40
            },
            "gpu2": {
                "device_name": "AMD MI250",
                "hardware_type": "gpu",
                "platform": "rocm",
                "memory_gb": 64
            },
            "cpu1": {
                "device_name": "Intel Xeon",
                "hardware_type": "cpu",
                "platform": "x86_64",
                "memory_gb": 128
            }
        }
        return hardware_info.get(hardware_id, {})
        
    def get_model_info(self, model_id):
        """Return mock model information."""
        model_info = {
            "bert-base-uncased": {
                "model_name": "BERT Base Uncased",
                "model_family": "bert",
                "modality": "text",
                "parameters_million": 110
            },
            "t5-base": {
                "model_name": "T5 Base",
                "model_family": "t5",
                "modality": "text",
                "parameters_million": 220
            },
            "vit-base-patch16-224": {
                "model_name": "ViT Base",
                "model_family": "vit",
                "modality": "vision",
                "parameters_million": 86
            }
        }
        return model_info.get(model_id, {})
        
    def _generate_performance_results(self):
        """Generate mock performance results."""
        results = []
        
        # Models to use
        models = ["bert-base-uncased", "t5-base", "vit-base-patch16-224"]
        
        # Hardware to use
        hardware = ["gpu1", "gpu2", "cpu1"]
        
        # Generate results for each model-hardware combination
        for model_id in models:
            for hardware_id in hardware:
                # Base performance values
                if hardware_id == "gpu1":
                    base_throughput = 150.0
                    base_latency = 30.0
                    base_memory = 2.0
                elif hardware_id == "gpu2":
                    base_throughput = 140.0
                    base_latency = 32.0
                    base_memory = 2.2
                else:  # CPU
                    base_throughput = 50.0
                    base_latency = 90.0
                    base_memory = 3.5
                
                # Adjust based on model
                if model_id == "bert-base-uncased":
                    model_factor = 1.0
                elif model_id == "t5-base":
                    model_factor = 0.8
                    base_memory *= 1.5
                else:  # ViT
                    model_factor = 1.2
                
                # Generate results for the past 7 days
                for i in range(7):
                    # Add some variation to the results
                    throughput_variation = (i % 3 - 1) * 5.0  # -5, 0, or 5
                    latency_variation = (i % 3 - 1) * 2.0  # -2, 0, or 2
                    memory_variation = (i % 3 - 1) * 0.1  # -0.1, 0, or 0.1
                    
                    # On day 5, introduce a regression for bert on gpu1
                    if i == 5 and model_id == "bert-base-uncased" and hardware_id == "gpu1":
                        throughput_variation = -20.0
                        latency_variation = 10.0
                    
                    # Calculate actual values
                    throughput = base_throughput * model_factor + throughput_variation
                    latency = base_latency / model_factor + latency_variation
                    memory = base_memory * model_factor + memory_variation
                    
                    # Create result
                    timestamp = datetime.now() - timedelta(days=i)
                    result = {
                        "model_id": model_id,
                        "hardware_id": hardware_id,
                        "run_id": f"run-{model_id}-{hardware_id}-{i}",
                        "batch_size": 8,
                        "precision": "fp16",
                        "total_time_seconds": 60.0,
                        "average_latency_ms": latency,
                        "throughput_items_per_second": throughput,
                        "memory_peak_mb": memory * 1024,
                        "timestamp": timestamp,
                        "is_simulated": False,
                        "worker_id": f"worker-{hardware_id}"
                    }
                    
                    results.append(result)
        
        return results


class TestDashboardIntegration(unittest.TestCase):
    """Integration tests for the dashboard system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if SKIP_TEST:
            return
            
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = cls.temp_dir.name
        
        # Create mock database manager
        cls.db_manager = MockDBManager()
        
        # Create result aggregator
        cls.result_aggregator = ResultAggregatorService(db_manager=cls.db_manager)
        
        # Configure result aggregator
        cls.result_aggregator.configure({
            "cache_ttl_seconds": 10,
            "anomaly_threshold": 2.0,
            "comparative_lookback_days": 7,
            "database_enabled": True,
            "normalize_metrics": True,
            "deduplication_enabled": True,
            "model_family_grouping": False
        })
        
        # Create visualization engine
        cls.visualization_engine = VisualizationEngine(
            result_aggregator=cls.result_aggregator,
            output_dir=os.path.join(cls.output_dir, "visualizations")
        )
        
        # Configure visualization engine
        cls.visualization_engine.configure({
            "theme": "light",
            "interactive": False,
            "static_format": "png",
            "width": 800,
            "height": 600,
            "dpi": 100,
            "include_annotations": True
        })
        
        # Create dashboard generator
        cls.dashboard_generator = DashboardGenerator(
            result_aggregator=cls.result_aggregator,
            output_dir=cls.output_dir
        )
        
        # Configure dashboard generator
        cls.dashboard_generator.configure({
            "theme": "light",
            "refresh_interval": 0,
            "include_performance_charts": True,
            "include_regression_detection": True,
            "include_dimension_analysis": True,
            "include_test_details": True,
            "include_worker_details": True,
            "max_items_per_section": 10
        })
        
        # Create dashboard server
        cls.dashboard_server = DashboardServer(
            host="localhost",
            port=8081,
            result_aggregator=cls.result_aggregator,
            output_dir=cls.output_dir
        )
        
        # Configure dashboard server
        cls.dashboard_server.configure({
            "auto_refresh": 0,
            "theme": "light",
            "max_items_per_page": 10,
            "default_report_type": "performance",
            "api_cache_time": 1
        })
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_create_visualization(self):
        """Test creating a visualization."""
        # Get time series data from result aggregator
        result = self.result_aggregator.aggregate_results(
            result_type="performance",
            aggregation_level="model",
            filter_params={"hardware_id": "gpu1"}
        )
        
        # Extract time series data for visualization
        time_series_data = {}
        if "results" in result and "basic_statistics" in result["results"]:
            for model_id, stats in result["results"]["basic_statistics"].items():
                if "throughput_items_per_second" in stats:
                    time_series_data[model_id] = [(datetime.now() - timedelta(days=i), 
                                                stats["throughput_items_per_second"]["mean"] * (1 - i*0.01)) 
                                               for i in range(7)]
        
        # Create visualization
        viz_path = self.visualization_engine.create_visualization(
            "time_series",
            {
                "time_series": time_series_data,
                "metric": "throughput_items_per_second",
                "title": "Model Throughput on GPU1"
            }
        )
        
        # Verify visualization was created
        self.assertIsNotNone(viz_path)
        self.assertTrue(os.path.exists(viz_path))
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_generate_dashboard(self):
        """Test generating a dashboard."""
        # Generate dashboard
        dashboard_path = self.dashboard_generator.generate_dashboard()
        
        # Verify dashboard was created
        self.assertIsNotNone(dashboard_path)
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Verify dashboard content
        with open(dashboard_path, "r") as f:
            dashboard_content = f.read()
            
        # Check for key dashboard elements
        self.assertIn("Performance Trends", dashboard_content)
        self.assertIn("Dimension Analysis", dashboard_content)
        self.assertIn("Regression Detection", dashboard_content)
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_generate_regression_report(self):
        """Test generating a regression report."""
        # Generate regression report
        report_path = self.dashboard_generator.generate_report("regression")
        
        # Verify report was created
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path))
        
        # Verify report content
        with open(report_path, "r") as f:
            report_content = f.read()
            
        # Check for key report elements
        self.assertIn("Performance Regression Report", report_content)
        self.assertIn("bert-base-uncased", report_content)
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_api_endpoints(self):
        """Test dashboard server API endpoints."""
        # Create event loop for async tests
        loop = # TODO: Remove event loop management - asyncio.new_event_loop()
        # TODO: Remove event loop management - asyncio.set_event_loop(loop)
        
        try:
            # Test API status endpoint
            status_response = loop.run_until_complete(self.dashboard_server.handle_api_status(None))
            status_data = json.loads(status_response.text)
            
            # Verify status response
            self.assertEqual(status_data["status"], "ok")
            self.assertIn("data", status_data)
            self.assertIn("api_version", status_data)
            
            # Test API tests endpoint
            tests_response = loop.run_until_complete(self.dashboard_server.handle_api_tests(None))
            tests_data = json.loads(tests_response.text)
            
            # Verify tests response
            self.assertEqual(tests_data["status"], "ok")
            self.assertIn("data", tests_data)
            self.assertIn("tests", tests_data["data"])
            
        finally:
            loop.close()
    
    @unittest.skipIf(SKIP_TEST, "Missing dependencies")
    def test_result_aggregator_integration(self):
        """Test integration with result aggregator."""
        # Get aggregated results
        results = self.result_aggregator.aggregate_results(
            result_type="performance",
            aggregation_level="hardware",
            filter_params={"model_id": "bert-base-uncased"}
        )
        
        # Verify results
        self.assertIn("results", results)
        self.assertIn("basic_statistics", results["results"])
        
        # Check for anomalies
        anomalies = self.result_aggregator.get_result_anomalies(
            result_type="performance",
            aggregation_level="model_hardware"
        )
        
        # Verify anomalies
        self.assertIn("anomalies", anomalies)
        
        # Get comparison report
        comparison = self.result_aggregator.get_comparison_report(
            result_type="performance",
            aggregation_level="model"
        )
        
        # Verify comparison
        self.assertIn("comparisons", comparison)


if __name__ == "__main__":
    unittest.main()