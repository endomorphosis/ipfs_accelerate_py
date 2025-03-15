#!/usr/bin/env python3
"""
Tests for the Validation Dashboard component.

This module contains unit tests for the ValidationDashboard class.
"""

import os
import sys
import unittest
import tempfile
from unittest.mock import MagicMock, patch
import datetime
import uuid
import random
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import the components to test
from duckdb_api.benchmark_validation.visualization.dashboard import ValidationDashboard

# Import base classes
from duckdb_api.benchmark_validation.core.base import (
    ValidationResult,
    BenchmarkResult,
    ValidationLevel,
    ValidationStatus,
    BenchmarkType,
    ValidationReporter
)

class TestValidationDashboard(unittest.TestCase):
    """Test cases for ValidationDashboard."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary output directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Configure dashboard
        self.config = {
            "output_directory": self.output_dir,
            "dashboard_directory": "dashboards",
            "monitoring_integration": False,
            "theme": "light"
        }
        
        # Create dashboard instance
        self.dashboard = ValidationDashboard(config=self.config)
        
        # Create sample validation results
        self.validation_results = self.create_sample_validation_results(count=10)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def create_sample_validation_results(self, count: int = 10) -> List[ValidationResult]:
        """Create sample validation results for testing."""
        validation_results = []
        
        model_ids = ["model-A", "model-B"]
        hardware_ids = ["hardware-X", "hardware-Y"]
        benchmark_types = [BenchmarkType.LATENCY, BenchmarkType.THROUGHPUT]
        validation_levels = [ValidationLevel.BASIC, ValidationLevel.ADVANCED]
        statuses = [ValidationStatus.VALID, ValidationStatus.INVALID, ValidationStatus.WARNING]
        
        for i in range(count):
            benchmark_result = BenchmarkResult(
                id=f"bench-{i}",
                model_id=random.choice(model_ids),
                hardware_id=random.choice(hardware_ids),
                benchmark_type=random.choice(benchmark_types),
                metrics={"latency_ms": random.uniform(10, 100)},
                metadata={"batch_size": 1}
            )
            
            validation_result = ValidationResult(
                id=f"val-{i}",
                benchmark_result=benchmark_result,
                validation_level=random.choice(validation_levels),
                status=random.choice(statuses),
                confidence_score=random.uniform(0.1, 1.0),
                issues=[{"description": f"Test issue {i}"}] if i % 2 == 0 else [],
                recommendations=[{"description": f"Test recommendation {i}"}] if i % 3 == 0 else []
            )
            
            validation_results.append(validation_result)
        
        return validation_results
    
    def test_initialization(self):
        """Test dashboard initialization."""
        # Check that dashboard was initialized properly
        self.assertIsNotNone(self.dashboard)
        self.assertEqual(self.dashboard.output_dir, self.output_dir)
        self.assertEqual(self.dashboard.dashboard_dir, os.path.join(self.output_dir, "dashboards"))
        
        # Verify default config values applied
        self.assertEqual(self.dashboard.config["theme"], "light")
        self.assertEqual(self.dashboard.config["monitoring_integration"], False)
    
    def test_create_dashboard_basic(self):
        """Test creating a basic dashboard."""
        # Mock advanced visualization components as unavailable
        with patch('duckdb_api.benchmark_validation.visualization.dashboard.ADVANCED_VIZ_AVAILABLE', False):
            # Create a new dashboard instance
            dashboard = ValidationDashboard(config=self.config)
            
            # Create basic dashboard
            dashboard_name = "test_basic_dashboard"
            dashboard_path = dashboard.create_dashboard(
                validation_results=self.validation_results,
                dashboard_name=dashboard_name
            )
            
            # Check that dashboard was created
            self.assertIsNotNone(dashboard_path)
            expected_path = os.path.join(self.output_dir, "dashboards", f"{dashboard_name}.html")
            self.assertEqual(dashboard_path, expected_path)
            
            # Verify dashboard file exists
            self.assertTrue(os.path.exists(dashboard_path))
            
            # Check that validation results were cached
            self.assertIn(dashboard_name, dashboard.validation_results_cache)
            self.assertEqual(
                len(dashboard.validation_results_cache[dashboard_name]), 
                len(self.validation_results)
            )
    
    @patch('duckdb_api.benchmark_validation.visualization.dashboard.ADVANCED_VIZ_AVAILABLE', True)
    @patch('duckdb_api.benchmark_validation.visualization.dashboard.CustomizableDashboard')
    def test_create_dashboard_advanced(self, mock_customizable_dashboard):
        """Test creating an advanced dashboard."""
        # Mock CustomizableDashboard
        mock_dashboard_instance = MagicMock()
        mock_dashboard_instance.create_dashboard.return_value = os.path.join(
            self.output_dir, "dashboards", "test_advanced", "dashboard.html"
        )
        mock_customizable_dashboard.return_value = mock_dashboard_instance
        
        # Create a new dashboard instance
        dashboard = ValidationDashboard(config=self.config)
        self.assertIsNotNone(dashboard.dashboard_instance)
        
        # Create advanced dashboard
        dashboard_name = "test_advanced_dashboard"
        dashboard_path = dashboard.create_dashboard(
            validation_results=self.validation_results,
            dashboard_name=dashboard_name,
            dashboard_title="Test Advanced Dashboard",
            dashboard_description="Test description"
        )
        
        # Check that dashboard was created
        self.assertIsNotNone(dashboard_path)
        self.assertEqual(
            dashboard_path, 
            os.path.join(self.output_dir, "dashboards", "test_advanced", "dashboard.html")
        )
        
        # Verify CustomizableDashboard.create_dashboard was called
        mock_dashboard_instance.create_dashboard.assert_called_once()
        
        # Verify arguments
        args, kwargs = mock_dashboard_instance.create_dashboard.call_args
        self.assertEqual(kwargs.get("dashboard_name"), dashboard_name)
        self.assertEqual(kwargs.get("title"), "Test Advanced Dashboard")
        self.assertEqual(kwargs.get("description"), "Test description")
        self.assertIsNotNone(kwargs.get("components"))
    
    def test_list_dashboards(self):
        """Test listing dashboards."""
        # Create two dashboards
        dashboard_names = ["test_list_dashboard_1", "test_list_dashboard_2"]
        
        for name in dashboard_names:
            # Create dashboard file
            dashboard_dir = os.path.join(self.output_dir, "dashboards")
            os.makedirs(dashboard_dir, exist_ok=True)
            dashboard_path = os.path.join(dashboard_dir, f"{name}.html")
            
            with open(dashboard_path, 'w') as f:
                f.write("<html><body>Test Dashboard</body></html>")
        
        # List dashboards
        dashboards = self.dashboard.list_dashboards()
        
        # Verify all dashboards were found
        self.assertEqual(len(dashboards), len(dashboard_names))
        
        # Verify dashboard details
        found_names = [d.get("name") for d in dashboards]
        for name in dashboard_names:
            self.assertIn(name, found_names)
    
    def test_export_dashboard(self):
        """Test exporting a dashboard."""
        # Create a dashboard first
        dashboard_name = "test_export_dashboard"
        
        # Cache validation results
        self.dashboard.validation_results_cache[dashboard_name] = self.validation_results
        
        # Export dashboard
        export_formats = ["html", "markdown", "json"]
        
        for format in export_formats:
            output_path = self.dashboard.export_dashboard(
                dashboard_name=dashboard_name,
                export_format=format
            )
            
            # Verify export file exists
            self.assertIsNotNone(output_path)
            self.assertTrue(os.path.exists(output_path))
            
            # Verify file extension
            extension = "md" if format == "markdown" else format
            self.assertTrue(output_path.endswith(f".{extension}"))
    
    def test_delete_dashboard(self):
        """Test deleting a dashboard."""
        # Create a dashboard first
        dashboard_name = "test_delete_dashboard"
        dashboard_path = os.path.join(self.output_dir, "dashboards", f"{dashboard_name}.html")
        
        # Create directory and file
        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
        with open(dashboard_path, 'w') as f:
            f.write("<html><body>Test Dashboard</body></html>")
        
        # Cache validation results
        self.dashboard.validation_results_cache[dashboard_name] = self.validation_results
        
        # Delete dashboard
        success = self.dashboard.delete_dashboard(dashboard_name)
        
        # Verify deletion was successful
        self.assertTrue(success)
        self.assertFalse(os.path.exists(dashboard_path))
        self.assertNotIn(dashboard_name, self.dashboard.validation_results_cache)
    
    @patch('duckdb_api.benchmark_validation.visualization.dashboard.MONITORING_DASHBOARD_AVAILABLE', True)
    @patch('duckdb_api.benchmark_validation.visualization.dashboard.VisualizationDashboardIntegration')
    def test_register_with_monitoring(self, mock_visualization_integration):
        """Test registering a dashboard with monitoring system."""
        # Mock monitoring integration
        mock_integration_instance = MagicMock()
        mock_integration_instance.create_embedded_dashboard.return_value = {
            "name": "test_register_dashboard",
            "page": "validation"
        }
        mock_visualization_integration.return_value = mock_integration_instance
        
        # Create dashboard with monitoring integration
        dashboard = ValidationDashboard(config={
            **self.config,
            "monitoring_integration": True
        })
        
        # Create list_dashboards mock to return our test dashboard
        dashboard.list_dashboards = MagicMock(return_value=[
            {
                "name": "test_register_dashboard",
                "title": "Test Register Dashboard",
                "description": "Test description"
            }
        ])
        
        # Register dashboard
        success = dashboard.register_with_monitoring_dashboard(
            dashboard_name="test_register_dashboard",
            page="validation",
            position="below"
        )
        
        # Verify registration was successful
        self.assertTrue(success)
        
        # Verify create_embedded_dashboard was called
        mock_integration_instance.create_embedded_dashboard.assert_called_once()
        
        # Verify arguments
        args, kwargs = mock_integration_instance.create_embedded_dashboard.call_args
        self.assertEqual(kwargs.get("name"), "test_register_dashboard")
        self.assertEqual(kwargs.get("page"), "validation")
        self.assertEqual(kwargs.get("position"), "below")

if __name__ == '__main__':
    unittest.main()