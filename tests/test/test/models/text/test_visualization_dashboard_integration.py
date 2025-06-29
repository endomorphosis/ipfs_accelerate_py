#!/usr/bin/env python3
"""
Integration Test for Advanced Visualization System and Monitoring Dashboard Integration.

This test validates the integration between the Advanced Visualization System's
Customizable Dashboard feature and the Distributed Testing Framework's Monitoring Dashboard.
"""

import os
import sys
import json
import unittest
import tempfile
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_visualization_dashboard_integration")

# Add parent directory to path for module imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import components to test
try:
    # Import visualization dashboard integration
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import (
        VisualizationDashboardIntegration
    )
    
    # Import monitoring dashboard
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard
    
    # Import visualization components
    from duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard import CustomizableDashboard
    
    # Check if test should be skipped due to missing dependencies
    SKIP_TEST = False
except ImportError as e:
    logger.warning(f"Import error: {e}")
    logger.warning("Some components are not available. Integration tests will be skipped.")
    SKIP_TEST = True


class MockCustomizableDashboard:
    """Mock implementation of CustomizableDashboard for testing."""
    
    def __init__(self, db_connection=None, output_dir="./mock_dashboard"):
        """Initialize with mock data."""
        self.output_dir = output_dir
        self.dashboards = {}
        self.templates = {
            "overview": {
                "title": "Overview Dashboard",
                "description": "Provides an overview of system performance",
                "grid_layout": {
                    "rows": 2,
                    "columns": 2
                }
            },
            "hardware_comparison": {
                "title": "Hardware Comparison Dashboard",
                "description": "Compares performance across different hardware",
                "grid_layout": {
                    "rows": 3,
                    "columns": 2
                }
            },
            "model_analysis": {
                "title": "Model Analysis Dashboard",
                "description": "Analyzes performance of specific models",
                "grid_layout": {
                    "rows": 2,
                    "columns": 3
                }
            },
            "empty": {
                "title": "Empty Dashboard",
                "description": "Empty dashboard template",
                "grid_layout": {
                    "rows": 4,
                    "columns": 4
                }
            }
        }
        self.components = {
            "3d": "3D visualization component",
            "heatmap": "Heatmap visualization component",
            "time-series": "Time series visualization component",
            "animated-time-series": "Animated time series visualization component"
        }
        self.title = "Mock Dashboard"
        self.description = "A mock dashboard for testing"
        self.dashboard_config = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_dashboard(self, dashboard_name, template="overview", title=None, 
                        description=None, components=None):
        """Create a mock dashboard."""
        dashboard_dir = os.path.join(self.output_dir, dashboard_name)
        os.makedirs(dashboard_dir, exist_ok=True)
        
        dashboard_html_path = os.path.join(dashboard_dir, "dashboard.html")
        
        # Generate simple HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title or f"Dashboard: {dashboard_name}"}</title>
        </head>
        <body>
            <h1>{title or f"Dashboard: {dashboard_name}"}</h1>
            <p>{description or "Mock dashboard for testing"}</p>
            <div class="dashboard-content">
                <p>Template: {template}</p>
                <p>Components: {len(components) if components else 0}</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(dashboard_html_path, "w") as f:
            f.write(html_content)
        
        # Store dashboard info
        self.dashboards[dashboard_name] = {
            "path": dashboard_html_path,
            "template": template,
            "title": title or f"Dashboard: {dashboard_name}",
            "description": description or "Mock dashboard for testing",
            "components": components or []
        }
        
        # Update instance attributes for the most recent dashboard
        self.title = title or f"Dashboard: {dashboard_name}"
        self.description = description or "Mock dashboard for testing"
        
        return dashboard_html_path
    
    def update_dashboard(self, dashboard_name, title=None, description=None):
        """Update a mock dashboard."""
        if dashboard_name not in self.dashboards:
            return None
        
        dashboard_info = self.dashboards[dashboard_name]
        dashboard_path = dashboard_info["path"]
        
        if title is not None:
            dashboard_info["title"] = title
            self.title = title
        
        if description is not None:
            dashboard_info["description"] = description
            self.description = description
        
        # Update timestamp
        self.dashboard_config["updated_at"] = datetime.now().isoformat()
        
        # Update HTML file
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard_info["title"]}</title>
        </head>
        <body>
            <h1>{dashboard_info["title"]}</h1>
            <p>{dashboard_info["description"]}</p>
            <div class="dashboard-content">
                <p>Template: {dashboard_info["template"]}</p>
                <p>Components: {len(dashboard_info["components"])}</p>
                <p>Updated at: {self.dashboard_config["updated_at"]}</p>
            </div>
        </body>
        </html>
        """
        
        with open(dashboard_path, "w") as f:
            f.write(html_content)
        
        return dashboard_path
    
    def list_available_templates(self):
        """List available templates."""
        return self.templates
    
    def list_available_components(self):
        """List available components."""
        return self.components
    
    def export_dashboard(self, dashboard_name, format, output_path):
        """Export a dashboard."""
        if dashboard_name not in self.dashboards:
            return None
        
        # For testing, just write a simple file
        with open(output_path, "w") as f:
            f.write(f"Exported dashboard: {dashboard_name} in {format} format")
        
        return output_path


class TestVisualizationDashboardIntegration(unittest.TestCase):
    """Test cases for the Visualization Dashboard Integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if SKIP_TEST:
            return
        
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = cls.temp_dir.name
        
        # Create dashboard and integration directories
        cls.dashboard_dir = os.path.join(cls.output_dir, "dashboards")
        cls.integration_dir = os.path.join(cls.output_dir, "integration")
        cls.static_dir = os.path.join(cls.output_dir, "static")
        
        os.makedirs(cls.dashboard_dir, exist_ok=True)
        os.makedirs(cls.integration_dir, exist_ok=True)
        os.makedirs(cls.static_dir, exist_ok=True)
        os.makedirs(os.path.join(cls.static_dir, "dashboards"), exist_ok=True)
        
        # Create symbolic link for dashboard directory
        try:
            if os.path.exists(os.path.join(cls.static_dir, "dashboards")):
                os.symlink(cls.dashboard_dir, os.path.join(cls.static_dir, "dashboards"), target_is_directory=True)
        except Exception as e:
            logger.warning(f"Could not create symbolic link: {e}")
        
        # Setup mock CustomizableDashboard if needed
        try:
            # Try to import the real CustomizableDashboard
            from duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard import CustomizableDashboard
            # Test if we can instantiate it
            test_dashboard = CustomizableDashboard(db_connection=None, output_dir=cls.dashboard_dir)
            cls.mock_dashboard = False
        except (ImportError, Exception) as e:
            logger.warning(f"Using mock CustomizableDashboard due to: {e}")
            # Use the mock implementation
            cls.mock_dashboard = True
        
        # Patch the import if needed
        if cls.mock_dashboard:
            # Store original import
            if "duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard" in sys.modules:
                cls.original_viz_customizable_dashboard = sys.modules["duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard"]
            else:
                cls.original_viz_customizable_dashboard = None
                
            # Create a mock module with MockCustomizableDashboard
            import types
            mock_module = types.ModuleType("viz_customizable_dashboard")
            mock_module.CustomizableDashboard = MockCustomizableDashboard
            sys.modules["duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard"] = mock_module
            
            # Re-import the visualization dashboard integration
            from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import VisualizationDashboardIntegration
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
        
        # Restore original import if patched
        if hasattr(cls, 'mock_dashboard') and cls.mock_dashboard:
            if cls.original_viz_customizable_dashboard is not None:
                sys.modules["duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard"] = cls.original_viz_customizable_dashboard
            else:
                del sys.modules["duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard"]
    
    def setUp(self):
        """Set up each test."""
        if SKIP_TEST:
            self.skipTest("Missing dependencies for integration test")
        
        # Create test instance
        self.integration = VisualizationDashboardIntegration(
            dashboard_dir=self.dashboard_dir,
            integration_dir=self.integration_dir
        )
    
    def test_initialization(self):
        """Test initialization of integration components."""
        self.assertTrue(hasattr(self.integration, 'dashboard_instance'))
        self.assertTrue(hasattr(self.integration, 'visualization_available'))
        self.assertTrue(hasattr(self.integration, 'embedded_dashboards'))
        self.assertEqual(self.integration.dashboard_dir, self.dashboard_dir)
        self.assertEqual(self.integration.integration_dir, self.integration_dir)
        self.assertTrue(os.path.exists(self.integration.registry_file))
    
    def test_create_embedded_dashboard(self):
        """Test creating an embedded dashboard."""
        # Create a simple dashboard
        dashboard_details = self.integration.create_embedded_dashboard(
            name="test_dashboard_1",
            page="overview",
            template="hardware_comparison",
            title="Test Hardware Dashboard",
            description="Dashboard for testing hardware performance",
            position="below"
        )
        
        # Verify dashboard details
        self.assertIsNotNone(dashboard_details)
        self.assertEqual(dashboard_details["name"], "test_dashboard_1")
        self.assertEqual(dashboard_details["page"], "overview")
        self.assertEqual(dashboard_details["template"], "hardware_comparison")
        self.assertEqual(dashboard_details["title"], "Test Hardware Dashboard")
        self.assertEqual(dashboard_details["position"], "below")
        
        # Verify dashboard exists in registry
        self.assertIn("test_dashboard_1", self.integration.embedded_dashboards)
        
        # Verify dashboard file exists
        self.assertTrue(os.path.exists(dashboard_details["path"]))
    
    def test_create_dashboard_with_components(self):
        """Test creating a dashboard with components."""
        # Create components
        components = [
            {
                "type": "3d",
                "config": {
                    "metrics": ["throughput", "latency", "memory"],
                    "dimensions": ["model", "hardware"],
                    "title": "3D Performance Metrics"
                },
                "width": 2,
                "height": 1
            },
            {
                "type": "heatmap",
                "config": {
                    "metric": "throughput",
                    "title": "Throughput Comparison"
                },
                "width": 1,
                "height": 1
            }
        ]
        
        # Create dashboard with components
        dashboard_details = self.integration.create_embedded_dashboard(
            name="test_dashboard_2",
            page="results",
            template="empty",
            title="Custom Components Dashboard",
            description="Dashboard with custom components",
            position="tab",
            components=components
        )
        
        # Verify dashboard details
        self.assertIsNotNone(dashboard_details)
        self.assertEqual(dashboard_details["name"], "test_dashboard_2")
        self.assertEqual(dashboard_details["page"], "results")
        self.assertEqual(dashboard_details["template"], "empty")
        self.assertEqual(dashboard_details["title"], "Custom Components Dashboard")
        self.assertEqual(dashboard_details["position"], "tab")
        
        # Verify dashboard exists in registry
        self.assertIn("test_dashboard_2", self.integration.embedded_dashboards)
    
    def test_get_embedded_dashboards_for_page(self):
        """Test getting embedded dashboards for a specific page."""
        # Create two dashboards for the same page
        self.integration.create_embedded_dashboard(
            name="page_dashboard_1",
            page="performance",
            template="overview",
            title="Performance Dashboard 1"
        )
        
        self.integration.create_embedded_dashboard(
            name="page_dashboard_2",
            page="performance",
            template="model_analysis",
            title="Performance Dashboard 2"
        )
        
        # Create one for a different page
        self.integration.create_embedded_dashboard(
            name="other_page_dashboard",
            page="compatibility",
            template="overview",
            title="Compatibility Dashboard"
        )
        
        # Get dashboards for performance page
        performance_dashboards = self.integration.get_embedded_dashboards_for_page("performance")
        
        # Verify correct dashboards returned
        self.assertEqual(len(performance_dashboards), 2)
        self.assertIn("page_dashboard_1", performance_dashboards)
        self.assertIn("page_dashboard_2", performance_dashboards)
        self.assertNotIn("other_page_dashboard", performance_dashboards)
    
    def test_update_embedded_dashboard(self):
        """Test updating an embedded dashboard."""
        # Create a dashboard
        original_details = self.integration.create_embedded_dashboard(
            name="update_test_dashboard",
            page="overview",
            template="overview",
            title="Original Title",
            description="Original description",
            position="below"
        )
        
        # Update dashboard
        updated_details = self.integration.update_embedded_dashboard(
            name="update_test_dashboard",
            title="Updated Title",
            description="Updated description",
            position="tab",
            page="results"
        )
        
        # Verify updates
        self.assertIsNotNone(updated_details)
        self.assertEqual(updated_details["title"], "Updated Title")
        self.assertEqual(updated_details["description"], "Updated description")
        self.assertEqual(updated_details["position"], "tab")
        self.assertEqual(updated_details["page"], "results")
        
        # Verify updates in registry
        registry_details = self.integration.get_embedded_dashboard("update_test_dashboard")
        self.assertEqual(registry_details["title"], "Updated Title")
        self.assertEqual(registry_details["description"], "Updated description")
        self.assertEqual(registry_details["position"], "tab")
        self.assertEqual(registry_details["page"], "results")
    
    def test_remove_embedded_dashboard(self):
        """Test removing an embedded dashboard."""
        # Create a dashboard
        self.integration.create_embedded_dashboard(
            name="remove_test_dashboard",
            page="overview",
            template="overview",
            title="Dashboard to Remove"
        )
        
        # Verify it exists in registry
        self.assertIn("remove_test_dashboard", self.integration.embedded_dashboards)
        
        # Remove dashboard
        removed = self.integration.remove_embedded_dashboard("remove_test_dashboard")
        
        # Verify removal
        self.assertTrue(removed)
        self.assertNotIn("remove_test_dashboard", self.integration.embedded_dashboards)
    
    def test_get_dashboard_iframe_html(self):
        """Test getting HTML iframe code for an embedded dashboard."""
        # Create a dashboard
        dashboard_details = self.integration.create_embedded_dashboard(
            name="iframe_test_dashboard",
            page="overview",
            template="overview",
            title="Dashboard for iframe test"
        )
        
        # Get iframe HTML
        iframe_html = self.integration.get_dashboard_iframe_html("iframe_test_dashboard")
        
        # Verify iframe HTML
        self.assertIn("<iframe", iframe_html)
        self.assertIn("</iframe>", iframe_html)
        self.assertIn("width=\"100%\"", iframe_html)
        self.assertIn("height=\"600px\"", iframe_html)
        
        # Get iframe with custom dimensions
        custom_iframe_html = self.integration.get_dashboard_iframe_html(
            name="iframe_test_dashboard",
            width="800px",
            height="400px"
        )
        
        # Verify custom dimensions
        self.assertIn("width=\"800px\"", custom_iframe_html)
        self.assertIn("height=\"400px\"", custom_iframe_html)
    
    def test_generate_dashboard_from_performance_data(self):
        """Test generating a dashboard from performance data."""
        # Mock performance data
        performance_data = {
            "metrics": {
                "throughput_items_per_second": {
                    "mean": 120.5,
                    "min": 100.2,
                    "max": 140.8
                },
                "average_latency_ms": {
                    "mean": 25.3,
                    "min": 20.1,
                    "max": 30.5
                },
                "memory_peak_mb": {
                    "mean": 1500.0,
                    "min": 1200.0,
                    "max": 1800.0
                }
            },
            "dimensions": {
                "model_family": ["bert", "t5", "vit"],
                "hardware_type": ["gpu", "cpu", "webgpu"]
            },
            "time_series": {
                "available": True,
                "time_points": 90
            }
        }
        
        # Generate dashboard
        dashboard_path = self.integration.generate_dashboard_from_performance_data(
            performance_data=performance_data,
            name="performance_dashboard",
            title="Generated Performance Dashboard"
        )
        
        # Verify dashboard was generated
        self.assertIsNotNone(dashboard_path)
        self.assertTrue(os.path.exists(dashboard_path))
    
    def test_templates_and_components(self):
        """Test listing available templates and components."""
        # Get templates
        templates = self.integration.list_available_templates()
        
        # Verify templates
        self.assertIsInstance(templates, dict)
        self.assertIn("overview", templates)
        self.assertIn("hardware_comparison", templates)
        self.assertIn("model_analysis", templates)
        self.assertIn("empty", templates)
        
        # Get components
        components = self.integration.list_available_components()
        
        # Verify components
        self.assertIsInstance(components, dict)
        self.assertIn("3d", components)
        self.assertIn("heatmap", components)
        self.assertIn("time-series", components)
        self.assertIn("animated-time-series", components)
    
    def test_export_embedded_dashboard(self):
        """Test exporting an embedded dashboard."""
        # Create a dashboard
        self.integration.create_embedded_dashboard(
            name="export_test_dashboard",
            page="overview",
            template="overview",
            title="Dashboard for export test"
        )
        
        # Export as HTML
        html_path = self.integration.export_embedded_dashboard(
            name="export_test_dashboard",
            format="html"
        )
        
        # Verify export
        self.assertIsNotNone(html_path)
        self.assertTrue(os.path.exists(html_path))
        
        # Export as another format
        pdf_path = self.integration.export_embedded_dashboard(
            name="export_test_dashboard",
            format="pdf"
        )
        
        # Verify export
        if pdf_path:  # Not all formats may be supported
            self.assertTrue(os.path.exists(pdf_path))


class TestMonitoringDashboardWithVisualization(unittest.TestCase):
    """Test cases for the Monitoring Dashboard with visualization integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if SKIP_TEST:
            return
        
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = cls.temp_dir.name
        
        # Create required directories
        cls.dashboard_dir = os.path.join(cls.output_dir, "dashboards")
        cls.static_dir = os.path.join(cls.output_dir, "static")
        cls.templates_dir = os.path.join(cls.output_dir, "templates")
        
        os.makedirs(cls.dashboard_dir, exist_ok=True)
        os.makedirs(cls.static_dir, exist_ok=True)
        os.makedirs(cls.templates_dir, exist_ok=True)
        os.makedirs(os.path.join(cls.static_dir, "dashboards"), exist_ok=True)
        
        # Determine if we need to use mock implementations
        cls.use_mocks = False
        
        try:
            # Try to import required components
            from duckdb_api.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard
            from duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard import CustomizableDashboard
            
            # Test if we can instantiate CustomizableDashboard
            test_dashboard = CustomizableDashboard(db_connection=None, output_dir=cls.dashboard_dir)
        except (ImportError, Exception) as e:
            logger.warning(f"Using mock implementations due to: {e}")
            cls.use_mocks = True
        
        # Patch imports if needed
        if cls.use_mocks:
            # Store original imports
            cls.original_modules = {}
            
            if "duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard" in sys.modules:
                cls.original_modules["viz_customizable_dashboard"] = sys.modules["duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard"]
            
            # Create a mock module with MockCustomizableDashboard
            import types
            mock_module = types.ModuleType("viz_customizable_dashboard")
            mock_module.CustomizableDashboard = MockCustomizableDashboard
            sys.modules["duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard"] = mock_module
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
        
        # Restore original imports if patched
        if hasattr(cls, 'use_mocks') and cls.use_mocks:
            for module_name, original_module in cls.original_modules.items():
                sys.modules[f"duckdb_api.visualization.advanced_visualization.{module_name}"] = original_module
    
    def setUp(self):
        """Set up each test."""
        if SKIP_TEST:
            self.skipTest("Missing dependencies for integration test")
        
        # Create mock monitoring dashboard
        self.monitoring_dashboard = MonitoringDashboard(
            host="localhost",
            port=8082,
            static_dir=self.static_dir,
            templates_dir=self.templates_dir,
            enable_visualization_integration=True,
            visualization_dashboard_dir=self.dashboard_dir
        )
    
    def test_monitoring_dashboard_initialization(self):
        """Test initialization of monitoring dashboard with visualization integration."""
        self.assertTrue(hasattr(self.monitoring_dashboard, 'visualization_integration'))
        self.assertIsNotNone(self.monitoring_dashboard.visualization_integration)
        self.assertTrue(self.monitoring_dashboard.enable_visualization_integration)
        self.assertEqual(self.monitoring_dashboard.visualization_dashboard_dir, self.dashboard_dir)
    
    def test_create_default_dashboards(self):
        """Test creating default dashboards for the monitoring dashboard."""
        # Create a method to create default dashboards
        def create_default_dashboards():
            integration = self.monitoring_dashboard.visualization_integration
            
            # Create a performance dashboard
            performance_dashboard = integration.create_embedded_dashboard(
                name="default_performance_dashboard",
                page="results",
                template="overview",
                title="Performance Overview",
                description="Overview of performance metrics",
                position="below"
            )
            
            # Create a hardware comparison dashboard
            hardware_dashboard = integration.create_embedded_dashboard(
                name="default_hardware_dashboard",
                page="results",
                template="hardware_comparison",
                title="Hardware Comparison",
                description="Comparison of different hardware platforms",
                position="below"
            )
            
            return [performance_dashboard, hardware_dashboard]
        
        # Create default dashboards
        dashboards = create_default_dashboards()
        
        # Verify dashboards were created
        self.assertEqual(len(dashboards), 2)
        self.assertIsNotNone(dashboards[0])
        self.assertIsNotNone(dashboards[1])
        
        # Verify they're in the registry
        performance_dashboard = self.monitoring_dashboard.visualization_integration.get_embedded_dashboard(
            "default_performance_dashboard"
        )
        hardware_dashboard = self.monitoring_dashboard.visualization_integration.get_embedded_dashboard(
            "default_hardware_dashboard"
        )
        
        self.assertIsNotNone(performance_dashboard)
        self.assertIsNotNone(hardware_dashboard)
        self.assertEqual(performance_dashboard["page"], "results")
        self.assertEqual(hardware_dashboard["page"], "results")
    
    def test_get_embedded_dashboards_html(self):
        """Test getting HTML for embedded dashboards on a page."""
        # Create a custom method for getting embedded dashboards HTML
        def get_embedded_dashboards_html(page, position="below"):
            integration = self.monitoring_dashboard.visualization_integration
            dashboards = integration.get_embedded_dashboards_for_page(page)
            
            # Filter by position
            position_dashboards = {name: details for name, details in dashboards.items()
                                  if details.get("position") == position}
            
            # Generate HTML
            html = ""
            for name, details in position_dashboards.items():
                html += integration.get_dashboard_iframe_html(name)
            
            return html
        
        # Create test dashboards
        integration = self.monitoring_dashboard.visualization_integration
        integration.create_embedded_dashboard(
            name="page1_dashboard1",
            page="page1",
            template="overview",
            title="Page 1 Dashboard 1",
            position="below"
        )
        integration.create_embedded_dashboard(
            name="page1_dashboard2",
            page="page1",
            template="hardware_comparison",
            title="Page 1 Dashboard 2",
            position="above"
        )
        
        # Get dashboards HTML for page1 with position below
        below_html = get_embedded_dashboards_html("page1", "below")
        
        # Verify HTML
        self.assertIn("<iframe", below_html)
        
        # Get dashboards HTML for page1 with position above
        above_html = get_embedded_dashboards_html("page1", "above")
        
        # Verify HTML
        self.assertIn("<iframe", above_html)
        
        # Get dashboards HTML for non-existent page
        none_html = get_embedded_dashboards_html("non_existent_page")
        
        # Verify empty HTML
        self.assertEqual(none_html, "")


def main():
    """Run tests."""
    unittest.main()


if __name__ == "__main__":
    main()