#!/usr/bin/env python3
"""
Test script for the Integrated Visualization and Reports System

This script tests the functionality of the integrated_visualization_reports.py script,
validating its ability to launch the visualization dashboard and generate reports.

Usage:
    python test_integrated_visualization_reports.py
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call
import tempfile
import subprocess
import argparse

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import module to test
import integrated_visualization_reports
from integrated_visualization_reports import IntegratedSystem


class TestIntegratedVisualizationReports(unittest.TestCase):
    """Test the Integrated Visualization and Reports System."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".duckdb")
        
        # Create mock arguments
        self.args = MagicMock()
        self.args.dashboard = False
        self.args.reports = False
        self.args.dashboard_export = False
        self.args.db_path = self.temp_db.name
        self.args.output_dir = self.temp_dir
        self.args.dashboard_port = 8050
        self.args.dashboard_host = "localhost"
        self.args.open_browser = False
        self.args.debug = False
        self.args.format = "html"
        self.args.simulation_validation = False
        self.args.cross_hardware_comparison = False
        self.args.combined_report = False
        self.args.historical = False
        self.args.days = 30
        self.args.badge_only = False
        self.args.ci = False
        self.args.github_pages = False
        self.args.export_metrics = False
        self.args.highlight_simulation = False
        self.args.tolerance = None
        self.args.include_visualizations = True
        self.args.visualization_format = "png"
        self.args.verbose = False
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary files
        self.temp_db.close()
        
        # Clean up temporary directory if it exists
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    @patch("subprocess.Popen")
    def test_start_dashboard(self, mock_popen):
        """Test starting the visualization dashboard."""
        # Set up mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdout.readline.return_value = "Dash is running on http://localhost:8050/"
        mock_popen.return_value = mock_process
        
        # Create system and start dashboard
        system = IntegratedSystem(self.args)
        result = system.start_dashboard(wait_for_startup=True)
        
        # Verify process was started with correct command
        self.assertEqual(result, mock_process)
        mock_popen.assert_called_once()
        # Extract args from the call
        args, kwargs = mock_popen.call_args
        # First argument should be the command
        cmd = args[0]
        # Command should contain visualization_dashboard.py
        self.assertTrue(any("visualization_dashboard.py" in arg for arg in cmd))
        # Command should have correct port and database path
        self.assertIn("--port", cmd)
        self.assertIn("8050", cmd)
        self.assertIn("--db-path", cmd)
        self.assertIn(self.temp_db.name, cmd)
    
    @patch("subprocess.run")
    def test_generate_reports(self, mock_run):
        """Test generating reports."""
        # Set up mock process result
        mock_result = MagicMock()
        mock_result.stdout = "Reports generated:\n- main: /path/to/report.html"
        mock_run.return_value = mock_result
        
        # Create system and generate reports
        system = IntegratedSystem(self.args)
        result = system.generate_reports()
        
        # Verify subprocess.run was called with correct command
        mock_run.assert_called_once()
        # Extract args from the call
        args, kwargs = mock_run.call_args
        # First argument should be the command
        cmd = args[0]
        # Command should contain enhanced_ci_cd_reports.py
        self.assertTrue(any("enhanced_ci_cd_reports.py" in arg for arg in cmd))
        # Command should have correct output directory and database path
        self.assertIn("--output-dir", cmd)
        self.assertIn(self.temp_dir, cmd)
        self.assertIn("--db-path", cmd)
        self.assertIn(self.temp_db.name, cmd)
        
        # Verify result contains the parsed report path
        self.assertEqual(result, {"main": "/path/to/report.html"})
    
    def test_export_dashboard_visualizations(self):
        """Test exporting dashboard visualizations."""
        # Create system and export visualizations
        system = IntegratedSystem(self.args)
        result = system.export_dashboard_visualizations()
        
        # Verify result contains the exported file path
        self.assertIn("index_html", result)
        
        # Verify the exported file exists
        export_dir = os.path.join(self.temp_dir, "dashboard_export")
        index_path = os.path.join(export_dir, "index.html")
        self.assertTrue(os.path.exists(index_path))
        
        # Verify the file contains expected content
        with open(index_path, 'r') as f:
            content = f.read()
            self.assertIn("Dashboard Export", content)
            self.assertIn("This is a static export of the visualization dashboard", content)
            self.assertIn("Overview", content)
            self.assertIn("Performance Analysis", content)
            self.assertIn("Hardware Comparison", content)
            self.assertIn("Time Series Analysis", content)
            self.assertIn("Simulation Validation", content)
    
    @patch("integrated_visualization_reports.IntegratedSystem.start_dashboard")
    @patch("integrated_visualization_reports.IntegratedSystem.generate_reports")
    def test_run_dashboard_and_reports(self, mock_generate_reports, mock_start_dashboard):
        """Test running the system with both dashboard and reports."""
        # Set up mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_start_dashboard.return_value = mock_process
        
        # Set up mock reports result
        mock_generate_reports.return_value = {"main": "/path/to/report.html"}
        
        # Configure args to run both dashboard and reports
        self.args.dashboard = True
        self.args.reports = True
        
        # Create system and run
        system = IntegratedSystem(self.args)
        system.run()
        
        # Verify dashboard was started and reports were generated
        mock_start_dashboard.assert_called_once()
        mock_generate_reports.assert_called_once()
    
    @patch("argparse.ArgumentParser.parse_args")
    def test_parse_arguments(self, mock_parse_args):
        """Test parsing command-line arguments."""
        # Set up mock args
        mock_args = MagicMock()
        mock_args.dashboard = True
        mock_args.reports = False
        mock_args.dashboard_export = False
        mock_parse_args.return_value = mock_args
        
        # Call parse_arguments
        args = integrated_visualization_reports.parse_arguments()
        
        # Verify the result matches the mock args
        self.assertEqual(args, mock_args)
        mock_parse_args.assert_called_once()
    
    @patch("integrated_visualization_reports.IntegratedSystem")
    @patch("integrated_visualization_reports.parse_arguments")
    def test_main(self, mock_parse_args, mock_system):
        """Test the main function."""
        # Set up mock args
        mock_args = MagicMock()
        mock_args.dashboard = True
        mock_args.reports = False
        mock_args.dashboard_export = False
        mock_parse_args.return_value = mock_args
        
        # Set up mock system
        mock_system_instance = MagicMock()
        mock_system.return_value = mock_system_instance
        
        # Call main
        result = integrated_visualization_reports.main()
        
        # Verify the result and method calls
        self.assertEqual(result, 0)
        mock_parse_args.assert_called_once()
        mock_system.assert_called_once_with(mock_args)
        mock_system_instance.run.assert_called_once()
    
    @patch("integrated_visualization_reports.parse_arguments")
    def test_main_no_action(self, mock_parse_args):
        """Test the main function when no action is specified."""
        # Set up mock args with no action
        mock_args = MagicMock()
        mock_args.dashboard = False
        mock_args.reports = False
        mock_args.dashboard_export = False
        mock_parse_args.return_value = mock_args
        
        # Call main
        result = integrated_visualization_reports.main()
        
        # Verify the result (should be 1 for error)
        self.assertEqual(result, 1)
        mock_parse_args.assert_called_once()


if __name__ == "__main__":
    unittest.main()