#!/usr/bin/env python3
"""
Integrated Visualization and Reports System

This script provides a single entry point for both the Visualization Dashboard
and the Enhanced CI/CD Reports Generator, allowing users to launch both
with a consistent configuration and database connection.

Features:
- Unified command-line interface for dashboard and reports
- Consistent database access across all components
- Report generation based on live dashboard data
- Easy-to-use commands for common scenarios
- Support for both interactive exploration and CI/CD integration

Usage:
    # Start the visualization dashboard
    python integrated_visualization_reports.py --dashboard
    
    # Generate CI/CD reports
    python integrated_visualization_reports.py --reports
    
    # Do both: start dashboard and generate reports
    python integrated_visualization_reports.py --dashboard --reports
    
    # Use a specific database
    python integrated_visualization_reports.py --dashboard --db-path ./my_benchmark_db.duckdb
    
    # Generate specific report types
    python integrated_visualization_reports.py --reports --simulation-validation
    
    # Generate a full dashboard export with all visualizations
    python integrated_visualization_reports.py --dashboard-export
"""

import os
import sys
import argparse
import logging
import subprocess
import threading
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Constants
DEFAULT_DB_PATH = os.environ.get("BENCHMARK_DB_PATH", os.path.join(test_dir, "test_template_db.duckdb"))
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(script_dir), "reports")
DEFAULT_DASHBOARD_PORT = 8050
DEFAULT_DASHBOARD_HOST = "localhost"


class IntegratedSystem:
    """
    Integrated Visualization and Reports System that combines the Visualization Dashboard
    and Enhanced CI/CD Reports Generator.
    """
    
    def __init__(self, args):
        """
        Initialize the integrated system with command-line arguments.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.db_path = args.db_path or DEFAULT_DB_PATH
        self.output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        self.dashboard_port = args.dashboard_port or DEFAULT_DASHBOARD_PORT
        self.dashboard_host = args.dashboard_host or DEFAULT_DASHBOARD_HOST
        self.dashboard_process = None
        self.open_browser = args.open_browser
        self.dashboard_only = args.dashboard and not args.reports and not args.dashboard_export
        self.reports_only = args.reports and not args.dashboard and not args.dashboard_export
        self.export_only = args.dashboard_export and not args.dashboard and not args.reports
        self.do_both = (args.dashboard and args.reports) or (args.dashboard and args.dashboard_export)
        
        # Set log level based on verbosity
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start_dashboard(self, wait_for_startup: bool = True) -> Optional[subprocess.Popen]:
        """
        Start the visualization dashboard as a separate process.
        
        Args:
            wait_for_startup: Whether to wait for dashboard startup before returning
            
        Returns:
            Dashboard process object
        """
        logger.info(f"Starting visualization dashboard at http://{self.dashboard_host}:{self.dashboard_port}")
        
        # Build command for launching dashboard
        cmd = [
            sys.executable,
            os.path.join(script_dir, "visualization_dashboard.py"),
            "--port", str(self.dashboard_port),
            "--host", self.dashboard_host,
            "--db-path", self.db_path
        ]
        
        if self.args.debug:
            cmd.append("--debug")
        
        # Start dashboard process
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Wait for dashboard to start if requested
            if wait_for_startup:
                success = self._wait_for_dashboard_startup(process)
                if not success:
                    logger.error("Failed to start visualization dashboard")
                    return None
            
            # Open browser if requested
            if self.open_browser:
                self._open_dashboard_in_browser()
            
            return process
            
        except Exception as e:
            logger.error(f"Error starting visualization dashboard: {str(e)}")
            return None
    
    def _wait_for_dashboard_startup(self, process: subprocess.Popen, timeout: int = 10) -> bool:
        """
        Wait for the dashboard process to start and become ready.
        
        Args:
            process: Dashboard process
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if dashboard started successfully, False otherwise
        """
        logger.info("Waiting for dashboard to start...")
        start_time = time.time()
        
        # Read process output to look for startup message
        while process.poll() is None and time.time() - start_time < timeout:
            line = process.stdout.readline().strip()
            if line:
                print(f"  {line}")
                if "Dash is running on" in line:
                    logger.info("Dashboard started successfully")
                    return True
            time.sleep(0.1)
        
        # Check if process exited
        if process.poll() is not None:
            logger.error(f"Dashboard process exited with code {process.returncode}")
            # Print any remaining output
            for line in process.stdout:
                print(f"  {line.strip()}")
            return False
        
        # Timeout reached
        logger.warning(f"Timeout waiting for dashboard to start (waited {timeout} seconds)")
        return False
    
    def _open_dashboard_in_browser(self):
        """Open the dashboard in a web browser."""
        url = f"http://{self.dashboard_host}:{self.dashboard_port}"
        try:
            logger.info(f"Opening dashboard in web browser: {url}")
            webbrowser.open(url)
        except Exception as e:
            logger.error(f"Error opening dashboard in browser: {str(e)}")
    
    def generate_reports(self) -> Dict[str, str]:
        """
        Generate reports using the Enhanced CI/CD Reports Generator.
        
        Returns:
            Dictionary mapping report names to file paths
        """
        logger.info("Generating reports...")
        
        # Build command for generating reports
        cmd = [
            sys.executable,
            os.path.join(script_dir, "enhanced_ci_cd_reports.py"),
            "--output-dir", self.output_dir,
            "--db-path", self.db_path
        ]
        
        # Add report format
        if self.args.format:
            cmd.extend(["--format", self.args.format])
        
        # Add report type options
        if self.args.simulation_validation:
            cmd.append("--simulation-validation")
        
        if self.args.cross_hardware_comparison:
            cmd.append("--cross-hardware-comparison")
        
        if self.args.combined_report:
            cmd.append("--combined-report")
        
        # Add other options
        if self.args.historical:
            cmd.append("--historical")
            if self.args.days:
                cmd.extend(["--days", str(self.args.days)])
        
        if self.args.badge_only:
            cmd.append("--badge-only")
        
        if self.args.ci:
            cmd.append("--ci")
        
        if self.args.github_pages:
            cmd.append("--github-pages")
        
        if self.args.export_metrics:
            cmd.append("--export-metrics")
        
        if self.args.highlight_simulation:
            cmd.append("--highlight-simulation")
        
        if self.args.tolerance is not None:
            cmd.extend(["--tolerance", str(self.args.tolerance)])
        
        if self.args.include_visualizations:
            cmd.append("--include-visualizations")
            if self.args.visualization_format:
                cmd.extend(["--visualization-format", self.args.visualization_format])
        
        if self.args.verbose:
            cmd.append("--verbose")
        
        # Run the command
        try:
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Parse output to find generated reports
            reports = {}
            for line in result.stdout.splitlines():
                if "Reports generated:" in line:
                    continue
                if line.startswith("- "):
                    try:
                        report_type, path = line[2:].split(": ", 1)
                        reports[report_type] = path
                    except:
                        pass
            
            logger.info(f"Generated {len(reports)} reports in {self.output_dir}")
            return reports
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating reports: {e.returncode}")
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Command error: {e.stderr}")
            return {}
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            return {}
    
    def export_dashboard_visualizations(self) -> Dict[str, str]:
        """
        Export visualizations from the dashboard for offline viewing.
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("Exporting dashboard visualizations...")
        
        # Create output directory for dashboard export
        export_dir = os.path.join(self.output_dir, "dashboard_export")
        os.makedirs(export_dir, exist_ok=True)
        
        # We'll create a simple HTML file with screenshots of the dashboard tabs
        index_html = os.path.join(export_dir, "index.html")
        
        # For now, just create a placeholder HTML file
        with open(index_html, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Export</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2em;
        }
        h1, h2, h3 {
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            color: #0366d6;
        }
        .note {
            background-color: #f8f9fa;
            border-left: 4px solid #0366d6;
            padding: 1em;
            margin: 1em 0;
        }
        .card {
            border: 1px solid #dfe2e5;
            border-radius: 3px;
            margin: 1em 0;
            padding: 1em;
        }
    </style>
</head>
<body>
    <h1>Dashboard Export</h1>
    
    <div class="note">
        <p>This is a static export of the visualization dashboard. For interactive exploration, please run the dashboard directly:</p>
        <pre>python integrated_visualization_reports.py --dashboard</pre>
    </div>
    
    <h2>Dashboard Links</h2>
    <p>Start the interactive dashboard to access these views:</p>
    
    <div class="card">
        <h3>Overview</h3>
        <p>High-level summary of test results, including success rates and distribution across models and hardware platforms.</p>
    </div>
    
    <div class="card">
        <h3>Performance Analysis</h3>
        <p>Detailed analysis of performance metrics (throughput, latency, memory usage) for specific models and hardware combinations.</p>
    </div>
    
    <div class="card">
        <h3>Hardware Comparison</h3>
        <p>Side-by-side comparison of different hardware platforms, with visualizations to identify optimal hardware for each model type.</p>
    </div>
    
    <div class="card">
        <h3>Time Series Analysis</h3>
        <p>Performance trends over time, with statistical analysis to identify significant changes and potential regressions.</p>
    </div>
    
    <div class="card">
        <h3>Simulation Validation</h3>
        <p>Validation of simulation accuracy by comparing performance metrics between simulated and real hardware.</p>
    </div>
    
    <h2>Export Generated</h2>
    <p>This export was generated on """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
</body>
</html>""")
        
        logger.info(f"Dashboard visualizations exported to {export_dir}")
        return {"index_html": index_html}
    
    def run(self):
        """Run the integrated system according to the specified options."""
        report_urls = {}
        
        # Start dashboard if requested
        if self.args.dashboard:
            self.dashboard_process = self.start_dashboard()
        
        try:
            # Generate reports if requested
            if self.args.reports:
                reports = self.generate_reports()
                
                # Create URLs for reports if dashboard is running
                if self.dashboard_process and reports:
                    for report_name, report_path in reports.items():
                        if os.path.isfile(report_path) and report_path.endswith((".html", ".svg", ".png")):
                            # The actual URL would need a web server; this is just for reference
                            relative_path = os.path.relpath(report_path, self.output_dir)
                            report_urls[report_name] = f"/reports/{relative_path}"
            
            # Export dashboard visualizations if requested
            if self.args.dashboard_export:
                export_results = self.export_dashboard_visualizations()
                
                # Create URLs for exports if dashboard is running
                if self.dashboard_process and export_results:
                    for export_name, export_path in export_results.items():
                        if os.path.isfile(export_path) and export_path.endswith((".html", ".svg", ".png")):
                            # The actual URL would need a web server; this is just for reference
                            relative_path = os.path.relpath(export_path, self.output_dir)
                            report_urls[export_name] = f"/reports/{relative_path}"
            
            # If only running the dashboard, wait for it to exit
            if self.dashboard_only and self.dashboard_process:
                try:
                    logger.info("Dashboard is running. Press Ctrl+C to exit.")
                    self.dashboard_process.wait()
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received, stopping dashboard...")
                finally:
                    if self.dashboard_process.poll() is None:
                        self.dashboard_process.terminate()
                        self.dashboard_process.wait()
            
        finally:
            # Terminate dashboard process if it's still running and not in dashboard-only mode
            if not self.dashboard_only and self.dashboard_process and self.dashboard_process.poll() is None:
                logger.info("Stopping dashboard...")
                self.dashboard_process.terminate()
                self.dashboard_process.wait()
        
        logger.info("Done.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrated Visualization and Reports System")
    
    # Main operation modes
    parser.add_argument("--dashboard", action="store_true",
                       help="Start the visualization dashboard")
    parser.add_argument("--reports", action="store_true",
                       help="Generate reports using the Enhanced CI/CD Reports Generator")
    parser.add_argument("--dashboard-export", action="store_true",
                       help="Export dashboard visualizations for offline viewing")
    
    # Dashboard options
    parser.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT,
                       help=f"Port for the visualization dashboard (default: {DEFAULT_DASHBOARD_PORT})")
    parser.add_argument("--dashboard-host", default=DEFAULT_DASHBOARD_HOST,
                       help=f"Host for the visualization dashboard (default: {DEFAULT_DASHBOARD_HOST})")
    parser.add_argument("--open-browser", action="store_true",
                       help="Open a web browser to the dashboard when it starts")
    parser.add_argument("--debug", action="store_true",
                       help="Run the dashboard in debug mode with hot reloading")
    
    # Report options
    parser.add_argument("--format", choices=["html", "markdown"],
                       help="Report format (default: html)")
    parser.add_argument("--simulation-validation", action="store_true",
                       help="Generate simulation validation report")
    parser.add_argument("--cross-hardware-comparison", action="store_true",
                       help="Generate cross-hardware performance comparison report")
    parser.add_argument("--combined-report", action="store_true",
                       help="Generate combined report with both simulation validation and cross-hardware comparison")
    parser.add_argument("--historical", action="store_true",
                       help="Include historical trend data in the report")
    parser.add_argument("--days", type=int,
                       help="Number of days to include in historical data (default: 30)")
    parser.add_argument("--badge-only", action="store_true",
                       help="Generate only status badges (no full reports)")
    parser.add_argument("--ci", action="store_true",
                       help="Generate reports for CI/CD integration")
    parser.add_argument("--github-pages", action="store_true",
                       help="Generate reports for GitHub Pages")
    parser.add_argument("--export-metrics", action="store_true",
                       help="Export performance metrics to CSV for further analysis")
    parser.add_argument("--highlight-simulation", action="store_true",
                       help="Highlight simulated hardware in reports")
    parser.add_argument("--tolerance", type=float,
                       help="Tolerance for simulation validation (as percentage)")
    parser.add_argument("--include-visualizations", action="store_true",
                       help="Include visualizations in reports")
    parser.add_argument("--visualization-format", choices=["png", "svg", "pdf"],
                       help="Format for visualization images (default: png)")
    
    # Common options
    parser.add_argument("--db-path", help=f"Path to DuckDB database file (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--output-dir", help=f"Directory to save reports (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Validate arguments
    if not args.dashboard and not args.reports and not args.dashboard_export:
        logger.error("Please specify at least one of --dashboard, --reports, or --dashboard-export")
        logger.info("Use --help for usage information")
        return 1
    
    # Run the integrated system
    system = IntegratedSystem(args)
    system.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())