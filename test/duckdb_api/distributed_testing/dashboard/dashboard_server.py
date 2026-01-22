#!/usr/bin/env python3
"""
Dashboard Server for Distributed Testing Framework

This module implements a web server for serving the visualization dashboard
for the distributed testing framework.

Features:
- Serves interactive HTML dashboards
- Real-time updates of test results
- REST API for accessing test data
- WebSocket support for live updates
"""

import os
import sys
import json
import logging
import asyncio
import threading
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dashboard_server")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import optional dependencies
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    logger.warning("aiohttp not available. Web server functionality will be limited.")
    AIOHTTP_AVAILABLE = False

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    logger.warning("jinja2 not available. Template rendering will be limited.")
    JINJA2_AVAILABLE = False

# Try to import the dashboard generator and visualization engine
try:
    from duckdb_api.distributed_testing.dashboard.dashboard_generator import DashboardGenerator
    from duckdb_api.distributed_testing.dashboard.visualization import VisualizationEngine
    DASHBOARD_COMPONENTS_AVAILABLE = True
except ImportError:
    logger.warning("Dashboard components not available. Dashboard functionality will be limited.")
    DASHBOARD_COMPONENTS_AVAILABLE = False

class DashboardServer:
    """Dashboard server for the distributed testing framework."""
    
    def __init__(self, host: str = "localhost", port: int = 8081,
                result_aggregator=None, coordinator_url: str = None,
                output_dir: str = "./dashboards"):
        """Initialize the dashboard server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            result_aggregator: Result aggregator for accessing result data
            coordinator_url: URL of the coordinator server
            output_dir: Directory to save generated dashboards
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for the dashboard server")
        
        self.host = host
        self.port = port
        self.result_aggregator = result_aggregator
        self.coordinator_url = coordinator_url
        self.output_dir = output_dir
        
        # Create dashboard generator and visualization engine
        self.dashboard_generator = None
        self.visualization_engine = None
        
        if DASHBOARD_COMPONENTS_AVAILABLE:
            # Create visualization engine
            self.visualization_engine = VisualizationEngine(
                result_aggregator=result_aggregator,
                output_dir=os.path.join(output_dir, "visualizations")
            )
            
            # Create dashboard generator
            self.dashboard_generator = DashboardGenerator(
                result_aggregator=result_aggregator,
                output_dir=output_dir
            )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "static"), exist_ok=True)
        
        # Web application
        self.app = web.Application()
        self.setup_routes()
        
        # Jinja2 templates
        self.setup_templates()
        
        # Configuration
        self.config = {
            "auto_refresh": 60,  # Auto-refresh interval in seconds (0 to disable)
            "theme": "light",  # light or dark
            "max_items_per_page": 50,  # Maximum number of items per page
            "default_report_type": "performance",  # Default report type
            "api_cache_time": 10,  # Cache time for API responses in seconds
        }
        
        # Cache for API responses
        self.api_cache = {}
        self.api_cache_time = {}
        
        # WebSocket connections
        self.websocket_connections = set()
        
        # Running flag
        self.running = False
        
        logger.info("Dashboard server initialized")
    
    def setup_routes(self):
        """Set up the routes for the web application."""
        # Main routes
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/dashboard", self.handle_dashboard)
        self.app.router.add_get("/report/{report_type}", self.handle_report)
        self.app.router.add_get("/visualization/{viz_type}", self.handle_visualization)
        
        # API routes
        self.app.router.add_get("/api/status", self.handle_api_status)
        self.app.router.add_get("/api/tests", self.handle_api_tests)
        self.app.router.add_get("/api/workers", self.handle_api_workers)
        self.app.router.add_get("/api/regressions", self.handle_api_regressions)
        self.app.router.add_get("/api/dimensions", self.handle_api_dimensions)
        self.app.router.add_get("/api/performance", self.handle_api_performance)
        
        # WebSocket route
        self.app.router.add_get("/ws", self.handle_websocket)
        
        # Static files
        self.app.router.add_static("/static", 
                                 os.path.join(self.output_dir, "static"),
                                 name="static")
        self.app.router.add_static("/visualizations", 
                                 os.path.join(self.output_dir, "visualizations"),
                                 name="visualizations")
    
    def setup_templates(self):
        """Set up the Jinja2 templates."""
        if JINJA2_AVAILABLE:
            # Create template directory if it doesn't exist
            template_dir = os.path.join(self.output_dir, "templates")
            os.makedirs(template_dir, exist_ok=True)
            
            # Create default templates if they don't exist
            self.create_default_templates(template_dir)
            
            # Set up Jinja2 environment
            self.jinja_env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )
        else:
            self.jinja_env = None
    
    def create_default_templates(self, template_dir: str):
        """Create default templates if they don't exist.
        
        Args:
            template_dir: Directory to create templates in
        """
        # Create index.html template
        index_path = os.path.join(template_dir, "index.html")
        if not os.path.exists(index_path):
            with open(index_path, "w") as f:
                f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Distributed Testing Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: {% if theme == "dark" %}#222222{% else %}#ffffff{% endif %};
            color: {% if theme == "dark" %}#f8f9fa{% else %}#333333{% endif %};
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }
        
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .card {
            background-color: {% if theme == "dark" %}#333333{% else %}#f8f9fa{% endif %};
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h2 {
            font-size: 20px;
            margin-top: 0;
            margin-bottom: 15px;
            color: {% if theme == "dark" %}#f8f9fa{% else %}#333333{% endif %};
        }
        
        .card p {
            margin-bottom: 15px;
            color: {% if theme == "dark" %}#cccccc{% else %}#666666{% endif %};
        }
        
        .card .btn {
            display: inline-block;
            background-color: #1f77b4;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            text-decoration: none;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        
        .card .btn:hover {
            background-color: #1a6091;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid {% if theme == "dark" %}#444444{% else %}#eeeeee{% endif %};
            color: {% if theme == "dark" %}#888888{% else %}#777777{% endif %};
        }
    </style>
    {% if auto_refresh > 0 %}
    <meta http-equiv="refresh" content="{{ auto_refresh }}">
    {% endif %}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Distributed Testing Dashboard</h1>
            <p>Interactive visualizations and reports for distributed test results</p>
        </div>
        
        <div class="card-grid">
            <div class="card">
                <h2>Performance Dashboard</h2>
                <p>Comprehensive view of test performance metrics across all dimensions.</p>
                <a href="/dashboard" class="btn">View Dashboard</a>
            </div>
            
            <div class="card">
                <h2>Regression Report</h2>
                <p>Analysis of performance regressions with statistical significance testing.</p>
                <a href="/report/regression" class="btn">View Report</a>
            </div>
            
            <div class="card">
                <h2>Dimension Analysis</h2>
                <p>Comparative analysis of test performance across different dimensions.</p>
                <a href="/report/dimension" class="btn">View Analysis</a>
            </div>
            
            <div class="card">
                <h2>Worker Performance</h2>
                <p>Performance metrics and statistics for worker nodes.</p>
                <a href="/report/worker" class="btn">View Performance</a>
            </div>
            
            <div class="card">
                <h2>Test Details</h2>
                <p>Detailed information about individual tests and their results.</p>
                <a href="/report/test" class="btn">View Details</a>
            </div>
            
            <div class="card">
                <h2>API Documentation</h2>
                <p>Documentation for the dashboard API endpoints.</p>
                <a href="/api/status" class="btn">View API</a>
            </div>
        </div>
        
        <div class="footer">
            <p>Distributed Testing Framework - Dashboard Server</p>
            <p>Generated: {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
""")
    
    def configure(self, config_updates: Dict[str, Any]):
        """Update the dashboard server configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        
        # Update dashboard generator and visualization engine if available
        if self.dashboard_generator:
            self.dashboard_generator.configure(config_updates)
            
        if self.visualization_engine:
            self.visualization_engine.configure(config_updates)
        
        logger.info(f"Dashboard server configuration updated: {config_updates}")
    
    async def start(self):
        """Start the dashboard server."""
        if self.running:
            logger.warning("Dashboard server already running")
            return
        
        try:
            # Start the web server
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            
            self.running = True
            logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
            
            # Keep the server running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error starting dashboard server: {e}")
            raise
    
    def start_async(self):
        """Start the dashboard server in a separate thread."""
        thread = threading.Thread(target=self._run_async_server, daemon=True)
        thread.start()
        return thread
    
    def _run_async_server(self):
        """Run the async server in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.start())
        except Exception as e:
            logger.error(f"Error in async server thread: {e}")
        finally:
            loop.close()
    
    async def stop(self):
        """Stop the dashboard server."""
        if not self.running:
            return
        
        # Close all WebSocket connections
        for ws in self.websocket_connections:
            await ws.close()
        
        self.running = False
        logger.info("Dashboard server stopped")
    
    def get_cached_api_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached API data if available and not expired.
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached data if available, None otherwise
        """
        if cache_key in self.api_cache:
            # Check if cache is still valid
            cache_time = self.api_cache_time.get(cache_key, 0)
            if (datetime.now() - cache_time).total_seconds() < self.config["api_cache_time"]:
                return self.api_cache[cache_key]
        
        return None
    
    def set_cached_api_data(self, cache_key: str, data: Dict[str, Any]):
        """Set cached API data.
        
        Args:
            cache_key: Cache key to set
            data: Data to cache
        """
        self.api_cache[cache_key] = data
        self.api_cache_time[cache_key] = datetime.now()
    
    async def handle_index(self, request):
        """Handle requests to the index page.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        if self.jinja_env:
            # Render template
            template = self.jinja_env.get_template("index.html")
            
            # Render with context
            html = template.render(
                theme=self.config["theme"],
                auto_refresh=self.config["auto_refresh"],
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            return web.Response(text=html, content_type="text/html")
        else:
            # Simple response without template
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Distributed Testing Dashboard</title>
</head>
<body>
    <h1>Distributed Testing Dashboard</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <ul>
        <li><a href="/dashboard">View Dashboard</a></li>
        <li><a href="/report/regression">View Regression Report</a></li>
        <li><a href="/api/status">API Status</a></li>
    </ul>
</body>
</html>
"""
            return web.Response(text=html, content_type="text/html")
    
    async def handle_dashboard(self, request):
        """Handle requests to the dashboard page.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        if not self.dashboard_generator:
            return web.Response(text="Dashboard generator not available", status=500)
        
        try:
            # Get data for the dashboard
            if self.result_aggregator:
                data = {
                    "overall_status": self.result_aggregator.get_overall_status(),
                    "test_analysis": self.result_aggregator.get_test_analysis(),
                    "worker_analysis": self.result_aggregator.worker_analysis,
                    "task_type_analysis": self.result_aggregator.task_type_analysis,
                    "dimension_analysis": self.result_aggregator.get_dimension_analysis(),
                    "regression_results": self.result_aggregator.get_regressions(),
                    "historical_performance": getattr(self.result_aggregator, 'historical_performance', {})
                }
            else:
                # Use static data for testing
                data = self.get_static_test_data()
            
            # Generate dashboard
            dashboard_path = self.dashboard_generator.generate_dashboard(data)
            
            if dashboard_path:
                # Read the generated HTML
                with open(dashboard_path, "r") as f:
                    html = f.read()
                
                return web.Response(text=html, content_type="text/html")
            else:
                return web.Response(text="Failed to generate dashboard", status=500)
                
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return web.Response(text=f"Error generating dashboard: {str(e)}", status=500)
    
    async def handle_report(self, request):
        """Handle requests for specific reports.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        if not self.dashboard_generator:
            return web.Response(text="Dashboard generator not available", status=500)
        
        # Get report type
        report_type = request.match_info.get("report_type", self.config["default_report_type"])
        
        try:
            # Get data for the report
            if self.result_aggregator:
                data = {
                    "overall_status": self.result_aggregator.get_overall_status(),
                    "test_analysis": self.result_aggregator.get_test_analysis(),
                    "worker_analysis": self.result_aggregator.worker_analysis,
                    "task_type_analysis": self.result_aggregator.task_type_analysis,
                    "dimension_analysis": self.result_aggregator.get_dimension_analysis(),
                    "regression_results": self.result_aggregator.get_regressions(),
                    "historical_performance": getattr(self.result_aggregator, 'historical_performance', {})
                }
            else:
                # Use static data for testing
                data = self.get_static_test_data()
            
            # Generate report
            report_path = self.dashboard_generator.generate_report(report_type, data)
            
            if report_path:
                # Read the generated HTML
                with open(report_path, "r") as f:
                    html = f.read()
                
                return web.Response(text=html, content_type="text/html")
            else:
                return web.Response(text=f"Failed to generate {report_type} report", status=500)
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return web.Response(text=f"Error generating report: {str(e)}", status=500)
    
    async def handle_visualization(self, request):
        """Handle requests for specific visualizations.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        if not self.visualization_engine:
            return web.Response(text="Visualization engine not available", status=500)
        
        # Get visualization type and parameters
        viz_type = request.match_info.get("viz_type", "")
        params = dict(request.query)
        
        try:
            # Get data for the visualization
            if self.result_aggregator:
                # Format depends on visualization type
                if viz_type == "time_series":
                    # Get time series data
                    entity_id = params.get("entity_id", "")
                    entity_type = params.get("entity_type", "worker")
                    metric = params.get("metric", "")
                    days = int(params.get("days", 30))
                    
                    if not entity_id or not metric:
                        return web.Response(text="Missing required parameters", status=400)
                    
                    time_series = self.result_aggregator.get_time_series(entity_id, entity_type, metric, days)
                    
                    data = {
                        "time_series": {entity_id: time_series},
                        "metric": metric,
                        "title": f"{metric.replace('_', ' ').title()} for {entity_id}"
                    }
                    
                elif viz_type == "dimension":
                    # Get dimension data
                    dimension = params.get("dimension", "")
                    metric = params.get("metric", "")
                    
                    if not dimension or not metric:
                        return web.Response(text="Missing required parameters", status=400)
                    
                    dimension_data = self.result_aggregator.get_dimension_analysis(dimension)
                    
                    # Extract values for the metric
                    values = {}
                    for value, metrics in dimension_data.items():
                        mean_key = f"{metric}_mean"
                        if mean_key in metrics:
                            values[value] = metrics[mean_key]
                    
                    data = {
                        "dimension": dimension,
                        "metric": metric,
                        "values": values,
                        "title": f"{metric.replace('_', ' ').title()} by {dimension.replace('_', ' ').title()}"
                    }
                    
                elif viz_type == "regression":
                    # Get regression data
                    significant_only = params.get("significant_only", "true").lower() == "true"
                    
                    data = {
                        "regressions": self.result_aggregator.get_regressions(significant_only=significant_only)
                    }
                    
                else:
                    return web.Response(text=f"Unknown visualization type: {viz_type}", status=400)
                
            else:
                # Use static data for testing
                data = self.get_static_viz_data(viz_type)
            
            # Generate visualization
            viz_path = self.visualization_engine.create_visualization(viz_type, data)
            
            if viz_path:
                # Serve the image
                return web.FileResponse(viz_path)
            else:
                return web.Response(text=f"Failed to generate {viz_type} visualization", status=500)
                
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return web.Response(text=f"Error generating visualization: {str(e)}", status=500)
    
    async def handle_api_status(self, request):
        """Handle requests for API status.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        # Check if we have cached data
        cached_data = self.get_cached_api_data("api_status")
        if cached_data:
            return web.json_response(cached_data)
        
        # Get status data
        if self.result_aggregator:
            status_data = self.result_aggregator.get_overall_status()
        else:
            # Use static data for testing
            status_data = {
                "test_count": 42,
                "worker_count": 5,
                "task_type_count": 3,
                "hardware_count": 4,
                "total_executions": 1250,
                "regression_count": 3,
                "significant_regression_count": 1
            }
        
        # Add API metadata
        api_data = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "data": status_data,
            "api_version": "1.0",
            "server_info": {
                "coordinator_url": self.coordinator_url,
                "dashboard_url": f"http://{self.host}:{self.port}",
                "auto_refresh": self.config["auto_refresh"]
            }
        }
        
        # Cache the data
        self.set_cached_api_data("api_status", api_data)
        
        return web.json_response(api_data)
    
    async def handle_api_tests(self, request):
        """Handle requests for test data.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        # Check if we have cached data
        cached_data = self.get_cached_api_data("api_tests")
        if cached_data:
            return web.json_response(cached_data)
        
        # Get pagination parameters
        page = int(request.query.get("page", 1))
        page_size = int(request.query.get("page_size", self.config["max_items_per_page"]))
        
        # Get sort parameters
        sort_by = request.query.get("sort_by", "execution_count")
        sort_order = request.query.get("sort_order", "desc")
        
        # Get test data
        if self.result_aggregator:
            test_data = self.result_aggregator.get_test_analysis()
        else:
            # Use static data for testing
            test_data = self.get_static_test_data().get("test_analysis", {})
        
        # Convert to list for pagination
        tests_list = []
        for test_id, analysis in test_data.items():
            analysis["test_id"] = test_id
            tests_list.append(analysis)
        
        # Sort the list
        reverse = sort_order.lower() == "desc"
        tests_list.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        
        # Paginate the list
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_tests = tests_list[start_idx:end_idx]
        
        # Create response
        api_data = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "tests": paginated_tests,
                "total_count": len(tests_list),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(tests_list) + page_size - 1) // page_size
            },
            "api_version": "1.0"
        }
        
        # Cache the data
        self.set_cached_api_data("api_tests", api_data)
        
        return web.json_response(api_data)
    
    async def handle_api_workers(self, request):
        """Handle requests for worker data.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        # Check if we have cached data
        cached_data = self.get_cached_api_data("api_workers")
        if cached_data:
            return web.json_response(cached_data)
        
        # Get pagination parameters
        page = int(request.query.get("page", 1))
        page_size = int(request.query.get("page_size", self.config["max_items_per_page"]))
        
        # Get sort parameters
        sort_by = request.query.get("sort_by", "execution_count")
        sort_order = request.query.get("sort_order", "desc")
        
        # Get worker data
        if self.result_aggregator:
            worker_data = self.result_aggregator.worker_analysis
        else:
            # Use static data for testing
            worker_data = self.get_static_test_data().get("worker_analysis", {})
        
        # Convert to list for pagination
        workers_list = []
        for worker_id, analysis in worker_data.items():
            analysis["worker_id"] = worker_id
            workers_list.append(analysis)
        
        # Sort the list
        reverse = sort_order.lower() == "desc"
        workers_list.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        
        # Paginate the list
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_workers = workers_list[start_idx:end_idx]
        
        # Create response
        api_data = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "workers": paginated_workers,
                "total_count": len(workers_list),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(workers_list) + page_size - 1) // page_size
            },
            "api_version": "1.0"
        }
        
        # Cache the data
        self.set_cached_api_data("api_workers", api_data)
        
        return web.json_response(api_data)
    
    async def handle_api_regressions(self, request):
        """Handle requests for regression data.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        # Check if we have cached data
        cached_data = self.get_cached_api_data("api_regressions")
        if cached_data:
            return web.json_response(cached_data)
        
        # Get parameters
        significant_only = request.query.get("significant_only", "false").lower() == "true"
        
        # Get regression data
        if self.result_aggregator:
            regression_data = self.result_aggregator.get_regressions(significant_only=significant_only)
        else:
            # Use static data for testing
            regression_data = self.get_static_test_data().get("regression_results", {})
            if significant_only:
                regression_data = {
                    k: v for k, v in regression_data.items() 
                    if v.get("has_significant_regression", False)
                }
        
        # Create response
        api_data = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "regressions": regression_data,
                "regression_count": len(regression_data),
                "significant_only": significant_only
            },
            "api_version": "1.0"
        }
        
        # Cache the data
        self.set_cached_api_data("api_regressions", api_data)
        
        return web.json_response(api_data)
    
    async def handle_api_dimensions(self, request):
        """Handle requests for dimension data.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        # Check if we have cached data
        cached_data = self.get_cached_api_data("api_dimensions")
        if cached_data:
            return web.json_response(cached_data)
        
        # Get parameters
        dimension = request.query.get("dimension", None)
        
        # Get dimension data
        if self.result_aggregator:
            dimension_data = self.result_aggregator.get_dimension_analysis(dimension)
        else:
            # Use static data for testing
            all_dimensions = self.get_static_test_data().get("dimension_analysis", {})
            dimension_data = all_dimensions.get(dimension, all_dimensions) if dimension else all_dimensions
        
        # Create response
        api_data = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "dimensions": dimension_data,
                "dimension_count": len(dimension_data) if isinstance(dimension_data, dict) else 0,
                "requested_dimension": dimension
            },
            "api_version": "1.0"
        }
        
        # Cache the data
        self.set_cached_api_data("api_dimensions", api_data)
        
        return web.json_response(api_data)
    
    async def handle_api_performance(self, request):
        """Handle requests for performance data.
        
        Args:
            request: HTTP request
            
        Returns:
            HTTP response
        """
        # Check if we have cached data
        cached_data = self.get_cached_api_data("api_performance")
        if cached_data:
            return web.json_response(cached_data)
        
        # Get parameters
        test_id = request.query.get("test_id", None)
        metric = request.query.get("metric", None)
        days = int(request.query.get("days", 30))
        
        # Get performance data
        performance_data = {}
        
        if self.result_aggregator and hasattr(self.result_aggregator, 'historical_performance'):
            if test_id:
                # Get data for specific test
                test_data = self.result_aggregator.historical_performance.get(test_id, {})
                performance_data = {test_id: test_data}
            else:
                # Get all performance data
                performance_data = self.result_aggregator.historical_performance
        else:
            # Use static data for testing
            performance_data = self.get_static_test_data().get("historical_performance", {})
            if test_id:
                performance_data = {test_id: performance_data.get(test_id, {})}
        
        # Process data for the response
        processed_data = {}
        
        for t_id, dates in performance_data.items():
            # Filter by date range
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            filtered_dates = {
                d: v for d, v in dates.items() 
                if d >= cutoff_date
            }
            
            # Filter by metric if specified
            if metric:
                for d, metrics_data in filtered_dates.items():
                    if "metrics" in metrics_data and metric in metrics_data["metrics"]:
                        if t_id not in processed_data:
                            processed_data[t_id] = {}
                        
                        if d not in processed_data[t_id]:
                            processed_data[t_id][d] = {"metrics": {}}
                        
                        processed_data[t_id][d]["metrics"][metric] = metrics_data["metrics"][metric]
            else:
                processed_data[t_id] = filtered_dates
        
        # Create response
        api_data = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "performance": processed_data,
                "test_count": len(processed_data),
                "days": days,
                "metric": metric
            },
            "api_version": "1.0"
        }
        
        # Cache the data
        self.set_cached_api_data("api_performance", api_data)
        
        return web.json_response(api_data)
    
    async def handle_websocket(self, request):
        """Handle WebSocket connections.
        
        Args:
            request: WebSocket request
            
        Returns:
            WebSocket response
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Add to connections
        self.websocket_connections.add(ws)
        logger.info(f"WebSocket connection established: {request.remote}")
        
        try:
            # Send initial status
            if self.result_aggregator:
                status_data = self.result_aggregator.get_overall_status()
            else:
                # Use static data for testing
                status_data = {
                    "test_count": 42,
                    "worker_count": 5,
                    "task_type_count": 3,
                    "hardware_count": 4,
                    "total_executions": 1250,
                    "regression_count": 3,
                    "significant_regression_count": 1
                }
                
            await ws.send_json({
                "type": "status",
                "data": status_data,
                "timestamp": datetime.now().isoformat()
            })
            
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Parse the message
                    try:
                        data = json.loads(msg.data)
                        message_type = data.get("type", "")
                        
                        if message_type == "ping":
                            # Respond to ping
                            await ws.send_json({
                                "type": "pong",
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        elif message_type == "get_status":
                            # Send status
                            if self.result_aggregator:
                                status_data = self.result_aggregator.get_overall_status()
                            else:
                                # Use static data for testing
                                status_data = {
                                    "test_count": 42,
                                    "worker_count": 5,
                                    "task_type_count": 3,
                                    "hardware_count": 4,
                                    "total_executions": 1250,
                                    "regression_count": 3,
                                    "significant_regression_count": 1
                                }
                                
                            await ws.send_json({
                                "type": "status",
                                "data": status_data,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON message: {msg.data}")
                
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
        finally:
            # Remove from connections
            self.websocket_connections.remove(ws)
            logger.info(f"WebSocket connection closed: {request.remote}")
        
        return ws
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all WebSocket connections.
        
        Args:
            message: Message to broadcast
        """
        if not self.websocket_connections:
            return
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()
        
        # Convert to JSON
        json_message = json.dumps(message)
        
        # Send to all connections
        for ws in list(self.websocket_connections):
            try:
                await ws.send_str(json_message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                try:
                    await ws.close()
                except:
                    pass
                self.websocket_connections.discard(ws)
    
    def get_static_test_data(self) -> Dict[str, Any]:
        """Get static test data for testing.
        
        Returns:
            Static test data
        """
        # Create static test data
        return {
            "overall_status": {
                "test_count": 42,
                "worker_count": 5,
                "task_type_count": 3,
                "hardware_count": 4,
                "total_executions": 1250,
                "regression_count": 3,
                "significant_regression_count": 1,
                "aggregated_metrics": {
                    "throughput_mean": 156.7,
                    "latency_mean": 42.3,
                    "memory_usage_mean": 2.45,
                    "success_rate_mean": 0.98
                }
            },
            "test_analysis": {
                "test1": {
                    "execution_count": 100,
                    "success_rate": 0.98,
                    "average_duration": 25.3,
                    "first_execution": datetime.now() - timedelta(days=10),
                    "last_execution": datetime.now() - timedelta(hours=1),
                    "hardware": "gpu1",
                    "model": "bert",
                    "throughput_mean": 150.2,
                    "latency_mean": 30.5
                },
                "test2": {
                    "execution_count": 80,
                    "success_rate": 0.95,
                    "average_duration": 32.1,
                    "first_execution": datetime.now() - timedelta(days=8),
                    "last_execution": datetime.now() - timedelta(hours=2),
                    "hardware": "gpu2",
                    "model": "t5",
                    "throughput_mean": 120.8,
                    "latency_mean": 45.2
                }
            },
            "worker_analysis": {
                "worker1": {
                    "execution_count": 500,
                    "success_rate": 0.99,
                    "average_duration": 28.7,
                    "first_execution": datetime.now() - timedelta(days=10),
                    "last_execution": datetime.now() - timedelta(hours=1),
                    "task_type_distribution": {
                        "type_counts": {"benchmark": 300, "inference": 200},
                        "type_percentages": {"benchmark": 60.0, "inference": 40.0}
                    }
                },
                "worker2": {
                    "execution_count": 400,
                    "success_rate": 0.97,
                    "average_duration": 32.5,
                    "first_execution": datetime.now() - timedelta(days=9),
                    "last_execution": datetime.now() - timedelta(hours=3),
                    "task_type_distribution": {
                        "type_counts": {"benchmark": 250, "inference": 150},
                        "type_percentages": {"benchmark": 62.5, "inference": 37.5}
                    }
                }
            },
            "dimension_analysis": {
                "hardware": {
                    "gpu1": {
                        "throughput_mean": 156.7,
                        "latency_mean": 28.3,
                        "memory_usage_mean": 2.1,
                        "success_rate_mean": 0.99
                    },
                    "gpu2": {
                        "throughput_mean": 142.3,
                        "latency_mean": 32.5,
                        "memory_usage_mean": 2.3,
                        "success_rate_mean": 0.97
                    }
                },
                "model": {
                    "bert": {
                        "throughput_mean": 155.3,
                        "latency_mean": 30.2,
                        "memory_usage_mean": 1.8,
                        "success_rate_mean": 0.98
                    },
                    "t5": {
                        "throughput_mean": 135.6,
                        "latency_mean": 35.7,
                        "memory_usage_mean": 2.5,
                        "success_rate_mean": 0.96
                    }
                }
            },
            "regression_results": {
                "test1": {
                    "metrics": {
                        "throughput": {
                            "is_regression": True,
                            "is_significant": True,
                            "percent_change": -15.3,
                            "baseline_mean": 165.7,
                            "current_mean": 140.2,
                            "p_value": 0.02
                        }
                    },
                    "baseline_period": ["2025-03-01", "2025-03-02", "2025-03-03"],
                    "current_period": ["2025-03-10", "2025-03-11", "2025-03-12"],
                    "has_significant_regression": True
                },
                "test2": {
                    "metrics": {
                        "latency": {
                            "is_regression": True,
                            "is_significant": False,
                            "percent_change": 8.5,
                            "baseline_mean": 32.1,
                            "current_mean": 34.8,
                            "p_value": 0.12
                        }
                    },
                    "baseline_period": ["2025-03-01", "2025-03-02", "2025-03-03"],
                    "current_period": ["2025-03-10", "2025-03-11", "2025-03-12"],
                    "has_significant_regression": False
                }
            },
            "historical_performance": {
                "test1": {
                    "2025-03-01": {
                        "metrics": {
                            "throughput": {
                                "mean": 165.2,
                                "std": 5.1,
                                "count": 10
                            },
                            "latency": {
                                "mean": 28.3,
                                "std": 2.1,
                                "count": 10
                            }
                        },
                        "count": 10
                    },
                    "2025-03-02": {
                        "metrics": {
                            "throughput": {
                                "mean": 166.5,
                                "std": 4.8,
                                "count": 10
                            },
                            "latency": {
                                "mean": 27.9,
                                "std": 1.9,
                                "count": 10
                            }
                        },
                        "count": 10
                    },
                    "2025-03-10": {
                        "metrics": {
                            "throughput": {
                                "mean": 142.1,
                                "std": 5.3,
                                "count": 10
                            },
                            "latency": {
                                "mean": 32.5,
                                "std": 2.4,
                                "count": 10
                            }
                        },
                        "count": 10
                    },
                    "2025-03-11": {
                        "metrics": {
                            "throughput": {
                                "mean": 138.7,
                                "std": 5.7,
                                "count": 10
                            },
                            "latency": {
                                "mean": 33.1,
                                "std": 2.6,
                                "count": 10
                            }
                        },
                        "count": 10
                    }
                }
            }
        }
    
    def get_static_viz_data(self, viz_type: str) -> Dict[str, Any]:
        """Get static visualization data for testing.
        
        Args:
            viz_type: Type of visualization
            
        Returns:
            Static visualization data
        """
        if viz_type == "time_series":
            # Create time series data
            dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
            values = [100 + i * 5 + (i % 3) * 10 for i in range(10)]
            
            return {
                "time_series": {
                    "test1": list(zip(dates, values))
                },
                "metric": "throughput",
                "title": "Throughput Over Time"
            }
            
        elif viz_type == "dimension":
            return {
                "dimension": "hardware",
                "metric": "throughput",
                "values": {
                    "gpu1": 156.7,
                    "gpu2": 142.3,
                    "cpu": 72.5
                },
                "title": "Throughput by Hardware"
            }
            
        elif viz_type == "regression":
            return {
                "regressions": {
                    "test1": {
                        "metrics": {
                            "throughput": {
                                "is_regression": True,
                                "is_significant": True,
                                "percent_change": -15.3,
                                "baseline_mean": 165.7,
                                "current_mean": 140.2,
                                "p_value": 0.02
                            }
                        },
                        "has_significant_regression": True
                    },
                    "test2": {
                        "metrics": {
                            "latency": {
                                "is_regression": True,
                                "is_significant": False,
                                "percent_change": 8.5,
                                "baseline_mean": 32.1,
                                "current_mean": 34.8,
                                "p_value": 0.12
                            }
                        },
                        "has_significant_regression": False
                    }
                }
            }
            
        elif viz_type == "correlation":
            return {
                "metric_x": "throughput",
                "metric_y": "latency",
                "points": [
                    (156.7, 28.3),
                    (142.3, 32.5),
                    (135.6, 35.7),
                    (120.8, 40.2),
                    (110.3, 45.1)
                ],
                "labels": ["gpu1", "gpu2", "gpu3", "cpu1", "cpu2"],
                "title": "Latency vs Throughput"
            }
            
        elif viz_type == "heatmap":
            return {
                "x_dimension": "model",
                "y_dimension": "hardware",
                "metric": "throughput",
                "values": [
                    [156.7, 142.3, 135.6],
                    [142.3, 135.6, 120.8],
                    [135.6, 120.8, 110.3]
                ],
                "x_labels": ["bert", "t5", "llama"],
                "y_labels": ["gpu1", "gpu2", "cpu"],
                "title": "Throughput by Model and Hardware"
            }
            
        return {}


# Command-line entry point
def main():
    parser = argparse.ArgumentParser(description="Dashboard Server for Distributed Testing Framework")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind the server to")
    parser.add_argument("--coordinator-url", help="URL of the coordinator server")
    parser.add_argument("--output-dir", default="./dashboards", help="Directory to save dashboards")
    parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Dashboard theme")
    parser.add_argument("--auto-open", action="store_true", help="Automatically open dashboard in browser")
    
    args = parser.parse_args()
    
    # Create and configure dashboard server
    server = DashboardServer(
        host=args.host,
        port=args.port,
        coordinator_url=args.coordinator_url,
        output_dir=args.output_dir
    )
    
    # Configure server
    server.configure({
        "theme": args.theme
    })
    
    # Start server
    print(f"Starting dashboard server at http://{args.host}:{args.port}")
    
    # Auto-open in browser if requested
    if args.auto_open:
        import webbrowser
        webbrowser.open(f"http://{args.host}:{args.port}")
    
    # Run the server (this will block until interrupted)
    asyncio.run(server.start())


if __name__ == "__main__":
    main()