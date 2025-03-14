"""
Monitoring Dashboard for Distributed Testing Framework

This module provides a web-based monitoring dashboard for the distributed testing framework,
allowing real-time monitoring of test execution, worker status, and result visualization.

Usage:
    python -m duckdb_api.distributed_testing.run_monitoring_dashboard
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import aiohttp
import jinja2
from aiohttp import web

from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_routes import setup_routes
from duckdb_api.distributed_testing.dashboard.websocket_handlers import WebSocketManager, setup_websocket_routes

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """Web-based monitoring dashboard for the distributed testing framework."""
    
    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 8080, 
        coordinator_url: Optional[str] = None, 
        result_aggregator_url: Optional[str] = None,
        refresh_interval: int = 5, 
        static_dir: Optional[str] = None, 
        template_dir: Optional[str] = None, 
        theme: str = 'light',
        enable_result_aggregator_integration: bool = False, 
        result_aggregator_integration: Optional[Any] = None,
        enable_e2e_test_integration: bool = False,
        e2e_test_integration: Optional[Any] = None,
        enable_performance_analytics: bool = True,
        enable_visualization_integration: bool = False,
        visualization_integration: Optional[Any] = None,
        dashboard_dir: str = "./dashboards"
    ):
        """Initialize the monitoring dashboard.
        
        Args:
            host: Host to bind the server to
            port: Port to listen on
            coordinator_url: URL of the coordinator service
            result_aggregator_url: URL of the result aggregator service
            refresh_interval: Interval in seconds to refresh data
            static_dir: Directory for static files
            template_dir: Directory for HTML templates
            theme: Dashboard theme (light or dark)
            enable_result_aggregator_integration: Whether to enable integration with result aggregator
            result_aggregator_integration: ResultAggregatorIntegration instance
            enable_e2e_test_integration: Whether to enable integration with E2E testing framework
            e2e_test_integration: E2ETestResultsIntegration instance
            enable_performance_analytics: Whether to enable performance analytics
            enable_visualization_integration: Whether to enable integration with Advanced Visualization System
            visualization_integration: VisualizationDashboardIntegration instance
            dashboard_dir: Directory to store dashboards
        """
        
        # Performance analytics data
        self.performance_analytics_data = {}
        self.enable_performance_analytics = enable_performance_analytics
        self.host = host
        self.port = port
        self.coordinator_url = coordinator_url
        self.result_aggregator_url = result_aggregator_url
        self.refresh_interval = refresh_interval
        self.theme = theme
        
        # Set up static and template directories
        module_dir = Path(__file__).parent
        self.static_dir = Path(static_dir) if static_dir else module_dir / 'static'
        self.template_dir = Path(template_dir) if template_dir else module_dir / 'templates'
        
        # Create static directory if it doesn't exist
        os.makedirs(self.static_dir, exist_ok=True)
        
        # Set up Jinja2 template environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Set up result aggregator integration
        self.enable_result_aggregator_integration = enable_result_aggregator_integration
        self.result_aggregator_integration = result_aggregator_integration
        
        # Set up E2E test integration
        self.enable_e2e_test_integration = enable_e2e_test_integration
        self.e2e_test_integration = e2e_test_integration
        
        # Set up Advanced Visualization System integration
        self.enable_visualization_integration = enable_visualization_integration
        self.visualization_integration = visualization_integration
        self.dashboard_dir = dashboard_dir
        
        # Initialize visualization integration if enabled but not provided
        if self.enable_visualization_integration and self.visualization_integration is None:
            try:
                # Import the visualization integration
                from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import (
                    VisualizationDashboardIntegration
                )
                
                # Create a symbolic link to dashboards in static directory for serving
                dashboards_static_dir = os.path.join(self.static_dir, 'dashboards')
                if not os.path.exists(dashboards_static_dir):
                    try:
                        # Use relative path for the link target if possible
                        target_path = os.path.relpath(self.dashboard_dir, os.path.dirname(dashboards_static_dir))
                        os.symlink(target_path, dashboards_static_dir, target_is_directory=True)
                    except Exception as e:
                        logger.error(f"Error creating symbolic link to dashboards: {e}")
                        # Fall back to normal directory
                        os.makedirs(dashboards_static_dir, exist_ok=True)
                
                # Initialize the integration
                self.visualization_integration = VisualizationDashboardIntegration(
                    dashboard_dir=self.dashboard_dir,
                    integration_dir=os.path.join(self.dashboard_dir, 'monitor_integration')
                )
                logger.info("Visualization Dashboard Integration initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Visualization Dashboard Integration: {e}")
                self.enable_visualization_integration = False
        
        # HTTP client session
        self.session = None
        
        logger.info(f"Initialized Monitoring Dashboard (host={host}, port={port})")
        logger.info(f"Coordinator URL: {coordinator_url}")
        logger.info(f"Result Aggregator URL: {result_aggregator_url}")
        logger.info(f"Static directory: {self.static_dir}")
        logger.info(f"Template directory: {self.template_dir}")
        logger.info(f"Result Aggregator Integration: {'Enabled' if enable_result_aggregator_integration else 'Disabled'}")
        logger.info(f"E2E Test Integration: {'Enabled' if enable_e2e_test_integration else 'Disabled'}")
        logger.info(f"Visualization Dashboard Integration: {'Enabled' if enable_visualization_integration else 'Disabled'}")
    
    async def get_coordinator_status(self) -> Dict[str, Any]:
        """Get status of the coordinator service.
        
        Returns:
            Dictionary containing coordinator status information
        """
        if not self.coordinator_url:
            return {
                "status": "Unknown",
                "error": "Coordinator URL not configured"
            }
        
        try:
            async with self.session.get(f"{self.coordinator_url}/api/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "status": "Error",
                        "error": f"HTTP {response.status}: {await response.text()}"
                    }
        except Exception as e:
            logger.error(f"Error getting coordinator status: {e}")
            return {
                "status": "Error",
                "error": str(e)
            }
    
    async def get_workers_status(self) -> Dict[str, Any]:
        """Get status of worker nodes.
        
        Returns:
            Dictionary containing worker status information
        """
        if not self.coordinator_url:
            return {
                "status": "Unknown",
                "error": "Coordinator URL not configured",
                "workers": []
            }
        
        try:
            async with self.session.get(f"{self.coordinator_url}/api/workers") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "status": "Error",
                        "error": f"HTTP {response.status}: {await response.text()}",
                        "workers": []
                    }
        except Exception as e:
            logger.error(f"Error getting workers status: {e}")
            return {
                "status": "Error",
                "error": str(e),
                "workers": []
            }
    
    async def get_workers_details(self) -> Dict[str, Any]:
        """Get detailed information about worker nodes.
        
        Returns:
            Dictionary containing detailed worker information
        """
        if not self.coordinator_url:
            return {
                "status": "Unknown",
                "error": "Coordinator URL not configured",
                "workers": []
            }
        
        try:
            async with self.session.get(f"{self.coordinator_url}/api/workers/details") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "status": "Error",
                        "error": f"HTTP {response.status}: {await response.text()}",
                        "workers": []
                    }
        except Exception as e:
            logger.error(f"Error getting workers details: {e}")
            return {
                "status": "Error",
                "error": str(e),
                "workers": []
            }
    
    async def get_tasks_status(self) -> Dict[str, Any]:
        """Get status of tasks.
        
        Returns:
            Dictionary containing task status information
        """
        if not self.coordinator_url:
            return {
                "status": "Unknown",
                "error": "Coordinator URL not configured",
                "tasks": []
            }
        
        try:
            async with self.session.get(f"{self.coordinator_url}/api/tasks") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "status": "Error",
                        "error": f"HTTP {response.status}: {await response.text()}",
                        "tasks": []
                    }
        except Exception as e:
            logger.error(f"Error getting tasks status: {e}")
            return {
                "status": "Error",
                "error": str(e),
                "tasks": []
            }
    
    async def get_task_history(self) -> Dict[str, Any]:
        """Get task execution history.
        
        Returns:
            Dictionary containing task history information
        """
        if not self.coordinator_url:
            return {
                "status": "Unknown",
                "error": "Coordinator URL not configured",
                "history": []
            }
        
        try:
            async with self.session.get(f"{self.coordinator_url}/api/tasks/history") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "status": "Error",
                        "error": f"HTTP {response.status}: {await response.text()}",
                        "history": []
                    }
        except Exception as e:
            logger.error(f"Error getting task history: {e}")
            return {
                "status": "Error",
                "error": str(e),
                "history": []
            }
    
    async def start(self):
        """Start the monitoring dashboard web server."""
        # Create client session
        self.session = aiohttp.ClientSession()
        
        # Create WebSocket manager
        self.websocket_manager = WebSocketManager()
        
        # Create web application
        app = web.Application()
        app['dashboard'] = self
        app['websocket_manager'] = self.websocket_manager
        
        # Setup routes
        setup_routes(app)
        setup_websocket_routes(app, self.websocket_manager)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Monitoring Dashboard started at http://{self.host}:{self.port}")
        logger.info(f"WebSocket server enabled for real-time monitoring")
        
        try:
            # Keep the server running
            while True:
                await asyncio.sleep(3600)  # Sleep for an hour
        except asyncio.CancelledError:
            logger.info("Stopping Monitoring Dashboard...")
        finally:
            # Cleanup
            await runner.cleanup()
            await self.session.close()
    
    def run(self):
        """Run the monitoring dashboard (blocking)."""
        asyncio.run(self.start())
        
    def store_performance_analytics_data(self, data: Dict[str, Any]) -> None:
        """Store performance analytics data.
        
        Args:
            data: Performance analytics data
        """
        if not self.enable_performance_analytics:
            logger.warning("Performance analytics is disabled")
            return
        
        logger.info("Storing performance analytics data")
        
        # Store data by time range
        time_range = 30  # Default time range
        
        # Store data
        self.performance_analytics_data[time_range] = data
        
        logger.info(f"Stored performance analytics data for {time_range} day time range")
    
    def get_performance_analytics_data(self, time_range: int = 30) -> Dict[str, Any]:
        """Get performance analytics data for the specified time range.
        
        Args:
            time_range: Time range in days
            
        Returns:
            Performance analytics data
        """
        if not self.enable_performance_analytics:
            logger.warning("Performance analytics is disabled")
            return {}
        
        # Get data for the specified time range
        if time_range in self.performance_analytics_data:
            return self.performance_analytics_data[time_range]
        
        # If no data for the specified time range, use the default
        if 30 in self.performance_analytics_data:
            return self.performance_analytics_data[30]
        
        # If no data at all, return empty dict
        return {}