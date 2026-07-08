#!/usr/bin/env python3
"""
Monitoring Dashboard Connector for the Simulation Accuracy and Validation Framework.

This module provides a connector between the visualization system and the monitoring dashboard,
enabling real-time visualization, alerts, and comprehensive monitoring of simulation validation results.
"""

import os
import sys
import logging
import json
import datetime
import uuid
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("monitoring_dashboard_connector")

# Import the validation visualizer db connector
from data.duckdb.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector

# Import the visualization system
from data.duckdb.simulation_validation.visualization.validation_visualizer import ValidationVisualizer

# Import base class for the database performance optimization
from data.duckdb.simulation_validation.db_performance_optimizer import DatabasePerformanceOptimizer

class MonitoringDashboardConnector:
    """
    Dashboard connector for the Simulation Accuracy and Validation Framework.
    
    This class provides high-level integration with the monitoring dashboard system,
    enabling real-time visualization, alerts, and comprehensive monitoring of simulation
    validation and database performance metrics.
    """
    
    def __init__(
        self,
        dashboard_url: str,
        dashboard_api_key: str,
        db_connector: Optional[ValidationVisualizerDBConnector] = None,
        db_optimizer: Optional[DatabasePerformanceOptimizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the monitoring dashboard connector.
        
        Args:
            dashboard_url: URL of the monitoring dashboard API
            dashboard_api_key: API key for the dashboard
            db_connector: ValidationVisualizerDBConnector instance
            db_optimizer: DatabasePerformanceOptimizer instance
            config: Configuration for the connector
        """
        self.dashboard_url = dashboard_url
        self.dashboard_api_key = dashboard_api_key
        self.db_connector = db_connector
        self.db_optimizer = db_optimizer
        self.config = config or {}
        
        # Authentication state
        self.session_token = None
        self.session_expires = None
        self.connected = False
        
        # Connect to the dashboard
        self._connect_to_dashboard()
        
        logger.info("MonitoringDashboardConnector initialized")
    
    def _connect_to_dashboard(self) -> bool:
        """
        Connect to the monitoring dashboard.
        
        Returns:
            True if connection was successful, False otherwise
        """
        if not self.dashboard_url or not self.dashboard_api_key:
            logger.warning("Dashboard URL or API key not provided. Cannot connect to dashboard.")
            return False
        
        try:
            # Create authentication URL
            auth_url = f"{self.dashboard_url.rstrip('/')}/auth"
            
            # Create authentication headers
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.dashboard_api_key
            }
            
            # Create authentication data
            auth_data = {
                "client": "simulation_validation_framework",
                "client_version": self.config.get("framework_version", "1.0.0"),
                "connection_type": "monitoring_dashboard_connector"
            }
            
            # Send authentication request
            logger.debug(f"Authenticating with dashboard at {auth_url}")
            response = requests.post(auth_url, headers=headers, json=auth_data, timeout=30)
            
            # Check response
            if response.status_code == 200:
                # Parse response
                result = response.json()
                
                # Store session token and expiration
                self.session_token = result.get("token")
                
                # Calculate expiration time
                expires_in = result.get("expires_in", 3600)  # Default to 1 hour
                self.session_expires = datetime.datetime.now() + datetime.timedelta(seconds=expires_in)
                
                # Update connection state
                self.connected = True
                
                logger.info("Successfully connected to monitoring dashboard")
                return True
            else:
                logger.error(f"Failed to authenticate with dashboard: {response.status_code} - {response.text}")
                self.connected = False
                return False
        
        except Exception as e:
            logger.error(f"Error connecting to dashboard: {str(e)}")
            self.connected = False
            return False
    
    def _ensure_connection(self) -> bool:
        """
        Ensure the dashboard connection is active, reconnecting if necessary.
        
        Returns:
            True if connection is active, False otherwise
        """
        # If not connected, attempt to connect
        if not self.connected:
            return self._connect_to_dashboard()
        
        # If token is expired or will expire soon, reconnect
        if (not self.session_expires or 
            datetime.datetime.now() >= self.session_expires - datetime.timedelta(minutes=5)):
            logger.info("Dashboard session token expired or about to expire. Reconnecting...")
            return self._connect_to_dashboard()
        
        # Connection is active
        return True
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for dashboard API requests.
        
        Returns:
            Dictionary of authentication headers
        """
        self._ensure_connection()
        
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.session_token}"
        }
    
    def create_dashboard(
        self,
        title: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        layout: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new dashboard.
        
        Args:
            title: Dashboard title
            description: Dashboard description
            tags: List of tags for the dashboard
            layout: Layout configuration for the dashboard
            
        Returns:
            Dictionary with dashboard creation status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        try:
            # Create dashboard endpoint URL
            dashboard_url = f"{self.dashboard_url.rstrip('/')}/dashboards"
            
            # Create dashboard data
            dashboard_data = {
                "title": title,
                "description": description,
                "tags": tags or [],
                "layout": layout or {"type": "grid", "columns": 12}
            }
            
            # Send dashboard creation request
            response = requests.post(
                dashboard_url,
                headers=self._get_auth_headers(),
                json=dashboard_data,
                timeout=30
            )
            
            # Check response
            if response.status_code in (200, 201):
                result = response.json()
                logger.info(f"Created dashboard: {title} with ID: {result.get('dashboard_id')}")
                return {
                    "success": True,
                    "dashboard_id": result.get("dashboard_id"),
                    "title": title,
                    "url": result.get("url")
                }
            else:
                logger.error(f"Failed to create dashboard: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Failed to create dashboard: {response.status_code} - {response.text}"
                }
        
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_panel(
        self,
        dashboard_id: str,
        panel_type: str,
        title: str,
        data_source: Dict[str, Any],
        position: Optional[Dict[str, Any]] = None,
        size: Optional[Dict[str, Any]] = None,
        refresh_interval: Optional[int] = None,
        visualization_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new panel in a dashboard.
        
        Args:
            dashboard_id: ID of the dashboard
            panel_type: Type of panel (e.g., line-chart, heatmap, etc.)
            title: Panel title
            data_source: Data source configuration for the panel
            position: Position of the panel in the dashboard
            size: Size of the panel
            refresh_interval: Refresh interval in seconds
            visualization_options: Visualization options for the panel
            
        Returns:
            Dictionary with panel creation status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        try:
            # Create panel endpoint URL
            panel_url = f"{self.dashboard_url.rstrip('/')}/dashboards/{dashboard_id}/panels"
            
            # Create panel data
            panel_data = {
                "type": panel_type,
                "title": title,
                "data_source": data_source,
                "position": position or {"x": 0, "y": 0},
                "size": size or {"width": 6, "height": 4},
                "refresh_interval": refresh_interval,
                "visualization_options": visualization_options or {}
            }
            
            # Send panel creation request
            response = requests.post(
                panel_url,
                headers=self._get_auth_headers(),
                json=panel_data,
                timeout=30
            )
            
            # Check response
            if response.status_code in (200, 201):
                result = response.json()
                logger.info(f"Created panel: {title} with ID: {result.get('panel_id')}")
                return {
                    "success": True,
                    "panel_id": result.get("panel_id"),
                    "dashboard_id": dashboard_id,
                    "title": title
                }
            else:
                logger.error(f"Failed to create panel: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Failed to create panel: {response.status_code} - {response.text}"
                }
        
        except Exception as e:
            logger.error(f"Error creating panel: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def update_panel_data(
        self,
        dashboard_id: str,
        panel_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update the data for a panel.
        
        Args:
            dashboard_id: ID of the dashboard
            panel_id: ID of the panel
            data: New data for the panel
            
        Returns:
            Dictionary with update status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        try:
            # Create panel data endpoint URL
            panel_data_url = f"{self.dashboard_url.rstrip('/')}/dashboards/{dashboard_id}/panels/{panel_id}/data"
            
            # Send panel data update request
            response = requests.put(
                panel_data_url,
                headers=self._get_auth_headers(),
                json={"data": data},
                timeout=30
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Updated panel data for panel ID: {panel_id}")
                return {
                    "success": True,
                    "panel_id": panel_id,
                    "dashboard_id": dashboard_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            else:
                logger.error(f"Failed to update panel data: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Failed to update panel data: {response.status_code} - {response.text}"
                }
        
        except Exception as e:
            logger.error(f"Error updating panel data: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_alert(
        self,
        name: str,
        description: str,
        dashboard_id: str,
        panel_id: str,
        condition: Dict[str, Any],
        notification_channels: List[str],
        severity: str = "warning",
        interval: int = 300
    ) -> Dict[str, Any]:
        """
        Create a new alert for a panel.
        
        Args:
            name: Alert name
            description: Alert description
            dashboard_id: ID of the dashboard
            panel_id: ID of the panel
            condition: Alert condition
            notification_channels: List of notification channel IDs
            severity: Alert severity (info, warning, error, critical)
            interval: Alert evaluation interval in seconds
            
        Returns:
            Dictionary with alert creation status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        try:
            # Create alert endpoint URL
            alert_url = f"{self.dashboard_url.rstrip('/')}/alerts"
            
            # Create alert data
            alert_data = {
                "name": name,
                "description": description,
                "dashboard_id": dashboard_id,
                "panel_id": panel_id,
                "condition": condition,
                "notification_channels": notification_channels,
                "severity": severity,
                "interval": interval
            }
            
            # Send alert creation request
            response = requests.post(
                alert_url,
                headers=self._get_auth_headers(),
                json=alert_data,
                timeout=30
            )
            
            # Check response
            if response.status_code in (200, 201):
                result = response.json()
                logger.info(f"Created alert: {name} with ID: {result.get('alert_id')}")
                return {
                    "success": True,
                    "alert_id": result.get("alert_id"),
                    "dashboard_id": dashboard_id,
                    "panel_id": panel_id,
                    "name": name
                }
            else:
                logger.error(f"Failed to create alert: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Failed to create alert: {response.status_code} - {response.text}"
                }
        
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_simulation_validation_dashboard(
        self,
        hardware_types: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        dashboard_title: Optional[str] = None,
        include_database_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Create a comprehensive dashboard for simulation validation.
        
        Args:
            hardware_types: List of hardware types to include
            model_types: List of model types to include
            metrics: List of metrics to visualize
            dashboard_title: Title for the dashboard
            include_database_metrics: Whether to include database performance metrics
            
        Returns:
            Dictionary with dashboard creation status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        # Create a default dashboard title if not provided
        if not dashboard_title:
            dashboard_title = "Simulation Validation Dashboard"
            if hardware_types and len(hardware_types) == 1:
                dashboard_title += f" - {hardware_types[0]}"
            if model_types and len(model_types) == 1:
                dashboard_title += f" - {model_types[0]}"
        
        # Create dashboard
        dashboard_result = self.create_dashboard(
            title=dashboard_title,
            description="Comprehensive dashboard for simulation validation",
            tags=["simulation", "validation"] + (hardware_types or []) + (model_types or [])
        )
        
        if not dashboard_result["success"]:
            return dashboard_result
        
        dashboard_id = dashboard_result["dashboard_id"]
        
        # Default metrics if not provided
        if not metrics:
            metrics = [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w"
            ]
        
        # Define panel layouts
        panel_layouts = {
            "mape_comparison": {"width": 12, "height": 6, "x": 0, "y": 0},
            "hardware_heatmap": {"width": 6, "height": 6, "x": 0, "y": 6},
            "time_series": {"width": 6, "height": 6, "x": 6, "y": 6},
            "calibration_impact": {"width": 12, "height": 6, "x": 0, "y": 12},
            "drift_detection": {"width": 12, "height": 6, "x": 0, "y": 18},
            "database_performance": {"width": 12, "height": 6, "x": 0, "y": 24}
        }
        
        # Create panels
        created_panels = []
        
        # Create MAPE comparison panel
        mape_panel = self.create_panel(
            dashboard_id=dashboard_id,
            panel_type="mape-comparison",
            title="MAPE Comparison by Hardware and Model",
            data_source={
                "type": "simulation_validation",
                "query": {
                    "hardware_types": hardware_types,
                    "model_types": model_types,
                    "metrics": metrics,
                    "aggregation": "avg",
                    "timeframe": "7d"
                }
            },
            position=panel_layouts["mape_comparison"],
            size={"width": panel_layouts["mape_comparison"]["width"], "height": panel_layouts["mape_comparison"]["height"]},
            refresh_interval=300,
            visualization_options={
                "chart_type": "bar",
                "color_scheme": "blue",
                "show_legend": True,
                "show_values": True
            }
        )
        
        if mape_panel["success"]:
            created_panels.append(mape_panel)
        
        # Create hardware heatmap panel
        heatmap_panel = self.create_panel(
            dashboard_id=dashboard_id,
            panel_type="hardware-heatmap",
            title="Hardware-Model Compatibility Matrix",
            data_source={
                "type": "simulation_validation",
                "query": {
                    "hardware_types": hardware_types,
                    "model_types": model_types,
                    "metrics": metrics[0] if metrics else "throughput_items_per_second",
                    "aggregation": "avg",
                    "timeframe": "7d"
                }
            },
            position=panel_layouts["hardware_heatmap"],
            size={"width": panel_layouts["hardware_heatmap"]["width"], "height": panel_layouts["hardware_heatmap"]["height"]},
            refresh_interval=300,
            visualization_options={
                "color_scheme": "viridis",
                "show_legend": True,
                "show_values": True
            }
        )
        
        if heatmap_panel["success"]:
            created_panels.append(heatmap_panel)
        
        # Create time series panel for each metric
        y_position = panel_layouts["time_series"]["y"]
        
        for i, metric in enumerate(metrics):
            if i > 0:  # Skip the first one as it's already positioned alongside the heatmap
                y_position += 6  # Move down 6 grid units for each new metric
            
            time_series_panel = self.create_panel(
                dashboard_id=dashboard_id,
                panel_type="time-series",
                title=f"{metric.replace('_', ' ').title()} Over Time",
                data_source={
                    "type": "simulation_validation",
                    "query": {
                        "hardware_types": hardware_types,
                        "model_types": model_types,
                        "metrics": metric,
                        "timeframe": "30d",
                        "aggregation": "avg",
                        "grouping": "daily"
                    }
                },
                position={"x": panel_layouts["time_series"]["x"], "y": y_position},
                size={"width": panel_layouts["time_series"]["width"], "height": panel_layouts["time_series"]["height"]},
                refresh_interval=300,
                visualization_options={
                    "chart_type": "line",
                    "show_legend": True,
                    "show_points": True,
                    "color_scheme": "category10"
                }
            )
            
            if time_series_panel["success"]:
                created_panels.append(time_series_panel)
        
        # Create calibration impact panel
        calibration_panel = self.create_panel(
            dashboard_id=dashboard_id,
            panel_type="calibration-impact",
            title="Calibration Impact Analysis",
            data_source={
                "type": "simulation_validation",
                "query": {
                    "hardware_types": hardware_types,
                    "model_types": model_types,
                    "metrics": metrics,
                    "timeframe": "90d"
                }
            },
            position=panel_layouts["calibration_impact"],
            size={"width": panel_layouts["calibration_impact"]["width"], "height": panel_layouts["calibration_impact"]["height"]},
            refresh_interval=600,
            visualization_options={
                "chart_type": "before-after",
                "show_legend": True,
                "show_percentage": True,
                "color_scheme": "green-red"
            }
        )
        
        if calibration_panel["success"]:
            created_panels.append(calibration_panel)
        
        # Create drift detection panel
        drift_panel = self.create_panel(
            dashboard_id=dashboard_id,
            panel_type="drift-detection",
            title="Simulation Accuracy Drift Detection",
            data_source={
                "type": "simulation_validation",
                "query": {
                    "hardware_types": hardware_types,
                    "model_types": model_types,
                    "metrics": metrics,
                    "timeframe": "90d",
                    "significance_threshold": 0.05
                }
            },
            position=panel_layouts["drift_detection"],
            size={"width": panel_layouts["drift_detection"]["width"], "height": panel_layouts["drift_detection"]["height"]},
            refresh_interval=600,
            visualization_options={
                "chart_type": "detection-timeline",
                "show_legend": True,
                "color_scheme": "diverging"
            }
        )
        
        if drift_panel["success"]:
            created_panels.append(drift_panel)
        
        # Include database performance metrics if requested
        if include_database_metrics and self.db_optimizer:
            db_performance_panel = self.create_panel(
                dashboard_id=dashboard_id,
                panel_type="database-performance",
                title="Database Performance Metrics",
                data_source={
                    "type": "database_performance",
                    "query": {
                        "metrics": ["query_time", "storage_size", "index_efficiency", "vacuum_status"],
                        "timeframe": "7d",
                        "aggregation": "avg",
                        "grouping": "hourly"
                    }
                },
                position=panel_layouts["database_performance"],
                size={"width": panel_layouts["database_performance"]["width"], "height": panel_layouts["database_performance"]["height"]},
                refresh_interval=300,
                visualization_options={
                    "chart_type": "multi-line",
                    "show_legend": True,
                    "show_thresholds": True,
                    "color_scheme": "blues"
                }
            )
            
            if db_performance_panel["success"]:
                created_panels.append(db_performance_panel)
        
        # Return result
        return {
            "success": True,
            "dashboard_id": dashboard_id,
            "dashboard_title": dashboard_title,
            "panels": created_panels,
            "panel_count": len(created_panels),
            "url": dashboard_result.get("url")
        }
    
    def setup_database_performance_monitoring(
        self,
        dashboard_id: str,
        metrics: Optional[List[str]] = None,
        interval: int = 300,
        create_alerts: bool = True,
        visualization_style: str = "detailed",
        status_indicators: bool = True,
        show_history: bool = True,
        custom_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Set up monitoring for database performance metrics.
        
        Args:
            dashboard_id: ID of the dashboard
            metrics: List of metrics to monitor
            interval: Monitoring interval in seconds
            create_alerts: Whether to create alerts for critical metrics
            visualization_style: Style of visualization ("detailed", "compact", or "overview")
            status_indicators: Whether to show status indicators for each metric
            show_history: Whether to show historical data for metrics
            custom_thresholds: Custom threshold values for metrics in format:
                               {metric_name: {"warning": value, "error": value}}
            
        Returns:
            Dictionary with setup status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        if not self.db_optimizer:
            return {"success": False, "error": "Database performance optimizer not available"}
        
        # Default metrics if not provided
        if not metrics:
            metrics = [
                "query_time",
                "storage_size",
                "index_efficiency",
                "vacuum_status",
                "compression_ratio",
                "read_efficiency",
                "write_efficiency",
                "cache_performance"  # Added cache performance to default metrics
            ]
        
        # Define default thresholds
        default_thresholds = {
            "query_time": {"warning": 500, "error": 1000},  # ms
            "storage_size": {"warning": 500000000, "error": 1000000000},  # bytes
            "index_efficiency": {"warning": 70, "error": 50},  # percent (lower is worse)
            "vacuum_status": {"warning": 60, "error": 40},  # percent (lower is worse)
            "compression_ratio": {"warning": 2.5, "error": 1.5},  # ratio (lower is worse)
            "read_efficiency": {"warning": 200, "error": 100},  # records/second (lower is worse)
            "write_efficiency": {"warning": 150, "error": 75},  # records/second (lower is worse)
            "cache_performance": {"warning": 50, "error": 30}  # percent (lower is worse)
        }
        
        # Merge custom thresholds with defaults
        if custom_thresholds:
            for metric, thresholds in custom_thresholds.items():
                if metric in default_thresholds:
                    default_thresholds[metric].update(thresholds)
        
        # Determine panel layout based on visualization style
        panel_layouts = {
            "detailed": {
                "metrics_per_row": 2,  # Two metrics per row
                "metric_width": 6, 
                "metric_height": 4,
                "summary_width": 12,
                "summary_height": 4,
                "trend_width": 12,
                "trend_height": 6
            },
            "compact": {
                "metrics_per_row": 3,  # Three metrics per row
                "metric_width": 4, 
                "metric_height": 3,
                "summary_width": 12,
                "summary_height": 3,
                "trend_width": 12,
                "trend_height": 4
            },
            "overview": {
                "metrics_per_row": 4,  # Four metrics per row
                "metric_width": 3, 
                "metric_height": 3,
                "summary_width": 12,
                "summary_height": 4,
                "trend_width": 12,
                "trend_height": 6
            }
        }
        
        # Use detailed layout as fallback
        layout = panel_layouts.get(visualization_style, panel_layouts["detailed"])
        
        # Create a panel for each metric
        created_panels = []
        alerts = []
        y_position = 0
        
        # Select appropriate visualization types based on metric
        visualization_types = {
            "query_time": "line",
            "storage_size": "area",
            "index_efficiency": "gauge",
            "vacuum_status": "gauge",
            "compression_ratio": "bar",
            "read_efficiency": "line",
            "write_efficiency": "line",
            "cache_performance": "gauge"
        }
        
        # Define units for metrics
        metric_units = {
            "query_time": "ms",
            "storage_size": "bytes",
            "index_efficiency": "%",
            "vacuum_status": "%",
            "compression_ratio": "ratio",
            "read_efficiency": "records/s",
            "write_efficiency": "records/s",
            "cache_performance": "%"
        }
        
        # First, create the summary panel at the top
        summary_panel = self.create_panel(
            dashboard_id=dashboard_id,
            panel_type="database-summary",
            title="Database Performance Summary",
            data_source={
                "type": "database_performance",
                "query": {
                    "metrics": metrics,
                    "timeframe": "1d",
                    "aggregation": "current"
                }
            },
            position={"x": 0, "y": y_position},
            size={"width": layout["summary_width"], "height": layout["summary_height"]},
            refresh_interval=interval,
            visualization_options={
                "chart_type": "stat-panel",
                "show_thresholds": True,
                "threshold_colors": ["green", "yellow", "red"],
                "layout": "grid",
                "show_status_indicators": status_indicators,
                "show_trend_indicators": True,
                "decimal_places": 1,
                "show_units": True,
                "metric_units": metric_units
            }
        )
        
        if summary_panel["success"]:
            created_panels.append(summary_panel)
            y_position += layout["summary_height"]
        
        # Add multi-metric trend panel if showing history
        if show_history:
            trend_panel = self.create_panel(
                dashboard_id=dashboard_id,
                panel_type="database-metrics-trend",
                title="Database Metrics Trend",
                data_source={
                    "type": "database_performance",
                    "query": {
                        "metrics": metrics[:5],  # Show up to 5 metrics for clarity
                        "timeframe": "30d",
                        "aggregation": "daily",
                        "grouping": "daily",
                        "normalize_values": True  # Normalize values to make them comparable
                    }
                },
                position={"x": 0, "y": y_position},
                size={"width": layout["trend_width"], "height": layout["trend_height"]},
                refresh_interval=interval,
                visualization_options={
                    "chart_type": "line",
                    "show_thresholds": False,
                    "color_scheme": "category10",
                    "legend_position": "right",
                    "y_axis_label": "Normalized Value",
                    "show_points": False,
                    "smooth_lines": True,
                    "fill_opacity": 0.1
                }
            )
            
            if trend_panel["success"]:
                created_panels.append(trend_panel)
                y_position += layout["trend_height"]
        
        # Create individual metric panels
        for i, metric in enumerate(metrics):
            panel_title = f"Database {metric.replace('_', ' ').title()}"
            
            # Calculate position based on layout
            x_position = (i % layout["metrics_per_row"]) * layout["metric_width"]
            if i % layout["metrics_per_row"] == 0 and i > 0:
                y_position += layout["metric_height"]
            
            # Set visualization type based on metric
            chart_type = visualization_types.get(metric, "line")
            
            # Add enhanced visualization options
            viz_options = {
                "chart_type": chart_type,
                "show_thresholds": True,
                "threshold_colors": ["green", "yellow", "red"],
                "color_scheme": "blues",
                "decimal_places": 1,
                "show_units": True,
                "unit": metric_units.get(metric, ""),
                "y_axis_label": f"{metric.replace('_', ' ').title()} ({metric_units.get(metric, '')})"
            }
            
            # Add status indicator config if enabled
            if status_indicators:
                viz_options["show_status_indicator"] = True
                viz_options["status_indicator_position"] = "top-right"
            
            # Add historical data config if enabled
            if show_history:
                viz_options["show_history"] = True
                viz_options["history_points"] = 30  # Show last 30 data points
                viz_options["show_trend_arrow"] = True
            
            # Create the metric panel
            panel_result = self.create_panel(
                dashboard_id=dashboard_id,
                panel_type="database-metric",
                title=panel_title,
                data_source={
                    "type": "database_performance",
                    "query": {
                        "metric": metric,
                        "timeframe": "7d",
                        "aggregation": "avg",
                        "grouping": "hourly",
                        "thresholds": default_thresholds.get(metric, {"warning": None, "error": None})
                    }
                },
                position={"x": x_position, "y": y_position},
                size={"width": layout["metric_width"], "height": layout["metric_height"]},
                refresh_interval=interval,
                visualization_options=viz_options
            )
            
            if panel_result["success"]:
                created_panels.append(panel_result)
                
                # Create alerts if requested
                if create_alerts:
                    # Get thresholds for this metric
                    thresholds = default_thresholds.get(metric, {})
                    
                    if "error" in thresholds and thresholds["error"] is not None:
                        # Determine comparison operator
                        # For metrics where lower is worse, use "lt" (less than)
                        # For metrics where higher is worse, use "gt" (greater than)
                        comparison = "lt" if metric in ["index_efficiency", "vacuum_status", "compression_ratio", 
                                                        "read_efficiency", "write_efficiency", "cache_performance"] else "gt"
                        
                        # Create error-level alert
                        error_alert_result = self.create_alert(
                            name=f"Database {metric.replace('_', ' ').title()} Critical Alert",
                            description=f"Critical alert when {metric.replace('_', ' ')} {comparison} {thresholds['error']} {metric_units.get(metric, '')}",
                            dashboard_id=dashboard_id,
                            panel_id=panel_result["panel_id"],
                            condition={
                                "metric": metric,
                                "comparison": comparison,
                                "value": thresholds["error"]
                            },
                            notification_channels=["email", "slack"],
                            severity="critical",
                            interval=interval
                        )
                        
                        if error_alert_result["success"]:
                            alerts.append(error_alert_result)
                    
                    if "warning" in thresholds and thresholds["warning"] is not None:
                        # Determine comparison operator as above
                        comparison = "lt" if metric in ["index_efficiency", "vacuum_status", "compression_ratio", 
                                                        "read_efficiency", "write_efficiency", "cache_performance"] else "gt"
                        
                        # Create warning-level alert
                        warning_alert_result = self.create_alert(
                            name=f"Database {metric.replace('_', ' ').title()} Warning Alert",
                            description=f"Warning alert when {metric.replace('_', ' ')} {comparison} {thresholds['warning']} {metric_units.get(metric, '')}",
                            dashboard_id=dashboard_id,
                            panel_id=panel_result["panel_id"],
                            condition={
                                "metric": metric,
                                "comparison": comparison,
                                "value": thresholds["warning"]
                            },
                            notification_channels=["email", "slack"],
                            severity="warning",
                            interval=interval * 2  # Less frequent for warnings
                        )
                        
                        if warning_alert_result["success"]:
                            alerts.append(warning_alert_result)
        
        # Return result with detailed information
        return {
            "success": True,
            "dashboard_id": dashboard_id,
            "visualization_style": visualization_style,
            "panels": created_panels,
            "panel_count": len(created_panels),
            "alerts": alerts,
            "alert_count": len(alerts),
            "metrics_monitored": metrics,
            "refresh_interval": interval,
            "thresholds": default_thresholds
        }
    
    def update_database_performance_metrics(
        self,
        dashboard_id: str,
        panel_ids: Optional[List[str]] = None,
        metrics_to_update: Optional[List[str]] = None,
        include_history: bool = True,
        include_thresholds: bool = True,
        format_values: bool = True,
        update_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update database performance metrics in monitoring dashboard.
        
        Args:
            dashboard_id: ID of the dashboard
            panel_ids: List of panel IDs to update (if None, update all database performance panels)
            metrics_to_update: List of specific metrics to update (if None, update all available metrics)
            include_history: Whether to include historical data in the update
            include_thresholds: Whether to include threshold data in the update
            format_values: Whether to format values for display (round numbers, add units)
            update_interval: If provided, update the panel refresh interval
            
        Returns:
            Dictionary with update status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        if not self.db_optimizer:
            return {"success": False, "error": "Database performance optimizer not available"}
        
        try:
            import requests
            
            # If no panel IDs provided, get all database performance panels
            if not panel_ids:
                # Fetch panels from dashboard
                response = requests.get(
                    f"{self.dashboard_url.rstrip('/')}/dashboards/{dashboard_id}/panels",
                    headers=self._get_auth_headers(),
                    timeout=30
                )
                
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Failed to get dashboard panels: {response.status_code} - {response.text}"
                    }
                
                # Filter database performance panels
                panels = response.json().get("panels", [])
                panel_ids = [
                    panel["panel_id"] for panel in panels
                    if panel.get("type", "").startswith("database-")
                ]
            
            # Get database performance metrics
            db_metrics = self.db_optimizer.get_performance_metrics()
            
            # Filter metrics if specific ones are requested
            if metrics_to_update:
                db_metrics = {k: v for k, v in db_metrics.items() if k in metrics_to_update}
            
            # Get overall database status
            overall_status = self.db_optimizer.get_overall_status()
            
            # Define units for metrics
            metric_units = {
                "query_time": "ms",
                "storage_size": "bytes",
                "index_efficiency": "%",
                "vacuum_status": "%",
                "compression_ratio": "ratio",
                "read_efficiency": "records/s",
                "write_efficiency": "records/s",
                "cache_performance": "%"
            }
            
            # Update each panel
            updated_panels = []
            metrics_updated = set()
            
            for panel_id in panel_ids:
                # Get panel details
                panel_response = requests.get(
                    f"{self.dashboard_url.rstrip('/')}/dashboards/{dashboard_id}/panels/{panel_id}",
                    headers=self._get_auth_headers(),
                    timeout=30
                )
                
                if panel_response.status_code != 200:
                    logger.warning(f"Failed to get panel details for {panel_id}: {panel_response.status_code}")
                    continue
                
                panel = panel_response.json()
                panel_type = panel.get("type", "")
                
                # Update panel data based on type
                if panel_type == "database-metric":
                    # Extract metric from data source
                    data_source = panel.get("data_source", {})
                    query = data_source.get("query", {})
                    metric = query.get("metric")
                    
                    if not metric or metric not in db_metrics:
                        continue
                    
                    # Track that we've updated this metric
                    metrics_updated.add(metric)
                    
                    # Get metric data
                    metric_data = db_metrics[metric]
                    
                    # Format value if requested
                    formatted_value = metric_data["value"]
                    if format_values:
                        # Round numeric values to appropriate precision
                        if isinstance(formatted_value, (int, float)):
                            if metric == "storage_size" and formatted_value > 1000000:
                                formatted_value = f"{formatted_value / 1000000:.2f} MB"
                            elif metric in ["index_efficiency", "vacuum_status", "cache_performance"]:
                                formatted_value = f"{formatted_value:.1f}{metric_units.get(metric, '')}"
                            elif metric == "compression_ratio":
                                formatted_value = f"{formatted_value:.2f}x"
                            else:
                                formatted_value = f"{formatted_value:.1f} {metric_units.get(metric, '')}"
                    
                    # Prepare update data
                    update_data = {
                        "value": metric_data["value"],
                        "formatted_value": formatted_value,
                        "previous_value": metric_data.get("previous_value"),
                        "change_pct": metric_data.get("change_pct"),
                        "unit": metric_data.get("unit", metric_units.get(metric, "")),
                        "status": metric_data.get("status", "good"),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Include history if requested
                    if include_history and "history" in metric_data:
                        update_data["history"] = metric_data["history"]
                        
                        # Calculate trend data if history exists
                        if metric_data["history"] and len(metric_data["history"]) > 1:
                            first_value = metric_data["history"][0]
                            last_value = metric_data["history"][-1]
                            if first_value and last_value:
                                trend_pct = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
                                update_data["trend_pct"] = trend_pct
                                update_data["trend_direction"] = "up" if trend_pct > 0 else "down" if trend_pct < 0 else "stable"
                                
                                # Determine if the trend is good or bad
                                # For metrics where higher is better
                                if metric in ["index_efficiency", "vacuum_status", "compression_ratio", 
                                            "read_efficiency", "write_efficiency", "cache_performance"]:
                                    update_data["trend_status"] = "good" if trend_pct > 0 else "warning" if trend_pct < 0 else "neutral"
                                # For metrics where lower is better
                                else:
                                    update_data["trend_status"] = "good" if trend_pct < 0 else "warning" if trend_pct > 0 else "neutral"
                    
                    # Include thresholds if requested
                    if include_thresholds:
                        update_data["thresholds"] = query.get("thresholds", {})
                    
                    # Update panel data
                    update_result = self.update_panel_data(
                        dashboard_id=dashboard_id,
                        panel_id=panel_id,
                        data=update_data
                    )
                    
                    if update_result["success"]:
                        updated_panels.append(update_result)
                
                elif panel_type == "database-summary":
                    # Prepare summary data
                    summary_data = {
                        "metrics": {},
                        "overall_status": overall_status,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Process each metric for the summary
                    for metric_name, metric_data in db_metrics.items():
                        # Format the value for display
                        formatted_value = metric_data["value"]
                        if format_values:
                            # Apply appropriate formatting based on metric type
                            if metric_name == "storage_size" and formatted_value > 1000000:
                                formatted_value = f"{formatted_value / 1000000:.2f} MB"
                            elif metric_name in ["index_efficiency", "vacuum_status", "cache_performance"]:
                                formatted_value = f"{formatted_value:.1f}{metric_units.get(metric_name, '')}"
                            elif metric_name == "compression_ratio":
                                formatted_value = f"{formatted_value:.2f}x"
                            else:
                                formatted_value = f"{formatted_value:.1f} {metric_units.get(metric_name, '')}"
                        
                        # Add to summary data
                        summary_data["metrics"][metric_name] = {
                            "value": metric_data["value"],
                            "formatted_value": formatted_value,
                            "status": metric_data.get("status", "good"),
                            "unit": metric_data.get("unit", metric_units.get(metric_name, "")),
                            "change_pct": metric_data.get("change_pct")
                        }
                    
                    # Update the summary panel
                    update_result = self.update_panel_data(
                        dashboard_id=dashboard_id,
                        panel_id=panel_id,
                        data=summary_data
                    )
                    
                    if update_result["success"]:
                        updated_panels.append(update_result)
                
                elif panel_type == "database-metrics-trend":
                    # Prepare trend data for multiple metrics
                    trend_data = {
                        "metrics": {},
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Extract query details
                    data_source = panel.get("data_source", {})
                    query = data_source.get("query", {})
                    requested_metrics = query.get("metrics", [])
                    
                    # Process each requested metric
                    for metric_name in requested_metrics:
                        if metric_name in db_metrics and "history" in db_metrics[metric_name]:
                            trend_data["metrics"][metric_name] = {
                                "history": db_metrics[metric_name]["history"],
                                "unit": db_metrics[metric_name].get("unit", metric_units.get(metric_name, "")),
                                "status": db_metrics[metric_name].get("status", "good")
                            }
                    
                    # Update the trend panel
                    update_result = self.update_panel_data(
                        dashboard_id=dashboard_id,
                        panel_id=panel_id,
                        data=trend_data
                    )
                    
                    if update_result["success"]:
                        updated_panels.append(update_result)
                
                # Update panel refresh interval if requested
                if update_interval is not None:
                    try:
                        # Update panel configuration with new refresh interval
                        refresh_response = requests.put(
                            f"{self.dashboard_url.rstrip('/')}/dashboards/{dashboard_id}/panels/{panel_id}/config",
                            headers=self._get_auth_headers(),
                            json={"refresh_interval": update_interval},
                            timeout=30
                        )
                        
                        if refresh_response.status_code != 200:
                            logger.warning(f"Failed to update refresh interval for panel {panel_id}: {refresh_response.status_code}")
                    except Exception as refresh_error:
                        logger.warning(f"Error updating refresh interval for panel {panel_id}: {str(refresh_error)}")
            
            # Return result with detailed information
            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "updated_panels": len(updated_panels),
                "metrics_updated": list(metrics_updated),
                "overall_status": overall_status,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error updating database performance metrics: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def setup_real_time_monitoring(
        self,
        hardware_types: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        interval: int = 300,
        dashboard_id: Optional[str] = None,
        create_new_dashboard: bool = True,
        include_database_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Set up real-time monitoring for simulation validation.
        
        Args:
            hardware_types: List of hardware types to monitor
            model_types: List of model types to monitor
            metrics: List of metrics to monitor
            interval: Monitoring interval in seconds
            dashboard_id: ID of the dashboard (if None and create_new_dashboard is False, fails)
            create_new_dashboard: Whether to create a new dashboard if dashboard_id is not provided
            include_database_metrics: Whether to include database performance metrics
            
        Returns:
            Dictionary with setup status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        # If no dashboard ID provided and create_new_dashboard is False, fail
        if not dashboard_id and not create_new_dashboard:
            return {"success": False, "error": "No dashboard ID provided and create_new_dashboard is False"}
        
        # Create a new dashboard if needed
        if not dashboard_id:
            dashboard_title = "Real-Time Monitoring Dashboard"
            if hardware_types and len(hardware_types) == 1:
                dashboard_title += f" - {hardware_types[0]}"
            if model_types and len(model_types) == 1:
                dashboard_title += f" - {model_types[0]}"
            
            dashboard_result = self.create_simulation_validation_dashboard(
                hardware_types=hardware_types,
                model_types=model_types,
                metrics=metrics,
                dashboard_title=dashboard_title,
                include_database_metrics=include_database_metrics
            )
            
            if not dashboard_result["success"]:
                return dashboard_result
            
            dashboard_id = dashboard_result["dashboard_id"]
        
        # Create real-time monitoring configuration
        monitoring_config = {
            "dashboard_id": dashboard_id,
            "hardware_types": hardware_types,
            "model_types": model_types,
            "metrics": metrics,
            "interval": interval,
            "include_database_metrics": include_database_metrics,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "active"
        }
        
        # Store monitoring configuration
        try:
            monitoring_url = f"{self.dashboard_url.rstrip('/')}/monitoring"
            
            response = requests.post(
                monitoring_url,
                headers=self._get_auth_headers(),
                json=monitoring_config,
                timeout=30
            )
            
            if response.status_code in (200, 201):
                result = response.json()
                logger.info(f"Set up real-time monitoring with ID: {result.get('monitoring_id')}")
                return {
                    "success": True,
                    "monitoring_id": result.get("monitoring_id"),
                    "dashboard_id": dashboard_id,
                    "interval": interval,
                    "url": result.get("url")
                }
            else:
                logger.error(f"Failed to set up monitoring: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Failed to set up monitoring: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            logger.error(f"Error setting up real-time monitoring: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_database_performance_dashboard(
        self,
        dashboard_title: str = "Database Performance Dashboard",
        metrics: Optional[List[str]] = None,
        refresh_interval: int = 300,
        visualization_style: str = "detailed",
        create_alerts: bool = True,
        auto_update: bool = True,
        update_interval: int = 3600,  # Default to hourly updates
        custom_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Creates a dedicated dashboard for database performance monitoring.
        
        This method creates a comprehensive dashboard focused on database performance
        metrics, including panels for individual metrics, summary information, and
        trend analysis. It also sets up automated updates if requested.
        
        Args:
            dashboard_title: Title for the dashboard
            metrics: List of metrics to include (if None, include all available metrics)
            refresh_interval: Panel refresh interval in seconds
            visualization_style: Style of visualization ("detailed", "compact", or "overview")
            create_alerts: Whether to create alerts for critical metrics
            auto_update: Whether to set up scheduled updates for the dashboard
            update_interval: Interval between dashboard updates in seconds (if auto_update is True)
            custom_thresholds: Custom threshold values for metrics
            
        Returns:
            Dictionary with dashboard creation status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        if not self.db_optimizer:
            return {"success": False, "error": "Database performance optimizer not available"}
        
        try:
            import requests
            
            # Get an initial set of metrics to determine what's available
            available_metrics = list(self.db_optimizer.get_performance_metrics().keys())
            
            # Filter metrics if specified
            if metrics:
                metrics_to_use = [m for m in metrics if m in available_metrics]
                if not metrics_to_use:  # If none of the requested metrics are available
                    metrics_to_use = available_metrics
            else:
                metrics_to_use = available_metrics
            
            # Create the dashboard
            dashboard_description = "Comprehensive monitoring dashboard for database performance metrics"
            dashboard_result = self.create_dashboard(
                title=dashboard_title,
                description=dashboard_description,
                tags=["database", "performance", "monitoring", "duckdb"]
            )
            
            if not dashboard_result["success"]:
                return dashboard_result
            
            dashboard_id = dashboard_result["dashboard_id"]
            
            # Set up database performance monitoring panels
            monitoring_result = self.setup_database_performance_monitoring(
                dashboard_id=dashboard_id,
                metrics=metrics_to_use,
                interval=refresh_interval,
                create_alerts=create_alerts,
                visualization_style=visualization_style,
                custom_thresholds=custom_thresholds
            )
            
            if not monitoring_result["success"]:
                return monitoring_result
            
            # If auto-update is requested, set up scheduled updates
            update_configuration = None
            if auto_update:
                try:
                    # Create update configuration
                    update_config = {
                        "dashboard_id": dashboard_id,
                        "update_interval": update_interval,
                        "update_type": "database_performance",
                        "configuration": {
                            "metrics": metrics_to_use,
                            "include_history": True,
                            "format_values": True,
                        },
                        "enabled": True
                    }
                    
                    # Create API endpoint
                    update_endpoint = f"{self.dashboard_url.rstrip('/')}/scheduled-updates"
                    
                    # Send request to set up scheduled updates
                    response = requests.post(
                        update_endpoint,
                        headers=self._get_auth_headers(),
                        json=update_config,
                        timeout=30
                    )
                    
                    if response.status_code in [200, 201]:
                        update_configuration = response.json()
                        logger.info(f"Successfully set up scheduled updates for dashboard: {dashboard_id}")
                    else:
                        logger.warning(f"Failed to set up scheduled updates: {response.status_code} - {response.text}")
                except Exception as e:
                    logger.warning(f"Error setting up scheduled updates: {str(e)}")
            
            # Return comprehensive result
            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "dashboard_title": dashboard_title,
                "dashboard_url": dashboard_result.get("url"),
                "panels_created": monitoring_result.get("panel_count", 0),
                "alerts_created": monitoring_result.get("alert_count", 0),
                "metrics_monitored": metrics_to_use,
                "refresh_interval": refresh_interval,
                "visualization_style": visualization_style,
                "auto_update": auto_update,
                "update_interval": update_interval if auto_update else None,
                "update_configuration": update_configuration
            }
        
        except Exception as e:
            logger.error(f"Error creating database performance dashboard: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_complete_monitoring_solution(
        self,
        dashboard_title: str = "Simulation & Database Monitoring Dashboard",
        include_database_performance: bool = True,
        include_validation_metrics: bool = True,
        hardware_types: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        performance_metrics: Optional[List[str]] = None,
        database_metrics: Optional[List[str]] = None,
        refresh_interval: int = 300,
        create_alerts: bool = True,
        visualization_style: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Creates a complete monitoring solution including both simulation validation 
        and database performance monitoring in a single dashboard.
        
        Args:
            dashboard_title: Title for the dashboard
            include_database_performance: Whether to include database performance monitoring
            include_validation_metrics: Whether to include simulation validation monitoring
            hardware_types: List of hardware types to monitor (for validation metrics)
            model_types: List of model types to monitor (for validation metrics)
            performance_metrics: List of simulation performance metrics to monitor
            database_metrics: List of database metrics to monitor
            refresh_interval: Panel refresh interval in seconds
            create_alerts: Whether to create alerts for critical metrics
            visualization_style: Style of visualization ("detailed", "compact", or "overview")
            
        Returns:
            Dictionary with dashboard creation status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        # Validate requirements
        if include_database_performance and not self.db_optimizer:
            return {"success": False, "error": "Database performance optimizer not available"}
        
        try:
            import requests
            
            # Set up default metrics if not provided
            if not performance_metrics:
                performance_metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"]
            
            if include_database_performance and not database_metrics and self.db_optimizer:
                # Get available database metrics
                try:
                    database_metrics = list(self.db_optimizer.get_performance_metrics().keys())
                except Exception:
                    database_metrics = [
                        "query_time", "storage_size", "index_efficiency", 
                        "vacuum_status", "compression_ratio"
                    ]
            
            # Create the dashboard
            tags = ["monitoring", "dashboard"]
            if include_database_performance:
                tags.append("database")
            if include_validation_metrics:
                tags.append("simulation")
            
            dashboard_result = self.create_dashboard(
                title=dashboard_title,
                description="Comprehensive monitoring solution for simulation validation and database performance",
                tags=tags
            )
            
            if not dashboard_result["success"]:
                return dashboard_result
            
            dashboard_id = dashboard_result["dashboard_id"]
            
            # Track created components
            components = []
            
            # 1. Add database performance panels if requested
            if include_database_performance and self.db_optimizer:
                # Determine layout position for database panels
                db_y_position = 0
                
                # Add database performance panels
                db_result = self.setup_database_performance_monitoring(
                    dashboard_id=dashboard_id,
                    metrics=database_metrics,
                    interval=refresh_interval,
                    create_alerts=create_alerts,
                    visualization_style=visualization_style
                )
                
                if db_result["success"]:
                    components.append({
                        "type": "database_performance",
                        "panels": db_result.get("panel_count", 0),
                        "alerts": db_result.get("alert_count", 0),
                        "metrics": database_metrics
                    })
                    
                    # Update total panel height based on visualization style
                    panel_height = len(database_metrics) * (4 if visualization_style == "detailed" else 3) // 2 + 8
                    db_y_position += panel_height
            
            # 2. Add simulation validation panels if requested
            if include_validation_metrics:
                # Determine position for simulation panels (below database panels if present)
                sim_y_position = db_y_position if include_database_performance else 0
                
                # For each hardware and model type combination, add relevant panels
                if hardware_types and model_types:
                    for hardware_type in hardware_types:
                        for model_type in model_types:
                            # Create MAPE comparison panel
                            mape_panel = self.create_panel(
                                dashboard_id=dashboard_id,
                                panel_type="mape-comparison",
                                title=f"MAPE Comparison - {model_type} on {hardware_type}",
                                data_source={
                                    "type": "simulation_validation",
                                    "query": {
                                        "hardware_types": [hardware_type],
                                        "model_types": [model_type],
                                        "metrics": performance_metrics,
                                        "aggregation": "avg",
                                        "timeframe": "7d"
                                    }
                                },
                                position={"x": 0, "y": sim_y_position},
                                size={"width": 6, "height": 6},
                                refresh_interval=refresh_interval
                            )
                            
                            if mape_panel["success"]:
                                components.append({
                                    "type": "simulation_validation",
                                    "panel_type": "mape_comparison",
                                    "hardware_type": hardware_type,
                                    "model_type": model_type
                                })
                            
                            # Create time series panel for each metric
                            for i, metric in enumerate(performance_metrics):
                                # Calculate position
                                x_position = 6 if i % 2 == 0 else 0
                                y_position = sim_y_position + (i // 2) * 3
                                
                                # Create panel
                                ts_panel = self.create_panel(
                                    dashboard_id=dashboard_id,
                                    panel_type="time-series",
                                    title=f"{metric.replace('_', ' ').title()} - {model_type} on {hardware_type}",
                                    data_source={
                                        "type": "simulation_validation",
                                        "query": {
                                            "hardware_types": [hardware_type],
                                            "model_types": [model_type],
                                            "metrics": [metric],
                                            "timeframe": "30d",
                                            "aggregation": "avg",
                                            "grouping": "daily"
                                        }
                                    },
                                    position={"x": x_position, "y": y_position},
                                    size={"width": 6, "height": 3},
                                    refresh_interval=refresh_interval
                                )
                                
                                if ts_panel["success"]:
                                    components.append({
                                        "type": "simulation_validation",
                                        "panel_type": "time_series",
                                        "metric": metric,
                                        "hardware_type": hardware_type,
                                        "model_type": model_type
                                    })
                            
                            # Update position for next hardware/model combination
                            sim_y_position += len(performance_metrics) * 2
            
            # 3. Set up scheduled updates for all panels
            update_config = {
                "dashboard_id": dashboard_id,
                "update_interval": refresh_interval * 10,  # Less frequent than panel refresh
                "components": components,
                "enabled": True
            }
            
            # Create API endpoint for scheduled updates
            update_endpoint = f"{self.dashboard_url.rstrip('/')}/scheduled-updates"
            
            # Send request to set up scheduled updates
            response = requests.post(
                update_endpoint,
                headers=self._get_auth_headers(),
                json=update_config,
                timeout=30
            )
            
            update_id = None
            if response.status_code in [200, 201]:
                update_result = response.json()
                update_id = update_result.get("update_id")
                logger.info(f"Successfully set up scheduled updates for dashboard: {dashboard_id}")
            
            # Return comprehensive result
            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "dashboard_title": dashboard_title,
                "dashboard_url": dashboard_result.get("url"),
                "components": components,
                "component_count": len(components),
                "includes_database_performance": include_database_performance,
                "includes_validation_metrics": include_validation_metrics,
                "refresh_interval": refresh_interval,
                "visualization_style": visualization_style,
                "scheduled_update_id": update_id
            }
        
        except Exception as e:
            logger.error(f"Error creating complete monitoring solution: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def update_simulation_validation_metrics(
        self,
        dashboard_id: str,
        panel_ids: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update simulation validation metrics in monitoring dashboard.
        
        Args:
            dashboard_id: ID of the dashboard
            panel_ids: List of panel IDs to update (if None, update all simulation validation panels)
            data: Data to update the panels with (if None, fetch from database)
            
        Returns:
            Dictionary with update status and details
        """
        if not self._ensure_connection():
            return {"success": False, "error": "Not connected to dashboard"}
        
        if not self.db_connector:
            return {"success": False, "error": "Database connector not available"}
        
        try:
            # If no panel IDs provided, get all simulation validation panels
            if not panel_ids:
                # Fetch panels from dashboard
                response = requests.get(
                    f"{self.dashboard_url.rstrip('/')}/dashboards/{dashboard_id}/panels",
                    headers=self._get_auth_headers(),
                    timeout=30
                )
                
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Failed to get dashboard panels: {response.status_code} - {response.text}"
                    }
                
                # Filter simulation validation panels
                panels = response.json().get("panels", [])
                panel_ids = [
                    panel["panel_id"] for panel in panels
                    if not panel.get("type", "").startswith("database-")
                ]
            
            # If no data provided, fetch from database
            if not data:
                # TODO: Implement data fetching from database
                pass
            
            # Update each panel
            updated_panels = []
            
            for panel_id in panel_ids:
                # Get panel details
                panel_response = requests.get(
                    f"{self.dashboard_url.rstrip('/')}/dashboards/{dashboard_id}/panels/{panel_id}",
                    headers=self._get_auth_headers(),
                    timeout=30
                )
                
                if panel_response.status_code != 200:
                    logger.warning(f"Failed to get panel details for {panel_id}: {panel_response.status_code}")
                    continue
                
                panel = panel_response.json()
                panel_type = panel.get("type", "")
                
                # Update panel data based on type
                # TODO: Implement panel-specific data updates
                
                # For now, update with the provided data
                if data:
                    update_result = self.update_panel_data(
                        dashboard_id=dashboard_id,
                        panel_id=panel_id,
                        data=data
                    )
                    
                    if update_result["success"]:
                        updated_panels.append(update_result)
            
            # Return result
            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "updated_panels": len(updated_panels),
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error updating simulation validation metrics: {str(e)}")
            return {"success": False, "error": str(e)}


def get_monitoring_dashboard_connector(
    dashboard_url: str,
    dashboard_api_key: str,
    db_connector: Optional[ValidationVisualizerDBConnector] = None,
    db_optimizer: Optional[DatabasePerformanceOptimizer] = None,
    config: Optional[Dict[str, Any]] = None
) -> MonitoringDashboardConnector:
    """
    Get a monitoring dashboard connector instance.
    
    Args:
        dashboard_url: URL of the monitoring dashboard API
        dashboard_api_key: API key for the dashboard
        db_connector: ValidationVisualizerDBConnector instance
        db_optimizer: DatabasePerformanceOptimizer instance
        config: Configuration for the connector
        
    Returns:
        MonitoringDashboardConnector instance
    """
    return MonitoringDashboardConnector(
        dashboard_url=dashboard_url,
        dashboard_api_key=dashboard_api_key,
        db_connector=db_connector,
        db_optimizer=db_optimizer,
        config=config
    )