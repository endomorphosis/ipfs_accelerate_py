#!/usr/bin/env python3
"""
Database Connector for the Validation Visualizer in the Simulation Accuracy and Validation Framework.

This module provides a connector between the database integration and visualization components,
allowing visualization to be generated directly from database queries.
"""

import os
import sys
import logging
import json
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("validation_visualizer_db_connector")

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


class ValidationVisualizerDBConnector:
    """
    Connector between the database integration and visualization components.
    
    This class retrieves data from the database and formats it for use with the
    ValidationVisualizer, enabling visualization directly from database queries.
    It also provides integration with the monitoring dashboard for real-time visualization.
    """
    
    def __init__(
        self,
        db_integration: Optional[SimulationValidationDBIntegration] = None,
        visualizer: Optional[ValidationVisualizer] = None,
        db_path: str = "./benchmark_db.duckdb",
        visualization_config: Optional[Dict[str, Any]] = None,
        dashboard_integration: bool = False,
        dashboard_url: Optional[str] = None,
        dashboard_api_key: Optional[str] = None
    ):
        """
        Initialize the connector.
        
        Args:
            db_integration: SimulationValidationDBIntegration instance
            visualizer: ValidationVisualizer instance
            db_path: Path to the DuckDB database (used if db_integration is None)
            visualization_config: Configuration for the visualizer
            dashboard_integration: Whether to enable monitoring dashboard integration
            dashboard_url: URL of the monitoring dashboard API (if integration is enabled)
            dashboard_api_key: API key for the dashboard (if integration is enabled)
        """
        # Initialize database integration
        self.db_integration = db_integration or SimulationValidationDBIntegration(db_path=db_path)
        
        # Initialize visualizer
        self.visualizer = visualizer or ValidationVisualizer(config=visualization_config)
        
        # Set up dashboard integration
        self.dashboard_integration = dashboard_integration
        self.dashboard_url = dashboard_url
        self.dashboard_api_key = dashboard_api_key
        self.dashboard_connected = False
        self.dashboard_session_token = None
        self.dashboard_session_expires = None
        
        # If dashboard integration is enabled, try to establish connection
        if self.dashboard_integration and self.dashboard_url:
            self._connect_to_dashboard()
        
        logger.info("ValidationVisualizerDBConnector initialized")
    
    def _connect_to_dashboard(self) -> bool:
        """
        Establishes a connection to the monitoring dashboard.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.dashboard_url or not self.dashboard_api_key:
            logger.warning("Dashboard URL or API key not provided. Cannot connect to dashboard.")
            return False
            
        try:
            import requests
            from urllib.parse import urljoin
            
            # Create authentication request
            auth_endpoint = urljoin(self.dashboard_url, "/auth")
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.dashboard_api_key
            }
            
            # Attempt to authenticate with the dashboard
            response = requests.post(
                auth_endpoint,
                headers=headers,
                json={"client": "simulation_validation_framework"}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.dashboard_session_token = data.get("token")
                # Store expiration time (default to 24 hours if not provided)
                expires_in = data.get("expires_in", 86400)  # 24 hours in seconds
                self.dashboard_session_expires = datetime.datetime.now() + datetime.timedelta(seconds=expires_in)
                self.dashboard_connected = True
                logger.info("Successfully connected to monitoring dashboard")
                return True
            else:
                logger.error(f"Failed to connect to dashboard. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                self.dashboard_connected = False
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to dashboard: {str(e)}")
            self.dashboard_connected = False
            return False
    
    def _ensure_dashboard_connection(self) -> bool:
        """
        Ensures an active connection to the dashboard, reconnecting if necessary.
        
        Returns:
            bool: True if connection is active, False otherwise
        """
        # If dashboard integration is not enabled, return False
        if not self.dashboard_integration:
            return False
            
        # If not connected, try to connect
        if not self.dashboard_connected:
            return self._connect_to_dashboard()
            
        # If token is expired or about to expire, reconnect
        if (self.dashboard_session_expires and 
            datetime.datetime.now() > self.dashboard_session_expires - datetime.timedelta(minutes=5)):
            logger.info("Dashboard session token expired or about to expire. Reconnecting...")
            return self._connect_to_dashboard()
            
        return True
    
    def create_dashboard_panel_from_db(
        self,
        panel_type: str,
        hardware_type: Optional[str] = None,
        model_type: Optional[str] = None,
        metric: Optional[str] = None,
        dashboard_id: Optional[str] = None,
        panel_title: Optional[str] = None,
        refresh_interval: int = 60,
        width: int = 6,
        height: int = 4,
        panel_description: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        position: Optional[int] = None,
        **additional_params
    ) -> Dict[str, Any]:
        """
        Creates a visualization panel in the monitoring dashboard.
        
        Args:
            panel_type: Type of panel to create (mape_comparison, hardware_heatmap, time_series, etc.)
            hardware_type: Hardware type for the panel
            model_type: Model type for the panel
            metric: Metric to display
            dashboard_id: ID of the dashboard to add the panel to
            panel_title: Title for the panel
            refresh_interval: How often to refresh the panel data (in seconds)
            width: Panel width in grid units
            height: Panel height in grid units
            panel_description: Optional description for the panel
            start_date: Start date for data filtering (ISO format)
            end_date: End date for data filtering (ISO format)
            position: Optional position index for the panel
            **additional_params: Additional parameters for the panel
            
        Returns:
            Dict with panel creation status and details
        """
        # Ensure dashboard connection
        if not self._ensure_dashboard_connection():
            logger.warning("Cannot create dashboard panel: Not connected to dashboard")
            return {"success": False, "error": "Not connected to dashboard"}
            
        try:
            import requests
            from urllib.parse import urljoin
            
            # Construct panel configuration
            panel_config = {
                "panel_type": panel_type,
                "hardware_type": hardware_type,
                "model_type": model_type,
                "metric": metric,
                "dashboard_id": dashboard_id,
                "title": panel_title,
                "refresh_interval": refresh_interval,
                "width": width,
                "height": height,
                "description": panel_description,
                "filters": {}
            }
            
            # Add position if provided
            if position is not None:
                panel_config["position"] = position
            
            # Add date filters if provided
            if start_date:
                panel_config["filters"]["start_date"] = start_date
            if end_date:
                panel_config["filters"]["end_date"] = end_date
                
            # Add any additional parameters
            panel_config.update(additional_params)
            
            # Create API endpoint
            panel_endpoint = urljoin(self.dashboard_url, "/panels")
            
            # Set up headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.dashboard_session_token}"
            }
            
            # Send request to create panel
            response = requests.post(
                panel_endpoint,
                headers=headers,
                json=panel_config
            )
            
            if response.status_code in [200, 201]:
                panel_data = response.json()
                logger.info(f"Successfully created dashboard panel: {panel_title or panel_type}")
                return {"success": True, "panel_id": panel_data.get("panel_id"), "data": panel_data}
            else:
                logger.error(f"Failed to create dashboard panel. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {"success": False, "error": f"API Error: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error creating dashboard panel: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_comprehensive_monitoring_dashboard(
        self,
        dashboard_title: str,
        hardware_type: Optional[str] = None,
        model_type: Optional[str] = None,
        dashboard_description: Optional[str] = None,
        refresh_interval: int = 60,
        include_panels: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        **additional_params
    ) -> Dict[str, Any]:
        """
        Creates a comprehensive monitoring dashboard with multiple panels.
        
        Args:
            dashboard_title: Title for the dashboard
            hardware_type: Hardware type for filtering
            model_type: Model type for filtering
            dashboard_description: Description for the dashboard
            refresh_interval: Default refresh interval for panels (in seconds)
            include_panels: List of panel types to include
            metrics: List of metrics to display
            **additional_params: Additional parameters for the dashboard
            
        Returns:
            Dict with dashboard creation status and details
        """
        # Ensure dashboard connection
        if not self._ensure_dashboard_connection():
            logger.warning("Cannot create comprehensive dashboard: Not connected to dashboard")
            return {"success": False, "error": "Not connected to dashboard"}
            
        try:
            import requests
            from urllib.parse import urljoin
            
            # Default panels to include if not specified
            if include_panels is None:
                include_panels = [
                    "mape_comparison",
                    "hardware_heatmap",
                    "time_series",
                    "simulation_vs_hardware",
                    "drift_detection",
                    "calibration_effectiveness"
                ]
                
            # Default metrics to include if not specified
            if metrics is None:
                metrics = ["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"]
                
            # Create dashboard configuration
            dashboard_config = {
                "title": dashboard_title,
                "description": dashboard_description,
                "filters": {}
            }
            
            # Add hardware and model filters if provided
            if hardware_type:
                dashboard_config["filters"]["hardware_type"] = hardware_type
            if model_type:
                dashboard_config["filters"]["model_type"] = model_type
                
            # Add any additional parameters
            dashboard_config.update(additional_params)
            
            # Create API endpoint
            dashboard_endpoint = urljoin(self.dashboard_url, "/dashboards")
            
            # Set up headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.dashboard_session_token}"
            }
            
            # Send request to create dashboard
            response = requests.post(
                dashboard_endpoint,
                headers=headers,
                json=dashboard_config
            )
            
            if response.status_code in [200, 201]:
                dashboard_data = response.json()
                dashboard_id = dashboard_data.get("dashboard_id")
                
                logger.info(f"Successfully created dashboard: {dashboard_title}")
                
                # Add panels to the dashboard
                added_panels = []
                
                # Panel layout configuration
                layout_configs = {
                    "mape_comparison": {"width": 6, "height": 4, "position": 0},
                    "hardware_heatmap": {"width": 6, "height": 4, "position": 1},
                    "time_series": {"width": 12, "height": 4, "position": 2},
                    "simulation_vs_hardware": {"width": 6, "height": 4, "position": 3},
                    "drift_detection": {"width": 6, "height": 4, "position": 4},
                    "calibration_effectiveness": {"width": 12, "height": 4, "position": 5}
                }
                
                # Create each requested panel
                for panel_type in include_panels:
                    layout = layout_configs.get(panel_type, {"width": 6, "height": 4})
                    
                    for metric in metrics:
                        panel_title = f"{panel_type.replace('_', ' ').title()} - {metric}"
                        
                        panel_result = self.create_dashboard_panel_from_db(
                            panel_type=panel_type,
                            hardware_type=hardware_type,
                            model_type=model_type,
                            metric=metric,
                            dashboard_id=dashboard_id,
                            panel_title=panel_title,
                            refresh_interval=refresh_interval,
                            width=layout["width"],
                            height=layout["height"],
                            position=layout.get("position")
                        )
                        
                        if panel_result["success"]:
                            added_panels.append(panel_result["panel_id"])
                
                return {
                    "success": True, 
                    "dashboard_id": dashboard_id, 
                    "url": urljoin(self.dashboard_url.replace('/api', ''), f"/dashboard/{dashboard_id}"),
                    "panels_added": len(added_panels),
                    "panel_ids": added_panels
                }
            else:
                logger.error(f"Failed to create dashboard. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {"success": False, "error": f"API Error: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def set_up_real_time_monitoring(
        self,
        hardware_type: str,
        model_type: str,
        metrics: List[str],
        monitoring_interval: int = 300,
        alert_thresholds: Optional[Dict[str, float]] = None,
        dashboard_id: Optional[str] = None,
        create_new_dashboard: bool = False,
        dashboard_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sets up real-time monitoring with alerting capabilities.
        
        Args:
            hardware_type: Hardware type to monitor
            model_type: Model type to monitor
            metrics: List of metrics to monitor
            monitoring_interval: Interval between data checks (in seconds)
            alert_thresholds: Dict mapping metrics to alert thresholds
            dashboard_id: Existing dashboard ID to add monitoring panels to
            create_new_dashboard: Whether to create a new dashboard if dashboard_id is not provided
            dashboard_title: Title for the new dashboard (if creating one)
            
        Returns:
            Dict with monitoring setup status and details
        """
        # Ensure dashboard connection
        if not self._ensure_dashboard_connection():
            logger.warning("Cannot set up real-time monitoring: Not connected to dashboard")
            return {"success": False, "error": "Not connected to dashboard"}
            
        try:
            import requests
            from urllib.parse import urljoin
            
            # If no dashboard_id provided and create_new_dashboard is True, create a new dashboard
            if not dashboard_id and create_new_dashboard:
                if not dashboard_title:
                    dashboard_title = f"Real-Time Monitoring - {model_type} on {hardware_type}"
                    
                dashboard_result = self.create_comprehensive_monitoring_dashboard(
                    dashboard_title=dashboard_title,
                    hardware_type=hardware_type,
                    model_type=model_type,
                    dashboard_description=f"Real-time monitoring dashboard for {model_type} on {hardware_type}",
                    refresh_interval=monitoring_interval,
                    include_panels=["time_series", "drift_detection"],
                    metrics=metrics
                )
                
                if dashboard_result["success"]:
                    dashboard_id = dashboard_result["dashboard_id"]
                else:
                    return dashboard_result  # Return the error
            
            # Default thresholds if not provided
            if alert_thresholds is None:
                alert_thresholds = {metric: 15.0 for metric in metrics}
                
            # Create monitoring configuration
            monitoring_config = {
                "hardware_type": hardware_type,
                "model_type": model_type,
                "metrics": metrics,
                "monitoring_interval": monitoring_interval,
                "alert_thresholds": alert_thresholds,
                "dashboard_id": dashboard_id
            }
            
            # Create API endpoint
            monitoring_endpoint = urljoin(self.dashboard_url, "/monitoring")
            
            # Set up headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.dashboard_session_token}"
            }
            
            # Send request to set up monitoring
            response = requests.post(
                monitoring_endpoint,
                headers=headers,
                json=monitoring_config
            )
            
            if response.status_code in [200, 201]:
                monitoring_data = response.json()
                logger.info(f"Successfully set up real-time monitoring for {model_type} on {hardware_type}")
                
                return {
                    "success": True,
                    "monitoring_id": monitoring_data.get("monitoring_id"),
                    "dashboard_id": dashboard_id,
                    "dashboard_url": urljoin(self.dashboard_url.replace('/api', ''), f"/dashboard/{dashboard_id}") if dashboard_id else None,
                    "alert_thresholds": alert_thresholds
                }
            else:
                logger.error(f"Failed to set up monitoring. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {"success": False, "error": f"API Error: {response.text}"}
                
        except Exception as e:
            logger.error(f"Error setting up real-time monitoring: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_mape_comparison_chart_from_db(
        self,
        hardware_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        metric_name: str = "all",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        dashboard_id: Optional[str] = None,
        create_dashboard_panel: bool = False
    ) -> Union[str, Dict[str, Any], None]:
        """
        Create a MAPE comparison chart using data from the database.
        
        Args:
            hardware_ids: List of hardware IDs to include
            model_ids: List of model IDs to include
            metric_name: Metric to visualize (or "all" for average)
            start_date: Start date for filtering results
            end_date: End date for filtering results
            interactive: Whether to create an interactive visualization
            output_path: Path to save the visualization
            title: Title for the visualization
            dashboard_id: ID of the dashboard to add the panel to (if create_dashboard_panel is True)
            create_dashboard_panel: Whether to create a dashboard panel instead of a local file
            
        Returns:
            If create_dashboard_panel is True, returns a dictionary with panel creation status.
            Otherwise, returns the path to the generated visualization file or None if unsuccessful.
        """
        # Check if we should create a dashboard panel instead of a local file
        if create_dashboard_panel and self.dashboard_integration:
            if not self._ensure_dashboard_connection():
                logger.warning("Cannot create dashboard panel: Not connected to dashboard")
                return {"success": False, "error": "Not connected to dashboard"}
                
            # Create panel in dashboard
            panel_title = title or f"MAPE Comparison - {metric_name}"
            return self.create_dashboard_panel_from_db(
                panel_type="mape_comparison",
                hardware_type=hardware_ids[0] if hardware_ids and len(hardware_ids) == 1 else None,
                model_type=model_ids[0] if model_ids and len(model_ids) == 1 else None,
                metric=metric_name,
                dashboard_id=dashboard_id,
                panel_title=panel_title,
                start_date=start_date,
                end_date=end_date,
                hardware_ids=hardware_ids,
                model_ids=model_ids
            )
            
        try:
            # Get validation results from the database
            validation_results = self._get_validation_results_from_db(
                hardware_ids=hardware_ids,
                model_ids=model_ids,
                start_date=start_date,
                end_date=end_date
            )
            
            if not validation_results:
                logger.warning("No validation results found in database")
                return None
            
            # Create visualization
            return self.visualizer.create_mape_comparison_chart(
                validation_results=validation_results,
                metric_name=metric_name,
                output_path=output_path,
                interactive=interactive,
                title=title
            )
            
        except Exception as e:
            logger.error(f"Error creating MAPE comparison chart: {str(e)}")
            return None
    
    def create_hardware_comparison_heatmap_from_db(
        self,
        metric_name: str = "throughput_items_per_second",
        model_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        dashboard_id: Optional[str] = None,
        create_dashboard_panel: bool = False
    ) -> Union[str, Dict[str, Any], None]:
        """
        Create a hardware comparison heatmap using data from the database.
        
        Args:
            metric_name: Metric to visualize
            model_ids: List of model IDs to include
            start_date: Start date for filtering results
            end_date: End date for filtering results
            interactive: Whether to create an interactive visualization
            output_path: Path to save the visualization
            title: Title for the visualization
            dashboard_id: ID of the dashboard to add the panel to (if create_dashboard_panel is True)
            create_dashboard_panel: Whether to create a dashboard panel instead of a local file
            
        Returns:
            If create_dashboard_panel is True, returns a dictionary with panel creation status.
            Otherwise, returns the path to the generated visualization file or None if unsuccessful.
        """
        # Check if we should create a dashboard panel instead of a local file
        if create_dashboard_panel and self.dashboard_integration:
            if not self._ensure_dashboard_connection():
                logger.warning("Cannot create dashboard panel: Not connected to dashboard")
                return {"success": False, "error": "Not connected to dashboard"}
                
            # Create panel in dashboard
            panel_title = title or f"Hardware Comparison Heatmap - {metric_name}"
            return self.create_dashboard_panel_from_db(
                panel_type="hardware_heatmap",
                metric=metric_name,
                dashboard_id=dashboard_id,
                panel_title=panel_title,
                start_date=start_date,
                end_date=end_date,
                model_ids=model_ids
            )
            
        try:
            # Get validation results from the database
            validation_results = self._get_validation_results_from_db(
                model_ids=model_ids,
                start_date=start_date,
                end_date=end_date
            )
            
            if not validation_results:
                logger.warning("No validation results found in database")
                return None
            
            # Extract hardware models and model types
            hardware_models = list(set(vr.hardware_result.hardware_id for vr in validation_results))
            model_types = list(set(vr.simulation_result.model_id for vr in validation_results))
            
            # Calculate MAPE values for each hardware and model combination
            mape_values = []
            for hw in hardware_models:
                hw_mapes = []
                for model in model_types:
                    # Filter validation results for this hardware and model
                    filtered_results = [
                        vr for vr in validation_results 
                        if vr.hardware_result.hardware_id == hw and vr.simulation_result.model_id == model
                    ]
                    
                    # Calculate average MAPE for this combination
                    if filtered_results:
                        metric_mapes = [
                            vr.metrics_comparison.get(metric_name, {}).get('mape', 0) 
                            for vr in filtered_results 
                            if metric_name in vr.metrics_comparison
                        ]
                        avg_mape = sum(metric_mapes) / len(metric_mapes) if metric_mapes else 0
                        hw_mapes.append(avg_mape)
                    else:
                        hw_mapes.append(0)
                mape_values.append(hw_mapes)
            
            # Create visualization
            return self.visualizer.create_hardware_comparison_heatmap(
                hardware_models=hardware_models,
                model_types=model_types,
                mape_values=mape_values,
                metric_name=metric_name,
                output_path=output_path,
                interactive=interactive,
                title=title
            )
            
        except Exception as e:
            logger.error(f"Error creating hardware comparison heatmap: {str(e)}")
            return None
    
    def create_time_series_chart_from_db(
        self,
        metric_name: str,
        hardware_id: str,
        model_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        dashboard_id: Optional[str] = None,
        create_dashboard_panel: bool = False
    ) -> Union[str, Dict[str, Any], None]:
        """
        Create a time series chart using data from the database.
        
        Args:
            metric_name: Metric to visualize
            hardware_id: Hardware ID to include
            model_id: Model ID to include
            start_date: Start date for filtering results
            end_date: End date for filtering results
            interactive: Whether to create an interactive visualization
            output_path: Path to save the visualization
            title: Title for the visualization
            dashboard_id: ID of the dashboard to add the panel to (if create_dashboard_panel is True)
            create_dashboard_panel: Whether to create a dashboard panel instead of a local file
            
        Returns:
            If create_dashboard_panel is True, returns a dictionary with panel creation status.
            Otherwise, returns the path to the generated visualization file or None if unsuccessful.
        """
        # Check if we should create a dashboard panel instead of a local file
        if create_dashboard_panel and self.dashboard_integration:
            if not self._ensure_dashboard_connection():
                logger.warning("Cannot create dashboard panel: Not connected to dashboard")
                return {"success": False, "error": "Not connected to dashboard"}
                
            # Create panel in dashboard
            panel_title = title or f"Time Series - {metric_name}"
            return self.create_dashboard_panel_from_db(
                panel_type="time_series",
                hardware_type=hardware_id,
                model_type=model_id,
                metric=metric_name,
                dashboard_id=dashboard_id,
                panel_title=panel_title,
                start_date=start_date,
                end_date=end_date
            )
            
        try:
            # Get validation results from the database
            validation_results = self._get_validation_results_from_db(
                hardware_ids=[hardware_id],
                model_ids=[model_id],
                start_date=start_date,
                end_date=end_date
            )
            
            if not validation_results:
                logger.warning("No validation results found in database")
                return None
            
            # Sort validation results by timestamp
            validation_results = sorted(
                validation_results, 
                key=lambda vr: vr.validation_timestamp
            )
            
            # Extract timestamps and values for the specified metric
            timestamps = [vr.validation_timestamp for vr in validation_results]
            simulation_values = [
                vr.metrics_comparison.get(metric_name, {}).get('simulation_value', 0) 
                for vr in validation_results
            ]
            hardware_values = [
                vr.metrics_comparison.get(metric_name, {}).get('hardware_value', 0) 
                for vr in validation_results
            ]
            
            # Create visualization
            return self.visualizer.create_time_series(
                timestamps=timestamps,
                simulation_values=simulation_values,
                hardware_values=hardware_values,
                metric_name=metric_name,
                output_path=output_path,
                interactive=interactive,
                title=title
            )
            
        except Exception as e:
            logger.error(f"Error creating time series chart: {str(e)}")
            return None
    
    def create_drift_visualization_from_db(
        self,
        hardware_type: str,
        model_type: str,
        metrics: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        dashboard_id: Optional[str] = None,
        create_dashboard_panel: bool = False
    ) -> Union[str, Dict[str, Any], None]:
        """
        Create a drift visualization using data from the database.
        
        Args:
            hardware_type: Hardware type to analyze
            model_type: Model type to analyze
            metrics: List of metrics to visualize (default: all available metrics)
            start_date: Start date for filtering results
            end_date: End date for filtering results
            interactive: Whether to create an interactive visualization
            output_path: Path to save the visualization
            title: Title for the visualization
            dashboard_id: ID of the dashboard to add the panel to (if create_dashboard_panel is True)
            create_dashboard_panel: Whether to create a dashboard panel instead of a local file
            
        Returns:
            If create_dashboard_panel is True, returns a dictionary with panel creation status.
            Otherwise, returns the path to the generated visualization file or None if unsuccessful.
        """
        # Check if we should create a dashboard panel instead of a local file
        if create_dashboard_panel and self.dashboard_integration:
            if not self._ensure_dashboard_connection():
                logger.warning("Cannot create dashboard panel: Not connected to dashboard")
                return {"success": False, "error": "Not connected to dashboard"}
                
            # Create panel in dashboard
            panel_title = title or f"Drift Detection - {model_type} on {hardware_type}"
            return self.create_dashboard_panel_from_db(
                panel_type="drift_detection",
                hardware_type=hardware_type,
                model_type=model_type,
                dashboard_id=dashboard_id,
                panel_title=panel_title,
                start_date=start_date,
                end_date=end_date,
                metrics=metrics
            )
            
        try:
            # Get drift detection results from the database
            drift_results = self.db_integration.get_drift_detection_results_with_filters(
                hardware_type=hardware_type,
                model_type=model_type
            )
            
            if not drift_results:
                logger.warning("No drift detection results found in database")
                return None
            
            # Filter by date if provided
            if start_date:
                drift_results = [dr for dr in drift_results if dr.timestamp >= start_date]
            if end_date:
                drift_results = [dr for dr in drift_results if dr.timestamp <= end_date]
            
            # Sort by timestamp
            drift_results = sorted(drift_results, key=lambda dr: dr.timestamp)
            
            # Filter metrics if provided
            if metrics:
                filtered_drift_results = []
                for dr in drift_results:
                    filtered_metrics = {
                        metric: data for metric, data in dr.drift_metrics.items() 
                        if metric in metrics
                    }
                    dr.drift_metrics = filtered_metrics
                    if filtered_metrics:  # Only include if it has at least one of the requested metrics
                        filtered_drift_results.append(dr)
                drift_results = filtered_drift_results
            
            # Create visualization
            return self.visualizer.create_drift_visualization(
                drift_results=drift_results,
                output_path=output_path,
                interactive=interactive,
                title=title
            )
            
        except Exception as e:
            logger.error(f"Error creating drift visualization: {str(e)}")
            return None
    
    def create_calibration_improvement_chart_from_db(
        self,
        hardware_type: str,
        model_type: str,
        metric_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        dashboard_id: Optional[str] = None,
        create_dashboard_panel: bool = False
    ) -> Union[str, Dict[str, Any], None]:
        """
        Create a calibration improvement chart using data from the database.
        
        Args:
            hardware_type: Hardware type to analyze
            model_type: Model type to analyze
            metric_name: Metric to visualize (default: all metrics)
            start_date: Start date for filtering results
            end_date: End date for filtering results
            interactive: Whether to create an interactive visualization
            output_path: Path to save the visualization
            title: Title for the visualization
            dashboard_id: ID of the dashboard to add the panel to (if create_dashboard_panel is True)
            create_dashboard_panel: Whether to create a dashboard panel instead of a local file
            
        Returns:
            If create_dashboard_panel is True, returns a dictionary with panel creation status.
            Otherwise, returns the path to the generated visualization file or None if unsuccessful.
        """
        # Check if we should create a dashboard panel instead of a local file
        if create_dashboard_panel and self.dashboard_integration:
            if not self._ensure_dashboard_connection():
                logger.warning("Cannot create dashboard panel: Not connected to dashboard")
                return {"success": False, "error": "Not connected to dashboard"}
                
            # Create panel in dashboard
            panel_title = title or f"Calibration Improvement - {model_type} on {hardware_type}"
            return self.create_dashboard_panel_from_db(
                panel_type="calibration_effectiveness",
                hardware_type=hardware_type,
                model_type=model_type,
                metric=metric_name,
                dashboard_id=dashboard_id,
                panel_title=panel_title,
                start_date=start_date,
                end_date=end_date
            )
            
        try:
            # Get calibration records from the database
            calibration_records = self.db_integration.get_calibration_records_with_filters(
                hardware_type=hardware_type,
                model_type=model_type
            )
            
            if not calibration_records:
                logger.warning("No calibration records found in database")
                return None
            
            # Filter by date if provided
            if start_date:
                calibration_records = [cr for cr in calibration_records if cr.timestamp >= start_date]
            if end_date:
                calibration_records = [cr for cr in calibration_records if cr.timestamp <= end_date]
            
            # Sort by timestamp
            calibration_records = sorted(calibration_records, key=lambda cr: cr.timestamp)
            
            # Create visualization
            return self.visualizer.create_calibration_improvement_chart(
                calibration_records=calibration_records,
                metric_name=metric_name,
                output_path=output_path,
                interactive=interactive,
                title=title
            )
            
        except Exception as e:
            logger.error(f"Error creating calibration improvement chart: {str(e)}")
            return None
    
    def create_comprehensive_dashboard_from_db(
        self,
        hardware_id: str,
        model_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        dashboard_id: Optional[str] = None,
        create_dashboard: bool = False
    ) -> Union[str, Dict[str, Any], None]:
        """
        Create a comprehensive dashboard using data from the database.
        
        Args:
            hardware_id: Hardware ID to analyze
            model_id: Model ID to analyze
            start_date: Start date for filtering results
            end_date: End date for filtering results
            interactive: Whether to create an interactive visualization
            output_path: Path to save the visualization
            title: Title for the visualization
            dashboard_id: ID of the dashboard (if create_dashboard is True and using existing)
            create_dashboard: Whether to create a dashboard in the monitoring system
            
        Returns:
            If create_dashboard is True, returns a dictionary with dashboard creation status.
            Otherwise, returns the path to the generated visualization file or None if unsuccessful.
        """
        # Check if we should create a monitoring dashboard instead of a local file
        if create_dashboard and self.dashboard_integration:
            if not self._ensure_dashboard_connection():
                logger.warning("Cannot create dashboard: Not connected to dashboard")
                return {"success": False, "error": "Not connected to dashboard"}
                
            # Create comprehensive dashboard in monitoring system
            dashboard_title = title or f"Comprehensive Dashboard - {model_id} on {hardware_id}"
            return self.create_comprehensive_monitoring_dashboard(
                dashboard_title=dashboard_title,
                hardware_type=hardware_id,
                model_type=model_id,
                dashboard_description=f"Comprehensive monitoring dashboard for {model_id} on {hardware_id}",
                dashboard_id=dashboard_id
            )
            
        try:
            # Get validation results from the database
            validation_results = self._get_validation_results_from_db(
                hardware_ids=[hardware_id],
                model_ids=[model_id],
                start_date=start_date,
                end_date=end_date
            )
            
            if not validation_results:
                logger.warning("No validation results found in database")
                return None
            
            # Create local dashboard title
            dashboard_title = title or f"Comprehensive Dashboard - {model_id} on {hardware_id}"
            
            # Define visualizations to include
            visualizations = []
            
            # Add MAPE comparison chart
            mape_chart_path = self.create_mape_comparison_chart_from_db(
                hardware_ids=[hardware_id],
                model_ids=[model_id],
                metric_name="all",
                start_date=start_date,
                end_date=end_date,
                interactive=interactive,
                output_path=os.path.join(os.path.dirname(output_path), "mape_comparison.html") if output_path else None,
                title=f"MAPE Comparison - {model_id} on {hardware_id}"
            )
            if mape_chart_path:
                visualizations.append({
                    "title": f"MAPE Comparison - {model_id} on {hardware_id}",
                    "path": mape_chart_path,
                    "type": "mape_comparison"
                })
            
            # Add time series charts for key metrics
            metrics = ["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"]
            for metric in metrics:
                ts_chart_path = self.create_time_series_chart_from_db(
                    metric_name=metric,
                    hardware_id=hardware_id,
                    model_id=model_id,
                    start_date=start_date,
                    end_date=end_date,
                    interactive=interactive,
                    output_path=os.path.join(os.path.dirname(output_path), f"time_series_{metric}.html") if output_path else None,
                    title=f"Time Series - {metric}"
                )
                if ts_chart_path:
                    visualizations.append({
                        "title": f"Time Series - {metric}",
                        "path": ts_chart_path,
                        "type": "time_series"
                    })
            
            # Create the comprehensive dashboard
            return self.visualizer.create_comprehensive_dashboard(
                visualizations=visualizations,
                title=dashboard_title,
                output_path=output_path
            )
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {str(e)}")
            return None
    
    def _get_validation_results_from_db(
        self,
        hardware_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[ValidationResult]:
        """
        Helper method to get validation results from the database with filtering.
        
        Args:
            hardware_ids: List of hardware IDs to filter by
            model_ids: List of model IDs to filter by
            start_date: Start date for filtering results
            end_date: End date for filtering results
            
        Returns:
            List of ValidationResult objects
        """
        # Get validation results from the database with filters
        validation_results = []
        
        if hardware_ids and len(hardware_ids) == 1 and model_ids and len(model_ids) == 1:
            # Simple case: one hardware ID and one model ID
            results = self.db_integration.get_validation_results_by_model_and_hardware(
                model_id=model_ids[0],
                hardware_id=hardware_ids[0]
            )
            validation_results.extend(results)
        elif hardware_ids and len(hardware_ids) == 1:
            # Filter by one hardware ID and multiple model IDs
            for model_id in model_ids:
                results = self.db_integration.get_validation_results_by_model_and_hardware(
                    model_id=model_id,
                    hardware_id=hardware_ids[0]
                )
                validation_results.extend(results)
        elif model_ids and len(model_ids) == 1:
            # Filter by one model ID and multiple hardware IDs
            for hardware_id in hardware_ids:
                results = self.db_integration.get_validation_results_by_model_and_hardware(
                    model_id=model_ids[0],
                    hardware_id=hardware_id
                )
                validation_results.extend(results)
        else:
            # Get all validation results and filter in-memory
            all_results = self.db_integration.get_validation_results_with_filters()
            validation_results = all_results
        
        # Filter by hardware IDs if specified
        if hardware_ids:
            validation_results = [
                vr for vr in validation_results 
                if vr.hardware_result.hardware_id in hardware_ids
            ]
        
        # Filter by model IDs if specified
        if model_ids:
            validation_results = [
                vr for vr in validation_results 
                if vr.simulation_result.model_id in model_ids
            ]
        
        # Filter by date if specified
        if start_date:
            validation_results = [
                vr for vr in validation_results 
                if vr.validation_timestamp >= start_date
            ]
        
        if end_date:
            validation_results = [
                vr for vr in validation_results 
                if vr.validation_timestamp <= end_date
            ]
        
        return validation_results