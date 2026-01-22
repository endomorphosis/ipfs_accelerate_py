#!/usr/bin/env python3
"""
Monitoring Dashboard Integration for Advanced Visualization System.

This module provides integration between the Advanced Visualization System
and the Monitoring Dashboard, allowing visualizations to be embedded and
synchronized with the monitoring dashboard.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("monitor_dashboard_integration")

# Try to import WebSocket client for real-time updates
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    logger.warning("Websocket client not available. Install with: pip install websocket-client")
    HAS_WEBSOCKET = False


class MonitorDashboardIntegration:
    """Integration with the Monitoring Dashboard system."""

    def __init__(self, dashboard_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Monitoring Dashboard integration.

        Args:
            dashboard_url: URL of the monitoring dashboard (default: http://localhost:8082)
            api_key: API key for authentication (optional)
        """
        self.dashboard_url = dashboard_url or "http://localhost:8082"
        self.api_key = api_key
        self.ws_connection = None
        self.is_connected = False
        
        # Component registry to track which visualizations have been registered with the dashboard
        self.registered_components = set()
        
        logger.info(f"Initialized Monitoring Dashboard integration with URL: {self.dashboard_url}")
    
    def connect(self) -> bool:
        """
        Connect to the monitoring dashboard API.
        
        Returns:
            Success status of the connection
        """
        try:
            import requests
            
            # Test connection to dashboard
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            response = requests.get(f"{self.dashboard_url}/api/health", headers=headers, timeout=5)
            
            if response.status_code == 200:
                logger.info("Successfully connected to Monitoring Dashboard API")
                return True
            else:
                logger.error(f"Failed to connect to Monitoring Dashboard API: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Monitoring Dashboard API: {e}")
            return False
    
    def connect_websocket(self) -> bool:
        """
        Connect to the Monitoring Dashboard WebSocket API for real-time updates.
        
        Returns:
            Success status of the WebSocket connection
        """
        if not HAS_WEBSOCKET:
            logger.error("WebSocket integration requires websocket-client package")
            return False
        
        try:
            # Derive WebSocket URL from HTTP URL (replace http with ws)
            if self.dashboard_url.startswith("http://"):
                ws_url = self.dashboard_url.replace("http://", "ws://")
            elif self.dashboard_url.startswith("https://"):
                ws_url = self.dashboard_url.replace("https://", "wss://")
            else:
                ws_url = f"ws://{self.dashboard_url}"
            
            # Connect to WebSocket API endpoint
            self.ws_connection = websocket.create_connection(
                f"{ws_url}/api/ws/visualizations",
                header={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            )
            
            self.is_connected = True
            logger.info("Successfully connected to Monitoring Dashboard WebSocket API")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Monitoring Dashboard WebSocket API: {e}")
            self.is_connected = False
            return False
    
    def register_visualization(
        self,
        visualization_id: str,
        visualization_type: str,
        metadata: Dict[str, Any],
        visualization_data: Optional[Dict[str, Any]] = None,
        html_content: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> bool:
        """
        Register a visualization with the monitoring dashboard.
        
        Args:
            visualization_id: Unique identifier for the visualization
            visualization_type: Type of visualization (3d, heatmap, power, time-series, dashboard)
            metadata: Metadata about the visualization (title, description, creation_time)
            visualization_data: Visualization data for API-based rendering (optional)
            html_content: HTML content of the visualization for iframe embedding (optional)
            image_path: Path to static image of the visualization (optional)
            
        Returns:
            Success status of the registration
        """
        try:
            import requests
            
            # Prepare the registration payload
            payload = {
                "visualization_id": visualization_id,
                "visualization_type": visualization_type,
                "metadata": metadata,
            }
            
            # Include visualization data if provided
            if visualization_data:
                payload["visualization_data"] = visualization_data
            
            # Prepare files for upload if provided
            files = {}
            if html_content:
                files["html"] = ("visualization.html", html_content, "text/html")
            
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    files["image"] = (os.path.basename(image_path), f.read(), "image/png")
            
            # Prepare headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Send registration request
            if files:
                response = requests.post(
                    f"{self.dashboard_url}/api/visualizations/register",
                    data={"payload": json.dumps(payload)},
                    files=files,
                    headers=headers
                )
            else:
                response = requests.post(
                    f"{self.dashboard_url}/api/visualizations/register",
                    json=payload,
                    headers=headers
                )
            
            if response.status_code in (200, 201):
                logger.info(f"Successfully registered visualization {visualization_id} with Monitoring Dashboard")
                self.registered_components.add(visualization_id)
                return True
            else:
                logger.error(f"Failed to register visualization: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering visualization with Monitoring Dashboard: {e}")
            return False
    
    def update_visualization(
        self,
        visualization_id: str,
        visualization_data: Optional[Dict[str, Any]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        html_content: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> bool:
        """
        Update an existing visualization in the monitoring dashboard.
        
        Args:
            visualization_id: Unique identifier for the visualization
            visualization_data: Updated visualization data (optional)
            metadata_updates: Metadata fields to update (optional)
            html_content: Updated HTML content (optional)
            image_path: Path to updated static image (optional)
            
        Returns:
            Success status of the update
        """
        try:
            import requests
            
            # Prepare the update payload
            payload = {
                "visualization_id": visualization_id,
            }
            
            # Include visualization data if provided
            if visualization_data:
                payload["visualization_data"] = visualization_data
            
            # Include metadata updates if provided
            if metadata_updates:
                payload["metadata_updates"] = metadata_updates
            
            # Prepare files for upload if provided
            files = {}
            if html_content:
                files["html"] = ("visualization.html", html_content, "text/html")
            
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    files["image"] = (os.path.basename(image_path), f.read(), "image/png")
            
            # Prepare headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Send update request
            if files:
                response = requests.put(
                    f"{self.dashboard_url}/api/visualizations/{visualization_id}",
                    data={"payload": json.dumps(payload)},
                    files=files,
                    headers=headers
                )
            else:
                response = requests.put(
                    f"{self.dashboard_url}/api/visualizations/{visualization_id}",
                    json=payload,
                    headers=headers
                )
            
            if response.status_code == 200:
                logger.info(f"Successfully updated visualization {visualization_id} in Monitoring Dashboard")
                return True
            else:
                logger.error(f"Failed to update visualization: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating visualization in Monitoring Dashboard: {e}")
            return False
    
    def send_real_time_update(
        self,
        visualization_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """
        Send a real-time update to a visualization via WebSocket.
        
        Args:
            visualization_id: Unique identifier for the visualization
            update_data: Data to update the visualization
            
        Returns:
            Success status of the update
        """
        if not self.is_connected:
            if not self.connect_websocket():
                return False
        
        try:
            # Prepare update message
            message = {
                "type": "visualization_update",
                "visualization_id": visualization_id,
                "data": update_data,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Send update via WebSocket
            self.ws_connection.send(json.dumps(message))
            logger.info(f"Sent real-time update for visualization {visualization_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending real-time update: {e}")
            self.is_connected = False
            return False
    
    def create_dashboard_panel(
        self,
        panel_title: str,
        visualization_ids: List[str],
        layout: Dict[str, Any],
        dashboard_id: str = "main",
        panel_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a panel in the monitoring dashboard containing multiple visualizations.
        
        Args:
            panel_title: Title for the panel
            visualization_ids: List of visualization IDs to include in the panel
            layout: Layout configuration for the panel
            dashboard_id: ID of the dashboard to add the panel to (default: main)
            panel_id: Unique identifier for the panel (default: generated from title)
            
        Returns:
            Panel ID if successful, None otherwise
        """
        try:
            import requests
            
            # Generate panel_id if not provided
            if panel_id is None:
                panel_id = panel_title.lower().replace(" ", "_") + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Prepare panel configuration
            panel_config = {
                "panel_id": panel_id,
                "title": panel_title,
                "visualization_ids": visualization_ids,
                "layout": layout,
                "dashboard_id": dashboard_id,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            # Prepare headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Send request to create panel
            response = requests.post(
                f"{self.dashboard_url}/api/dashboards/{dashboard_id}/panels",
                json=panel_config,
                headers=headers
            )
            
            if response.status_code in (200, 201):
                logger.info(f"Successfully created panel {panel_id} in dashboard {dashboard_id}")
                return panel_id
            else:
                logger.error(f"Failed to create dashboard panel: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating dashboard panel: {e}")
            return None
    
    def register_visualization_batch(
        self,
        visualizations: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Register multiple visualizations with the monitoring dashboard in a batch.
        
        Args:
            visualizations: Dictionary mapping visualization IDs to metadata and paths
            
        Returns:
            List of successfully registered visualization IDs
        """
        try:
            import requests
            
            # Prepare batch request
            visualization_batch = []
            for viz_id, viz_info in visualizations.items():
                viz_entry = {
                    "visualization_id": viz_id,
                    "visualization_type": viz_info.get("type"),
                    "metadata": viz_info.get("metadata", {})
                }
                
                # Include data if available
                if "data" in viz_info:
                    viz_entry["visualization_data"] = viz_info["data"]
                
                visualization_batch.append(viz_entry)
            
            # Prepare headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Send batch registration request
            response = requests.post(
                f"{self.dashboard_url}/api/visualizations/register_batch",
                json={"visualizations": visualization_batch},
                headers=headers
            )
            
            if response.status_code in (200, 201):
                result = response.json()
                successful_ids = result.get("successful_ids", [])
                failed_ids = result.get("failed_ids", [])
                
                logger.info(f"Successfully registered {len(successful_ids)} visualizations with Monitoring Dashboard")
                if failed_ids:
                    logger.warning(f"Failed to register {len(failed_ids)} visualizations: {', '.join(failed_ids)}")
                
                # Register successful visualizations in local registry
                self.registered_components.update(successful_ids)
                
                return successful_ids
            else:
                logger.error(f"Failed to register visualization batch: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error registering visualization batch with Monitoring Dashboard: {e}")
            return []
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """
        Get the current status of the monitoring dashboard.
        
        Returns:
            Dashboard status information
        """
        try:
            import requests
            
            # Prepare headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Send status request
            response = requests.get(
                f"{self.dashboard_url}/api/status",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get dashboard status: {response.status_code} - {response.text}")
                return {"status": "error", "message": f"Failed to get dashboard status: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error getting dashboard status: {e}")
            return {"status": "error", "message": str(e)}
    
    def export_dashboard_snapshot(
        self,
        dashboard_id: str = "main",
        include_visualizations: bool = True,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export a snapshot of the entire dashboard with all visualizations.
        
        Args:
            dashboard_id: ID of the dashboard to export (default: main)
            include_visualizations: Whether to include visualization data
            output_path: Path to save the exported snapshot (default: generated)
            
        Returns:
            Path to the exported snapshot file if successful, None otherwise
        """
        try:
            import requests
            
            # Prepare headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Send export request
            response = requests.get(
                f"{self.dashboard_url}/api/dashboards/{dashboard_id}/export",
                params={"include_visualizations": str(include_visualizations).lower()},
                headers=headers
            )
            
            if response.status_code == 200:
                # Generate output path if not provided
                if output_path is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"dashboard_snapshot_{dashboard_id}_{timestamp}.json"
                
                # Create directory if needed
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                # Save snapshot to file
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(response.json(), f, indent=2)
                
                logger.info(f"Successfully exported dashboard snapshot to {output_path}")
                return output_path
            else:
                logger.error(f"Failed to export dashboard snapshot: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting dashboard snapshot: {e}")
            return None
    
    def import_dashboard_snapshot(
        self,
        snapshot_path: str,
        target_dashboard_id: Optional[str] = None,
        merge_strategy: str = "replace"
    ) -> bool:
        """
        Import a dashboard snapshot into the monitoring dashboard.
        
        Args:
            snapshot_path: Path to the snapshot file
            target_dashboard_id: ID of the dashboard to import into (default: from snapshot)
            merge_strategy: Strategy for handling conflicts ('replace', 'merge', 'keep_existing')
            
        Returns:
            Success status of the import
        """
        try:
            import requests
            
            # Check if snapshot file exists
            if not os.path.exists(snapshot_path):
                logger.error(f"Snapshot file not found: {snapshot_path}")
                return False
            
            # Read snapshot file
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snapshot_data = json.load(f)
            
            # Prepare headers
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Prepare import params
            params = {
                "merge_strategy": merge_strategy
            }
            if target_dashboard_id:
                params["dashboard_id"] = target_dashboard_id
            
            # Send import request
            response = requests.post(
                f"{self.dashboard_url}/api/dashboards/import",
                json=snapshot_data,
                params=params,
                headers=headers
            )
            
            if response.status_code in (200, 201):
                logger.info(f"Successfully imported dashboard snapshot from {snapshot_path}")
                return True
            else:
                logger.error(f"Failed to import dashboard snapshot: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error importing dashboard snapshot: {e}")
            return False
    
    def synchronize_visualizations(
        self,
        visualization_dir: str,
        file_pattern: str = "*.html",
        recursive: bool = True,
        visualization_type_mapping: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Scan a directory for visualization files and synchronize them with the dashboard.
        
        Args:
            visualization_dir: Directory containing visualization files
            file_pattern: Glob pattern for finding visualization files
            recursive: Whether to scan recursively
            visualization_type_mapping: Mapping from filename patterns to visualization types
            
        Returns:
            Number of successfully synchronized visualizations
        """
        try:
            import glob
            
            # Prepare visualization type mapping if not provided
            if visualization_type_mapping is None:
                visualization_type_mapping = {
                    "*3d*": "3d",
                    "*heatmap*": "heatmap",
                    "*power*": "power",
                    "*time*series*": "time-series",
                    "*dashboard*": "dashboard"
                }
            
            # Find visualization files
            if recursive:
                visualization_files = glob.glob(os.path.join(visualization_dir, "**", file_pattern), recursive=True)
            else:
                visualization_files = glob.glob(os.path.join(visualization_dir, file_pattern))
            
            if not visualization_files:
                logger.warning(f"No visualization files found in {visualization_dir} with pattern {file_pattern}")
                return 0
            
            # Prepare visualizations for batch registration
            visualizations = {}
            for viz_file in visualization_files:
                # Generate visualization ID from filename
                viz_id = os.path.splitext(os.path.basename(viz_file))[0]
                
                # Determine visualization type based on filename
                viz_type = None
                for pattern, v_type in visualization_type_mapping.items():
                    import fnmatch
                    if fnmatch.fnmatch(viz_file.lower(), pattern.lower()):
                        viz_type = v_type
                        break
                
                if viz_type is None:
                    viz_type = "generic"
                
                # Extract creation time from file metadata
                creation_time = datetime.datetime.fromtimestamp(os.path.getctime(viz_file)).isoformat()
                
                # Read HTML content
                with open(viz_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                # Add to visualization collection
                visualizations[viz_id] = {
                    "type": viz_type,
                    "metadata": {
                        "title": viz_id.replace("_", " ").title(),
                        "creation_time": creation_time,
                        "file_path": viz_file
                    },
                    "html_content": html_content
                }
            
            # Register visualizations in batch
            registered_ids = self.register_visualization_batch(visualizations)
            
            # Count successful registrations
            num_registered = len(registered_ids)
            
            logger.info(f"Successfully synchronized {num_registered} visualizations with the dashboard")
            return num_registered
                
        except Exception as e:
            logger.error(f"Error synchronizing visualizations: {e}")
            return 0

    def setup_auto_sync(
        self,
        visualization_dir: str,
        interval_seconds: int = 60,
        file_pattern: str = "*.html",
        recursive: bool = True,
        max_runtime_minutes: Optional[int] = None
    ) -> None:
        """
        Set up automatic synchronization of visualizations with the dashboard.
        
        Args:
            visualization_dir: Directory to monitor for visualization files
            interval_seconds: Interval between synchronization runs
            file_pattern: Glob pattern for finding visualization files
            recursive: Whether to scan recursively
            max_runtime_minutes: Maximum runtime in minutes (None for indefinite)
        """
        import threading
        import time
        
        logger.info(f"Setting up automatic synchronization for {visualization_dir} every {interval_seconds} seconds")
        
        def sync_thread():
            start_time = time.time()
            run_count = 0
            
            while True:
                # Check if max runtime reached
                if max_runtime_minutes is not None:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= max_runtime_minutes:
                        logger.info(f"Auto-sync stopping after {elapsed_minutes:.1f} minutes")
                        break
                
                # Run synchronization
                run_count += 1
                logger.info(f"Auto-sync run #{run_count}")
                try:
                    self.synchronize_visualizations(
                        visualization_dir=visualization_dir,
                        file_pattern=file_pattern,
                        recursive=recursive
                    )
                except Exception as e:
                    logger.error(f"Error during auto-sync: {e}")
                
                # Sleep until next run
                time.sleep(interval_seconds)
        
        # Start sync thread
        sync_thread = threading.Thread(target=sync_thread, daemon=True)
        sync_thread.start()
        
        logger.info("Auto-sync started in background thread")


class MonitorDashboardIntegrationMixin:
    """Mixin class for adding monitoring dashboard integration to the visualization system."""
    
    def __init__(self, dashboard_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the monitoring dashboard integration mixin.
        
        Args:
            dashboard_url: URL of the monitoring dashboard
            api_key: API key for authentication
        """
        # Initialize dashboard integration
        self._dashboard_integration = MonitorDashboardIntegration(
            dashboard_url=dashboard_url,
            api_key=api_key
        )
        
        # Track which visualizations have been registered
        self._dashboard_registered_visualizations = set()
    
    def connect_to_dashboard(self) -> bool:
        """
        Connect to the monitoring dashboard.
        
        Returns:
            Success status of the connection
        """
        return self._dashboard_integration.connect()
    
    def register_visualization_with_dashboard(
        self,
        visualization_id: str,
        visualization_type: str,
        title: str,
        html_path: Optional[str] = None,
        image_path: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a visualization with the monitoring dashboard.
        
        Args:
            visualization_id: Unique identifier for the visualization
            visualization_type: Type of visualization
            title: Title of the visualization
            html_path: Path to HTML visualization file
            image_path: Path to image file
            data: Visualization data for API-based rendering
            metadata: Additional metadata about the visualization
            
        Returns:
            Success status of the registration
        """
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "title": title,
            "creation_time": datetime.datetime.now().isoformat()
        })
        
        # Read HTML content if path provided
        html_content = None
        if html_path and os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        
        # Register with dashboard
        result = self._dashboard_integration.register_visualization(
            visualization_id=visualization_id,
            visualization_type=visualization_type,
            metadata=metadata,
            visualization_data=data,
            html_content=html_content,
            image_path=image_path
        )
        
        if result:
            self._dashboard_registered_visualizations.add(visualization_id)
        
        return result
    
    def create_dashboard_panel_from_visualizations(
        self,
        panel_title: str,
        visualization_ids: List[str],
        layout_columns: int = 2,
        dashboard_id: str = "main"
    ) -> Optional[str]:
        """
        Create a panel in the monitoring dashboard from multiple visualizations.
        
        Args:
            panel_title: Title for the panel
            visualization_ids: List of visualization IDs to include
            layout_columns: Number of columns in the layout
            dashboard_id: ID of the dashboard to add the panel to
            
        Returns:
            Panel ID if successful, None otherwise
        """
        # Check if visualizations are registered
        unregistered = [viz_id for viz_id in visualization_ids 
                        if viz_id not in self._dashboard_registered_visualizations]
        
        if unregistered:
            logger.warning(f"Some visualizations are not registered with the dashboard: {', '.join(unregistered)}")
        
        # Create simple grid layout
        num_items = len(visualization_ids)
        layout = {
            "type": "grid",
            "columns": layout_columns,
            "rows": (num_items + layout_columns - 1) // layout_columns,
            "items": []
        }
        
        # Add items to layout
        for i, viz_id in enumerate(visualization_ids):
            row = i // layout_columns
            col = i % layout_columns
            
            layout["items"].append({
                "visualization_id": viz_id,
                "row": row,
                "col": col,
                "width": 1,
                "height": 1
            })
        
        # Create panel
        return self._dashboard_integration.create_dashboard_panel(
            panel_title=panel_title,
            visualization_ids=visualization_ids,
            layout=layout,
            dashboard_id=dashboard_id
        )
    
    def sync_visualizations_with_dashboard(
        self,
        output_dir: Optional[str] = None
    ) -> int:
        """
        Synchronize all visualizations in the output directory with the dashboard.
        
        Args:
            output_dir: Directory containing visualization files (default: self.output_dir)
            
        Returns:
            Number of successfully synchronized visualizations
        """
        # Get output directory
        vis_dir = output_dir or getattr(self, 'output_dir', None)
        if vis_dir is None:
            logger.error("No output directory specified")
            return 0
        
        # Synchronize visualizations
        return self._dashboard_integration.synchronize_visualizations(
            visualization_dir=vis_dir,
            file_pattern="*.html",
            recursive=True
        )
    
    def setup_auto_dashboard_sync(
        self,
        interval_seconds: int = 60,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Set up automatic synchronization with the monitoring dashboard.
        
        Args:
            interval_seconds: Interval between synchronization runs
            output_dir: Directory to monitor for visualization files
        """
        # Get output directory
        vis_dir = output_dir or getattr(self, 'output_dir', None)
        if vis_dir is None:
            logger.error("No output directory specified")
            return
        
        # Setup auto-sync
        self._dashboard_integration.setup_auto_sync(
            visualization_dir=vis_dir,
            interval_seconds=interval_seconds,
            file_pattern="*.html",
            recursive=True
        )
    
    def export_dashboard_snapshot(
        self,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export a snapshot of the monitoring dashboard.
        
        Args:
            output_path: Path to save the snapshot (default: generated in output_dir)
            
        Returns:
            Path to the snapshot file if successful, None otherwise
        """
        # Generate default output path if not provided
        if output_path is None:
            # Use self.output_dir if available
            output_dir = getattr(self, 'output_dir', './exports')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"dashboard_snapshot_{timestamp}.json")
        
        # Export snapshot
        return self._dashboard_integration.export_dashboard_snapshot(
            output_path=output_path,
            include_visualizations=True
        )