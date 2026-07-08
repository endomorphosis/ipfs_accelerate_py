"""
Monitoring Dashboard Integration for Dynamic Resource Management Visualization

This module provides integration between the Monitoring Dashboard and the 
Dynamic Resource Management Visualization module, allowing display of resource
allocation, utilization, and scaling visualizations in the monitoring dashboard.
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("drm_visualization_integration")

class DRMVisualizationIntegration:
    """
    Integration between Monitoring Dashboard and Dynamic Resource Management Visualization.
    
    This class provides methods to integrate the DRM visualization capabilities
    with the monitoring dashboard, allowing real-time display of resource metrics.
    """
    
    def __init__(
        self, 
        output_dir: str = "./drm_visualizations",
        update_interval: int = 60,
        resource_manager = None
    ):
        """
        Initialize the integration with the monitoring dashboard.
        
        Args:
            output_dir: Directory to store DRM visualizations
            update_interval: Interval in seconds between visualization updates
            resource_manager: Optional DynamicResourceManager instance
        """
        self.output_dir = output_dir
        self.update_interval = update_interval
        self.resource_manager = resource_manager
        self.drm_visualization = None
        self.last_update_time = 0
        self.visualization_registry = {}
        self.registry_file = os.path.join(output_dir, "visualization_registry.json")
        self.visualization_available = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load registry if it exists
        self._load_registry()
        
        # Try to import DRMVisualization
        try:
            from dynamic_resource_management_visualization import DRMVisualization
            self.visualization_available = True
            
            # Initialize visualization if resource manager is provided
            if resource_manager:
                self.drm_visualization = DRMVisualization(
                    dynamic_resource_manager=resource_manager,
                    output_dir=output_dir,
                    interactive=True,
                    update_interval=update_interval
                )
                logger.info("DRMVisualization initialized successfully")
        except ImportError:
            logger.warning("DRMVisualization not available. Visualization integration will be limited.")
    
    def _load_registry(self):
        """Load the visualization registry from disk."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    self.visualization_registry = json.load(f)
                logger.info(f"Loaded {len(self.visualization_registry)} visualizations from registry")
            except Exception as e:
                logger.error(f"Error loading visualization registry: {e}")
                self.visualization_registry = {}
    
    def _save_registry(self):
        """Save the visualization registry to disk."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.visualization_registry, f, indent=4)
            logger.info(f"Saved {len(self.visualization_registry)} visualizations to registry")
        except Exception as e:
            logger.error(f"Error saving visualization registry: {e}")
    
    def set_resource_manager(self, resource_manager):
        """
        Set or update the Dynamic Resource Manager instance.
        
        Args:
            resource_manager: DynamicResourceManager instance
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.visualization_available:
            logger.error("DRMVisualization not available. Cannot set resource manager.")
            return False
        
        try:
            # Update resource manager
            self.resource_manager = resource_manager
            
            # Initialize or update visualization
            if self.drm_visualization is None:
                from dynamic_resource_management_visualization import DRMVisualization
                self.drm_visualization = DRMVisualization(
                    dynamic_resource_manager=resource_manager,
                    output_dir=self.output_dir,
                    interactive=True,
                    update_interval=self.update_interval
                )
            else:
                self.drm_visualization.drm = resource_manager
            
            logger.info("Resource manager set successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting resource manager: {e}")
            return False
    
    def update_visualizations(self, force=False):
        """
        Update all DRM visualizations.
        
        Args:
            force: Force update even if update interval hasn't elapsed
        
        Returns:
            Dict[str, str]: Dictionary of visualization paths, keyed by type
        """
        if not self.visualization_available or self.drm_visualization is None:
            logger.error("DRMVisualization not available or not initialized.")
            return {}
        
        # Check if update interval has elapsed
        current_time = time.time()
        if not force and current_time - self.last_update_time < self.update_interval:
            logger.debug("Update interval has not elapsed. Skipping update.")
            return {key: value.get("path") for key, value in self.visualization_registry.items()}
        
        try:
            # Generate all visualizations
            result = {}
            
            # Resource utilization heatmap
            heatmap_path = self.drm_visualization.create_resource_utilization_heatmap()
            if heatmap_path:
                self.visualization_registry["resource_heatmap"] = {
                    "type": "resource_heatmap",
                    "title": "Resource Utilization Heatmap",
                    "description": "Heatmap showing resource utilization across workers",
                    "path": heatmap_path,
                    "updated_at": datetime.now().isoformat()
                }
                result["resource_heatmap"] = heatmap_path
            
            # Scaling history visualization
            scaling_path = self.drm_visualization.create_scaling_history_visualization()
            if scaling_path:
                self.visualization_registry["scaling_history"] = {
                    "type": "scaling_history",
                    "title": "Scaling History",
                    "description": "Visualization of scaling decisions over time",
                    "path": scaling_path,
                    "updated_at": datetime.now().isoformat()
                }
                result["scaling_history"] = scaling_path
            
            # Resource allocation visualization
            allocation_path = self.drm_visualization.create_resource_allocation_visualization()
            if allocation_path:
                self.visualization_registry["resource_allocation"] = {
                    "type": "resource_allocation",
                    "title": "Resource Allocation",
                    "description": "Visualization of resource allocation across workers",
                    "path": allocation_path,
                    "updated_at": datetime.now().isoformat()
                }
                result["resource_allocation"] = allocation_path
            
            # Resource efficiency visualization
            efficiency_path = self.drm_visualization.create_resource_efficiency_visualization()
            if efficiency_path:
                self.visualization_registry["resource_efficiency"] = {
                    "type": "resource_efficiency",
                    "title": "Resource Efficiency",
                    "description": "Visualization of resource allocation efficiency",
                    "path": efficiency_path,
                    "updated_at": datetime.now().isoformat()
                }
                result["resource_efficiency"] = efficiency_path
            
            # Cloud resource visualization if available
            if hasattr(self.drm_visualization, "create_cloud_resource_visualization") and \
               bool(getattr(self.drm_visualization, "cloud_usage_history", {})):
                cloud_path = self.drm_visualization.create_cloud_resource_visualization()
                if cloud_path:
                    self.visualization_registry["cloud_resources"] = {
                        "type": "cloud_resources",
                        "title": "Cloud Resource Usage",
                        "description": "Visualization of cloud provider resource usage",
                        "path": cloud_path,
                        "updated_at": datetime.now().isoformat()
                    }
                    result["cloud_resources"] = cloud_path
            
            # Create comprehensive dashboard
            try:
                dashboard_path = self.drm_visualization.create_resource_dashboard()
                if dashboard_path:
                    self.visualization_registry["dashboard"] = {
                        "type": "dashboard",
                        "title": "Resource Management Dashboard",
                        "description": "Comprehensive dashboard with all visualizations",
                        "path": dashboard_path,
                        "updated_at": datetime.now().isoformat()
                    }
                    result["dashboard"] = dashboard_path
            except Exception as e:
                logger.error(f"Error creating dashboard: {e}")
            
            # Update last update time
            self.last_update_time = current_time
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Updated {len(result)} visualizations")
            return result
        
        except Exception as e:
            logger.error(f"Error updating visualizations: {e}")
            return {}
    
    def get_visualization(self, viz_type):
        """
        Get a specific visualization by type.
        
        Args:
            viz_type: Visualization type (resource_heatmap, scaling_history, etc.)
        
        Returns:
            Dict: Visualization details or None if not found
        """
        return self.visualization_registry.get(viz_type)
    
    def get_all_visualizations(self):
        """
        Get all available visualizations.
        
        Returns:
            Dict: All visualization details
        """
        return self.visualization_registry
    
    def get_iframe_html(self, viz_type, width="100%", height="600px"):
        """
        Get HTML iframe code for embedding a visualization.
        
        Args:
            viz_type: Visualization type (resource_heatmap, scaling_history, etc.)
            width: iframe width
            height: iframe height
        
        Returns:
            str: HTML iframe code or empty string if visualization not found
        """
        viz_details = self.visualization_registry.get(viz_type)
        if not viz_details:
            return ""
        
        path = viz_details.get("path")
        if not path or not os.path.exists(path):
            return ""
        
        # Convert path to relative URL
        rel_path = os.path.relpath(path, self.output_dir)
        
        # For HTML dashboards
        if viz_type == "dashboard" and path.endswith('.html'):
            iframe_html = f"""
            <div class="embedded-dashboard" style="margin-top: 20px; margin-bottom: 20px;">
                <h3>{viz_details.get('title', 'Resource Dashboard')}</h3>
                <p>{viz_details.get('description', '')}</p>
                <iframe src="/static/drm_visualizations/{rel_path}" 
                        width="{width}" height="{height}" frameborder="0" style="border: 1px solid #ddd;">
                </iframe>
                <p class="text-muted small">Last updated: {viz_details.get('updated_at', '').replace('T', ' ').split('.')[0]}</p>
            </div>
            """
            return iframe_html
        
        # For image visualizations
        if path.endswith(('.png', '.jpg', '.jpeg', '.svg')):
            img_html = f"""
            <div class="visualization-container" style="margin-top: 20px; margin-bottom: 20px;">
                <h3>{viz_details.get('title', viz_type.replace('_', ' ').title())}</h3>
                <p>{viz_details.get('description', '')}</p>
                <div style="text-align: center;">
                    <img src="/static/drm_visualizations/{rel_path}" 
                         alt="{viz_details.get('title', viz_type)}" 
                         style="max-width: {width}; height: auto; border: 1px solid #ddd;" />
                </div>
                <p class="text-muted small">Last updated: {viz_details.get('updated_at', '').replace('T', ' ').split('.')[0]}</p>
            </div>
            """
            return img_html
            
        return ""
    
    def start_dashboard_server(self, port=None):
        """
        Start the DRM dashboard server.
        
        Args:
            port: Port to use for the dashboard server
        
        Returns:
            str: Dashboard URL if successful, None otherwise
        """
        if not self.visualization_available or self.drm_visualization is None:
            logger.error("DRMVisualization not available or not initialized.")
            return None
        
        try:
            # Start the dashboard server
            url = self.drm_visualization.start_dashboard_server(port=port)
            
            if url:
                logger.info(f"DRM dashboard server started at {url}")
            
            return url
        except Exception as e:
            logger.error(f"Error starting dashboard server: {e}")
            return None
    
    def stop_dashboard_server(self):
        """
        Stop the DRM dashboard server.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.visualization_available or self.drm_visualization is None:
            logger.error("DRMVisualization not available or not initialized.")
            return False
        
        try:
            # Stop the dashboard server
            self.drm_visualization.stop_dashboard_server()
            logger.info("DRM dashboard server stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping dashboard server: {e}")
            return False
    
    def cleanup(self):
        """
        Clean up resources used by the integration.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Stop dashboard server
            if self.visualization_available and self.drm_visualization:
                self.stop_dashboard_server()
                
                # Clean up visualization resources
                self.drm_visualization.cleanup()
            
            logger.info("DRM visualization integration cleaned up")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up DRM visualization integration: {e}")
            return False