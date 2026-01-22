"""
Monitoring Dashboard Visualization Integration

This module provides integration between the Monitoring Dashboard and the Advanced Visualization
System's Customizable Dashboard functionality.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dashboard_visualization_integration")

# Try to import visualization components
try:
    from duckdb_api.visualization.advanced_visualization import CustomizableDashboard
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("CustomizableDashboard not available. Advanced visualizations will be disabled.")
    VISUALIZATION_AVAILABLE = False

class VisualizationDashboardIntegration:
    """Integration between Monitoring Dashboard and Advanced Visualization System."""
    
    def __init__(self, dashboard_dir: str = "./dashboards", integration_dir: str = None):
        """Initialize the integration with the monitoring dashboard.
        
        Args:
            dashboard_dir: Directory to store visualization dashboards
            integration_dir: Directory to store integration-specific files
        """
        self.dashboard_dir = dashboard_dir
        self.integration_dir = integration_dir or os.path.join(dashboard_dir, "monitor_integration")
        self.visualization_available = VISUALIZATION_AVAILABLE
        self.dashboard_instance = None
        
        # Create directories
        os.makedirs(self.dashboard_dir, exist_ok=True)
        os.makedirs(self.integration_dir, exist_ok=True)
        
        # Embedded dashboard registry
        self.embedded_dashboards = {}
        self.registry_file = os.path.join(self.integration_dir, "embedded_dashboards.json")
        
        # Load existing registry if available
        self._load_registry()
        
        if self.visualization_available:
            try:
                self.dashboard_instance = CustomizableDashboard(
                    db_connection=None,
                    output_dir=self.dashboard_dir
                )
                logger.info("CustomizableDashboard initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing CustomizableDashboard: {e}")
                self.visualization_available = False
    
    def _load_registry(self):
        """Load the embedded dashboard registry."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    self.embedded_dashboards = json.load(f)
                logger.info(f"Loaded {len(self.embedded_dashboards)} embedded dashboards from registry")
            except Exception as e:
                logger.error(f"Error loading embedded dashboard registry: {e}")
                self.embedded_dashboards = {}
    
    def _save_registry(self):
        """Save the embedded dashboard registry."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.embedded_dashboards, f, indent=4)
            logger.info(f"Saved {len(self.embedded_dashboards)} embedded dashboards to registry")
        except Exception as e:
            logger.error(f"Error saving embedded dashboard registry: {e}")
    
    def create_embedded_dashboard(self, 
                                 name: str, 
                                 page: str, 
                                 template: str = "overview", 
                                 title: str = None, 
                                 description: str = None, 
                                 position: str = "below", 
                                 components: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new dashboard to embed in the monitoring dashboard.
        
        Args:
            name: Unique name for the dashboard
            page: Monitoring dashboard page where this will be embedded
            template: Dashboard template to use
            title: Dashboard title
            description: Dashboard description
            position: Embedding position (above, below, tab)
            components: Optional list of component configurations
        
        Returns:
            Dictionary with dashboard details
        """
        if not self.visualization_available:
            logger.error("CustomizableDashboard not available. Cannot create embedded dashboard.")
            return None
        
        try:
            # Create the dashboard
            dashboard_path = self.dashboard_instance.create_dashboard(
                dashboard_name=name,
                template=template,
                title=title,
                description=description,
                components=components
            )
            
            # Add to embedded dashboard registry
            dashboard_details = {
                "name": name,
                "page": page,
                "template": template,
                "title": title or self.dashboard_instance.title,
                "description": description or self.dashboard_instance.description,
                "path": dashboard_path,
                "position": position,
                "created_at": self.dashboard_instance.dashboard_config.get("created_at", ""),
                "updated_at": self.dashboard_instance.dashboard_config.get("updated_at", "")
            }
            
            self.embedded_dashboards[name] = dashboard_details
            self._save_registry()
            
            return dashboard_details
        
        except Exception as e:
            logger.error(f"Error creating embedded dashboard: {e}")
            return None
    
    def get_embedded_dashboards_for_page(self, page: str) -> Dict[str, Dict[str, Any]]:
        """Get all embedded dashboards for a specific monitoring dashboard page.
        
        Args:
            page: Monitoring dashboard page name
        
        Returns:
            Dictionary of dashboard details for the specified page
        """
        return {name: details for name, details in self.embedded_dashboards.items() 
                if details.get("page") == page}
    
    def get_embedded_dashboard(self, name: str) -> Dict[str, Any]:
        """Get details of a specific embedded dashboard.
        
        Args:
            name: Dashboard name
        
        Returns:
            Dictionary with dashboard details
        """
        return self.embedded_dashboards.get(name)
    
    def update_embedded_dashboard(self, name: str, title: str = None, description: str = None, 
                                  position: str = None, page: str = None) -> Dict[str, Any]:
        """Update an embedded dashboard's configuration.
        
        Args:
            name: Dashboard name
            title: New dashboard title
            description: New dashboard description
            position: New embedding position
            page: New monitoring dashboard page
        
        Returns:
            Updated dashboard details
        """
        if not self.visualization_available:
            logger.error("CustomizableDashboard not available. Cannot update embedded dashboard.")
            return None
        
        if name not in self.embedded_dashboards:
            logger.error(f"Embedded dashboard '{name}' not found")
            return None
        
        try:
            # Get current details
            current_details = self.embedded_dashboards[name]
            
            # Update monitoring dashboard specific details
            if position is not None:
                current_details["position"] = position
            if page is not None:
                current_details["page"] = page
            
            # Update visualization dashboard
            if title is not None or description is not None:
                dashboard_path = self.dashboard_instance.update_dashboard(
                    dashboard_name=name,
                    title=title,
                    description=description
                )
                
                if dashboard_path:
                    # Update details
                    if title is not None:
                        current_details["title"] = title
                    if description is not None:
                        current_details["description"] = description
                    
                    current_details["updated_at"] = self.dashboard_instance.dashboard_config.get("updated_at", "")
                    current_details["path"] = dashboard_path
            
            # Save registry
            self._save_registry()
            
            return current_details
        
        except Exception as e:
            logger.error(f"Error updating embedded dashboard: {e}")
            return None
    
    def remove_embedded_dashboard(self, name: str) -> bool:
        """Remove an embedded dashboard.
        
        Args:
            name: Dashboard name
        
        Returns:
            True if successful, False otherwise
        """
        if name not in self.embedded_dashboards:
            logger.error(f"Embedded dashboard '{name}' not found")
            return False
        
        try:
            # Remove dashboard from registry
            dashboard_details = self.embedded_dashboards.pop(name)
            self._save_registry()
            
            # Don't delete the actual dashboard file, just remove from embedded registry
            logger.info(f"Removed embedded dashboard '{name}' from registry")
            
            return True
        
        except Exception as e:
            logger.error(f"Error removing embedded dashboard: {e}")
            return False
    
    def get_dashboard_iframe_html(self, name: str, width: str = "100%", height: str = "600px") -> str:
        """Get HTML to embed dashboard using an iframe.
        
        Args:
            name: Dashboard name
            width: iframe width
            height: iframe height
        
        Returns:
            HTML string with iframe code
        """
        if name not in self.embedded_dashboards:
            logger.error(f"Embedded dashboard '{name}' not found")
            return ""
        
        dashboard_details = self.embedded_dashboards[name]
        dashboard_path = dashboard_details.get("path", "")
        
        if not dashboard_path or not os.path.exists(dashboard_path):
            logger.error(f"Dashboard file for '{name}' not found: {dashboard_path}")
            return ""
        
        # Create iframe HTML
        iframe_html = f"""
        <div class="embedded-dashboard">
            <iframe src="/static/dashboards/{os.path.basename(os.path.dirname(dashboard_path))}/dashboard.html" 
                    width="{width}" height="{height}" frameborder="0">
            </iframe>
        </div>
        """
        
        return iframe_html
    
    def generate_dashboard_from_performance_data(self, 
                                               performance_data: Dict[str, Any],
                                               name: str = None,
                                               title: str = "Performance Analysis Dashboard") -> str:
        """Generate a visualization dashboard from performance analytics data.
        
        Args:
            performance_data: Performance data from monitoring dashboard
            name: Dashboard name (generated if not provided)
            title: Dashboard title
        
        Returns:
            Path to the generated dashboard
        """
        if not self.visualization_available:
            logger.error("CustomizableDashboard not available. Cannot generate dashboard.")
            return None
        
        try:
            # Generate dashboard name if not provided
            if not name:
                from datetime import datetime
                import uuid
                name = f"performance_dashboard_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
            
            # Extract metrics and dimensions from performance data
            metrics = []
            dimensions = []
            
            if "metrics" in performance_data:
                metrics = list(performance_data["metrics"].keys())
            if "dimensions" in performance_data:
                dimensions = list(performance_data["dimensions"].keys())
            
            # Create components based on available data
            components = []
            
            # Add 3D visualization if we have at least 3 metrics
            if len(metrics) >= 3:
                components.append({
                    "type": "3d",
                    "config": {
                        "metrics": metrics[:3],
                        "dimensions": dimensions[:2] if dimensions else [],
                        "title": "3D Performance Metrics"
                    },
                    "width": 2,
                    "height": 1
                })
            
            # Add heatmap if we have metrics and dimensions
            if metrics and dimensions:
                components.append({
                    "type": "heatmap",
                    "config": {
                        "metric": metrics[0],
                        "title": f"{metrics[0].replace('_', ' ').title()} Comparison"
                    },
                    "width": 1,
                    "height": 1
                })
            
            # Add time-series visualization if we have time data
            if "time_series" in performance_data and metrics:
                components.append({
                    "type": "time-series",
                    "config": {
                        "metric": metrics[0],
                        "dimensions": dimensions[:2] if dimensions else [],
                        "time_range": 90,
                        "title": f"{metrics[0].replace('_', ' ').title()} Trends"
                    },
                    "width": 2,
                    "height": 1
                })
            
            # Add animated time-series if we have time data
            if "time_series" in performance_data and metrics:
                components.append({
                    "type": "animated-time-series",
                    "config": {
                        "metric": metrics[0],
                        "dimensions": dimensions[:2] if dimensions else [],
                        "time_range": 90,
                        "title": f"Animated {metrics[0].replace('_', ' ').title()} Trends"
                    },
                    "width": 2,
                    "height": 1
                })
            
            # Create the dashboard
            dashboard_path = self.dashboard_instance.create_dashboard(
                dashboard_name=name,
                title=title,
                description=f"Performance dashboard generated from monitoring data on {datetime.now().strftime('%Y-%m-%d')}",
                components=components
            )
            
            return dashboard_path
        
        except Exception as e:
            logger.error(f"Error generating dashboard from performance data: {e}")
            return None

    def list_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """List all available dashboard templates.
        
        Returns:
            Dictionary of template details
        """
        if not self.visualization_available:
            logger.error("CustomizableDashboard not available.")
            return {}
        
        try:
            return self.dashboard_instance.list_available_templates()
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return {}
    
    def list_available_components(self) -> Dict[str, str]:
        """List all available component types.
        
        Returns:
            Dictionary of component types and descriptions
        """
        if not self.visualization_available:
            logger.error("CustomizableDashboard not available.")
            return {}
        
        try:
            return self.dashboard_instance.list_available_components()
        except Exception as e:
            logger.error(f"Error listing components: {e}")
            return {}
    
    def export_embedded_dashboard(self, name: str, format: str = "html") -> str:
        """Export an embedded dashboard to a specific format.
        
        Args:
            name: Dashboard name
            format: Export format (html, png, pdf)
        
        Returns:
            Path to the exported file
        """
        if not self.visualization_available:
            logger.error("CustomizableDashboard not available.")
            return None
        
        if name not in self.embedded_dashboards:
            logger.error(f"Embedded dashboard '{name}' not found")
            return None
        
        try:
            output_path = os.path.join(self.integration_dir, f"{name}.{format}")
            return self.dashboard_instance.export_dashboard(name, format, output_path)
        except Exception as e:
            logger.error(f"Error exporting dashboard: {e}")
            return None