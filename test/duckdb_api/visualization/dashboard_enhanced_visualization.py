#!/usr/bin/env python3
"""
Dashboard-Enhanced Advanced Visualization System.

This module extends the AdvancedVisualizationSystem with monitoring dashboard integration.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dashboard_enhanced_visualization")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the base AdvancedVisualizationSystem
from duckdb_api.visualization.advanced_visualization import AdvancedVisualizationSystem

# Import dashboard integration
try:
    from duckdb_api.visualization.advanced_visualization.monitor_dashboard_integration import (
        MonitorDashboardIntegration,
        MonitorDashboardIntegrationMixin
    )
    HAS_DASHBOARD_INTEGRATION = True
except ImportError:
    logger.warning("MonitorDashboardIntegration not available. Install with: pip install requests websocket-client")
    HAS_DASHBOARD_INTEGRATION = False
    # Create dummy classes if not available
    class MonitorDashboardIntegration:
        def __init__(self, dashboard_url=None, api_key=None):
            pass
    
    class MonitorDashboardIntegrationMixin:
        def __init__(self, dashboard_url=None, api_key=None):
            pass


class DashboardEnhancedVisualizationSystem(AdvancedVisualizationSystem, MonitorDashboardIntegrationMixin):
    """
    Enhanced Advanced Visualization System with Monitoring Dashboard Integration.
    
    This class extends the AdvancedVisualizationSystem with the ability to synchronize
    visualizations with a monitoring dashboard.
    """
    
    def __init__(
        self, 
        db_api=None, 
        output_dir: str = "./advanced_visualizations", 
        dashboard_url: Optional[str] = None, 
        api_key: Optional[str] = None
    ):
        """
        Initialize the dashboard-enhanced visualization system.
        
        Args:
            db_api: Database API for accessing performance data
            output_dir: Directory to save visualizations
            dashboard_url: URL of the monitoring dashboard
            api_key: API key for dashboard authentication
        """
        # Initialize base classes
        AdvancedVisualizationSystem.__init__(self, db_api=db_api, output_dir=output_dir)
        
        if HAS_DASHBOARD_INTEGRATION:
            MonitorDashboardIntegrationMixin.__init__(self, dashboard_url=dashboard_url, api_key=api_key)
            logger.info("Dashboard integration initialized")
        else:
            logger.warning("Dashboard integration not available. Some features will be disabled.")
    
    def synchronize_with_dashboard(
        self,
        dashboard_url: Optional[str] = None,
        api_key: Optional[str] = None,
        create_panel: bool = True,
        panel_title: str = "Performance Visualization Panel"
    ) -> int:
        """
        Synchronize visualizations with the monitoring dashboard.
        
        This method synchronizes all visualizations in the output directory with
        the monitoring dashboard, optionally creating a panel to display them.
        
        Args:
            dashboard_url: URL of the monitoring dashboard (overrides the one from initialization)
            api_key: API key for authentication (overrides the one from initialization)
            create_panel: Whether to create a dashboard panel with the visualizations
            panel_title: Title for the panel if created
            
        Returns:
            Number of synchronized visualizations
        """
        if not HAS_DASHBOARD_INTEGRATION:
            logger.error("Dashboard integration not available. Install with: pip install requests websocket-client")
            return 0
        
        # Connect to dashboard and check if successful
        if not self.connect_to_dashboard():
            # If connection fails and we have new credentials, try with those
            if dashboard_url or api_key:
                # Update integration instance with new credentials
                self._dashboard_integration = MonitorDashboardIntegration(
                    dashboard_url=dashboard_url or self._dashboard_integration.dashboard_url,
                    api_key=api_key or self._dashboard_integration.api_key
                )
                if not self.connect_to_dashboard():
                    logger.error("Failed to connect to dashboard even with updated credentials")
                    return 0
            else:
                logger.error("Failed to connect to dashboard")
                return 0
                
        # Synchronize visualizations
        num_synced = self.sync_visualizations_with_dashboard(self.output_dir)
        
        if num_synced == 0:
            logger.warning("No visualizations synchronized with dashboard")
            return 0
        
        # Create panel if requested
        if create_panel and num_synced > 0:
            # Find visualization files
            import glob
            visualization_files = glob.glob(os.path.join(self.output_dir, "*.html"), recursive=False)
            
            if visualization_files:
                # Extract visualization IDs
                visualization_ids = [os.path.splitext(os.path.basename(path))[0] for path in visualization_files]
                
                # Create panel
                panel_id = self.create_dashboard_panel_from_visualizations(
                    panel_title=panel_title,
                    visualization_ids=visualization_ids,
                    layout_columns=2
                )
                
                if panel_id:
                    logger.info(f"Created dashboard panel with ID: {panel_id}")
                else:
                    logger.warning("Failed to create dashboard panel")
        
        return num_synced
    
    def setup_auto_dashboard_sync(
        self,
        interval_seconds: int = 60,
        max_runtime_minutes: Optional[int] = None,
        dashboard_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> bool:
        """
        Set up automatic synchronization with the monitoring dashboard.
        
        Args:
            interval_seconds: Interval between synchronization runs
            max_runtime_minutes: Maximum runtime in minutes (None for indefinite)
            dashboard_url: URL of the monitoring dashboard (overrides the one from initialization)
            api_key: API key for authentication (overrides the one from initialization)
            
        Returns:
            Success status of the setup
        """
        if not HAS_DASHBOARD_INTEGRATION:
            logger.error("Dashboard integration not available. Install with: pip install requests websocket-client")
            return False
        
        # Connect to dashboard and check if successful
        if not self.connect_to_dashboard():
            # If connection fails and we have new credentials, try with those
            if dashboard_url or api_key:
                # Update integration instance with new credentials
                self._dashboard_integration = MonitorDashboardIntegration(
                    dashboard_url=dashboard_url or self._dashboard_integration.dashboard_url,
                    api_key=api_key or self._dashboard_integration.api_key
                )
                if not self.connect_to_dashboard():
                    logger.error("Failed to connect to dashboard even with updated credentials")
                    return False
            else:
                logger.error("Failed to connect to dashboard")
                return False
        
        # Set up auto-sync
        self.setup_auto_dashboard_sync(
            interval_seconds=interval_seconds,
            output_dir=self.output_dir
        )
        
        logger.info(f"Automatic synchronization set up with interval {interval_seconds} seconds")
        
        if max_runtime_minutes is not None:
            logger.info(f"Automatic synchronization will run for {max_runtime_minutes} minutes")
            
            # Import in function to avoid issues if not available
            import threading
            import time
            
            def stop_after_timeout():
                time.sleep(max_runtime_minutes * 60)
                logger.info(f"Automatic synchronization completed after {max_runtime_minutes} minutes")
                
            # Start timeout thread
            timeout_thread = threading.Thread(target=stop_after_timeout, daemon=True)
            timeout_thread.start()
            
        return True
    
    def create_visualization_and_sync(
        self,
        visualization_type: str,
        visualization_params: Dict[str, Any],
        sync_with_dashboard: bool = True,
        dashboard_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a visualization and synchronize it with the dashboard.
        
        Args:
            visualization_type: Type of visualization to create (3d, heatmap, power, time-series)
            visualization_params: Parameters for creating the visualization
            sync_with_dashboard: Whether to synchronize with the dashboard
            dashboard_url: URL of the monitoring dashboard (overrides the one from initialization)
            api_key: API key for authentication (overrides the one from initialization)
            
        Returns:
            Path to the created visualization if successful, None otherwise
        """
        # Create visualization based on type
        viz_path = None
        
        if visualization_type == "3d":
            viz_path = self.create_3d_performance_visualization(**visualization_params)
        elif visualization_type == "heatmap":
            viz_path = self.create_hardware_comparison_heatmap(**visualization_params)
        elif visualization_type == "power":
            viz_path = self.create_power_efficiency_visualization(**visualization_params)
        elif visualization_type == "time-series":
            viz_path = self.create_animated_time_series_visualization(**visualization_params)
        else:
            logger.error(f"Unsupported visualization type: {visualization_type}")
            return None
        
        if not viz_path:
            logger.error("Failed to create visualization")
            return None
        
        # Synchronize with dashboard if requested
        if sync_with_dashboard and HAS_DASHBOARD_INTEGRATION:
            # Connect to dashboard
            if not self.connect_to_dashboard():
                if dashboard_url or api_key:
                    # Try with provided credentials
                    self._dashboard_integration = MonitorDashboardIntegration(
                        dashboard_url=dashboard_url or self._dashboard_integration.dashboard_url,
                        api_key=api_key or self._dashboard_integration.api_key
                    )
                    if not self.connect_to_dashboard():
                        logger.warning("Failed to connect to dashboard, visualization not synchronized")
                else:
                    logger.warning("Failed to connect to dashboard, visualization not synchronized")
            else:
                # Register visualization with dashboard
                viz_id = os.path.splitext(os.path.basename(viz_path))[0]
                viz_type = visualization_type
                title = visualization_params.get("title", f"{viz_type.title()} Visualization")
                
                result = self.register_visualization_with_dashboard(
                    visualization_id=viz_id,
                    visualization_type=viz_type,
                    title=title,
                    html_path=viz_path
                )
                
                if result:
                    logger.info(f"Visualization {viz_id} synchronized with dashboard")
                else:
                    logger.warning(f"Failed to synchronize visualization {viz_id} with dashboard")
        
        return viz_path
    
    def export_dashboard_snapshot(
        self,
        output_path: Optional[str] = None,
        include_visualizations: bool = True
    ) -> Optional[str]:
        """
        Export a snapshot of the monitoring dashboard.
        
        Args:
            output_path: Path to save the snapshot (default: generated in output_dir)
            include_visualizations: Whether to include visualization data in the snapshot
            
        Returns:
            Path to the snapshot file if successful, None otherwise
        """
        if not HAS_DASHBOARD_INTEGRATION:
            logger.error("Dashboard integration not available. Install with: pip install requests websocket-client")
            return None
        
        if not self.connect_to_dashboard():
            logger.error("Failed to connect to dashboard")
            return None
        
        # Generate default output path if not provided
        if output_path is None:
            # Use self.output_dir if available
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"dashboard_snapshot_{timestamp}.json")
        
        # Export snapshot
        return self._dashboard_integration.export_dashboard_snapshot(
            output_path=output_path,
            include_visualizations=include_visualizations
        )
    
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
        if not HAS_DASHBOARD_INTEGRATION:
            logger.error("Dashboard integration not available. Install with: pip install requests websocket-client")
            return False
        
        if not self.connect_to_dashboard():
            logger.error("Failed to connect to dashboard")
            return False
        
        # Check if snapshot file exists
        if not os.path.exists(snapshot_path):
            logger.error(f"Snapshot file not found: {snapshot_path}")
            return False
        
        # Import snapshot
        return self._dashboard_integration.import_dashboard_snapshot(
            snapshot_path=snapshot_path,
            target_dashboard_id=target_dashboard_id,
            merge_strategy=merge_strategy
        )


# Add the class to __all__ if this module is imported
__all__ = ['DashboardEnhancedVisualizationSystem']