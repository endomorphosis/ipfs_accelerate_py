"""
Advanced Visualization System - Provides interactive and static visualizations 
for exploring performance data across hardware types, model families, and metrics.

This package includes various visualization components:
- BaseVisualization: Common functionality for all visualization components
- HardwareHeatmapVisualization: Heatmaps for comparing performance across hardware platforms
- TimeSeriesVisualization: Time series plots for tracking metrics over time
- Visualization3D: Interactive 3D visualizations for multi-dimensional data exploration
- AnimatedTimeSeriesVisualization: Animated time series visualizations with interactive controls
- CustomizableDashboard: Customizable dashboard system for combining multiple visualizations
- MonitorDashboardIntegration: Integration with Monitoring Dashboard for centralized visualization
"""

from duckdb_api.visualization.advanced_visualization.base import BaseVisualization
from duckdb_api.visualization.advanced_visualization.viz_heatmap import HardwareHeatmapVisualization
from duckdb_api.visualization.advanced_visualization.viz_time_series import TimeSeriesVisualization
from duckdb_api.visualization.advanced_visualization.viz_3d import Visualization3D
from duckdb_api.visualization.advanced_visualization.viz_animated_time_series import AnimatedTimeSeriesVisualization
from duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard import CustomizableDashboard

# Import dashboard integration
try:
    from duckdb_api.visualization.advanced_visualization.monitor_dashboard_integration import (
        MonitorDashboardIntegration,
        MonitorDashboardIntegrationMixin
    )
    HAS_DASHBOARD_INTEGRATION = True
except ImportError:
    import logging
    logging.getLogger(__name__).warning(
        "MonitorDashboardIntegration not available. Install with: pip install requests websocket-client"
    )
    HAS_DASHBOARD_INTEGRATION = False

__all__ = [
    'BaseVisualization',
    'HardwareHeatmapVisualization',
    'TimeSeriesVisualization',
    'Visualization3D',
    'AnimatedTimeSeriesVisualization',
    'CustomizableDashboard',
]

# Add dashboard integration to __all__ if available
if HAS_DASHBOARD_INTEGRATION:
    __all__.extend(['MonitorDashboardIntegration', 'MonitorDashboardIntegrationMixin'])