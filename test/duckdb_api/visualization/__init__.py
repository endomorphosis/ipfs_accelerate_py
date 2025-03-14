"""
Visualization module for the IPFS Accelerate Python Framework.

This module provides various visualization components for analyzing performance data
and creating interactive visualizations for different aspects of the system.

Submodules:
- advanced_visualization: Interactive visualization components for performance analysis
"""

# Import advanced visualization components for easier access
from duckdb_api.visualization.advanced_visualization import (
    BaseVisualization,
    HardwareHeatmapVisualization, 
    TimeSeriesVisualization,
    Visualization3D
)

__all__ = [
    "BaseVisualization",
    "HardwareHeatmapVisualization",
    "TimeSeriesVisualization",
    "Visualization3D",
]
