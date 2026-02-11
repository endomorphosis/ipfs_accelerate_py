"""
Base Visualization Component for the Advanced Visualization System.

This module provides the base class for all visualization components in the
Advanced Visualization System. It handles common functionality like database
connections, configuration management, and output handling.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("base_visualization")

# Import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False


class BaseVisualization(ABC):
    """
    Base class for all visualization components.
    
    This abstract class provides common functionality for all visualization
    components including configuration management, data access, and output handling.
    """
    
    def __init__(self, db_connection=None, theme="light", debug=False):
        """Initialize the visualization with database connection and theme."""
        self.db_connection = db_connection
        self.theme = theme
        self.debug = debug
        self.figure = None
        self.data = None
        self._configure_theme()
    
    def _configure_theme(self):
        """Configure visualization style based on the theme."""
        self.theme_colors = {
            "light": {
                "background": "#ffffff",
                "text": "#333333",
                "grid": "#eeeeee",
                "accent1": "#1f77b4",
                "accent2": "#ff7f0e",
                "accent3": "#2ca02c",
                "accent4": "#d62728",
                "accent5": "#9467bd"
            },
            "dark": {
                "background": "#222222",
                "text": "#ffffff",
                "grid": "#444444",
                "accent1": "#1f77b4",
                "accent2": "#ff7f0e",
                "accent3": "#2ca02c",
                "accent4": "#d62728",
                "accent5": "#9467bd"
            }
        }.get(self.theme, {})
        
    def get_data(self, query_params=None):
        """Get data from database for visualization."""
        if self.db_connection is None:
            raise ValueError("Database connection is required")
        
        # Implementation will depend on specific visualization requirements
        pass
    
    def create_visualization(self, **kwargs):
        """Create the visualization."""
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement create_visualization")
    
    def export(self, filename, format="png", width=1200, height=800, scale=2):
        """Export visualization to a file."""
        if self.figure is None:
            raise ValueError("No visualization has been created")
        
        if format == "png":
            self.figure.write_image(filename, width=width, height=height, scale=scale)
        elif format == "html":
            self.figure.write_html(filename)
        elif format == "json":
            self.figure.write_json(filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        return True
    
    def show(self):
        """Display the visualization."""
        if self.figure is None:
            raise ValueError("No visualization has been created")
        return self.figure.show()
    
    def load_data(self, data_source):
        """Load data from various sources (DataFrame, file, etc.)"""
        if isinstance(data_source, pd.DataFrame):
            return data_source
        
        if isinstance(data_source, dict):
            return pd.DataFrame.from_dict(data_source)
        
        if isinstance(data_source, str):
            # Assume it's a file path
            path = Path(data_source)
            if not path.exists():
                logger.error(f"Data file not found: {path}")
                return pd.DataFrame()
            
            # Load based on file extension
            suffix = path.suffix.lower()
            try:
                if suffix == '.csv':
                    return pd.read_csv(path)
                elif suffix == '.json':
                    return pd.read_json(path)
                elif suffix in ['.xlsx', '.xls']:
                    return pd.read_excel(path)
                elif suffix == '.parquet':
                    return pd.read_parquet(path)
                else:
                    logger.error(f"Unsupported file format: {suffix}")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error loading data from {path}: {e}")
                return pd.DataFrame()
        
        logger.error(f"Unsupported data source type: {type(data_source)}")
        return pd.DataFrame()