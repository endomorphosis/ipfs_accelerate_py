#!/usr/bin/env python3
"""
Export Integration for Advanced Visualization System.

This module provides export methods that integrate with the AdvancedVisualizationSystem class.
"""

import os
import logging
import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
from plotly.graph_objects import Figure

# Import export utilities
from duckdb_api.visualization.advanced_visualization.export_manager import ExportManager

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("export_integration")


class ExportIntegration:
    """Export integration mixin for AdvancedVisualizationSystem."""
    
    def __init__(self):
        """Initialize export integration."""
        # This will be initialized when mixed into AdvancedVisualizationSystem
        self._export_manager = None
    
    def initialize_export_manager(self, output_dir: str) -> None:
        """
        Initialize the export manager.
        
        Args:
            output_dir: Directory to save exported files
        """
        self._export_manager = ExportManager(output_dir=output_dir)
        logger.info(f"Export manager initialized with output directory: {output_dir}")
    
    def export_visualization(
        self,
        component_type: str,
        component_data: Dict[str, Any],
        visualization_id: Optional[str] = None,
        formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export a visualization component to multiple formats.
        
        Args:
            component_type: Type of visualization ('3d', 'heatmap', 'power', 'time-series', 'dashboard')
            component_data: Visualization component data dictionary
            visualization_id: Unique identifier for the visualization (default: generated from type and timestamp)
            formats: List of formats to export to (default: component-specific defaults)
            output_dir: Directory to save exports (default: self.output_dir)
            settings: Export settings to override defaults
            
        Returns:
            Dictionary mapping formats to exported file paths
        """
        # Initialize export manager if needed
        if self._export_manager is None:
            self.initialize_export_manager(output_dir or getattr(self, 'output_dir', './exports'))
        elif output_dir and output_dir != self._export_manager.output_dir:
            # If output_dir is different from current export manager dir, create a new one
            self._export_manager = ExportManager(output_dir=output_dir)
        
        # Generate visualization_id if not provided
        if visualization_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            visualization_id = f"{component_type}_{timestamp}"
        
        # Export the visualization
        return self._export_manager.export_visualization(
            visualization_id=visualization_id,
            component_type=component_type,
            component_data=component_data,
            formats=formats,
            settings=settings
        )
    
    def export_3d_visualization(
        self,
        visualization_data: Dict[str, Any],
        formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        visualization_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export a 3D visualization to multiple formats.
        
        Args:
            visualization_data: 3D visualization data dictionary
            formats: List of formats to export to (default: ['html', 'png', 'pdf', 'json'])
            output_dir: Directory to save exports (default: self.output_dir)
            visualization_id: Unique identifier for the visualization
            settings: Export settings to override defaults
            
        Returns:
            Dictionary mapping formats to exported file paths
        """
        if formats is None:
            formats = ['html', 'png', 'pdf', 'json']
        
        return self.export_visualization(
            component_type='3d',
            component_data=visualization_data,
            visualization_id=visualization_id,
            formats=formats,
            output_dir=output_dir,
            settings=settings
        )
    
    def export_heatmap_visualization(
        self,
        visualization_data: Dict[str, Any],
        formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        visualization_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export a heatmap visualization to multiple formats.
        
        Args:
            visualization_data: Heatmap visualization data dictionary
            formats: List of formats to export to (default: ['html', 'png', 'pdf', 'json'])
            output_dir: Directory to save exports (default: self.output_dir)
            visualization_id: Unique identifier for the visualization
            settings: Export settings to override defaults
            
        Returns:
            Dictionary mapping formats to exported file paths
        """
        if formats is None:
            formats = ['html', 'png', 'pdf', 'json']
        
        return self.export_visualization(
            component_type='heatmap',
            component_data=visualization_data,
            visualization_id=visualization_id,
            formats=formats,
            output_dir=output_dir,
            settings=settings
        )
    
    def export_power_visualization(
        self,
        visualization_data: Dict[str, Any],
        formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        visualization_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export a power efficiency visualization to multiple formats.
        
        Args:
            visualization_data: Power visualization data dictionary
            formats: List of formats to export to (default: ['html', 'png', 'pdf', 'json'])
            output_dir: Directory to save exports (default: self.output_dir)
            visualization_id: Unique identifier for the visualization
            settings: Export settings to override defaults
            
        Returns:
            Dictionary mapping formats to exported file paths
        """
        if formats is None:
            formats = ['html', 'png', 'pdf', 'json']
        
        return self.export_visualization(
            component_type='power',
            component_data=visualization_data,
            visualization_id=visualization_id,
            formats=formats,
            output_dir=output_dir,
            settings=settings
        )
    
    def export_time_series_visualization(
        self,
        visualization_data: Dict[str, Any],
        formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        visualization_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        include_animation: bool = True
    ) -> Dict[str, str]:
        """
        Export a time series visualization to multiple formats, including animations.
        
        Args:
            visualization_data: Time series visualization data dictionary
            formats: List of formats to export to (default: ['html', 'png', 'pdf', 'json', 'mp4', 'gif'])
            output_dir: Directory to save exports (default: self.output_dir)
            visualization_id: Unique identifier for the visualization
            settings: Export settings to override defaults
            include_animation: Whether to include animation formats (mp4, gif)
            
        Returns:
            Dictionary mapping formats to exported file paths
        """
        if formats is None:
            formats = ['html', 'png', 'pdf', 'json']
            if include_animation:
                formats.extend(['mp4', 'gif'])
        
        return self.export_visualization(
            component_type='time-series',
            component_data=visualization_data,
            visualization_id=visualization_id,
            formats=formats,
            output_dir=output_dir,
            settings=settings
        )
    
    def export_dashboard(
        self,
        dashboard_name: str,
        format: str = "html",
        output_path: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export a dashboard to the specified format. This is an enhanced version of the previous export_dashboard method.
        
        Args:
            dashboard_name: Name of the dashboard to export
            format: Export format ('html', 'pdf', 'png')
            output_path: Path to save the exported file (default: derived from dashboard_name and format)
            settings: Export settings to override defaults
            
        Returns:
            Path to the exported file
        """
        # This function is expected to be provided by the main visualization system
        # We're just enhancing it with better integration with the export system
        
        # Get the dashboard data
        dashboard = self.get_dashboard(dashboard_name)
        if not dashboard:
            logger.error(f"Dashboard '{dashboard_name}' not found")
            return ""
        
        # Generate default output path if not provided
        if output_path is None:
            # Use self.output_dir if available, otherwise use export manager's output_dir
            base_dir = getattr(self, 'output_dir', None)
            if base_dir is None and self._export_manager:
                base_dir = self._export_manager.output_dir
            if base_dir is None:
                base_dir = "./exports"
            
            # Generate output filename
            output_path = os.path.join(base_dir, f"{dashboard_name}.{format}")
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Current implementation depends on the dashboard generation method in main system
        # This is just a placeholder for the enhanced export capabilities
        
        # For HTML format, existing dashboard HTML generation should be used
        if format == 'html':
            html_content = self._generate_dashboard_html(dashboard, theme=dashboard.get('theme', 'light'))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Exported dashboard to HTML: {output_path}")
            return output_path
        
        # For PDF and PNG formats, use export capabilities through Playwright
        elif format in ['pdf', 'png']:
            # First, export as HTML
            html_path = os.path.join(output_dir, f"{dashboard_name}_temp.html")
            html_content = self._generate_dashboard_html(dashboard, theme=dashboard.get('theme', 'light'))
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            try:
                # Use Playwright to export to PDF or PNG
                self._export_html_to_format(html_path, output_path, format, settings)
                logger.info(f"Exported dashboard to {format.upper()}: {output_path}")
            except Exception as e:
                logger.error(f"Error exporting dashboard to {format}: {e}")
                return ""
            finally:
                # Clean up temporary HTML file
                if os.path.exists(html_path):
                    os.remove(html_path)
            
            return output_path
        
        else:
            logger.error(f"Unsupported export format for dashboard: {format}")
            return ""
    
    def _export_html_to_format(
        self,
        html_path: str,
        output_path: str,
        format: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Export an HTML file to PDF or PNG format using Playwright.
        
        Args:
            html_path: Path to the HTML file
            output_path: Path to save the exported file
            format: Export format ('pdf' or 'png')
            settings: Export settings including width, height, etc.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.error("Playwright is required for PDF/PNG export. Install with: pip install playwright")
            logger.error("After installation, run: playwright install chromium")
            raise
        
        if settings is None:
            settings = {}
        
        width = settings.get('width', 1200)
        height = settings.get('height', 1600)
        scale = settings.get('scale', 1)
        
        html_file_url = f"file://{os.path.abspath(html_path)}"
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=scale
            )
            page = context.new_page()
            page.goto(html_file_url)
            
            # Wait for all visualizations to load
            page.wait_for_timeout(2000)
            
            if format == 'pdf':
                # Export to PDF
                page.pdf(
                    path=output_path,
                    width=f"{width}px",
                    height=f"{height}px",
                    print_background=True,
                    margin={"top": "20px", "right": "20px", "bottom": "20px", "left": "20px"}
                )
            elif format == 'png':
                # Export to PNG
                page.screenshot(
                    path=output_path,
                    full_page=True,
                    type='png'
                )
            
            browser.close()
    
    def export_all_visualizations(
        self,
        visualizations: Dict[str, Dict[str, Any]],
        output_dir: Optional[str] = None,
        formats: Optional[Dict[str, List[str]]] = None,
        settings: Optional[Dict[str, Any]] = None,
        create_index: bool = True,
        title: str = "All Visualizations"
    ) -> Dict[str, Dict[str, str]]:
        """
        Export all visualizations in a batch.
        
        Args:
            visualizations: Dictionary mapping visualization IDs to component data dictionaries
            output_dir: Directory to save exports (default: self.output_dir)
            formats: Dictionary mapping component types to lists of formats
            settings: Export settings to override defaults
            create_index: Whether to create an HTML index page with links to all exports
            title: Title for the index page
            
        Returns:
            Nested dictionary mapping visualization IDs to format->path mappings
        """
        # Initialize export manager if needed
        if self._export_manager is None:
            self.initialize_export_manager(output_dir or getattr(self, 'output_dir', './exports'))
        elif output_dir and output_dir != self._export_manager.output_dir:
            # If output_dir is different from current export manager dir, create a new one
            self._export_manager = ExportManager(output_dir=output_dir)
        
        # Use export manager to export multiple visualizations
        return self._export_manager.export_multiple_visualizations(
            visualizations=visualizations,
            formats=formats,
            settings=settings,
            create_index=create_index,
            title=title
        )
    
    def generate_export_report(
        self,
        title: str = "Visualization Export Report",
        description: str = "Comprehensive report of exported visualizations",
        output_path: Optional[str] = None,
        visualization_ids: Optional[List[str]] = None
    ) -> str:
        """
        Generate a comprehensive report of all exported visualizations.
        
        Args:
            title: Title for the report
            description: Description for the report
            output_path: Path to save the report (default: output_dir/export_report.html)
            visualization_ids: List of specific visualization IDs to include (default: all)
            
        Returns:
            Path to the generated report file
        """
        # Initialize export manager if needed
        if self._export_manager is None:
            self.initialize_export_manager(getattr(self, 'output_dir', './exports'))
        
        # Generate the report
        return self._export_manager.create_export_report(
            visualization_ids=visualization_ids,
            output_path=output_path,
            title=title,
            description=description
        )
    
    def export_animated_time_series(
        self,
        visualization_data: Dict[str, Any],
        format: str = "mp4",
        output_path: Optional[str] = None,
        visualization_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export an animated time series visualization with optimized settings.
        
        Args:
            visualization_data: Time series visualization data dictionary
            format: Export format ('mp4' or 'gif')
            output_path: Path to save the exported file
            visualization_id: Unique identifier for the visualization
            settings: Export settings to override defaults
            
        Returns:
            Path to the exported file
        """
        # Initialize export manager if needed
        if self._export_manager is None:
            output_dir = getattr(self, 'output_dir', './exports')
            self.initialize_export_manager(output_dir)
        
        # Generate visualization_id if not provided
        if visualization_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            visualization_id = f"time_series_animation_{timestamp}"
        
        # Export the animation
        return self._export_manager.export_animation_optimized(
            visualization_id=visualization_id,
            component_data=visualization_data,
            format=format,
            settings=settings,
            output_path=output_path
        )
    
    def configure_export_settings(self, settings: Dict[str, Any]) -> None:
        """
        Configure default export settings.
        
        Args:
            settings: Dictionary of export settings to use as defaults
        """
        # Initialize export manager if needed
        if self._export_manager is None:
            self.initialize_export_manager(getattr(self, 'output_dir', './exports'))
        
        # Configure settings
        self._export_manager.set_default_settings(settings)
        logger.info(f"Configured export settings: {settings}")
    
    def save_export_metadata(self, output_path: Optional[str] = None) -> str:
        """
        Save metadata about all exports to a JSON file.
        
        Args:
            output_path: Path to save the metadata file
            
        Returns:
            Path to the metadata file
        """
        # Initialize export manager if needed
        if self._export_manager is None:
            self.initialize_export_manager(getattr(self, 'output_dir', './exports'))
        
        # Save metadata
        return self._export_manager.save_export_metadata(output_path=output_path)