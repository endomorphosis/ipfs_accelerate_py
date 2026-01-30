#!/usr/bin/env python3
"""
Export Manager for Advanced Visualization System.

This module integrates the export utilities with the Advanced Visualization System,
providing a unified interface for exporting visualizations to various formats.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import plotly.graph_objects as go

# Import export utilities
from data.duckdb.visualization.advanced_visualization.export_utils import (
    export_figure,
    export_animation,
    export_data,
    export_visualization_component,
    export_visualization_component_all_formats,
    batch_export_visualizations,
    create_export_index,
    SUPPORTED_FORMATS,
    DEFAULT_EXPORT_SETTINGS
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("export_manager")


class ExportManager:
    """Manager for exporting visualizations to various formats."""
    
    def __init__(self, output_dir: str = "./exports"):
        """
        Initialize the export manager.

        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Default export settings
        self.default_settings = DEFAULT_EXPORT_SETTINGS.copy()
        
        # Keep track of exported visualizations
        self.exports = {}
        
        logger.info(f"Export manager initialized with output directory: {output_dir}")
    
    def set_default_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update default export settings.

        Args:
            settings: Dictionary of export settings
        """
        self.default_settings.update(settings)
        logger.info(f"Updated default export settings: {settings}")
    
    def export_visualization(
        self,
        visualization_id: str,
        component_type: str,
        component_data: Dict[str, Any],
        formats: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None,
        create_manifest: bool = True,
        subfolder: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export a visualization to multiple formats.

        Args:
            visualization_id: Unique identifier for the visualization
            component_type: Type of visualization component ('3d', 'heatmap', 'power', 'time-series')
            component_data: Dictionary containing component data including figure and metadata
            formats: List of formats to export to (default uses appropriate formats for the component type)
            settings: Dictionary of export settings (overrides default settings)
            create_manifest: Whether to create a manifest file documenting the exports
            subfolder: Optional subfolder within output_dir to save exports

        Returns:
            Dictionary mapping formats to file paths
        """
        # Set output directory
        export_dir = self.output_dir
        if subfolder:
            export_dir = os.path.join(export_dir, subfolder)
            os.makedirs(export_dir, exist_ok=True)
        
        # Merge settings with defaults
        merged_settings = self.default_settings.copy()
        if settings:
            merged_settings.update(settings)
        
        # Export the visualization
        exports = export_visualization_component_all_formats(
            component_type=component_type,
            component_data=component_data,
            output_dir=export_dir,
            base_name=visualization_id,
            formats=formats,
            settings=merged_settings,
            create_manifest=create_manifest
        )
        
        # Store export information
        self.exports[visualization_id] = {
            "component_type": component_type,
            "exports": exports,
            "timestamp": datetime.datetime.now().isoformat(),
            "export_dir": export_dir
        }
        
        return exports
    
    def export_multiple_visualizations(
        self,
        visualizations: Dict[str, Dict[str, Any]],
        formats: Optional[Dict[str, List[str]]] = None,
        settings: Optional[Dict[str, Any]] = None,
        create_index: bool = True,
        subfolder: Optional[str] = None,
        title: str = "Visualization Exports"
    ) -> Dict[str, Dict[str, str]]:
        """
        Export multiple visualizations to various formats.

        Args:
            visualizations: Dictionary mapping visualization IDs to component data
                           Each component data dict must have a 'component_type' key
            formats: Dictionary mapping component types to lists of formats
            settings: Dictionary of export settings (overrides default settings)
            create_index: Whether to create an HTML index page for all exports
            subfolder: Optional subfolder within output_dir to save exports
            title: Title for the index page

        Returns:
            Nested dictionary mapping visualization IDs to format->path mappings
        """
        # Set output directory
        export_dir = self.output_dir
        if subfolder:
            export_dir = os.path.join(export_dir, subfolder)
            os.makedirs(export_dir, exist_ok=True)
        
        # Merge settings with defaults
        merged_settings = self.default_settings.copy()
        if settings:
            merged_settings.update(settings)
        
        # Build visualizations dictionary with component types
        typed_visualizations = {}
        for viz_id, viz_data in visualizations.items():
            if "component_type" not in viz_data:
                logger.warning(f"Skipping visualization {viz_id}: missing component_type")
                continue
            typed_visualizations[viz_id] = viz_data
        
        # Export the visualizations
        exports = batch_export_visualizations(
            visualizations=typed_visualizations,
            output_dir=export_dir,
            formats=formats,
            settings=merged_settings
        )
        
        # Create index page if requested
        if create_index and exports:
            index_path = os.path.join(export_dir, "index.html")
            create_export_index(exports, index_path, title)
            
            # Add index to exports
            for viz_id in exports:
                exports[viz_id]["index"] = index_path
        
        # Store export information
        for viz_id, viz_exports in exports.items():
            self.exports[viz_id] = {
                "component_type": typed_visualizations[viz_id]["component_type"],
                "exports": viz_exports,
                "timestamp": datetime.datetime.now().isoformat(),
                "export_dir": export_dir
            }
        
        return exports
    
    def get_export_history(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the history of exported visualizations.

        Returns:
            Dictionary mapping visualization IDs to export information
        """
        return self.exports
    
    def create_export_report(
        self,
        visualization_ids: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        title: str = "Visualization Export Report",
        description: str = "Report of exported visualizations"
    ) -> str:
        """
        Create a comprehensive report of exported visualizations.

        Args:
            visualization_ids: List of visualization IDs to include (default: all)
            output_path: Path to save the report (default: output_dir/export_report.html)
            title: Title for the report
            description: Description for the report

        Returns:
            Path to the report file
        """
        if not self.exports:
            logger.warning("No exports available to create report")
            return ""
        
        # Set default output path
        if output_path is None:
            output_path = os.path.join(self.output_dir, "export_report.html")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Filter exports if visualization_ids provided
        if visualization_ids:
            exports_to_include = {viz_id: self.exports[viz_id] for viz_id in visualization_ids 
                                if viz_id in self.exports}
        else:
            exports_to_include = self.exports
        
        if not exports_to_include:
            logger.warning("No matching exports found to create report")
            return ""
        
        # Build report HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
        }}
        p {{
            margin-bottom: 20px;
        }}
        .report-summary {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .visualization {{
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }}
        .formats {{
            margin-top: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .format-link {{
            background-color: #3498db;
            color: white;
            padding: 5px 15px;
            text-decoration: none;
            border-radius: 3px;
            display: inline-block;
            transition: background-color 0.3s;
        }}
        .format-link:hover {{
            background-color: #2980b9;
        }}
        .preview {{
            margin-top: 20px;
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }}
        .preview img {{
            max-width: 100%;
            display: block;
        }}
        .metadata {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            font-size: 0.9em;
        }}
        .metadata h3 {{
            margin-top: 0;
            color: #555;
        }}
        .metadata-item {{
            margin-bottom: 5px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="report-summary">
        <p>{description}</p>
        <p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Total Visualizations: {len(exports_to_include)}</p>
    </div>
"""

        # Add each visualization
        for viz_id, viz_info in exports_to_include.items():
            component_type = viz_info.get("component_type", "unknown")
            exports = viz_info.get("exports", {})
            timestamp = viz_info.get("timestamp", "")
            
            # Format timestamp
            try:
                timestamp_dt = datetime.datetime.fromisoformat(timestamp)
                formatted_timestamp = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                formatted_timestamp = timestamp
            
            html_content += f"""
    <div class="visualization">
        <h2>{viz_id}</h2>
        <div class="timestamp">Exported: {formatted_timestamp}</div>
        <div class="metadata-item">Type: {component_type}</div>
        <div class="formats">
"""
            
            # Add links to each format
            for format, path in exports.items():
                if format != 'manifest' and format != 'index':
                    rel_path = os.path.relpath(path, os.path.dirname(output_path))
                    html_content += f'            <a class="format-link" href="{rel_path}">{format.upper()}</a>\n'
            
            html_content += "        </div>\n"
            
            # Add preview (PNG if available, otherwise skip)
            if 'png' in exports:
                rel_path = os.path.relpath(exports['png'], os.path.dirname(output_path))
                html_content += f"""
        <div class="preview">
            <img src="{rel_path}" alt="{viz_id} preview">
        </div>
"""
            
            html_content += "    </div>\n"

        # Close HTML
        html_content += """
</body>
</html>
"""

        # Write the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Created export report at {output_path}")
        return output_path
    
    def export_animation_optimized(
        self,
        visualization_id: str,
        component_data: Dict[str, Any],
        format: str = "mp4",
        settings: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export an animated visualization with optimized settings for high-quality output.

        Args:
            visualization_id: Unique identifier for the visualization
            component_data: Dictionary containing component data including figure
            format: Export format ('mp4' or 'gif')
            settings: Dictionary of export settings (overrides default settings)
            output_path: Path to save the exported file (default: output_dir/visualization_id.format)

        Returns:
            Path to the exported file
        """
        if format not in ['mp4', 'gif']:
            raise ValueError(f"Unsupported animation format: {format}. Use 'mp4' or 'gif'.")
        
        # Set default output path
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{visualization_id}.{format}")
        
        # Merge settings with defaults
        merged_settings = self.default_settings.copy()
        merged_settings.update({
            'width': 1920,  # HD resolution
            'height': 1080,
            'scale': 2,
            'fps': 30,  # High frame rate
            'duration': 15000,  # 15 seconds
            'frame_duration': 50,  # Smooth animation
            'transition_duration': 100,
            'redraw': True,
            'easing': 'cubic-in-out'
        })
        if settings:
            merged_settings.update(settings)
        
        # Get figure from component data
        fig = component_data.get('figure')
        if fig is None:
            raise ValueError(f"No figure found in component data")
        
        # Export the animation
        exported_path = export_animation(
            fig=fig,
            output_path=output_path,
            format=format,
            settings=merged_settings
        )
        
        # Store export information
        self.exports[visualization_id] = {
            "component_type": "time-series",
            "exports": {format: exported_path},
            "timestamp": datetime.datetime.now().isoformat(),
            "export_dir": os.path.dirname(output_path)
        }
        
        return exported_path
    
    def save_export_metadata(self, output_path: Optional[str] = None) -> str:
        """
        Save metadata about all exports to a JSON file.

        Args:
            output_path: Path to save the metadata file (default: output_dir/export_metadata.json)

        Returns:
            Path to the metadata file
        """
        # Set default output path
        if output_path is None:
            output_path = os.path.join(self.output_dir, "export_metadata.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build metadata
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "export_count": len(self.exports),
            "output_directory": self.output_dir,
            "exports": {}
        }
        
        # Add metadata for each export
        for viz_id, viz_info in self.exports.items():
            metadata["exports"][viz_id] = {
                "component_type": viz_info.get("component_type", "unknown"),
                "timestamp": viz_info.get("timestamp", ""),
                "export_dir": viz_info.get("export_dir", ""),
                "formats": list(viz_info.get("exports", {}).keys())
            }
        
        # Write the file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved export metadata to {output_path}")
        return output_path