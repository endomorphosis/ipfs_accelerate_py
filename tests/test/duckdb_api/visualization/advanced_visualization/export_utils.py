#!/usr/bin/env python3
"""
Export utilities for the Advanced Visualization System.

This module provides functions for exporting visualizations to various formats,
including HTML, PNG, PDF, SVG, JSON, CSV, MP4, and GIF.
"""

import os
import re
import base64
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.io import write_image, to_html, to_json

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("visualization_export")

# Dictionary of supported export formats and their file extensions
SUPPORTED_FORMATS = {
    'html': '.html',
    'png': '.png',
    'pdf': '.pdf',
    'svg': '.svg',
    'json': '.json',
    'csv': '.csv',
    'mp4': '.mp4',
    'gif': '.gif',
}

# Default export settings
DEFAULT_EXPORT_SETTINGS = {
    'width': 1200,
    'height': 800,
    'scale': 2,  # Higher scale for better resolution
    'include_plotlyjs': True,
    'include_mathjax': False,
    'full_html': True,
}


def export_figure(
    fig: go.Figure,
    output_path: str,
    format: str = "html",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export a Plotly figure to the specified format.

    Args:
        fig: The Plotly figure to export
        output_path: Path to save the exported file
        format: Export format ('html', 'png', 'pdf', 'svg', 'json')
        settings: Dictionary of export settings

    Returns:
        Path to the exported file
    """
    if settings is None:
        settings = DEFAULT_EXPORT_SETTINGS.copy()

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # If no file extension is provided, add the appropriate one
    if not os.path.splitext(output_path)[1]:
        output_path += SUPPORTED_FORMATS.get(format, '.html')

    try:
        if format == 'html':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(to_html(
                    fig,
                    include_plotlyjs=settings.get('include_plotlyjs', True),
                    include_mathjax=settings.get('include_mathjax', False),
                    full_html=settings.get('full_html', True),
                    auto_open=settings.get('auto_open', False),
                ))
        elif format in ['png', 'pdf', 'svg']:
            write_image(
                fig,
                output_path,
                width=settings.get('width', 1200),
                height=settings.get('height', 800),
                scale=settings.get('scale', 2),
                engine=settings.get('engine', 'kaleido'),
            )
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(to_json(fig))
        else:
            raise ValueError(f"Unsupported export format: {format}")
    except Exception as e:
        logger.error(f"Error exporting figure to {format}: {e}")
        raise

    logger.info(f"Exported figure to {output_path}")
    return output_path


def export_animation(
    fig: go.Figure,
    output_path: str,
    format: str = "mp4",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export an animated figure to MP4 or GIF format.

    Args:
        fig: The Plotly figure to export (must have animation frames)
        output_path: Path to save the exported file
        format: Export format ('mp4' or 'gif')
        settings: Dictionary of export settings

    Returns:
        Path to the exported file
    """
    if settings is None:
        settings = DEFAULT_EXPORT_SETTINGS.copy()

    if format not in ['mp4', 'gif']:
        raise ValueError(f"Unsupported animation format: {format}. Use 'mp4' or 'gif'.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # If no file extension is provided, add the appropriate one
    if not os.path.splitext(output_path)[1]:
        output_path += SUPPORTED_FORMATS.get(format, f'.{format}')

    # First, export as HTML with animation
    temp_html_path = f"{os.path.splitext(output_path)[0]}_temp.html"
    
    with open(temp_html_path, 'w', encoding='utf-8') as f:
        f.write(to_html(
            fig,
            include_plotlyjs=True,
            full_html=True,
            auto_play=True,
            animation_opts={
                'frame': {
                    'duration': settings.get('frame_duration', 500),
                    'redraw': settings.get('redraw', True),
                },
                'transition': {
                    'duration': settings.get('transition_duration', 500),
                    'easing': settings.get('easing', 'cubic-in-out'),
                },
            },
        ))

    try:
        # Use playwright to capture the animation
        capture_animation_with_playwright(
            temp_html_path,
            output_path,
            format=format,
            width=settings.get('width', 1200),
            height=settings.get('height', 800),
            duration=settings.get('duration', 10000),  # Default 10 seconds
            fps=settings.get('fps', 30),  # Frames per second
        )
    except Exception as e:
        logger.error(f"Error exporting animation to {format}: {e}")
        
        # Fallback to static image if animation export fails
        logger.warning(f"Falling back to static image export")
        export_figure(fig, output_path, format='png' if format == 'gif' else 'png', settings=settings)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)

    logger.info(f"Exported animation to {output_path}")
    return output_path


def capture_animation_with_playwright(
    html_path: str,
    output_path: str,
    format: str = "mp4",
    width: int = 1200,
    height: int = 800,
    duration: int = 10000,  # Duration in milliseconds
    fps: int = 30,
) -> None:
    """
    Use Playwright to capture an HTML animation as MP4 or GIF.

    Args:
        html_path: Path to the HTML file with animation
        output_path: Path to save the exported file
        format: Export format ('mp4' or 'gif')
        width: Width of the output video
        height: Height of the output video
        duration: Duration of the capture in milliseconds
        fps: Frames per second for the capture
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright is required for animation export. Install with: pip install playwright")
        logger.error("After installation, run: playwright install chromium")
        raise

    html_file_url = f"file://{os.path.abspath(html_path)}"
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(html_file_url)
        
        # Wait for the animation to load
        page.wait_for_timeout(1000)  # Give it a second to initialize
        
        if format == 'mp4':
            # Start recording
            page.video.start(path=output_path, width=width, height=height, fps=fps)
            # Let the animation play for the specified duration
            page.wait_for_timeout(duration)
            # Stop recording and close
            browser.close()
        else:  # GIF format
            # For GIF, we need to take screenshots and combine them
            temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            
            frame_count = int(duration * fps / 1000)
            frame_interval = duration / frame_count
            
            # Capture frames
            frames = []
            for i in range(frame_count):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                page.screenshot(path=frame_path)
                frames.append(frame_path)
                page.wait_for_timeout(frame_interval)
            
            browser.close()
            
            # Combine frames into GIF using ImageMagick
            try:
                subprocess.run([
                    "convert",
                    "-delay", str(100 // fps),  # Convert fps to delay (1/100 seconds)
                    "-loop", "0",  # Loop forever
                    *frames,
                    output_path
                ], check=True)
            except subprocess.CalledProcessError:
                logger.error("ImageMagick is required for GIF creation. Please install ImageMagick.")
                raise
            finally:
                # Clean up temporary frames
                for frame in frames:
                    if os.path.exists(frame):
                        os.remove(frame)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)


def export_data(
    data: Union[pd.DataFrame, Dict, List],
    output_path: str,
    format: str = "csv",
) -> str:
    """
    Export data to CSV or JSON format.

    Args:
        data: The data to export (DataFrame, dict, or list)
        output_path: Path to save the exported file
        format: Export format ('csv' or 'json')

    Returns:
        Path to the exported file
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # If no file extension is provided, add the appropriate one
    if not os.path.splitext(output_path)[1]:
        output_path += SUPPORTED_FORMATS.get(format, f'.{format}')

    try:
        if format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)
            else:
                pd.DataFrame(data).to_csv(output_path, index=False)
        elif format == 'json':
            if isinstance(data, pd.DataFrame):
                data.to_json(output_path, orient='records', indent=2)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported data export format: {format}")
    except Exception as e:
        logger.error(f"Error exporting data to {format}: {e}")
        raise

    logger.info(f"Exported data to {output_path}")
    return output_path


def export_visualization_component(
    component_type: str,
    component_data: Dict[str, Any],
    output_path: str,
    format: str = "html",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export a visualization component to the specified format.

    Args:
        component_type: Type of visualization component ('3d', 'heatmap', 'power', 'time-series')
        component_data: Dictionary containing component data including figure and metadata
        output_path: Path to save the exported file
        format: Export format ('html', 'png', 'pdf', 'svg', 'json', 'csv', 'mp4', 'gif')
        settings: Dictionary of export settings

    Returns:
        Path to the exported file
    """
    if settings is None:
        settings = DEFAULT_EXPORT_SETTINGS.copy()

    fig = component_data.get('figure')
    if fig is None:
        raise ValueError(f"No figure found in component data")

    # Export based on format and component type
    if format in ['html', 'png', 'pdf', 'svg', 'json']:
        return export_figure(fig, output_path, format, settings)
    elif format in ['mp4', 'gif'] and component_type == 'time-series':
        return export_animation(fig, output_path, format, settings)
    elif format in ['csv', 'json'] and 'data' in component_data:
        return export_data(component_data['data'], output_path, format)
    else:
        # Fallback to standard figure export
        logger.warning(f"Format {format} not supported for component type {component_type}, falling back to HTML export")
        return export_figure(fig, output_path, 'html', settings)


def extract_base64_images(html_content: str, output_dir: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract base64 encoded images from HTML content and save them as files.

    Args:
        html_content: HTML content containing base64 encoded images
        output_dir: Directory to save the extracted images

    Returns:
        Tuple of (updated HTML content, dictionary of image paths)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = {}
    pattern = r'data:image/([a-zA-Z]+);base64,([^"\'\\]+)'
    
    def replace_match(match):
        image_format = match.group(1)
        base64_data = match.group(2)
        
        # Generate a unique filename
        filename = f"image_{len(image_paths)}.{image_format}"
        image_path = os.path.join(output_dir, filename)
        
        # Save the image
        try:
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(base64_data))
            
            image_paths[filename] = image_path
            
            # Return the image URL
            return f'./images/{filename}'
        except Exception as e:
            logger.error(f"Error saving base64 image: {e}")
            return match.group(0)  # Return the original base64 data
    
    # Replace base64 images with file references
    updated_html = re.sub(pattern, replace_match, html_content)
    
    return updated_html, image_paths


def get_visualization_metadata(component_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and format metadata from visualization component data.
    
    Args:
        component_data: Dictionary containing component data
        
    Returns:
        Dictionary of formatted metadata
    """
    metadata = {}
    
    # Extract common metadata
    for key in ["title", "description", "metrics", "dimensions", "timestamp", "config"]:
        if key in component_data:
            metadata[key] = component_data[key]
    
    # Add any component-specific metadata
    if "component_type" in component_data:
        metadata["type"] = component_data["component_type"]
    
    if "creation_date" in component_data:
        metadata["created"] = component_data["creation_date"]
        
    # Add data summary if data is available
    if "data" in component_data and isinstance(component_data["data"], pd.DataFrame):
        df = component_data["data"]
        metadata["data_summary"] = {
            "rows": len(df),
            "columns": list(df.columns),
            "metrics": {col: {"min": float(df[col].min()), 
                              "max": float(df[col].max()), 
                              "mean": float(df[col].mean())} 
                       for col in df.select_dtypes(include=['number']).columns}
        }
    
    return metadata


def create_export_manifest(
    exports: Dict[str, str],
    component_data: Dict[str, Any],
    manifest_path: str
) -> str:
    """
    Create a manifest file documenting all exported files with metadata.
    
    Args:
        exports: Dictionary mapping export formats to file paths
        component_data: Original component data containing metadata
        manifest_path: Path to save the manifest file
        
    Returns:
        Path to the manifest file
    """
    # Extract metadata
    metadata = get_visualization_metadata(component_data)
    
    # Build manifest
    manifest = {
        "title": metadata.get("title", "Exported Visualization"),
        "type": metadata.get("type", "visualization"),
        "created": metadata.get("creation_date", pd.Timestamp.now().isoformat()),
        "exports": {format: os.path.basename(path) for format, path in exports.items()},
        "metadata": metadata
    }
    
    # Write manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created export manifest at {manifest_path}")
    return manifest_path


def export_visualization_component_all_formats(
    component_type: str,
    component_data: Dict[str, Any],
    output_dir: str,
    base_name: str,
    formats: Optional[List[str]] = None,
    settings: Optional[Dict[str, Any]] = None,
    create_manifest: bool = True
) -> Dict[str, str]:
    """
    Export a visualization component to multiple formats.

    Args:
        component_type: Type of visualization component ('3d', 'heatmap', 'power', 'time-series')
        component_data: Dictionary containing component data including figure and metadata
        output_dir: Directory to save the exported files
        base_name: Base name for output files
        formats: List of formats to export to (default: html, png, pdf, json)
        settings: Dictionary of export settings
        create_manifest: Whether to create a manifest file documenting the exports

    Returns:
        Dictionary mapping formats to file paths
    """
    if formats is None:
        if component_type == 'time-series':
            formats = ['html', 'png', 'pdf', 'json', 'mp4', 'gif']
        else:
            formats = ['html', 'png', 'pdf', 'json']
    
    if settings is None:
        settings = DEFAULT_EXPORT_SETTINGS.copy()

    os.makedirs(output_dir, exist_ok=True)
    exports = {}

    for format in formats:
        output_path = os.path.join(output_dir, f"{base_name}.{format}")
        try:
            exported_path = export_visualization_component(
                component_type, component_data, output_path, format, settings
            )
            exports[format] = exported_path
        except Exception as e:
            logger.error(f"Error exporting to {format}: {e}")
            # Continue with other formats

    # Create manifest file if requested
    if create_manifest and exports:
        manifest_path = os.path.join(output_dir, f"{base_name}_manifest.json")
        create_export_manifest(exports, component_data, manifest_path)
        exports['manifest'] = manifest_path

    return exports


def batch_export_visualizations(
    visualizations: Dict[str, Dict[str, Any]],
    output_dir: str,
    formats: Optional[Dict[str, List[str]]] = None,
    settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, str]]:
    """
    Batch export multiple visualizations to various formats.

    Args:
        visualizations: Dictionary mapping visualization names to component data
        output_dir: Directory to save the exported files
        formats: Dictionary mapping component types to lists of formats
        settings: Dictionary of export settings

    Returns:
        Nested dictionary mapping visualization names to format->path mappings
    """
    if formats is None:
        formats = {
            '3d': ['html', 'png', 'pdf', 'json'],
            'heatmap': ['html', 'png', 'pdf', 'json'],
            'power': ['html', 'png', 'pdf', 'json'],
            'time-series': ['html', 'png', 'pdf', 'json', 'mp4', 'gif']
        }
    
    if settings is None:
        settings = DEFAULT_EXPORT_SETTINGS.copy()

    os.makedirs(output_dir, exist_ok=True)
    exports = {}

    for name, component_data in visualizations.items():
        component_type = component_data.get('component_type', '3d')
        component_formats = formats.get(component_type, ['html', 'png'])
        
        component_exports = export_visualization_component_all_formats(
            component_type=component_type,
            component_data=component_data,
            output_dir=output_dir,
            base_name=name,
            formats=component_formats,
            settings=settings
        )
        
        exports[name] = component_exports

    return exports


def create_export_index(
    exports: Dict[str, Dict[str, str]],
    output_path: str,
    title: str = "Visualization Exports"
) -> str:
    """
    Create an HTML index page for all exported visualizations.

    Args:
        exports: Nested dictionary mapping visualization names to format->path mappings
        output_path: Path to save the index file
        title: Title for the index page

    Returns:
        Path to the index file
    """
    # Build HTML content
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
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
"""

    # Add each visualization
    for name, format_paths in exports.items():
        html_content += f"""
    <div class="visualization">
        <h2>{name}</h2>
        <div class="formats">
"""
        
        # Add links to each format
        for format, path in format_paths.items():
            if format != 'manifest':
                rel_path = os.path.basename(path)
                html_content += f'            <a class="format-link" href="{rel_path}">{format.upper()}</a>\n'
        
        html_content += "        </div>\n"
        
        # Add preview (PNG if available, otherwise skip)
        if 'png' in format_paths:
            rel_path = os.path.basename(format_paths['png'])
            html_content += f"""
        <div class="preview">
            <img src="{rel_path}" alt="{name} preview">
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

    logger.info(f"Created export index at {output_path}")
    return output_path