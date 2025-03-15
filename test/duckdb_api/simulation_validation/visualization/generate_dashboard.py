#!/usr/bin/env python3
"""
Dashboard Generator for Simulation Validation Framework

This module generates comprehensive interactive dashboards from validation results,
providing visualizations for simulation accuracy, performance metrics, hardware
comparisons, and drift detection.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_dashboard")

# Try importing visualization dependencies
try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.subplots as sp
    visualization_dependencies = True
except ImportError:
    logger.warning("Visualization dependencies not found. Install numpy, pandas, and plotly for full functionality.")
    visualization_dependencies = False

class DashboardGenerator:
    """
    Generates comprehensive dashboards from validation results.
    
    This class provides methods for creating interactive dashboards that visualize
    simulation accuracy, performance metrics, hardware comparisons, and drift detection.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        run_id: Optional[str] = None,
        interactive: bool = True,
        title: Optional[str] = None
    ):
        """
        Initialize the Dashboard Generator.
        
        Args:
            input_dir: Directory containing validation results
            output_dir: Directory to save generated dashboard
            run_id: Unique identifier for this run
            interactive: Whether to create interactive visualizations
            title: Title for the dashboard
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.run_id = run_id or datetime.now().strftime("%Y%m%d%H%M%S")
        self.interactive = interactive
        self.title = title or f"Simulation Validation Dashboard - {self.run_id}"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize validation results
        self.validation_results = None
        self.validation_items = None
        self.summary = None
        
        # Check if visualization dependencies are available
        if not visualization_dependencies:
            logger.error("Visualization dependencies not found. Install numpy, pandas, and plotly for full functionality.")
            raise ImportError("Visualization dependencies not found")
        
        logger.info(f"Dashboard Generator initialized with run ID: {self.run_id}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
    
    def load_validation_results(self) -> bool:
        """
        Load validation results from input directory.
        
        Returns:
            True if validation results were loaded successfully, False otherwise
        """
        validation_file = None
        
        # Try different filenames for validation results
        for filename in ["validation_results.json", "simulation_vs_hardware_results.json", "validation_summary.json", "results.json"]:
            path = os.path.join(self.input_dir, filename)
            if os.path.exists(path):
                validation_file = path
                break
        
        # If no file found with standard names, look for any JSON file with validation in the name
        if validation_file is None:
            for path in Path(self.input_dir).glob("*validation*.json"):
                validation_file = str(path)
                break
        
        # If still no file found, look for any JSON file
        if validation_file is None:
            for path in Path(self.input_dir).glob("*.json"):
                validation_file = str(path)
                break
        
        # If no validation file found, return False
        if validation_file is None:
            logger.error(f"No validation results found in {self.input_dir}")
            return False
        
        # Load validation results
        try:
            with open(validation_file, 'r') as f:
                self.validation_results = json.load(f)
            
            # Extract validation items and summary
            self.validation_items = self.validation_results.get("validation_items", [])
            self.summary = self.validation_results.get("summary", {})
            
            logger.info(f"Loaded validation results from {validation_file}")
            logger.info(f"Found {len(self.validation_items)} validation items")
            
            return True
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")
            return False
    
    def convert_to_dataframe(self) -> pd.DataFrame:
        """
        Convert validation items to a pandas DataFrame for easier analysis.
        
        Returns:
            DataFrame with validation data
        """
        if not self.validation_items:
            logger.warning("No validation items available")
            return pd.DataFrame()
        
        # Extract data from validation items
        data = []
        
        for item in self.validation_items:
            hardware_id = item.get("hardware_id", "unknown")
            model_id = item.get("model_id", "unknown")
            metrics_comparison = item.get("metrics_comparison", {})
            
            for metric_name, metric_data in metrics_comparison.items():
                if isinstance(metric_data, dict):
                    simulation_value = metric_data.get("simulation_value")
                    hardware_value = metric_data.get("hardware_value")
                    mape = metric_data.get("mape")
                    
                    if simulation_value is not None and hardware_value is not None:
                        data.append({
                            "hardware_id": hardware_id,
                            "model_id": model_id,
                            "metric_name": metric_name,
                            "simulation_value": simulation_value,
                            "hardware_value": hardware_value,
                            "mape": mape if mape is not None else 0
                        })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"Created DataFrame with {len(df)} rows")
        
        return df
    
    def create_dashboard(self) -> str:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Returns:
            Path to the generated dashboard HTML file
        """
        # Load validation results if not loaded already
        if self.validation_results is None:
            success = self.load_validation_results()
            if not success:
                logger.error("Failed to load validation results")
                return ""
        
        # Check if we have validation items
        if not self.validation_items:
            logger.error("No validation items available")
            return ""
        
        # Convert to DataFrame for analysis
        df = self.convert_to_dataframe()
        
        # Create output path
        output_path = os.path.join(self.output_dir, "validation_report.html")
        
        # Create dashboard sections
        dashboard_sections = []
        
        # Create summary section
        summary_html = self._create_summary_section()
        dashboard_sections.append(summary_html)
        
        # Create overall MAPE comparison chart
        try:
            mape_chart = self._create_mape_comparison_chart(df)
            dashboard_sections.append(mape_chart)
        except Exception as e:
            logger.error(f"Error creating MAPE comparison chart: {e}")
        
        # Create hardware comparison heatmap
        try:
            heatmap = self._create_hardware_comparison_heatmap(df)
            dashboard_sections.append(heatmap)
        except Exception as e:
            logger.error(f"Error creating hardware comparison heatmap: {e}")
        
        # Create metric details section
        try:
            metric_details = self._create_metric_details_section(df)
            dashboard_sections.append(metric_details)
        except Exception as e:
            logger.error(f"Error creating metric details section: {e}")
        
        # Create hardware profiles section
        try:
            hardware_profiles = self._create_hardware_profiles_section(df)
            dashboard_sections.append(hardware_profiles)
        except Exception as e:
            logger.error(f"Error creating hardware profiles section: {e}")
        
        # Create model profiles section
        try:
            model_profiles = self._create_model_profiles_section(df)
            dashboard_sections.append(model_profiles)
        except Exception as e:
            logger.error(f"Error creating model profiles section: {e}")
        
        # Create visualization gallery
        try:
            visualization_gallery = self._create_visualization_gallery()
            dashboard_sections.append(visualization_gallery)
        except Exception as e:
            logger.error(f"Error creating visualization gallery: {e}")
        
        # Create drift detection section
        try:
            drift_detection = self._create_drift_detection_section()
            dashboard_sections.append(drift_detection)
        except Exception as e:
            logger.error(f"Error creating drift detection section: {e}")
        
        # Combine all sections into a complete dashboard
        dashboard_html = self._create_dashboard_html(dashboard_sections)
        
        # Write dashboard to file
        try:
            with open(output_path, 'w') as f:
                f.write(dashboard_html)
            
            logger.info(f"Dashboard saved to {output_path}")
            
            # Create additional dashboard files
            self._create_additional_dashboards(df)
            
            return output_path
        except Exception as e:
            logger.error(f"Error writing dashboard to file: {e}")
            return ""
    
    def _create_summary_section(self) -> str:
        """
        Create a summary section for the dashboard.
        
        Returns:
            HTML string with summary section
        """
        # Extract summary data
        overall_mape = self.summary.get("overall_mape", 0)
        if isinstance(overall_mape, str):
            try:
                overall_mape = float(overall_mape)
            except ValueError:
                overall_mape = 0
        
        status = self.summary.get("status", "unknown")
        timestamp = self.summary.get("timestamp", datetime.now().isoformat())
        
        # Format timestamp if it's a string
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        
        # Calculate counts
        hardware_count = len(set(item.get("hardware_id", "unknown") for item in self.validation_items))
        model_count = len(set(item.get("model_id", "unknown") for item in self.validation_items))
        
        # Determine status color
        if status.lower() == "success":
            status_class = "text-success"
        elif status.lower() == "failure":
            status_class = "text-danger"
        else:
            status_class = "text-warning"
        
        # Determine MAPE color
        if overall_mape < 0.05:  # 5%
            mape_class = "text-success"
        elif overall_mape < 0.10:  # 10%
            mape_class = "text-warning"
        else:
            mape_class = "text-danger"
        
        # Create summary HTML
        summary_html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h3>Validation Summary</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <table class="table table-bordered">
                            <tr>
                                <th>Run ID</th>
                                <td>{self.run_id}</td>
                            </tr>
                            <tr>
                                <th>Timestamp</th>
                                <td>{timestamp}</td>
                            </tr>
                            <tr>
                                <th>Status</th>
                                <td class="{status_class}">{status.upper()}</td>
                            </tr>
                            <tr>
                                <th>Overall MAPE</th>
                                <td class="{mape_class}">{overall_mape:.2%}</td>
                            </tr>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <table class="table table-bordered">
                            <tr>
                                <th>Validation Items</th>
                                <td>{len(self.validation_items)}</td>
                            </tr>
                            <tr>
                                <th>Hardware Types</th>
                                <td>{hardware_count}</td>
                            </tr>
                            <tr>
                                <th>Model Types</th>
                                <td>{model_count}</td>
                            </tr>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return summary_html
    
    def _create_mape_comparison_chart(self, df: pd.DataFrame) -> str:
        """
        Create a MAPE comparison chart for the dashboard.
        
        Args:
            df: DataFrame with validation data
            
        Returns:
            HTML string with MAPE comparison chart
        """
        # Create output path for the chart
        output_path = os.path.join(self.output_dir, "mape_comparison.html")
        
        # Calculate average MAPE by hardware and model
        hardware_model_mape = df.groupby(['hardware_id', 'model_id'])['mape'].mean().reset_index()
        
        # Sort by MAPE (ascending) for better visualization
        hardware_model_mape = hardware_model_mape.sort_values('mape')
        
        # Create the chart
        fig = px.bar(
            hardware_model_mape,
            x='hardware_id',
            y='mape',
            color='model_id',
            barmode='group',
            labels={'mape': 'Mean Absolute Percentage Error (MAPE)', 'hardware_id': 'Hardware', 'model_id': 'Model'},
            title='MAPE Comparison by Hardware and Model'
        )
        
        # Update layout for better visualization
        fig.update_layout(
            legend_title="Model",
            font=dict(size=12),
            yaxis=dict(tickformat=".1%"),
            height=600
        )
        
        # Save the chart
        try:
            fig.write_html(output_path)
            logger.info(f"MAPE comparison chart saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving MAPE comparison chart: {e}")
        
        # Create HTML section
        chart_html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h3>MAPE Comparison by Hardware and Model</h3>
            </div>
            <div class="card-body">
                <div class="ratio ratio-16x9">
                    <iframe src="mape_comparison.html" allowfullscreen></iframe>
                </div>
            </div>
        </div>
        """
        
        return chart_html
    
    def _create_hardware_comparison_heatmap(self, df: pd.DataFrame) -> str:
        """
        Create a hardware comparison heatmap for the dashboard.
        
        Args:
            df: DataFrame with validation data
            
        Returns:
            HTML string with hardware comparison heatmap
        """
        # Create output path for the heatmap
        output_path = os.path.join(self.output_dir, "hardware_heatmap.html")
        
        # Pivot data to create a 2D matrix of hardware vs model with MAPE values
        pivot_df = df.pivot_table(
            values='mape',
            index='hardware_id',
            columns='model_id',
            aggfunc='mean'
        ).fillna(0)
        
        # Sort rows and columns for better visualization
        pivot_df = pivot_df.sort_index()
        pivot_df = pivot_df.sort_index(axis=1)
        
        # Create heatmap
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Model", y="Hardware", color="MAPE"),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale='RdYlGn_r',  # Red (high MAPE) to Green (low MAPE)
            title='Hardware-Model MAPE Heatmap'
        )
        
        # Update layout for better visualization
        fig.update_layout(
            font=dict(size=12),
            coloraxis_colorbar=dict(title="MAPE", tickformat=".1%"),
            height=600
        )
        
        # Add values as text annotations
        fig.update_traces(text=[[f"{value:.1%}" for value in row] for row in pivot_df.values],
                        texttemplate="%{text}")
        
        # Save the heatmap
        try:
            fig.write_html(output_path)
            logger.info(f"Hardware comparison heatmap saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving hardware comparison heatmap: {e}")
        
        # Create HTML section
        heatmap_html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h3>Hardware-Model MAPE Heatmap</h3>
            </div>
            <div class="card-body">
                <div class="ratio ratio-16x9">
                    <iframe src="hardware_heatmap.html" allowfullscreen></iframe>
                </div>
            </div>
        </div>
        """
        
        return heatmap_html
    
    def _create_metric_details_section(self, df: pd.DataFrame) -> str:
        """
        Create a metric details section for the dashboard.
        
        Args:
            df: DataFrame with validation data
            
        Returns:
            HTML string with metric details section
        """
        # Create output path for the metrics chart
        output_path = os.path.join(self.output_dir, "metrics_comparison.html")
        
        # Calculate average MAPE by metric
        metric_mape = df.groupby('metric_name')['mape'].mean().reset_index()
        
        # Sort by MAPE (ascending) for better visualization
        metric_mape = metric_mape.sort_values('mape')
        
        # Create the chart
        fig = px.bar(
            metric_mape,
            x='metric_name',
            y='mape',
            color='mape',
            color_continuous_scale='RdYlGn_r',  # Red (high MAPE) to Green (low MAPE)
            labels={'mape': 'Mean Absolute Percentage Error (MAPE)', 'metric_name': 'Metric'},
            title='MAPE Comparison by Metric'
        )
        
        # Update layout for better visualization
        fig.update_layout(
            font=dict(size=12),
            yaxis=dict(tickformat=".1%"),
            height=600
        )
        
        # Save the chart
        try:
            fig.write_html(output_path)
            logger.info(f"Metrics comparison chart saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving metrics comparison chart: {e}")
        
        # Create HTML section
        metrics_html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h3>Metric Performance Analysis</h3>
            </div>
            <div class="card-body">
                <div class="ratio ratio-16x9">
                    <iframe src="metrics_comparison.html" allowfullscreen></iframe>
                </div>
            </div>
        </div>
        """
        
        return metrics_html
    
    def _create_hardware_profiles_section(self, df: pd.DataFrame) -> str:
        """
        Create a hardware profiles section for the dashboard.
        
        Args:
            df: DataFrame with validation data
            
        Returns:
            HTML string with hardware profiles section
        """
        # Create output path for the hardware profiles chart
        output_path = os.path.join(self.output_dir, "hardware_profiles.html")
        
        # Get unique hardware IDs
        hardware_ids = df['hardware_id'].unique()
        
        # Create subplots for each hardware type
        fig = sp.make_subplots(
            rows=len(hardware_ids),
            cols=1,
            subplot_titles=[f"{hw_id} Performance" for hw_id in hardware_ids],
            vertical_spacing=0.1
        )
        
        # Add traces for each hardware type
        for i, hw_id in enumerate(hardware_ids, 1):
            # Filter data for this hardware type
            hw_data = df[df['hardware_id'] == hw_id]
            
            # Calculate average MAPE by metric for this hardware
            hw_metric_mape = hw_data.groupby('metric_name')['mape'].mean().reset_index()
            
            # Sort by MAPE (ascending) for better visualization
            hw_metric_mape = hw_metric_mape.sort_values('mape')
            
            # Add trace
            fig.add_trace(
                go.Bar(
                    x=hw_metric_mape['metric_name'],
                    y=hw_metric_mape['mape'],
                    text=[f"{mape:.1%}" for mape in hw_metric_mape['mape']],
                    name=hw_id,
                    marker_color=hw_metric_mape['mape'],
                    marker=dict(colorscale='RdYlGn_r', cmin=0, cmax=0.2)
                ),
                row=i,
                col=1
            )
            
            # Update yaxis for this subplot
            fig.update_yaxes(title_text="MAPE", tickformat=".1%", row=i, col=1)
        
        # Update layout for better visualization
        fig.update_layout(
            title="Hardware Profiles - Metric Performance",
            showlegend=False,
            font=dict(size=12),
            height=300 * len(hardware_ids)
        )
        
        # Save the chart
        try:
            fig.write_html(output_path)
            logger.info(f"Hardware profiles chart saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving hardware profiles chart: {e}")
        
        # Create HTML section
        hw_profiles_html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h3>Hardware Profiles</h3>
            </div>
            <div class="card-body">
                <div class="ratio ratio-16x9">
                    <iframe src="hardware_profiles.html" allowfullscreen></iframe>
                </div>
            </div>
        </div>
        """
        
        return hw_profiles_html
    
    def _create_model_profiles_section(self, df: pd.DataFrame) -> str:
        """
        Create a model profiles section for the dashboard.
        
        Args:
            df: DataFrame with validation data
            
        Returns:
            HTML string with model profiles section
        """
        # Create output path for the model profiles chart
        output_path = os.path.join(self.output_dir, "model_profiles.html")
        
        # Get unique model IDs
        model_ids = df['model_id'].unique()
        
        # Create subplots for each model type
        fig = sp.make_subplots(
            rows=len(model_ids),
            cols=1,
            subplot_titles=[f"{model_id} Performance" for model_id in model_ids],
            vertical_spacing=0.1
        )
        
        # Add traces for each model type
        for i, model_id in enumerate(model_ids, 1):
            # Filter data for this model type
            model_data = df[df['model_id'] == model_id]
            
            # Calculate average MAPE by hardware for this model
            model_hw_mape = model_data.groupby('hardware_id')['mape'].mean().reset_index()
            
            # Sort by MAPE (ascending) for better visualization
            model_hw_mape = model_hw_mape.sort_values('mape')
            
            # Add trace
            fig.add_trace(
                go.Bar(
                    x=model_hw_mape['hardware_id'],
                    y=model_hw_mape['mape'],
                    text=[f"{mape:.1%}" for mape in model_hw_mape['mape']],
                    name=model_id,
                    marker_color=model_hw_mape['mape'],
                    marker=dict(colorscale='RdYlGn_r', cmin=0, cmax=0.2)
                ),
                row=i,
                col=1
            )
            
            # Update yaxis for this subplot
            fig.update_yaxes(title_text="MAPE", tickformat=".1%", row=i, col=1)
        
        # Update layout for better visualization
        fig.update_layout(
            title="Model Profiles - Hardware Performance",
            showlegend=False,
            font=dict(size=12),
            height=300 * len(model_ids)
        )
        
        # Save the chart
        try:
            fig.write_html(output_path)
            logger.info(f"Model profiles chart saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving model profiles chart: {e}")
        
        # Create HTML section
        model_profiles_html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h3>Model Profiles</h3>
            </div>
            <div class="card-body">
                <div class="ratio ratio-16x9">
                    <iframe src="model_profiles.html" allowfullscreen></iframe>
                </div>
            </div>
        </div>
        """
        
        return model_profiles_html
    
    def _create_visualization_gallery(self) -> str:
        """
        Create a visualization gallery for the dashboard.
        
        Returns:
            HTML string with visualization gallery
        """
        # Create output directory for the gallery
        gallery_dir = os.path.join(self.output_dir, "gallery")
        os.makedirs(gallery_dir, exist_ok=True)
        
        # Create output path for the gallery HTML
        gallery_path = os.path.join(self.output_dir, "visualization_gallery.html")
        
        # Create gallery HTML content
        gallery_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Validation - Visualization Gallery</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding: 20px; }
                .gallery-item { margin-bottom: 30px; }
                .gallery-item img { width: 100%; border-radius: 5px; }
                .gallery-caption { margin-top: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Visualization Gallery</h1>
                <p>This gallery showcases various visualizations generated from the validation results.</p>
                
                <div class="row">
                    <div class="col-md-6 gallery-item">
                        <h3>MAPE Comparison</h3>
                        <div class="ratio ratio-16x9">
                            <iframe src="../mape_comparison.html" allowfullscreen></iframe>
                        </div>
                        <div class="gallery-caption">
                            Comparison of Mean Absolute Percentage Error (MAPE) across different hardware and model combinations.
                        </div>
                    </div>
                    
                    <div class="col-md-6 gallery-item">
                        <h3>Hardware Comparison Heatmap</h3>
                        <div class="ratio ratio-16x9">
                            <iframe src="../hardware_heatmap.html" allowfullscreen></iframe>
                        </div>
                        <div class="gallery-caption">
                            Heatmap visualization showing MAPE values across different hardware and model combinations.
                        </div>
                    </div>
                    
                    <div class="col-md-6 gallery-item">
                        <h3>Metrics Comparison</h3>
                        <div class="ratio ratio-16x9">
                            <iframe src="../metrics_comparison.html" allowfullscreen></iframe>
                        </div>
                        <div class="gallery-caption">
                            Comparison of MAPE values across different metrics, showing which metrics have higher simulation accuracy.
                        </div>
                    </div>
                    
                    <div class="col-md-6 gallery-item">
                        <h3>Hardware Profiles</h3>
                        <div class="ratio ratio-16x9">
                            <iframe src="../hardware_profiles.html" allowfullscreen></iframe>
                        </div>
                        <div class="gallery-caption">
                            Detailed profiles of each hardware type, showing metric-specific MAPE values.
                        </div>
                    </div>
                    
                    <div class="col-md-6 gallery-item">
                        <h3>Model Profiles</h3>
                        <div class="ratio ratio-16x9">
                            <iframe src="../model_profiles.html" allowfullscreen></iframe>
                        </div>
                        <div class="gallery-caption">
                            Detailed profiles of each model type, showing hardware-specific MAPE values.
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write gallery HTML to file
        try:
            with open(gallery_path, 'w') as f:
                f.write(gallery_html)
            
            logger.info(f"Visualization gallery saved to {gallery_path}")
        except Exception as e:
            logger.error(f"Error saving visualization gallery: {e}")
        
        # Create HTML section for the dashboard
        gallery_section_html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h3>Visualization Gallery</h3>
            </div>
            <div class="card-body">
                <p>Explore a gallery of interactive visualizations generated from the validation results.</p>
                <a href="visualization_gallery.html" class="btn btn-primary" target="_blank">Open Visualization Gallery</a>
            </div>
        </div>
        """
        
        return gallery_section_html
    
    def _create_drift_detection_section(self) -> str:
        """
        Create a drift detection section for the dashboard.
        
        Returns:
            HTML string with drift detection section
        """
        # Check if drift detection results are available
        drift_results = self.validation_results.get("drift_detection", {})
        
        if not drift_results:
            # If no drift results, create a placeholder section
            drift_html = f"""
            <div class="card mb-4">
                <div class="card-header">
                    <h3>Drift Detection</h3>
                </div>
                <div class="card-body">
                    <div class="alert alert-info">
                        <p>No drift detection results available for this validation run.</p>
                        <p>Drift detection analyzes changes in simulation accuracy over time, which requires historical validation data.</p>
                    </div>
                </div>
            </div>
            """
            
            return drift_html
        
        # Extract drift detection information
        is_significant = drift_results.get("is_significant", False)
        drift_metrics = drift_results.get("drift_metrics", {})
        
        # Create output path for the drift detection chart
        output_path = os.path.join(self.output_dir, "drift_detection_report.html")
        
        # Create HTML content for drift detection report
        if is_significant:
            status_class = "alert-danger"
            status_text = "Significant drift detected"
        else:
            status_class = "alert-success"
            status_text = "No significant drift detected"
        
        drift_report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Validation - Drift Detection Report</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ padding: 20px; }}
                .metric-card {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Drift Detection Report</h1>
                
                <div class="alert {status_class} mt-4">
                    <h4 class="alert-heading">{status_text}</h4>
                    <p>This report shows the results of drift detection analysis, which compares recent validation results with historical data to identify significant changes in simulation accuracy over time.</p>
                </div>
                
                <h2 class="mt-4">Drift Metrics</h2>
                <div class="row">
        """
        
        # Add metric-specific drift information
        for metric_name, metric_data in drift_metrics.items():
            drift_detected = metric_data.get("drift_detected", False)
            mean_change_pct = metric_data.get("mean_change_pct", 0)
            p_value = metric_data.get("p_value", 1.0)
            
            # Determine status class based on drift detection
            metric_status_class = "danger" if drift_detected else "success"
            
            drift_report_html += f"""
                    <div class="col-md-6">
                        <div class="card metric-card border-{metric_status_class}">
                            <div class="card-header bg-{metric_status_class} text-white">
                                {metric_name}
                            </div>
                            <div class="card-body">
                                <p><strong>Drift Detected:</strong> {drift_detected}</p>
                                <p><strong>Mean Change:</strong> {mean_change_pct:.2f}%</p>
                                <p><strong>P-Value:</strong> {p_value:.4f}</p>
                            </div>
                        </div>
                    </div>
            """
        
        drift_report_html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write drift detection report to file
        try:
            with open(output_path, 'w') as f:
                f.write(drift_report_html)
            
            logger.info(f"Drift detection report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving drift detection report: {e}")
        
        # Create HTML section for the dashboard
        drift_html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h3>Drift Detection</h3>
            </div>
            <div class="card-body">
                <div class="alert {status_class}">
                    <h4 class="alert-heading">{status_text}</h4>
                    <p>Drift detection analyzes changes in simulation accuracy over time.</p>
                </div>
                <a href="drift_detection_report.html" class="btn btn-primary" target="_blank">View Drift Detection Report</a>
            </div>
        </div>
        """
        
        return drift_html
    
    def _create_additional_dashboards(self, df: pd.DataFrame) -> None:
        """
        Create additional dashboard files for specific sections.
        
        Args:
            df: DataFrame with validation data
        """
        # Create performance analysis dashboard
        self._create_performance_analysis_dashboard(df)
        
        # Create calibration report if calibration data is available
        if "calibration" in self.validation_results:
            self._create_calibration_report()
    
    def _create_performance_analysis_dashboard(self, df: pd.DataFrame) -> None:
        """
        Create a performance analysis dashboard.
        
        Args:
            df: DataFrame with validation data
        """
        # Create output path for the performance analysis dashboard
        output_path = os.path.join(self.output_dir, "performance_analysis.html")
        
        # Calculate average values for simulation and hardware
        perf_data = []
        
        for item in self.validation_items:
            hardware_id = item.get("hardware_id", "unknown")
            model_id = item.get("model_id", "unknown")
            metrics_comparison = item.get("metrics_comparison", {})
            
            for metric_name, metric_data in metrics_comparison.items():
                if isinstance(metric_data, dict):
                    simulation_value = metric_data.get("simulation_value")
                    hardware_value = metric_data.get("hardware_value")
                    mape = metric_data.get("mape")
                    
                    if simulation_value is not None and hardware_value is not None:
                        # Special handling for latency metrics (lower is better)
                        is_latency = "latency" in metric_name.lower()
                        
                        perf_data.append({
                            "hardware_id": hardware_id,
                            "model_id": model_id,
                            "metric_name": metric_name,
                            "simulation_value": simulation_value,
                            "hardware_value": hardware_value,
                            "mape": mape if mape is not None else 0,
                            "relative_performance": hardware_value / simulation_value if not is_latency else simulation_value / hardware_value
                        })
        
        # Convert to DataFrame
        perf_df = pd.DataFrame(perf_data)
        
        # Create scatter plot comparing simulation vs hardware values
        scatter_fig = px.scatter(
            perf_df,
            x="simulation_value",
            y="hardware_value",
            color="metric_name",
            hover_name="model_id",
            hover_data=["hardware_id", "mape"],
            labels={"simulation_value": "Simulation Value", "hardware_value": "Hardware Value", "metric_name": "Metric"},
            title="Simulation vs Hardware Values"
        )
        
        # Add identity line (perfect prediction)
        scatter_fig.add_trace(
            go.Scatter(
                x=[perf_df["simulation_value"].min(), perf_df["simulation_value"].max()],
                y=[perf_df["simulation_value"].min(), perf_df["simulation_value"].max()],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="black", dash="dash")
            )
        )
        
        # Update layout for better visualization
        scatter_fig.update_layout(
            font=dict(size=12),
            height=600
        )
        
        # Calculate relative performance by hardware and model
        rel_perf = perf_df.groupby(['hardware_id', 'model_id'])['relative_performance'].mean().reset_index()
        
        # Create bar chart for relative performance
        rel_perf_fig = px.bar(
            rel_perf,
            x="hardware_id",
            y="relative_performance",
            color="model_id",
            barmode="group",
            labels={"relative_performance": "Relative Performance", "hardware_id": "Hardware", "model_id": "Model"},
            title="Relative Performance by Hardware and Model"
        )
        
        # Update layout for better visualization
        rel_perf_fig.update_layout(
            font=dict(size=12),
            height=600
        )
        
        # Create metrics comparison table
        metrics_table = perf_df.groupby("metric_name").agg({
            "mape": ["mean", "min", "max"],
            "relative_performance": ["mean", "min", "max"]
        }).reset_index()
        
        metrics_table.columns = ["metric_name", "mape_mean", "mape_min", "mape_max", "rel_perf_mean", "rel_perf_min", "rel_perf_max"]
        
        # Create HTML content for performance analysis dashboard
        perf_analysis_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Validation - Performance Analysis</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ padding: 20px; }}
                .viz-container {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Performance Analysis Dashboard</h1>
                <p>This dashboard provides detailed analysis of simulation vs hardware performance metrics.</p>
                
                <div class="viz-container">
                    <h2>Simulation vs Hardware Values</h2>
                    <div class="ratio ratio-16x9">
                        {scatter_fig.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
                
                <div class="viz-container">
                    <h2>Relative Performance by Hardware and Model</h2>
                    <div class="ratio ratio-16x9">
                        {rel_perf_fig.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
                
                <div class="viz-container">
                    <h2>Metrics Comparison</h2>
                    <table class="table table-bordered table-striped">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Mean MAPE</th>
                                <th>Min MAPE</th>
                                <th>Max MAPE</th>
                                <th>Mean Rel. Perf.</th>
                                <th>Min Rel. Perf.</th>
                                <th>Max Rel. Perf.</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Add rows for each metric
        for _, row in metrics_table.iterrows():
            perf_analysis_html += f"""
                            <tr>
                                <td>{row['metric_name']}</td>
                                <td>{row['mape_mean']:.2%}</td>
                                <td>{row['mape_min']:.2%}</td>
                                <td>{row['mape_max']:.2%}</td>
                                <td>{row['rel_perf_mean']:.2f}</td>
                                <td>{row['rel_perf_min']:.2f}</td>
                                <td>{row['rel_perf_max']:.2f}</td>
                            </tr>
            """
        
        perf_analysis_html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write performance analysis dashboard to file
        try:
            with open(output_path, 'w') as f:
                f.write(perf_analysis_html)
            
            logger.info(f"Performance analysis dashboard saved to {output_path}")
            
            # Create output path for the metrics comparison dashboard
            metrics_path = os.path.join(self.output_dir, "metrics_comparison.html")
            
            # Create a standalone metrics comparison chart
            metrics_fig = px.bar(
                metrics_table,
                x="metric_name",
                y="mape_mean",
                error_y=metrics_table["mape_max"] - metrics_table["mape_mean"],
                labels={"metric_name": "Metric", "mape_mean": "Mean MAPE"},
                title="Metrics Comparison - MAPE"
            )
            
            # Update layout for better visualization
            metrics_fig.update_layout(
                font=dict(size=12),
                yaxis=dict(tickformat=".1%"),
                height=600
            )
            
            # Save the metrics comparison chart
            metrics_fig.write_html(metrics_path)
            logger.info(f"Metrics comparison chart saved to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error saving performance analysis dashboard: {e}")
    
    def _create_calibration_report(self) -> None:
        """
        Create a calibration report if calibration data is available.
        """
        # Create output path for the calibration report
        output_path = os.path.join(self.output_dir, "calibration_report.html")
        
        # Extract calibration information
        calibration_data = self.validation_results.get("calibration", {})
        
        if not calibration_data:
            # If no calibration data, create a placeholder report
            calibration_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Simulation Validation - Calibration Report</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body {{ padding: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Calibration Report</h1>
                    
                    <div class="alert alert-info mt-4">
                        <p>No calibration data available for this validation run.</p>
                        <p>Calibration improves simulation accuracy by adjusting simulation parameters based on validation results.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Write calibration report to file
            try:
                with open(output_path, 'w') as f:
                    f.write(calibration_html)
                
                logger.info(f"Calibration report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving calibration report: {e}")
            
            return
        
        # Extract calibration metrics
        overall_improvement = calibration_data.get("overall_improvement", 0)
        before_calibration = calibration_data.get("before_calibration", {})
        after_calibration = calibration_data.get("after_calibration", {})
        
        # Create HTML content for calibration report
        calibration_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Validation - Calibration Report</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ padding: 20px; }}
                .summary-card {{ margin-bottom: 30px; }}
                .improvement-card {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Calibration Report</h1>
                <p>This report shows the effectiveness of simulation calibration in improving simulation accuracy.</p>
                
                <div class="card summary-card">
                    <div class="card-header">
                        <h2>Calibration Summary</h2>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-primary">
                            <h4 class="alert-heading">Overall Improvement: {overall_improvement:.2f}%</h4>
                            <p>The calibration process improved simulation accuracy by {overall_improvement:.2f}%.</p>
                        </div>
                    </div>
                </div>
                
                <h2>Metric-Specific Improvements</h2>
                <div class="row">
        """
        
        # Add metric-specific improvement information
        for metric_name, metric_data in calibration_data.get("metric_improvements", {}).items():
            improvement_pct = metric_data.get("improvement_percentage", 0)
            before_mape = metric_data.get("before_mape", 0)
            after_mape = metric_data.get("after_mape", 0)
            
            # Determine improvement class based on percentage
            if improvement_pct > 20:
                improvement_class = "success"
            elif improvement_pct > 10:
                improvement_class = "primary"
            elif improvement_pct > 0:
                improvement_class = "info"
            else:
                improvement_class = "warning"
            
            calibration_html += f"""
                    <div class="col-md-6">
                        <div class="card improvement-card border-{improvement_class}">
                            <div class="card-header bg-{improvement_class} text-white">
                                {metric_name}
                            </div>
                            <div class="card-body">
                                <h5 class="card-title">Improvement: {improvement_pct:.2f}%</h5>
                                <p class="card-text">MAPE Before: {before_mape:.2%}</p>
                                <p class="card-text">MAPE After: {after_mape:.2%}</p>
                            </div>
                        </div>
                    </div>
            """
        
        calibration_html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write calibration report to file
        try:
            with open(output_path, 'w') as f:
                f.write(calibration_html)
            
            logger.info(f"Calibration report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving calibration report: {e}")
    
    def _create_dashboard_html(self, sections: List[str]) -> str:
        """
        Create the complete dashboard HTML from individual sections.
        
        Args:
            sections: List of HTML strings for dashboard sections
            
        Returns:
            Complete dashboard HTML
        """
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
            <style>
                body {{ padding-top: 60px; padding-bottom: 20px; }}
                .sidebar {{ position: fixed; top: 60px; bottom: 0; left: 0; z-index: 1000; padding: 20px; overflow-x: hidden; overflow-y: auto; background-color: #f8f9fa; }}
                .main-content {{ margin-left: 260px; }}
                .section {{ margin-bottom: 30px; }}
                @media (max-width: 768px) {{ 
                    .sidebar {{ position: static; height: auto; }} 
                    .main-content {{ margin-left: 0; }}
                }}
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-md navbar-dark bg-dark fixed-top">
                <div class="container-fluid">
                    <a class="navbar-brand" href="#">Simulation Validation Dashboard</a>
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarCollapse" aria-controls="navbarCollapse" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarCollapse">
                        <ul class="navbar-nav me-auto mb-2 mb-md-0">
                            <li class="nav-item">
                                <a class="nav-link active" href="validation_report.html">Dashboard</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="visualization_gallery.html">Gallery</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="performance_analysis.html">Performance</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="metrics_comparison.html">Metrics</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="drift_detection_report.html">Drift</a>
                            </li>
                        </ul>
                    </div>
                </div>
            </nav>

            <div class="container-fluid">
                <div class="row">
                    <nav id="sidebar" class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse">
                        <div class="position-sticky pt-3">
                            <ul class="nav flex-column">
                                <li class="nav-item">
                                    <a class="nav-link active" href="#summary">
                                        Summary
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="#mape-comparison">
                                        MAPE Comparison
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="#hardware-heatmap">
                                        Hardware Heatmap
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="#metric-details">
                                        Metric Details
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="#hardware-profiles">
                                        Hardware Profiles
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="#model-profiles">
                                        Model Profiles
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="#visualization-gallery">
                                        Visualization Gallery
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="#drift-detection">
                                        Drift Detection
                                    </a>
                                </li>
                            </ul>
                        </div>
                    </nav>

                    <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4 main-content">
                        <div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
                            <h1>{self.title}</h1>
                            <div class="btn-toolbar mb-2 mb-md-0">
                                <div class="btn-group me-2">
                                    <a href="performance_analysis.html" class="btn btn-sm btn-outline-secondary">Performance Analysis</a>
                                    <a href="visualization_gallery.html" class="btn btn-sm btn-outline-secondary">Gallery</a>
                                </div>
                            </div>
                        </div>

                        <div id="summary" class="section">
                            {sections[0] if len(sections) > 0 else ""}
                        </div>

                        <div id="mape-comparison" class="section">
                            {sections[1] if len(sections) > 1 else ""}
                        </div>

                        <div id="hardware-heatmap" class="section">
                            {sections[2] if len(sections) > 2 else ""}
                        </div>

                        <div id="metric-details" class="section">
                            {sections[3] if len(sections) > 3 else ""}
                        </div>

                        <div id="hardware-profiles" class="section">
                            {sections[4] if len(sections) > 4 else ""}
                        </div>

                        <div id="model-profiles" class="section">
                            {sections[5] if len(sections) > 5 else ""}
                        </div>

                        <div id="visualization-gallery" class="section">
                            {sections[6] if len(sections) > 6 else ""}
                        </div>

                        <div id="drift-detection" class="section">
                            {sections[7] if len(sections) > 7 else ""}
                        </div>
                    </main>
                </div>
            </div>
        </body>
        </html>
        """
        
        return dashboard_html


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate comprehensive dashboard from validation results")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing validation results")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated dashboard")
    parser.add_argument("--run-id", type=str, help="Unique identifier for this run")
    parser.add_argument("--interactive", action="store_true", help="Create interactive visualizations")
    parser.add_argument("--title", type=str, help="Title for the dashboard")
    
    args = parser.parse_args()
    
    # Create dashboard generator
    generator = DashboardGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        interactive=args.interactive,
        title=args.title
    )
    
    # Create dashboard
    dashboard_path = generator.create_dashboard()
    
    if dashboard_path:
        print(f"Dashboard successfully generated at: {dashboard_path}")
        return 0
    else:
        print("Failed to generate dashboard")
        return 1


if __name__ == "__main__":
    sys.exit(main())