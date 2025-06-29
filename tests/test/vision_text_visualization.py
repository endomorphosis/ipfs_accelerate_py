#!/usr/bin/env python3

"""
Vision-Text Model Visualization

This script generates interactive visualizations for vision-text model (CLIP, BLIP) test results
stored in the DuckDB database. It provides performance comparisons, compatibility matrices,
and statistical analysis with interactive charts.

Features:
1. Hardware comparison visualizations across models
2. Model performance visualizations across hardware
3. Time series performance tracking
4. Compatibility heatmaps
5. Statistical analysis of performance metrics
6. Interactive dashboard generation
7. Export to various formats (HTML, PNG, PDF)

Dependencies:
- plotly
- pandas
- duckdb
- numpy
- scipy (for statistical analysis)

Usage:
  python vision_text_visualization.py --performance-comparison
  python vision_text_visualization.py --compatibility-heatmap
  python vision_text_visualization.py --time-series
  python vision_text_visualization.py --dashboard
  python vision_text_visualization.py --export-format html
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"vision_text_visualization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Path constants
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = ROOT_DIR / "reports"
VISUALIZATIONS_DIR = ROOT_DIR / "visualizations" / "vision_text"
DB_PATH = ROOT_DIR / "benchmark_db.duckdb"

# Ensure directories exist
REPORTS_DIR.mkdir(exist_ok=True)
VISUALIZATIONS_DIR.mkdir(exist_ok=True, parents=True)

# Hardware platforms display names
HARDWARE_DISPLAY_NAMES = {
    "cpu": "CPU",
    "cuda": "CUDA (NVIDIA)",
    "openvino": "OpenVINO (Intel)",
    "rocm": "ROCm (AMD)",
    "mps": "MPS (Apple)",
    "webnn": "WebNN",
    "webgpu": "WebGPU"
}

# Chart themes (matching existing system)
CHART_THEMES = {
    "light": {
        "background": "#ffffff",
        "text": "#333333",
        "grid": "#eeeeee",
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    },
    "dark": {
        "background": "#222222",
        "text": "#ffffff",
        "grid": "#444444",
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    }
}

def connect_to_db() -> Any:
    """Connect to DuckDB database."""
    try:
        import duckdb
        conn = duckdb.connect(str(DB_PATH))
        return conn
    except ImportError:
        logger.error("DuckDB not installed. Please install with: pip install duckdb")
        return None
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def get_performance_data(conn) -> Dict:
    """Retrieve performance data for vision-text models from database."""
    try:
        # Query performance data
        performance_data = conn.execute("""
        SELECT 
            vr.model_id, 
            vr.model_type,
            vr.hardware_platform, 
            AVG(vr.avg_inference_time) as avg_time,
            COUNT(*) as test_count,
            mc.task
        FROM vision_text_results vr
        JOIN model_compatibility mc ON vr.model_id = mc.model_id
        WHERE vr.success = TRUE
        GROUP BY vr.model_id, vr.model_type, vr.hardware_platform, mc.task
        ORDER BY vr.model_type, avg_time
        """).fetchdf()
        
        if performance_data.empty:
            logger.warning("No performance data found in the database")
            return {"success": False, "message": "No performance data found"}
        
        return {"success": True, "data": performance_data}
    except Exception as e:
        logger.error(f"Error retrieving performance data: {e}")
        return {"success": False, "message": str(e)}

def get_compatibility_data(conn) -> Dict:
    """Retrieve compatibility data for vision-text models from database."""
    try:
        # Query compatibility data
        compatibility_data = conn.execute("""
        SELECT 
            model_id, 
            model_family, 
            model_type, 
            task, 
            cpu, 
            cuda, 
            openvino,
            rocm,
            mps,
            webnn,
            webgpu,
            last_tested
        FROM model_compatibility
        WHERE model_type IN ('clip', 'blip')
        ORDER BY model_family, model_id
        """).fetchdf()
        
        if compatibility_data.empty:
            logger.warning("No compatibility data found in the database")
            return {"success": False, "message": "No compatibility data found"}
        
        return {"success": True, "data": compatibility_data}
    except Exception as e:
        logger.error(f"Error retrieving compatibility data: {e}")
        return {"success": False, "message": str(e)}

def get_time_series_data(conn) -> Dict:
    """Retrieve time series performance data for vision-text models."""
    try:
        # Query time series data
        time_series_data = conn.execute("""
        SELECT 
            model_id, 
            model_type,
            hardware_platform, 
            timestamp,
            avg_inference_time
        FROM vision_text_results
        WHERE success = TRUE
        ORDER BY model_id, hardware_platform, timestamp
        """).fetchdf()
        
        if time_series_data.empty:
            logger.warning("No time series data found in the database")
            return {"success": False, "message": "No time series data found"}
        
        # Convert timestamp string to datetime if needed
        if time_series_data['timestamp'].dtype == 'object':
            try:
                time_series_data['timestamp'] = pd.to_datetime(time_series_data['timestamp'])
            except Exception as e:
                logger.warning(f"Could not convert timestamps to datetime: {e}")
        
        return {"success": True, "data": time_series_data}
    except Exception as e:
        logger.error(f"Error retrieving time series data: {e}")
        return {"success": False, "message": str(e)}

def create_performance_comparison(data, theme="light", export_format=None) -> Tuple[bool, str]:
    """Create a performance comparison visualization."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
        
        # Group data by model type
        model_types = data["model_type"].unique()
        performance_figs = []
        
        for model_type in model_types:
            # Filter data for this model type
            model_data = data[data["model_type"] == model_type]
            
            # Get unique models and hardware platforms
            models = model_data["model_id"].unique()
            
            # Use a shorter model name for display
            model_data = model_data.copy()
            model_data["short_model_id"] = model_data["model_id"].apply(
                lambda x: x.split("/")[-1] if "/" in x else x
            )
            
            # Create a grouped bar chart
            fig = px.bar(
                model_data, 
                x="short_model_id",
                y="avg_time",
                color="hardware_platform",
                barmode="group",
                labels={
                    "short_model_id": "Model",
                    "avg_time": "Average Inference Time (s)",
                    "hardware_platform": "Hardware Platform"
                },
                title=f"{model_type.upper()} Model Performance Comparison",
                category_orders={"hardware_platform": sorted(model_data["hardware_platform"].unique())},
                color_discrete_map={
                    "cpu": "#1f77b4",
                    "cuda": "#ff7f0e",
                    "openvino": "#2ca02c",
                    "rocm": "#d62728",
                    "mps": "#9467bd",
                    "webnn": "#8c564b",
                    "webgpu": "#17becf"
                }
            )
            
            # Update layout for theme
            theme_settings = CHART_THEMES[theme]
            fig.update_layout(
                template="plotly_white" if theme == "light" else "plotly_dark",
                paper_bgcolor=theme_settings["background"],
                plot_bgcolor=theme_settings["background"],
                font=dict(color=theme_settings["text"]),
                legend_title_text="Hardware Platform",
                xaxis_title="Model",
                yaxis_title="Average Inference Time (seconds)",
                height=600,
                width=1000,
                margin=dict(t=50, b=100)
            )
            
            # Add hover template with more details
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                "Hardware: %{marker.color}<br>" +
                "Avg Inference Time: %{y:.4f} s<br>" +
                "Tests: %{customdata}"
            )
            
            # Add custom data for hover template
            for i, trace in enumerate(fig.data):
                fig.data[i].customdata = model_data[model_data["hardware_platform"] == trace.name]["test_count"]
            
            # Save the figure
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            html_file = VISUALIZATIONS_DIR / f"{model_type}_performance_comparison_{timestamp}.html"
            
            fig.write_html(str(html_file), include_plotlyjs="cdn")
            logger.info(f"Created {model_type} performance comparison: {html_file}")
            
            # Export in requested format if specified
            if export_format and export_format != "html":
                export_file = VISUALIZATIONS_DIR / f"{model_type}_performance_comparison_{timestamp}.{export_format}"
                fig.write_image(str(export_file))
                logger.info(f"Exported {model_type} performance comparison as {export_format}: {export_file}")
            
            performance_figs.append(html_file)
        
        return True, str(performance_figs)
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        logger.error("Please install required libraries: pip install plotly pandas")
        return False, f"Required library not installed: {e}"
    except Exception as e:
        logger.error(f"Error creating performance comparison: {e}")
        return False, str(e)

def create_compatibility_heatmap(data, theme="light", export_format=None) -> Tuple[bool, str]:
    """Create a compatibility heatmap visualization."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
        import numpy as np
        
        # Group data by model type
        model_types = data["model_type"].unique()
        heatmap_figs = []
        
        for model_type in model_types:
            # Filter data for this model type
            model_data = data[data["model_type"] == model_type]
            
            # Get unique models and hardware platforms
            models = model_data["model_id"].unique()
            
            # Use a shorter model name for display
            model_data = model_data.copy()
            model_data["short_model_id"] = model_data["model_id"].apply(
                lambda x: x.split("/")[-1] if "/" in x else x
            )
            
            # Prepare data for heatmap
            hardware_cols = ["cpu", "cuda", "openvino", "rocm", "mps", "webnn", "webgpu"]
            matrix_data = []
            
            for _, row in model_data.iterrows():
                for hw in hardware_cols:
                    matrix_data.append({
                        "Model": row["short_model_id"],
                        "Hardware": HARDWARE_DISPLAY_NAMES.get(hw, hw),
                        "Compatible": row[hw],
                        "Task": row["task"],
                        "Value": 1 if row[hw] else 0
                    })
            
            matrix_df = pd.DataFrame(matrix_data)
            
            # Create heatmap
            fig = px.imshow(
                pd.pivot_table(
                    matrix_df, 
                    values="Value", 
                    index="Model", 
                    columns="Hardware"
                ),
                labels=dict(x="Hardware Platform", y="Model", color="Compatible"),
                title=f"{model_type.upper()} Model Compatibility Matrix",
                color_continuous_scale=[[0, 'red'], [1, 'green']],
                zmin=0,
                zmax=1
            )
            
            # Update layout for theme
            theme_settings = CHART_THEMES[theme]
            fig.update_layout(
                template="plotly_white" if theme == "light" else "plotly_dark",
                paper_bgcolor=theme_settings["background"],
                plot_bgcolor=theme_settings["background"],
                font=dict(color=theme_settings["text"]),
                height=max(600, len(models) * 40),
                width=1000,
                xaxis_title="Hardware Platform",
                yaxis_title="Model",
                coloraxis_colorbar=dict(
                    title="Compatible",
                    tickvals=[0, 1],
                    ticktext=["No", "Yes"]
                ),
                margin=dict(t=50, b=100)
            )
            
            # Add custom hover template
            tasks = {model: task for model, task in zip(model_data["short_model_id"], model_data["task"])}
            
            hovertemplate = "<b>%{y}</b><br>" + \
                            "Hardware: %{x}<br>" + \
                            "Task: %{customdata}<br>" + \
                            "Compatible: %{z}<br>"
            
            fig.update_traces(
                customdata=[[tasks.get(y, "")] for y in fig.data[0].y],
                hovertemplate=hovertemplate
            )
            
            # Save the figure
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            html_file = VISUALIZATIONS_DIR / f"{model_type}_compatibility_heatmap_{timestamp}.html"
            
            fig.write_html(str(html_file), include_plotlyjs="cdn")
            logger.info(f"Created {model_type} compatibility heatmap: {html_file}")
            
            # Export in requested format if specified
            if export_format and export_format != "html":
                export_file = VISUALIZATIONS_DIR / f"{model_type}_compatibility_heatmap_{timestamp}.{export_format}"
                fig.write_image(str(export_file))
                logger.info(f"Exported {model_type} compatibility heatmap as {export_format}: {export_file}")
            
            heatmap_figs.append(html_file)
        
        return True, str(heatmap_figs)
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        logger.error("Please install required libraries: pip install plotly pandas")
        return False, f"Required library not installed: {e}"
    except Exception as e:
        logger.error(f"Error creating compatibility heatmap: {e}")
        return False, str(e)

def create_time_series_visualization(data, theme="light", export_format=None) -> Tuple[bool, str]:
    """Create a time series visualization for model performance over time."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
        
        # Group data by model type
        model_types = data["model_type"].unique()
        time_series_figs = []
        
        for model_type in model_types:
            # Filter data for this model type
            model_data = data[data["model_type"] == model_type]
            
            # Get unique models and hardware platforms
            models = model_data["model_id"].unique()
            
            # Create a line plot for each model
            for model in models:
                model_specific_data = model_data[model_data["model_id"] == model]
                
                # Use a shorter model name for display
                short_model_id = model.split("/")[-1] if "/" in model else model
                
                # Create figure
                fig = px.line(
                    model_specific_data, 
                    x="timestamp", 
                    y="avg_inference_time", 
                    color="hardware_platform",
                    labels={
                        "timestamp": "Date",
                        "avg_inference_time": "Average Inference Time (s)",
                        "hardware_platform": "Hardware Platform"
                    },
                    title=f"{short_model_id} Performance Over Time",
                    color_discrete_map={
                        "cpu": "#1f77b4",
                        "cuda": "#ff7f0e",
                        "openvino": "#2ca02c",
                        "rocm": "#d62728",
                        "mps": "#9467bd",
                        "webnn": "#8c564b",
                        "webgpu": "#17becf"
                    }
                )
                
                # Add markers
                fig.update_traces(mode="lines+markers", marker=dict(size=8))
                
                # Update layout for theme
                theme_settings = CHART_THEMES[theme]
                fig.update_layout(
                    template="plotly_white" if theme == "light" else "plotly_dark",
                    paper_bgcolor=theme_settings["background"],
                    plot_bgcolor=theme_settings["background"],
                    font=dict(color=theme_settings["text"]),
                    legend_title_text="Hardware Platform",
                    xaxis_title="Date",
                    yaxis_title="Average Inference Time (seconds)",
                    height=600,
                    width=1000,
                    margin=dict(t=50, b=100),
                    hovermode="x unified"
                )
                
                # Add hover template with more details
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>" +
                    "Hardware: %{marker.color}<br>" +
                    "Avg Inference Time: %{y:.4f} s<br>"
                )
                
                # Add range slider
                fig.update_xaxes(rangeslider_visible=True)
                
                # Save the figure
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                html_file = VISUALIZATIONS_DIR / f"{short_model_id}_time_series_{timestamp}.html"
                
                fig.write_html(str(html_file), include_plotlyjs="cdn")
                logger.info(f"Created {short_model_id} time series: {html_file}")
                
                # Export in requested format if specified
                if export_format and export_format != "html":
                    export_file = VISUALIZATIONS_DIR / f"{short_model_id}_time_series_{timestamp}.{export_format}"
                    fig.write_image(str(export_file))
                    logger.info(f"Exported {short_model_id} time series as {export_format}: {export_file}")
                
                time_series_figs.append(html_file)
        
        return True, str(time_series_figs)
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        logger.error("Please install required libraries: pip install plotly pandas")
        return False, f"Required library not installed: {e}"
    except Exception as e:
        logger.error(f"Error creating time series visualization: {e}")
        return False, str(e)

def create_interactive_dashboard(performance_data, compatibility_data, theme="light") -> Tuple[bool, str]:
    """Create an interactive dashboard combining multiple visualizations."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        
        # Prepare data
        perf_data = performance_data.copy()
        compat_data = compatibility_data.copy()
        
        # Use shorter model names
        perf_data["short_model_id"] = perf_data["model_id"].apply(
            lambda x: x.split("/")[-1] if "/" in x else x
        )
        compat_data["short_model_id"] = compat_data["model_id"].apply(
            lambda x: x.split("/")[-1] if "/" in x else x
        )
        
        # Create a 2x2 dashboard layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Performance Comparison",
                "Compatibility Matrix",
                "Hardware Distribution",
                "Model Type Distribution"
            ),
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "pie"}, {"type": "pie"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Performance comparison (top left)
        for hardware in sorted(perf_data["hardware_platform"].unique()):
            hw_data = perf_data[perf_data["hardware_platform"] == hardware]
            # Take only the first few models to avoid cluttering
            hw_data = hw_data.head(10)
            
            fig.add_trace(
                go.Bar(
                    x=hw_data["short_model_id"],
                    y=hw_data["avg_time"],
                    name=hardware,
                    legendgroup="hardware",
                    hovertemplate="<b>%{x}</b><br>" +
                    f"Hardware: {hardware}<br>" +
                    "Avg Inference Time: %{y:.4f} s<br>"
                ),
                row=1, col=1
            )
        
        # 2. Compatibility matrix (top right)
        # Prepare data for heatmap
        hardware_cols = ["cpu", "cuda", "openvino", "rocm", "mps", "webnn", "webgpu"]
        hardware_names = [HARDWARE_DISPLAY_NAMES.get(hw, hw) for hw in hardware_cols]
        
        # Take only the first few models to avoid cluttering
        compat_matrix_data = compat_data.head(10)
        z_data = compat_matrix_data[hardware_cols].values
        
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=hardware_names,
                y=compat_matrix_data["short_model_id"],
                colorscale=[[0, 'red'], [1, 'green']],
                showscale=False,
                zmin=0,
                zmax=1
            ),
            row=1, col=2
        )
        
        # 3. Hardware distribution (bottom left)
        hardware_counts = perf_data.groupby("hardware_platform").size().reset_index(name="count")
        
        fig.add_trace(
            go.Pie(
                labels=hardware_counts["hardware_platform"],
                values=hardware_counts["count"],
                name="Hardware",
                legendgroup="hardware_pie",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Model type distribution (bottom right)
        model_type_counts = compat_data.groupby("model_type").size().reset_index(name="count")
        
        fig.add_trace(
            go.Pie(
                labels=model_type_counts["model_type"],
                values=model_type_counts["count"],
                name="Model Type",
                legendgroup="model_type",
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout for theme
        theme_settings = CHART_THEMES[theme]
        fig.update_layout(
            template="plotly_white" if theme == "light" else "plotly_dark",
            paper_bgcolor=theme_settings["background"],
            plot_bgcolor=theme_settings["background"],
            font=dict(color=theme_settings["text"]),
            title="Vision-Text Models Dashboard",
            height=900,
            width=1200,
            showlegend=True,
            legend_title_text="Hardware Platform",
            margin=dict(t=50, b=50)
        )
        
        # Update x and y axis labels
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Avg Inference Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Hardware Platform", row=1, col=2)
        fig.update_yaxes(title_text="Model", row=1, col=2)
        
        # Save the dashboard
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = VISUALIZATIONS_DIR / f"vision_text_dashboard_{timestamp}.html"
        
        fig.write_html(str(dashboard_file), include_plotlyjs="cdn")
        logger.info(f"Created interactive dashboard: {dashboard_file}")
        
        return True, str(dashboard_file)
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        logger.error("Please install required libraries: pip install plotly pandas")
        return False, f"Required library not installed: {e}"
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {e}")
        return False, str(e)

def create_statistical_analysis(data, theme="light", export_format=None) -> Tuple[bool, str]:
    """Create statistical analysis visualizations with confidence intervals."""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        from scipy import stats
        
        # Group data by model and hardware
        model_hardware_groups = data.groupby(["model_id", "hardware_platform"])
        
        # Store analysis results
        analysis_results = []
        
        # For each model-hardware combination, calculate stats
        for (model, hardware), group in model_hardware_groups:
            # Only analyze groups with enough data points
            if len(group) < 2:
                continue
                
            # Calculate statistics
            mean = group["avg_inference_time"].mean()
            std = group["avg_inference_time"].std()
            n = len(group)
            
            # Calculate 95% confidence interval
            conf_interval = stats.t.interval(
                0.95, 
                n-1, 
                loc=mean, 
                scale=std/np.sqrt(n)
            )
            
            # Store results
            short_model_id = model.split("/")[-1] if "/" in model else model
            
            analysis_results.append({
                "model_id": model,
                "short_model_id": short_model_id,
                "hardware_platform": hardware,
                "mean": mean,
                "std": std,
                "conf_low": conf_interval[0],
                "conf_high": conf_interval[1],
                "sample_size": n,
                "model_type": group["model_type"].iloc[0]
            })
        
        if not analysis_results:
            logger.warning("Not enough data for statistical analysis")
            return False, "Not enough data for statistical analysis"
        
        # Convert to DataFrame
        analysis_df = pd.DataFrame(analysis_results)
        
        # Group by model type
        model_types = analysis_df["model_type"].unique()
        analysis_figs = []
        
        for model_type in model_types:
            # Filter data for this model type
            model_data = analysis_df[analysis_df["model_type"] == model_type]
            
            # Create figure
            fig = go.Figure()
            
            for hardware in sorted(model_data["hardware_platform"].unique()):
                hw_data = model_data[model_data["hardware_platform"] == hardware]
                
                # Plot mean values
                fig.add_trace(go.Bar(
                    x=hw_data["short_model_id"],
                    y=hw_data["mean"],
                    name=hardware,
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=hw_data["conf_high"] - hw_data["mean"],
                        arrayminus=hw_data["mean"] - hw_data["conf_low"],
                        visible=True
                    ),
                    hovertemplate="<b>%{x}</b><br>" +
                    f"Hardware: {hardware}<br>" +
                    "Avg Inference Time: %{y:.4f} s<br>" +
                    "95% CI: [%{error_y.arrayminus:.4f}, %{error_y.array:.4f}]<br>" +
                    "Sample Size: %{customdata}"
                ))
                
                # Add custom data for hover template
                fig.data[-1].customdata = hw_data["sample_size"]
            
            # Update layout for theme
            theme_settings = CHART_THEMES[theme]
            fig.update_layout(
                template="plotly_white" if theme == "light" else "plotly_dark",
                paper_bgcolor=theme_settings["background"],
                plot_bgcolor=theme_settings["background"],
                font=dict(color=theme_settings["text"]),
                title=f"{model_type.upper()} Model Performance with 95% Confidence Intervals",
                xaxis_title="Model",
                yaxis_title="Average Inference Time (seconds)",
                legend_title="Hardware Platform",
                height=600,
                width=1000,
                barmode='group'
            )
            
            # Save the figure
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            html_file = VISUALIZATIONS_DIR / f"{model_type}_statistical_analysis_{timestamp}.html"
            
            fig.write_html(str(html_file), include_plotlyjs="cdn")
            logger.info(f"Created {model_type} statistical analysis: {html_file}")
            
            # Export in requested format if specified
            if export_format and export_format != "html":
                export_file = VISUALIZATIONS_DIR / f"{model_type}_statistical_analysis_{timestamp}.{export_format}"
                fig.write_image(str(export_file))
                logger.info(f"Exported {model_type} statistical analysis as {export_format}: {export_file}")
            
            analysis_figs.append(html_file)
        
        return True, str(analysis_figs)
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        logger.error("Please install required libraries: pip install plotly pandas scipy numpy")
        return False, f"Required library not installed: {e}"
    except Exception as e:
        logger.error(f"Error creating statistical analysis: {e}")
        return False, str(e)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Vision-Text Model Visualization")
    
    # Visualization types
    viz_group = parser.add_mutually_exclusive_group(required=True)
    viz_group.add_argument("--performance-comparison", action="store_true", 
                         help="Create performance comparison visualization")
    viz_group.add_argument("--compatibility-heatmap", action="store_true",
                         help="Create compatibility heatmap visualization")
    viz_group.add_argument("--time-series", action="store_true",
                         help="Create time series visualization")
    viz_group.add_argument("--statistical-analysis", action="store_true",
                         help="Create statistical analysis with confidence intervals")
    viz_group.add_argument("--dashboard", action="store_true",
                         help="Create interactive dashboard")
    viz_group.add_argument("--all", action="store_true",
                         help="Create all visualization types")
    
    # Additional options
    parser.add_argument("--theme", choices=["light", "dark"], default="light",
                       help="Visualization theme")
    parser.add_argument("--export-format", choices=["html", "png", "svg", "pdf"],
                       default="html", help="Export format")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH),
                       help="Path to DuckDB database")
    parser.add_argument("--output-dir", type=str, default=str(VISUALIZATIONS_DIR),
                       help="Output directory for visualizations")
    parser.add_argument("--browser", action="store_true",
                       help="Open visualizations in browser")
    
    args = parser.parse_args()
    
    # Update paths if specified
    global DB_PATH, VISUALIZATIONS_DIR
    DB_PATH = Path(args.db_path)
    VISUALIZATIONS_DIR = Path(args.output_dir)
    VISUALIZATIONS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Connect to database
    conn = connect_to_db()
    if not conn:
        return 1
    
    try:
        # Import required libraries
        import pandas as pd
        
        # Get required data
        if args.performance_comparison or args.dashboard or args.statistical_analysis or args.all:
            performance_result = get_performance_data(conn)
            if not performance_result["success"]:
                logger.error(f"Failed to get performance data: {performance_result['message']}")
                return 1
            performance_data = performance_result["data"]
        
        if args.compatibility_heatmap or args.dashboard or args.all:
            compatibility_result = get_compatibility_data(conn)
            if not compatibility_result["success"]:
                logger.error(f"Failed to get compatibility data: {compatibility_result['message']}")
                return 1
            compatibility_data = compatibility_result["data"]
        
        if args.time_series or args.all:
            time_series_result = get_time_series_data(conn)
            if not time_series_result["success"]:
                logger.error(f"Failed to get time series data: {time_series_result['message']}")
                return 1
            time_series_data = time_series_result["data"]
        
        # Create requested visualizations
        if args.performance_comparison or args.all:
            success, message = create_performance_comparison(
                performance_data, args.theme, args.export_format
            )
            if not success:
                logger.error(f"Failed to create performance comparison: {message}")
        
        if args.compatibility_heatmap or args.all:
            success, message = create_compatibility_heatmap(
                compatibility_data, args.theme, args.export_format
            )
            if not success:
                logger.error(f"Failed to create compatibility heatmap: {message}")
        
        if args.time_series or args.all:
            success, message = create_time_series_visualization(
                time_series_data, args.theme, args.export_format
            )
            if not success:
                logger.error(f"Failed to create time series visualization: {message}")
        
        if args.statistical_analysis or args.all:
            success, message = create_statistical_analysis(
                performance_data, args.theme, args.export_format
            )
            if not success:
                logger.error(f"Failed to create statistical analysis: {message}")
        
        if args.dashboard or args.all:
            success, message = create_interactive_dashboard(
                performance_data, compatibility_data, args.theme
            )
            if not success:
                logger.error(f"Failed to create interactive dashboard: {message}")
        
        # Open browser if requested
        if args.browser:
            import webbrowser
            for file in VISUALIZATIONS_DIR.glob(f"*_{datetime.datetime.now().strftime('%Y%m%d')}*"):
                webbrowser.open(f"file://{file}")
        
        logger.info("Visualization creation complete.")
        return 0
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        logger.error("Please install required libraries: pip install plotly pandas numpy")
        return 1
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return 1
    finally:
        # Close database connection
        if conn:
            conn.close()

if __name__ == "__main__":
    sys.exit(main())