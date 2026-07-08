#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Dashboard Generator

Creates a comprehensive visualization dashboard for test results, hardware compatibility,
and performance metrics using Plotly and Dash.

Features:
- Interactive visualizations of test results
- Hardware compatibility matrices
- Performance metrics comparison
- Real-time test monitoring capabilities
- Model architecture analysis
- Test history and trends
"""

import os
import sys
import json
import glob
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Static HTML charts will not be generated.")

try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output, State
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: Dash not available. Interactive dashboard will not be generated.")

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("Warning: DuckDB not available. Database integration will be limited.")

# Model Architecture Categories
ARCHITECTURE_CATEGORIES = {
    "encoder-only": ["bert", "roberta", "albert", "distilbert", "electra", "mpnet"],
    "decoder-only": ["gpt2", "llama", "gpt_neo", "gpt_neox", "gptj", "bloom", "opt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "mt5", "longt5"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "resnet"],
    "multimodal": ["clip", "blip", "llava", "paligemma", "flava", "blip2"],
    "audio": ["whisper", "wav2vec2", "hubert", "speecht5", "seamless"]
}

# Hardware Platform Colors
HARDWARE_COLORS = {
    "cpu": "#1f77b4",  # blue
    "cuda": "#ff7f0e",  # orange
    "mps": "#2ca02c",   # green
    "openvino": "#d62728",  # red
    "webnn": "#9467bd",  # purple
    "webgpu": "#8c564b",  # brown
}

# Status Icons
STATUS_ICONS = {
    "success": "✅",
    "failure": "❌",
    "mock": "🔷",
    "real": "🚀"
}

def load_test_results(results_dir="collected_results", days=7):
    """
    Load test results from JSON files in the specified directory
    """
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return pd.DataFrame()
    
    # Load each file into a list of records
    all_results = []
    current_time = datetime.now().timestamp()
    cutoff_time = current_time - (days * 24 * 60 * 60)  # Convert days to seconds
    
    for json_file in json_files:
        try:
            # Check file modification time
            file_time = os.path.getmtime(json_file)
            if file_time < cutoff_time:
                continue  # Skip files older than cutoff
                
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Process data based on file format
            if isinstance(data, dict):
                # Transform data to our standard format if needed
                if "model_name" not in data and "model_id" in data:
                    data["model_name"] = data["model_id"]
                
                # Add file metadata
                data["filename"] = os.path.basename(json_file)
                data["timestamp"] = file_time
                
                # Add mock vs real inference info if not present
                if "using_real_inference" not in data:
                    has_torch = data.get("has_torch", False)
                    has_transformers = data.get("has_transformers", False)
                    data["using_real_inference"] = has_torch and has_transformers
                    data["using_mocks"] = not data["using_real_inference"]
                
                all_results.append(data)
            elif isinstance(data, list):
                # Handle list format (e.g., from distributed testing)
                for item in data:
                    if isinstance(item, dict):
                        item["filename"] = os.path.basename(json_file)
                        item["timestamp"] = file_time
                        
                        # Add mock vs real inference info if not present
                        if "using_real_inference" not in item:
                            has_torch = item.get("has_torch", False)
                            has_transformers = item.get("has_transformers", False)
                            item["using_real_inference"] = has_torch and has_transformers
                            item["using_mocks"] = not item["using_real_inference"]
                            
                        all_results.append(item)
                    
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not all_results:
        print(f"No valid test results found in {results_dir}")
        return pd.DataFrame()
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_results)
    
    # Add formatted timestamp
    if "timestamp" in df.columns:
        df["timestamp_str"] = df["timestamp"].apply(
            lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
        )
    
    # Add model architecture category
    df["architecture"] = "unknown"
    
    for arch, models in ARCHITECTURE_CATEGORIES.items():
        for model in models:
            # Check model name (could be in several formats)
            for col in ["model_name", "model_id", "model"]:
                if col in df.columns:
                    mask = df[col].astype(str).str.contains(model, case=False, na=False)
                    df.loc[mask, "architecture"] = arch
    
    # Ensure hardware column exists
    if "hardware" not in df.columns:
        df["hardware"] = "cpu"  # Default to CPU
    
    return df

def load_hardware_results(hardware_db="hardware_compatibility_matrix.duckdb"):
    """
    Load hardware compatibility results from DuckDB database
    """
    if not DUCKDB_AVAILABLE:
        print("DuckDB not available, loading hardware results from JSON files")
        
        # Try to find hardware results in compatibility_reports directory
        reports_dir = "compatibility_reports"
        if os.path.exists(reports_dir):
            json_path = os.path.join(reports_dir, "hardware_compatibility_results.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    return pd.DataFrame(data)
                except Exception as e:
                    print(f"Error loading hardware results: {e}")
        
        return pd.DataFrame()
    
    if not os.path.exists(hardware_db):
        print(f"Hardware database not found: {hardware_db}")
        return pd.DataFrame()
    
    try:
        # Connect to DuckDB
        conn = duckdb.connect(hardware_db)
        
        # Query hardware results
        df_results = conn.execute("""
            SELECT 
                model_id,
                model_type,
                hardware,
                success,
                load_time,
                inference_time,
                memory_usage,
                error,
                timestamp,
                output_shape
            FROM hardware_results
            ORDER BY timestamp DESC
        """).fetchdf()
        
        # Query hardware detection
        df_detection = conn.execute("""
            SELECT 
                hardware_type,
                available,
                name,
                features,
                timestamp
            FROM hardware_detection
            ORDER BY timestamp DESC
        """).fetchdf()
        
        conn.close()
        
        return df_results, df_detection
    
    except Exception as e:
        print(f"Error querying hardware database: {e}")
        return pd.DataFrame(), pd.DataFrame()

def load_distributed_results(dist_dir="distributed_results", days=7):
    """
    Load distributed testing results
    """
    if not os.path.exists(dist_dir):
        print(f"Distributed results directory not found: {dist_dir}")
        return pd.DataFrame()
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(dist_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {dist_dir}")
        return pd.DataFrame()
    
    all_results = []
    worker_results = []
    current_time = datetime.now().timestamp()
    cutoff_time = current_time - (days * 24 * 60 * 60)
    
    for json_file in json_files:
        try:
            # Check file modification time
            file_time = os.path.getmtime(json_file)
            if file_time < cutoff_time:
                continue
                
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Add file metadata
            data["filename"] = os.path.basename(json_file)
            data["timestamp"] = file_time
            all_results.append(data)
            
            # Extract worker-specific results
            if "results" in data:
                for result in data["results"]:
                    result["test_name"] = data.get("test_name", "unknown")
                    result["timestamp"] = file_time
                    worker_results.append(result)
        
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not all_results:
        print(f"No valid distributed results found in {dist_dir}")
        return pd.DataFrame(), pd.DataFrame()
    
    df_tests = pd.DataFrame(all_results)
    df_workers = pd.DataFrame(worker_results) if worker_results else pd.DataFrame()
    
    # Add formatted timestamp
    if "timestamp" in df_tests.columns:
        df_tests["timestamp_str"] = df_tests["timestamp"].apply(
            lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
        )
    
    if "timestamp" in df_workers.columns:
        df_workers["timestamp_str"] = df_workers["timestamp"].apply(
            lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
        )
    
    return df_tests, df_workers

def create_model_coverage_chart(df):
    """
    Create model coverage chart by architecture
    """
    if not PLOTLY_AVAILABLE or df.empty:
        return None
    
    # Count tests by architecture
    arch_counts = df.groupby("architecture").size().reset_index(name="count")
    
    # Create pie chart
    fig = px.pie(
        arch_counts, 
        values="count", 
        names="architecture",
        title="Model Tests by Architecture",
        color="architecture",
        color_discrete_map={
            "encoder-only": "#1f77b4",
            "decoder-only": "#ff7f0e",
            "encoder-decoder": "#2ca02c",
            "vision": "#d62728",
            "multimodal": "#9467bd",
            "audio": "#8c564b",
            "unknown": "#7f7f7f"
        }
    )
    
    fig.update_traces(textinfo="value+percent", hole=0.3)
    
    return fig

def create_hardware_comparison_chart(df):
    """
    Create hardware comparison chart for inference time
    """
    if not PLOTLY_AVAILABLE or df.empty:
        return None
    
    # Get successful tests only
    df_success = df[df.get("success", True) == True].copy()
    
    if "hardware" not in df_success.columns or "inference_time" not in df_success.columns:
        return None
    
    # Aggregate data by hardware
    hw_stats = df_success.groupby("hardware")["inference_time"].agg(
        ["mean", "median", "min", "max", "count"]
    ).reset_index()
    
    # Sort by median time
    hw_stats = hw_stats.sort_values("median")
    
    # Create bar chart
    fig = go.Figure()
    
    for i, row in hw_stats.iterrows():
        hw = row["hardware"]
        fig.add_trace(go.Bar(
            name=hw,
            x=[hw],
            y=[row["median"]],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["max"] - row["median"]],
                arrayminus=[row["median"] - row["min"]]
            ),
            text=f"Count: {row['count']}",
            marker_color=HARDWARE_COLORS.get(hw, "#7f7f7f")
        ))
    
    fig.update_layout(
        title="Inference Time by Hardware Platform",
        xaxis_title="Hardware Platform",
        yaxis_title="Inference Time (seconds)",
        barmode="group"
    )
    
    return fig

def create_test_success_chart(df):
    """
    Create test success rate chart
    """
    if not PLOTLY_AVAILABLE or df.empty:
        return None
    
    # Calculate success rate by architecture
    success_data = []
    
    for arch in df["architecture"].unique():
        arch_df = df[df["architecture"] == arch]
        total = len(arch_df)
        
        if "success" in df.columns:
            success = arch_df["success"].sum()
        else:
            # Assume success if no column
            success = total
        
        mock_count = arch_df.get("using_mocks", pd.Series([False] * len(arch_df))).sum()
        real_count = total - mock_count
        
        success_data.append({
            "architecture": arch,
            "success": success,
            "failure": total - success,
            "mock": mock_count,
            "real": real_count,
            "success_rate": success / total if total > 0 else 0,
            "total": total
        })
    
    success_df = pd.DataFrame(success_data)
    
    # Create stacked bar chart
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=("Success vs Failure", "Mock vs Real Inference")
    )
    
    for i, row in success_df.iterrows():
        fig.add_trace(
            go.Bar(
                name="Success",
                x=[row["architecture"]],
                y=[row["success"]],
                marker_color="green",
                customdata=[row["total"]],
                hovertemplate="%{y} successful out of %{customdata} tests<br>(%{text})",
                text=[f"{row['success_rate']:.1%}"],
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name="Failure",
                x=[row["architecture"]],
                y=[row["failure"]],
                marker_color="red",
                customdata=[row["total"]],
                hovertemplate="%{y} failed out of %{customdata} tests<br>(%{text})",
                text=[f"{(row['failure']/row['total']):.1%}" if row["total"] > 0 else "0%"],
            ),
            row=1, col=1
        )
    
    # Add mock vs real inference pie chart
    mock_total = success_df["mock"].sum()
    real_total = success_df["real"].sum()
    
    fig.add_trace(
        go.Pie(
            labels=["Mock Objects", "Real Inference"],
            values=[mock_total, real_total],
            marker_colors=["royalblue", "gold"],
            textinfo="label+percent",
            hole=0.3,
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Test Success Rate and Inference Type",
        barmode="stack",
        height=500
    )
    
    return fig

def create_performance_trend_chart(df):
    """
    Create performance trend chart over time
    """
    if not PLOTLY_AVAILABLE or df.empty or "timestamp" not in df.columns:
        return None
    
    # Filter to successful tests with inference time
    if "inference_time" not in df.columns:
        return None
    
    df_trend = df.copy()
    df_trend = df_trend.sort_values("timestamp")
    
    # Group by date and architecture
    df_trend["date"] = pd.to_datetime(df_trend["timestamp"], unit="s").dt.date
    
    # Calculate daily averages by architecture
    avg_by_date = df_trend.groupby(["date", "architecture"])["inference_time"].mean().reset_index()
    
    # Create line chart
    fig = px.line(
        avg_by_date,
        x="date",
        y="inference_time",
        color="architecture",
        title="Inference Time Trend by Architecture",
        labels={"inference_time": "Average Inference Time (s)", "date": "Date"}
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_hardware_matrix_heatmap(hardware_df):
    """
    Create hardware compatibility matrix heatmap
    """
    if not PLOTLY_AVAILABLE or hardware_df.empty:
        return None
    
    # Create pivot table for model type vs hardware
    if "model_type" not in hardware_df.columns or "hardware" not in hardware_df.columns:
        return None
    
    # Calculate success rate for each model type and hardware combination
    success_rate = hardware_df.groupby(["model_type", "hardware"])["success"].mean().reset_index()
    pivot_df = success_rate.pivot(index="model_type", columns="hardware", values="success")
    
    # Fill NaN with 0
    pivot_df = pivot_df.fillna(0)
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Hardware Platform", y="Model Architecture", color="Success Rate"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="RdYlGn",
        title="Hardware Compatibility Matrix (Success Rate)",
        zmin=0,
        zmax=1,
        aspect="auto"
    )
    
    fig.update_layout(
        xaxis=dict(side="bottom"),
        coloraxis_colorbar=dict(
            title="Success Rate",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        )
    )
    
    # Add values as text
    for i, row in enumerate(pivot_df.index):
        for j, col in enumerate(pivot_df.columns):
            value = pivot_df.iloc[i, j]
            fig.add_annotation(
                x=col,
                y=row,
                text=f"{value:.0%}",
                showarrow=False,
                font=dict(
                    color="white" if value < 0.5 else "black",
                    size=10
                )
            )
    
    return fig

def create_performance_comparison_chart(hardware_df):
    """
    Create performance comparison chart across hardware platforms
    """
    if not PLOTLY_AVAILABLE or hardware_df.empty:
        return None
    
    # Filter successful tests with inference time
    if "success" not in hardware_df.columns or "inference_time" not in hardware_df.columns:
        return None
    
    df_perf = hardware_df[hardware_df["success"] == True].copy()
    
    if df_perf.empty:
        return None
    
    # Group by model type and hardware
    perf_stats = df_perf.groupby(["model_type", "hardware"])["inference_time"].median().reset_index()
    
    # Create grouped bar chart
    fig = px.bar(
        perf_stats,
        x="model_type",
        y="inference_time",
        color="hardware",
        barmode="group",
        color_discrete_map=HARDWARE_COLORS,
        title="Median Inference Time by Model Type and Hardware",
        labels={"inference_time": "Median Inference Time (s)", "model_type": "Model Architecture", "hardware": "Hardware Platform"}
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_memory_usage_chart(hardware_df):
    """
    Create memory usage chart across model types
    """
    if not PLOTLY_AVAILABLE or hardware_df.empty:
        return None
    
    # Filter successful tests with memory usage
    if "success" not in hardware_df.columns or "memory_usage" not in hardware_df.columns:
        return None
    
    df_mem = hardware_df[hardware_df["success"] == True].copy()
    
    if df_mem.empty:
        return None
    
    # Group by model type
    mem_stats = df_mem.groupby(["model_type", "hardware"])["memory_usage"].median().reset_index()
    
    # Sort by median memory usage
    mem_stats = mem_stats.sort_values(["model_type", "memory_usage"], ascending=[True, False])
    
    # Create grouped bar chart
    fig = px.bar(
        mem_stats,
        x="model_type",
        y="memory_usage",
        color="hardware",
        barmode="group",
        color_discrete_map=HARDWARE_COLORS,
        title="Median Memory Usage by Model Type and Hardware",
        labels={"memory_usage": "Median Memory Usage (MB)", "model_type": "Model Architecture", "hardware": "Hardware Platform"}
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_dashboard_data(results_dir="collected_results", dist_dir="distributed_results", hardware_db="hardware_compatibility_matrix.duckdb", days=30):
    """
    Create all dashboard data
    """
    # Load data
    print("Loading test results...")
    test_df = load_test_results(results_dir, days)
    
    print("Loading hardware results...")
    if DUCKDB_AVAILABLE and os.path.exists(hardware_db):
        hardware_results, hardware_detection = load_hardware_results(hardware_db)
    else:
        hardware_results = pd.DataFrame()
        hardware_detection = pd.DataFrame()
    
    print("Loading distributed results...")
    dist_tests, dist_workers = load_distributed_results(dist_dir, days)
    
    # Check if we have any data
    if test_df.empty and hardware_results.empty and dist_tests.empty:
        print("No data available to create dashboard")
        return None
    
    # Create directory for dashboard data
    dashboard_dir = "dashboard_data"
    os.makedirs(dashboard_dir, exist_ok=True)
    
    data = {
        "test_results": test_df.to_dict("records") if not test_df.empty else [],
        "hardware_results": hardware_results.to_dict("records") if not hardware_results.empty else [],
        "hardware_detection": hardware_detection.to_dict("records") if not hardware_detection.empty else [],
        "distributed_tests": dist_tests.to_dict("records") if not dist_tests.empty else [],
        "distributed_workers": dist_workers.to_dict("records") if not dist_workers.empty else [],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_count": len(test_df) if not test_df.empty else 0,
            "hardware_count": len(hardware_results) if not hardware_results.empty else 0,
            "distributed_count": len(dist_tests) if not dist_tests.empty else 0,
        }
    }
    
    # Save data as JSON
    data_file = os.path.join(dashboard_dir, "dashboard_data.json")
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Dashboard data saved to {data_file}")
    
    # Create charts if Plotly is available
    if PLOTLY_AVAILABLE:
        print("Creating charts...")
        charts = {}
        
        # Test results charts
        if not test_df.empty:
            charts["model_coverage"] = create_model_coverage_chart(test_df)
            charts["test_success"] = create_test_success_chart(test_df)
            charts["performance_trend"] = create_performance_trend_chart(test_df)
        
        # Hardware results charts
        if not hardware_results.empty:
            charts["hardware_comparison"] = create_hardware_comparison_chart(hardware_results)
            charts["hardware_matrix"] = create_hardware_matrix_heatmap(hardware_results)
            charts["performance_comparison"] = create_performance_comparison_chart(hardware_results)
            charts["memory_usage"] = create_memory_usage_chart(hardware_results)
        
        # Save charts as HTML
        for name, fig in charts.items():
            if fig is not None:
                chart_file = os.path.join(dashboard_dir, f"{name}.html")
                fig.write_html(chart_file)
                print(f"Chart saved to {chart_file}")
    
    return data

def create_interactive_dashboard(data=None, results_dir="collected_results", dist_dir="distributed_results", hardware_db="hardware_compatibility_matrix.duckdb", port=8050):
    """
    Create interactive dashboard with Dash
    """
    if not DASH_AVAILABLE:
        print("Dash not available, cannot create interactive dashboard")
        print("Please install with: pip install dash")
        return
    
    # Load data if not provided
    if data is None:
        # Load all data
        test_df = load_test_results(results_dir)
        
        if DUCKDB_AVAILABLE and os.path.exists(hardware_db):
            hardware_results, hardware_detection = load_hardware_results(hardware_db)
        else:
            hardware_results = pd.DataFrame()
            hardware_detection = pd.DataFrame()
        
        dist_tests, dist_workers = load_distributed_results(dist_dir)
    else:
        # Use provided data
        test_df = pd.DataFrame(data.get("test_results", []))
        hardware_results = pd.DataFrame(data.get("hardware_results", []))
        hardware_detection = pd.DataFrame(data.get("hardware_detection", []))
        dist_tests = pd.DataFrame(data.get("distributed_tests", []))
        dist_workers = pd.DataFrame(data.get("distributed_workers", []))
    
    # Initialize Dash app
    app = dash.Dash(__name__, title="HuggingFace Test Dashboard")
    
    # Define layout
    app.layout = html.Div([
        html.H1("HuggingFace Testing Framework Dashboard"),
        
        # Dashboard tabs
        dcc.Tabs([
            # Overview Tab
            dcc.Tab(label="Overview", children=[
                html.Div([
                    html.H2("Testing Overview"),
                    
                    # Statistics cards
                    html.Div([
                        html.Div([
                            html.H3("Test Results"),
                            html.P(f"{len(test_df)} test results"),
                            html.P(f"{test_df['architecture'].nunique()} model architectures")
                            if not test_df.empty else html.P("No data"),
                        ], className="stat-card"),
                        
                        html.Div([
                            html.H3("Hardware Compatibility"),
                            html.P(f"{len(hardware_results)} hardware tests"),
                            html.P(f"{hardware_results['hardware'].nunique()} hardware platforms")
                            if not hardware_results.empty else html.P("No data"),
                        ], className="stat-card"),
                        
                        html.Div([
                            html.H3("Distributed Testing"),
                            html.P(f"{len(dist_tests)} distributed test runs"),
                            html.P(f"{len(dist_workers)} worker executions")
                            if not dist_workers.empty else html.P("No data"),
                        ], className="stat-card"),
                    ], className="stat-card-container"),
                    
                    # Charts
                    html.Div([
                        html.Div([
                            html.H3("Model Coverage"),
                            dcc.Graph(
                                id="model-coverage-chart",
                                figure=create_model_coverage_chart(test_df) if not test_df.empty else {}
                            )
                        ], className="chart-container"),
                        
                        html.Div([
                            html.H3("Test Success Rate"),
                            dcc.Graph(
                                id="test-success-chart",
                                figure=create_test_success_chart(test_df) if not test_df.empty else {}
                            )
                        ], className="chart-container"),
                    ]),
                    
                    # Hardware comparison
                    html.Div([
                        html.H3("Hardware Compatibility Matrix"),
                        dcc.Graph(
                            id="hardware-matrix-chart",
                            figure=create_hardware_matrix_heatmap(hardware_results) if not hardware_results.empty else {}
                        )
                    ], className="full-width-chart"),
                ])
            ]),
            
            # Test Results Tab
            dcc.Tab(label="Test Results", children=[
                html.Div([
                    html.H2("Test Results"),
                    
                    # Filters
                    html.Div([
                        html.Div([
                            html.Label("Architecture"),
                            dcc.Dropdown(
                                id="architecture-filter",
                                options=[
                                    {"label": arch.title(), "value": arch}
                                    for arch in test_df["architecture"].unique()
                                ] if not test_df.empty else [],
                                multi=True,
                                placeholder="All Architectures"
                            )
                        ], className="filter"),
                        
                        html.Div([
                            html.Label("Status"),
                            dcc.Dropdown(
                                id="status-filter",
                                options=[
                                    {"label": "Success", "value": "success"},
                                    {"label": "Failure", "value": "failure"},
                                    {"label": "Mock Objects", "value": "mock"},
                                    {"label": "Real Inference", "value": "real"}
                                ],
                                multi=True,
                                placeholder="All Statuses"
                            )
                        ], className="filter"),
                        
                        html.Button("Apply Filters", id="apply-filters", className="filter-button")
                    ], className="filter-container"),
                    
                    # Results table
                    html.Div([
                        dash_table.DataTable(
                            id="test-results-table",
                            columns=[
                                {"name": "Model", "id": "model_name"},
                                {"name": "Architecture", "id": "architecture"},
                                {"name": "Status", "id": "status"},
                                {"name": "Inference Type", "id": "inference_type"},
                                {"name": "Timestamp", "id": "timestamp_str"}
                            ],
                            data=[],
                            filter_action="native",
                            sort_action="native",
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_data_conditional=[
                                {
                                    "if": {"filter_query": "{status} = 'Success'"},
                                    "backgroundColor": "rgba(0, 255, 0, 0.1)"
                                },
                                {
                                    "if": {"filter_query": "{status} = 'Failure'"},
                                    "backgroundColor": "rgba(255, 0, 0, 0.1)"
                                }
                            ]
                        )
                    ], className="table-container"),
                    
                    # Performance trend chart
                    html.Div([
                        html.H3("Performance Trend"),
                        dcc.Graph(
                            id="performance-trend-chart",
                            figure=create_performance_trend_chart(test_df) if not test_df.empty else {}
                        )
                    ], className="full-width-chart")
                ])
            ]),
            
            # Hardware Compatibility Tab
            dcc.Tab(label="Hardware Compatibility", children=[
                html.Div([
                    html.H2("Hardware Compatibility"),
                    
                    # Hardware detection
                    html.Div([
                        html.H3("Available Hardware"),
                        html.Div(id="hardware-detection-cards", className="stat-card-container")
                    ]),
                    
                    # Hardware comparison chart
                    html.Div([
                        html.H3("Hardware Performance Comparison"),
                        dcc.Graph(
                            id="hardware-comparison-chart",
                            figure=create_hardware_comparison_chart(hardware_results) if not hardware_results.empty else {}
                        )
                    ], className="full-width-chart"),
                    
                    # Performance by model type
                    html.Div([
                        html.H3("Performance by Model Type"),
                        dcc.Graph(
                            id="performance-comparison-chart",
                            figure=create_performance_comparison_chart(hardware_results) if not hardware_results.empty else {}
                        )
                    ], className="full-width-chart"),
                    
                    # Memory usage chart
                    html.Div([
                        html.H3("Memory Usage"),
                        dcc.Graph(
                            id="memory-usage-chart",
                            figure=create_memory_usage_chart(hardware_results) if not hardware_results.empty else {}
                        )
                    ], className="full-width-chart")
                ])
            ]),
            
            # Distributed Testing Tab
            dcc.Tab(label="Distributed Testing", children=[
                html.Div([
                    html.H2("Distributed Testing"),
                    
                    # Test runs
                    html.Div([
                        html.H3("Distributed Test Runs"),
                        dash_table.DataTable(
                            id="distributed-tests-table",
                            columns=[
                                {"name": "Test Name", "id": "test_name"},
                                {"name": "Workers", "id": "worker_count"},
                                {"name": "Success Rate", "id": "success_rate"},
                                {"name": "Duration (s)", "id": "total_duration"},
                                {"name": "Timestamp", "id": "timestamp_str"}
                            ],
                            data=[],
                            filter_action="native",
                            sort_action="native",
                            page_size=10,
                            style_table={"overflowX": "auto"}
                        )
                    ], className="table-container"),
                    
                    # Worker performance
                    html.Div([
                        html.H3("Worker Performance"),
                        dcc.Graph(id="worker-performance-chart")
                    ], className="full-width-chart"),
                    
                    # Hardware distribution
                    html.Div([
                        html.H3("Hardware Distribution"),
                        dcc.Graph(id="hardware-distribution-chart")
                    ], className="full-width-chart")
                ])
            ])
        ])
    ])
    
    # Define callbacks
    @app.callback(
        Output("test-results-table", "data"),
        [Input("apply-filters", "n_clicks")],
        [State("architecture-filter", "value"),
         State("status-filter", "value")]
    )
    def update_results_table(n_clicks, architecture, status):
        if test_df.empty:
            return []
        
        # Create a copy of the DataFrame
        filtered_df = test_df.copy()
        
        # Apply architecture filter
        if architecture and len(architecture) > 0:
            filtered_df = filtered_df[filtered_df["architecture"].isin(architecture)]
        
        # Apply status filter
        if status and len(status) > 0:
            if "success" in status and "failure" in status:
                # No filtering needed for success/failure
                pass
            elif "success" in status:
                filtered_df = filtered_df[filtered_df.get("success", True) == True]
            elif "failure" in status:
                filtered_df = filtered_df[filtered_df.get("success", True) == False]
            
            if "mock" in status and "real" in status:
                # No filtering needed for mock/real
                pass
            elif "mock" in status:
                filtered_df = filtered_df[filtered_df.get("using_mocks", False) == True]
            elif "real" in status:
                filtered_df = filtered_df[filtered_df.get("using_real_inference", True) == True]
        
        # Prepare data for table
        result_data = []
        
        for _, row in filtered_df.iterrows():
            # Get model name from available columns
            model_name = None
            for col in ["model_name", "model_id", "model"]:
                if col in row and row[col]:
                    model_name = row[col]
                    break
            
            if not model_name:
                continue
            
            # Get success status
            success = row.get("success", True)
            
            # Get inference type
            using_mocks = row.get("using_mocks", False)
            
            result_data.append({
                "model_name": model_name,
                "architecture": row.get("architecture", "unknown"),
                "status": "Success" if success else "Failure",
                "inference_type": "Mock Objects" if using_mocks else "Real Inference",
                "timestamp_str": row.get("timestamp_str", "")
            })
        
        return result_data
    
    @app.callback(
        Output("hardware-detection-cards", "children"),
        [Input("hardware-compatibility", "active_tab")]  # Just a trigger, not actually used
    )
    def update_hardware_cards(_):
        if hardware_detection.empty:
            return html.P("No hardware detection data available")
        
        # Get the latest detection record for each hardware type
        latest_detection = hardware_detection.sort_values("timestamp", ascending=False)
        latest_detection = latest_detection.drop_duplicates("hardware_type")
        
        # Create cards for each hardware type
        cards = []
        
        for _, row in latest_detection.iterrows():
            hw_type = row["hardware_type"]
            available = row["available"]
            name = row["name"] if available else "Not Available"
            
            # Parse features if available
            features_str = ""
            if available and "features" in row and row["features"]:
                try:
                    if isinstance(row["features"], str):
                        features = json.loads(row["features"])
                    else:
                        features = row["features"]
                    
                    for k, v in features.items():
                        features_str += f"{k}: {v}<br>"
                except:
                    features_str = str(row["features"])
            
            # Create card
            card = html.Div([
                html.H4(f"{hw_type.upper()}"),
                html.P("✅ Available" if available else "❌ Not Available"),
                html.P(name) if available else None,
                html.P(html.Span(
                    dangerouslySetInnerHTML={"__html": features_str}
                )) if features_str else None
            ], className="stat-card")
            
            cards.append(card)
        
        return cards
    
    @app.callback(
        [Output("distributed-tests-table", "data"),
         Output("worker-performance-chart", "figure"),
         Output("hardware-distribution-chart", "figure")],
        [Input("distributed-testing", "active_tab")]  # Just a trigger, not actually used
    )
    def update_distributed_data(_):
        # Table data
        table_data = []
        
        if not dist_tests.empty:
            for _, row in dist_tests.iterrows():
                # Calculate worker count
                worker_count = 0
                if "results" in row and isinstance(row["results"], list):
                    worker_ids = set()
                    for result in row["results"]:
                        if isinstance(result, dict) and "worker_id" in result:
                            worker_ids.add(result["worker_id"])
                    worker_count = len(worker_ids)
                
                # Calculate success rate
                success_rate = 0
                if "successful_tasks" in row and "total_tasks" in row:
                    if row["total_tasks"] > 0:
                        success_rate = row["successful_tasks"] / row["total_tasks"]
                elif "success_rate" in row:
                    success_rate = row["success_rate"]
                
                table_data.append({
                    "test_name": row.get("test_name", "Unknown"),
                    "worker_count": worker_count,
                    "success_rate": f"{success_rate:.1%}",
                    "total_duration": f"{row.get('total_duration', 0):.2f}",
                    "timestamp_str": row.get("timestamp_str", "")
                })
        
        # Worker performance chart
        if dist_workers.empty:
            worker_fig = go.Figure()
        else:
            # Group by worker_id and calculate statistics
            worker_stats = dist_workers.groupby("worker_id").agg({
                "execution_time": ["mean", "median", "count"]
            }).reset_index()
            
            worker_stats.columns = ["worker_id", "mean_time", "median_time", "task_count"]
            
            # Sort by median time
            worker_stats = worker_stats.sort_values("median_time")
            
            # Create bar chart
            worker_fig = px.bar(
                worker_stats,
                x="worker_id",
                y="median_time",
                color="task_count",
                hover_data=["mean_time", "task_count"],
                title="Worker Performance (Median Execution Time)",
                labels={"median_time": "Median Execution Time (s)", "worker_id": "Worker ID", "task_count": "Tasks Executed"}
            )
        
        # Hardware distribution chart
        if dist_workers.empty:
            hw_fig = go.Figure()
        else:
            # Count tasks by hardware
            if "hardware" in dist_workers.columns:
                hw_counts = dist_workers["hardware"].value_counts().reset_index()
                hw_counts.columns = ["hardware", "count"]
                
                # Create pie chart
                hw_fig = px.pie(
                    hw_counts,
                    values="count",
                    names="hardware",
                    title="Task Distribution by Hardware",
                    color="hardware",
                    color_discrete_map=HARDWARE_COLORS
                )
                
                hw_fig.update_traces(textinfo="value+percent", hole=0.3)
            else:
                hw_fig = go.Figure()
        
        return table_data, worker_fig, hw_fig
    
    # Add custom CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }
                h1, h2, h3 {
                    color: #333;
                    padding: 15px;
                    margin: 0;
                }
                h1 {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                }
                .stat-card-container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin: 15px;
                }
                .stat-card {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 15px;
                    margin-bottom: 15px;
                    flex: 1;
                    min-width: 200px;
                    margin-right: 15px;
                }
                .chart-container {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 15px;
                    margin: 15px;
                    flex: 1;
                }
                .full-width-chart {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 15px;
                    margin: 15px;
                }
                .filter-container {
                    display: flex;
                    flex-wrap: wrap;
                    align-items: flex-end;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 15px;
                    margin: 15px;
                }
                .filter {
                    margin-right: 15px;
                    min-width: 200px;
                    flex: 1;
                }
                .filter-button {
                    background-color: #2c3e50;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 10px 15px;
                    cursor: pointer;
                }
                .table-container {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 15px;
                    margin: 15px;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Run the server
    print(f"Starting dashboard server on port {port}...")
    app.run_server(debug=False, port=port)

def create_static_dashboard(data=None, results_dir="collected_results", dist_dir="distributed_results", hardware_db="hardware_compatibility_matrix.duckdb", output_dir="dashboard"):
    """
    Create static dashboard HTML files
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, cannot create static dashboard")
        print("Please install with: pip install plotly")
        return
    
    # Load data if not provided
    if data is None:
        # Create dashboard data
        data = create_dashboard_data(results_dir, dist_dir, hardware_db)
        
        if data is None:
            print("No data available to create dashboard")
            return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert data to DataFrames
    test_df = pd.DataFrame(data.get("test_results", []))
    hardware_results = pd.DataFrame(data.get("hardware_results", []))
    hardware_detection = pd.DataFrame(data.get("hardware_detection", []))
    dist_tests = pd.DataFrame(data.get("distributed_tests", []))
    dist_workers = pd.DataFrame(data.get("distributed_workers", []))
    
    # Create HTML dashboard
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>HuggingFace Testing Dashboard</title>
            <style>
                body {{
                    font-family: sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                h1, h2, h3 {{
                    color: #333;
                    padding: 15px;
                    margin: 0;
                }}
                h1 {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                }}
                .card-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin: 15px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 15px;
                    margin-bottom: 15px;
                    flex: 1;
                    min-width: 200px;
                    margin-right: 15px;
                }}
                .chart-container {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 15px;
                    margin: 15px;
                }}
                .chart {{
                    width: 100%;
                    height: 500px;
                    border: none;
                }}
                .section {{
                    margin: 20px 0;
                }}
                .tabs {{
                    display: flex;
                    border-bottom: 1px solid #ccc;
                    margin-bottom: 20px;
                }}
                .tab {{
                    padding: 10px 20px;
                    cursor: pointer;
                    background-color: #f1f1f1;
                    border: 1px solid #ccc;
                    border-bottom: none;
                    margin-right: 5px;
                    border-radius: 5px 5px 0 0;
                }}
                .tab.active {{
                    background-color: white;
                    border-bottom: 1px solid white;
                    margin-bottom: -1px;
                }}
                .tab-content {{
                    display: none;
                }}
                .tab-content.active {{
                    display: block;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                    }}
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).className += " active";
                    evt.currentTarget.className += " active";
                }}
            </script>
        </head>
        <body>
            <h1>HuggingFace Testing Framework Dashboard</h1>
            
            <div class="tabs">
                <button class="tab active" onclick="openTab(event, 'overview')">Overview</button>
                <button class="tab" onclick="openTab(event, 'test-results')">Test Results</button>
                <button class="tab" onclick="openTab(event, 'hardware')">Hardware Compatibility</button>
                <button class="tab" onclick="openTab(event, 'distributed')">Distributed Testing</button>
            </div>
            
            <div id="overview" class="tab-content active">
                <h2>Testing Overview</h2>
                
                <div class="card-container">
                    <div class="card">
                        <h3>Test Results</h3>
                        <p>{len(test_df)} test results</p>
                        <p>{test_df['architecture'].nunique() if not test_df.empty else 0} model architectures</p>
                    </div>
                    
                    <div class="card">
                        <h3>Hardware Compatibility</h3>
                        <p>{len(hardware_results)} hardware tests</p>
                        <p>{hardware_results['hardware'].nunique() if not hardware_results.empty else 0} hardware platforms</p>
                    </div>
                    
                    <div class="card">
                        <h3>Distributed Testing</h3>
                        <p>{len(dist_tests)} distributed test runs</p>
                        <p>{len(dist_workers)} worker executions</p>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>Model Coverage</h3>
                    <iframe class="chart" src="model_coverage.html"></iframe>
                </div>
                
                <div class="chart-container">
                    <h3>Test Success Rate</h3>
                    <iframe class="chart" src="test_success.html"></iframe>
                </div>
                
                <div class="chart-container">
                    <h3>Hardware Compatibility Matrix</h3>
                    <iframe class="chart" src="hardware_matrix.html"></iframe>
                </div>
            </div>
            
            <div id="test-results" class="tab-content">
                <h2>Test Results</h2>
                
                <div class="chart-container">
                    <h3>Performance Trend</h3>
                    <iframe class="chart" src="performance_trend.html"></iframe>
                </div>
                
                <div class="chart-container">
                    <h3>Recent Test Results</h3>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Architecture</th>
                            <th>Status</th>
                            <th>Inference Type</th>
                            <th>Timestamp</th>
                        </tr>
                        {"".join([f"<tr><td>{row.get('model_name', '')}</td><td>{row.get('architecture', '')}</td><td>{'✅' if row.get('success', True) else '❌'}</td><td>{'🔷 Mock' if row.get('using_mocks', False) else '🚀 Real'}</td><td>{row.get('timestamp_str', '')}</td></tr>" for _, row in test_df.head(20).iterrows()]) if not test_df.empty else "<tr><td colspan='5'>No data available</td></tr>"}
                    </table>
                </div>
            </div>
            
            <div id="hardware" class="tab-content">
                <h2>Hardware Compatibility</h2>
                
                <div class="card-container">
                    {"".join([f'<div class="card"><h3>{row["hardware_type"].upper()}</h3><p>{"✅ Available" if row["available"] else "❌ Not Available"}</p>{f"<p>{row.get("name", "")}</p>" if row["available"] else ""}</div>' for _, row in hardware_detection.drop_duplicates("hardware_type").iterrows()]) if not hardware_detection.empty else '<div class="card"><h3>No Hardware Data</h3><p>No hardware detection data available</p></div>'}
                </div>
                
                <div class="chart-container">
                    <h3>Hardware Performance Comparison</h3>
                    <iframe class="chart" src="hardware_comparison.html"></iframe>
                </div>
                
                <div class="chart-container">
                    <h3>Performance by Model Type</h3>
                    <iframe class="chart" src="performance_comparison.html"></iframe>
                </div>
                
                <div class="chart-container">
                    <h3>Memory Usage</h3>
                    <iframe class="chart" src="memory_usage.html"></iframe>
                </div>
            </div>
            
            <div id="distributed" class="tab-content">
                <h2>Distributed Testing</h2>
                
                <div class="chart-container">
                    <h3>Distributed Test Runs</h3>
                    <table>
                        <tr>
                            <th>Test Name</th>
                            <th>Success Rate</th>
                            <th>Workers</th>
                            <th>Duration (s)</th>
                            <th>Timestamp</th>
                        </tr>
                        {"".join([f"<tr><td>{row.get('test_name', '')}</td><td>{row.get('success_rate', 0):.1%}</td><td>{len(set([r.get('worker_id', '') for r in row.get('results', []) if isinstance(r, dict) and 'worker_id' in r]))}</td><td>{row.get('total_duration', 0):.2f}</td><td>{row.get('timestamp_str', '')}</td></tr>" for _, row in dist_tests.head(10).iterrows()]) if not dist_tests.empty else "<tr><td colspan='5'>No data available</td></tr>"}
                    </table>
                </div>
            </div>
            
            <footer>
                <p style="text-align: center; padding: 20px; color: #777;">
                    Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </p>
            </footer>
        </body>
        </html>
        """)
    
    # Generate chart HTML files
    print("Generating chart files...")
    
    # Test results charts
    if not test_df.empty:
        model_coverage = create_model_coverage_chart(test_df)
        if model_coverage:
            model_coverage.write_html(os.path.join(output_dir, "model_coverage.html"))
        
        test_success = create_test_success_chart(test_df)
        if test_success:
            test_success.write_html(os.path.join(output_dir, "test_success.html"))
        
        performance_trend = create_performance_trend_chart(test_df)
        if performance_trend:
            performance_trend.write_html(os.path.join(output_dir, "performance_trend.html"))
    
    # Hardware results charts
    if not hardware_results.empty:
        hardware_comparison = create_hardware_comparison_chart(hardware_results)
        if hardware_comparison:
            hardware_comparison.write_html(os.path.join(output_dir, "hardware_comparison.html"))
        
        hardware_matrix = create_hardware_matrix_heatmap(hardware_results)
        if hardware_matrix:
            hardware_matrix.write_html(os.path.join(output_dir, "hardware_matrix.html"))
        
        performance_comparison = create_performance_comparison_chart(hardware_results)
        if performance_comparison:
            performance_comparison.write_html(os.path.join(output_dir, "performance_comparison.html"))
        
        memory_usage = create_memory_usage_chart(hardware_results)
        if memory_usage:
            memory_usage.write_html(os.path.join(output_dir, "memory_usage.html"))
    
    # Worker performance chart
    if not dist_workers.empty:
        # Group by worker_id and calculate statistics
        if "worker_id" in dist_workers.columns and "execution_time" in dist_workers.columns:
            worker_stats = dist_workers.groupby("worker_id").agg({
                "execution_time": ["mean", "median", "count"]
            }).reset_index()
            
            worker_stats.columns = ["worker_id", "mean_time", "median_time", "task_count"]
            
            # Create bar chart
            worker_fig = px.bar(
                worker_stats,
                x="worker_id",
                y="median_time",
                color="task_count",
                hover_data=["mean_time", "task_count"],
                title="Worker Performance (Median Execution Time)",
                labels={"median_time": "Median Execution Time (s)", "worker_id": "Worker ID", "task_count": "Tasks Executed"}
            )
            
            worker_fig.write_html(os.path.join(output_dir, "worker_performance.html"))
    
    print(f"Static dashboard generated in {output_dir}")
    print(f"Open {os.path.join(output_dir, 'index.html')} in a web browser to view")

def main():
    parser = argparse.ArgumentParser(description="Test Dashboard Generator")
    parser.add_argument("--results-dir", type=str, default="collected_results",
                       help="Directory containing test results JSON files")
    parser.add_argument("--dist-dir", type=str, default="distributed_results",
                       help="Directory containing distributed test results")
    parser.add_argument("--hardware-db", type=str, default="hardware_compatibility_matrix.duckdb",
                       help="Path to hardware compatibility database")
    parser.add_argument("--days", type=int, default=30,
                       help="Number of days of data to include")
    parser.add_argument("--output-dir", type=str, default="dashboard",
                       help="Output directory for static dashboard")
    parser.add_argument("--port", type=int, default=8050,
                       help="Port for interactive dashboard server")
    parser.add_argument("--interactive", action="store_true",
                       help="Launch interactive dashboard server")
    parser.add_argument("--static", action="store_true",
                       help="Generate static HTML dashboard")
    args = parser.parse_args()
    
    # Create dashboard data
    print("Creating dashboard data...")
    data = create_dashboard_data(args.results_dir, args.dist_dir, args.hardware_db, args.days)
    
    if data is None:
        print("No data available to create dashboard")
        return
    
    # Create static dashboard if requested
    if args.static:
        print("Generating static dashboard...")
        create_static_dashboard(data, args.results_dir, args.dist_dir, args.hardware_db, args.output_dir)
    
    # Launch interactive dashboard if requested
    if args.interactive:
        print("Launching interactive dashboard...")
        create_interactive_dashboard(data, args.results_dir, args.dist_dir, args.hardware_db, args.port)
    
    # Default to static if neither specified
    if not args.static and not args.interactive:
        print("Generating static dashboard by default...")
        create_static_dashboard(data, args.results_dir, args.dist_dir, args.hardware_db, args.output_dir)

if __name__ == "__main__":
    main()