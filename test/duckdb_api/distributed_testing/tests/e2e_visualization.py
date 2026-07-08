#!/usr/bin/env python3
"""
End-to-End Test Visualization for Distributed Testing Framework

This script provides visualization capabilities for end-to-end test results, 
creating interactive visualizations that can be embedded in HTML reports or 
viewed directly in the monitoring dashboard.

Usage:
    python -m duckdb_api.distributed_testing.tests.e2e_visualization [options]

Options:
    --report-dir DIR               Directory containing test reports (default: ./e2e_test_reports)
    --output-dir DIR               Directory for generated visualizations (default: ./e2e_visualizations)
    --test-id ID                   Specific test ID to visualize (default: latest)
    --visualization-types TYPES    Types of visualizations to generate [summary,component,timing,failures,all] (default: all)
    --dashboard-integration        Integrate visualizations with monitoring dashboard
    --dashboard-url URL            Monitoring dashboard URL (default: http://localhost:8080)
    --generate-standalone          Generate standalone HTML visualizations
    --theme THEME                  Visualization theme [light,dark] (default: light)
"""

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from plotly.offline import plot
from plotly.subplots import make_subplots

# Add parent directory to path to ensure imports work properly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Function to find the latest test report
def find_latest_report(report_dir: str) -> Optional[str]:
    """Find the most recent test report file in the given directory."""
    report_files = glob.glob(os.path.join(report_dir, "*_results.json"))
    if not report_files:
        return None
    
    # Sort by modification time, most recent first
    report_files.sort(key=os.path.getmtime, reverse=True)
    return report_files[0]

# Function to load a test report
def load_test_report(report_file: str) -> Dict:
    """Load and parse a test report file."""
    with open(report_file, 'r') as f:
        return json.load(f)

# Function to extract test ID from filename
def extract_test_id(filename: str) -> str:
    """Extract the test ID from a report filename."""
    match = re.search(r'(e2e_test_[0-9_]+)_results\.json', os.path.basename(filename))
    if match:
        return match.group(1)
    return os.path.basename(filename).replace('_results.json', '')

# Function to create output directory if it doesn't exist
def ensure_output_dir(output_dir: str) -> None:
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)

# Create summary visualization
def create_summary_visualization(report: Dict, output_dir: str, test_id: str, theme: str = 'light') -> str:
    """Create a summary visualization of test results."""
    # Get theme configuration
    theme_config = {
        'light': {
            'bg_color': '#ffffff',
            'text_color': '#333333',
            'grid_color': '#eeeeee',
            'success_color': '#28a745',
            'failure_color': '#dc3545',
            'warning_color': '#ffc107',
            'info_color': '#17a2b8'
        },
        'dark': {
            'bg_color': '#222222',
            'text_color': '#f2f2f2',
            'grid_color': '#444444',
            'success_color': '#28a745',
            'failure_color': '#dc3545',
            'warning_color': '#ffc107',
            'info_color': '#17a2b8'
        }
    }[theme]
    
    # Extract validation results
    validation_results = report.get('validation_results', {})
    summary = report.get('summary', {})
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Component Validation Status",
            "Test Configuration",
            "Validation Details",
            "Hardware Profiles"
        ),
        specs=[
            [{"type": "pie"}, {"type": "table"}],
            [{"type": "bar"}, {"type": "pie"}],
        ]
    )
    
    # Add component validation status pie chart
    labels = list(summary.keys())
    values = [1 if v else 0 for v in summary.values()]
    
    # Remove overall_success from the pie chart
    if 'overall_success' in labels:
        idx = labels.index('overall_success')
        labels.pop(idx)
        values.pop(idx)
    
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=[theme_config['success_color'] if v else theme_config['failure_color'] for v in values],
            textinfo='label+percent',
            hoverinfo='label+percent',
            hole=0.3,
        ),
        row=1, col=1
    )
    
    # Add test configuration table
    config = report.get('configuration', {})
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Configuration Parameter', 'Value'],
                fill_color=theme_config['info_color'],
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    list(config.keys()),
                    [str(v) if not isinstance(v, list) else ', '.join(v) for v in config.values()]
                ],
                fill_color=theme_config['bg_color'],
                align='left',
                font=dict(color=theme_config['text_color'], size=11)
            )
        ),
        row=1, col=2
    )
    
    # Add validation details bar chart
    validation_categories = []
    validation_pass_counts = []
    validation_fail_counts = []
    
    for category, details in validation_results.items():
        validation_categories.append(category)
        pass_count = sum(1 for k, v in details.items() if v and k != category + '_accessible')
        fail_count = sum(1 for k, v in details.items() if not v and k != category + '_accessible')
        validation_pass_counts.append(pass_count)
        validation_fail_counts.append(fail_count)
    
    fig.add_trace(
        go.Bar(
            name='Pass',
            x=validation_categories,
            y=validation_pass_counts,
            marker_color=theme_config['success_color']
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Fail',
            x=validation_categories,
            y=validation_fail_counts,
            marker_color=theme_config['failure_color']
        ),
        row=2, col=1
    )
    
    # Add hardware profiles pie chart
    hardware_profiles = config.get('hardware_profiles', [])
    if not isinstance(hardware_profiles, list):
        hardware_profiles = [hardware_profiles]
    
    if hardware_profiles:
        # If 'all' is in the list, expand it to include all hardware types
        if 'all' in hardware_profiles:
            hardware_profiles = ['cpu', 'gpu', 'webgpu', 'webnn', 'multi']
        
        fig.add_trace(
            go.Pie(
                labels=hardware_profiles,
                values=[1] * len(hardware_profiles),  # Equal weight for each profile
                textinfo='label',
                hoverinfo='label',
                marker_colors=[
                    '#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'
                ][:len(hardware_profiles)]
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text=f"End-to-End Test Results Summary (Test ID: {test_id})",
        barmode='stack',
        height=800,
        width=1200,
        paper_bgcolor=theme_config['bg_color'],
        plot_bgcolor=theme_config['bg_color'],
        font=dict(color=theme_config['text_color']),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    # Save the figure to an HTML file
    output_file = os.path.join(output_dir, f'{test_id}_summary.html')
    plot(fig, filename=output_file, auto_open=False)
    
    return output_file

# Create component status visualization
def create_component_visualization(report: Dict, output_dir: str, test_id: str, theme: str = 'light') -> str:
    """Create a visualization of component status during the test."""
    # Get theme configuration
    theme_config = {
        'light': {
            'bg_color': '#ffffff',
            'text_color': '#333333',
            'grid_color': '#eeeeee',
            'success_color': '#28a745',
            'failure_color': '#dc3545',
            'warning_color': '#ffc107',
            'info_color': '#17a2b8'
        },
        'dark': {
            'bg_color': '#222222',
            'text_color': '#f2f2f2',
            'grid_color': '#444444',
            'success_color': '#28a745',
            'failure_color': '#dc3545',
            'warning_color': '#ffc107',
            'info_color': '#17a2b8'
        }
    }[theme]
    
    # Extract validation results for each component
    validation_results = report.get('validation_results', {})
    
    # Define the components
    components = [
        "Result Aggregator",
        "Coordinator",
        "Monitoring Dashboard",
        "Worker Nodes",
        "Integration System"
    ]
    
    # Map validation results to components
    component_status = {
        "Result Aggregator": validation_results.get('result_aggregation', {}).get('summary_accessible', False),
        "Coordinator": True,  # Assuming coordinator is working if the test ran
        "Monitoring Dashboard": validation_results.get('dashboard', {}).get('dashboard_accessible', False),
        "Worker Nodes": True,  # Assuming workers are functioning if the test ran
        "Integration System": validation_results.get('integration', {}).get('visualization_data_accessible', False)
    }
    
    # Create a figure
    fig = go.Figure()
    
    # Add component status as a horizontal bar chart
    fig.add_trace(
        go.Bar(
            x=[1 if status else 0 for status in component_status.values()],
            y=list(component_status.keys()),
            orientation='h',
            marker_color=[theme_config['success_color'] if status else theme_config['failure_color'] 
                         for status in component_status.values()],
            text=['Operational' if status else 'Failed' for status in component_status.values()],
            textposition='auto',
            name='Component Status'
        )
    )
    
    # Add detailed status information
    for i, (component, status) in enumerate(component_status.items()):
        if component == "Result Aggregator":
            has_data = validation_results.get('result_aggregation', {}).get('has_test_results', False)
            fig.add_annotation(
                x=1.1,
                y=i,
                text=f"Data Present: {'Yes' if has_data else 'No'}",
                showarrow=False,
                font=dict(color=theme_config['text_color']),
                align="left"
            )
        elif component == "Monitoring Dashboard":
            dashboard_content = validation_results.get('dashboard', {}).get('dashboard_content_length', 0)
            fig.add_annotation(
                x=1.1,
                y=i,
                text=f"Content Size: {dashboard_content} bytes",
                showarrow=False,
                font=dict(color=theme_config['text_color']),
                align="left"
            )
        elif component == "Integration System":
            has_data = validation_results.get('integration', {}).get('has_data', False)
            fig.add_annotation(
                x=1.1,
                y=i,
                text=f"Integration Data: {'Available' if has_data else 'Missing'}",
                showarrow=False,
                font=dict(color=theme_config['text_color']),
                align="left"
            )
    
    # Update layout
    fig.update_layout(
        title_text=f"Component Status (Test ID: {test_id})",
        xaxis=dict(
            title="Status",
            tickvals=[0, 1],
            ticktext=["Failed", "Operational"],
            range=[0, 2]  # Extend range to make room for annotations
        ),
        yaxis=dict(
            title="Component"
        ),
        height=600,
        width=1000,
        paper_bgcolor=theme_config['bg_color'],
        plot_bgcolor=theme_config['bg_color'],
        font=dict(color=theme_config['text_color']),
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    # Save the figure to an HTML file
    output_file = os.path.join(output_dir, f'{test_id}_component_status.html')
    plot(fig, filename=output_file, auto_open=False)
    
    return output_file

# Create timing visualization
def create_timing_visualization(report: Dict, output_dir: str, test_id: str, theme: str = 'light') -> str:
    """Create a visualization of test timing information."""
    # Get theme configuration
    theme_config = {
        'light': {
            'bg_color': '#ffffff',
            'text_color': '#333333',
            'grid_color': '#eeeeee',
            'success_color': '#28a745',
            'failure_color': '#dc3545',
            'warning_color': '#ffc107',
            'info_color': '#17a2b8'
        },
        'dark': {
            'bg_color': '#222222',
            'text_color': '#f2f2f2',
            'grid_color': '#444444',
            'success_color': '#28a745',
            'failure_color': '#dc3545',
            'warning_color': '#ffc107',
            'info_color': '#17a2b8'
        }
    }[theme]
    
    # Simulated test timing data (in a real scenario, this would come from the test execution)
    # In a production system, actual timestamps from the test would be used
    timing_data = {
        "start_services": 0,
        "services_started": 5,
        "start_workers": 5,
        "workers_started": 10,
        "submit_workloads": 10,
        "workloads_submitted": 12,
        "inject_failures": report.get('configuration', {}).get('include_failures', False) and 15 or None,
        "failures_injected": report.get('configuration', {}).get('include_failures', False) and 20 or None,
        "test_execution": 12,
        "test_completed": 12 + report.get('configuration', {}).get('test_duration', 60),
        "validation_started": 12 + report.get('configuration', {}).get('test_duration', 60),
        "validation_completed": 12 + report.get('configuration', {}).get('test_duration', 60) + 5,
        "cleanup_started": 12 + report.get('configuration', {}).get('test_duration', 60) + 5,
        "cleanup_completed": 12 + report.get('configuration', {}).get('test_duration', 60) + 10
    }
    
    # Create pairs of start and end times for Gantt chart
    phases = [
        {"Task": "Start Services", "Start": timing_data["start_services"], "Finish": timing_data["services_started"], "Resource": "Infrastructure"},
        {"Task": "Start Workers", "Start": timing_data["start_workers"], "Finish": timing_data["workers_started"], "Resource": "Infrastructure"},
        {"Task": "Submit Workloads", "Start": timing_data["submit_workloads"], "Finish": timing_data["workloads_submitted"], "Resource": "Workload"}
    ]
    
    if timing_data["inject_failures"] is not None:
        phases.append({"Task": "Inject Failures", "Start": timing_data["inject_failures"], "Finish": timing_data["failures_injected"], "Resource": "Fault Injection"})
    
    phases.extend([
        {"Task": "Test Execution", "Start": timing_data["test_execution"], "Finish": timing_data["test_completed"], "Resource": "Execution"},
        {"Task": "Validation", "Start": timing_data["validation_started"], "Finish": timing_data["validation_completed"], "Resource": "Validation"},
        {"Task": "Cleanup", "Start": timing_data["cleanup_started"], "Finish": timing_data["cleanup_completed"], "Resource": "Infrastructure"}
    ])
    
    # Convert to DataFrame for Plotly
    df = pd.DataFrame(phases)
    
    # Define colors for each resource
    resource_colors = {
        "Infrastructure": "#7986CB",  # Indigo
        "Workload": "#4DB6AC",        # Teal
        "Fault Injection": "#FF8A65", # Deep Orange
        "Execution": "#FFD54F",       # Amber
        "Validation": "#AED581",      # Light Green
    }
    
    # Create Gantt chart
    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Task",
        color="Resource",
        color_discrete_map=resource_colors,
        title=f"Test Execution Timeline (Test ID: {test_id})"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Time (seconds)",
        height=600,
        width=1000,
        paper_bgcolor=theme_config['bg_color'],
        plot_bgcolor=theme_config['bg_color'],
        font=dict(color=theme_config['text_color']),
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    # Add vertical line at failure injection if failures were included
    if timing_data["inject_failures"] is not None:
        fig.add_vline(
            x=timing_data["inject_failures"],
            line_width=2,
            line_dash="dash",
            line_color=theme_config['warning_color'],
            annotation_text="Failure Injection Started",
            annotation_position="top right"
        )
    
    # Save the figure to an HTML file
    output_file = os.path.join(output_dir, f'{test_id}_timing.html')
    plot(fig, filename=output_file, auto_open=False)
    
    return output_file

# Create failure visualization
def create_failure_visualization(report: Dict, output_dir: str, test_id: str, theme: str = 'light') -> str:
    """Create a visualization of test failures if any were injected."""
    # Get theme configuration
    theme_config = {
        'light': {
            'bg_color': '#ffffff',
            'text_color': '#333333',
            'grid_color': '#eeeeee',
            'success_color': '#28a745',
            'failure_color': '#dc3545',
            'warning_color': '#ffc107',
            'info_color': '#17a2b8'
        },
        'dark': {
            'bg_color': '#222222',
            'text_color': '#f2f2f2',
            'grid_color': '#444444',
            'success_color': '#28a745',
            'failure_color': '#dc3545',
            'warning_color': '#ffc107',
            'info_color': '#17a2b8'
        }
    }[theme]
    
    # Check if failures were included in the test
    include_failures = report.get('configuration', {}).get('include_failures', False)
    
    if not include_failures:
        # Create a simple figure indicating no failures were injected
        fig = go.Figure()
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': theme_config['success_color']},
                    'steps': [
                        {'range': [0, 40], 'color': theme_config['failure_color']},
                        {'range': [40, 70], 'color': theme_config['warning_color']},
                        {'range': [70, 100], 'color': theme_config['success_color']}
                    ]
                }
            )
        )
        
        # Add annotation
        fig.add_annotation(
            x=0.5,
            y=0.3,
            text="No failures were injected during this test",
            showarrow=False,
            font=dict(color=theme_config['text_color'], size=16),
            align="center"
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"System Health (Test ID: {test_id})",
            height=600,
            width=1000,
            paper_bgcolor=theme_config['bg_color'],
            plot_bgcolor=theme_config['bg_color'],
            font=dict(color=theme_config['text_color']),
            margin=dict(l=50, r=50, t=100, b=50),
        )
    else:
        # Simulated failure data
        # In a real scenario, this would come from actual test failure data
        failures = [
            {"type": "Worker Termination", "time": 15, "duration": 2, "recovery": "Automatic", "impact": "Medium"},
            {"type": "Malformed Request (Coordinator)", "time": 20, "duration": 1, "recovery": "Automatic", "impact": "Low"},
            {"type": "Malformed Request (Result Aggregator)", "time": 22, "duration": 1, "recovery": "Automatic", "impact": "Low"}
        ]
        
        # Create a figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Injected Failures",
                "Recovery Timeline"
            ),
            specs=[
                [{"type": "table"}],
                [{"type": "scatter"}]
            ],
            row_heights=[0.4, 0.6]
        )
        
        # Add failure table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Failure Type", "Time (s)", "Duration (s)", "Recovery", "Impact"],
                    fill_color=theme_config['info_color'],
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[
                        [f["type"] for f in failures],
                        [f["time"] for f in failures],
                        [f["duration"] for f in failures],
                        [f["recovery"] for f in failures],
                        [f["impact"] for f in failures]
                    ],
                    fill_color=theme_config['bg_color'],
                    align='left',
                    font=dict(color=theme_config['text_color'], size=11)
                )
            ),
            row=1, col=1
        )
        
        # Create recovery timeline scatter plot
        x_values = []
        y_values = []
        colors = []
        sizes = []
        hover_texts = []
        
        # System health over time (0-100%)
        # Start at 100%
        health_times = [0]
        health_values = [100]
        
        for f in failures:
            # Add failure point
            x_values.append(f["time"])
            y_values.append(0)  # Bottom of chart
            colors.append(theme_config['failure_color'])
            sizes.append(15)
            hover_texts.append(f"Failure: {f['type']}<br>Time: {f['time']}s<br>Impact: {f['impact']}")
            
            # Add recovery point
            recovery_time = f["time"] + f["duration"]
            x_values.append(recovery_time)
            y_values.append(1)  # Top of chart
            colors.append(theme_config['success_color'])
            sizes.append(15)
            hover_texts.append(f"Recovery: {f['type']}<br>Time: {recovery_time}s<br>Method: {f['recovery']}")
            
            # Add health data points
            impact_value = 30 if f["impact"] == "High" else 15 if f["impact"] == "Medium" else 5
            health_times.append(f["time"])
            health_values.append(health_values[-1] - impact_value)
            
            health_times.append(recovery_time)
            health_values.append(health_values[-1] + impact_value)
        
        # Add end point at 100% health
        health_times.append(report.get('configuration', {}).get('test_duration', 60))
        health_values.append(100)
        
        # Add system health line
        fig.add_trace(
            go.Scatter(
                x=health_times,
                y=health_values,
                mode='lines+markers',
                name='System Health',
                line=dict(color=theme_config['info_color'], width=3),
                hoverinfo='y',
            ),
            row=2, col=1
        )
        
        # Add failure and recovery markers
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=sizes,
                    symbol='circle'
                ),
                hovertext=hover_texts,
                hoverinfo='text',
                name='Events'
            ),
            row=2, col=1
        )
        
        # Update y-axis for the scatter plot to make it invisible (just need health line)
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.1, 1.1],
            row=2, col=1
        )
        
        # Add a secondary y-axis for the health values
        fig.update_layout(
            yaxis2=dict(
                title="System Health (%)",
                anchor="x",
                overlaying="y",
                side="right",
                range=[0, 110],
                showgrid=False
            )
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Fault Tolerance Testing (Test ID: {test_id})",
            height=800,
            width=1000,
            paper_bgcolor=theme_config['bg_color'],
            plot_bgcolor=theme_config['bg_color'],
            font=dict(color=theme_config['text_color']),
            margin=dict(l=50, r=50, t=100, b=50),
        )
    
    # Save the figure to an HTML file
    output_file = os.path.join(output_dir, f'{test_id}_failures.html')
    plot(fig, filename=output_file, auto_open=False)
    
    return output_file

# Function to create a unified HTML dashboard
def create_unified_dashboard(
    test_id: str,
    summary_file: str,
    component_file: str,
    timing_file: str,
    failures_file: str,
    output_dir: str,
    theme: str = 'light'
) -> str:
    """Create a unified HTML dashboard with all visualizations."""
    # Get theme configuration
    theme_config = {
        'light': {
            'bg_color': '#ffffff',
            'text_color': '#333333',
            'accent_color': '#007bff',
            'link_color': '#0056b3',
            'border_color': '#dee2e6'
        },
        'dark': {
            'bg_color': '#222222',
            'text_color': '#f2f2f2',
            'accent_color': '#0d6efd',
            'link_color': '#6ea8fe',
            'border_color': '#495057'
        }
    }[theme]
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>End-to-End Test Results Dashboard - {test_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: {theme_config['bg_color']};
            color: {theme_config['text_color']};
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background-color: {theme_config['accent_color']};
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .tab-container {{
            margin-bottom: 20px;
        }}
        .tabs {{
            display: flex;
            border-bottom: 1px solid {theme_config['border_color']};
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border: 1px solid {theme_config['border_color']};
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
            background-color: {theme_config['bg_color']};
            color: {theme_config['text_color']};
        }}
        .tab.active {{
            background-color: {theme_config['accent_color']};
            color: white;
            border-color: {theme_config['accent_color']};
        }}
        .tab-content {{
            display: none;
            padding: 20px;
            border: 1px solid {theme_config['border_color']};
            border-radius: 0 0 5px 5px;
        }}
        .tab-content.active {{
            display: block;
        }}
        iframe {{
            border: none;
            width: 100%;
            height: 800px;
        }}
        footer {{
            margin-top: 50px;
            text-align: center;
            padding: 20px;
            font-size: 14px;
            border-top: 1px solid {theme_config['border_color']};
        }}
    </style>
</head>
<body>
    <header>
        <h1>End-to-End Test Results Dashboard</h1>
        <p>Test ID: {test_id} | Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </header>
    
    <div class="container">
        <div class="tab-container">
            <div class="tabs">
                <div class="tab active" onclick="openTab(event, 'summary')">Summary</div>
                <div class="tab" onclick="openTab(event, 'components')">Component Status</div>
                <div class="tab" onclick="openTab(event, 'timing')">Test Timing</div>
                <div class="tab" onclick="openTab(event, 'failures')">Fault Tolerance</div>
            </div>
            
            <div id="summary" class="tab-content active">
                <iframe src="{os.path.basename(summary_file)}"></iframe>
            </div>
            
            <div id="components" class="tab-content">
                <iframe src="{os.path.basename(component_file)}"></iframe>
            </div>
            
            <div id="timing" class="tab-content">
                <iframe src="{os.path.basename(timing_file)}"></iframe>
            </div>
            
            <div id="failures" class="tab-content">
                <iframe src="{os.path.basename(failures_file)}"></iframe>
            </div>
        </div>
    </div>
    
    <footer>
        <p>IPFS Accelerate Python Framework - Distributed Testing Framework</p>
        <p>© {datetime.datetime.now().year} - End-to-End Testing Visualization Tool</p>
    </footer>
    
    <script>
        function openTab(evt, tabName) {{
            // Hide all tab content
            var tabcontent = document.getElementsByClassName("tab-content");
            for (var i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].classList.remove("active");
            }}
            
            // Remove active class from all tabs
            var tabs = document.getElementsByClassName("tab");
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove("active");
            }}
            
            // Show the current tab and add an "active" class to the button that opened the tab
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }}
    </script>
</body>
</html>
"""
    
    # Write HTML to file
    output_file = os.path.join(output_dir, f'{test_id}_dashboard.html')
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file

# Function to integrate with monitoring dashboard
def integrate_with_dashboard(
    test_id: str,
    summary_file: str,
    component_file: str,
    timing_file: str,
    failures_file: str,
    dashboard_url: str
) -> bool:
    """Integrate visualizations with the monitoring dashboard."""
    try:
        # Read visualization files
        with open(summary_file, 'r') as f:
            summary_content = f.read()
        
        with open(component_file, 'r') as f:
            component_content = f.read()
        
        with open(timing_file, 'r') as f:
            timing_content = f.read()
        
        with open(failures_file, 'r') as f:
            failures_content = f.read()
        
        # Prepare data to send to dashboard
        data = {
            'test_id': test_id,
            'visualizations': {
                'summary': summary_content,
                'component': component_content,
                'timing': timing_content,
                'failures': failures_content
            }
        }
        
        # Send to dashboard API
        response = requests.post(
            f"{dashboard_url}/api/e2e-test-results",
            json=data
        )
        
        if response.status_code == 200:
            print(f"Successfully integrated visualizations with dashboard at {dashboard_url}")
            return True
        else:
            print(f"Failed to integrate with dashboard: HTTP {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print(f"Error integrating with dashboard: {e}")
        return False

def main():
    """Main entry point for the visualization tool."""
    parser = argparse.ArgumentParser(description='End-to-End Test Visualization Tool')
    parser.add_argument('--report-dir', type=str, default='./e2e_test_reports',
                       help='Directory containing test reports')
    parser.add_argument('--output-dir', type=str, default='./e2e_visualizations',
                       help='Directory for generated visualizations')
    parser.add_argument('--test-id', type=str, default='latest',
                       help='Specific test ID to visualize')
    parser.add_argument('--visualization-types', type=str, default='all',
                       help='Types of visualizations to generate [summary,component,timing,failures,all]')
    parser.add_argument('--dashboard-integration', action='store_true',
                       help='Integrate visualizations with monitoring dashboard')
    parser.add_argument('--dashboard-url', type=str, default='http://localhost:8080',
                       help='Monitoring dashboard URL')
    parser.add_argument('--generate-standalone', action='store_true',
                       help='Generate standalone HTML visualizations')
    parser.add_argument('--theme', type=str, choices=['light', 'dark'], default='light',
                       help='Visualization theme [light,dark]')
    parser.add_argument('--open-browser', action='store_true',
                       help='Open the dashboard in a web browser when done')
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_output_dir(args.output_dir)
    
    # Find and load test report
    if args.test_id == 'latest':
        report_file = find_latest_report(args.report_dir)
        if not report_file:
            print(f"No test reports found in {args.report_dir}")
            return 1
        
        test_id = extract_test_id(report_file)
    else:
        test_id = args.test_id
        report_file = os.path.join(args.report_dir, f"{test_id}_results.json")
        if not os.path.exists(report_file):
            print(f"Test report for ID {test_id} not found: {report_file}")
            return 1
    
    # Load the report
    report = load_test_report(report_file)
    
    # Determine which visualizations to generate
    visualization_types = args.visualization_types.lower().split(',')
    if 'all' in visualization_types:
        visualization_types = ['summary', 'component', 'timing', 'failures']
    
    # Generate visualizations
    generated_files = {}
    
    if 'summary' in visualization_types:
        print(f"Generating summary visualization for test {test_id}...")
        summary_file = create_summary_visualization(report, args.output_dir, test_id, args.theme)
        generated_files['summary'] = summary_file
        print(f"Generated: {summary_file}")
    
    if 'component' in visualization_types:
        print(f"Generating component status visualization for test {test_id}...")
        component_file = create_component_visualization(report, args.output_dir, test_id, args.theme)
        generated_files['component'] = component_file
        print(f"Generated: {component_file}")
    
    if 'timing' in visualization_types:
        print(f"Generating timing visualization for test {test_id}...")
        timing_file = create_timing_visualization(report, args.output_dir, test_id, args.theme)
        generated_files['timing'] = timing_file
        print(f"Generated: {timing_file}")
    
    if 'failures' in visualization_types:
        print(f"Generating failure visualization for test {test_id}...")
        failures_file = create_failure_visualization(report, args.output_dir, test_id, args.theme)
        generated_files['failures'] = failures_file
        print(f"Generated: {failures_file}")
    
    # Generate unified dashboard if requested
    if args.generate_standalone and 'summary' in generated_files and 'component' in generated_files and 'timing' in generated_files and 'failures' in generated_files:
        print(f"Generating unified dashboard for test {test_id}...")
        dashboard_file = create_unified_dashboard(
            test_id,
            generated_files['summary'],
            generated_files['component'],
            generated_files['timing'],
            generated_files['failures'],
            args.output_dir,
            args.theme
        )
        print(f"Generated dashboard: {dashboard_file}")
        
        if args.open_browser:
            print(f"Opening dashboard in web browser...")
            webbrowser.open(f"file://{os.path.abspath(dashboard_file)}")
    
    # Integrate with monitoring dashboard if requested
    if args.dashboard_integration and 'summary' in generated_files and 'component' in generated_files and 'timing' in generated_files and 'failures' in generated_files:
        print(f"Integrating visualizations with monitoring dashboard at {args.dashboard_url}...")
        success = integrate_with_dashboard(
            test_id,
            generated_files['summary'],
            generated_files['component'],
            generated_files['timing'],
            generated_files['failures'],
            args.dashboard_url
        )
        
        if success:
            dashboard_url = f"{args.dashboard_url}/e2e-test-results/{test_id}"
            print(f"Visualizations integrated with dashboard. View at: {dashboard_url}")
            
            if args.open_browser:
                print(f"Opening dashboard in web browser...")
                webbrowser.open(dashboard_url)
        else:
            print("Failed to integrate with dashboard. See error logs for details.")
    
    print(f"Visualization complete. Output files in: {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())