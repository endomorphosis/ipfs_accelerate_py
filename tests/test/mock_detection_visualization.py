#!/usr/bin/env python3
"""
Visualization module for mock detection results in the testing framework.
This module creates visualizations to help understand and analyze the test environment
and dependency mocking status across various test configurations.
"""

import os
import sys
import json
import glob
import argparse
import datetime
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Constants
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
TEST_RESULTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "skills", "results")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "visualizations")

# Ensure directories exist
for directory in [RESULT_DIR, DEFAULT_OUTPUT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_mock_test_results(results_pattern: str = "mock_detection_*.json") -> List[Dict[str, Any]]:
    """
    Load mock detection test results from JSON files.
    
    Args:
        results_pattern: Glob pattern to match result files
        
    Returns:
        List of result dictionaries
    """
    results = []
    for result_file in glob.glob(os.path.join(RESULT_DIR, results_pattern)):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                # Add filename as metadata
                data['_source_file'] = os.path.basename(result_file)
                results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {result_file}: {e}")
    
    return results

def scan_test_files_for_mock_status(test_dir: str = "skills/fixed_tests") -> List[Dict[str, Any]]:
    """
    Scan test files for mock detection status indicators.
    
    Args:
        test_dir: Directory containing test files
        
    Returns:
        List of dictionaries with test file analysis
    """
    results = []
    
    for test_file in glob.glob(os.path.join(test_dir, "test_*.py")):
        try:
            with open(test_file, 'r') as f:
                content = f.read()
                
            model_name = os.path.basename(test_file).replace("test_", "").replace(".py", "")
            
            result = {
                "file": os.path.basename(test_file),
                "model": model_name,
                "has_mock_detection": "using_mocks" in content,
                "has_real_inference_check": "using_real_inference" in content,
                "has_emoji_indicators": "🚀" in content or "🔷" in content,
                "dependency_checks": {
                    "torch": "HAS_TORCH" in content,
                    "transformers": "HAS_TRANSFORMERS" in content,
                    "tokenizers": "HAS_TOKENIZERS" in content,
                    "sentencepiece": "HAS_SENTENCEPIECE" in content,
                }
            }
            
            results.append(result)
        except IOError as e:
            print(f"Error reading {test_file}: {e}")
    
    return results

def create_dependency_heatmap(scan_results: List[Dict[str, Any]], 
                             output_file: str = None,
                             title: str = "Mock Detection Implementation in Test Files") -> go.Figure:
    """
    Create a heatmap showing which test files have mock detection implemented.
    
    Args:
        scan_results: List of dictionaries with test file analysis
        output_file: Path to save the visualization
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Convert results to DataFrame
    df = pd.DataFrame(scan_results)
    
    # Explode the dependency_checks dict into separate columns
    for dep in ["torch", "transformers", "tokenizers", "sentencepiece"]:
        df[f"has_{dep}_check"] = df["dependency_checks"].apply(lambda x: x.get(dep, False))
    
    # Create core features matrix
    feature_columns = [
        "has_mock_detection", 
        "has_real_inference_check", 
        "has_emoji_indicators",
        "has_torch_check",
        "has_transformers_check",
        "has_tokenizers_check",
        "has_sentencepiece_check"
    ]
    
    # Sort by model name for better visualization
    df = df.sort_values("model")
    
    # Create heatmap
    fig = go.Figure()
    
    # Add heatmap trace
    heatmap_data = df[feature_columns].astype(int).values
    
    # Get model names for y-axis
    models = df["model"].tolist()
    
    # Create readable labels for x-axis
    feature_labels = [
        "Mock Detection", 
        "Real Inference Check", 
        "Emoji Indicators", 
        "Torch Check", 
        "Transformers Check", 
        "Tokenizers Check", 
        "SentencePiece Check"
    ]
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=feature_labels,
        y=models,
        colorscale=[[0, '#f8f9fa'], [1, '#4CAF50']],
        showscale=False,
        hovertemplate='Model: %{y}<br>Feature: %{x}<br>Implemented: %{z}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Features",
            tickangle=-45
        ),
        yaxis=dict(
            title="Model Test Files",
            autorange="reversed"  # To match matrix orientation
        ),
        height=max(600, len(models) * 20),  # Dynamic height based on number of models
        margin=dict(l=140, r=40, t=50, b=100)
    )
    
    # Save if output file is specified
    if output_file:
        fig.write_html(output_file)
        print(f"Saved dependency heatmap to {output_file}")
    
    return fig

def create_mock_detection_summary(scan_results: List[Dict[str, Any]], 
                                output_file: str = None) -> go.Figure:
    """
    Create a summary visualization of mock detection implementation status.
    
    Args:
        scan_results: List of dictionaries with test file analysis
        output_file: Path to save the visualization
        
    Returns:
        Plotly figure object
    """
    # Convert results to DataFrame
    df = pd.DataFrame(scan_results)
    
    # Create subplot with 2 rows and 1 column
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=("Implementation Rate of Mock Detection Features", 
                       "Dependency Check Implementation"),
        vertical_spacing=0.25
    )
    
    # Calculate implementation rates for core features
    feature_columns = ["has_mock_detection", "has_real_inference_check", "has_emoji_indicators"]
    feature_labels = ["Mock Detection", "Real Inference Check", "Emoji Indicators"]
    
    implementation_rates = [df[col].mean() * 100 for col in feature_columns]
    
    # Add bar chart for core features
    fig.add_trace(
        go.Bar(
            x=feature_labels,
            y=implementation_rates,
            marker_color='rgba(58, 71, 80, 0.6)',
            text=[f"{rate:.1f}%" for rate in implementation_rates],
            textposition='auto',
            hovertemplate='%{x}<br>Implementation Rate: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Calculate implementation rates for dependency checks
    dependency_columns = ["dependency_checks"]
    dependency_labels = ["torch", "transformers", "tokenizers", "sentencepiece"]
    
    # Calculate rates for each dependency check
    dependency_rates = []
    for dep in dependency_labels:
        rate = df["dependency_checks"].apply(lambda x: x.get(dep, False)).mean() * 100
        dependency_rates.append(rate)
    
    # Add bar chart for dependency checks
    fig.add_trace(
        go.Bar(
            x=dependency_labels,
            y=dependency_rates,
            marker_color='rgba(71, 58, 131, 0.6)',
            text=[f"{rate:.1f}%" for rate in dependency_rates],
            textposition='auto',
            hovertemplate='%{x}<br>Implementation Rate: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Mock Detection Implementation Summary",
        height=700,
        showlegend=False
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Implementation Rate (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Implementation Rate (%)", range=[0, 100], row=2, col=1)
    
    # Save if output file is specified
    if output_file:
        fig.write_html(output_file)
        print(f"Saved mock detection summary to {output_file}")
    
    return fig

def create_model_family_analysis(scan_results: List[Dict[str, Any]], 
                                output_file: str = None) -> go.Figure:
    """
    Create an analysis of mock detection implementation by model family.
    
    Args:
        scan_results: List of dictionaries with test file analysis
        output_file: Path to save the visualization
        
    Returns:
        Plotly figure object
    """
    # Convert results to DataFrame
    df = pd.DataFrame(scan_results)
    
    # Extract model family from model name
    def extract_family(model_name):
        if 'hf_' in model_name:
            model_name = model_name.replace('hf_', '')
        
        # Handle common model families
        for family in ['bert', 'gpt', 't5', 'vit', 'clip', 'blip', 'llama', 
                     'whisper', 'wav2vec2', 'roberta', 'bart']:
            if family in model_name.lower():
                return family
        
        # Default to "other" if no match
        return "other"
    
    df['model_family'] = df['model'].apply(extract_family)
    
    # Group by model family and calculate implementation rates
    family_stats = df.groupby('model_family').agg({
        'has_mock_detection': 'mean',
        'has_real_inference_check': 'mean',
        'has_emoji_indicators': 'mean',
        'file': 'count'
    }).reset_index()
    
    # Rename columns for clarity
    family_stats.rename(columns={
        'has_mock_detection': 'Mock Detection Rate',
        'has_real_inference_check': 'Real Inference Check Rate',
        'has_emoji_indicators': 'Emoji Indicators Rate',
        'file': 'Number of Models'
    }, inplace=True)
    
    # Sort by number of models (descending)
    family_stats = family_stats.sort_values('Number of Models', ascending=False)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each feature
    features = ['Mock Detection Rate', 'Real Inference Check Rate', 'Emoji Indicators Rate']
    colors = ['rgba(71, 58, 131, 0.8)', 'rgba(58, 131, 71, 0.8)', 'rgba(131, 58, 71, 0.8)']
    
    for i, feature in enumerate(features):
        fig.add_trace(go.Bar(
            x=family_stats['model_family'],
            y=family_stats[feature] * 100,  # Convert to percentage
            name=feature,
            marker_color=colors[i],
            text=[f"{rate*100:.1f}%" for rate in family_stats[feature]],
            textposition='auto',
            hovertemplate='%{x}<br>' + feature + ': %{y:.1f}%<extra></extra>'
        ))
    
    # Add model count as a text annotation
    for i, row in family_stats.iterrows():
        fig.add_annotation(
            x=row['model_family'],
            y=105,  # Position above the bars
            text=f"n={int(row['Number of Models'])}",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title="Mock Detection Implementation by Model Family",
        xaxis=dict(title="Model Family"),
        yaxis=dict(
            title="Implementation Rate (%)",
            range=[0, 110]  # Leave room for the count annotation
        ),
        barmode='group',
        legend=dict(
            x=0,
            y=1.1,
            orientation='h'
        ),
        height=600
    )
    
    # Save if output file is specified
    if output_file:
        fig.write_html(output_file)
        print(f"Saved model family analysis to {output_file}")
    
    return fig

def create_test_result_analysis(mock_results: List[Dict[str, Any]], 
                               output_file: str = None) -> go.Figure:
    """
    Create visualization analyzing test results with real vs. mock dependencies.
    
    Args:
        mock_results: List of mock test result dictionaries
        output_file: Path to save the visualization
        
    Returns:
        Plotly figure object
    """
    # If no results provided, try to load them
    if not mock_results:
        mock_results = load_mock_test_results()
    
    if not mock_results:
        print("No mock test results available. Run tests first.")
        return None
    
    # Prepare data for visualization
    test_data = []
    
    for result in mock_results:
        # Extract data from each test result
        for test_entry in result.get('test_results', []):
            entry = {
                'test_name': test_entry.get('test_name', 'Unknown'),
                'model_id': test_entry.get('model_id', 'Unknown'),
                'success': test_entry.get('success', False),
                'using_mocks': test_entry.get('using_mocks', True),
                'duration_ms': test_entry.get('duration_ms', 0),
                'error': test_entry.get('error', None),
                'timestamp': result.get('timestamp', ''),
                'environment': result.get('environment', {}),
            }
            test_data.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(test_data)
    
    if df.empty:
        print("No test data found in results.")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Success Rate by Mock Status",
            "Average Duration by Mock Status",
            "Success Rate by Model",
            "Test Count by Model"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar", "colspan": 2}, None]
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    # 1. Success Rate by Mock Status
    mock_success = df.groupby('using_mocks')['success'].mean() * 100
    
    mock_labels = ["Real Dependencies", "Mock Dependencies"]
    
    fig.add_trace(
        go.Bar(
            x=mock_labels,
            y=mock_success.values,
            marker_color=['rgba(58, 131, 71, 0.6)', 'rgba(71, 58, 131, 0.6)'],
            text=[f"{rate:.1f}%" for rate in mock_success.values],
            textposition='auto',
            hovertemplate='%{x}<br>Success Rate: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Average Duration by Mock Status
    # Only include successful tests for fair comparison
    success_df = df[df['success'] == True]
    mock_duration = success_df.groupby('using_mocks')['duration_ms'].mean()
    
    fig.add_trace(
        go.Bar(
            x=mock_labels,
            y=mock_duration.values,
            marker_color=['rgba(58, 131, 71, 0.6)', 'rgba(71, 58, 131, 0.6)'],
            text=[f"{dur:.0f}ms" for dur in mock_duration.values],
            textposition='auto',
            hovertemplate='%{x}<br>Avg Duration: %{y:.0f}ms<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Success Rate by Model (Top N models)
    N = 15  # Show top N models
    model_success = df.groupby('model_id')['success'].agg(['mean', 'count']).reset_index()
    model_success['success_rate'] = model_success['mean'] * 100
    
    # Sort by count (descending) to show most frequent models
    model_success = model_success.sort_values('count', ascending=False).head(N)
    
    # Truncate long model names
    model_success['display_name'] = model_success['model_id'].apply(
        lambda x: (x[:20] + '...') if len(x) > 20 else x
    )
    
    fig.add_trace(
        go.Bar(
            x=model_success['display_name'],
            y=model_success['success_rate'],
            marker_color='rgba(131, 58, 71, 0.6)',
            text=[f"{rate:.1f}%" for rate in model_success['success_rate']],
            textposition='auto',
            hovertemplate='Model: %{x}<br>Success Rate: %{y:.1f}%<br>Test Count: %{customdata}<extra></extra>',
            customdata=model_success['count']
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Mock vs. Real Dependencies Test Result Analysis",
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Duration (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 100], row=2, col=1)
    
    fig.update_xaxes(title_text="Dependency Status", row=1, col=1)
    fig.update_xaxes(title_text="Dependency Status", row=1, col=2)
    fig.update_xaxes(title_text="Model", tickangle=-45, row=2, col=1)
    
    # Save if output file is specified
    if output_file:
        fig.write_html(output_file)
        print(f"Saved test result analysis to {output_file}")
    
    return fig

def create_interactive_dashboard(scan_results: List[Dict[str, Any]],
                               mock_results: List[Dict[str, Any]] = None,
                               output_file: str = None) -> go.Figure:
    """
    Create an interactive dashboard combining all visualizations.
    
    Args:
        scan_results: List of dictionaries with test file analysis
        mock_results: List of mock test result dictionaries
        output_file: Path to save the visualization
        
    Returns:
        Plotly figure object
    """
    # Create subplots with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Implementation Rate of Mock Detection Features",
            "Mock Detection by Model Family",
            "Test Files with Mock Detection Implementation",
            "Test Result Analysis"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "bar"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Implementation Rate of Features (top left)
    df = pd.DataFrame(scan_results)
    
    # Calculate implementation rates for core features
    feature_columns = ["has_mock_detection", "has_real_inference_check", "has_emoji_indicators"]
    feature_labels = ["Mock Detection", "Real Inference Check", "Emoji Indicators"]
    
    implementation_rates = [df[col].mean() * 100 for col in feature_columns]
    
    fig.add_trace(
        go.Bar(
            x=feature_labels,
            y=implementation_rates,
            marker_color='rgba(58, 71, 80, 0.6)',
            text=[f"{rate:.1f}%" for rate in implementation_rates],
            textposition='auto',
            hovertemplate='%{x}<br>Implementation Rate: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Model Family Analysis (top right)
    # Extract model family from model name
    def extract_family(model_name):
        if 'hf_' in model_name:
            model_name = model_name.replace('hf_', '')
        
        # Handle common model families
        for family in ['bert', 'gpt', 't5', 'vit', 'clip', 'blip', 'llama', 
                     'whisper', 'wav2vec2', 'roberta', 'bart']:
            if family in model_name.lower():
                return family
        
        # Default to "other" if no match
        return "other"
    
    df['model_family'] = df['model'].apply(extract_family)
    
    # Group by model family and calculate implementation rates
    family_stats = df.groupby('model_family').agg({
        'has_mock_detection': 'mean',
        'file': 'count'
    }).reset_index()
    
    # Sort by count (descending)
    family_stats = family_stats.sort_values('file', ascending=False)
    family_stats = family_stats.head(10)  # Top 10 families
    
    fig.add_trace(
        go.Bar(
            x=family_stats['model_family'],
            y=family_stats['has_mock_detection'] * 100,
            marker_color='rgba(71, 58, 131, 0.6)',
            text=[f"{rate*100:.1f}%" for rate in family_stats['has_mock_detection']],
            textposition='auto',
            hovertemplate='%{x}<br>Implementation Rate: %{y:.1f}%<br>Models: %{customdata}<extra></extra>',
            customdata=family_stats['file']
        ),
        row=1, col=2
    )
    
    # 3. Heatmap (bottom left)
    # Select top N models for heatmap (to avoid overcrowding)
    N = 20
    
    # Sort by model name for better visualization
    df_sorted = df.sort_values("model")
    
    # Select subset of models
    df_subset = df_sorted.head(N)
    
    # Create feature matrix
    feature_columns = [
        "has_mock_detection", 
        "has_real_inference_check", 
        "has_emoji_indicators"
    ]
    
    # Create readable labels
    feature_labels = [
        "Mock Detection", 
        "Real Inference Check", 
        "Emoji Indicators"
    ]
    
    # Extract data for heatmap
    heatmap_data = df_subset[feature_columns].astype(int).values
    models = df_subset["model"].tolist()
    
    # Add heatmap trace
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=feature_labels,
            y=models,
            colorscale=[[0, '#f8f9fa'], [1, '#4CAF50']],
            showscale=False,
            hovertemplate='Model: %{y}<br>Feature: %{x}<br>Implemented: %{z}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Test Result Analysis (bottom right)
    if mock_results:
        # Prepare data for visualization
        test_data = []
        
        for result in mock_results:
            # Extract data from each test result
            for test_entry in result.get('test_results', []):
                entry = {
                    'model_id': test_entry.get('model_id', 'Unknown'),
                    'success': test_entry.get('success', False),
                    'using_mocks': test_entry.get('using_mocks', True),
                }
                test_data.append(entry)
        
        # Convert to DataFrame
        test_df = pd.DataFrame(test_data)
        
        if not test_df.empty:
            # Calculate success rates
            mock_success = test_df.groupby('using_mocks')['success'].mean() * 100
            
            mock_labels = ["Real Dependencies", "Mock Dependencies"]
            
            fig.add_trace(
                go.Bar(
                    x=mock_labels,
                    y=mock_success.values,
                    marker_color=['rgba(58, 131, 71, 0.6)', 'rgba(71, 58, 131, 0.6)'],
                    text=[f"{rate:.1f}%" for rate in mock_success.values],
                    textposition='auto',
                    hovertemplate='%{x}<br>Success Rate: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Set y-axis range for the success rate chart
            fig.update_yaxes(title_text="Success Rate (%)", range=[0, 100], row=2, col=2)
            fig.update_xaxes(title_text="Dependency Status", row=2, col=2)
    
    # Update axes and layout
    fig.update_yaxes(title_text="Implementation Rate (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Implementation Rate (%)", range=[0, 100], row=1, col=2)
    
    fig.update_xaxes(title_text="Features", row=1, col=1)
    fig.update_xaxes(title_text="Model Family", tickangle=-45, row=1, col=2)
    fig.update_xaxes(title_text="Features", row=2, col=1)
    
    fig.update_yaxes(title_text="Models", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title="Mock Detection Implementation Dashboard",
        height=1000,
        showlegend=False,
        margin=dict(l=150, r=40, t=100, b=100)
    )
    
    # Save if output file is specified
    if output_file:
        fig.write_html(output_file)
        print(f"Saved interactive dashboard to {output_file}")
    
    return fig

def generate_report(scan_results: List[Dict[str, Any]], 
                  mock_results: List[Dict[str, Any]] = None,
                  output_file: str = None) -> str:
    """
    Generate a markdown report summarizing mock detection status.
    
    Args:
        scan_results: List of dictionaries with test file analysis
        mock_results: List of mock test result dictionaries
        output_file: Path to save the report
        
    Returns:
        Markdown formatted report text
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(scan_results)
    
    # Calculate implementation statistics
    total_files = len(df)
    with_mock_detection = df['has_mock_detection'].sum()
    with_real_inference = df['has_real_inference_check'].sum()
    with_emoji = df['has_emoji_indicators'].sum()
    
    # Create model family analysis
    def extract_family(model_name):
        if 'hf_' in model_name:
            model_name = model_name.replace('hf_', '')
        
        for family in ['bert', 'gpt', 't5', 'vit', 'clip', 'blip', 'llama', 
                     'whisper', 'wav2vec2', 'roberta', 'bart']:
            if family in model_name.lower():
                return family
        
        return "other"
    
    df['model_family'] = df['model'].apply(extract_family)
    family_stats = df.groupby('model_family').agg({
        'has_mock_detection': 'mean',
        'file': 'count'
    }).reset_index()
    
    family_stats = family_stats.sort_values('file', ascending=False)
    
    # Generate report text
    report = f"""# Mock Detection Implementation Report

## Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Test Files Analyzed**: {total_files}
- **Files with Mock Detection**: {with_mock_detection} ({with_mock_detection/total_files*100:.1f}%)
- **Files with Real Inference Check**: {with_real_inference} ({with_real_inference/total_files*100:.1f}%)
- **Files with Emoji Indicators**: {with_emoji} ({with_emoji/total_files*100:.1f}%)

## Implementation by Model Family

| Model Family | Test Files | Implementation Rate |
|--------------|------------|---------------------|
"""
    
    # Add rows for each model family
    for _, row in family_stats.iterrows():
        family = row['model_family']
        count = int(row['file'])
        rate = row['has_mock_detection'] * 100
        report += f"| {family} | {count} | {rate:.1f}% |\n"
    
    # Add mock test results if available
    if mock_results:
        test_data = []
        
        for result in mock_results:
            for test_entry in result.get('test_results', []):
                entry = {
                    'model_id': test_entry.get('model_id', 'Unknown'),
                    'success': test_entry.get('success', False),
                    'using_mocks': test_entry.get('using_mocks', True),
                }
                test_data.append(entry)
        
        test_df = pd.DataFrame(test_data)
        
        if not test_df.empty:
            # Calculate success rates
            mock_success = test_df.groupby('using_mocks')['success'].mean() * 100
            real_success_rate = mock_success.get(False, 0)
            mock_success_rate = mock_success.get(True, 0)
            
            # Count tests
            real_tests = test_df[test_df['using_mocks'] == False].shape[0]
            mock_tests = test_df[test_df['using_mocks'] == True].shape[0]
            
            report += f"""
## Test Results Analysis

| Dependency Type | Tests Run | Success Rate |
|-----------------|-----------|--------------|
| Real Dependencies | {real_tests} | {real_success_rate:.1f}% |
| Mock Dependencies | {mock_tests} | {mock_success_rate:.1f}% |

"""
    
    # Add recommendations
    report += """
## Recommendations

1. **Implementation Consistency**: Ensure all test files implement mock detection consistently.
2. **Dependency Checking**: Add checks for all relevant dependencies in each test file.
3. **Emoji Indicators**: Use standard emoji indicators (🚀/🔷) for better visual distinction.
4. **Test Coverage**: Continue to expand test coverage with both real and mock dependency testing.
5. **Documentation**: Keep mock detection documentation updated.

## Next Steps

1. Create automation to add mock detection to all test files.
2. Integrate mock detection status into test reports.
3. Improve reporting by adding model-specific mock behavior details.
4. Enhance visualization integration with the main dashboard.
"""
    
    # Save if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Saved report to {output_file}")
    
    return report

def main():
    """Main function to parse arguments and generate visualizations."""
    parser = argparse.ArgumentParser(description="Generate visualizations for mock detection results")
    parser.add_argument('--test-dir', type=str, default="skills/fixed_tests",
                      help="Directory containing test files to analyze")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                      help="Directory to save visualization outputs")
    parser.add_argument('--results-pattern', type=str, default="mock_detection_*.json",
                      help="Pattern to match mock detection result files")
    parser.add_argument('--no-heatmap', action='store_true',
                      help="Skip generating the dependency heatmap")
    parser.add_argument('--no-summary', action='store_true',
                      help="Skip generating the mock detection summary")
    parser.add_argument('--no-family', action='store_true',
                      help="Skip generating the model family analysis")
    parser.add_argument('--no-test-results', action='store_true',
                      help="Skip generating the test result analysis")
    parser.add_argument('--no-dashboard', action='store_true',
                      help="Skip generating the interactive dashboard")
    parser.add_argument('--no-report', action='store_true',
                      help="Skip generating the markdown report")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Scan test files
    print(f"Scanning test files in {args.test_dir}...")
    scan_results = scan_test_files_for_mock_status(args.test_dir)
    print(f"Found {len(scan_results)} test files.")
    
    # Load mock test results if available
    mock_results = load_mock_test_results(args.results_pattern)
    print(f"Loaded {len(mock_results)} mock test result files.")
    
    # Generate visualizations
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if not args.no_heatmap:
        output_file = os.path.join(args.output_dir, f"mock_detection_heatmap_{timestamp}.html")
        create_dependency_heatmap(scan_results, output_file)
    
    if not args.no_summary:
        output_file = os.path.join(args.output_dir, f"mock_detection_summary_{timestamp}.html")
        create_mock_detection_summary(scan_results, output_file)
    
    if not args.no_family:
        output_file = os.path.join(args.output_dir, f"mock_detection_family_{timestamp}.html")
        create_model_family_analysis(scan_results, output_file)
    
    if not args.no_test_results and mock_results:
        output_file = os.path.join(args.output_dir, f"mock_detection_test_results_{timestamp}.html")
        create_test_result_analysis(mock_results, output_file)
    
    if not args.no_dashboard:
        output_file = os.path.join(args.output_dir, f"mock_detection_dashboard_{timestamp}.html")
        create_interactive_dashboard(scan_results, mock_results, output_file)
    
    if not args.no_report:
        output_file = os.path.join(args.output_dir, f"mock_detection_report_{timestamp}.md")
        generate_report(scan_results, mock_results, output_file)
    
    print(f"All visualizations have been saved to {args.output_dir}")

if __name__ == "__main__":
    main()