#!/usr/bin/env python3
"""
Performance Analytics for End-to-End Testing Framework

This module provides comprehensive performance analytics and reporting for the
end-to-end testing framework, including statistical analysis, trend detection,
and automated performance regression detection.

Usage:
    python -m duckdb_api.distributed_testing.tests.performance_analytics [options]

Options:
    --report-dir DIR               Directory containing test reports (default: ./e2e_test_reports)
    --output-dir DIR               Directory for generated reports (default: ./e2e_performance_reports)
    --db-path PATH                 Path to the DuckDB database (default: ./benchmark_db.duckdb)
    --time-range DAYS              Time range in days for analysis (default: 30)
    --baseline-days DAYS           Days to use for baseline (default: 7)
    --comparison-days DAYS         Days to use for comparison (default: 1)
    --metrics LIST                 Metrics to analyze [latency,throughput,memory,cpu,all] (default: all)
    --models LIST                  Models to include in analysis (default: all)
    --hardware-types LIST          Hardware types to include [cpu,gpu,webgpu,webnn,multi,all] (default: all)
    --regression-threshold FLOAT   Threshold for regression detection (default: 0.1)
    --generate-report              Generate HTML report
    --upload-to-dashboard          Upload results to dashboard
    --dashboard-url URL            URL of monitoring dashboard (default: http://localhost:8082)
    --debug                        Enable debug logging
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path to ensure imports work properly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Try to import DuckDB
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("Warning: DuckDB not available. Database functionality will be limited.")

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """Performance analytics for the end-to-end testing framework."""
    
    def __init__(
        self, 
        report_dir: str = "./e2e_test_reports",
        output_dir: str = "./e2e_performance_reports",
        db_path: str = "./benchmark_db.duckdb",
        time_range_days: int = 30,
        baseline_days: int = 7,
        comparison_days: int = 1,
        debug: bool = False
    ):
        """Initialize the performance analytics.
        
        Args:
            report_dir: Directory containing test reports
            output_dir: Directory for generated reports
            db_path: Path to the DuckDB database
            time_range_days: Time range in days for analysis
            baseline_days: Days to use for baseline
            comparison_days: Days to use for comparison
            debug: Enable debug logging
        """
        self.report_dir = Path(report_dir)
        self.output_dir = Path(output_dir)
        self.db_path = db_path
        self.time_range_days = time_range_days
        self.baseline_days = baseline_days
        self.comparison_days = comparison_days
        self.debug = debug
        
        # Create directories if they don't exist
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB connection if available
        self.db_conn = None
        if DUCKDB_AVAILABLE and os.path.exists(self.db_path):
            try:
                self.db_conn = duckdb.connect(self.db_path)
                logger.info(f"Connected to DuckDB database at {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to connect to DuckDB database: {e}")
                self.db_conn = None
        
        # Test data cache
        self.test_data = {}
        self.performance_data = None
        self.regression_analysis = None
        
        logger.info(f"Initialized performance analytics with {time_range_days} day time range")
        logger.info(f"Baseline: {baseline_days} days, Comparison: {comparison_days} days")
    
    def load_test_reports(self, max_reports: int = 1000) -> Dict[str, Any]:
        """Load test reports from the report directory.
        
        Args:
            max_reports: Maximum number of reports to load
            
        Returns:
            Dictionary of test reports by ID
        """
        logger.info(f"Loading test reports from {self.report_dir}")
        
        # Find all report files
        report_files = list(self.report_dir.glob("*_results.json"))
        
        # Sort by modification time, newest first
        report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Limit the number of reports
        report_files = report_files[:max_reports]
        
        # Load reports
        reports = {}
        for report_file in report_files:
            try:
                # Extract test ID from filename
                test_id = report_file.stem.replace("_results", "")
                
                # Load report
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                # Store report
                reports[test_id] = report_data
                
                if self.debug:
                    logger.debug(f"Loaded report {test_id}")
            except Exception as e:
                logger.error(f"Failed to load report {report_file}: {e}")
        
        logger.info(f"Loaded {len(reports)} test reports")
        return reports
    
    def load_performance_data_from_db(self, 
        time_range_days: Optional[int] = None,
        models: Optional[List[str]] = None,
        hardware_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load performance data from the DuckDB database.
        
        Args:
            time_range_days: Time range in days (None to use default)
            models: List of models to include (None for all)
            hardware_types: List of hardware types to include (None for all)
            
        Returns:
            DataFrame with performance data
        """
        if not DUCKDB_AVAILABLE or not self.db_conn:
            logger.warning("DuckDB not available or not connected")
            return pd.DataFrame()
        
        # Use specified time range or default
        days = time_range_days if time_range_days is not None else self.time_range_days
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Loading performance data from {start_date} to {end_date}")
        
        # Build query
        query = """
            SELECT 
                test_id,
                model_name, 
                hardware_type,
                test_timestamp,
                latency_ms,
                throughput_items_sec,
                memory_usage_mb,
                cpu_usage_percent,
                test_duration_sec,
                batch_size,
                precision_type,
                additional_info
            FROM performance_results
            WHERE test_timestamp BETWEEN ? AND ?
        """
        
        params = [start_date, end_date]
        
        # Add model filter if specified
        if models and len(models) > 0 and 'all' not in models:
            model_list = ', '.join([f"'{model}'" for model in models])
            query += f" AND model_name IN ({model_list})"
        
        # Add hardware type filter if specified
        if hardware_types and len(hardware_types) > 0 and 'all' not in hardware_types:
            hw_list = ', '.join([f"'{hw}'" for hw in hardware_types])
            query += f" AND hardware_type IN ({hw_list})"
        
        # Order by timestamp
        query += " ORDER BY test_timestamp DESC"
        
        try:
            # Execute query
            result = self.db_conn.execute(query, params)
            
            # Convert to DataFrame
            df = result.fetch_df()
            
            logger.info(f"Loaded {len(df)} performance records")
            return df
        except Exception as e:
            logger.error(f"Failed to load performance data from database: {e}")
            return pd.DataFrame()
    
    def analyze_performance_trends(self, 
        df: Optional[pd.DataFrame] = None, 
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze performance trends from the data.
        
        Args:
            df: DataFrame with performance data (None to use cached)
            metrics: List of metrics to analyze (None for all)
            
        Returns:
            Dictionary with trend analysis
        """
        # Use provided DataFrame or load from database
        if df is None:
            if self.performance_data is None:
                self.performance_data = self.load_performance_data_from_db()
            
            df = self.performance_data
        
        if df.empty:
            logger.warning("No performance data available for trend analysis")
            return {}
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ['latency_ms', 'throughput_items_sec', 'memory_usage_mb', 'cpu_usage_percent']
        
        logger.info(f"Analyzing performance trends for metrics: {metrics}")
        
        # Ensure test_timestamp is datetime
        if 'test_timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['test_timestamp']):
            df['test_timestamp'] = pd.to_datetime(df['test_timestamp'])
        
        # Group by model and hardware type
        grouped = df.groupby(['model_name', 'hardware_type'])
        
        # Analyze trends for each group
        trends = {}
        for (model, hw), group_df in grouped:
            # Sort by timestamp
            group_df = group_df.sort_values('test_timestamp')
            
            # Skip groups with too few data points
            if len(group_df) < 3:
                continue
            
            # Analyze each metric
            model_hw_key = f"{model}_{hw}"
            trends[model_hw_key] = {
                'model': model,
                'hardware_type': hw,
                'metrics': {}
            }
            
            for metric in metrics:
                if metric not in group_df.columns:
                    continue
                
                # Get metric values
                values = group_df[metric].values
                
                # Skip metrics with no data
                if len(values) == 0 or np.isnan(values).all():
                    continue
                
                # Calculate basic statistics
                stats_data = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': int(len(values))
                }
                
                # Calculate trend (linear regression)
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                # Determine trend direction
                if slope > 0.01:
                    trend_direction = 'increasing'
                elif slope < -0.01:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
                
                # Determine if this is good or bad based on metric
                is_good = (
                    (metric == 'throughput_items_sec' and trend_direction == 'increasing') or
                    ((metric in ['latency_ms', 'memory_usage_mb', 'cpu_usage_percent']) and trend_direction == 'decreasing')
                )
                
                # Add trend analysis
                stats_data.update({
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'trend_direction': trend_direction,
                    'is_good_trend': is_good
                })
                
                # Store results
                trends[model_hw_key]['metrics'][metric] = stats_data
        
        logger.info(f"Analyzed trends for {len(trends)} model-hardware combinations")
        return trends
    
    def detect_performance_regressions(self,
        df: Optional[pd.DataFrame] = None,
        regression_threshold: float = 0.1,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Detect performance regressions by comparing recent data to baseline.
        
        Args:
            df: DataFrame with performance data (None to use cached)
            regression_threshold: Threshold for regression detection (0.1 = 10%)
            metrics: List of metrics to analyze (None for all)
            
        Returns:
            Dictionary with regression analysis
        """
        # Use provided DataFrame or load from database
        if df is None:
            if self.performance_data is None:
                self.performance_data = self.load_performance_data_from_db()
            
            df = self.performance_data
        
        if df.empty:
            logger.warning("No performance data available for regression detection")
            return {}
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ['latency_ms', 'throughput_items_sec', 'memory_usage_mb', 'cpu_usage_percent']
        
        logger.info(f"Detecting performance regressions with threshold {regression_threshold}")
        
        # Ensure test_timestamp is datetime
        if 'test_timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['test_timestamp']):
            df['test_timestamp'] = pd.to_datetime(df['test_timestamp'])
        
        # Calculate date ranges
        now = datetime.now()
        baseline_start = now - timedelta(days=self.baseline_days + self.comparison_days)
        baseline_end = now - timedelta(days=self.comparison_days)
        comparison_start = baseline_end
        comparison_end = now
        
        logger.info(f"Baseline period: {baseline_start} to {baseline_end}")
        logger.info(f"Comparison period: {comparison_start} to {comparison_end}")
        
        # Filter data for baseline and comparison periods
        baseline_df = df[(df['test_timestamp'] >= baseline_start) & (df['test_timestamp'] < baseline_end)]
        comparison_df = df[(df['test_timestamp'] >= comparison_start) & (df['test_timestamp'] <= comparison_end)]
        
        if baseline_df.empty or comparison_df.empty:
            logger.warning("Insufficient data for baseline or comparison period")
            return {}
        
        # Group by model and hardware type
        baseline_grouped = baseline_df.groupby(['model_name', 'hardware_type'])
        comparison_grouped = comparison_df.groupby(['model_name', 'hardware_type'])
        
        # Analyze regressions for each group
        regressions = {}
        for (model, hw), baseline_group in baseline_grouped:
            # Skip if no comparison data
            if (model, hw) not in comparison_grouped.groups:
                continue
            
            comparison_group = comparison_grouped.get_group((model, hw))
            
            # Skip groups with too few data points
            if len(baseline_group) < 3 or len(comparison_group) < 3:
                continue
            
            # Analyze each metric
            model_hw_key = f"{model}_{hw}"
            regressions[model_hw_key] = {
                'model': model,
                'hardware_type': hw,
                'metrics': {}
            }
            
            for metric in metrics:
                if metric not in baseline_group.columns or metric not in comparison_group.columns:
                    continue
                
                # Get metric values
                baseline_values = baseline_group[metric].values
                comparison_values = comparison_group[metric].values
                
                # Skip metrics with no data
                if (len(baseline_values) == 0 or np.isnan(baseline_values).all() or
                    len(comparison_values) == 0 or np.isnan(comparison_values).all()):
                    continue
                
                # Calculate statistics
                baseline_mean = float(np.mean(baseline_values))
                comparison_mean = float(np.mean(comparison_values))
                
                # Calculate change
                if baseline_mean == 0:
                    # Avoid division by zero
                    change_ratio = 0 if comparison_mean == 0 else 1
                else:
                    change_ratio = (comparison_mean - baseline_mean) / baseline_mean
                
                # Determine if this is a regression based on metric
                is_regression = (
                    (metric == 'throughput_items_sec' and change_ratio < -regression_threshold) or
                    ((metric in ['latency_ms', 'memory_usage_mb', 'cpu_usage_percent']) and change_ratio > regression_threshold)
                )
                
                # Add regression analysis
                regressions[model_hw_key]['metrics'][metric] = {
                    'baseline_mean': baseline_mean,
                    'comparison_mean': comparison_mean,
                    'change_ratio': float(change_ratio),
                    'change_percent': float(change_ratio * 100),
                    'is_regression': is_regression,
                    'baseline_count': int(len(baseline_values)),
                    'comparison_count': int(len(comparison_values)),
                    'baseline_std': float(np.std(baseline_values)),
                    'comparison_std': float(np.std(comparison_values))
                }
        
        # Cache regression analysis
        self.regression_analysis = regressions
        
        # Count total regressions
        total_regressions = sum(
            1 for model_data in regressions.values()
            for metric_data in model_data['metrics'].values()
            if metric_data['is_regression']
        )
        
        logger.info(f"Detected {total_regressions} regressions across {len(regressions)} model-hardware combinations")
        return regressions
    
    def create_trend_visualization(self, 
        trends: Dict[str, Any] = None,
        top_n: int = 5,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Create visualizations for performance trends.
        
        Args:
            trends: Dictionary with trend analysis (None to use cached)
            top_n: Number of top models to include
            metrics: List of metrics to visualize (None for all)
            
        Returns:
            Dictionary with visualizations
        """
        # Use provided trends or analyze
        if trends is None:
            if self.performance_data is None:
                self.performance_data = self.load_performance_data_from_db()
            
            trends = self.analyze_performance_trends(self.performance_data, metrics)
        
        if not trends:
            logger.warning("No trend data available for visualization")
            return {}
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ['latency_ms', 'throughput_items_sec', 'memory_usage_mb', 'cpu_usage_percent']
        
        logger.info(f"Creating trend visualizations for top {top_n} models")
        
        # Map metric names to display names
        metric_display = {
            'latency_ms': 'Latency (ms)',
            'throughput_items_sec': 'Throughput (items/sec)',
            'memory_usage_mb': 'Memory Usage (MB)',
            'cpu_usage_percent': 'CPU Usage (%)'
        }
        
        # Prepare data for trending metrics
        trending_metrics = {}
        for metric in metrics:
            # Collect data for this metric
            metric_data = []
            for model_hw, data in trends.items():
                if metric in data['metrics']:
                    metric_info = data['metrics'][metric]
                    metric_data.append({
                        'model': data['model'],
                        'hardware_type': data['hardware_type'],
                        'model_hw': model_hw,
                        'mean': metric_info['mean'],
                        'slope': metric_info['slope'],
                        'r_squared': metric_info['r_squared'],
                        'trend_direction': metric_info['trend_direction'],
                        'is_good_trend': metric_info['is_good_trend']
                    })
            
            # Skip empty metrics
            if not metric_data:
                continue
            
            # Convert to DataFrame
            metric_df = pd.DataFrame(metric_data)
            
            # Get top models by absolute slope
            metric_df['abs_slope'] = metric_df['slope'].abs()
            top_models = metric_df.nlargest(top_n, 'abs_slope')
            
            # Create figure
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    f"Top {top_n} Models by Trend Magnitude",
                    "Trend Direction Distribution"
                ),
                specs=[[{"type": "bar"}, {"type": "pie"}]]
            )
            
            # Add bar chart for top models
            fig.add_trace(
                go.Bar(
                    y=[f"{row['model']} ({row['hardware_type']})" for _, row in top_models.iterrows()],
                    x=top_models['slope'],
                    orientation='h',
                    marker_color=[
                        'green' if is_good else 'red'
                        for is_good in top_models['is_good_trend']
                    ],
                    name="Trend Slope",
                    text=[
                        f"{slope:.4f} ({direction})"
                        for slope, direction in zip(top_models['slope'], top_models['trend_direction'])
                    ],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Add pie chart for trend directions
            trend_counts = metric_df['trend_direction'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=trend_counts.index,
                    values=trend_counts.values,
                    marker_colors={
                        'increasing': 'green' if metric == 'throughput_items_sec' else 'red',
                        'decreasing': 'red' if metric == 'throughput_items_sec' else 'green',
                        'stable': 'gray'
                    },
                    textinfo='percent+label'
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text=f"Performance Trends: {metric_display.get(metric, metric)}",
                height=500,
                width=1000,
                showlegend=False
            )
            
            # Update axes
            fig.update_xaxes(title_text="Trend Slope", row=1, col=1)
            
            # Store visualization
            trending_metrics[metric] = {
                'figure': fig,
                'top_models': top_models.to_dict('records'),
                'trend_distribution': {
                    k: int(v) for k, v in trend_counts.items()
                }
            }
        
        logger.info(f"Created trend visualizations for {len(trending_metrics)} metrics")
        return trending_metrics
    
    def create_regression_visualization(self,
        regressions: Dict[str, Any] = None,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Create visualizations for performance regressions.
        
        Args:
            regressions: Dictionary with regression analysis (None to use cached)
            metrics: List of metrics to visualize (None for all)
            
        Returns:
            Dictionary with visualizations
        """
        # Use provided regressions or detect
        if regressions is None:
            if self.regression_analysis is None:
                if self.performance_data is None:
                    self.performance_data = self.load_performance_data_from_db()
                
                self.regression_analysis = self.detect_performance_regressions(self.performance_data)
            
            regressions = self.regression_analysis
        
        if not regressions:
            logger.warning("No regression data available for visualization")
            return {}
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ['latency_ms', 'throughput_items_sec', 'memory_usage_mb', 'cpu_usage_percent']
        
        logger.info("Creating regression visualizations")
        
        # Map metric names to display names
        metric_display = {
            'latency_ms': 'Latency (ms)',
            'throughput_items_sec': 'Throughput (items/sec)',
            'memory_usage_mb': 'Memory Usage (MB)',
            'cpu_usage_percent': 'CPU Usage (%)'
        }
        
        # Prepare data for regression visualizations
        regression_visualizations = {}
        for metric in metrics:
            # Collect data for this metric
            metric_data = []
            for model_hw, data in regressions.items():
                if metric in data['metrics']:
                    metric_info = data['metrics'][metric]
                    if metric_info['is_regression']:
                        metric_data.append({
                            'model': data['model'],
                            'hardware_type': data['hardware_type'],
                            'model_hw': model_hw,
                            'baseline_mean': metric_info['baseline_mean'],
                            'comparison_mean': metric_info['comparison_mean'],
                            'change_percent': metric_info['change_percent'],
                            'baseline_count': metric_info['baseline_count'],
                            'comparison_count': metric_info['comparison_count']
                        })
            
            # Skip metrics with no regressions
            if not metric_data:
                continue
            
            # Convert to DataFrame
            metric_df = pd.DataFrame(metric_data)
            
            # Sort by change percentage (absolute value)
            metric_df['abs_change'] = metric_df['change_percent'].abs()
            metric_df = metric_df.sort_values('abs_change', ascending=False)
            
            # Create figure
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=[f"Performance Regressions: {metric_display.get(metric, metric)}"]
            )
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    y=[f"{row['model']} ({row['hardware_type']})" for _, row in metric_df.iterrows()],
                    x=metric_df['change_percent'],
                    orientation='h',
                    marker_color='red',
                    name="Change %",
                    text=[
                        f"{pct:.1f}% (Baseline: {baseline:.2f}, Current: {current:.2f})"
                        for pct, baseline, current in zip(
                            metric_df['change_percent'], 
                            metric_df['baseline_mean'], 
                            metric_df['comparison_mean']
                        )
                    ],
                    textposition='auto'
                )
            )
            
            # Update layout
            fig.update_layout(
                title_text=f"Performance Regressions: {metric_display.get(metric, metric)}",
                height=500,
                width=1000,
                showlegend=False,
                margin=dict(l=200)  # Extra margin for model names
            )
            
            # Update axes
            fig.update_xaxes(title_text="Change (%)")
            
            # Store visualization
            regression_visualizations[metric] = {
                'figure': fig,
                'regression_count': len(metric_df),
                'max_regression': float(metric_df['abs_change'].max()) if len(metric_df) > 0 else 0,
                'affected_models': metric_df.to_dict('records')
            }
        
        logger.info(f"Created regression visualizations for {len(regression_visualizations)} metrics")
        return regression_visualizations
    
    def create_html_report(self, 
        trend_visualizations: Dict[str, Any] = None,
        regression_visualizations: Dict[str, Any] = None,
        report_title: str = "Performance Analytics Report"
    ) -> str:
        """Create HTML report with visualizations.
        
        Args:
            trend_visualizations: Dictionary with trend visualizations (None to create)
            regression_visualizations: Dictionary with regression visualizations (None to create)
            report_title: Title for the report
            
        Returns:
            Path to the generated HTML file
        """
        # Create visualizations if not provided
        if trend_visualizations is None:
            trend_visualizations = self.create_trend_visualization()
        
        if regression_visualizations is None:
            regression_visualizations = self.create_regression_visualization()
        
        # Generate timestamp for the report
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename
        
        logger.info(f"Creating HTML report: {report_path}")
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #4a5568;
                    color: white;
                    padding: 20px;
                    text-align: center;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .section-title {{
                    border-bottom: 2px solid #4a5568;
                    padding-bottom: 10px;
                    color: #2d3748;
                    margin-top: 30px;
                }}
                .viz-container {{
                    margin: 20px 0;
                    padding: 15px;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    background-color: white;
                }}
                .summary-card {{
                    background-color: #edf2f7;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .metric-title {{
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #4a5568;
                }}
                .regression-alert {{
                    background-color: #fed7d7;
                    border-left: 4px solid #f56565;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                }}
                .good-trend {{
                    background-color: #c6f6d5;
                    border-left: 4px solid #48bb78;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                }}
                .model-list {{
                    list-style-type: none;
                    padding: 0;
                }}
                .model-list li {{
                    padding: 8px;
                    border-bottom: 1px solid #e2e8f0;
                }}
                .footer {{
                    margin-top: 50px;
                    padding: 20px;
                    text-align: center;
                    font-size: 0.9em;
                    color: #a0aec0;
                    border-top: 1px solid #e2e8f0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_title}</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2 class="section-title">Executive Summary</h2>
                    <div class="summary-card">
        """
        
        # Add regression summary
        total_regressions = sum(
            v.get('regression_count', 0)
            for v in regression_visualizations.values()
        )
        
        if total_regressions > 0:
            html_content += f"""
                        <div class="regression-alert">
                            <h3>⚠️ Performance Regressions Detected</h3>
                            <p>There are <strong>{total_regressions}</strong> performance regressions detected across {len(regression_visualizations)} metrics.</p>
                            <p>The most affected metrics are:</p>
                            <ul>
            """
            
            # Add top affected metrics
            for metric, data in regression_visualizations.items():
                metric_display = {
                    'latency_ms': 'Latency',
                    'throughput_items_sec': 'Throughput',
                    'memory_usage_mb': 'Memory Usage',
                    'cpu_usage_percent': 'CPU Usage'
                }.get(metric, metric)
                
                html_content += f"""
                                <li><strong>{metric_display}</strong>: {data['regression_count']} regressions (up to {data['max_regression']:.1f}% change)</li>
                """
            
            html_content += """
                            </ul>
                        </div>
            """
        else:
            html_content += """
                        <div class="good-trend">
                            <h3>✅ No Performance Regressions</h3>
                            <p>No significant performance regressions were detected.</p>
                        </div>
            """
        
        # Add trend summary
        good_trends = sum(
            1 for v in trend_visualizations.values()
            for model in v.get('top_models', [])
            if model.get('is_good_trend', False)
        )
        
        if good_trends > 0:
            html_content += f"""
                        <div class="good-trend">
                            <h3>📈 Positive Performance Trends</h3>
                            <p>{good_trends} metrics show positive performance trends.</p>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
        """
        
        # Add regression visualizations section
        if regression_visualizations:
            html_content += """
                <div class="section">
                    <h2 class="section-title">Performance Regressions</h2>
            """
            
            for metric, data in regression_visualizations.items():
                metric_display = {
                    'latency_ms': 'Latency (ms)',
                    'throughput_items_sec': 'Throughput (items/sec)',
                    'memory_usage_mb': 'Memory Usage (MB)',
                    'cpu_usage_percent': 'CPU Usage (%)'
                }.get(metric, metric)
                
                if data['regression_count'] > 0:
                    html_content += f"""
                        <div class="viz-container">
                            <div class="metric-title">{metric_display} Regressions</div>
                            <div id="regression-{metric}" class="plotly-chart"></div>
                            <script>
                                var data = {data['figure'].to_json()};
                                Plotly.newPlot('regression-{metric}', data.data, data.layout);
                            </script>
                            
                            <div class="regression-details">
                                <h4>Affected Models:</h4>
                                <ul class="model-list">
                    """
                    
                    # Add affected models
                    for model in data['affected_models']:
                        html_content += f"""
                                    <li>
                                        <strong>{model['model']} ({model['hardware_type']})</strong>: 
                                        {model['change_percent']:.1f}% change
                                        (Baseline: {model['baseline_mean']:.2f}, Current: {model['comparison_mean']:.2f})
                                    </li>
                        """
                    
                    html_content += """
                                </ul>
                            </div>
                        </div>
                    """
            
            html_content += """
                </div>
            """
        
        # Add trend visualizations section
        if trend_visualizations:
            html_content += """
                <div class="section">
                    <h2 class="section-title">Performance Trends</h2>
            """
            
            for metric, data in trend_visualizations.items():
                metric_display = {
                    'latency_ms': 'Latency (ms)',
                    'throughput_items_sec': 'Throughput (items/sec)',
                    'memory_usage_mb': 'Memory Usage (MB)',
                    'cpu_usage_percent': 'CPU Usage (%)'
                }.get(metric, metric)
                
                html_content += f"""
                    <div class="viz-container">
                        <div class="metric-title">{metric_display} Trends</div>
                        <div id="trend-{metric}" class="plotly-chart"></div>
                        <script>
                            var data = {data['figure'].to_json()};
                            Plotly.newPlot('trend-{metric}', data.data, data.layout);
                        </script>
                        
                        <div class="trend-details">
                            <h4>Top Models by Trend Magnitude:</h4>
                            <ul class="model-list">
                """
                
                # Add top models
                for model in data['top_models']:
                    trend_class = "good-trend" if model['is_good_trend'] else "regression-alert"
                    trend_icon = "📈" if model['is_good_trend'] else "📉"
                    
                    html_content += f"""
                                <li class="{trend_class}">
                                    {trend_icon} <strong>{model['model']} ({model['hardware_type']})</strong>: 
                                    Slope: {model['slope']:.4f}, Direction: {model['trend_direction']}
                                </li>
                    """
                
                html_content += """
                            </ul>
                        </div>
                    </div>
                """
            
            html_content += """
                </div>
            """
        
        # Add footer
        html_content += f"""
                <div class="footer">
                    <p>Performance Analytics Report - {timestamp}</p>
                    <p>Generated by Distributed Testing Framework</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created HTML report: {report_path}")
        return str(report_path)
    
    async def upload_to_dashboard(self,
        dashboard_url: str,
        trend_visualizations: Dict[str, Any] = None,
        regression_visualizations: Dict[str, Any] = None
    ) -> bool:
        """Upload performance analytics to the monitoring dashboard.
        
        Args:
            dashboard_url: URL of the monitoring dashboard
            trend_visualizations: Dictionary with trend visualizations (None to create)
            regression_visualizations: Dictionary with regression visualizations (None to create)
            
        Returns:
            Success status
        """
        # Create visualizations if not provided
        if trend_visualizations is None:
            trend_visualizations = self.create_trend_visualization()
        
        if regression_visualizations is None:
            regression_visualizations = self.create_regression_visualization()
        
        logger.info(f"Uploading performance analytics to dashboard: {dashboard_url}")
        
        # Prepare data for upload
        upload_data = {
            'type': 'performance_analytics',
            'timestamp': datetime.now().isoformat(),
            'trend_visualizations': {},
            'regression_visualizations': {},
            'summary': {
                'total_regressions': sum(
                    v.get('regression_count', 0)
                    for v in regression_visualizations.values()
                ),
                'positive_trends': sum(
                    1 for v in trend_visualizations.values()
                    for model in v.get('top_models', [])
                    if model.get('is_good_trend', False)
                )
            }
        }
        
        # Convert plotly figures to HTML
        for metric, data in trend_visualizations.items():
            upload_data['trend_visualizations'][metric] = {
                'figure_html': data['figure'].to_html(include_plotlyjs=False, full_html=False),
                'top_models': data['top_models'],
                'trend_distribution': data['trend_distribution']
            }
        
        for metric, data in regression_visualizations.items():
            upload_data['regression_visualizations'][metric] = {
                'figure_html': data['figure'].to_html(include_plotlyjs=False, full_html=False),
                'regression_count': data['regression_count'],
                'max_regression': data['max_regression'],
                'affected_models': data['affected_models']
            }
        
        # Send data to dashboard
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{dashboard_url}/api/performance-analytics",
                    json=upload_data
                ) as response:
                    if response.status == 200:
                        logger.info("Successfully uploaded performance analytics to dashboard")
                        return True
                    else:
                        logger.error(f"Failed to upload to dashboard: HTTP {response.status} - {await response.text()}")
                        return False
        except Exception as e:
            logger.error(f"Error uploading to dashboard: {e}")
            return False

async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Performance Analytics for End-to-End Testing Framework")
    
    # Basic options
    parser.add_argument("--report-dir", default="./e2e_test_reports",
                       help="Directory containing test reports")
    parser.add_argument("--output-dir", default="./e2e_performance_reports",
                       help="Directory for generated reports")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    
    # Analysis options
    parser.add_argument("--time-range", type=int, default=30,
                       help="Time range in days for analysis")
    parser.add_argument("--baseline-days", type=int, default=7,
                       help="Days to use for baseline")
    parser.add_argument("--comparison-days", type=int, default=1,
                       help="Days to use for comparison")
    parser.add_argument("--metrics", default="all",
                       help="Metrics to analyze [latency,throughput,memory,cpu,all]")
    parser.add_argument("--models", default="all",
                       help="Models to include in analysis")
    parser.add_argument("--hardware-types", default="all",
                       help="Hardware types to include [cpu,gpu,webgpu,webnn,multi,all]")
    parser.add_argument("--regression-threshold", type=float, default=0.1,
                       help="Threshold for regression detection (default: 0.1)")
    
    # Output options
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate HTML report")
    parser.add_argument("--upload-to-dashboard", action="store_true",
                       help="Upload results to dashboard")
    parser.add_argument("--dashboard-url", default="http://localhost:8082",
                       help="URL of monitoring dashboard")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    # Parse metrics
    if args.metrics.lower() == 'all':
        metrics = ['latency_ms', 'throughput_items_sec', 'memory_usage_mb', 'cpu_usage_percent']
    else:
        metrics = args.metrics.split(',')
        # Map shorthand names to full names
        metric_map = {
            'latency': 'latency_ms',
            'throughput': 'throughput_items_sec',
            'memory': 'memory_usage_mb',
            'cpu': 'cpu_usage_percent'
        }
        metrics = [metric_map.get(m.lower(), m) for m in metrics]
    
    # Parse models
    if args.models.lower() == 'all':
        models = None
    else:
        models = args.models.split(',')
    
    # Parse hardware types
    if args.hardware_types.lower() == 'all':
        hardware_types = None
    else:
        hardware_types = args.hardware_types.split(',')
    
    # Create analytics object
    analytics = PerformanceAnalytics(
        report_dir=args.report_dir,
        output_dir=args.output_dir,
        db_path=args.db_path,
        time_range_days=args.time_range,
        baseline_days=args.baseline_days,
        comparison_days=args.comparison_days,
        debug=args.debug
    )
    
    # Load performance data
    performance_data = analytics.load_performance_data_from_db(
        time_range_days=args.time_range,
        models=models,
        hardware_types=hardware_types
    )
    
    if performance_data.empty:
        logger.warning("No performance data available for analysis")
        return 1
    
    # Store performance data
    analytics.performance_data = performance_data
    
    # Analyze performance trends
    trends = analytics.analyze_performance_trends(performance_data, metrics)
    
    # Detect performance regressions
    regressions = analytics.detect_performance_regressions(
        performance_data,
        regression_threshold=args.regression_threshold,
        metrics=metrics
    )
    
    # Create visualizations
    trend_visualizations = analytics.create_trend_visualization(trends, metrics=metrics)
    regression_visualizations = analytics.create_regression_visualization(regressions, metrics=metrics)
    
    # Generate HTML report if requested
    if args.generate_report:
        report_path = analytics.create_html_report(
            trend_visualizations,
            regression_visualizations
        )
        print(f"Generated HTML report: {report_path}")
        
        # Open the report in browser
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
        except:
            pass
    
    # Upload to dashboard if requested
    if args.upload_to_dashboard:
        success = await analytics.upload_to_dashboard(
            args.dashboard_url,
            trend_visualizations,
            regression_visualizations
        )
        
        if success:
            print(f"Successfully uploaded to dashboard: {args.dashboard_url}/performance-analytics")
        else:
            print("Failed to upload to dashboard")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))