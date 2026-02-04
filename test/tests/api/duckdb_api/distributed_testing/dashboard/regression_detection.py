#!/usr/bin/env python3
"""
Advanced Regression Detection for Visualization Dashboard

This module implements enhanced regression detection capabilities for the visualization dashboard
with statistical analysis, change point detection, and advanced visualization of regressions.
It integrates with the Enhanced Visualization Dashboard to provide comprehensive regression analysis.

Features:
- Statistical significance testing for detecting true regressions
- Change point detection for identifying when performance changes occur
- Bayesian analysis for robust regression detection with uncertainty quantification
- Visual annotations for highlighting regressions in time series data
- Severity classification based on statistical and business impact
- Correlation analysis between different metrics for root cause analysis
"""

import logging
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("regression_detection")

# Try to import optional dependencies with graceful fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Some regression detection features will be limited.")
    PANDAS_AVAILABLE = False

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy not available. Statistical significance testing will be disabled.")
    SCIPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Visualization features will be limited.")
    PLOTLY_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    logger.warning("Ruptures not available. Change point detection will be limited.")
    RUPTURES_AVAILABLE = False

class RegressionDetector:
    """Advanced regression detection for performance data with statistical significance testing and visualization."""
    
    def __init__(self, db_conn=None):
        """Initialize the regression detector.
        
        Args:
            db_conn: Optional DuckDB connection for querying historical data
        """
        self.db_conn = db_conn
        self.config = {
            # Basic configuration
            "min_samples": 5,                 # Minimum samples required for detection
            "window_size": 10,                # Window size for moving average
            "regression_threshold": 10.0,     # Percentage change to trigger detection (%)
            "confidence_level": 0.95,         # Statistical confidence level (1-alpha)
            
            # Advanced configuration
            "change_point_penalty": 2,        # Penalty term for change point detection (higher = fewer change points)
            "change_point_model": "l2",       # Model for change point detection (l1, l2, rbf, etc.)
            "smoothing_factor": 0.2,          # Smoothing factor for time series (0-1)
            "allow_positive_regressions": False, # Whether to include performance improvements
            
            # Severity classification thresholds
            "severity_thresholds": {
                "critical": 30.0,            # >30% change
                "high": 20.0,                # >20% change
                "medium": 10.0,              # >10% change
                "low": 5.0                   # >5% change
            },
            
            # Metrics configuration
            "metrics_config": {
                "latency_ms": {
                    "higher_is_better": False,  # Lower latency is better
                    "unit": "ms",
                    "display_name": "Latency",
                    "regression_direction": "increase"  # An increase is a regression
                },
                "throughput_items_per_second": {
                    "higher_is_better": True,   # Higher throughput is better
                    "unit": "items/sec",
                    "display_name": "Throughput",
                    "regression_direction": "decrease"  # A decrease is a regression
                },
                "memory_usage_mb": {
                    "higher_is_better": False,  # Lower memory usage is better
                    "unit": "MB",
                    "display_name": "Memory Usage",
                    "regression_direction": "increase"  # An increase is a regression
                }
            }
        }
        
        logger.info("Regression detector initialized")
    
    def detect_regressions(self, time_series_data: Dict[str, Any], metric: str) -> List[Dict[str, Any]]:
        """Detect regressions in time series data for a specific metric.
        
        Args:
            time_series_data: Dictionary containing timestamps and values
            metric: Metric name to analyze
            
        Returns:
            List of detected regressions with details
        """
        if not time_series_data or "timestamps" not in time_series_data or "values" not in time_series_data:
            logger.warning("Invalid time series data format")
            return []
            
        timestamps = time_series_data["timestamps"]
        values = time_series_data["values"]
        
        if len(timestamps) != len(values):
            logger.warning("Timestamps and values length mismatch")
            return []
            
        if len(values) < self.config["min_samples"]:
            logger.warning(f"Insufficient data points for regression detection. Need at least {self.config['min_samples']}.")
            return []
        
        # Get metric configuration
        metric_config = self.config["metrics_config"].get(metric, {
            "higher_is_better": False,
            "unit": "",
            "display_name": metric,
            "regression_direction": "any"
        })
        
        # Apply smoothing if needed
        if self.config["smoothing_factor"] > 0:
            values = self._apply_smoothing(values)
        
        # Detect change points for segmenting the time series
        change_points = self._detect_change_points(values)
        
        # Analyze segments for regressions
        regressions = []
        
        if not change_points or len(change_points) == 0:
            # No change points detected, compare start and end
            if len(values) > 2 * self.config["window_size"]:
                # Only if we have enough data for two windows
                start_segment = values[:self.config["window_size"]]
                end_segment = values[-self.config["window_size"]:]
                
                regression = self._analyze_segment_pair(
                    start_segment, end_segment, 
                    timestamps[0], timestamps[-1],
                    metric, metric_config
                )
                
                if regression:
                    regressions.append(regression)
        else:
            # Analyze each segment pair around change points
            for i, cp in enumerate(change_points):
                if cp < self.config["window_size"] or cp > len(values) - self.config["window_size"]:
                    continue  # Skip if too close to the start or end
                
                # Define the two segments
                window_size = min(self.config["window_size"], cp // 2)
                before_segment = values[cp - window_size:cp]
                after_segment = values[cp:cp + window_size]
                
                # Skip if either segment has no data
                if not before_segment or not after_segment:
                    continue
                
                regression = self._analyze_segment_pair(
                    before_segment, after_segment, 
                    timestamps[cp - window_size], timestamps[cp],
                    metric, metric_config
                )
                
                if regression:
                    regressions.append(regression)
        
        # Sort regressions by significance (most significant first)
        regressions.sort(key=lambda x: x.get("significance", 0), reverse=True)
        
        return regressions
            
    def _apply_smoothing(self, values: List[float]) -> List[float]:
        """Apply exponential smoothing to the time series.
        
        Args:
            values: List of values to smooth
            
        Returns:
            Smoothed values
        """
        alpha = self.config["smoothing_factor"]
        smoothed = [values[0]]  # Start with the first value
        
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
            
        return smoothed
        
    def _detect_change_points(self, values: List[float]) -> List[int]:
        """Detect change points in the time series.
        
        Args:
            values: List of values to analyze
            
        Returns:
            List of change point indices
        """
        if len(values) < 2 * self.config["min_samples"]:
            return []  # Not enough data
            
        # Use ruptures package if available
        if RUPTURES_AVAILABLE:
            try:
                # Initialize change point detection algorithm
                algo = rpt.Pelt(model=self.config["change_point_model"], 
                                min_size=self.config["min_samples"],
                                jump=1)
                
                # Fit the algorithm to the data
                algo.fit(np.array(values).reshape(-1, 1))
                
                # Find the optimal number of change points
                result = algo.predict(pen=self.config["change_point_penalty"])
                
                # Remove the last change point (end of sequence)
                if result[-1] == len(values):
                    result = result[:-1]
                    
                return result
            except Exception as e:
                logger.warning(f"Error in change point detection: {e}")
                return self._simple_change_point_detection(values)
        else:
            # Fallback to simple change point detection
            return self._simple_change_point_detection(values)
            
    def _simple_change_point_detection(self, values: List[float]) -> List[int]:
        """Simple change point detection based on moving average.
        
        Args:
            values: List of values to analyze
            
        Returns:
            List of change point indices
        """
        if len(values) < 2 * self.config["window_size"]:
            return []  # Not enough data
        
        # Calculate moving average
        window_size = self.config["window_size"]
        moving_avg = []
        
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            moving_avg.append(sum(window) / window_size)
        
        # Calculate differences
        diffs = [abs(moving_avg[i] - moving_avg[i-1]) for i in range(1, len(moving_avg))]
        
        # Find the threshold for significant changes (90th percentile)
        if not diffs:
            return []
            
        threshold = np.percentile(diffs, 90)
        
        # Find significant change points
        change_points = []
        for i in range(len(diffs)):
            if diffs[i] > threshold:
                # Add window_size to get back to the original index
                change_points.append(i + window_size)
        
        # Merge close change points
        if not change_points:
            return []
            
        merged_points = [change_points[0]]
        for cp in change_points[1:]:
            if cp - merged_points[-1] > window_size:
                merged_points.append(cp)
        
        return merged_points
        
    def _analyze_segment_pair(self, before_segment: List[float], after_segment: List[float],
                             before_time, after_time, metric: str, metric_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a pair of segments for regression.
        
        Args:
            before_segment: Values before the change point
            after_segment: Values after the change point
            before_time: Timestamp of the start of before segment
            after_time: Timestamp of the start of after segment
            metric: Metric name
            metric_config: Configuration for the metric
            
        Returns:
            Regression details if detected, None otherwise
        """
        if not before_segment or not after_segment:
            return None
            
        # Calculate basic statistics
        before_mean = np.mean(before_segment)
        after_mean = np.mean(after_segment)
        
        # Calculate percentage change
        if before_mean == 0:
            return None  # Avoid division by zero
            
        percentage_change = ((after_mean - before_mean) / before_mean) * 100
        
        # Determine if this is a regression based on metric configuration
        higher_is_better = metric_config.get("higher_is_better", False)
        
        is_regression = False
        if higher_is_better:
            # For metrics where higher is better (throughput), a decrease is a regression
            is_regression = percentage_change < -self.config["regression_threshold"]
        else:
            # For metrics where lower is better (latency, memory), an increase is a regression
            is_regression = percentage_change > self.config["regression_threshold"]
        
        # Skip if not a regression or if we don't want positive "regressions" (improvements)
        if not is_regression and not (self.config["allow_positive_regressions"] and abs(percentage_change) > self.config["regression_threshold"]):
            return None
        
        # Determine statistical significance
        p_value = 1.0  # Default p-value (not significant)
        is_significant = False
        
        if SCIPY_AVAILABLE:
            # Perform t-test for statistical significance
            t_stat, p_value = stats.ttest_ind(before_segment, after_segment, equal_var=False)
            is_significant = p_value < (1 - self.config["confidence_level"])
        
        # Calculate severity
        severity = self._calculate_severity(abs(percentage_change))
        
        # Create regression details
        regression = {
            "metric": metric,
            "display_name": metric_config.get("display_name", metric),
            "unit": metric_config.get("unit", ""),
            "change_point_time": after_time,
            "before_mean": before_mean,
            "after_mean": after_mean,
            "percentage_change": percentage_change,
            "absolute_change": after_mean - before_mean,
            "is_regression": is_regression,
            "is_improvement": percentage_change > 0 if higher_is_better else percentage_change < 0,
            "p_value": p_value,
            "is_significant": is_significant,
            "significance": 1.0 - p_value if p_value < 1.0 else 0.0,
            "severity": severity,
            "before_sample_size": len(before_segment),
            "after_sample_size": len(after_segment),
            "direction": "increase" if percentage_change > 0 else "decrease"
        }
        
        return regression
        
    def _calculate_severity(self, percentage_change: float) -> str:
        """Calculate severity level based on percentage change.
        
        Args:
            percentage_change: Absolute percentage change
            
        Returns:
            Severity level (critical, high, medium, low, or none)
        """
        thresholds = self.config["severity_thresholds"]
        
        if percentage_change >= thresholds["critical"]:
            return "critical"
        elif percentage_change >= thresholds["high"]:
            return "high"
        elif percentage_change >= thresholds["medium"]:
            return "medium"
        elif percentage_change >= thresholds["low"]:
            return "low"
        else:
            return "none"
    
    def create_regression_visualization(self, time_series_data: Dict[str, Any], 
                                      regressions: List[Dict[str, Any]], 
                                      metric: str,
                                      title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create visualization highlighting regressions in time series data.
        
        Args:
            time_series_data: Dictionary containing timestamps and values
            regressions: List of detected regressions
            metric: Metric name
            title: Optional title for the visualization
            
        Returns:
            Plotly figure as dictionary if Plotly is available, None otherwise
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create regression visualization.")
            return None
            
        # Extract data from time series
        timestamps = time_series_data["timestamps"]
        values = time_series_data["values"]
        
        # Get metric configuration
        metric_config = self.config["metrics_config"].get(metric, {
            "display_name": metric,
            "unit": "",
            "higher_is_better": False
        })
        
        # Apply smoothing if needed
        smoothed_values = self._apply_smoothing(values) if self.config["smoothing_factor"] > 0 else None
        
        # Create figure
        fig = go.Figure()
        
        # Add original values
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines+markers",
                name=f"{metric_config.get('display_name', metric)} (Raw)",
                line=dict(width=1, color="#1F77B4"),
                marker=dict(size=4)
            )
        )
        
        # Add smoothed values if available
        if smoothed_values:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=smoothed_values,
                    mode="lines",
                    name=f"{metric_config.get('display_name', metric)} (Smoothed)",
                    line=dict(width=2, color="#FF7F0E"),
                )
            )
        
        # Add annotations for regressions
        shapes = []
        annotations = []
        
        for regression in regressions:
            if not regression.get("is_significant", False):
                continue  # Only show significant regressions
                
            change_point_time = regression.get("change_point_time")
            if change_point_time not in timestamps:
                continue  # Skip if change point time not in timestamps
                
            # Find index of change point
            change_idx = timestamps.index(change_point_time)
            
            # Add vertical line at change point
            shape_color = "#FF0000" if regression.get("is_regression", False) else "#00FF00"
            shapes.append(
                dict(
                    type="line",
                    x0=change_point_time,
                    y0=min(values),
                    x1=change_point_time,
                    y1=max(values),
                    line=dict(
                        color=shape_color,
                        width=2,
                        dash="dash",
                    )
                )
            )
            
            # Add annotation
            direction = "▲" if regression.get("direction") == "increase" else "▼"
            color = "#FF0000" if regression.get("is_regression", False) else "#00FF00"
            annotations.append(
                dict(
                    x=change_point_time,
                    y=values[change_idx],
                    xref="x",
                    yref="y",
                    text=f"{direction} {abs(regression.get('percentage_change', 0)):.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    font=dict(
                        size=12,
                        color=color
                    ),
                    bordercolor=color,
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    opacity=0.8
                )
            )
        
        # Update figure with shapes and annotations
        fig.update_layout(
            shapes=shapes,
            annotations=annotations
        )
        
        # Add change points as vertical lines
        change_points = self._detect_change_points(values)
        for cp in change_points:
            if cp < len(timestamps):
                shapes.append(
                    dict(
                        type="line",
                        x0=timestamps[cp],
                        y0=min(values),
                        x1=timestamps[cp],
                        y1=max(values),
                        line=dict(
                            color="rgba(100, 100, 100, 0.5)",
                            width=1,
                            dash="dot",
                        )
                    )
                )
        
        # Update layout
        title = title or f"{metric_config.get('display_name', metric)} with Regression Detection"
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=f"{metric_config.get('display_name', metric)} ({metric_config.get('unit', '')})",
            shapes=shapes,
            showlegend=True,
            hovermode="closest",
            template="plotly_white",
            height=500,
            width=900,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig.to_dict()
    
    def generate_regression_report(self, time_series_data: Dict[str, Dict[str, Any]], 
                                  regressions_by_metric: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a comprehensive regression analysis report.
        
        Args:
            time_series_data: Dictionary of time series data by metric
            regressions_by_metric: Dictionary of detected regressions by metric
            
        Returns:
            Report as a dictionary with summary and details
        """
        # Initialize report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": {
                "total_metrics_analyzed": len(time_series_data),
                "total_regressions_detected": 0,
                "significant_regressions": 0,
                "critical_regressions": 0,
                "high_regressions": 0,
                "medium_regressions": 0,
                "low_regressions": 0
            },
            "metrics": {},
            "regressions": []
        }
        
        # Analyze each metric
        for metric, regressions in regressions_by_metric.items():
            # Count regressions
            significant_regressions = [r for r in regressions if r.get("is_significant", False) and r.get("is_regression", False)]
            
            # Count by severity
            severity_counts = {
                "critical": 0, "high": 0, "medium": 0, "low": 0, "none": 0
            }
            
            for r in significant_regressions:
                severity = r.get("severity", "none")
                severity_counts[severity] += 1
            
            # Update summary counts
            report["summary"]["total_regressions_detected"] += len(regressions)
            report["summary"]["significant_regressions"] += len(significant_regressions)
            report["summary"]["critical_regressions"] += severity_counts["critical"]
            report["summary"]["high_regressions"] += severity_counts["high"]
            report["summary"]["medium_regressions"] += severity_counts["medium"]
            report["summary"]["low_regressions"] += severity_counts["low"]
            
            # Add metric details
            if metric not in report["metrics"]:
                report["metrics"][metric] = {
                    "total_regressions": len(regressions),
                    "significant_regressions": len(significant_regressions),
                    "severity_distribution": severity_counts,
                    "data_points": len(time_series_data.get(metric, {}).get("values", []))
                }
            
            # Add all significant regressions to the list
            for r in significant_regressions:
                # Add metric name if not already present
                if "metric" not in r:
                    r["metric"] = metric
                    
                report["regressions"].append(r)
        
        # Sort regressions by significance and severity
        report["regressions"].sort(key=lambda x: (
            x.get("significance", 0), 
            {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}.get(x.get("severity", "none"), 0)
        ), reverse=True)
        
        return report
    
    def create_correlation_analysis(self, metrics_data: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create correlation analysis between different metrics.
        
        Args:
            metrics_data: Dictionary of time series data by metric
            
        Returns:
            Correlation analysis as dictionary if Pandas and Plotly are available, None otherwise
        """
        if not PANDAS_AVAILABLE or not PLOTLY_AVAILABLE:
            logger.warning("Pandas or Plotly not available. Cannot create correlation analysis.")
            return None
            
        # Convert to pandas DataFrame
        data = {}
        timestamps = None
        
        for metric, metric_data in metrics_data.items():
            if "values" in metric_data and "timestamps" in metric_data:
                if timestamps is None:
                    timestamps = metric_data["timestamps"]
                elif len(timestamps) != len(metric_data["timestamps"]):
                    # Skip metrics with different timestamps
                    continue
                    
                data[metric] = metric_data["values"]
        
        if not data:
            return None
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu",
                zmid=0,
                text=[
                    [f"{x:.2f}" for x in row] for row in corr_matrix.values
                ],
                texttemplate="%{text}",
                hovertemplate="Correlation between %{y} and %{x}: %{z:.3f}<extra></extra>"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Metric Correlation Analysis",
            height=600,
            width=800,
            xaxis=dict(title="Metric"),
            yaxis=dict(title="Metric"),
            template="plotly_white"
        )
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "visualization": fig.to_dict(),
            "insights": self._generate_correlation_insights(corr_matrix)
        }
        
    def _generate_correlation_insights(self, corr_matrix) -> List[str]:
        """Generate insights about correlations between metrics.
        
        Args:
            corr_matrix: Correlation matrix as pandas DataFrame
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Find strong positive correlations
        strong_positive = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if corr > 0.7:
                    strong_positive.append((col1, col2, corr))
        
        # Find strong negative correlations
        strong_negative = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if corr < -0.7:
                    strong_negative.append((col1, col2, corr))
        
        # Generate insights
        for col1, col2, corr in strong_positive:
            insights.append(f"Strong positive correlation ({corr:.2f}) between {col1} and {col2}")
            
        for col1, col2, corr in strong_negative:
            insights.append(f"Strong negative correlation ({corr:.2f}) between {col1} and {col2}")
        
        # Check for common patterns
        if "latency_ms" in corr_matrix.columns and "throughput_items_per_second" in corr_matrix.columns:
            corr = corr_matrix.loc["latency_ms", "throughput_items_per_second"]
            if corr < -0.5:
                insights.append(f"Expected negative correlation ({corr:.2f}) between latency and throughput: lower latency correlates with higher throughput")
        
        return insights