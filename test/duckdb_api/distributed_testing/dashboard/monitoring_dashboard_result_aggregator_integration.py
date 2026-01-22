#!/usr/bin/env python3
"""
Enhanced Result Aggregator Integration for Monitoring Dashboard

This module provides enhanced integration between the Monitoring Dashboard
and the Result Aggregator components of the Distributed Testing Framework.
It enables comprehensive visualization of aggregated test results within
the dashboard interface.

Implementation Date: March 17, 2025
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dashboard_aggregator_integration")

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import optional visualization dependencies
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Advanced visualizations will be limited.")
    PLOTLY_AVAILABLE = False

# Import result aggregator constants
try:
    from duckdb_api.distributed_testing.result_aggregator.service import (
        RESULT_TYPE_PERFORMANCE, RESULT_TYPE_COMPATIBILITY,
        RESULT_TYPE_INTEGRATION, RESULT_TYPE_WEB_PLATFORM,
        AGGREGATION_LEVEL_TEST_RUN, AGGREGATION_LEVEL_MODEL,
        AGGREGATION_LEVEL_HARDWARE, AGGREGATION_LEVEL_MODEL_HARDWARE,
        AGGREGATION_LEVEL_TASK_TYPE, AGGREGATION_LEVEL_WORKER
    )
    AGGREGATOR_CONSTANTS_AVAILABLE = True
except ImportError:
    logger.warning("Result aggregator constants not available. Using fallback definitions.")
    AGGREGATOR_CONSTANTS_AVAILABLE = False
    # Fallback definitions
    RESULT_TYPE_PERFORMANCE = "performance"
    RESULT_TYPE_COMPATIBILITY = "compatibility"
    RESULT_TYPE_INTEGRATION = "integration"
    RESULT_TYPE_WEB_PLATFORM = "web_platform"
    AGGREGATION_LEVEL_TEST_RUN = "test_run"
    AGGREGATION_LEVEL_MODEL = "model"
    AGGREGATION_LEVEL_HARDWARE = "hardware"
    AGGREGATION_LEVEL_MODEL_HARDWARE = "model_hardware"
    AGGREGATION_LEVEL_TASK_TYPE = "task_type"
    AGGREGATION_LEVEL_WORKER = "worker"

class ResultAggregatorIntegration:
    """
    Enhanced integration between Monitoring Dashboard and Result Aggregator.
    
    This class provides advanced visualization and integration capabilities
    to display result aggregator data within the monitoring dashboard.
    """
    
    def __init__(self, result_aggregator=None, output_dir="./monitoring_dashboard"):
        """
        Initialize the result aggregator integration.
        
        Args:
            result_aggregator: Result aggregator instance
            output_dir: Directory to save visualization files
        """
        self.result_aggregator = result_aggregator
        self.output_dir = output_dir
        self.visualizations_dir = os.path.join(output_dir, "visualizations")
        
        # Create directory if it doesn't exist
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Configuration
        self.config = {
            "default_result_type": RESULT_TYPE_PERFORMANCE,
            "default_aggregation_level": AGGREGATION_LEVEL_MODEL_HARDWARE,
            "theme": "dark",
            "max_items_in_charts": 12,
            "chart_height": 500,
            "chart_width": 900,
            "enable_annotations": True,
            "color_scales": {
                "performance": "RdYlGn",  # Red-Yellow-Green
                "compatibility": "YlGnBu",  # Yellow-Green-Blue
                "integration": "Viridis",
                "web_platform": "Plasma"
            },
            "anomaly_highlight_color": "rgba(255, 0, 0, 0.3)",
            "improvement_highlight_color": "rgba(0, 255, 0, 0.3)",
            "color_scheme": {
                "dark": {
                    "background": "#1a1a1a",
                    "paper_bgcolor": "#2a2a2a",
                    "grid_color": "#444444",
                    "text_color": "#ffffff",
                    "line_colors": ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]
                },
                "light": {
                    "background": "#ffffff",
                    "paper_bgcolor": "#f8f9fa",
                    "grid_color": "#e9ecef",
                    "text_color": "#333333",
                    "line_colors": ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]
                }
            }
        }
        
        logger.info("Result aggregator integration initialized")
    
    def configure(self, config_updates: Dict[str, Any]):
        """
        Update the integration configuration.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        self.config.update(config_updates)
        logger.info(f"Result aggregator integration configuration updated: {config_updates}")
    
    def get_performance_summary(self, time_range_days=7) -> Dict[str, Any]:
        """
        Get a summary of performance metrics for dashboard display.
        
        Args:
            time_range_days: Number of days to include in the summary
            
        Returns:
            Dictionary with performance summary data
        """
        if not self.result_aggregator:
            return {"error": "Result aggregator not available"}
        
        # Define time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_range_days)
        time_range = (start_time, end_time)
        
        try:
            # Get aggregated performance results
            performance_results = self.result_aggregator.aggregate_results(
                result_type=RESULT_TYPE_PERFORMANCE,
                aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
                time_range=time_range,
                use_cache=True
            )
            
            # Extract basic statistics
            stats = performance_results.get("results", {}).get("basic_statistics", {})
            
            # Calculate overall summary
            total_tests = sum(
                group_stats.get("result_count", 0) 
                for group_stats in stats.values()
            )
            
            # Extract anomalies
            anomalies = performance_results.get("results", {}).get("anomalies", {})
            total_anomalies = sum(
                len(metric_anomalies.get("anomalies", []))
                for group_anomalies in anomalies.values()
                for metric_anomalies in group_anomalies.values()
            )
            
            # Extract comparisons for improvements/regressions
            comparisons = performance_results.get("results", {}).get("comparisons", {})
            improvements = 0
            regressions = 0
            for group_comparisons in comparisons.values():
                for metric_comparison in group_comparisons.values():
                    if metric_comparison.get("significance", False):
                        if metric_comparison.get("is_improvement", False):
                            improvements += 1
                        else:
                            regressions += 1
            
            # Prepare summary data
            summary = {
                "total_tests": total_tests,
                "total_model_hardware_pairs": len(stats),
                "total_anomalies": total_anomalies,
                "total_improvements": improvements,
                "total_regressions": regressions,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "days": time_range_days
                },
                "models": set(
                    group_key.split(":")[0] if ":" in group_key else group_key
                    for group_key in stats.keys()
                ),
                "hardware": set(
                    group_key.split(":")[1] if ":" in group_key else "unknown"
                    for group_key in stats.keys()
                    if ":" in group_key
                )
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def get_compatibility_summary(self, time_range_days=7) -> Dict[str, Any]:
        """
        Get a summary of compatibility metrics for dashboard display.
        
        Args:
            time_range_days: Number of days to include in the summary
            
        Returns:
            Dictionary with compatibility summary data
        """
        if not self.result_aggregator:
            return {"error": "Result aggregator not available"}
        
        # Define time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_range_days)
        time_range = (start_time, end_time)
        
        try:
            # Get aggregated compatibility results
            compatibility_results = self.result_aggregator.aggregate_results(
                result_type=RESULT_TYPE_COMPATIBILITY,
                aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
                time_range=time_range,
                use_cache=True
            )
            
            # Extract distribution data for compatible/incompatible counts
            distributions = compatibility_results.get("results", {}).get("distributions", {})
            
            # Calculate compatibility statistics
            compatibility_stats = {
                "total_pairs_tested": 0,
                "compatible_pairs": 0,
                "incompatible_pairs": 0,
                "compatibility_rate": 0.0,
                "model_counts": {},
                "hardware_counts": {}
            }
            
            # Process each model-hardware pair
            for group_key, group_data in distributions.items():
                if ":" not in group_key:
                    continue
                    
                model_id, hardware_id = group_key.split(":", 1)
                
                # Check if is_compatible distribution exists
                if "is_compatible" in group_data:
                    is_compatible_data = group_data["is_compatible"]
                    total_values = is_compatible_data.get("total_values", 0)
                    
                    compatibility_stats["total_pairs_tested"] += 1
                    
                    # Count compatible instances
                    true_count = is_compatible_data.get("distribution", {}).get("True", {}).get("count", 0)
                    
                    if true_count > 0:
                        compatibility_stats["compatible_pairs"] += 1
                    else:
                        compatibility_stats["incompatible_pairs"] += 1
                    
                    # Track counts by model and hardware
                    if model_id not in compatibility_stats["model_counts"]:
                        compatibility_stats["model_counts"][model_id] = {
                            "total": 0, "compatible": 0
                        }
                    
                    compatibility_stats["model_counts"][model_id]["total"] += 1
                    if true_count > 0:
                        compatibility_stats["model_counts"][model_id]["compatible"] += 1
                        
                    if hardware_id not in compatibility_stats["hardware_counts"]:
                        compatibility_stats["hardware_counts"][hardware_id] = {
                            "total": 0, "compatible": 0
                        }
                        
                    compatibility_stats["hardware_counts"][hardware_id]["total"] += 1
                    if true_count > 0:
                        compatibility_stats["hardware_counts"][hardware_id]["compatible"] += 1
            
            # Calculate overall compatibility rate
            if compatibility_stats["total_pairs_tested"] > 0:
                compatibility_stats["compatibility_rate"] = (
                    compatibility_stats["compatible_pairs"] / 
                    compatibility_stats["total_pairs_tested"]
                ) * 100
            
            # Add calculated rates for models and hardware
            for model_id, counts in compatibility_stats["model_counts"].items():
                if counts["total"] > 0:
                    counts["compatibility_rate"] = (counts["compatible"] / counts["total"]) * 100
            
            for hardware_id, counts in compatibility_stats["hardware_counts"].items():
                if counts["total"] > 0:
                    counts["compatibility_rate"] = (counts["compatible"] / counts["total"]) * 100
            
            return compatibility_stats
            
        except Exception as e:
            logger.error(f"Error getting compatibility summary: {e}")
            return {"error": str(e)}
    
    def get_integration_test_summary(self, time_range_days=7) -> Dict[str, Any]:
        """
        Get a summary of integration test results for dashboard display.
        
        Args:
            time_range_days: Number of days to include in the summary
            
        Returns:
            Dictionary with integration test summary data
        """
        if not self.result_aggregator:
            return {"error": "Result aggregator not available"}
        
        # Define time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_range_days)
        time_range = (start_time, end_time)
        
        try:
            # Get aggregated integration test results
            integration_results = self.result_aggregator.aggregate_results(
                result_type=RESULT_TYPE_INTEGRATION,
                aggregation_level=AGGREGATION_LEVEL_TASK_TYPE,
                time_range=time_range,
                use_cache=True
            )
            
            # Extract distribution data for pass/fail counts
            distributions = integration_results.get("results", {}).get("distributions", {})
            
            # Calculate integration test statistics
            test_stats = {
                "total_tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "pass_rate": 0.0,
                "module_results": {},
                "test_class_results": {}
            }
            
            # Process status distributions
            for module_key, module_data in distributions.items():
                if "status" in module_data:
                    status_data = module_data["status"]
                    total_values = status_data.get("total_values", 0)
                    test_stats["total_tests_run"] += total_values
                    
                    # Count passed tests
                    pass_count = status_data.get("distribution", {}).get("pass", {}).get("count", 0)
                    test_stats["tests_passed"] += pass_count
                    
                    # Count failed tests (all non-pass results)
                    fail_count = total_values - pass_count
                    test_stats["tests_failed"] += fail_count
                    
                    # Track results by module
                    test_stats["module_results"][module_key] = {
                        "total": total_values,
                        "passed": pass_count,
                        "failed": fail_count,
                        "pass_rate": (pass_count / total_values) * 100 if total_values > 0 else 0
                    }
                
                # Process test class distribution if available
                if "test_class" in module_data:
                    class_data = module_data["test_class"]
                    for class_name, class_info in class_data.get("distribution", {}).items():
                        # Add to test class results
                        test_stats["test_class_results"][class_name] = {
                            "total": class_info.get("count", 0),
                            "module": module_key
                        }
            
            # Calculate overall pass rate
            if test_stats["total_tests_run"] > 0:
                test_stats["pass_rate"] = (
                    test_stats["tests_passed"] / 
                    test_stats["total_tests_run"]
                ) * 100
            
            return test_stats
            
        except Exception as e:
            logger.error(f"Error getting integration test summary: {e}")
            return {"error": str(e)}
    
    def get_web_platform_summary(self, time_range_days=7) -> Dict[str, Any]:
        """
        Get a summary of web platform test results for dashboard display.
        
        Args:
            time_range_days: Number of days to include in the summary
            
        Returns:
            Dictionary with web platform test summary data
        """
        if not self.result_aggregator:
            return {"error": "Result aggregator not available"}
        
        # Define time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_range_days)
        time_range = (start_time, end_time)
        
        try:
            # Get aggregated web platform results
            web_results = self.result_aggregator.aggregate_results(
                result_type=RESULT_TYPE_WEB_PLATFORM,
                aggregation_level=AGGREGATION_LEVEL_TASK_TYPE,  # This groups by platform:browser
                time_range=time_range,
                use_cache=True
            )
            
            # Extract distribution data for browser success rates
            distributions = web_results.get("results", {}).get("distributions", {})
            
            # Calculate web platform statistics
            web_stats = {
                "total_tests_run": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0,
                "platforms": {},
                "browsers": {},
                "platform_browser_pairs": {}
            }
            
            # Process success distributions
            for platform_browser, platform_data in distributions.items():
                if ":" not in platform_browser:
                    continue
                    
                platform, browser = platform_browser.split(":", 1)
                
                if "success" in platform_data:
                    success_data = platform_data["success"]
                    total_values = success_data.get("total_values", 0)
                    web_stats["total_tests_run"] += total_values
                    
                    # Count successful tests
                    success_count = success_data.get("distribution", {}).get("True", {}).get("count", 0)
                    web_stats["successful_tests"] += success_count
                    
                    # Count failed tests
                    fail_count = total_values - success_count
                    web_stats["failed_tests"] += fail_count
                    
                    # Track by platform
                    if platform not in web_stats["platforms"]:
                        web_stats["platforms"][platform] = {
                            "total": 0, "successful": 0, "success_rate": 0
                        }
                    
                    web_stats["platforms"][platform]["total"] += total_values
                    web_stats["platforms"][platform]["successful"] += success_count
                    
                    # Track by browser
                    if browser not in web_stats["browsers"]:
                        web_stats["browsers"][browser] = {
                            "total": 0, "successful": 0, "success_rate": 0
                        }
                    
                    web_stats["browsers"][browser]["total"] += total_values
                    web_stats["browsers"][browser]["successful"] += success_count
                    
                    # Track platform-browser pairs
                    web_stats["platform_browser_pairs"][platform_browser] = {
                        "total": total_values,
                        "successful": success_count,
                        "failed": fail_count,
                        "success_rate": (success_count / total_values) * 100 if total_values > 0 else 0,
                        "platform": platform,
                        "browser": browser
                    }
            
            # Calculate success rates
            if web_stats["total_tests_run"] > 0:
                web_stats["success_rate"] = (
                    web_stats["successful_tests"] / 
                    web_stats["total_tests_run"]
                ) * 100
            
            # Calculate platform success rates
            for platform, data in web_stats["platforms"].items():
                if data["total"] > 0:
                    data["success_rate"] = (data["successful"] / data["total"]) * 100
            
            # Calculate browser success rates
            for browser, data in web_stats["browsers"].items():
                if data["total"] > 0:
                    data["success_rate"] = (data["successful"] / data["total"]) * 100
            
            return web_stats
            
        except Exception as e:
            logger.error(f"Error getting web platform summary: {e}")
            return {"error": str(e)}
    
    def create_performance_trend_chart(self, 
                                      time_range_days=7, 
                                      metric="average_latency_ms", 
                                      top_n=8) -> Optional[str]:
        """
        Create a performance trend chart for the dashboard.
        
        Args:
            time_range_days: Number of days to include in the chart
            metric: Performance metric to visualize
            top_n: Number of top model-hardware pairs to include
            
        Returns:
            Path to the generated chart file, or None if creation failed
        """
        if not self.result_aggregator:
            logger.error("Result aggregator not available")
            return None
            
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for chart creation")
            return None
        
        # Define output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.visualizations_dir, 
            f"performance_trend_{metric}_{timestamp}.html"
        )
        
        # Define time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_range_days)
        time_range = (start_time, end_time)
        
        try:
            # Get aggregated performance results
            performance_results = self.result_aggregator.aggregate_results(
                result_type=RESULT_TYPE_PERFORMANCE,
                aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
                time_range=time_range,
                use_cache=True
            )
            
            # Extract basic statistics
            stats = performance_results.get("results", {}).get("basic_statistics", {})
            
            # Sort model-hardware pairs by the specified metric
            sorted_pairs = []
            for group_key, group_stats in stats.items():
                if metric in group_stats:
                    metric_value = group_stats[metric].get("mean", 0)
                    sorted_pairs.append((group_key, metric_value))
            
            # Determine sorting direction (lower is better for latency, higher for throughput)
            reverse_sort = "throughput" in metric or "score" in metric
            sorted_pairs.sort(key=lambda x: x[1], reverse=reverse_sort)
            
            # Take top N pairs
            top_pairs = sorted_pairs[:top_n]
            
            if not top_pairs:
                logger.warning(f"No data for performance trend chart with metric: {metric}")
                return None
            
            # Get trend data for each pair
            trend_data = {}
            for pair, _ in top_pairs:
                # Parse pair to get model_id and hardware_id
                if ":" in pair:
                    model_id, hardware_id = pair.split(":", 1)
                    
                    # Define filter params
                    filter_params = {
                        "model_id": model_id,
                        "hardware_id": hardware_id
                    }
                    
                    # Get raw results for trend
                    raw_results = self.result_aggregator._fetch_results(
                        result_type=RESULT_TYPE_PERFORMANCE,
                        aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
                        filter_params=filter_params,
                        time_range=time_range
                    )
                    
                    # Extract metric values and timestamps
                    values = []
                    for result in raw_results:
                        if metric in result and result[metric] is not None:
                            values.append({
                                "timestamp": result.get("timestamp", ""),
                                "value": result[metric],
                                "batch_size": result.get("batch_size", 1),
                                "precision": result.get("precision", "unknown")
                            })
                    
                    # Sort by timestamp
                    values.sort(key=lambda x: x["timestamp"])
                    
                    # Add to trend data
                    if values:
                        trend_data[pair] = values
            
            # Create the plot
            fig = go.Figure()
            
            # Set up color scheme based on theme
            theme = self.config["theme"]
            color_scheme = self.config["color_scheme"][theme]
            line_colors = color_scheme["line_colors"]
            
            # Add a trace for each model-hardware pair
            for i, (pair, values) in enumerate(trend_data.items()):
                color_idx = i % len(line_colors)
                line_color = line_colors[color_idx]
                
                # Extract x and y values
                x_values = [val["timestamp"] for val in values]
                y_values = [val["value"] for val in values]
                
                # Create hover text with additional information
                hover_text = [
                    f"Model-Hardware: {pair}<br>"
                    f"Timestamp: {val['timestamp']}<br>"
                    f"{metric}: {val['value']:.2f}<br>"
                    f"Batch Size: {val['batch_size']}<br>"
                    f"Precision: {val['precision']}"
                    for val in values
                ]
                
                # Add trace
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=pair,
                    line=dict(color=line_color),
                    hovertext=hover_text,
                    hoverinfo='text'
                ))
            
            # Add annotations for significant improvements or regressions
            if self.config["enable_annotations"]:
                comparisons = performance_results.get("results", {}).get("comparisons", {})
                
                for pair in trend_data.keys():
                    if pair in comparisons and metric in comparisons[pair]:
                        comparison = comparisons[pair][metric]
                        
                        # Check if change is significant
                        if comparison.get("significance", False):
                            is_improvement = comparison.get("is_improvement", False)
                            pct_change = comparison.get("pct_change_mean", 0)
                            
                            # Add annotation
                            annotation_text = (
                                f"{'↑' if is_improvement else '↓'} "
                                f"{abs(pct_change):.1f}% "
                                f"{'improvement' if is_improvement else 'regression'}"
                            )
                            
                            # Add annotation to the chart
                            fig.add_annotation(
                                x=end_time.strftime("%Y-%m-%d"),
                                y=comparison.get("current_mean", 0),
                                text=annotation_text,
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor="#2ecc71" if is_improvement else "#e74c3c",
                                bgcolor="#2ecc7133" if is_improvement else "#e74c3c33",
                                bordercolor="#2ecc71" if is_improvement else "#e74c3c",
                                borderwidth=1,
                                borderpad=4,
                                font=dict(color=color_scheme["text_color"])
                            )
            
            # Update layout
            fig.update_layout(
                title=f"Performance Trend: {metric} (Last {time_range_days} Days)",
                xaxis_title="Date",
                yaxis_title=metric,
                legend_title="Model-Hardware Pairs",
                height=self.config["chart_height"],
                width=self.config["chart_width"],
                template="plotly_dark" if theme == "dark" else "plotly_white",
                paper_bgcolor=color_scheme["paper_bgcolor"],
                plot_bgcolor=color_scheme["background"],
                font=dict(color=color_scheme["text_color"]),
                xaxis=dict(
                    gridcolor=color_scheme["grid_color"],
                    tickformat="%Y-%m-%d"
                ),
                yaxis=dict(
                    gridcolor=color_scheme["grid_color"]
                ),
                hovermode="closest"
            )
            
            # Create reversed axis for metrics where lower is better
            if "latency" in metric or "time" in metric:
                fig.update_layout(
                    yaxis=dict(
                        autorange="reversed",
                        gridcolor=color_scheme["grid_color"]
                    )
                )
            
            # Write to file
            fig.write_html(output_path)
            
            # Return the path relative to output_dir for template usage
            return os.path.relpath(output_path, os.path.dirname(self.output_dir))
            
        except Exception as e:
            logger.error(f"Error creating performance trend chart: {e}")
            return None
    
    def create_compatibility_matrix_chart(self, time_range_days=7) -> Optional[str]:
        """
        Create a compatibility matrix chart for the dashboard.
        
        Args:
            time_range_days: Number of days to include in the chart
            
        Returns:
            Path to the generated chart file, or None if creation failed
        """
        if not self.result_aggregator:
            logger.error("Result aggregator not available")
            return None
            
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for chart creation")
            return None
        
        # Define output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.visualizations_dir, 
            f"compatibility_matrix_{timestamp}.html"
        )
        
        # Define time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_range_days)
        time_range = (start_time, end_time)
        
        try:
            # Get compatibility summary
            compatibility_stats = self.get_compatibility_summary(time_range_days)
            
            if "error" in compatibility_stats:
                logger.error(f"Error getting compatibility data: {compatibility_stats['error']}")
                return None
            
            # Extract model and hardware information
            models = list(compatibility_stats["model_counts"].keys())
            hardware = list(compatibility_stats["hardware_counts"].keys())
            
            # Limit number of items if too many
            max_items = self.config["max_items_in_charts"]
            if len(models) > max_items:
                # Sort by compatibility rate
                sorted_models = sorted(
                    models,
                    key=lambda m: compatibility_stats["model_counts"][m].get("compatibility_rate", 0),
                    reverse=True
                )
                models = sorted_models[:max_items]
            
            if len(hardware) > max_items:
                # Sort by compatibility rate
                sorted_hardware = sorted(
                    hardware,
                    key=lambda h: compatibility_stats["hardware_counts"][h].get("compatibility_rate", 0),
                    reverse=True
                )
                hardware = sorted_hardware[:max_items]
            
            # Create compatibility matrix
            matrix = []
            annotations = []
            
            for h_idx, hw in enumerate(hardware):
                row = []
                for m_idx, model in enumerate(models):
                    pair_key = f"{model}:{hw}"
                    
                    # Check if this pair is in our compatibility results
                    is_compatible = False
                    
                    # Get aggregated compatibility results for this specific pair
                    filter_params = {
                        "model_id": model,
                        "hardware_id": hw
                    }
                    pair_results = self.result_aggregator.aggregate_results(
                        result_type=RESULT_TYPE_COMPATIBILITY,
                        aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
                        filter_params=filter_params,
                        time_range=time_range,
                        use_cache=True
                    )
                    
                    # Check if we have distribution data
                    distributions = pair_results.get("results", {}).get("distributions", {})
                    for group_key, group_data in distributions.items():
                        if group_key == pair_key and "is_compatible" in group_data:
                            is_compatible_data = group_data["is_compatible"]
                            distribution = is_compatible_data.get("distribution", {})
                            true_count = distribution.get("True", {}).get("count", 0)
                            is_compatible = true_count > 0
                    
                    # Add to matrix
                    row.append(1 if is_compatible else 0)
                    
                    # Add annotation
                    annotations.append(
                        dict(
                            x=m_idx,
                            y=h_idx,
                            text="✓" if is_compatible else "✕",
                            showarrow=False,
                            font=dict(
                                color="white",
                                size=16
                            )
                        )
                    )
                
                matrix.append(row)
            
            # Create the heatmap
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=models,
                y=hardware,
                colorscale=[
                    [0, "rgb(220, 53, 69)"],  # red for incompatible
                    [1, "rgb(40, 167, 69)"]   # green for compatible
                ],
                showscale=False,
                hovertemplate="Model: %{x}<br>Hardware: %{y}<br>Compatible: %{z}<extra></extra>"
            ))
            
            # Add annotations
            fig.update_layout(annotations=annotations)
            
            # Set up color scheme based on theme
            theme = self.config["theme"]
            color_scheme = self.config["color_scheme"][theme]
            
            # Update layout
            fig.update_layout(
                title="Model-Hardware Compatibility Matrix",
                height=self.config["chart_height"],
                width=self.config["chart_width"],
                template="plotly_dark" if theme == "dark" else "plotly_white",
                paper_bgcolor=color_scheme["paper_bgcolor"],
                plot_bgcolor=color_scheme["background"],
                font=dict(color=color_scheme["text_color"]),
                xaxis=dict(
                    title="Models",
                    gridcolor=color_scheme["grid_color"],
                    tickangle=-45
                ),
                yaxis=dict(
                    title="Hardware",
                    gridcolor=color_scheme["grid_color"]
                )
            )
            
            # Write to file
            fig.write_html(output_path)
            
            # Return the path relative to output_dir for template usage
            return os.path.relpath(output_path, os.path.dirname(self.output_dir))
            
        except Exception as e:
            logger.error(f"Error creating compatibility matrix chart: {e}")
            return None
    
    def create_test_pass_rate_chart(self, time_range_days=7) -> Optional[str]:
        """
        Create a test pass rate chart for the dashboard.
        
        Args:
            time_range_days: Number of days to include in the chart
            
        Returns:
            Path to the generated chart file, or None if creation failed
        """
        if not self.result_aggregator:
            logger.error("Result aggregator not available")
            return None
            
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for chart creation")
            return None
        
        # Define output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.visualizations_dir, 
            f"test_pass_rate_{timestamp}.html"
        )
        
        try:
            # Get integration test summary
            test_stats = self.get_integration_test_summary(time_range_days)
            
            if "error" in test_stats:
                logger.error(f"Error getting integration test data: {test_stats['error']}")
                return None
            
            # Extract module results
            module_results = test_stats.get("module_results", {})
            
            # Sort modules by pass rate
            sorted_modules = sorted(
                module_results.items(),
                key=lambda x: x[1]["pass_rate"],
                reverse=True
            )
            
            # Limit number of modules if too many
            max_items = self.config["max_items_in_charts"]
            if len(sorted_modules) > max_items:
                sorted_modules = sorted_modules[:max_items]
            
            # Prepare data for chart
            modules = [module for module, _ in sorted_modules]
            pass_rates = [data["pass_rate"] for _, data in sorted_modules]
            total_tests = [data["total"] for _, data in sorted_modules]
            passed_tests = [data["passed"] for _, data in sorted_modules]
            failed_tests = [data["failed"] for _, data in sorted_modules]
            
            # Create hover text
            hover_text = [
                f"Module: {module}<br>"
                f"Pass Rate: {data['pass_rate']:.1f}%<br>"
                f"Passed: {data['passed']} / {data['total']}<br>"
                f"Failed: {data['failed']}"
                for module, data in sorted_modules
            ]
            
            # Create the chart
            fig = go.Figure()
            
            # Set up color scheme based on theme
            theme = self.config["theme"]
            color_scheme = self.config["color_scheme"][theme]
            
            # Add pass rate bar
            fig.add_trace(go.Bar(
                x=modules,
                y=pass_rates,
                name="Pass Rate (%)",
                marker_color="#2ecc71",  # green
                opacity=0.7,
                hovertext=hover_text,
                hoverinfo="text"
            ))
            
            # Add total tests line
            fig.add_trace(go.Scatter(
                x=modules,
                y=total_tests,
                mode="lines+markers",
                name="Total Tests",
                line=dict(color="#3498db", width=3),  # blue
                marker=dict(size=8),
                yaxis="y2"
            ))
            
            # Update layout for dual axis
            fig.update_layout(
                title="Integration Test Pass Rates by Module",
                xaxis=dict(
                    title="Test Modules",
                    tickangle=-45,
                    gridcolor=color_scheme["grid_color"]
                ),
                yaxis=dict(
                    title="Pass Rate (%)",
                    range=[0, 100],
                    gridcolor=color_scheme["grid_color"],
                    tickformat=".1f"
                ),
                yaxis2=dict(
                    title="Test Count",
                    overlaying="y",
                    side="right",
                    gridcolor=color_scheme["grid_color"]
                ),
                height=self.config["chart_height"],
                width=self.config["chart_width"],
                template="plotly_dark" if theme == "dark" else "plotly_white",
                paper_bgcolor=color_scheme["paper_bgcolor"],
                plot_bgcolor=color_scheme["background"],
                font=dict(color=color_scheme["text_color"]),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="closest"
            )
            
            # Write to file
            fig.write_html(output_path)
            
            # Return the path relative to output_dir for template usage
            return os.path.relpath(output_path, os.path.dirname(self.output_dir))
            
        except Exception as e:
            logger.error(f"Error creating test pass rate chart: {e}")
            return None
    
    def create_web_platform_browser_chart(self, time_range_days=7) -> Optional[str]:
        """
        Create a web platform browser comparison chart for the dashboard.
        
        Args:
            time_range_days: Number of days to include in the chart
            
        Returns:
            Path to the generated chart file, or None if creation failed
        """
        if not self.result_aggregator:
            logger.error("Result aggregator not available")
            return None
            
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for chart creation")
            return None
        
        # Define output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.visualizations_dir, 
            f"web_platform_browsers_{timestamp}.html"
        )
        
        try:
            # Get web platform summary
            web_stats = self.get_web_platform_summary(time_range_days)
            
            if "error" in web_stats:
                logger.error(f"Error getting web platform data: {web_stats['error']}")
                return None
            
            # Extract browser and platform data
            browsers = web_stats.get("browsers", {})
            platforms = web_stats.get("platforms", {})
            platform_browser_pairs = web_stats.get("platform_browser_pairs", {})
            
            # Create a subplot with 2 charts
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "Browser Success Rates", 
                    "Platform Success Rates"
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Set up color scheme based on theme
            theme = self.config["theme"]
            color_scheme = self.config["color_scheme"][theme]
            
            # Add browser success rates
            browser_names = list(browsers.keys())
            browser_rates = [browsers[b]["success_rate"] for b in browser_names]
            browser_total = [browsers[b]["total"] for b in browser_names]
            
            # Create hover text for browsers
            browser_hover = [
                f"Browser: {browser}<br>"
                f"Success Rate: {data['success_rate']:.1f}%<br>"
                f"Successful: {data['successful']} / {data['total']}"
                for browser, data in browsers.items()
            ]
            
            # Add browser bar chart
            fig.add_trace(
                go.Bar(
                    x=browser_names,
                    y=browser_rates,
                    name="Browser Success Rate",
                    marker_color="#3498db",  # blue
                    hovertext=browser_hover,
                    hoverinfo="text"
                ),
                row=1, col=1
            )
            
            # Add platform success rates
            platform_names = list(platforms.keys())
            platform_rates = [platforms[p]["success_rate"] for p in platform_names]
            platform_total = [platforms[p]["total"] for p in platform_names]
            
            # Create hover text for platforms
            platform_hover = [
                f"Platform: {platform}<br>"
                f"Success Rate: {data['success_rate']:.1f}%<br>"
                f"Successful: {data['successful']} / {data['total']}"
                for platform, data in platforms.items()
            ]
            
            # Add platform bar chart
            fig.add_trace(
                go.Bar(
                    x=platform_names,
                    y=platform_rates,
                    name="Platform Success Rate",
                    marker_color="#2ecc71",  # green
                    hovertext=platform_hover,
                    hoverinfo="text"
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Web Platform Success Rates",
                height=self.config["chart_height"],
                width=self.config["chart_width"] * 1.5,  # Wider for two charts
                template="plotly_dark" if theme == "dark" else "plotly_white",
                paper_bgcolor=color_scheme["paper_bgcolor"],
                plot_bgcolor=color_scheme["background"],
                font=dict(color=color_scheme["text_color"]),
                showlegend=False,
                hovermode="closest"
            )
            
            # Update y-axis range for both charts
            fig.update_yaxes(
                title="Success Rate (%)",
                range=[0, 100],
                tickformat=".1f",
                gridcolor=color_scheme["grid_color"],
                row=1, col=1
            )
            
            fig.update_yaxes(
                title="Success Rate (%)",
                range=[0, 100],
                tickformat=".1f",
                gridcolor=color_scheme["grid_color"],
                row=1, col=2
            )
            
            # Update x-axis
            fig.update_xaxes(
                title="Browsers",
                tickangle=-45,
                gridcolor=color_scheme["grid_color"],
                row=1, col=1
            )
            
            fig.update_xaxes(
                title="Platforms",
                tickangle=-45,
                gridcolor=color_scheme["grid_color"],
                row=1, col=2
            )
            
            # Write to file
            fig.write_html(output_path)
            
            # Return the path relative to output_dir for template usage
            return os.path.relpath(output_path, os.path.dirname(self.output_dir))
            
        except Exception as e:
            logger.error(f"Error creating web platform browser chart: {e}")
            return None
    
    def create_anomaly_chart(self, time_range_days=7) -> Optional[str]:
        """
        Create a chart visualizing detected anomalies in performance metrics.
        
        Args:
            time_range_days: Number of days to include in the chart
            
        Returns:
            Path to the generated chart file, or None if creation failed
        """
        if not self.result_aggregator:
            logger.error("Result aggregator not available")
            return None
            
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for chart creation")
            return None
        
        # Define output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.visualizations_dir, 
            f"performance_anomalies_{timestamp}.html"
        )
        
        # Define time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_range_days)
        time_range = (start_time, end_time)
        
        try:
            # Get anomalies from result aggregator
            anomalies = self.result_aggregator.get_result_anomalies(
                result_type=RESULT_TYPE_PERFORMANCE,
                aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
                time_range=time_range,
                use_cache=True
            )
            
            if "error" in anomalies:
                logger.error(f"Error getting anomalies: {anomalies['error']}")
                return None
            
            # Extract anomaly data
            anomaly_list = anomalies.get("anomalies", [])
            
            if not anomaly_list:
                logger.warning("No anomalies found for chart")
                return None
            
            # Group anomalies by metric
            metric_anomalies = {}
            for anomaly in anomaly_list:
                metric = anomaly.get("metric")
                if metric not in metric_anomalies:
                    metric_anomalies[metric] = []
                metric_anomalies[metric].append(anomaly)
            
            # Create subplots for each metric
            metrics = list(metric_anomalies.keys())
            fig = make_subplots(
                rows=len(metrics), 
                cols=1,
                subplot_titles=[f"Anomalies in {m}" for m in metrics],
                vertical_spacing=0.1
            )
            
            # Set up color scheme based on theme
            theme = self.config["theme"]
            color_scheme = self.config["color_scheme"][theme]
            
            # Add anomalies for each metric
            for i, metric in enumerate(metrics):
                metric_data = metric_anomalies[metric]
                
                # Group by group (model-hardware pair)
                group_anomalies = {}
                for anomaly in metric_data:
                    group = anomaly.get("group")
                    if group not in group_anomalies:
                        group_anomalies[group] = []
                    group_anomalies[group].append(anomaly)
                
                # Add a trace for each group
                for j, (group, anomalies) in enumerate(group_anomalies.items()):
                    color_idx = j % len(color_scheme["line_colors"])
                    line_color = color_scheme["line_colors"][color_idx]
                    
                    # Extract value, mean, and z-score
                    values = [a["value"] for a in anomalies]
                    means = [a["mean"] for a in anomalies]
                    z_scores = [a["z_score"] for a in anomalies]
                    severities = [a["severity"] for a in anomalies]
                    
                    # Create hover text
                    hover_text = [
                        f"Group: {group}<br>"
                        f"Value: {v}<br>"
                        f"Mean: {m}<br>"
                        f"Z-Score: {z:.2f}<br>"
                        f"Severity: {s}"
                        for v, m, z, s in zip(values, means, z_scores, severities)
                    ]
                    
                    # Create a small dataset to show the anomaly
                    x_pos = [j] * len(values)
                    
                    # Add scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=x_pos,
                            y=values,
                            mode="markers",
                            name=group,
                            marker=dict(
                                color=line_color,
                                size=12,
                                symbol="circle",
                                line=dict(
                                    color="red" if "bad" in severities else "green",
                                    width=2
                                )
                            ),
                            hovertext=hover_text,
                            hoverinfo="text"
                        ),
                        row=i+1, col=1
                    )
                    
                    # Add reference mean line
                    if means:
                        fig.add_trace(
                            go.Scatter(
                                x=[j-0.2, j+0.2],
                                y=[means[0], means[0]],
                                mode="lines",
                                line=dict(
                                    color="rgba(255, 255, 255, 0.5)",
                                    width=2,
                                    dash="dash"
                                ),
                                showlegend=False
                            ),
                            row=i+1, col=1
                        )
                
                # Update axes for this subplot
                fig.update_xaxes(
                    title="Model-Hardware Pairs",
                    tickvals=list(range(len(group_anomalies))),
                    ticktext=list(group_anomalies.keys()),
                    tickangle=-45,
                    gridcolor=color_scheme["grid_color"],
                    row=i+1, col=1
                )
                
                fig.update_yaxes(
                    title=metric,
                    gridcolor=color_scheme["grid_color"],
                    row=i+1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title="Performance Metric Anomalies",
                height=self.config["chart_height"] * len(metrics),
                width=self.config["chart_width"],
                template="plotly_dark" if theme == "dark" else "plotly_white",
                paper_bgcolor=color_scheme["paper_bgcolor"],
                plot_bgcolor=color_scheme["background"],
                font=dict(color=color_scheme["text_color"]),
                showlegend=False,
                hovermode="closest"
            )
            
            # Write to file
            fig.write_html(output_path)
            
            # Return the path relative to output_dir for template usage
            return os.path.relpath(output_path, os.path.dirname(self.output_dir))
            
        except Exception as e:
            logger.error(f"Error creating anomaly chart: {e}")
            return None
    
    def create_performance_comparison_chart(self, time_range_days=7) -> Optional[str]:
        """
        Create a performance comparison chart showing improvements/regressions.
        
        Args:
            time_range_days: Number of days to include in the chart
            
        Returns:
            Path to the generated chart file, or None if creation failed
        """
        if not self.result_aggregator:
            logger.error("Result aggregator not available")
            return None
            
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for chart creation")
            return None
        
        # Define output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.visualizations_dir, 
            f"performance_comparison_{timestamp}.html"
        )
        
        try:
            # Get comparison report from aggregator
            comparison_report = self.result_aggregator.get_comparison_report(
                result_type=RESULT_TYPE_PERFORMANCE,
                aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
                time_range=None,  # Use default
                use_cache=True
            )
            
            if "error" in comparison_report:
                logger.error(f"Error getting comparison report: {comparison_report['error']}")
                return None
            
            # Extract comparison data
            comparisons = comparison_report.get("comparisons", [])
            
            if not comparisons:
                logger.warning("No comparison data available for chart")
                return None
            
            # Filter for significant changes only
            significant_comparisons = [c for c in comparisons if c.get("significance", False)]
            
            if not significant_comparisons:
                logger.warning("No significant changes found for comparison chart")
                return None
            
            # Group by metric
            metric_comparisons = {}
            for comp in significant_comparisons:
                metric = comp.get("metric")
                if metric not in metric_comparisons:
                    metric_comparisons[metric] = []
                metric_comparisons[metric].append(comp)
            
            # Create subplots for each metric
            metrics = list(metric_comparisons.keys())
            fig = make_subplots(
                rows=len(metrics), 
                cols=1,
                subplot_titles=[f"Changes in {m}" for m in metrics],
                vertical_spacing=0.1
            )
            
            # Set up color scheme based on theme
            theme = self.config["theme"]
            color_scheme = self.config["color_scheme"][theme]
            
            # Process each metric
            for i, metric in enumerate(metrics):
                metric_data = metric_comparisons[metric]
                
                # Sort by percentage change
                sorted_data = sorted(
                    metric_data, 
                    key=lambda x: abs(x.get("pct_change_mean", 0)),
                    reverse=True
                )
                
                # Limit to max items
                max_items = self.config["max_items_in_charts"]
                if len(sorted_data) > max_items:
                    sorted_data = sorted_data[:max_items]
                
                # Extract data for chart
                groups = [comp["group"] for comp in sorted_data]
                pct_changes = [comp["pct_change_mean"] for comp in sorted_data]
                is_improvements = [comp["is_improvement"] for comp in sorted_data]
                
                # Create hover text
                hover_text = [
                    f"Group: {comp['group']}<br>"
                    f"Current: {comp['current_mean']:.2f}<br>"
                    f"Historical: {comp['historical_mean']:.2f}<br>"
                    f"Change: {comp['pct_change_mean']:.2f}%<br>"
                    f"{'Improvement' if comp['is_improvement'] else 'Regression'}"
                    for comp in sorted_data
                ]
                
                # Add bar chart
                colors = [
                    "#2ecc71" if improve else "#e74c3c"  # green for improvement, red for regression
                    for improve in is_improvements
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=groups,
                        y=pct_changes,
                        marker_color=colors,
                        hovertext=hover_text,
                        hoverinfo="text"
                    ),
                    row=i+1, col=1
                )
                
                # Add zero line for reference
                fig.add_trace(
                    go.Scatter(
                        x=[groups[0], groups[-1]],
                        y=[0, 0],
                        mode="lines",
                        line=dict(
                            color="rgba(255, 255, 255, 0.5)",
                            width=1,
                            dash="dash"
                        ),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
                
                # Update axes for this subplot
                fig.update_xaxes(
                    title="Model-Hardware Pairs",
                    tickangle=-45,
                    gridcolor=color_scheme["grid_color"],
                    row=i+1, col=1
                )
                
                fig.update_yaxes(
                    title="% Change",
                    gridcolor=color_scheme["grid_color"],
                    tickformat=".1f",
                    row=i+1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title="Performance Metric Changes (Current vs. Historical)",
                height=self.config["chart_height"] * len(metrics),
                width=self.config["chart_width"],
                template="plotly_dark" if theme == "dark" else "plotly_white",
                paper_bgcolor=color_scheme["paper_bgcolor"],
                plot_bgcolor=color_scheme["background"],
                font=dict(color=color_scheme["text_color"]),
                showlegend=False,
                hovermode="closest"
            )
            
            # Write to file
            fig.write_html(output_path)
            
            # Return the path relative to output_dir for template usage
            return os.path.relpath(output_path, os.path.dirname(self.output_dir))
            
        except Exception as e:
            logger.error(f"Error creating performance comparison chart: {e}")
            return None
    
    def create_dashboard_result_summary(self, time_range_days=7) -> Dict[str, Any]:
        """
        Create a comprehensive summary of all result types for the dashboard.
        
        Args:
            time_range_days: Number of days to include in the summary
            
        Returns:
            Dictionary with summary data and visualization paths
        """
        if not self.result_aggregator:
            return {"error": "Result aggregator not available"}
        
        try:
            # Collect all summary data
            performance_summary = self.get_performance_summary(time_range_days)
            compatibility_summary = self.get_compatibility_summary(time_range_days)
            integration_summary = self.get_integration_test_summary(time_range_days)
            web_platform_summary = self.get_web_platform_summary(time_range_days)
            
            # Create visualizations
            perf_trend_chart = self.create_performance_trend_chart(
                time_range_days=time_range_days,
                metric="average_latency_ms"
            )
            
            compatibility_chart = self.create_compatibility_matrix_chart(
                time_range_days=time_range_days
            )
            
            test_pass_chart = self.create_test_pass_rate_chart(
                time_range_days=time_range_days
            )
            
            web_platform_chart = self.create_web_platform_browser_chart(
                time_range_days=time_range_days
            )
            
            anomaly_chart = self.create_anomaly_chart(
                time_range_days=time_range_days
            )
            
            comparison_chart = self.create_performance_comparison_chart(
                time_range_days=time_range_days
            )
            
            # Combine into dashboard summary
            dashboard_summary = {
                "time_range_days": time_range_days,
                "summaries": {
                    "performance": performance_summary,
                    "compatibility": compatibility_summary,
                    "integration": integration_summary,
                    "web_platform": web_platform_summary
                },
                "visualizations": {
                    "performance_trend": perf_trend_chart,
                    "compatibility_matrix": compatibility_chart,
                    "test_pass_rate": test_pass_chart,
                    "web_platform_browsers": web_platform_chart,
                    "anomalies": anomaly_chart,
                    "performance_comparison": comparison_chart
                },
                "overall_stats": {
                    "total_tests_run": (
                        performance_summary.get("total_tests", 0) +
                        integration_summary.get("total_tests_run", 0) +
                        web_platform_summary.get("total_tests_run", 0)
                    ),
                    "total_model_hardware_pairs": performance_summary.get("total_model_hardware_pairs", 0),
                    "total_anomalies": performance_summary.get("total_anomalies", 0),
                    "compatibility_rate": compatibility_summary.get("compatibility_rate", 0),
                    "integration_pass_rate": integration_summary.get("pass_rate", 0),
                    "web_platform_success_rate": web_platform_summary.get("success_rate", 0)
                },
                "completion_timestamp": datetime.now().isoformat()
            }
            
            return dashboard_summary
            
        except Exception as e:
            logger.error(f"Error creating dashboard result summary: {e}")
            return {"error": str(e)}

# For command-line usage
def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Result Aggregator Dashboard Integration")
    parser.add_argument("--output-dir", default="./monitoring_dashboard",
                       help="Output directory for visualizations")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to DuckDB database")
    parser.add_argument("--time-range", type=int, default=7,
                       help="Time range in days for visualizations")
    parser.add_argument("--theme", choices=["light", "dark"], default="dark",
                       help="Theme for visualizations")
    
    args = parser.parse_args()
    
    # Create database manager
    try:
        from duckdb_api.core.db_manager import BenchmarkDBManager
        db_manager = BenchmarkDBManager(args.db_path)
        print(f"Connected to database at {args.db_path}")
    except ImportError:
        print("BenchmarkDBManager not available. Using mock database.")
        db_manager = None
    
    # Create result aggregator
    try:
        from duckdb_api.distributed_testing.result_aggregator.service import ResultAggregatorService
        result_aggregator = ResultAggregatorService(db_manager=db_manager)
        print("Result aggregator created successfully")
    except ImportError:
        print("ResultAggregatorService not available. Using mock service.")
        result_aggregator = None
    
    # Create integration
    integration = ResultAggregatorIntegration(
        result_aggregator=result_aggregator,
        output_dir=args.output_dir
    )
    
    # Configure integration
    integration.configure({
        "theme": args.theme
    })
    
    # Create dashboard summary
    print(f"Creating dashboard summary for the last {args.time_range} days...")
    summary = integration.create_dashboard_result_summary(args.time_range)
    
    # Print summary information
    if "error" in summary:
        print(f"Error: {summary['error']}")
    else:
        print("\nDashboard Summary:")
        print(f"Total tests run: {summary['overall_stats']['total_tests_run']}")
        print(f"Model-hardware pairs: {summary['overall_stats']['total_model_hardware_pairs']}")
        print(f"Compatibility rate: {summary['overall_stats']['compatibility_rate']:.1f}%")
        print(f"Integration test pass rate: {summary['overall_stats']['integration_pass_rate']:.1f}%")
        print(f"Web platform success rate: {summary['overall_stats']['web_platform_success_rate']:.1f}%")
        
        print("\nVisualizations created:")
        for name, path in summary["visualizations"].items():
            if path:
                print(f"- {name}: {path}")

if __name__ == "__main__":
    main()