#!/usr/bin/env python3
"""
Error Recovery Visualization for Distributed Testing Framework

This module provides visualization tools for the Error Recovery system with performance tracking,
allowing for better understanding of recovery strategy effectiveness, performance trends,
and system health during error recovery operations.
"""

import os
import sys
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import argparse

# Try to import visualization libraries, with graceful fallback
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import numpy as np
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("error_recovery_visualization")


class ErrorRecoveryVisualizer:
    """
    Visualization tools for the error recovery system with performance tracking.
    
    This class provides methods to visualize various aspects of the error recovery system, including:
    - Strategy performance comparison
    - Success rate trends over time
    - Recovery time analysis
    - Impact and stability metrics
    - Progressive recovery statistics
    - Hardware-aware recovery effectiveness
    """
    
    def __init__(self, output_dir: str = None, file_format: str = "png"):
        """
        Initialize the error recovery visualizer.
        
        Args:
            output_dir: Directory to save visualization files (default: current directory)
            file_format: File format for saved visualizations (png, pdf, svg)
        """
        # Check if visualization libraries are available
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib is not available. Visualizations will be limited.")
        
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas is not available. Data processing capabilities will be limited.")
        
        # Set output directory
        self.output_dir = output_dir or os.path.join(os.getcwd(), "images")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set file format
        self.file_format = file_format
        
        # Set styles for consistent visualization
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8-darkgrid')
            
    def visualize_strategy_performance(self, 
                                   performance_data: Dict[str, Any],
                                   filename: str = "strategy_performance_dashboard") -> str:
        """
        Visualize performance metrics for recovery strategies.
        
        Args:
            performance_data: Performance data from get_performance_metrics()
            filename: Base filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib is required for this visualization"
        
        # Extract data
        strategy_stats = performance_data.get("strategy_stats", {})
        
        if not strategy_stats:
            logger.warning("No strategy performance data available")
            return "No data available"
        
        # Prepare data for visualization
        strategy_ids = []
        success_rates = []
        recovery_times = []
        overall_scores = []
        sample_counts = []
        
        for strategy_id, stats in strategy_stats.items():
            strategy_ids.append(strategy_id)
            success_rates.append(stats["success_rate"])
            recovery_times.append(stats["avg_recovery_time"])
            overall_scores.append(stats["overall_score"])
            sample_counts.append(stats["total_samples"])
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Success Rate bar chart (top left)
        ax1 = plt.subplot(2, 2, 1)
        bars = ax1.bar(strategy_ids, success_rates, color='green', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_title('Strategy Success Rates')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Success Rate (0-1)')
        ax1.set_ylim(0, 1.1)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        
        # 2. Recovery Time bar chart (top right)
        ax2 = plt.subplot(2, 2, 2)
        colors = ['green' if t < 10 else 'orange' if t < 30 else 'red' for t in recovery_times]
        bars = ax2.bar(strategy_ids, recovery_times, color=colors, alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
        
        ax2.set_title('Average Recovery Time')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_ylim(0, max(recovery_times) * 1.2)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        
        # 3. Overall Score bar chart (bottom left)
        ax3 = plt.subplot(2, 2, 3)
        colors = ['red' if s < 0.4 else 'orange' if s < 0.7 else 'green' for s in overall_scores]
        bars = ax3.bar(strategy_ids, overall_scores, color=colors, alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_title('Overall Performance Score')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Score (0-1)')
        ax3.set_ylim(0, 1.1)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
        
        # 4. Sample Count bar chart (bottom right)
        ax4 = plt.subplot(2, 2, 4)
        bars = ax4.bar(strategy_ids, sample_counts, color='blue', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_title('Sample Count')
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('Number of Samples')
        ax4.set_ylim(0, max(sample_counts) * 1.2)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
        
        # Add title and adjust layout
        fig.suptitle('Recovery Strategy Performance Dashboard', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_error_recovery_heatmap(self,
                                    performance_data: Dict[str, Any],
                                    filename: str = "error_recovery_heatmap") -> str:
        """
        Create a heatmap visualization showing strategy effectiveness for different error types.
        
        Args:
            performance_data: Performance data from get_performance_metrics()
            filename: Base filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib is required for this visualization"
        
        # Extract data
        strategy_stats = performance_data.get("strategy_stats", {})
        
        if not strategy_stats:
            logger.warning("No strategy performance data available")
            return "No data available"
        
        # Collect error types and strategies
        error_types = set()
        strategies = {}
        
        for strategy_id, stats in strategy_stats.items():
            strategies[strategy_id] = stats["name"]
            
            for error_type in stats.get("by_error_type", {}):
                error_types.add(error_type)
        
        error_types = sorted(list(error_types))
        strategy_ids = sorted(list(strategies.keys()))
        
        if not error_types or not strategy_ids:
            logger.warning("Insufficient data for heatmap")
            return "Insufficient data"
        
        # Create data matrix for heatmap
        data = np.zeros((len(strategy_ids), len(error_types)))
        
        for i, strategy_id in enumerate(strategy_ids):
            stats = strategy_stats.get(strategy_id, {})
            by_error_type = stats.get("by_error_type", {})
            
            for j, error_type in enumerate(error_types):
                if error_type in by_error_type:
                    data[i, j] = by_error_type[error_type].get("overall_score", 0.0)
        
        # Create heatmap visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(data, cmap='YlGnBu', aspect='auto')
        
        # Set tick labels
        ax.set_xticks(np.arange(len(error_types)))
        ax.set_yticks(np.arange(len(strategy_ids)))
        ax.set_xticklabels(error_types)
        ax.set_yticklabels([f"{sid} ({strategies[sid]})" for sid in strategy_ids])
        
        # Rotate the x-tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Effectiveness Score", rotation=-90, va="bottom")
        
        # Add values in each cell
        for i in range(len(strategy_ids)):
            for j in range(len(error_types)):
                text = ax.text(j, i, f"{data[i, j]:.2f}",
                               ha="center", va="center", color="black" if data[i, j] > 0.6 else "white")
        
        # Add title and labels
        ax.set_title("Strategy Effectiveness by Error Type")
        ax.set_xlabel("Error Type")
        ax.set_ylabel("Recovery Strategy")
        
        # Adjust layout
        fig.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_performance_trends(self,
                                 time_series_data: List[Dict[str, Any]],
                                 filename: str = "performance_trend_graphs") -> str:
        """
        Visualize performance trends over time.
        
        Args:
            time_series_data: Time series data from analyze_recovery_performance()["time_series"]
            filename: Base filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib is required for this visualization"
        
        if not time_series_data:
            logger.warning("No time series data available")
            return "No data available"
        
        # Extract data
        dates = [datetime.strptime(entry["date"], "%Y-%m-%d") for entry in time_series_data]
        recovery_counts = [entry["total_recoveries"] for entry in time_series_data]
        success_rates = [entry["success_rate"] for entry in time_series_data]
        execution_times = [entry["avg_execution_time"] for entry in time_series_data]
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # 1. Recovery Count trend
        ax1.plot(dates, recovery_counts, marker='o', linestyle='-', color='blue', alpha=0.7)
        ax1.set_title('Daily Recovery Operations')
        ax1.set_ylabel('Number of Recoveries')
        ax1.grid(True, alpha=0.3)
        
        # 2. Success Rate trend
        ax2.plot(dates, success_rates, marker='o', linestyle='-', color='green', alpha=0.7)
        ax2.set_title('Daily Success Rate')
        ax2.set_ylabel('Success Rate (0-1)')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # 3. Execution Time trend
        ax3.plot(dates, execution_times, marker='o', linestyle='-', color='orange', alpha=0.7)
        ax3.set_title('Average Recovery Time')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
        fig.autofmt_xdate()
        
        # Add overall title
        fig.suptitle('Error Recovery Performance Trends', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_progressive_recovery(self,
                                   progressive_data: Dict[str, Any],
                                   filename: str = "progressive_recovery_analysis") -> str:
        """
        Visualize progressive recovery statistics.
        
        Args:
            progressive_data: Progressive recovery data from get_progressive_recovery_history()
            filename: Base filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib is required for this visualization"
        
        # Extract summary data
        summary = progressive_data.get("summary", {})
        
        if not summary:
            logger.warning("No progressive recovery data available")
            return "No data available"
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(12, 10))
        
        # 1. Recovery Level Distribution (pie chart)
        ax1 = plt.subplot(2, 2, 1)
        
        labels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
        sizes = [
            summary.get("level_1_count", 0),
            summary.get("level_2_count", 0),
            summary.get("level_3_count", 0),
            summary.get("level_4_count", 0),
            summary.get("level_5_count", 0)
        ]
        
        # Filter out zero values
        non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
        non_zero_sizes = [size for size in sizes if size > 0]
        
        if non_zero_sizes:
            colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
            # Filter colors to match non-zero data
            non_zero_colors = [colors[i] for i, size in enumerate(sizes) if size > 0]
            
            ax1.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, autopct='%1.1f%%',
                   startangle=90, shadow=False)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax1.set_title('Recovery Level Distribution')
        else:
            ax1.text(0.5, 0.5, 'No data available', horizontalalignment='center',
                    verticalalignment='center', transform=ax1.transAxes)
        
        # 2. Success vs. Failure (pie chart)
        ax2 = plt.subplot(2, 2, 2)
        
        success_count = summary.get("successful_recoveries", 0)
        failure_count = summary.get("failed_recoveries", 0)
        
        if success_count > 0 or failure_count > 0:
            labels = ['Success', 'Failure']
            sizes = [success_count, failure_count]
            colors = ['#4CAF50', '#F44336']
            
            # Filter out zero values
            non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
            non_zero_sizes = [size for size in sizes if size > 0]
            non_zero_colors = [color for color, size in zip(colors, sizes) if size > 0]
            
            ax2.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, autopct='%1.1f%%',
                   startangle=90, shadow=False)
            ax2.axis('equal')
        else:
            ax2.text(0.5, 0.5, 'No data available', horizontalalignment='center',
                    verticalalignment='center', transform=ax2.transAxes)
            
        ax2.set_title('Recovery Success vs. Failure')
        
        # 3. Progression Analysis (bar chart)
        ax3 = plt.subplot(2, 1, 2)
        
        # Count number of errors at each level
        level_counts = [
            summary.get("level_1_count", 0),
            summary.get("level_2_count", 0),
            summary.get("level_3_count", 0),
            summary.get("level_4_count", 0),
            summary.get("level_5_count", 0)
        ]
        
        # Calculate progression rates (percentage of errors that escalated to next level)
        progression_rates = []
        
        for i in range(4):  # Only 4 transitions (1→2, 2→3, 3→4, 4→5)
            if level_counts[i] > 0:
                rate = level_counts[i+1] / level_counts[i]
                progression_rates.append(rate * 100)  # Convert to percentage
            else:
                progression_rates.append(0)
        
        if any(level_counts):
            # Plot progression rates
            x = range(len(progression_rates))
            bars = ax3.bar([f'Level {i+1} → {i+2}' for i in x], progression_rates, color='#FF9800', alpha=0.7)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax3.set_title('Error Progression Rates')
            ax3.set_xlabel('Progression')
            ax3.set_ylabel('Percentage of Errors that Escalated')
            ax3.set_ylim(0, max(progression_rates + [5]) * 1.2)  # Ensure space for labels
        else:
            ax3.text(0.5, 0.5, 'No data available', horizontalalignment='center',
                    verticalalignment='center', transform=ax3.transAxes)
        
        # Add title and adjust layout
        fig.suptitle('Progressive Recovery Analysis', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def generate_recovery_dashboard(self,
                                performance_data: Dict[str, Any],
                                time_series_data: List[Dict[str, Any]],
                                progressive_data: Dict[str, Any],
                                filename: str = "recovery_dashboard") -> str:
        """
        Generate a comprehensive error recovery dashboard combining multiple visualizations.
        
        Args:
            performance_data: Performance data from get_performance_metrics()
            time_series_data: Time series data from analyze_recovery_performance()["time_series"]
            progressive_data: Progressive recovery data from get_progressive_recovery_history()
            filename: Base filename for the dashboard
            
        Returns:
            Path to the saved dashboard file
        """
        # Generate individual visualizations
        strategy_perf_path = self.visualize_strategy_performance(performance_data)
        heatmap_path = self.visualize_error_recovery_heatmap(performance_data)
        trends_path = self.visualize_performance_trends(time_series_data)
        progressive_path = self.visualize_progressive_recovery(progressive_data)
        
        # Generate HTML dashboard that includes all visualizations
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error Recovery Performance Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                h1, h2, h3 {{ color: #333; }}
                .dashboard-container {{ display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px; margin: 20px 0; }}
                .dashboard-item {{ background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 15px; }}
                .dashboard-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .full-width {{ grid-column: 1 / span 2; }}
                .summary-box {{ background-color: #e9f5ff; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); grid-gap: 10px; }}
                .stat-item {{ background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .stat-number {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .poor {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Error Recovery Performance Dashboard</h1>
            
            <div class="summary-box">
                <h2>Recovery System Overview</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div>Total Recoveries</div>
                        <div class="stat-number">{progressive_data.get("summary", {}).get("successful_recoveries", 0) + progressive_data.get("summary", {}).get("failed_recoveries", 0)}</div>
                    </div>
                    <div class="stat-item">
                        <div>Success Rate</div>
                        <div class="stat-number {self._get_success_rate_class(performance_data.get("summary", {}).get("average_success_rate", 0))}">
                            {performance_data.get("summary", {}).get("average_success_rate", 0):.1%}
                        </div>
                    </div>
                    <div class="stat-item">
                        <div>Avg Recovery Time</div>
                        <div class="stat-number {self._get_recovery_time_class(performance_data.get("summary", {}).get("average_recovery_time", 0))}">
                            {performance_data.get("summary", {}).get("average_recovery_time", 0):.1f}s
                        </div>
                    </div>
                    <div class="stat-item">
                        <div>Strategies</div>
                        <div class="stat-number">{performance_data.get("summary", {}).get("total_strategies", 0)}</div>
                    </div>
                </div>
            </div>
            
            <div class="dashboard-container">
                <div class="dashboard-item">
                    <h3>Strategy Performance</h3>
                    <img src="{os.path.basename(strategy_perf_path)}" alt="Strategy Performance Dashboard">
                </div>
                <div class="dashboard-item">
                    <h3>Error Recovery Heatmap</h3>
                    <img src="{os.path.basename(heatmap_path)}" alt="Error Recovery Heatmap">
                </div>
                <div class="dashboard-item">
                    <h3>Performance Trends</h3>
                    <img src="{os.path.basename(trends_path)}" alt="Performance Trend Graphs">
                </div>
                <div class="dashboard-item">
                    <h3>Progressive Recovery Analysis</h3>
                    <img src="{os.path.basename(progressive_path)}" alt="Progressive Recovery Analysis">
                </div>
            </div>
            
            <div class="dashboard-item full-width">
                <h2>Top Performing Strategies by Error Type</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f2f2f2;">
                        <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Error Type</th>
                        <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Best Strategy</th>
                        <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Score</th>
                    </tr>
        """
        
        # Add rows for top strategies
        top_strategies = performance_data.get("top_strategies", {})
        for error_type, strategy_info in sorted(top_strategies.items()):
            score = strategy_info.get("score", 0)
            score_class = "good" if score >= 0.7 else "warning" if score >= 0.4 else "poor"
            
            html_content += f"""
                    <tr>
                        <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">{error_type}</td>
                        <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">{strategy_info.get("strategy_name", "Unknown")}</td>
                        <td style="padding: 8px; text-align: left; border: 1px solid #ddd;" class="{score_class}">{score:.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div style="text-align: center; margin-top: 20px; color: #777; font-size: 12px;">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </body>
        </html>
        """
        
        # Save HTML dashboard
        output_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _get_success_rate_class(self, rate: float) -> str:
        """Get CSS class for success rate value."""
        if rate >= 0.7:
            return "good"
        elif rate >= 0.4:
            return "warning"
        else:
            return "poor"
    
    def _get_recovery_time_class(self, time_seconds: float) -> str:
        """Get CSS class for recovery time value."""
        if time_seconds <= 10:
            return "good"
        elif time_seconds <= 30:
            return "warning"
        else:
            return "poor"


# Main function to run as a standalone tool
def main():
    """Main function when the module is run as a script."""
    parser = argparse.ArgumentParser(description="Visualize error recovery performance metrics")
    parser.add_argument("--db-path", type=str, help="Path to the database file")
    parser.add_argument("--output-dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--days", type=int, default=30, help="Number of days of history to analyze")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output file format")
    parser.add_argument("--generate-dashboard", action="store_true", help="Generate comprehensive dashboard")
    args = parser.parse_args()
    
    # Import necessary modules only when needed
    import duckdb
    from error_recovery_with_performance_tracking import PerformanceBasedErrorRecovery
    
    # Connect to database
    if args.db_path:
        db_connection = duckdb.connect(args.db_path)
    else:
        # Try to connect to default path or create in-memory database
        try:
            db_connection = duckdb.connect("./error_recovery_with_performance_test_db/recovery_data.duckdb")
        except:
            logger.warning("No database file specified, using in-memory database")
            db_connection = duckdb.connect(":memory:")
    
    # Create dummy coordinator for interface
    class DummyCoordinator:
        def __init__(self):
            self.db = db_connection
    
    # Create recovery system
    recovery_system = PerformanceBasedErrorRecovery(
        error_handler=None,
        coordinator=DummyCoordinator(),
        db_connection=db_connection
    )
    
    # Get data for visualizations
    performance_data = recovery_system.get_performance_metrics()
    progressive_data = recovery_system.get_progressive_recovery_history()
    
    # Create asynchronous function to run analyze_recovery_performance
    
    async def get_analysis_data():
        return await recovery_system.analyze_recovery_performance(days=args.days)
    
    # Run the async function
    analysis_data = anyio.run(get_analysis_data())
    time_series_data = analysis_data.get("time_series", [])
    
    # Create visualizer
    visualizer = ErrorRecoveryVisualizer(output_dir=args.output_dir, file_format=args.format)
    
    # Generate visualizations
    strategy_perf_path = visualizer.visualize_strategy_performance(performance_data)
    heatmap_path = visualizer.visualize_error_recovery_heatmap(performance_data)
    trends_path = visualizer.visualize_performance_trends(time_series_data)
    progressive_path = visualizer.visualize_progressive_recovery(progressive_data)
    
    logger.info(f"Strategy performance visualization saved to: {strategy_perf_path}")
    logger.info(f"Error recovery heatmap saved to: {heatmap_path}")
    logger.info(f"Performance trends visualization saved to: {trends_path}")
    logger.info(f"Progressive recovery visualization saved to: {progressive_path}")
    
    # Generate dashboard if requested
    if args.generate_dashboard:
        dashboard_path = visualizer.generate_recovery_dashboard(
            performance_data,
            time_series_data,
            progressive_data
        )
        logger.info(f"Recovery dashboard saved to: {dashboard_path}")
    
    print(f"Visualizations saved to: {visualizer.output_dir}")


if __name__ == "__main__":
    main()