#!/usr/bin/env python3
"""
Fault Tolerance Visualization Module

This module provides visualization capabilities for the Hardware-Aware Fault Tolerance System,
allowing users to generate insightful visualizations of failure patterns, recovery strategies,
and system performance.

Implementation Date: March 13, 2025 (Earlier than the comprehensive monitoring dashboard planned for June 19-26, 2025)

This visualization system was implemented ahead of schedule as an enhancement to the hardware-aware
fault tolerance system and serves as a foundation for the comprehensive monitoring dashboard
planned for future development.
"""

import os
import json
import time
import logging
import matplotlib
# Use non-interactive backend for server environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter

# Import fault tolerance components
from duckdb_api.distributed_testing.hardware_aware_fault_tolerance import (
    FailureType, RecoveryStrategy, FailureContext, RecoveryAction
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("fault_tolerance_viz")


class FaultToleranceVisualizer:
    """Visualization engine for fault tolerance data."""
    
    def __init__(self, output_dir="./visualizations"):
        """
        Initialize the fault tolerance visualizer.
        
        Args:
            output_dir: Directory where visualization files will be saved
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default style
        plt.style.use('ggplot')
        
        logger.info(f"Fault tolerance visualizer initialized (output: {output_dir})")
    
    def visualize_failure_distribution(self, 
                                      failure_history: List[Dict[str, Any]], 
                                      title="Failure Distribution by Type and Hardware",
                                      filename="failure_distribution.png"):
        """
        Create a visualization of failure distribution by type and hardware.
        
        Args:
            failure_history: List of failure context dictionaries
            title: Title for the visualization
            filename: Output filename
            
        Returns:
            Path to the generated visualization file
        """
        if not failure_history:
            logger.warning("No failure history data provided for visualization")
            return None
        
        # Extract data for visualization
        failure_types = [f.get("error_type", "UNKNOWN") for f in failure_history]
        hardware_classes = [
            f.get("hardware_profile", {}).get("hardware_class", "UNKNOWN") 
            for f in failure_history
        ]
        
        # Count occurrences
        failure_type_counts = Counter(failure_types)
        hardware_class_counts = Counter(hardware_classes)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot failure types
        failure_types, ft_counts = zip(*failure_type_counts.items()) if failure_type_counts else ([], [])
        ax1.bar(failure_types, ft_counts, color='#3498db')
        ax1.set_title("Failures by Type")
        ax1.set_xlabel("Failure Type")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot hardware classes
        hw_classes, hw_counts = zip(*hardware_class_counts.items()) if hardware_class_counts else ([], [])
        ax2.bar(hw_classes, hw_counts, color='#e74c3c')
        ax2.set_title("Failures by Hardware Class")
        ax2.set_xlabel("Hardware Class")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis='x', rotation=45)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated failure distribution visualization: {output_path}")
        return output_path
    
    def visualize_recovery_effectiveness(self,
                                       recovery_history: Dict[str, List[Dict[str, Any]]],
                                       title="Recovery Strategy Effectiveness",
                                       filename="recovery_effectiveness.png"):
        """
        Create a visualization of recovery strategy effectiveness.
        
        Args:
            recovery_history: Dictionary mapping task IDs to lists of recovery actions
            title: Title for the visualization
            filename: Output filename
            
        Returns:
            Path to the generated visualization file
        """
        if not recovery_history:
            logger.warning("No recovery history data provided for visualization")
            return None
        
        # Flatten recovery history and extract strategies
        strategies = []
        for task_id, actions in recovery_history.items():
            for action in actions:
                strategy = action.get("strategy", "UNKNOWN")
                strategies.append(strategy)
        
        # Count strategy occurrences
        strategy_counts = Counter(strategies)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot strategies
        strategies, counts = zip(*strategy_counts.items()) if strategy_counts else ([], [])
        bars = ax.bar(strategies, counts, color='#2ecc71')
        
        # Add count labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_xlabel("Recovery Strategy")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        
        # Save the figure
        fig.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated recovery effectiveness visualization: {output_path}")
        return output_path
    
    def visualize_failure_timeline(self,
                                  failure_history: List[Dict[str, Any]],
                                  title="Failure Timeline",
                                  filename="failure_timeline.png"):
        """
        Create a timeline visualization of failures.
        
        Args:
            failure_history: List of failure context dictionaries
            title: Title for the visualization
            filename: Output filename
            
        Returns:
            Path to the generated visualization file
        """
        if not failure_history:
            logger.warning("No failure history data provided for visualization")
            return None
        
        # Extract timestamps and convert to datetime
        timestamps = []
        error_types = []
        
        for failure in failure_history:
            # Get timestamp
            ts_str = failure.get("timestamp")
            if ts_str:
                try:
                    timestamp = datetime.fromisoformat(ts_str)
                    timestamps.append(timestamp)
                    error_types.append(failure.get("error_type", "UNKNOWN"))
                except (ValueError, TypeError):
                    # Skip entries with invalid timestamps
                    pass
        
        if not timestamps:
            logger.warning("No valid timestamp data for timeline visualization")
            return None
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'timestamp': timestamps,
            'error_type': error_types
        })
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Count failures by day and type
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby(['date', 'error_type']).size().unstack(fill_value=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot daily counts as stacked area chart
        daily_counts.plot(kind='area', stacked=True, alpha=0.7, ax=ax)
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Failures")
        ax.legend(title="Failure Type")
        
        # Save the figure
        fig.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated failure timeline visualization: {output_path}")
        return output_path
    
    def visualize_recovery_success_rates(self,
                                        ml_detector,
                                        title="Recovery Strategy Success Rates by Hardware",
                                        filename="recovery_success_rates.png"):
        """
        Create a visualization of recovery strategy success rates.
        
        Args:
            ml_detector: ML pattern detector with strategy success rate data
            title: Title for the visualization
            filename: Output filename
            
        Returns:
            Path to the generated visualization file
        """
        if not ml_detector or not hasattr(ml_detector, 'strategy_success_rates'):
            logger.warning("No ML detector provided or no success rate data available")
            return None
        
        # Extract success rate data
        success_rates = {}
        
        for hw_class, strategies in ml_detector.strategy_success_rates.items():
            for strategy, counts in strategies.items():
                total = counts.get("total", 0)
                if total > 0:
                    success = counts.get("success", 0)
                    rate = success / total
                    
                    if hw_class not in success_rates:
                        success_rates[hw_class] = {}
                    
                    success_rates[hw_class][strategy] = {
                        "rate": rate,
                        "total": total
                    }
        
        if not success_rates:
            logger.warning("No success rate data available for visualization")
            return None
        
        # Create figure with subplots for each hardware class
        n_classes = len(success_rates)
        fig_height = 5 * n_classes
        fig, axes = plt.subplots(n_classes, 1, figsize=(12, fig_height))
        
        # Handle case with only one hardware class
        if n_classes == 1:
            axes = [axes]
        
        # Plot success rates for each hardware class
        for i, (hw_class, strategies) in enumerate(success_rates.items()):
            ax = axes[i]
            
            strategy_names = list(strategies.keys())
            rates = [strategies[s]["rate"] for s in strategy_names]
            totals = [strategies[s]["total"] for s in strategy_names]
            
            # Create bar heights based on success rates
            bars = ax.bar(strategy_names, rates, color='#9b59b6')
            
            # Add rate labels above bars
            for bar, total in zip(bars, totals):
                height = bar.get_height()
                ax.annotate(f'{height:.1%} ({total})',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            ax.set_title(f"Success Rates for {hw_class}")
            ax.set_xlabel("Recovery Strategy")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1.1)  # Limit y-axis to 0-110%
            ax.tick_params(axis='x', rotation=45)
            
            # Add a horizontal line at 50% success rate
            ax.axhline(y=0.5, linestyle='--', alpha=0.5, color='black')
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated recovery success rates visualization: {output_path}")
        return output_path
    
    def visualize_ml_patterns(self,
                             ml_patterns,
                             title="Machine Learning-Detected Patterns",
                             filename="ml_patterns.png"):
        """
        Create a visualization of ML-detected patterns.
        
        Args:
            ml_patterns: List of ML-detected patterns
            title: Title for the visualization
            filename: Output filename
            
        Returns:
            Path to the generated visualization file
        """
        if not ml_patterns:
            logger.warning("No ML patterns provided for visualization")
            return None
        
        # Extract data for visualization
        pattern_types = [p.get("type", "UNKNOWN") for p in ml_patterns]
        confidences = [p.get("confidence", 0.0) for p in ml_patterns]
        pattern_ids = [f"Pattern {i+1}" for i in range(len(ml_patterns))]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot pattern types distribution
        type_counts = Counter(pattern_types)
        types, counts = zip(*type_counts.items()) if type_counts else ([], [])
        ax1.bar(types, counts, color='#1abc9c')
        ax1.set_title("Pattern Types")
        ax1.set_xlabel("Pattern Type")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot confidence scores
        if pattern_ids and confidences:
            # Sort by confidence for better visualization
            sorted_data = sorted(zip(pattern_ids, confidences), key=lambda x: x[1], reverse=True)
            pattern_ids, confidences = zip(*sorted_data)
            
            bars = ax2.bar(pattern_ids, confidences, color='#f39c12')
            
            # Add confidence labels above bars
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            ax2.set_title("Pattern Confidence Scores")
            ax2.set_xlabel("Pattern")
            ax2.set_ylabel("Confidence Score")
            ax2.set_ylim(0, 1.1)  # Limit y-axis to 0-1.1
            ax2.tick_params(axis='x', rotation=45)
            
            # Add a horizontal line at 0.7 confidence threshold
            ax2.axhline(y=0.7, linestyle='--', alpha=0.5, color='black')
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated ML patterns visualization: {output_path}")
        return output_path
    
    def create_hardware_failure_heatmap(self,
                                      failure_history: List[Dict[str, Any]],
                                      title="Hardware Failure Heatmap",
                                      filename="hardware_failure_heatmap.png"):
        """
        Create a heatmap visualization of failures by hardware class and error type.
        
        Args:
            failure_history: List of failure context dictionaries
            title: Title for the visualization
            filename: Output filename
            
        Returns:
            Path to the generated visualization file
        """
        if not failure_history:
            logger.warning("No failure history data provided for heatmap")
            return None
        
        # Extract hardware classes and error types
        hardware_classes = []
        error_types = []
        
        for failure in failure_history:
            hw_class = failure.get("hardware_profile", {}).get("hardware_class", "UNKNOWN")
            error_type = failure.get("error_type", "UNKNOWN")
            
            hardware_classes.append(hw_class)
            error_types.append(error_type)
        
        # Count occurrences for each combination
        hw_error_counts = defaultdict(lambda: defaultdict(int))
        
        for hw_class, error_type in zip(hardware_classes, error_types):
            hw_error_counts[hw_class][error_type] += 1
        
        # Convert to DataFrame for easier plotting
        unique_hw_classes = sorted(set(hardware_classes))
        unique_error_types = sorted(set(error_types))
        
        data = []
        for hw_class in unique_hw_classes:
            row = []
            for error_type in unique_error_types:
                count = hw_error_counts[hw_class][error_type]
                row.append(count)
            data.append(row)
        
        df = pd.DataFrame(data, index=unique_hw_classes, columns=unique_error_types)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(df.values, cmap='YlOrRd')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
        
        # Show all ticks and label them
        ax.set_xticks(np.arange(len(unique_error_types)))
        ax.set_yticks(np.arange(len(unique_hw_classes)))
        ax.set_xticklabels(unique_error_types)
        ax.set_yticklabels(unique_hw_classes)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(unique_hw_classes)):
            for j in range(len(unique_error_types)):
                text = ax.text(j, i, df.iloc[i, j],
                              ha="center", va="center", color="black")
        
        ax.set_title(title)
        fig.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated hardware failure heatmap: {output_path}")
        return output_path
    
    def create_comprehensive_report(self,
                                  fault_tolerance_manager,
                                  ml_detector=None,
                                  output_filename="fault_tolerance_report.html"):
        """
        Create a comprehensive HTML report with multiple visualizations.
        
        Args:
            fault_tolerance_manager: Hardware-aware fault tolerance manager instance
            ml_detector: Optional ML pattern detector instance
            output_filename: Output HTML filename
            
        Returns:
            Path to the generated HTML report file
        """
        try:
            # Prepare data from fault tolerance manager
            failure_history = []
            recovery_history = {}
            ml_patterns = []
            
            if fault_tolerance_manager:
                # Extract failure history
                for f_contexts in fault_tolerance_manager.failure_history:
                    failure_history.append({
                        "task_id": f_contexts.task_id,
                        "worker_id": f_contexts.worker_id,
                        "error_type": f_contexts.error_type.name,
                        "timestamp": f_contexts.timestamp.isoformat(),
                        "hardware_profile": {
                            "hardware_class": f_contexts.hardware_profile.hardware_class.name
                            if f_contexts.hardware_profile else "UNKNOWN"
                        }
                    })
                
                # Extract recovery history
                for task_id, actions in fault_tolerance_manager.recovery_history.items():
                    recovery_history[task_id] = [
                        {
                            "strategy": action.strategy.name,
                            "worker_id": action.worker_id,
                            "message": action.message
                        }
                        for action in actions
                    ]
            
            # Get ML patterns if detector is available
            if ml_detector:
                ml_patterns = ml_detector.detect_patterns()
            
            # Generate visualizations
            vis_paths = []
            
            # Failure distribution
            if failure_history:
                failure_dist_path = self.visualize_failure_distribution(failure_history)
                if failure_dist_path:
                    vis_paths.append(("Failure Distribution", os.path.basename(failure_dist_path)))
            
            # Recovery effectiveness
            if recovery_history:
                recovery_eff_path = self.visualize_recovery_effectiveness(recovery_history)
                if recovery_eff_path:
                    vis_paths.append(("Recovery Effectiveness", os.path.basename(recovery_eff_path)))
            
            # Failure timeline
            if failure_history:
                timeline_path = self.visualize_failure_timeline(failure_history)
                if timeline_path:
                    vis_paths.append(("Failure Timeline", os.path.basename(timeline_path)))
            
            # Hardware failure heatmap
            if failure_history:
                heatmap_path = self.create_hardware_failure_heatmap(failure_history)
                if heatmap_path:
                    vis_paths.append(("Hardware Failure Heatmap", os.path.basename(heatmap_path)))
            
            # Recovery success rates
            if ml_detector:
                success_rates_path = self.visualize_recovery_success_rates(ml_detector)
                if success_rates_path:
                    vis_paths.append(("Recovery Success Rates", os.path.basename(success_rates_path)))
            
            # ML patterns
            if ml_patterns:
                ml_patterns_path = self.visualize_ml_patterns(ml_patterns)
                if ml_patterns_path:
                    vis_paths.append(("ML-Detected Patterns", os.path.basename(ml_patterns_path)))
            
            # Generate HTML report
            html_content = self._generate_html_report(vis_paths, fault_tolerance_manager, ml_detector)
            
            # Save HTML report
            output_path = os.path.join(self.output_dir, output_filename)
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated comprehensive fault tolerance report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating comprehensive report: {e}")
            return None
    
    def _generate_html_report(self, vis_paths, fault_tolerance_manager, ml_detector):
        """
        Generate HTML content for the comprehensive report.
        
        Args:
            vis_paths: List of (title, path) tuples for visualizations
            fault_tolerance_manager: Fault tolerance manager instance
            ml_detector: ML pattern detector instance
            
        Returns:
            HTML content as a string
        """
        # Get statistics
        total_failures = len(fault_tolerance_manager.failure_history) if fault_tolerance_manager else 0
        total_patterns = len(fault_tolerance_manager.failure_patterns) if fault_tolerance_manager else 0
        total_ml_patterns = len(ml_detector.detect_patterns()) if ml_detector else 0
        
        # Get timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Hardware-Aware Fault Tolerance Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .report-header {{
                    background-color: #3498db;
                    color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .stats-container {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    width: 30%;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .stat-value {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .visualizations {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: space-between;
                    margin-bottom: 30px;
                }}
                .visualization-item {{
                    width: 48%;
                    margin-bottom: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .visualization-item img {{
                    width: 100%;
                    height: auto;
                    border-radius: 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    color: #7f8c8d;
                }}
                @media (max-width: 768px) {{
                    .stats-container {{
                        flex-direction: column;
                    }}
                    .stat-card {{
                        width: 100%;
                        margin-bottom: 15px;
                    }}
                    .visualization-item {{
                        width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <h1>Hardware-Aware Fault Tolerance Report</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="stats-container">
                <div class="stat-card">
                    <h3>Total Failures</h3>
                    <div class="stat-value">{total_failures}</div>
                </div>
                <div class="stat-card">
                    <h3>Detected Patterns</h3>
                    <div class="stat-value">{total_patterns}</div>
                </div>
                <div class="stat-card">
                    <h3>ML-Detected Patterns</h3>
                    <div class="stat-value">{total_ml_patterns}</div>
                </div>
            </div>
            
            <h2>Visualizations</h2>
            <div class="visualizations">
        """
        
        # Add visualization items
        for title, img_path in vis_paths:
            html += f"""
                <div class="visualization-item">
                    <h3>{title}</h3>
                    <img src="{img_path}" alt="{title}">
                </div>
            """
        
        # Add footer and close HTML
        html += """
            </div>
            
            <div class="footer">
                <p>Generated by Hardware-Aware Fault Tolerance Visualizer</p>
            </div>
        </body>
        </html>
        """
        
        return html


# Utility functions for easier access to visualization capabilities
def create_fault_tolerance_visualizer(output_dir="./visualizations"):
    """Create a fault tolerance visualizer instance."""
    return FaultToleranceVisualizer(output_dir=output_dir)

def visualize_fault_tolerance_system(fault_tolerance_manager, output_dir="./visualizations"):
    """
    Create a comprehensive visualization of a fault tolerance system.
    
    Args:
        fault_tolerance_manager: Hardware-aware fault tolerance manager instance
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the generated report
    """
    visualizer = FaultToleranceVisualizer(output_dir=output_dir)
    
    # Get ML detector if available
    ml_detector = getattr(fault_tolerance_manager, "ml_detector", None)
    
    # Create comprehensive report
    return visualizer.create_comprehensive_report(
        fault_tolerance_manager=fault_tolerance_manager,
        ml_detector=ml_detector
    )