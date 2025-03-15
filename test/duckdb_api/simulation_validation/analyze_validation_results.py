#!/usr/bin/env python3
"""
Validation Results Analyzer for Simulation Validation Framework

This module analyzes validation results from the Simulation Validation Framework
and generates summary reports in different formats (text, markdown, HTML, JSON).
It provides insights into validation accuracy, performance across hardware/model
combinations, and trends over time.
"""

import os
import sys
import json
import argparse
import logging
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

# Optional imports - used for visualization if available
try:
    import numpy as np
    import pandas as pd
    np_available = True
except ImportError:
    np_available = False

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analyze_validation_results")

class ValidationResultsAnalyzer:
    """
    Analyzes validation results from the Simulation Validation Framework.
    """
    
    def __init__(self):
        """Initialize the validation results analyzer."""
        pass
    
    def load_validation_results(self, results_dir: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load validation results from a directory.
        
        Args:
            results_dir: Directory containing validation results
            run_id: Optional run ID to filter results
            
        Returns:
            Dictionary of validation results
        """
        try:
            # Look for validation results in the directory
            validation_files = []
            
            # If run_id is provided, look for files matching that run ID
            if run_id:
                patterns = [
                    f"validation_results_{run_id}.json",
                    f"validation_summary_{run_id}.json",
                    f"results_{run_id}.json"
                ]
                
                for pattern in patterns:
                    matches = list(Path(results_dir).glob(pattern))
                    if matches:
                        validation_files.extend(matches)
            
            # If no files found with run_id, look for standard filenames
            if not validation_files:
                patterns = [
                    "validation_results.json",
                    "simulation_vs_hardware_results.json",
                    "validation_summary.json",
                    "results.json"
                ]
                
                for pattern in patterns:
                    file_path = os.path.join(results_dir, pattern)
                    if os.path.exists(file_path):
                        validation_files.append(Path(file_path))
            
            # If still no files found, look for any JSON file with validation in name
            if not validation_files:
                validation_files = list(Path(results_dir).glob("*validation*.json"))
            
            # If still no files found, try finding any JSON file
            if not validation_files:
                validation_files = list(Path(results_dir).glob("*.json"))
                
            # If no files found at all, return empty results
            if not validation_files:
                logger.error(f"No validation results found in {results_dir}")
                return {"status": "error", "message": "No validation results found"}
            
            # Load the first valid validation file
            for file_path in validation_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        # Check if this is a valid results file
                        if "validation_items" in data or "summary" in data:
                            logger.info(f"Loaded validation results from {file_path}")
                            return data
                        
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
            
            # If we get here, no valid validation results were found
            logger.error(f"No valid validation results found in {results_dir}")
            return {"status": "error", "message": "No valid validation results found"}
            
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")
            return {"status": "error", "message": f"Error loading validation results: {e}"}
    
    def analyze_results(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze validation results to extract insights.
        
        Args:
            validation_results: Validation results data
            
        Returns:
            Dictionary with validation analysis
        """
        # Check for errors in results
        if validation_results.get("status") == "error":
            return validation_results
        
        # Extract key components
        validation_items = validation_results.get("validation_items", [])
        summary = validation_results.get("summary", {})
        
        if not validation_items and not summary:
            return {
                "status": "error", 
                "message": "No validation items or summary found in results"
            }
        
        # Extract metrics from the first validation item to determine available metrics
        available_metrics = set()
        if validation_items:
            metrics_comparison = validation_items[0].get("metrics_comparison", {})
            available_metrics = set(metrics_comparison.keys())
        
        # Group validation items by hardware and model
        hardware_model_pairs = {}
        
        for item in validation_items:
            hardware_id = item.get("hardware_id", "unknown")
            model_id = item.get("model_id", "unknown")
            key = f"{hardware_id}_{model_id}"
            
            if key not in hardware_model_pairs:
                hardware_model_pairs[key] = {
                    "hardware_id": hardware_id,
                    "model_id": model_id,
                    "items": []
                }
            
            hardware_model_pairs[key]["items"].append(item)
        
        # Calculate metrics by hardware-model pair
        hardware_model_metrics = {}
        
        for key, pair_data in hardware_model_pairs.items():
            hardware_id = pair_data["hardware_id"]
            model_id = pair_data["model_id"]
            items = pair_data["items"]
            
            metrics_data = {}
            for metric in available_metrics:
                # Collect values for this metric across all items
                mape_values = []
                
                for item in items:
                    metrics_comparison = item.get("metrics_comparison", {})
                    if metric in metrics_comparison:
                        metric_data = metrics_comparison[metric]
                        if "mape" in metric_data:
                            mape_values.append(metric_data["mape"])
                
                # Calculate statistics if we have values
                if mape_values:
                    # Use numpy if available for better statistics
                    if np_available:
                        metrics_data[metric] = {
                            "mean_mape": float(np.mean(mape_values)),
                            "median_mape": float(np.median(mape_values)),
                            "min_mape": float(np.min(mape_values)),
                            "max_mape": float(np.max(mape_values)),
                            "std_mape": float(np.std(mape_values)),
                            "sample_count": len(mape_values)
                        }
                    else:
                        # Fallback to basic statistics
                        mean_mape = sum(mape_values) / len(mape_values)
                        sorted_mapes = sorted(mape_values)
                        median_idx = len(sorted_mapes) // 2
                        metrics_data[metric] = {
                            "mean_mape": mean_mape,
                            "median_mape": sorted_mapes[median_idx],
                            "min_mape": min(mape_values),
                            "max_mape": max(mape_values),
                            "sample_count": len(mape_values)
                        }
            
            # Calculate overall MAPE across all metrics
            overall_mape_values = []
            for item in items:
                metrics_comparison = item.get("metrics_comparison", {})
                item_mape_values = []
                
                for metric, metric_data in metrics_comparison.items():
                    if "mape" in metric_data:
                        item_mape_values.append(metric_data["mape"])
                
                if item_mape_values:
                    overall_mape_values.append(sum(item_mape_values) / len(item_mape_values))
            
            # Calculate overall statistics
            if overall_mape_values:
                if np_available:
                    metrics_data["overall"] = {
                        "mean_mape": float(np.mean(overall_mape_values)),
                        "median_mape": float(np.median(overall_mape_values)),
                        "min_mape": float(np.min(overall_mape_values)),
                        "max_mape": float(np.max(overall_mape_values)),
                        "std_mape": float(np.std(overall_mape_values)),
                        "sample_count": len(overall_mape_values)
                    }
                else:
                    mean_mape = sum(overall_mape_values) / len(overall_mape_values)
                    sorted_mapes = sorted(overall_mape_values)
                    median_idx = len(sorted_mapes) // 2
                    metrics_data["overall"] = {
                        "mean_mape": mean_mape,
                        "median_mape": sorted_mapes[median_idx],
                        "min_mape": min(overall_mape_values),
                        "max_mape": max(overall_mape_values),
                        "sample_count": len(overall_mape_values)
                    }
            
            hardware_model_metrics[key] = {
                "hardware_id": hardware_id,
                "model_id": model_id,
                "metrics": metrics_data,
                "validation_count": len(items)
            }
        
        # Identify best and worst performing hardware-model pairs
        if hardware_model_metrics:
            # Sort by overall mean MAPE
            sorted_pairs = sorted(
                hardware_model_metrics.values(),
                key=lambda x: x["metrics"].get("overall", {}).get("mean_mape", float('inf'))
            )
            
            best_pairs = sorted_pairs[:3] if len(sorted_pairs) > 3 else sorted_pairs
            worst_pairs = sorted_pairs[-3:] if len(sorted_pairs) > 3 else sorted_pairs
            worst_pairs.reverse()  # Reverse to show highest MAPE first
        else:
            best_pairs = []
            worst_pairs = []
        
        # Create analysis results
        analysis = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "validation_items_count": len(validation_items),
            "hardware_model_pairs_count": len(hardware_model_pairs),
            "available_metrics": list(available_metrics),
            "hardware_model_metrics": hardware_model_metrics,
            "best_performing_pairs": best_pairs,
            "worst_performing_pairs": worst_pairs
        }
        
        # Add summary information if available
        if summary:
            analysis["summary"] = summary
        
        return analysis
    
    def generate_report(
        self, 
        analysis: Dict[str, Any], 
        format: str = "text", 
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a validation results report.
        
        Args:
            analysis: Validation analysis data
            format: Report format (text, markdown, html, json)
            output_file: File to write report to (optional)
            
        Returns:
            Report content
        """
        if analysis.get("status") != "success":
            return f"Error: {analysis.get('message', 'Unknown error')}"
        
        # Generate report in requested format
        if format == "json":
            result = json.dumps(analysis, indent=2)
            
        elif format == "markdown":
            lines = ["# Validation Results Analysis\n"]
            
            # Summary section
            lines.append("## Summary\n")
            lines.append(f"- **Timestamp**: {analysis['timestamp']}")
            lines.append(f"- **Validation Items**: {analysis['validation_items_count']}")
            lines.append(f"- **Hardware-Model Pairs**: {analysis['hardware_model_pairs_count']}")
            lines.append(f"- **Available Metrics**: {', '.join(analysis['available_metrics'])}\n")
            
            # Add summary from validation results if available
            if "summary" in analysis:
                summary = analysis["summary"]
                if "overall_mape" in summary:
                    lines.append(f"- **Overall MAPE**: {summary['overall_mape']:.2%}")
                if "status" in summary:
                    lines.append(f"- **Validation Status**: {summary['status']}")
                if "timestamp" in summary:
                    lines.append(f"- **Validation Timestamp**: {summary['timestamp']}")
                lines.append("")
            
            # Best performing pairs
            if analysis["best_performing_pairs"]:
                lines.append("## Best Performing Hardware-Model Pairs\n")
                lines.append("| Hardware | Model | Mean MAPE | Median MAPE | Min MAPE | Max MAPE |")
                lines.append("|----------|-------|-----------|-------------|----------|----------|")
                
                for pair in analysis["best_performing_pairs"]:
                    overall_metrics = pair["metrics"].get("overall", {})
                    mean_mape = overall_metrics.get("mean_mape", 0) * 100
                    median_mape = overall_metrics.get("median_mape", 0) * 100
                    min_mape = overall_metrics.get("min_mape", 0) * 100
                    max_mape = overall_metrics.get("max_mape", 0) * 100
                    
                    lines.append(f"| {pair['hardware_id']} | {pair['model_id']} | {mean_mape:.2f}% | {median_mape:.2f}% | {min_mape:.2f}% | {max_mape:.2f}% |")
                
                lines.append("")
            
            # Worst performing pairs
            if analysis["worst_performing_pairs"]:
                lines.append("## Worst Performing Hardware-Model Pairs\n")
                lines.append("| Hardware | Model | Mean MAPE | Median MAPE | Min MAPE | Max MAPE |")
                lines.append("|----------|-------|-----------|-------------|----------|----------|")
                
                for pair in analysis["worst_performing_pairs"]:
                    overall_metrics = pair["metrics"].get("overall", {})
                    mean_mape = overall_metrics.get("mean_mape", 0) * 100
                    median_mape = overall_metrics.get("median_mape", 0) * 100
                    min_mape = overall_metrics.get("min_mape", 0) * 100
                    max_mape = overall_metrics.get("max_mape", 0) * 100
                    
                    lines.append(f"| {pair['hardware_id']} | {pair['model_id']} | {mean_mape:.2f}% | {median_mape:.2f}% | {min_mape:.2f}% | {max_mape:.2f}% |")
                
                lines.append("")
            
            # Metric Performance By Hardware-Model Pair
            if analysis["hardware_model_metrics"]:
                lines.append("## Metric Performance By Hardware-Model Pair\n")
                
                # Sort hardware-model pairs by overall mean MAPE
                sorted_pairs = sorted(
                    analysis["hardware_model_metrics"].values(),
                    key=lambda x: x["metrics"].get("overall", {}).get("mean_mape", float('inf'))
                )
                
                for pair in sorted_pairs:
                    lines.append(f"### {pair['hardware_id']} - {pair['model_id']}\n")
                    lines.append(f"- **Validation Count**: {pair['validation_count']}\n")
                    
                    # Create table of metrics
                    lines.append("| Metric | Mean MAPE | Median MAPE | Min MAPE | Max MAPE |")
                    lines.append("|--------|-----------|-------------|----------|----------|")
                    
                    # Sort metrics by mean MAPE
                    metrics = pair["metrics"]
                    sorted_metrics = {}
                    
                    # Put 'overall' first
                    if "overall" in metrics:
                        sorted_metrics["overall"] = metrics["overall"]
                    
                    # Sort remaining metrics
                    for metric, data in sorted(
                        metrics.items(),
                        key=lambda x: x[1].get("mean_mape", float('inf'))
                    ):
                        if metric != "overall":
                            sorted_metrics[metric] = data
                    
                    # Create rows for each metric
                    for metric, data in sorted_metrics.items():
                        mean_mape = data.get("mean_mape", 0) * 100
                        median_mape = data.get("median_mape", 0) * 100
                        min_mape = data.get("min_mape", 0) * 100
                        max_mape = data.get("max_mape", 0) * 100
                        
                        lines.append(f"| {metric} | {mean_mape:.2f}% | {median_mape:.2f}% | {min_mape:.2f}% | {max_mape:.2f}% |")
                    
                    lines.append("")
            
            result = "\n".join(lines)
            
        elif format == "html":
            html_lines = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <title>Validation Results Analysis</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 20px; }",
                "        .summary { background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; }",
                "        .good { color: #2e7d32; }",
                "        .moderate { color: #ff9800; }",
                "        .poor { color: #d32f2f; }",
                "        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
                "        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }",
                "        th { background-color: #f5f5f5; }",
                "        .card { border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 20px; }",
                "        .card-header { font-weight: bold; margin-bottom: 10px; }",
                "        .progress-container { width: 100px; background-color: #f1f1f1; border-radius: 3px; }",
                "        .progress-bar { height: 15px; border-radius: 3px; }",
                "        .excellent { background-color: #4caf50; }",
                "        .good { background-color: #8bc34a; }",
                "        .moderate { background-color: #ffeb3b; }",
                "        .poor { background-color: #f44336; }",
                "    </style>",
                "</head>",
                "<body>",
                "    <h1>Validation Results Analysis</h1>",
                ""
            ]
            
            # Summary section
            html_lines.extend([
                "    <div class='summary'>",
                "        <h2>Summary</h2>",
                f"        <p><strong>Timestamp</strong>: {analysis['timestamp']}</p>",
                f"        <p><strong>Validation Items</strong>: {analysis['validation_items_count']}</p>",
                f"        <p><strong>Hardware-Model Pairs</strong>: {analysis['hardware_model_pairs_count']}</p>",
                f"        <p><strong>Available Metrics</strong>: {', '.join(analysis['available_metrics'])}</p>"
            ])
            
            # Add summary from validation results if available
            if "summary" in analysis:
                summary = analysis["summary"]
                if "overall_mape" in summary:
                    overall_mape = summary["overall_mape"] * 100
                    mape_class = "good" if overall_mape < 10 else "moderate" if overall_mape < 20 else "poor"
                    html_lines.append(f"        <p><strong>Overall MAPE</strong>: <span class='{mape_class}'>{overall_mape:.2f}%</span></p>")
                if "status" in summary:
                    html_lines.append(f"        <p><strong>Validation Status</strong>: {summary['status']}</p>")
                if "timestamp" in summary:
                    html_lines.append(f"        <p><strong>Validation Timestamp</strong>: {summary['timestamp']}</p>")
            
            html_lines.append("    </div>")
            
            # Best performing pairs
            if analysis["best_performing_pairs"]:
                html_lines.extend([
                    "    <h2>Best Performing Hardware-Model Pairs</h2>",
                    "    <table>",
                    "        <tr>",
                    "            <th>Hardware</th>",
                    "            <th>Model</th>",
                    "            <th>Mean MAPE</th>",
                    "            <th>Median MAPE</th>",
                    "            <th>Min MAPE</th>",
                    "            <th>Max MAPE</th>",
                    "            <th>Performance</th>",
                    "        </tr>"
                ])
                
                for pair in analysis["best_performing_pairs"]:
                    overall_metrics = pair["metrics"].get("overall", {})
                    mean_mape = overall_metrics.get("mean_mape", 0) * 100
                    median_mape = overall_metrics.get("median_mape", 0) * 100
                    min_mape = overall_metrics.get("min_mape", 0) * 100
                    max_mape = overall_metrics.get("max_mape", 0) * 100
                    
                    bar_class = "excellent" if mean_mape < 5 else "good" if mean_mape < 10 else "moderate" if mean_mape < 20 else "poor"
                    
                    html_lines.extend([
                        "        <tr>",
                        f"            <td>{pair['hardware_id']}</td>",
                        f"            <td>{pair['model_id']}</td>",
                        f"            <td>{mean_mape:.2f}%</td>",
                        f"            <td>{median_mape:.2f}%</td>",
                        f"            <td>{min_mape:.2f}%</td>",
                        f"            <td>{max_mape:.2f}%</td>",
                        f"            <td><div class='progress-container'><div class='progress-bar {bar_class}' style='width:{100-mean_mape}%'></div></div></td>",
                        "        </tr>"
                    ])
                
                html_lines.append("    </table>")
            
            # Worst performing pairs
            if analysis["worst_performing_pairs"]:
                html_lines.extend([
                    "    <h2>Worst Performing Hardware-Model Pairs</h2>",
                    "    <table>",
                    "        <tr>",
                    "            <th>Hardware</th>",
                    "            <th>Model</th>",
                    "            <th>Mean MAPE</th>",
                    "            <th>Median MAPE</th>",
                    "            <th>Min MAPE</th>",
                    "            <th>Max MAPE</th>",
                    "            <th>Performance</th>",
                    "        </tr>"
                ])
                
                for pair in analysis["worst_performing_pairs"]:
                    overall_metrics = pair["metrics"].get("overall", {})
                    mean_mape = overall_metrics.get("mean_mape", 0) * 100
                    median_mape = overall_metrics.get("median_mape", 0) * 100
                    min_mape = overall_metrics.get("min_mape", 0) * 100
                    max_mape = overall_metrics.get("max_mape", 0) * 100
                    
                    bar_class = "excellent" if mean_mape < 5 else "good" if mean_mape < 10 else "moderate" if mean_mape < 20 else "poor"
                    
                    html_lines.extend([
                        "        <tr>",
                        f"            <td>{pair['hardware_id']}</td>",
                        f"            <td>{pair['model_id']}</td>",
                        f"            <td>{mean_mape:.2f}%</td>",
                        f"            <td>{median_mape:.2f}%</td>",
                        f"            <td>{min_mape:.2f}%</td>",
                        f"            <td>{max_mape:.2f}%</td>",
                        f"            <td><div class='progress-container'><div class='progress-bar {bar_class}' style='width:{100-mean_mape if mean_mape < 100 else 0}%'></div></div></td>",
                        "        </tr>"
                    ])
                
                html_lines.append("    </table>")
            
            # Metric Performance By Hardware-Model Pair
            if analysis["hardware_model_metrics"]:
                html_lines.append("    <h2>Metric Performance By Hardware-Model Pair</h2>")
                
                # Sort hardware-model pairs by overall mean MAPE
                sorted_pairs = sorted(
                    analysis["hardware_model_metrics"].values(),
                    key=lambda x: x["metrics"].get("overall", {}).get("mean_mape", float('inf'))
                )
                
                for pair in sorted_pairs:
                    html_lines.extend([
                        "    <div class='card'>",
                        f"        <div class='card-header'>{pair['hardware_id']} - {pair['model_id']}</div>",
                        f"        <p><strong>Validation Count</strong>: {pair['validation_count']}</p>",
                        "        <table>",
                        "            <tr>",
                        "                <th>Metric</th>",
                        "                <th>Mean MAPE</th>",
                        "                <th>Median MAPE</th>",
                        "                <th>Min MAPE</th>",
                        "                <th>Max MAPE</th>",
                        "                <th>Performance</th>",
                        "            </tr>"
                    ])
                    
                    # Sort metrics by mean MAPE
                    metrics = pair["metrics"]
                    sorted_metrics = {}
                    
                    # Put 'overall' first
                    if "overall" in metrics:
                        sorted_metrics["overall"] = metrics["overall"]
                    
                    # Sort remaining metrics
                    for metric, data in sorted(
                        metrics.items(),
                        key=lambda x: x[1].get("mean_mape", float('inf'))
                    ):
                        if metric != "overall":
                            sorted_metrics[metric] = data
                    
                    # Create rows for each metric
                    for metric, data in sorted_metrics.items():
                        mean_mape = data.get("mean_mape", 0) * 100
                        median_mape = data.get("median_mape", 0) * 100
                        min_mape = data.get("min_mape", 0) * 100
                        max_mape = data.get("max_mape", 0) * 100
                        
                        bar_class = "excellent" if mean_mape < 5 else "good" if mean_mape < 10 else "moderate" if mean_mape < 20 else "poor"
                        
                        html_lines.extend([
                            "            <tr>",
                            f"                <td>{metric}</td>",
                            f"                <td>{mean_mape:.2f}%</td>",
                            f"                <td>{median_mape:.2f}%</td>",
                            f"                <td>{min_mape:.2f}%</td>",
                            f"                <td>{max_mape:.2f}%</td>",
                            f"                <td><div class='progress-container'><div class='progress-bar {bar_class}' style='width:{100-mean_mape if mean_mape < 100 else 0}%'></div></div></td>",
                            "            </tr>"
                        ])
                    
                    html_lines.extend([
                        "        </table>",
                        "    </div>"
                    ])
            
            html_lines.extend([
                "</body>",
                "</html>"
            ])
            
            result = "\n".join(html_lines)
            
        else:
            # Plain text output
            lines = ["Validation Results Analysis", "=" * 27, ""]
            
            # Summary section
            lines.append("Summary:")
            lines.append(f"  Timestamp: {analysis['timestamp']}")
            lines.append(f"  Validation Items: {analysis['validation_items_count']}")
            lines.append(f"  Hardware-Model Pairs: {analysis['hardware_model_pairs_count']}")
            lines.append(f"  Available Metrics: {', '.join(analysis['available_metrics'])}")
            
            # Add summary from validation results if available
            if "summary" in analysis:
                summary = analysis["summary"]
                if "overall_mape" in summary:
                    lines.append(f"  Overall MAPE: {summary['overall_mape']:.2%}")
                if "status" in summary:
                    lines.append(f"  Validation Status: {summary['status']}")
                if "timestamp" in summary:
                    lines.append(f"  Validation Timestamp: {summary['timestamp']}")
            lines.append("")
            
            # Best performing pairs
            if analysis["best_performing_pairs"]:
                lines.append("Best Performing Hardware-Model Pairs:")
                for pair in analysis["best_performing_pairs"]:
                    overall_metrics = pair["metrics"].get("overall", {})
                    mean_mape = overall_metrics.get("mean_mape", 0) * 100
                    lines.append(f"  {pair['hardware_id']} - {pair['model_id']}: {mean_mape:.2f}% mean MAPE")
                lines.append("")
            
            # Worst performing pairs
            if analysis["worst_performing_pairs"]:
                lines.append("Worst Performing Hardware-Model Pairs:")
                for pair in analysis["worst_performing_pairs"]:
                    overall_metrics = pair["metrics"].get("overall", {})
                    mean_mape = overall_metrics.get("mean_mape", 0) * 100
                    lines.append(f"  {pair['hardware_id']} - {pair['model_id']}: {mean_mape:.2f}% mean MAPE")
                lines.append("")
            
            # Metric Performance Summary
            lines.append("Metric Performance Summary:")
            
            # Sort hardware-model pairs by overall mean MAPE
            sorted_pairs = sorted(
                analysis["hardware_model_metrics"].values(),
                key=lambda x: x["metrics"].get("overall", {}).get("mean_mape", float('inf'))
            )
            
            for pair in sorted_pairs:
                lines.append(f"  {pair['hardware_id']} - {pair['model_id']}:")
                
                # Sort metrics by mean MAPE
                metrics = pair["metrics"]
                sorted_metrics = {}
                
                # Put 'overall' first
                if "overall" in metrics:
                    sorted_metrics["overall"] = metrics["overall"]
                
                # Sort remaining metrics
                for metric, data in sorted(
                    metrics.items(),
                    key=lambda x: x[1].get("mean_mape", float('inf'))
                ):
                    if metric != "overall":
                        sorted_metrics[metric] = data
                
                # Output each metric
                for metric, data in sorted_metrics.items():
                    mean_mape = data.get("mean_mape", 0) * 100
                    lines.append(f"    {metric}: {mean_mape:.2f}% mean MAPE")
                
                lines.append("")
            
            result = "\n".join(lines)
        
        # Write to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
                
        return result


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Analyze validation results for Simulation Validation Framework")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing validation results")
    parser.add_argument("--run-id", type=str, help="Optional run ID to filter results")
    parser.add_argument("--output-format", type=str, default="text", choices=["text", "json", "markdown", "html"], help="Output format")
    parser.add_argument("--output-file", type=str, help="Output file")
    
    args = parser.parse_args()
    
    # Validate results directory
    if not os.path.isdir(args.results_dir):
        logger.error(f"Results directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = ValidationResultsAnalyzer()
    
    # Load and analyze validation results
    validation_results = analyzer.load_validation_results(args.results_dir, args.run_id)
    analysis = analyzer.analyze_results(validation_results)
    
    # Generate report
    report = analyzer.generate_report(
        analysis, 
        format=args.output_format, 
        output_file=args.output_file
    )
    
    # Output report if not writing to file
    if not args.output_file:
        print(report)
    else:
        print(f"Validation analysis report written to {args.output_file}")
    
    # Return success if analysis was successful
    return 0 if analysis.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())