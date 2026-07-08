#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android Thermal Analysis Tool

This script provides a tool for analyzing thermal behavior of Android devices 
during model execution, including thermal profiling, throttling detection,
and battery impact correlation.

Usage:
    python android_thermal_analysis.py --model <model_path> --duration <seconds>
"""

import os
import time
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Local imports
from android_test_harness import AndroidDevice, AndroidModelRunner, AndroidThermalMonitor

try:
    from .database_integration import AndroidDatabaseAPI
    ANDROID_DB_AVAILABLE = True
except ImportError:
    logger.warning("Could not import Android database integration. Database functionality will be limited.")
    ANDROID_DB_AVAILABLE = False


def run_thermal_analysis(
    model_path: str,
    model_name: Optional[str] = None,
    device_serial: Optional[str] = None,
    duration_seconds: int = 300,
    sample_interval: float = 1.0,
    output_path: Optional[str] = None,
    db_path: Optional[str] = None,
    batch_size: int = 1,
    accelerator: str = "auto",
    threads: int = 4,
    model_type: str = "onnx",
    save_to_db: bool = False
) -> Dict[str, Any]:
    """
    Run a thermal analysis of a model on an Android device.
    
    Args:
        model_path: Path to the model file
        model_name: Optional name of the model
        device_serial: Optional serial number for the device
        duration_seconds: Duration of the analysis in seconds
        sample_interval: Thermal sampling interval in seconds
        output_path: Optional path to save the analysis results
        db_path: Optional path to database for storing results
        batch_size: Batch size to use for inference
        accelerator: Hardware accelerator to use
        threads: Number of threads to use
        model_type: Type of model (onnx, tflite)
        save_to_db: Whether to save results to database
        
    Returns:
        Dictionary with analysis results
    """
    # Determine model name if not provided
    if not model_name:
        model_name = os.path.basename(model_path)
    
    logger.info(f"Starting thermal analysis for model: {model_name}")
    logger.info(f"Duration: {duration_seconds} seconds")
    
    # Connect to device
    device = AndroidDevice(device_serial)
    
    if not device.connected:
        logger.error("Failed to connect to Android device")
        return {"status": "error", "message": "Failed to connect to device"}
    
    logger.info(f"Connected to device: {device.device_info.get('model', device.serial)}")
    
    # Create thermal monitor
    thermal_monitor = AndroidThermalMonitor(device)
    thermal_monitor.monitoring_interval = sample_interval
    
    # Create model runner
    model_runner = AndroidModelRunner(device)
    
    try:
        # Start thermal monitoring
        logger.info("Starting thermal monitoring")
        thermal_monitor.start_monitoring()
        
        # Store initial thermal state
        baseline_temps = thermal_monitor.get_current_temperatures()
        baseline_battery = device.get_battery_info()
        start_time = time.time()
        
        # Prepare model
        logger.info(f"Preparing model: {model_name}")
        remote_model_path = model_runner.prepare_model(model_path, model_type)
        
        if not remote_model_path:
            logger.error("Failed to prepare model")
            thermal_monitor.stop_monitoring()
            return {"status": "error", "message": "Failed to prepare model"}
        
        # Initialize results
        results = {
            "status": "success",
            "model_name": model_name,
            "model_path": model_path,
            "device_info": device.to_dict(),
            "start_time": start_time,
            "duration_seconds": duration_seconds,
            "sample_interval": sample_interval,
            "baseline": {
                "temperatures": baseline_temps,
                "battery": baseline_battery
            },
            "configuration": {
                "batch_size": batch_size,
                "accelerator": accelerator,
                "threads": threads
            },
            "time_series": [],
            "thermal_events": []
        }
        
        # Prepare runner
        logger.info(f"Preparing runner for {model_type}")
        model_runner.prepare_runner(model_type)
        
        # Run continuous inference for the specified duration
        logger.info(f"Running continuous inference for {duration_seconds} seconds")
        end_time = start_time + duration_seconds
        iteration = 0
        
        while time.time() < end_time:
            # Record time point
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Run a single inference
            logger.info(f"Running iteration {iteration + 1}")
            inference_result = model_runner.run_model(
                model_path=remote_model_path,
                iterations=1,
                batch_size=batch_size,
                threads=threads,
                accelerator=accelerator
            )
            
            # Get current thermal state
            current_temps = thermal_monitor.get_current_temperatures()
            current_battery = device.get_battery_info()
            throttling_stats = thermal_monitor.get_throttling_stats()
            
            # Record time series data point
            time_point = {
                "timestamp": current_time,
                "elapsed_seconds": elapsed_time,
                "iteration": iteration,
                "temperatures": current_temps,
                "battery": {
                    "level": current_battery["level"],
                    "temperature": current_battery["temperature"]
                },
                "throttling": {
                    "detected": throttling_stats["throttling_detected"],
                    "level": throttling_stats["throttling_level"],
                    "performance_impact": throttling_stats["performance_impact"]
                },
                "latency_ms": inference_result.get("latency_ms", {}).get("mean", 0)
            }
            
            # Add to time series
            results["time_series"].append(time_point)
            
            # Print status update
            if throttling_stats["throttling_detected"]:
                logger.warning(
                    f"Throttling detected: Level {throttling_stats['throttling_level']} "
                    f"({throttling_stats['level_description']})"
                )
            
            hottest_zone = max(current_temps.items(), key=lambda x: x[1], default=(None, 0))
            logger.info(
                f"Iteration {iteration + 1}: Elapsed {elapsed_time:.1f}s, "
                f"Latency {time_point['latency_ms']:.2f}ms, "
                f"Hottest zone: {hottest_zone[0]} at {hottest_zone[1]:.1f}°C"
            )
            
            iteration += 1
            
            # Sleep to control load (optional)
            time.sleep(0.1)
        
        # Get final thermal state
        final_temps = thermal_monitor.get_current_temperatures()
        final_battery = device.get_battery_info()
        final_time = time.time()
        
        # Get thermal report
        thermal_report = thermal_monitor.get_thermal_report()
        
        # Add final data to results
        results["final"] = {
            "temperatures": final_temps,
            "battery": final_battery,
            "duration_seconds": final_time - start_time,
            "iterations": iteration,
            "thermal_report": thermal_report
        }
        
        # Calculate thermal impact
        temp_deltas = {
            zone: final_temps[zone] - baseline_temps.get(zone, 0)
            for zone in final_temps.keys()
        }
        
        battery_impact = {
            "level_delta": baseline_battery["level"] - final_battery["level"],
            "temperature_delta": final_battery["temperature"] - baseline_battery["temperature"],
            "percent_per_hour": (baseline_battery["level"] - final_battery["level"]) * (3600 / (final_time - start_time))
        }
        
        # Calculate throttling impact
        throttling_duration = thermal_monitor.get_throttling_stats()["throttling_time_seconds"]
        throttling_percentage = (throttling_duration / (final_time - start_time)) * 100
        
        # Calculate performance correlation
        if len(results["time_series"]) > 1:
            # Extract latencies and temperatures
            latencies = [point["latency_ms"] for point in results["time_series"]]
            
            # Calculate correlation between temperature and latency
            temp_latency_correlation = {}
            
            for zone in final_temps.keys():
                temps = [point["temperatures"].get(zone, 0) for point in results["time_series"]]
                
                # Calculate correlation if enough data points
                if len(temps) > 5:
                    import numpy as np
                    try:
                        correlation = np.corrcoef(temps, latencies)[0, 1]
                        temp_latency_correlation[zone] = correlation
                    except:
                        temp_latency_correlation[zone] = 0.0
            
            # Add performance analysis to results
            results["performance_analysis"] = {
                "latency_ms": {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": sum(latencies) / len(latencies),
                    "range": max(latencies) - min(latencies)
                },
                "temperature_correlation": temp_latency_correlation
            }
        
        # Add impact analysis to results
        results["impact_analysis"] = {
            "temperature_deltas": temp_deltas,
            "battery_impact": battery_impact,
            "throttling_seconds": throttling_duration,
            "throttling_percentage": throttling_percentage,
            "overall_impact_score": max(0.0, min(1.0, (
                max(temp_deltas.values()) / 15.0 +  # Temperature impact
                throttling_percentage / 100.0 +     # Throttling impact
                battery_impact["percent_per_hour"] / 30.0  # Battery impact
            ) / 3.0))
        }
        
        # Generate recommendations
        results["recommendations"] = thermal_report["recommendations"]
        
        # Save results if output path provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Analysis results saved to: {output_path}")
        
        # Save to database if requested
        if save_to_db and db_path and ANDROID_DB_AVAILABLE:
            try:
                db_api = AndroidDatabaseAPI(db_path)
                analysis_id = db_api.store_thermal_analysis(results)
                if analysis_id:
                    logger.info(f"Thermal analysis saved to database with ID: {analysis_id}")
                    results["database_id"] = analysis_id
                else:
                    logger.warning("Failed to save thermal analysis to database")
            except Exception as e:
                logger.error(f"Error saving thermal analysis to database: {e}")
        
        return results
    
    finally:
        # Stop thermal monitoring
        logger.info("Stopping thermal monitoring")
        thermal_monitor.stop_monitoring()


def generate_report(results: Dict[str, Any], report_format: str = "markdown", output_path: Optional[str] = None) -> str:
    """
    Generate a report from thermal analysis results.
    
    Args:
        results: Analysis results
        report_format: Report format (markdown, html)
        output_path: Optional path to save the report
        
    Returns:
        Generated report
    """
    if report_format == "html":
        report = _generate_html_report(results)
    else:
        report = _generate_markdown_report(results)
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to: {output_path}")
    
    return report


def _generate_markdown_report(results: Dict[str, Any]) -> str:
    """
    Generate a markdown report from analysis results.
    
    Args:
        results: Analysis results
        
    Returns:
        Markdown report
    """
    model_name = results.get("model_name", "Unknown")
    device_info = results.get("device_info", {})
    device_model = device_info.get("model", "Unknown")
    
    report = f"# Thermal Analysis Report: {model_name}\n\n"
    report += f"Generated: {datetime.datetime.now().isoformat()}\n\n"
    
    # Device information
    report += "## Device Information\n\n"
    report += f"- **Model**: {device_model}\n"
    report += f"- **Manufacturer**: {device_info.get('manufacturer', 'Unknown')}\n"
    report += f"- **Android Version**: {device_info.get('android_version', 'Unknown')}\n"
    report += f"- **Chipset**: {device_info.get('chipset', 'Unknown')}\n\n"
    
    # Test configuration
    config = results.get("configuration", {})
    report += "## Test Configuration\n\n"
    report += f"- **Model**: {model_name}\n"
    report += f"- **Batch Size**: {config.get('batch_size', 1)}\n"
    report += f"- **Accelerator**: {config.get('accelerator', 'auto')}\n"
    report += f"- **Threads**: {config.get('threads', 4)}\n"
    report += f"- **Duration**: {results.get('duration_seconds', 0)} seconds\n"
    report += f"- **Iterations**: {results.get('final', {}).get('iterations', 0)}\n\n"
    
    # Impact analysis
    impact = results.get("impact_analysis", {})
    report += "## Thermal Impact Analysis\n\n"
    
    # Temperature changes
    report += "### Temperature Impact\n\n"
    report += "| Zone | Initial (°C) | Final (°C) | Change (°C) |\n"
    report += "|------|-------------|------------|-------------|\n"
    
    temp_deltas = impact.get("temperature_deltas", {})
    initial_temps = results.get("baseline", {}).get("temperatures", {})
    final_temps = results.get("final", {}).get("temperatures", {})
    
    for zone in sorted(temp_deltas.keys()):
        initial = initial_temps.get(zone, 0)
        final = final_temps.get(zone, 0)
        delta = temp_deltas.get(zone, 0)
        
        report += f"| {zone} | {initial:.1f} | {final:.1f} | {delta:+.1f} |\n"
    
    report += "\n"
    
    # Battery impact
    battery_impact = impact.get("battery_impact", {})
    report += "### Battery Impact\n\n"
    report += f"- **Level Change**: {battery_impact.get('level_delta', 0)}%\n"
    report += f"- **Temperature Change**: {battery_impact.get('temperature_delta', 0):.1f}°C\n"
    report += f"- **Estimated Drain**: {battery_impact.get('percent_per_hour', 0):.1f}% per hour\n\n"
    
    # Throttling impact
    report += "### Throttling Impact\n\n"
    report += f"- **Throttling Duration**: {impact.get('throttling_seconds', 0):.1f} seconds\n"
    report += f"- **Throttling Percentage**: {impact.get('throttling_percentage', 0):.1f}% of test duration\n\n"
    
    # Overall impact
    report += "### Overall Impact\n\n"
    report += f"- **Impact Score**: {impact.get('overall_impact_score', 0):.2f} (0-1 scale)\n"
    
    # Determine impact rating
    impact_score = impact.get('overall_impact_score', 0)
    if impact_score < 0.3:
        impact_rating = "Low"
    elif impact_score < 0.6:
        impact_rating = "Medium"
    else:
        impact_rating = "High"
    
    report += f"- **Impact Rating**: {impact_rating}\n\n"
    
    # Performance analysis
    perf = results.get("performance_analysis", {})
    if perf:
        report += "## Performance Analysis\n\n"
        
        # Latency analysis
        latency = perf.get("latency_ms", {})
        report += "### Latency (ms)\n\n"
        report += f"- **Min**: {latency.get('min', 0):.2f}\n"
        report += f"- **Max**: {latency.get('max', 0):.2f}\n"
        report += f"- **Mean**: {latency.get('mean', 0):.2f}\n"
        report += f"- **Range**: {latency.get('range', 0):.2f}\n\n"
        
        # Temperature correlation
        correlation = perf.get("temperature_correlation", {})
        if correlation:
            report += "### Temperature-Latency Correlation\n\n"
            report += "| Zone | Correlation |\n"
            report += "|------|-------------|\n"
            
            for zone, corr in sorted(correlation.items(), key=lambda x: abs(x[1]), reverse=True):
                report += f"| {zone} | {corr:.3f} |\n"
            
            report += "\n"
            
            # Add correlation interpretation
            max_corr_zone, max_corr = max(correlation.items(), key=lambda x: abs(x[1]), default=(None, 0))
            if max_corr_zone and abs(max_corr) > 0.5:
                if max_corr > 0:
                    report += f"**Note**: Strong positive correlation between {max_corr_zone} temperature and latency "
                    report += f"({max_corr:.3f}), indicating performance degradation as temperature increases.\n\n"
                else:
                    report += f"**Note**: Strong negative correlation detected ({max_corr:.3f}). "
                    report += "This unusual pattern may indicate thermal throttling is effectively managing performance.\n\n"
    
    # Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        report += "## Recommendations\n\n"
        
        for rec in recommendations:
            report += f"- {rec}\n"
        
        report += "\n"
    
    # Conclusion
    report += "## Conclusion\n\n"
    
    if impact_score < 0.3:
        report += f"The model '{model_name}' shows a **low thermal impact** on the device. "
        report += "It should be suitable for extended use without significant performance degradation or battery drain.\n\n"
    elif impact_score < 0.6:
        report += f"The model '{model_name}' shows a **moderate thermal impact** on the device. "
        report += "For extended use, consider implementing thermal management strategies such as periodic cooling breaks or optimizing the model.\n\n"
    else:
        report += f"The model '{model_name}' shows a **high thermal impact** on the device. "
        report += "Extended use may lead to significant thermal throttling, performance degradation, and battery drain. "
        report += "Consider model optimization, quantization, or using a more powerful device for this workload.\n\n"
    
    return report


def _generate_html_report(results: Dict[str, Any]) -> str:
    """
    Generate an HTML report from analysis results.
    
    Args:
        results: Analysis results
        
    Returns:
        HTML report
    """
    model_name = results.get("model_name", "Unknown")
    device_info = results.get("device_info", {})
    device_model = device_info.get("model", "Unknown")
    
    # Similar to the markdown report but with HTML formatting
    # This would include charts and visualizations in a full implementation
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Thermal Analysis Report: {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3, h4 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .chart {{ width: 100%; height: 300px; margin-bottom: 20px; }}
        .positive-delta {{ color: red; }}
        .negative-delta {{ color: green; }}
        .high-impact {{ color: red; font-weight: bold; }}
        .medium-impact {{ color: orange; font-weight: bold; }}
        .low-impact {{ color: green; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Thermal Analysis Report: {model_name}</h1>
    <p>Generated: {datetime.datetime.now().isoformat()}</p>
    
    <h2>Device Information</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>Model</td><td>{device_model}</td></tr>
        <tr><td>Manufacturer</td><td>{device_info.get('manufacturer', 'Unknown')}</td></tr>
        <tr><td>Android Version</td><td>{device_info.get('android_version', 'Unknown')}</td></tr>
        <tr><td>Chipset</td><td>{device_info.get('chipset', 'Unknown')}</td></tr>
    </table>
"""
    
    # Test configuration
    config = results.get("configuration", {})
    html += f"""
    <h2>Test Configuration</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Model</td><td>{model_name}</td></tr>
        <tr><td>Batch Size</td><td>{config.get('batch_size', 1)}</td></tr>
        <tr><td>Accelerator</td><td>{config.get('accelerator', 'auto')}</td></tr>
        <tr><td>Threads</td><td>{config.get('threads', 4)}</td></tr>
        <tr><td>Duration</td><td>{results.get('duration_seconds', 0)} seconds</td></tr>
        <tr><td>Iterations</td><td>{results.get('final', {}).get('iterations', 0)}</td></tr>
    </table>
"""
    
    # Impact analysis
    impact = results.get("impact_analysis", {})
    html += """
    <h2>Thermal Impact Analysis</h2>
    
    <h3>Temperature Impact</h3>
    <table>
        <tr>
            <th>Zone</th>
            <th>Initial (°C)</th>
            <th>Final (°C)</th>
            <th>Change (°C)</th>
        </tr>
"""
    
    temp_deltas = impact.get("temperature_deltas", {})
    initial_temps = results.get("baseline", {}).get("temperatures", {})
    final_temps = results.get("final", {}).get("temperatures", {})
    
    for zone in sorted(temp_deltas.keys()):
        initial = initial_temps.get(zone, 0)
        final = final_temps.get(zone, 0)
        delta = temp_deltas.get(zone, 0)
        
        delta_class = "positive-delta" if delta > 0 else "negative-delta" if delta < 0 else ""
        
        html += f"""
        <tr>
            <td>{zone}</td>
            <td>{initial:.1f}</td>
            <td>{final:.1f}</td>
            <td class="{delta_class}">{delta:+.1f}</td>
        </tr>"""
    
    html += """
    </table>
"""
    
    # Battery impact
    battery_impact = impact.get("battery_impact", {})
    html += f"""
    <h3>Battery Impact</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Level Change</td><td>{battery_impact.get('level_delta', 0)}%</td></tr>
        <tr><td>Temperature Change</td><td>{battery_impact.get('temperature_delta', 0):.1f}°C</td></tr>
        <tr><td>Estimated Drain</td><td>{battery_impact.get('percent_per_hour', 0):.1f}% per hour</td></tr>
    </table>
"""
    
    # Throttling impact
    html += f"""
    <h3>Throttling Impact</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Throttling Duration</td><td>{impact.get('throttling_seconds', 0):.1f} seconds</td></tr>
        <tr><td>Throttling Percentage</td><td>{impact.get('throttling_percentage', 0):.1f}% of test duration</td></tr>
    </table>
"""
    
    # Overall impact
    impact_score = impact.get('overall_impact_score', 0)
    if impact_score < 0.3:
        impact_rating = "Low"
        impact_class = "low-impact"
    elif impact_score < 0.6:
        impact_rating = "Medium"
        impact_class = "medium-impact"
    else:
        impact_rating = "High"
        impact_class = "high-impact"
    
    html += f"""
    <h3>Overall Impact</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Impact Score</td><td>{impact_score:.2f} (0-1 scale)</td></tr>
        <tr><td>Impact Rating</td><td class="{impact_class}">{impact_rating}</td></tr>
    </table>
"""
    
    # Performance analysis
    perf = results.get("performance_analysis", {})
    if perf:
        html += """
    <h2>Performance Analysis</h2>
"""
        
        # Latency analysis
        latency = perf.get("latency_ms", {})
        html += f"""
    <h3>Latency (ms)</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Min</td><td>{latency.get('min', 0):.2f}</td></tr>
        <tr><td>Max</td><td>{latency.get('max', 0):.2f}</td></tr>
        <tr><td>Mean</td><td>{latency.get('mean', 0):.2f}</td></tr>
        <tr><td>Range</td><td>{latency.get('range', 0):.2f}</td></tr>
    </table>
"""
        
        # Temperature correlation
        correlation = perf.get("temperature_correlation", {})
        if correlation:
            html += """
    <h3>Temperature-Latency Correlation</h3>
    <table>
        <tr>
            <th>Zone</th>
            <th>Correlation</th>
        </tr>
"""
            
            for zone, corr in sorted(correlation.items(), key=lambda x: abs(x[1]), reverse=True):
                html += f"""
        <tr>
            <td>{zone}</td>
            <td>{corr:.3f}</td>
        </tr>"""
            
            html += """
    </table>
"""
            
            # Add correlation interpretation
            max_corr_zone, max_corr = max(correlation.items(), key=lambda x: abs(x[1]), default=(None, 0))
            if max_corr_zone and abs(max_corr) > 0.5:
                if max_corr > 0:
                    html += f"""
    <p><strong>Note</strong>: Strong positive correlation between {max_corr_zone} temperature and latency 
    ({max_corr:.3f}), indicating performance degradation as temperature increases.</p>
"""
                else:
                    html += f"""
    <p><strong>Note</strong>: Strong negative correlation detected ({max_corr:.3f}). 
    This unusual pattern may indicate thermal throttling is effectively managing performance.</p>
"""
    
    # Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        html += """
    <h2>Recommendations</h2>
    <ul>
"""
        
        for rec in recommendations:
            html += f"""
        <li>{rec}</li>"""
        
        html += """
    </ul>
"""
    
    # Conclusion
    html += """
    <h2>Conclusion</h2>
"""
    
    if impact_score < 0.3:
        html += f"""
    <p>The model '{model_name}' shows a <span class="low-impact">low thermal impact</span> on the device. 
    It should be suitable for extended use without significant performance degradation or battery drain.</p>
"""
    elif impact_score < 0.6:
        html += f"""
    <p>The model '{model_name}' shows a <span class="medium-impact">moderate thermal impact</span> on the device. 
    For extended use, consider implementing thermal management strategies such as periodic cooling breaks or optimizing the model.</p>
"""
    else:
        html += f"""
    <p>The model '{model_name}' shows a <span class="high-impact">high thermal impact</span> on the device. 
    Extended use may lead to significant thermal throttling, performance degradation, and battery drain. 
    Consider model optimization, quantization, or using a more powerful device for this workload.</p>
"""
    
    html += """
</body>
</html>
"""
    
    return html


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Android Thermal Analysis Tool")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--name", help="Model name (defaults to filename)")
    parser.add_argument("--type", default="onnx", choices=["onnx", "tflite"], help="Model type")
    parser.add_argument("--serial", help="Device serial number")
    parser.add_argument("--duration", type=int, default=300, help="Analysis duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Thermal sampling interval in seconds")
    parser.add_argument("--output", help="Path to save analysis results")
    parser.add_argument("--report", help="Path to save analysis report")
    parser.add_argument("--report-format", default="markdown", choices=["markdown", "html"], help="Report format")
    parser.add_argument("--db-path", help="Path to database for storing results")
    parser.add_argument("--save-to-db", action="store_true", help="Save results to database")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--accelerator", default="auto", help="Hardware accelerator")
    parser.add_argument("--threads", type=int, default=4, help="Thread count")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run thermal analysis
        results = run_thermal_analysis(
            model_path=args.model,
            model_name=args.name,
            device_serial=args.serial,
            duration_seconds=args.duration,
            sample_interval=args.interval,
            output_path=args.output,
            db_path=args.db_path,
            batch_size=args.batch_size,
            accelerator=args.accelerator,
            threads=args.threads,
            model_type=args.type,
            save_to_db=args.save_to_db
        )
        
        if results.get("status") == "success":
            # Generate report if requested
            if args.report:
                generate_report(
                    results=results,
                    report_format=args.report_format,
                    output_path=args.report
                )
            
            print("\nThermal Analysis Summary:")
            
            # Print device info
            device_model = results.get("device_info", {}).get("model", "Unknown")
            print(f"Device: {device_model}")
            
            # Print impact analysis
            impact = results.get("impact_analysis", {})
            impact_score = impact.get("overall_impact_score", 0)
            
            if impact_score < 0.3:
                impact_text = "LOW"
            elif impact_score < 0.6:
                impact_text = "MEDIUM"
            else:
                impact_text = "HIGH"
            
            print(f"Thermal Impact: {impact_text} ({impact_score:.2f})")
            print(f"Battery Impact: {impact.get('battery_impact', {}).get('percent_per_hour', 0):.1f}% per hour")
            print(f"Throttling: {impact.get('throttling_percentage', 0):.1f}% of test duration")
            
            # Print top recommendations
            recommendations = results.get("recommendations", [])
            if recommendations:
                print("\nTop Recommendations:")
                for i, rec in enumerate(recommendations[:3]):
                    print(f"  {i+1}. {rec}")
            
            # Print report location if saved
            if args.report:
                print(f"\nDetailed report saved to: {args.report}")
            if args.output:
                print(f"Raw analysis data saved to: {args.output}")
            
            return 0
        else:
            print(f"Error: {results.get('message', 'Unknown error')}")
            return 1
    
    except Exception as e:
        logger.exception(f"Error during thermal analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(main())