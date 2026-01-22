#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobile Performance Regression Detection Tool

This script analyzes benchmark data to detect performance regressions across mobile platforms.
It compares current benchmark results with historical data to identify significant performance
degradations that may require attention.

Usage:
    python check_mobile_regressions.py --data-file DATA_FILE [--threshold THRESHOLD]
    [--output OUTPUT] [--format {json,markdown,text}] [--days DAYS] [--verbose]

Examples:
    # Check for regressions in analysis data with default settings
    python check_mobile_regressions.py --data-file analysis_results.json
    
    # Use custom threshold and output format
    python check_mobile_regressions.py --data-file analysis_results.json --threshold 10
        --output regression_report.md --format markdown
    
    # Compare with historical data from last 30 days
    python check_mobile_regressions.py --data-file analysis_results.json --days 30

Date: April 2025
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local imports
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


class MobileRegressionDetector:
    """
    Analyzes benchmark data to detect performance regressions across mobile platforms.
    
    This class compares current benchmark results with historical data to identify
    significant performance degradations that may require attention.
    """
    
    # Regression severity levels
    SEVERITY_LEVELS = {
        "critical": {"color": "red", "threshold": 25},
        "high": {"color": "orange", "threshold": 15},
        "medium": {"color": "yellow", "threshold": 10},
        "low": {"color": "blue", "threshold": 5}
    }
    
    def __init__(self, 
                 data_file: str,
                 db_path: Optional[str] = None,
                 threshold: float = 10.0,
                 days: int = 14,
                 verbose: bool = False):
        """
        Initialize the mobile regression detector.
        
        Args:
            data_file: Path to JSON data file with benchmark results
            db_path: Optional path to DuckDB database for historical data
            threshold: Percentage threshold for regression detection
            days: Number of days to look back for historical data
            verbose: Enable verbose logging
        """
        self.data_file = data_file
        self.db_path = db_path
        self.threshold = threshold
        self.days_lookback = days
        self.verbose = verbose
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize variables
        self.current_data = {}
        self.historical_data = {}
        self.db_api = None
        self.regressions = []
        
        # Calculate the date range for historical comparisons
        self.end_date = datetime.datetime.now()
        self.start_date = self.end_date - datetime.timedelta(days=self.days_lookback)
    
    def load_current_data(self) -> bool:
        """
        Load current benchmark data from JSON file.
        
        Returns:
            Success status
        """
        try:
            if not os.path.exists(self.data_file):
                logger.error(f"Data file not found: {self.data_file}")
                return False
            
            with open(self.data_file, 'r') as f:
                self.current_data = json.load(f)
            
            logger.info(f"Loaded current data from: {self.data_file}")
            
            # Verify data structure
            if not isinstance(self.current_data, dict):
                logger.error("Invalid data format: expected JSON object")
                return False
            
            if "platforms" not in self.current_data:
                logger.error("Invalid data format: missing 'platforms' section")
                return False
            
            if "models" not in self.current_data:
                logger.error("Invalid data format: missing 'models' section")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading current data: {e}")
            return False
    
    def connect_to_db(self) -> bool:
        """
        Connect to DuckDB database for historical data.
        
        Returns:
            Success status
        """
        # Skip if db_path not provided
        if not self.db_path:
            logger.info("No database path provided, skipping historical data")
            return False
        
        try:
            if not os.path.exists(self.db_path):
                logger.error(f"Database file not found: {self.db_path}")
                return False
            
            self.db_api = BenchmarkDBAPI(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def load_historical_data(self) -> bool:
        """
        Load historical benchmark data from database.
        
        Returns:
            Success status
        """
        if not self.db_api:
            if not self.connect_to_db():
                return False
        
        try:
            # Format date strings for database query
            start_date_str = self.start_date.strftime("%Y-%m-%d")
            end_date_str = self.end_date.strftime("%Y-%m-%d")
            
            logger.info(f"Loading historical data from {start_date_str} to {end_date_str}")
            
            # Get benchmark runs in the date range
            benchmark_runs = self.db_api.query(f"""
                SELECT *
                FROM benchmark_runs
                WHERE timestamp >= '{start_date_str}'
                  AND timestamp <= '{end_date_str}'
                ORDER BY timestamp DESC
            """)
            
            # Initialize historical data structure
            self.historical_data = {
                "platforms": {},
                "models": {}
            }
            
            # Process each benchmark run
            for run in benchmark_runs:
                run_id = run.get("id")
                device_info = run.get("device_info", {})
                platform = device_info.get("platform", "unknown")
                model_name = run.get("model_name", "unknown")
                
                # Initialize platform data if needed
                if platform not in self.historical_data["platforms"]:
                    self.historical_data["platforms"][platform] = {
                        "devices": {},
                        "models": {}
                    }
                
                # Initialize model data if needed
                if model_name not in self.historical_data["models"]:
                    self.historical_data["models"][model_name] = {
                        "platforms": {}
                    }
                
                # Get benchmark results for this run
                results = self.db_api.query(f"""
                    SELECT r.*, c.configuration
                    FROM benchmark_results r
                    JOIN benchmark_configurations c ON r.config_id = c.id
                    WHERE r.run_id = '{run_id}'
                """)
                
                # Process results
                for result in results:
                    config = result.get("configuration", {})
                    batch_size = config.get("batch_size", 1)
                    
                    # Extract metrics
                    throughput = result.get("throughput_items_per_second", 0)
                    latency = result.get("latency_ms", {}).get("mean", 0)
                    
                    # Add to platform data
                    platform_models = self.historical_data["platforms"][platform]["models"]
                    if model_name not in platform_models:
                        platform_models[model_name] = {
                            "batch_sizes": {}
                        }
                    
                    if batch_size not in platform_models[model_name]["batch_sizes"]:
                        platform_models[model_name]["batch_sizes"][batch_size] = {
                            "throughput": [],
                            "latency": []
                        }
                    
                    platform_models[model_name]["batch_sizes"][batch_size]["throughput"].append(throughput)
                    platform_models[model_name]["batch_sizes"][batch_size]["latency"].append(latency)
                    
                    # Add to model data
                    model_platforms = self.historical_data["models"][model_name]["platforms"]
                    if platform not in model_platforms:
                        model_platforms[platform] = {
                            "batch_sizes": {}
                        }
                    
                    if batch_size not in model_platforms[platform]["batch_sizes"]:
                        model_platforms[platform]["batch_sizes"][batch_size] = {
                            "throughput": [],
                            "latency": []
                        }
                    
                    model_platforms[platform]["batch_sizes"][batch_size]["throughput"].append(throughput)
                    model_platforms[platform]["batch_sizes"][batch_size]["latency"].append(latency)
            
            # Calculate averages for all metrics
            self._calculate_historical_averages()
            
            logger.info(f"Loaded historical data for {len(self.historical_data['platforms'])} platforms "
                        f"and {len(self.historical_data['models'])} models")
            return True
        
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
    
    def _calculate_historical_averages(self) -> None:
        """Calculate average metrics for historical data."""
        # Process platform data
        for platform, platform_data in self.historical_data["platforms"].items():
            for model_name, model_data in platform_data["models"].items():
                for batch_size, metrics in model_data["batch_sizes"].items():
                    # Calculate throughput average
                    throughput_values = metrics["throughput"]
                    if throughput_values:
                        metrics["avg_throughput"] = sum(throughput_values) / len(throughput_values)
                    else:
                        metrics["avg_throughput"] = 0
                    
                    # Calculate latency average
                    latency_values = metrics["latency"]
                    if latency_values:
                        metrics["avg_latency"] = sum(latency_values) / len(latency_values)
                    else:
                        metrics["avg_latency"] = 0
        
        # Process model data
        for model_name, model_data in self.historical_data["models"].items():
            for platform, platform_data in model_data["platforms"].items():
                for batch_size, metrics in platform_data["batch_sizes"].items():
                    # Calculate throughput average
                    throughput_values = metrics["throughput"]
                    if throughput_values:
                        metrics["avg_throughput"] = sum(throughput_values) / len(throughput_values)
                    else:
                        metrics["avg_throughput"] = 0
                    
                    # Calculate latency average
                    latency_values = metrics["latency"]
                    if latency_values:
                        metrics["avg_latency"] = sum(latency_values) / len(latency_values)
                    else:
                        metrics["avg_latency"] = 0
    
    def detect_regressions(self) -> List[Dict[str, Any]]:
        """
        Detect performance regressions by comparing current and historical data.
        
        Returns:
            List of detected regressions
        """
        self.regressions = []
        
        # Skip if no current data
        if not self.current_data:
            logger.error("No current data loaded")
            return self.regressions
        
        # Process each platform in current data
        for platform, platform_data in self.current_data.get("platforms", {}).items():
            for model_name, model_data in platform_data.get("models", {}).items():
                for batch_size_str, metrics in model_data.get("batch_sizes", {}).items():
                    # Convert batch size to int
                    try:
                        batch_size = int(batch_size_str)
                    except ValueError:
                        batch_size = 1
                    
                    # Get current metrics
                    current_throughput = metrics.get("throughput", 0)
                    current_latency = metrics.get("latency", 0)
                    
                    # Check for regressions if historical data is available
                    if self.historical_data and platform in self.historical_data["platforms"]:
                        platform_hist = self.historical_data["platforms"][platform]
                        
                        if model_name in platform_hist.get("models", {}):
                            model_hist = platform_hist["models"][model_name]
                            
                            if batch_size in model_hist.get("batch_sizes", {}):
                                batch_hist = model_hist["batch_sizes"][batch_size]
                                
                                # Compare throughput
                                hist_throughput = batch_hist.get("avg_throughput", 0)
                                if hist_throughput > 0 and current_throughput > 0:
                                    throughput_change = ((hist_throughput - current_throughput) / hist_throughput) * 100
                                    
                                    if throughput_change > self.threshold:
                                        severity = self._get_severity_level(throughput_change)
                                        
                                        self.regressions.append({
                                            "platform": platform,
                                            "model": model_name,
                                            "batch_size": batch_size,
                                            "metric": "throughput",
                                            "current_value": current_throughput,
                                            "historical_value": hist_throughput,
                                            "change_percent": throughput_change,
                                            "severity": severity,
                                            "samples": len(batch_hist["throughput"])
                                        })
                                
                                # Compare latency
                                hist_latency = batch_hist.get("avg_latency", 0)
                                if hist_latency > 0 and current_latency > 0:
                                    # For latency, higher is worse, so formula is reversed
                                    latency_change = ((current_latency - hist_latency) / hist_latency) * 100
                                    
                                    if latency_change > self.threshold:
                                        severity = self._get_severity_level(latency_change)
                                        
                                        self.regressions.append({
                                            "platform": platform,
                                            "model": model_name,
                                            "batch_size": batch_size,
                                            "metric": "latency",
                                            "current_value": current_latency,
                                            "historical_value": hist_latency,
                                            "change_percent": latency_change,
                                            "severity": severity,
                                            "samples": len(batch_hist["latency"])
                                        })
        
        # Sort regressions by severity and change percentage
        self.regressions.sort(key=lambda x: (
            list(self.SEVERITY_LEVELS.keys()).index(x["severity"]),
            -x["change_percent"]
        ))
        
        logger.info(f"Detected {len(self.regressions)} regressions above threshold {self.threshold}%")
        return self.regressions
    
    def _get_severity_level(self, change_percent: float) -> str:
        """
        Determine the severity level based on the change percentage.
        
        Args:
            change_percent: Percentage change in performance
            
        Returns:
            Severity level string
        """
        for level, config in sorted(
            self.SEVERITY_LEVELS.items(),
            key=lambda x: x[1]["threshold"],
            reverse=True
        ):
            if change_percent >= config["threshold"]:
                return level
        
        return "low"
    
    def has_critical_regressions(self) -> bool:
        """
        Check if there are any critical regressions.
        
        Returns:
            True if critical regressions exist
        """
        return any(r["severity"] == "critical" for r in self.regressions)
    
    def generate_report(self, output_path: str, format: str = "markdown") -> str:
        """
        Generate a regression report.
        
        Args:
            output_path: Path to output report file
            format: Report format (json, markdown, text)
            
        Returns:
            Path to report file
        """
        if not self.regressions:
            logger.info("No regressions to report")
            
            # Create empty report based on format
            if format == "json":
                report_content = json.dumps({"regressions": []}, indent=2)
            elif format == "markdown":
                report_content = "# Mobile Performance Regression Report\n\n"
                report_content += f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                report_content += "No performance regressions detected.\n"
            else:  # text
                report_content = "Mobile Performance Regression Report\n"
                report_content += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                report_content += "No performance regressions detected.\n"
            
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            return output_path
        
        # Generate report based on format
        if format == "json":
            report_content = json.dumps({
                "timestamp": datetime.datetime.now().isoformat(),
                "threshold": self.threshold,
                "days_analyzed": self.days_lookback,
                "regressions": self.regressions
            }, indent=2)
        
        elif format == "markdown":
            report_content = "# Mobile Performance Regression Report\n\n"
            report_content += f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report_content += f"**Analysis Threshold:** {self.threshold}%\n"
            report_content += f"**Historical Period:** {self.days_lookback} days\n"
            report_content += f"**Regressions Detected:** {len(self.regressions)}\n\n"
            
            if self.has_critical_regressions():
                report_content += "⚠️ **CRITICAL REGRESSIONS DETECTED** ⚠️\n\n"
            
            report_content += "## Summary\n\n"
            report_content += "| Platform | Model | Batch Size | Metric | Change (%) | Severity |\n"
            report_content += "|----------|-------|------------|--------|------------|----------|\n"
            
            for reg in self.regressions:
                platform = reg["platform"]
                model = reg["model"]
                batch_size = reg["batch_size"]
                metric = reg["metric"]
                change = reg["change_percent"]
                severity = reg["severity"]
                
                report_content += f"| {platform} | {model} | {batch_size} | {metric} | {change:.2f}% | {severity} |\n"
            
            report_content += "\n## Detailed Regressions\n\n"
            
            for i, reg in enumerate(self.regressions):
                platform = reg["platform"]
                model = reg["model"]
                batch_size = reg["batch_size"]
                metric = reg["metric"]
                current = reg["current_value"]
                historical = reg["historical_value"]
                change = reg["change_percent"]
                severity = reg["severity"]
                samples = reg["samples"]
                
                report_content += f"### Regression {i+1}: {platform} - {model}\n\n"
                report_content += f"**Severity:** {severity}\n"
                report_content += f"**Metric:** {metric}\n"
                report_content += f"**Batch Size:** {batch_size}\n"
                report_content += f"**Current Value:** {current:.2f}\n"
                report_content += f"**Historical Average:** {historical:.2f} (from {samples} samples)\n"
                report_content += f"**Change:** {change:.2f}%\n\n"
        
        else:  # text
            report_content = "Mobile Performance Regression Report\n"
            report_content += "="*40 + "\n"
            report_content += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += f"Analysis Threshold: {self.threshold}%\n"
            report_content += f"Historical Period: {self.days_lookback} days\n"
            report_content += f"Regressions Detected: {len(self.regressions)}\n\n"
            
            if self.has_critical_regressions():
                report_content += "!!! CRITICAL REGRESSIONS DETECTED !!!\n\n"
            
            report_content += "Summary:\n"
            report_content += "-"*40 + "\n"
            
            for reg in self.regressions:
                platform = reg["platform"]
                model = reg["model"]
                batch_size = reg["batch_size"]
                metric = reg["metric"]
                change = reg["change_percent"]
                severity = reg["severity"]
                
                report_content += f"{platform} - {model} (batch {batch_size}): {metric} regressed by {change:.2f}% [{severity}]\n"
            
            report_content += "\nDetailed Regressions:\n"
            report_content += "="*40 + "\n\n"
            
            for i, reg in enumerate(self.regressions):
                platform = reg["platform"]
                model = reg["model"]
                batch_size = reg["batch_size"]
                metric = reg["metric"]
                current = reg["current_value"]
                historical = reg["historical_value"]
                change = reg["change_percent"]
                severity = reg["severity"]
                samples = reg["samples"]
                
                report_content += f"Regression {i+1}: {platform} - {model}\n"
                report_content += f"  Severity: {severity}\n"
                report_content += f"  Metric: {metric}\n"
                report_content += f"  Batch Size: {batch_size}\n"
                report_content += f"  Current Value: {current:.2f}\n"
                report_content += f"  Historical Average: {historical:.2f} (from {samples} samples)\n"
                report_content += f"  Change: {change:.2f}%\n\n"
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Regression report saved to {output_path}")
        return output_path
    
    def run(self, output_path: Optional[str] = None, format: str = "text") -> bool:
        """
        Run the complete regression detection process.
        
        Args:
            output_path: Optional path for regression report
            format: Report format (json, markdown, text)
            
        Returns:
            Success status
        """
        # Load current data
        if not self.load_current_data():
            return False
        
        # Attempt to load historical data if database provided
        if self.db_path:
            self.load_historical_data()
        
        # Detect regressions
        self.detect_regressions()
        
        # Generate report if path provided
        if output_path:
            self.generate_report(output_path, format)
        
        # Print summary to console
        if self.regressions:
            print("\nMobile Performance Regressions Detected:")
            print("-"*50)
            
            for reg in self.regressions:
                platform = reg["platform"]
                model = reg["model"]
                metric = reg["metric"]
                change = reg["change_percent"]
                severity = reg["severity"]
                
                print(f"{platform} - {model}: {metric} regressed by {change:.2f}% [{severity}]")
            
            print("\nSee report for details.")
            
            # Return failure if there are critical regressions
            if self.has_critical_regressions():
                logger.warning("Critical regressions detected!")
                return False
        else:
            print("\nNo performance regressions detected above threshold.")
        
        return True


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Mobile Performance Regression Detection Tool")
    
    parser.add_argument("--data-file", required=True, help="Path to JSON data file with benchmark results")
    parser.add_argument("--db-path", help="Optional path to DuckDB database for historical data")
    parser.add_argument("--threshold", type=float, default=10.0, 
                       help="Percentage threshold for regression detection (default: 10.0)")
    parser.add_argument("--output", help="Path to output report file")
    parser.add_argument("--format", choices=["json", "markdown", "text"], default="text",
                       help="Report format (default: text)")
    parser.add_argument("--days", type=int, default=14,
                       help="Number of days to look back for historical data (default: 14)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = MobileRegressionDetector(
            data_file=args.data_file,
            db_path=args.db_path,
            threshold=args.threshold,
            days=args.days,
            verbose=args.verbose
        )
        
        # Run detection process
        result = detector.run(args.output, args.format)
        
        # Return exit code based on result
        return 0 if result else 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())