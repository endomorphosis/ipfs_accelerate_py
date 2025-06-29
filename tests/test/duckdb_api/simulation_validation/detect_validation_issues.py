#!/usr/bin/env python3
"""
Validation Issue Detection for Simulation Validation Framework

This module analyzes validation results and detects potential issues or anomalies.
It identifies high discrepancies between simulation and hardware results,
unexpected patterns in error distribution, and provides severity classifications.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("detect_validation_issues")

class ValidationIssueDetector:
    """
    Detects validation issues and anomalies in simulation validation results.
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize the issue detector.
        
        Args:
            threshold: Threshold for MAPE above which to flag issues (0.1 = 10%)
        """
        self.threshold = threshold
        
    def load_validation_results(self, results_dir: str) -> Dict[str, Any]:
        """
        Load validation results from a directory.
        
        Args:
            results_dir: Directory containing validation results
            
        Returns:
            Dictionary of validation results
        """
        try:
            # Look for validation_results.json in the directory
            validation_file = os.path.join(results_dir, "validation_results.json")
            if os.path.exists(validation_file):
                with open(validation_file, 'r') as f:
                    return json.load(f)
            
            # If not found, try alternatives
            alternatives = [
                "simulation_vs_hardware_results.json",
                "validation_summary.json",
                "results.json"
            ]
            
            for alt in alternatives:
                alt_file = os.path.join(results_dir, alt)
                if os.path.exists(alt_file):
                    with open(alt_file, 'r') as f:
                        return json.load(f)
            
            # If no existing files, look for any JSON file with validation in name
            json_files = list(Path(results_dir).glob("*validation*.json"))
            if json_files:
                with open(json_files[0], 'r') as f:
                    return json.load(f)
                    
            logger.error(f"No validation results found in {results_dir}")
            return {}
            
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")
            return {}
    
    def detect_issues(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect issues in validation results.
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Check if the results structure contains the expected data
        if not validation_results or not isinstance(validation_results, dict):
            issues.append({
                "issue_type": "missing_data",
                "description": "No validation results found or invalid format",
                "severity": "high",
                "recommendation": "Verify validation process completed successfully"
            })
            return issues
        
        # Extract key components from validation results
        validation_items = validation_results.get("validation_items", [])
        summary = validation_results.get("summary", {})
        metrics = validation_results.get("metrics", {})
        
        # Check for empty validation results
        if not validation_items and not summary and not metrics:
            issues.append({
                "issue_type": "empty_results",
                "description": "Validation results contain no data",
                "severity": "high",
                "recommendation": "Check validation process and inputs"
            })
            return issues
        
        # Check overall MAPE if available
        if "overall_mape" in summary:
            overall_mape = summary["overall_mape"]
            if overall_mape > self.threshold * 2:  # Higher threshold for overall
                issues.append({
                    "issue_type": "high_overall_error",
                    "description": f"Overall MAPE of {overall_mape:.2%} exceeds threshold of {self.threshold*2:.2%}",
                    "severity": "high",
                    "metric": "overall_mape",
                    "value": overall_mape,
                    "threshold": self.threshold * 2,
                    "recommendation": "Review simulation parameters and calibration"
                })
        
        # Check for hardware-model pairs with high error
        high_error_pairs = []
        
        # Process validation items
        for item in validation_items:
            hardware_id = item.get("hardware_id", "unknown")
            model_id = item.get("model_id", "unknown")
            metrics_comparison = item.get("metrics_comparison", {})
            
            for metric_name, metric_data in metrics_comparison.items():
                # Skip non-numeric metrics
                if not isinstance(metric_data.get("mape"), (int, float)):
                    continue
                    
                mape = metric_data["mape"]
                if mape > self.threshold:
                    high_error_pairs.append({
                        "hardware_id": hardware_id,
                        "model_id": model_id,
                        "metric": metric_name,
                        "mape": mape,
                        "simulation_value": metric_data.get("simulation_value"),
                        "hardware_value": metric_data.get("hardware_value")
                    })
        
        # Report high error pairs if found
        if high_error_pairs:
            # Group by hardware and model
            hardware_models = {}
            for pair in high_error_pairs:
                key = f"{pair['hardware_id']}_{pair['model_id']}"
                if key not in hardware_models:
                    hardware_models[key] = {
                        "hardware_id": pair["hardware_id"],
                        "model_id": pair["model_id"],
                        "metrics": []
                    }
                hardware_models[key]["metrics"].append({
                    "name": pair["metric"],
                    "mape": pair["mape"],
                    "simulation_value": pair["simulation_value"],
                    "hardware_value": pair["hardware_value"]
                })
            
            # Create issues for each hardware-model pair
            for _, hm_data in hardware_models.items():
                # Calculate average MAPE for this hardware-model pair
                avg_mape = sum(m["mape"] for m in hm_data["metrics"]) / len(hm_data["metrics"])
                
                # Determine severity based on MAPE and number of metrics
                if avg_mape > self.threshold * 2 or len(hm_data["metrics"]) > 2:
                    severity = "high"
                elif avg_mape > self.threshold * 1.5:
                    severity = "medium"
                else:
                    severity = "low"
                
                issues.append({
                    "issue_type": "high_error_hardware_model",
                    "description": f"High error detected for {hm_data['hardware_id']} - {hm_data['model_id']} (avg MAPE: {avg_mape:.2%})",
                    "severity": severity,
                    "hardware_id": hm_data["hardware_id"],
                    "model_id": hm_data["model_id"],
                    "metrics": hm_data["metrics"],
                    "avg_mape": avg_mape,
                    "threshold": self.threshold,
                    "recommendation": "Calibrate simulation parameters for this hardware-model pair"
                })
        
        # Check for metrics with consistently high error across hardware
        metric_errors = {}
        for item in validation_items:
            for metric_name, metric_data in item.get("metrics_comparison", {}).items():
                if not isinstance(metric_data.get("mape"), (int, float)):
                    continue
                    
                if metric_name not in metric_errors:
                    metric_errors[metric_name] = []
                    
                metric_errors[metric_name].append(metric_data["mape"])
        
        # Analyze metrics with consistently high errors
        for metric_name, errors in metric_errors.items():
            if len(errors) < 2:
                continue
                
            avg_error = sum(errors) / len(errors)
            if avg_error > self.threshold:
                severity = "high" if avg_error > self.threshold * 2 else "medium"
                issues.append({
                    "issue_type": "consistent_metric_error",
                    "description": f"Metric '{metric_name}' shows consistently high error across hardware (avg: {avg_error:.2%})",
                    "severity": severity,
                    "metric": metric_name,
                    "avg_error": avg_error,
                    "error_values": errors,
                    "threshold": self.threshold,
                    "recommendation": "Review simulation logic for this specific metric"
                })
        
        # Check for unusual patterns in error distribution
        for metric_name, errors in metric_errors.items():
            if len(errors) < 5:  # Need enough data points
                continue
                
            # Look for bimodal distribution (some very high, some very low)
            errors_array = np.array(errors)
            if len(errors_array) > 0:
                q25 = np.percentile(errors_array, 25)
                q75 = np.percentile(errors_array, 75)
                iqr = q75 - q25
                
                # Check for bimodal distribution or high variance
                if iqr > self.threshold or np.std(errors_array) / np.mean(errors_array) > 1.0:
                    issues.append({
                        "issue_type": "unusual_error_distribution",
                        "description": f"Unusual error distribution for metric '{metric_name}' (high variance)",
                        "severity": "medium",
                        "metric": metric_name,
                        "stats": {
                            "mean": float(np.mean(errors_array)),
                            "std": float(np.std(errors_array)),
                            "min": float(np.min(errors_array)),
                            "max": float(np.max(errors_array)),
                            "q25": float(q25),
                            "q75": float(q75),
                            "iqr": float(iqr)
                        },
                        "recommendation": "Investigate hardware-specific factors affecting this metric"
                    })
        
        # Check drift detection results if available
        drift_results = validation_results.get("drift_detection", {})
        if drift_results and drift_results.get("is_significant", False):
            issues.append({
                "issue_type": "significant_drift",
                "description": "Significant drift detected in simulation accuracy",
                "severity": "high",
                "drift_metrics": drift_results.get("drift_metrics", {}),
                "recommendation": "Review recent changes to simulation or hardware and calibrate"
            })
            
        return issues
    
    def generate_report(
        self, 
        issues: List[Dict[str, Any]], 
        format: str = "json", 
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a report of detected issues.
        
        Args:
            issues: List of detected issues
            format: Report format (json, markdown, html)
            output_file: File to write report to (optional)
            
        Returns:
            Report content
        """
        if not issues:
            summary = "No validation issues detected."
            if format == "json":
                result = json.dumps({"status": "ok", "issues": [], "summary": summary}, indent=2)
            elif format == "markdown":
                result = f"# Validation Issues Report\n\n{summary}\n"
            elif format == "html":
                result = f"<h1>Validation Issues Report</h1><p>{summary}</p>"
            else:
                result = summary
                
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(result)
                    
            return result
        
        # Count issues by severity
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity = issue.get("severity", "low")
            severity_counts[severity] += 1
        
        # Generate report in requested format
        if format == "json":
            result = json.dumps({
                "status": "issues_found",
                "issues": issues,
                "summary": {
                    "total_issues": len(issues),
                    "by_severity": severity_counts,
                    "timestamp": datetime.now().isoformat()
                }
            }, indent=2)
            
        elif format == "markdown":
            lines = ["# Validation Issues Report\n"]
            
            # Summary section
            lines.append("## Summary\n")
            lines.append(f"- **Total Issues**: {len(issues)}")
            lines.append(f"- **High Severity Issues**: {severity_counts['high']}")
            lines.append(f"- **Medium Severity Issues**: {severity_counts['medium']}")
            lines.append(f"- **Low Severity Issues**: {severity_counts['low']}")
            lines.append(f"- **Timestamp**: {datetime.now().isoformat()}\n")
            
            # High severity issues first
            if severity_counts["high"] > 0:
                lines.append("## High Severity Issues\n")
                for i, issue in enumerate([i for i in issues if i.get("severity") == "high"]):
                    lines.append(f"### Issue {i+1}: {issue['issue_type']}\n")
                    lines.append(f"**Description**: {issue['description']}")
                    if "hardware_id" in issue and "model_id" in issue:
                        lines.append(f"\n**Hardware**: {issue['hardware_id']}")
                        lines.append(f"**Model**: {issue['model_id']}")
                    if "metric" in issue:
                        lines.append(f"**Metric**: {issue['metric']}")
                    if "recommendation" in issue:
                        lines.append(f"\n**Recommendation**: {issue['recommendation']}")
                    lines.append("\n")
            
            # Medium severity issues
            if severity_counts["medium"] > 0:
                lines.append("## Medium Severity Issues\n")
                for i, issue in enumerate([i for i in issues if i.get("severity") == "medium"]):
                    lines.append(f"### Issue {i+1}: {issue['issue_type']}\n")
                    lines.append(f"**Description**: {issue['description']}")
                    if "hardware_id" in issue and "model_id" in issue:
                        lines.append(f"\n**Hardware**: {issue['hardware_id']}")
                        lines.append(f"**Model**: {issue['model_id']}")
                    if "metric" in issue:
                        lines.append(f"**Metric**: {issue['metric']}")
                    if "recommendation" in issue:
                        lines.append(f"\n**Recommendation**: {issue['recommendation']}")
                    lines.append("\n")
            
            # Low severity issues (abbreviated)
            if severity_counts["low"] > 0:
                lines.append("## Low Severity Issues\n")
                for i, issue in enumerate([i for i in issues if i.get("severity") == "low"]):
                    lines.append(f"- **{issue['issue_type']}**: {issue['description']}")
                    if "recommendation" in issue:
                        lines.append(f" - *Recommendation*: {issue['recommendation']}")
                    lines.append("\n")
                    
            result = "\n".join(lines)
            
        elif format == "html":
            # Create HTML report
            html_lines = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <title>Validation Issues Report</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 20px; }",
                "        .high { background-color: #ffdddd; border-left: 5px solid #f44336; padding: 10px; }",
                "        .medium { background-color: #ffffcc; border-left: 5px solid #ffeb3b; padding: 10px; }",
                "        .low { background-color: #e7f3fe; border-left: 5px solid #2196F3; padding: 10px; }",
                "        .summary { background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; }",
                "    </style>",
                "</head>",
                "<body>",
                "    <h1>Validation Issues Report</h1>",
                ""
            ]
            
            # Summary section
            html_lines.extend([
                "    <div class='summary'>",
                "        <h2>Summary</h2>",
                f"        <p><strong>Total Issues</strong>: {len(issues)}</p>",
                f"        <p><strong>High Severity Issues</strong>: {severity_counts['high']}</p>",
                f"        <p><strong>Medium Severity Issues</strong>: {severity_counts['medium']}</p>",
                f"        <p><strong>Low Severity Issues</strong>: {severity_counts['low']}</p>",
                f"        <p><strong>Timestamp</strong>: {datetime.now().isoformat()}</p>",
                "    </div>",
                ""
            ])
            
            # High severity issues first
            if severity_counts["high"] > 0:
                html_lines.append("    <h2>High Severity Issues</h2>")
                for i, issue in enumerate([i for i in issues if i.get("severity") == "high"]):
                    html_lines.extend([
                        "    <div class='high'>",
                        f"        <h3>Issue {i+1}: {issue['issue_type']}</h3>",
                        f"        <p><strong>Description</strong>: {issue['description']}</p>"
                    ])
                    
                    if "hardware_id" in issue and "model_id" in issue:
                        html_lines.extend([
                            f"        <p><strong>Hardware</strong>: {issue['hardware_id']}<br>",
                            f"        <strong>Model</strong>: {issue['model_id']}</p>"
                        ])
                        
                    if "metric" in issue:
                        html_lines.append(f"        <p><strong>Metric</strong>: {issue['metric']}</p>")
                        
                    if "recommendation" in issue:
                        html_lines.append(f"        <p><strong>Recommendation</strong>: {issue['recommendation']}</p>")
                        
                    html_lines.append("    </div>")
                    html_lines.append("")
            
            # Medium severity issues
            if severity_counts["medium"] > 0:
                html_lines.append("    <h2>Medium Severity Issues</h2>")
                for i, issue in enumerate([i for i in issues if i.get("severity") == "medium"]):
                    html_lines.extend([
                        "    <div class='medium'>",
                        f"        <h3>Issue {i+1}: {issue['issue_type']}</h3>",
                        f"        <p><strong>Description</strong>: {issue['description']}</p>"
                    ])
                    
                    if "hardware_id" in issue and "model_id" in issue:
                        html_lines.extend([
                            f"        <p><strong>Hardware</strong>: {issue['hardware_id']}<br>",
                            f"        <strong>Model</strong>: {issue['model_id']}</p>"
                        ])
                        
                    if "metric" in issue:
                        html_lines.append(f"        <p><strong>Metric</strong>: {issue['metric']}</p>")
                        
                    if "recommendation" in issue:
                        html_lines.append(f"        <p><strong>Recommendation</strong>: {issue['recommendation']}</p>")
                        
                    html_lines.append("    </div>")
                    html_lines.append("")
            
            # Low severity issues
            if severity_counts["low"] > 0:
                html_lines.append("    <h2>Low Severity Issues</h2>")
                html_lines.append("    <div class='low'>")
                html_lines.append("        <ul>")
                
                for issue in [i for i in issues if i.get("severity") == "low"]:
                    html_lines.append(f"            <li><strong>{issue['issue_type']}</strong>: {issue['description']}")
                    if "recommendation" in issue:
                        html_lines.append(f" - <em>Recommendation</em>: {issue['recommendation']}")
                    html_lines.append("            </li>")
                    
                html_lines.append("        </ul>")
                html_lines.append("    </div>")
            
            html_lines.extend([
                "</body>",
                "</html>"
            ])
            
            result = "\n".join(html_lines)
            
        else:
            # Plain text output
            lines = ["Validation Issues Report", "=" * 25, ""]
            lines.append(f"Total Issues: {len(issues)}")
            lines.append(f"High Severity Issues: {severity_counts['high']}")
            lines.append(f"Medium Severity Issues: {severity_counts['medium']}")
            lines.append(f"Low Severity Issues: {severity_counts['low']}")
            lines.append(f"Timestamp: {datetime.now().isoformat()}")
            lines.append("")
            
            # List all issues
            for i, issue in enumerate(issues):
                severity = issue.get("severity", "low")
                lines.append(f"Issue {i+1}: {issue['issue_type']} ({severity.upper()})")
                lines.append(f"Description: {issue['description']}")
                
                if "hardware_id" in issue and "model_id" in issue:
                    lines.append(f"Hardware: {issue['hardware_id']}")
                    lines.append(f"Model: {issue['model_id']}")
                    
                if "metric" in issue:
                    lines.append(f"Metric: {issue['metric']}")
                    
                if "recommendation" in issue:
                    lines.append(f"Recommendation: {issue['recommendation']}")
                    
                lines.append("")
                
            result = "\n".join(lines)
        
        # Write to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
                
        return result


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Detect validation issues in simulation validation results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing validation results")
    parser.add_argument("--threshold", type=float, default=0.1, help="MAPE threshold for flagging issues (default: 0.1)")
    parser.add_argument("--output-format", type=str, default="text", choices=["text", "json", "markdown", "html"], help="Output format")
    parser.add_argument("--output-file", type=str, help="Output file (default: validation_issues.<format>)")
    
    args = parser.parse_args()
    
    # Validate results directory
    if not os.path.isdir(args.results_dir):
        logger.error(f"Results directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    # Create issue detector
    detector = ValidationIssueDetector(threshold=args.threshold)
    
    # Load validation results
    validation_results = detector.load_validation_results(args.results_dir)
    
    # Detect issues
    issues = detector.detect_issues(validation_results)
    
    # Default output file based on format
    if not args.output_file:
        if args.output_format == "json":
            output_file = os.path.join(args.results_dir, "validation_issues.json")
        elif args.output_format == "markdown":
            output_file = os.path.join(args.results_dir, "validation_issues.md")
        elif args.output_format == "html":
            output_file = os.path.join(args.results_dir, "validation_issues.html")
        else:
            output_file = os.path.join(args.results_dir, "validation_issues.txt")
    else:
        output_file = args.output_file
    
    # Generate report
    report = detector.generate_report(issues, format=args.output_format, output_file=output_file)
    
    # For JSON output, print to stdout for potential parsing
    if args.output_format == "json":
        print(report)
    else:
        high_severity = sum(1 for i in issues if i.get("severity") == "high")
        medium_severity = sum(1 for i in issues if i.get("severity") == "medium")
        
        if high_severity > 0:
            print(f"Found {len(issues)} issues ({high_severity} high, {medium_severity} medium)")
            print(f"Full report written to {output_file}")
        else:
            print(f"Found {len(issues)} issues (no high severity issues)")
            print(f"Full report written to {output_file}")
    
    # Return 0 if no high severity issues, 1 otherwise (for CI integration)
    return 0 if sum(1 for i in issues if i.get("severity") == "high") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())