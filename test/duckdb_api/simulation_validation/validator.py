"""
Simulation Accuracy Validation Framework for IPFS Accelerate.

This module provides tools for validating the accuracy of simulation results
against real hardware results. It implements statistical validation methods,
calibration systems, and drift detection for simulation accuracy.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("simulation_validator")

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class ValidationResult:
    """Stores results of a simulation validation run."""
    
    def __init__(self, real_data=None, sim_data=None):
        """Initialize with real and simulated data."""
        self.real_data = real_data
        self.sim_data = sim_data
        self.metrics = {}
        self.valid = False
        self.error_details = {}
        self.timestamp = datetime.now().isoformat()
        
    def calculate_metrics(self, metric_names=None):
        """Calculate validation metrics."""
        if self.real_data is None or self.sim_data is None:
            raise ValueError("Real and simulated data must be provided")
        
        # Default metrics if none specified
        if metric_names is None:
            metric_names = ["mae", "mape", "rmse", "r2"]
        
        # Calculate each requested metric
        for metric in metric_names:
            if metric == "mae":
                self.metrics["mae"] = self._calculate_mae()
            elif metric == "mape":
                self.metrics["mape"] = self._calculate_mape()
            elif metric == "rmse":
                self.metrics["rmse"] = self._calculate_rmse()
            elif metric == "r2":
                self.metrics["r2"] = self._calculate_r2()
        
        return self.metrics
    
    def _calculate_mae(self):
        """Calculate Mean Absolute Error."""
        result = {}
        for col in self.real_data.select_dtypes(include=['number']).columns:
            if col in self.sim_data.columns:
                real_values = self.real_data[col].values
                sim_values = self.sim_data[col].values
                result[col] = np.mean(np.abs(real_values - sim_values))
        return result
    
    def _calculate_mape(self):
        """Calculate Mean Absolute Percentage Error."""
        result = {}
        for col in self.real_data.select_dtypes(include=['number']).columns:
            if col in self.sim_data.columns:
                real_values = self.real_data[col].values
                sim_values = self.sim_data[col].values
                # Avoid division by zero
                mask = real_values != 0
                if mask.any():
                    result[col] = np.mean(np.abs((real_values[mask] - sim_values[mask]) / real_values[mask])) * 100
                else:
                    result[col] = np.nan
        return result
    
    def _calculate_rmse(self):
        """Calculate Root Mean Square Error."""
        result = {}
        for col in self.real_data.select_dtypes(include=['number']).columns:
            if col in self.sim_data.columns:
                real_values = self.real_data[col].values
                sim_values = self.sim_data[col].values
                result[col] = np.sqrt(np.mean((real_values - sim_values) ** 2))
        return result
    
    def _calculate_r2(self):
        """Calculate R-squared (coefficient of determination)."""
        result = {}
        for col in self.real_data.select_dtypes(include=['number']).columns:
            if col in self.sim_data.columns:
                real_values = self.real_data[col].values
                sim_values = self.sim_data[col].values
                
                # Calculate R²
                mean_real = np.mean(real_values)
                ss_total = np.sum((real_values - mean_real) ** 2)
                ss_residual = np.sum((real_values - sim_values) ** 2)
                
                if ss_total > 0:
                    result[col] = 1 - (ss_residual / ss_total)
                else:
                    result[col] = np.nan
        return result
    
    def is_valid(self, tolerance=0.1):
        """Check if simulation is valid within tolerance."""
        if not self.metrics:
            self.calculate_metrics()
        
        # Check MAPE for each metric
        if "mape" in self.metrics:
            invalid_metrics = {}
            for metric, value in self.metrics["mape"].items():
                if not np.isnan(value) and value > tolerance * 100:  # tolerance as percentage
                    invalid_metrics[metric] = value
            
            self.valid = len(invalid_metrics) == 0
            self.error_details = invalid_metrics
        else:
            # If MAPE not available, use MAE
            if "mae" in self.metrics:
                invalid_metrics = {}
                for metric, value in self.metrics["mae"].items():
                    # For MAE, we need reference values to determine validity
                    mean_real = np.mean(self.real_data[metric].values)
                    if mean_real > 0 and value > tolerance * mean_real:
                        invalid_metrics[metric] = value
                
                self.valid = len(invalid_metrics) == 0
                self.error_details = invalid_metrics
        
        return self.valid
    
    def summarize(self):
        """Generate a summary of validation results."""
        if not self.metrics:
            self.calculate_metrics()
        
        summary = {
            "timestamp": self.timestamp,
            "valid": self.is_valid(),
            "metrics": self.metrics,
            "error_details": self.error_details,
            "sample_count": len(self.real_data) if self.real_data is not None else 0
        }
        
        return summary


class SimulationValidator:
    """Core validation framework for simulation accuracy."""
    
    def __init__(self, db_connection=None, real_data_source=None, sim_data_source=None, debug=False):
        """Initialize the validation framework."""
        self.db_connection = db_connection
        self.real_data_source = real_data_source
        self.sim_data_source = sim_data_source
        self.debug = debug
        self.validation_results = {}
        self.metrics = {}
        
    def validate_simulation(self, model_name=None, hardware_type=None, 
                          metrics=None, tolerance=0.1):
        """
        Validate simulation accuracy against real hardware results.
        
        Args:
            model_name: Model to validate (or None for all)
            hardware_type: Hardware to validate (or None for all)
            metrics: List of metrics to validate
            tolerance: Acceptable error tolerance
            
        Returns:
            ValidationResult object with validation results
        """
        # Get real and simulated data
        real_data = self._get_real_data(model_name, hardware_type, metrics)
        sim_data = self._get_simulated_data(model_name, hardware_type, metrics)
        
        if real_data is None or sim_data is None:
            if self.debug:
                # Generate sample data for debugging
                real_data, sim_data = self._generate_sample_data(model_name, hardware_type, metrics)
            else:
                raise ValueError("Failed to retrieve real or simulated data")
        
        # Align data
        aligned_real, aligned_sim = self._align_data(real_data, sim_data)
        
        # Create validation result
        result = ValidationResult(aligned_real, aligned_sim)
        result.calculate_metrics()
        result.is_valid(tolerance)
        
        # Store result
        key = f"{model_name or 'all'}-{hardware_type or 'all'}"
        self.validation_results[key] = result
        
        return result
    
    def _get_real_data(self, model_name=None, hardware_type=None, metrics=None):
        """Get real hardware benchmark data."""
        if self.db_connection is None and self.real_data_source is None:
            if self.debug:
                # Return None for debugging mode
                return None
            else:
                raise ValueError("Database connection or real data source is required")
        
        # If real data source is provided directly, use it
        if isinstance(self.real_data_source, pd.DataFrame):
            df = self.real_data_source.copy()
            
            # Apply filters
            if model_name is not None and "model_name" in df.columns:
                df = df[df["model_name"] == model_name]
            if hardware_type is not None and "hardware_type" in df.columns:
                df = df[df["hardware_type"] == hardware_type]
            
            return df
        
        # Otherwise, query the database
        if self.db_connection is not None:
            # Build query
            query = """
            SELECT *
            FROM benchmark_results
            WHERE is_simulation = FALSE
            """
            
            if model_name:
                query += f" AND model_name = '{model_name}'"
            if hardware_type:
                query += f" AND hardware_type = '{hardware_type}'"
            
            # Execute query
            try:
                return self.db_connection.execute(query).fetchdf()
            except Exception as e:
                logger.error(f"Error querying database for real data: {e}")
                return None
        
        return None
    
    def _get_simulated_data(self, model_name=None, hardware_type=None, metrics=None):
        """Get simulated benchmark data."""
        if self.db_connection is None and self.sim_data_source is None:
            if self.debug:
                # Return None for debugging mode
                return None
            else:
                raise ValueError("Database connection or simulation data source is required")
        
        # If simulation data source is provided directly, use it
        if isinstance(self.sim_data_source, pd.DataFrame):
            df = self.sim_data_source.copy()
            
            # Apply filters
            if model_name is not None and "model_name" in df.columns:
                df = df[df["model_name"] == model_name]
            if hardware_type is not None and "hardware_type" in df.columns:
                df = df[df["hardware_type"] == hardware_type]
            
            return df
        
        # Otherwise, query the database
        if self.db_connection is not None:
            # Build query
            query = """
            SELECT *
            FROM benchmark_results
            WHERE is_simulation = TRUE
            """
            
            if model_name:
                query += f" AND model_name = '{model_name}'"
            if hardware_type:
                query += f" AND hardware_type = '{hardware_type}'"
            
            # Execute query
            try:
                return self.db_connection.execute(query).fetchdf()
            except Exception as e:
                logger.error(f"Error querying database for simulation data: {e}")
                return None
        
        return None
    
    def _generate_sample_data(self, model_name=None, hardware_type=None, metrics=None):
        """Generate sample data for testing in debug mode."""
        logger.info("Generating sample data for simulation validation")
        
        # Set default model and hardware if not specified
        if model_name is None:
            model_name = "BERT"
        if hardware_type is None:
            hardware_types = ["CPU", "GPU", "WebGPU"]
        else:
            hardware_types = [hardware_type]
        
        # Create basic columns for the dataframes
        columns = ["model_name", "hardware_type", "batch_size", "throughput_items_per_second", "average_latency_ms", "memory_peak_mb"]
        if metrics is not None:
            for metric in metrics:
                if metric not in columns:
                    columns.append(metric)
        
        # Generate sample data for real hardware
        real_rows = []
        for hw in hardware_types:
            throughput_base = 200 if hw == "GPU" else 100
            latency_base = 20 if hw == "GPU" else 50
            memory_base = 1000 if hw == "GPU" else 500
            
            batch_sizes = [1, 2, 4, 8, 16]
            for batch_size in batch_sizes:
                # Scale with batch size
                throughput = throughput_base * (batch_size ** 0.8) * (1 + np.random.normal(0, 0.1))
                latency = latency_base * (batch_size ** 0.2) * (1 + np.random.normal(0, 0.1))
                memory = memory_base * (batch_size ** 0.5) * (1 + np.random.normal(0, 0.05))
                
                row = {
                    "model_name": model_name,
                    "hardware_type": hw,
                    "batch_size": batch_size,
                    "throughput_items_per_second": throughput,
                    "average_latency_ms": latency,
                    "memory_peak_mb": memory
                }
                
                # Add custom metrics if specified
                if metrics is not None:
                    for metric in metrics:
                        if metric not in row:
                            row[metric] = 100 * (1 + np.random.normal(0, 0.2))
                
                real_rows.append(row)
        
        # Generate sample data for simulations with some systematic bias
        sim_rows = []
        for hw in hardware_types:
            # Add some systematic bias to simulations (e.g., throughput overestimation)
            throughput_bias = 1.2 if hw == "GPU" else 1.1  # 20% or 10% overestimation
            latency_bias = 0.9 if hw == "GPU" else 0.95    # 10% or 5% underestimation
            memory_bias = 0.9                             # 10% underestimation
            
            batch_sizes = [1, 2, 4, 8, 16]
            for batch_size in batch_sizes:
                # Scale with batch size
                throughput = throughput_base * (batch_size ** 0.8) * throughput_bias * (1 + np.random.normal(0, 0.15))
                latency = latency_base * (batch_size ** 0.2) * latency_bias * (1 + np.random.normal(0, 0.15))
                memory = memory_base * (batch_size ** 0.5) * memory_bias * (1 + np.random.normal(0, 0.1))
                
                row = {
                    "model_name": model_name,
                    "hardware_type": hw,
                    "batch_size": batch_size,
                    "throughput_items_per_second": throughput,
                    "average_latency_ms": latency,
                    "memory_peak_mb": memory
                }
                
                # Add custom metrics if specified
                if metrics is not None:
                    for metric in metrics:
                        if metric not in row:
                            # Add some bias to simulation metrics
                            bias = 1.1 if np.random.random() > 0.5 else 0.9
                            row[metric] = 100 * bias * (1 + np.random.normal(0, 0.25))
                
                sim_rows.append(row)
        
        # Create dataframes
        real_df = pd.DataFrame(real_rows)
        sim_df = pd.DataFrame(sim_rows)
        
        return real_df, sim_df
    
    def _align_data(self, real_data, sim_data):
        """Align real and simulated data for comparison."""
        # This is a simplified alignment - in practice, this would be more sophisticated
        
        # Identify common columns for grouping
        group_cols = ['model_name', 'hardware_type', 'batch_size']
        metric_cols = ['throughput_items_per_second', 'average_latency_ms', 'memory_peak_mb']
        
        # Filter to only include columns that exist in both dataframes
        group_cols = [col for col in group_cols if col in real_data.columns and col in sim_data.columns]
        metric_cols = [col for col in metric_cols if col in real_data.columns and col in sim_data.columns]
        
        # Add any additional numeric columns
        for col in real_data.columns:
            if col not in group_cols and col not in metric_cols and pd.api.types.is_numeric_dtype(real_data[col]) and col in sim_data.columns:
                metric_cols.append(col)
        
        if not group_cols or not metric_cols:
            raise ValueError("Insufficient common columns for alignment")
        
        # Group and aggregate
        real_agg = real_data.groupby(group_cols)[metric_cols].mean().reset_index()
        sim_agg = sim_data.groupby(group_cols)[metric_cols].mean().reset_index()
        
        # Merge on common columns
        merged = pd.merge(real_agg, sim_agg, on=group_cols, suffixes=('_real', '_sim'))
        
        # Separate back into real and simulated dataframes
        real_cols = group_cols + [col for col in merged.columns if col.endswith('_real')]
        sim_cols = group_cols + [col for col in merged.columns if col.endswith('_sim')]
        
        aligned_real = merged[real_cols].copy()
        aligned_sim = merged[sim_cols].copy()
        
        # Rename columns to remove suffixes
        aligned_real.columns = [col.replace('_real', '') for col in aligned_real.columns]
        aligned_sim.columns = [col.replace('_sim', '') for col in aligned_sim.columns]
        
        return aligned_real, aligned_sim
    
    def get_validation_metrics(self):
        """Get validation metrics from the most recent validation run."""
        if not self.validation_results:
            raise ValueError("No validation results available")
        
        # Get the most recent validation result
        latest_key = list(self.validation_results.keys())[-1]
        return self.validation_results[latest_key].metrics
    
    def generate_validation_report(self, output_path=None, format="html"):
        """Generate a validation report based on validation results."""
        if not self.validation_results:
            raise ValueError("No validation results available")
        
        if format == "html":
            return self._generate_html_report(output_path)
        elif format == "json":
            return self._generate_json_report(output_path)
        elif format == "markdown":
            return self._generate_markdown_report(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_report(self, output_path=None):
        """Generate HTML validation report."""
        # This would use Jinja2 templates in a real implementation
        # Here's a simplified version
        
        html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }
                .summary { margin-bottom: 20px; }
                .metrics { margin-bottom: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .valid { color: green; }
                .invalid { color: red; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Simulation Validation Report</h1>
                <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            </div>
            
            <div class="summary">
                <h2>Validation Summary</h2>
                <table>
                    <tr>
                        <th>Configuration</th>
                        <th>Status</th>
                        <th>Sample Count</th>
                        <th>Invalid Metrics</th>
                    </tr>
        """
        
        for key, result in self.validation_results.items():
            status_class = "valid" if result.valid else "invalid"
            status_text = "VALID" if result.valid else "INVALID"
            invalid_metrics = ", ".join([f"{k}: {v:.2f}%" for k, v in result.error_details.items()])
            
            html += f"""
                    <tr>
                        <td>{key}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{len(result.real_data) if result.real_data is not None else 0}</td>
                        <td>{invalid_metrics}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="metrics">
                <h2>Validation Metrics</h2>
        """
        
        # Add metrics for each validation result
        for key, result in self.validation_results.items():
            html += f"""
                <h3>Configuration: {key}</h3>
                
                <h4>Mean Absolute Percentage Error (MAPE)</h4>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>MAPE (%)</th>
                    </tr>
            """
            
            if "mape" in result.metrics:
                for metric, value in result.metrics["mape"].items():
                    status_class = "valid" if value <= 10 else "invalid"  # 10% threshold
                    html += f"""
                        <tr>
                            <td>{metric}</td>
                            <td class="{status_class}">{value:.2f}%</td>
                        </tr>
                    """
            
            html += """
                </table>
                
                <h4>R-Squared (R²)</h4>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>R²</th>
                    </tr>
            """
            
            if "r2" in result.metrics:
                for metric, value in result.metrics["r2"].items():
                    status_class = "valid" if value >= 0.9 else "invalid"  # 0.9 threshold
                    html += f"""
                        <tr>
                            <td>{metric}</td>
                            <td class="{status_class}">{value:.4f}</td>
                        </tr>
                    """
            
            html += """
                </table>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            return output_path
        else:
            return html
    
    def _generate_json_report(self, output_path=None):
        """Generate JSON validation report."""
        import json
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "validation_results": {}
        }
        
        for key, result in self.validation_results.items():
            report["validation_results"][key] = result.summarize()
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            return output_path
        else:
            return json.dumps(report, indent=2)
    
    def _generate_markdown_report(self, output_path=None):
        """Generate Markdown validation report."""
        md = f"# Simulation Validation Report\n\n"
        md += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        md += "## Validation Summary\n\n"
        md += "| Configuration | Status | Sample Count | Invalid Metrics |\n"
        md += "| ------------- | ------ | ------------ | --------------- |\n"
        
        for key, result in self.validation_results.items():
            status_text = "✅ VALID" if result.valid else "❌ INVALID"
            invalid_metrics = ", ".join([f"{k}: {v:.2f}%" for k, v in result.error_details.items()])
            
            md += f"| {key} | {status_text} | {len(result.real_data) if result.real_data is not None else 0} | {invalid_metrics} |\n"
        
        md += "\n## Validation Metrics\n\n"
        
        # Add metrics for each validation result
        for key, result in self.validation_results.items():
            md += f"### Configuration: {key}\n\n"
            
            md += "#### Mean Absolute Percentage Error (MAPE)\n\n"
            md += "| Metric | MAPE (%) |\n"
            md += "| ------ | -------- |\n"
            
            if "mape" in result.metrics:
                for metric, value in result.metrics["mape"].items():
                    status_emoji = "✅" if value <= 10 else "❌"  # 10% threshold
                    md += f"| {metric} | {status_emoji} {value:.2f}% |\n"
            
            md += "\n#### R-Squared (R²)\n\n"
            md += "| Metric | R² |\n"
            md += "| ------ | -- |\n"
            
            if "r2" in result.metrics:
                for metric, value in result.metrics["r2"].items():
                    status_emoji = "✅" if value >= 0.9 else "❌"  # 0.9 threshold
                    md += f"| {metric} | {status_emoji} {value:.4f} |\n"
            
            md += "\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(md)
            return output_path
        else:
            return md