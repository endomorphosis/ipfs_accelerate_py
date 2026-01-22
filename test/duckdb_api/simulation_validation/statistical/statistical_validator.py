#!/usr/bin/env python3
"""
Statistical Validation Tools for the Simulation Accuracy and Validation Framework.

This module provides comprehensive statistical methods for validating simulation accuracy
against real hardware measurements, including error metrics, statistical tests,
confidence scoring, and visualization utilities.
"""

import os
import logging
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("statistical_validator")

# Add parent directories to path for module imports
import sys
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import base classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult,
    SimulationValidator
)

class StatisticalValidator(SimulationValidator):
    """
    Comprehensive statistical validator for simulation results.
    
    This class extends the basic validator to provide advanced statistical methods
    for validating simulation accuracy, including multiple error metrics, statistical
    tests, and confidence scoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the statistical validator.
        
        Args:
            config: Configuration options for the validator
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            # Core validation metrics
            "metrics_to_validate": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w",
                "initialization_time_ms",
                "warmup_time_ms"
            ],
            
            # Error metrics to calculate
            "error_metrics": [
                "absolute_error",           # Absolute difference
                "relative_error",           # Relative difference (decimal)
                "mape",                     # Mean Absolute Percentage Error
                "percent_error",            # Signed percentage error
                "rmse",                     # Root Mean Square Error (for multiple points)
                "normalized_rmse"           # RMSE normalized by range
            ],
            
            # Statistical tests
            "statistical_tests": {
                "enabled": True,
                "tests": [
                    "t_test",               # Student's t-test for mean differences
                    "mann_whitney",         # Mann-Whitney U test for distribution differences
                    "ks_test",              # Kolmogorov-Smirnov test for distribution differences
                    "pearson_correlation",  # Pearson correlation coefficient
                    "spearman_correlation"  # Spearman rank correlation coefficient
                ],
                "significance_level": 0.05  # p-value threshold for significance
            },
            
            # Accuracy thresholds for classification
            "accuracy_thresholds": {
                "mape": {
                    "excellent": 5.0,      # MAPE <= 5%
                    "good": 10.0,          # MAPE <= 10%
                    "acceptable": 15.0,    # MAPE <= 15%
                    "poor": 25.0,          # MAPE <= 25%
                    "unacceptable": float('inf')  # MAPE > 25%
                },
                "rmse": {
                    "excellent": 0.05,     # RMSE <= 0.05
                    "good": 0.10,          # RMSE <= 0.10
                    "acceptable": 0.15,    # RMSE <= 0.15
                    "poor": 0.25,          # RMSE <= 0.25
                    "unacceptable": float('inf')  # RMSE > 0.25
                }
            },
            
            # Advanced metrics
            "advanced_metrics": {
                "enabled": True,
                "metrics": [
                    "confidence_score",     # Overall confidence in simulation accuracy
                    "bias_score",           # Measure of systematic bias in simulation
                    "precision_score",      # Measure of simulation precision (consistency)
                    "reliability_score",    # Overall reliability score
                    "fidelity_score"        # Measure of how well simulation preserves relationships
                ]
            },
            
            # Multivariate analysis
            "multivariate_analysis": {
                "enabled": True,
                "methods": [
                    "pca",                  # Principal Component Analysis
                    "multidimensional_scaling"  # MDS for visualizing relationships
                ]
            },
            
            # Validation version
            "validation_version": "statistical_v1.0"
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def validate(
        self,
        simulation_result: SimulationResult,
        hardware_result: HardwareResult
    ) -> ValidationResult:
        """
        Validate a simulation result against real hardware measurements.
        
        Args:
            simulation_result: The simulation result to validate
            hardware_result: The real hardware result to compare against
            
        Returns:
            A ValidationResult with comparison metrics
        """
        # Check that model and hardware IDs match
        if simulation_result.model_id != hardware_result.model_id:
            logger.warning(f"Model ID mismatch: {simulation_result.model_id} vs {hardware_result.model_id}")
        
        if simulation_result.hardware_id != hardware_result.hardware_id:
            logger.warning(f"Hardware ID mismatch: {simulation_result.hardware_id} vs {hardware_result.hardware_id}")
        
        # Check that batch size and precision match
        if simulation_result.batch_size != hardware_result.batch_size:
            logger.warning(f"Batch size mismatch: {simulation_result.batch_size} vs {hardware_result.batch_size}")
        
        if simulation_result.precision != hardware_result.precision:
            logger.warning(f"Precision mismatch: {simulation_result.precision} vs {hardware_result.precision}")
        
        # Calculate metrics comparison
        metrics_comparison = self._calculate_error_metrics(
            simulation_result.metrics,
            hardware_result.metrics
        )
        
        # Run statistical tests if enabled
        if self.config["statistical_tests"]["enabled"]:
            statistical_test_results = self._run_statistical_tests(
                simulation_result.metrics,
                hardware_result.metrics
            )
            
            # Add statistical test results to metrics comparison
            for metric, tests in statistical_test_results.items():
                if metric in metrics_comparison:
                    metrics_comparison[metric].update(tests)
        
        # Generate additional metrics
        additional_metrics = self._calculate_advanced_metrics(
            simulation_result,
            hardware_result,
            metrics_comparison
        )
        
        # Create validation result
        validation_result = ValidationResult(
            simulation_result=simulation_result,
            hardware_result=hardware_result,
            metrics_comparison=metrics_comparison,
            validation_version=self.config["validation_version"],
            additional_metrics=additional_metrics
        )
        
        return validation_result
    
    def validate_batch(
        self,
        simulation_results: List[SimulationResult],
        hardware_results: List[HardwareResult]
    ) -> List[ValidationResult]:
        """
        Validate multiple simulation results against real hardware measurements.
        
        Args:
            simulation_results: The simulation results to validate
            hardware_results: The real hardware results to compare against
            
        Returns:
            A list of ValidationResults with comparison metrics
        """
        validation_results = []
        
        # Group results by model_id, hardware_id, batch_size, precision
        grouped_sim_results = self._group_results(simulation_results)
        grouped_hw_results = self._group_results(hardware_results)
        
        # Validate each group of results
        for group_key, sim_group in grouped_sim_results.items():
            if group_key in grouped_hw_results:
                hw_group = grouped_hw_results[group_key]
                
                # If multiple results in both groups, do a batch comparison
                if len(sim_group) > 1 and len(hw_group) > 1:
                    batch_validation = self._validate_result_groups(sim_group, hw_group)
                    if batch_validation:
                        validation_results.append(batch_validation)
                
                # Also do pairwise comparisons
                for i, sim_result in enumerate(sim_group):
                    if i < len(hw_group):
                        hw_result = hw_group[i]
                        validation_result = self.validate(sim_result, hw_result)
                        validation_results.append(validation_result)
            else:
                logger.warning(f"No matching hardware results for group: {group_key}")
        
        return validation_results
    
    def summarize_validation(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Generate a summary of validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            A dictionary with summary statistics
        """
        if not validation_results:
            return {"status": "error", "message": "No validation results provided"}
        
        summary = {
            "num_validations": len(validation_results),
            "metrics": {},
            "models": {},
            "hardware": {},
            "overall": {},
            "advanced": {}
        }
        
        # Extract all metrics being compared
        all_metrics = set()
        for val_result in validation_results:
            all_metrics.update(val_result.metrics_comparison.keys())
        
        # Calculate summary statistics for each metric
        for metric in all_metrics:
            metric_values = {
                "mape": [],
                "rmse": [],
                "absolute_error": [],
                "relative_error": []
            }
            
            for val_result in validation_results:
                if metric in val_result.metrics_comparison:
                    for error_metric, values in metric_values.items():
                        if error_metric in val_result.metrics_comparison[metric]:
                            value = val_result.metrics_comparison[metric][error_metric]
                            if not np.isnan(value):
                                values.append(value)
            
            summary["metrics"][metric] = {}
            for error_metric, values in metric_values.items():
                if values:
                    summary["metrics"][metric][error_metric] = {
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "std_dev": np.std(values),
                        "count": len(values)
                    }
        
        # Calculate summary statistics by model
        model_results = {}
        for val_result in validation_results:
            model_id = val_result.simulation_result.model_id
            
            if model_id not in model_results:
                model_results[model_id] = {
                    "mape": [],
                    "rmse": [],
                    "absolute_error": [],
                    "relative_error": []
                }
            
            # Collect error metrics across all performance metrics
            for metric, comparison in val_result.metrics_comparison.items():
                for error_metric in ["mape", "rmse", "absolute_error", "relative_error"]:
                    if error_metric in comparison:
                        value = comparison[error_metric]
                        if not np.isnan(value):
                            model_results[model_id][error_metric].append(value)
        
        # Calculate summary for each model
        for model_id, error_metrics in model_results.items():
            summary["models"][model_id] = {}
            
            for error_metric, values in error_metrics.items():
                if values:
                    summary["models"][model_id][error_metric] = {
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "std_dev": np.std(values),
                        "count": len(values)
                    }
        
        # Calculate summary statistics by hardware
        hardware_results = {}
        for val_result in validation_results:
            hardware_id = val_result.simulation_result.hardware_id
            
            if hardware_id not in hardware_results:
                hardware_results[hardware_id] = {
                    "mape": [],
                    "rmse": [],
                    "absolute_error": [],
                    "relative_error": []
                }
            
            # Collect error metrics across all performance metrics
            for metric, comparison in val_result.metrics_comparison.items():
                for error_metric in ["mape", "rmse", "absolute_error", "relative_error"]:
                    if error_metric in comparison:
                        value = comparison[error_metric]
                        if not np.isnan(value):
                            hardware_results[hardware_id][error_metric].append(value)
        
        # Calculate summary for each hardware
        for hardware_id, error_metrics in hardware_results.items():
            summary["hardware"][hardware_id] = {}
            
            for error_metric, values in error_metrics.items():
                if values:
                    summary["hardware"][hardware_id][error_metric] = {
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "std_dev": np.std(values),
                        "count": len(values)
                    }
        
        # Calculate overall statistics
        all_error_metrics = {
            "mape": [],
            "rmse": [],
            "absolute_error": [],
            "relative_error": []
        }
        
        for val_result in validation_results:
            for metric, comparison in val_result.metrics_comparison.items():
                for error_metric in all_error_metrics.keys():
                    if error_metric in comparison:
                        value = comparison[error_metric]
                        if not np.isnan(value):
                            all_error_metrics[error_metric].append(value)
        
        for error_metric, values in all_error_metrics.items():
            if values:
                summary["overall"][error_metric] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std_dev": np.std(values),
                    "count": len(values)
                }
        
        # Evaluate overall status based on mean MAPE
        if "mape" in summary["overall"]:
            mean_mape = summary["overall"]["mape"]["mean"]
            thresholds = self.config["accuracy_thresholds"]["mape"]
            
            if mean_mape <= thresholds["excellent"]:
                summary["overall"]["status"] = "excellent"
            elif mean_mape <= thresholds["good"]:
                summary["overall"]["status"] = "good"
            elif mean_mape <= thresholds["acceptable"]:
                summary["overall"]["status"] = "acceptable"
            elif mean_mape <= thresholds["poor"]:
                summary["overall"]["status"] = "poor"
            else:
                summary["overall"]["status"] = "unacceptable"
        
        # Collect advanced metrics
        if self.config["advanced_metrics"]["enabled"]:
            advanced_metrics = self.config["advanced_metrics"]["metrics"]
            for metric in advanced_metrics:
                values = []
                
                for val_result in validation_results:
                    if (val_result.additional_metrics and 
                        metric in val_result.additional_metrics):
                        value = val_result.additional_metrics[metric]
                        if not np.isnan(value):
                            values.append(value)
                
                if values:
                    summary["advanced"][metric] = {
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "std_dev": np.std(values),
                        "count": len(values)
                    }
        
        return summary
    
    def calculate_confidence_score(
        self,
        validation_results: List[ValidationResult],
        hardware_id: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Calculate a confidence score for simulation accuracy based on validation history.
        
        Args:
            validation_results: List of validation results to analyze
            hardware_id: Hardware identifier
            model_id: Model identifier
            
        Returns:
            Dictionary with confidence metrics
        """
        if not validation_results:
            return {
                "overall_confidence": 0.0,
                "components": {},
                "message": "No validation results available"
            }
        
        # Filter validation results for the specified hardware and model
        filtered_results = [
            result for result in validation_results
            if result.hardware_result.hardware_id == hardware_id and 
               result.hardware_result.model_id == model_id
        ]
        
        if not filtered_results:
            return {
                "overall_confidence": 0.0,
                "components": {},
                "message": f"No validation results for {hardware_id}/{model_id}"
            }
        
        # Calculate confidence components
        
        # 1. Accuracy component (based on MAPE)
        accuracy_scores = []
        for result in filtered_results:
            mape_values = []
            for metric, comparison in result.metrics_comparison.items():
                if "mape" in comparison and not np.isnan(comparison["mape"]):
                    mape_values.append(comparison["mape"])
            
            if mape_values:
                avg_mape = np.mean(mape_values)
                thresholds = self.config["accuracy_thresholds"]["mape"]
                
                if avg_mape <= thresholds["excellent"]:
                    accuracy_scores.append(1.0)
                elif avg_mape <= thresholds["good"]:
                    accuracy_scores.append(0.8)
                elif avg_mape <= thresholds["acceptable"]:
                    accuracy_scores.append(0.6)
                elif avg_mape <= thresholds["poor"]:
                    accuracy_scores.append(0.3)
                else:
                    accuracy_scores.append(0.0)
        
        accuracy_component = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # 2. Sample size component
        sample_count = len(filtered_results)
        sample_size_component = min(1.0, sample_count / 10)  # 10+ samples for full confidence
        
        # 3. Recency component
        now = datetime.datetime.now()
        timestamps = []
        for result in filtered_results:
            try:
                timestamp = datetime.datetime.fromisoformat(result.validation_timestamp)
                timestamps.append(timestamp)
            except (ValueError, TypeError):
                pass
        
        recency_component = 0.0
        if timestamps:
            most_recent = max(timestamps)
            age_days = (now - most_recent).days
            recency_component = max(0.0, 1.0 - (age_days / 30))  # 30 days for full decay
        
        # 4. Consistency component
        consistency_component = 0.0
        if len(filtered_results) >= 2:
            mape_stds = []
            for metric in self.config["metrics_to_validate"]:
                mape_values = []
                for result in filtered_results:
                    if (metric in result.metrics_comparison and 
                        "mape" in result.metrics_comparison[metric] and 
                        not np.isnan(result.metrics_comparison[metric]["mape"])):
                        mape_values.append(result.metrics_comparison[metric]["mape"])
                
                if len(mape_values) >= 2:
                    mape_stds.append(np.std(mape_values))
            
            if mape_stds:
                avg_std = np.mean(mape_stds)
                consistency_component = max(0.0, 1.0 - (avg_std / 20))  # 20% std for zero confidence
        
        # Combine components with weights
        component_weights = {
            "accuracy": 0.4,
            "sample_size": 0.2,
            "recency": 0.2,
            "consistency": 0.2
        }
        
        components = {
            "accuracy": accuracy_component,
            "sample_size": sample_size_component,
            "recency": recency_component,
            "consistency": consistency_component
        }
        
        overall_confidence = sum(
            component * component_weights[name]
            for name, component in components.items()
        )
        
        return {
            "overall_confidence": overall_confidence,
            "components": components,
            "interpretation": (
                "Very high confidence" if overall_confidence >= 0.8 else
                "High confidence" if overall_confidence >= 0.6 else
                "Moderate confidence" if overall_confidence >= 0.4 else
                "Low confidence" if overall_confidence >= 0.2 else
                "Very low confidence"
            )
        }
    
    def _calculate_error_metrics(
        self,
        simulation_metrics: Dict[str, float],
        hardware_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate error metrics between simulation and hardware results.
        
        Args:
            simulation_metrics: Metrics from simulation result
            hardware_metrics: Metrics from hardware result
            
        Returns:
            Dictionary mapping metric names to dictionaries of error metrics
        """
        metrics_comparison = {}
        
        # Determine which metrics to compare
        metrics_to_validate = self.config["metrics_to_validate"]
        error_metrics = self.config["error_metrics"]
        
        for metric_name in metrics_to_validate:
            # Skip if either simulation or hardware metric is missing
            if metric_name not in simulation_metrics or metric_name not in hardware_metrics:
                continue
            
            # Get metric values
            sim_value = simulation_metrics[metric_name]
            hw_value = hardware_metrics[metric_name]
            
            # Skip if either value is None
            if sim_value is None or hw_value is None:
                continue
            
            # Initialize comparison dict for this metric
            metrics_comparison[metric_name] = {}
            
            # Calculate error metrics
            if "absolute_error" in error_metrics:
                metrics_comparison[metric_name]["absolute_error"] = abs(sim_value - hw_value)
            
            if "relative_error" in error_metrics:
                if hw_value != 0:
                    metrics_comparison[metric_name]["relative_error"] = abs(sim_value - hw_value) / abs(hw_value)
                else:
                    # Handle division by zero
                    metrics_comparison[metric_name]["relative_error"] = float('nan')
            
            if "mape" in error_metrics:
                if hw_value != 0:
                    metrics_comparison[metric_name]["mape"] = abs(sim_value - hw_value) / abs(hw_value) * 100.0
                else:
                    # Handle division by zero
                    metrics_comparison[metric_name]["mape"] = float('nan')
            
            if "percent_error" in error_metrics:
                if hw_value != 0:
                    metrics_comparison[metric_name]["percent_error"] = (sim_value - hw_value) / abs(hw_value) * 100.0
                else:
                    # Handle division by zero
                    metrics_comparison[metric_name]["percent_error"] = float('nan')
            
            if "rmse" in error_metrics:
                # For a single point, RMSE equals absolute error
                metrics_comparison[metric_name]["rmse"] = abs(sim_value - hw_value)
            
            if "normalized_rmse" in error_metrics:
                # We need a normalization factor
                if hw_value != 0:
                    metrics_comparison[metric_name]["normalized_rmse"] = abs(sim_value - hw_value) / abs(hw_value)
                else:
                    metrics_comparison[metric_name]["normalized_rmse"] = float('nan')
        
        return metrics_comparison
    
    def _run_statistical_tests(
        self,
        simulation_metrics: Dict[str, float],
        hardware_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run statistical tests on simulation and hardware metrics.
        
        Args:
            simulation_metrics: Metrics from simulation result
            hardware_metrics: Metrics from hardware result
            
        Returns:
            Dictionary mapping metric names to dictionaries of test results
        """
        test_results = {}
        
        # Skip if we don't have scipy for statistical tests
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available for statistical tests")
            return test_results
        
        # Determine which metrics to test
        metrics_to_validate = self.config["metrics_to_validate"]
        tests = self.config["statistical_tests"]["tests"]
        significance_level = self.config["statistical_tests"]["significance_level"]
        
        # For single data points, we can't run some tests
        # Here, we're treating each metric as a separate data point for correlation tests
        
        # Extract values for metrics present in both results
        sim_values = []
        hw_values = []
        metric_keys = []
        
        for metric_name in metrics_to_validate:
            if (metric_name in simulation_metrics and metric_name in hardware_metrics and
                simulation_metrics[metric_name] is not None and hardware_metrics[metric_name] is not None):
                sim_values.append(simulation_metrics[metric_name])
                hw_values.append(hardware_metrics[metric_name])
                metric_keys.append(metric_name)
        
        # Skip tests if we don't have enough data points
        if len(sim_values) < 2:
            return test_results
        
        # Initialize the overall test results
        test_results["overall"] = {}
        
        # Pearson correlation
        if "pearson_correlation" in tests:
            try:
                r, p_value = stats.pearsonr(sim_values, hw_values)
                
                test_results["overall"]["pearson_correlation"] = {
                    "coefficient": float(r),
                    "p_value": float(p_value),
                    "is_significant": p_value < significance_level,
                    "interpretation": (
                        "Very strong correlation" if abs(r) >= 0.9 else
                        "Strong correlation" if abs(r) >= 0.7 else
                        "Moderate correlation" if abs(r) >= 0.5 else
                        "Weak correlation" if abs(r) >= 0.3 else
                        "Negligible correlation"
                    )
                }
            except Exception as e:
                logger.warning(f"Error calculating Pearson correlation: {e}")
        
        # Spearman correlation
        if "spearman_correlation" in tests:
            try:
                rho, p_value = stats.spearmanr(sim_values, hw_values)
                
                test_results["overall"]["spearman_correlation"] = {
                    "coefficient": float(rho),
                    "p_value": float(p_value),
                    "is_significant": p_value < significance_level,
                    "interpretation": (
                        "Very strong correlation" if abs(rho) >= 0.9 else
                        "Strong correlation" if abs(rho) >= 0.7 else
                        "Moderate correlation" if abs(rho) >= 0.5 else
                        "Weak correlation" if abs(rho) >= 0.3 else
                        "Negligible correlation"
                    )
                }
            except Exception as e:
                logger.warning(f"Error calculating Spearman correlation: {e}")
        
        # For tests on individual metrics, we'd need multiple data points for each metric
        # These are generally not available from a single SimulationResult and HardwareResult
        # These tests would be run in the validate_batch method when comparing groups of results
        
        return test_results
    
    def _calculate_advanced_metrics(
        self,
        simulation_result: SimulationResult,
        hardware_result: HardwareResult,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate advanced metrics for validation.
        
        Args:
            simulation_result: Simulation result to validate
            hardware_result: Hardware result to compare against
            metrics_comparison: Basic metrics comparison
            
        Returns:
            Dictionary with advanced metrics
        """
        if not self.config["advanced_metrics"]["enabled"]:
            return {}
        
        advanced_metrics = {}
        
        # Calculate overall MAPE
        mape_values = []
        for metric, comparison in metrics_comparison.items():
            if "mape" in comparison and not np.isnan(comparison["mape"]):
                mape_values.append(comparison["mape"])
        
        if mape_values:
            advanced_metrics["overall_mape"] = np.mean(mape_values)
            advanced_metrics["mape_std_dev"] = np.std(mape_values)
        
        # Calculate confidence score
        if "confidence_score" in self.config["advanced_metrics"]["metrics"]:
            # For a single validation, confidence is based on MAPE
            if "overall_mape" in advanced_metrics:
                mape = advanced_metrics["overall_mape"]
                thresholds = self.config["accuracy_thresholds"]["mape"]
                
                if mape <= thresholds["excellent"]:
                    confidence = 0.9  # Very high confidence
                elif mape <= thresholds["good"]:
                    confidence = 0.7  # High confidence
                elif mape <= thresholds["acceptable"]:
                    confidence = 0.5  # Moderate confidence
                elif mape <= thresholds["poor"]:
                    confidence = 0.3  # Low confidence
                else:
                    confidence = 0.1  # Very low confidence
                
                advanced_metrics["confidence_score"] = confidence
                
                # Add interpretation
                if confidence >= 0.8:
                    advanced_metrics["confidence_interpretation"] = "Very high confidence"
                elif confidence >= 0.6:
                    advanced_metrics["confidence_interpretation"] = "High confidence"
                elif confidence >= 0.4:
                    advanced_metrics["confidence_interpretation"] = "Moderate confidence"
                elif confidence >= 0.2:
                    advanced_metrics["confidence_interpretation"] = "Low confidence"
                else:
                    advanced_metrics["confidence_interpretation"] = "Very low confidence"
        
        # Calculate bias score
        if "bias_score" in self.config["advanced_metrics"]["metrics"]:
            # Bias is based on signed percent errors
            percent_errors = []
            for metric, comparison in metrics_comparison.items():
                if "percent_error" in comparison and not np.isnan(comparison["percent_error"]):
                    percent_errors.append(comparison["percent_error"])
            
            if percent_errors:
                mean_percent_error = np.mean(percent_errors)
                
                # Normalize to a 0-1 scale where 0 is perfect (no bias)
                normalized_bias = min(1.0, abs(mean_percent_error) / 100.0)
                
                # Convert to a 0-1 score where 1 is perfect (no bias)
                bias_score = 1.0 - normalized_bias
                
                advanced_metrics["bias_score"] = bias_score
                advanced_metrics["mean_percent_error"] = mean_percent_error
                
                # Add bias direction
                if mean_percent_error > 1.0:
                    advanced_metrics["bias_direction"] = "simulation overestimates"
                elif mean_percent_error < -1.0:
                    advanced_metrics["bias_direction"] = "simulation underestimates"
                else:
                    advanced_metrics["bias_direction"] = "minimal bias"
        
        # Calculate precision score
        if "precision_score" in self.config["advanced_metrics"]["metrics"]:
            # Precision is based on consistency of errors
            if "mape_std_dev" in advanced_metrics:
                std_dev = advanced_metrics["mape_std_dev"]
                
                # Normalize to a 0-1 scale where 0 is least precise
                normalized_precision = max(0.0, 1.0 - (std_dev / 50.0))  # 50% std dev or more = 0 precision
                
                advanced_metrics["precision_score"] = normalized_precision
                
                # Add interpretation
                if normalized_precision >= 0.8:
                    advanced_metrics["precision_interpretation"] = "Very high precision"
                elif normalized_precision >= 0.6:
                    advanced_metrics["precision_interpretation"] = "High precision"
                elif normalized_precision >= 0.4:
                    advanced_metrics["precision_interpretation"] = "Moderate precision"
                elif normalized_precision >= 0.2:
                    advanced_metrics["precision_interpretation"] = "Low precision"
                else:
                    advanced_metrics["precision_interpretation"] = "Very low precision"
        
        # Calculate reliability score
        if "reliability_score" in self.config["advanced_metrics"]["metrics"]:
            # Reliability is a combination of bias and precision
            if "bias_score" in advanced_metrics and "precision_score" in advanced_metrics:
                reliability = (advanced_metrics["bias_score"] * 0.5 + 
                              advanced_metrics["precision_score"] * 0.5)
                
                advanced_metrics["reliability_score"] = reliability
                
                # Add interpretation
                if reliability >= 0.8:
                    advanced_metrics["reliability_interpretation"] = "Very high reliability"
                elif reliability >= 0.6:
                    advanced_metrics["reliability_interpretation"] = "High reliability"
                elif reliability >= 0.4:
                    advanced_metrics["reliability_interpretation"] = "Moderate reliability"
                elif reliability >= 0.2:
                    advanced_metrics["reliability_interpretation"] = "Low reliability"
                else:
                    advanced_metrics["reliability_interpretation"] = "Very low reliability"
        
        # Calculate fidelity score
        if "fidelity_score" in self.config["advanced_metrics"]["metrics"]:
            # Fidelity is based on how well simulation preserves relationships
            # Use Pearson correlation from statistical tests if available
            if ("overall" in metrics_comparison and 
                "pearson_correlation" in metrics_comparison["overall"]):
                
                correlation = metrics_comparison["overall"]["pearson_correlation"]["coefficient"]
                
                # Normalize to a 0-1 scale where 1 is perfect fidelity
                fidelity = (correlation + 1.0) / 2.0
                
                advanced_metrics["fidelity_score"] = fidelity
                
                # Add interpretation
                if fidelity >= 0.9:
                    advanced_metrics["fidelity_interpretation"] = "Very high fidelity"
                elif fidelity >= 0.8:
                    advanced_metrics["fidelity_interpretation"] = "High fidelity"
                elif fidelity >= 0.7:
                    advanced_metrics["fidelity_interpretation"] = "Moderate fidelity"
                elif fidelity >= 0.6:
                    advanced_metrics["fidelity_interpretation"] = "Low fidelity"
                else:
                    advanced_metrics["fidelity_interpretation"] = "Very low fidelity"
            else:
                # If correlation isn't available, use alternative method
                # Calculate correlation between simulation and hardware metrics
                sim_values = []
                hw_values = []
                
                for metric in metrics_comparison.keys():
                    if metric != "overall" and metric in simulation_result.metrics and metric in hardware_result.metrics:
                        sim_value = simulation_result.metrics[metric]
                        hw_value = hardware_result.metrics[metric]
                        
                        if sim_value is not None and hw_value is not None:
                            sim_values.append(sim_value)
                            hw_values.append(hw_value)
                
                if len(sim_values) >= 2:  # Need at least 2 points for correlation
                    try:
                        correlation = np.corrcoef(sim_values, hw_values)[0, 1]
                        
                        # Normalize to a 0-1 scale where 1 is perfect fidelity
                        fidelity = (correlation + 1.0) / 2.0
                        
                        advanced_metrics["fidelity_score"] = fidelity
                        
                        # Add interpretation
                        if fidelity >= 0.9:
                            advanced_metrics["fidelity_interpretation"] = "Very high fidelity"
                        elif fidelity >= 0.8:
                            advanced_metrics["fidelity_interpretation"] = "High fidelity"
                        elif fidelity >= 0.7:
                            advanced_metrics["fidelity_interpretation"] = "Moderate fidelity"
                        elif fidelity >= 0.6:
                            advanced_metrics["fidelity_interpretation"] = "Low fidelity"
                        else:
                            advanced_metrics["fidelity_interpretation"] = "Very low fidelity"
                    except Exception as e:
                        logger.warning(f"Error calculating correlation for fidelity score: {e}")
        
        return advanced_metrics
    
    def _group_results(
        self,
        results: List[Union[SimulationResult, HardwareResult]]
    ) -> Dict[str, List[Union[SimulationResult, HardwareResult]]]:
        """
        Group results by model_id, hardware_id, batch_size, precision.
        
        Args:
            results: List of results to group
            
        Returns:
            Dictionary mapping group keys to lists of results
        """
        grouped_results = {}
        
        for result in results:
            group_key = f"{result.model_id}_{result.hardware_id}_{result.batch_size}_{result.precision}"
            
            if group_key not in grouped_results:
                grouped_results[group_key] = []
            
            grouped_results[group_key].append(result)
        
        return grouped_results
    
    def _validate_result_groups(
        self,
        simulation_group: List[SimulationResult],
        hardware_group: List[HardwareResult]
    ) -> Optional[ValidationResult]:
        """
        Validate groups of simulation and hardware results.
        
        Args:
            simulation_group: Group of simulation results
            hardware_group: Group of hardware results
            
        Returns:
            ValidationResult for the group comparison, or None if validation fails
        """
        if not simulation_group or not hardware_group:
            return None
        
        # We'll use the first result in each group for the result containers
        sim_result = simulation_group[0]
        hw_result = hardware_group[0]
        
        # Collect metrics for statistical analysis
        metrics_data = {}
        for metric in self.config["metrics_to_validate"]:
            sim_values = []
            hw_values = []
            
            for result in simulation_group:
                if metric in result.metrics and result.metrics[metric] is not None:
                    sim_values.append(result.metrics[metric])
            
            for result in hardware_group:
                if metric in result.metrics and result.metrics[metric] is not None:
                    hw_values.append(result.metrics[metric])
            
            if sim_values and hw_values:
                metrics_data[metric] = {
                    "simulation": sim_values,
                    "hardware": hw_values
                }
        
        if not metrics_data:
            return None
        
        # Calculate metrics comparison
        metrics_comparison = {}
        for metric, data in metrics_data.items():
            sim_values = data["simulation"]
            hw_values = data["hardware"]
            
            # Use minimum length to match pairs
            min_len = min(len(sim_values), len(hw_values))
            
            if min_len == 0:
                continue
            
            metrics_comparison[metric] = {}
            
            # Calculate basic error metrics for paired data
            absolute_errors = []
            relative_errors = []
            mape_values = []
            percent_errors = []
            
            for i in range(min_len):
                sim_value = sim_values[i]
                hw_value = hw_values[i]
                
                absolute_error = abs(sim_value - hw_value)
                absolute_errors.append(absolute_error)
                
                if hw_value != 0:
                    relative_error = absolute_error / abs(hw_value)
                    relative_errors.append(relative_error)
                    
                    mape = relative_error * 100.0
                    mape_values.append(mape)
                    
                    percent_error = (sim_value - hw_value) / abs(hw_value) * 100.0
                    percent_errors.append(percent_error)
            
            # Calculate aggregate metrics
            if "absolute_error" in self.config["error_metrics"] and absolute_errors:
                metrics_comparison[metric]["absolute_error"] = np.mean(absolute_errors)
            
            if "relative_error" in self.config["error_metrics"] and relative_errors:
                metrics_comparison[metric]["relative_error"] = np.mean(relative_errors)
            
            if "mape" in self.config["error_metrics"] and mape_values:
                metrics_comparison[metric]["mape"] = np.mean(mape_values)
            
            if "percent_error" in self.config["error_metrics"] and percent_errors:
                metrics_comparison[metric]["percent_error"] = np.mean(percent_errors)
            
            if "rmse" in self.config["error_metrics"] and absolute_errors:
                metrics_comparison[metric]["rmse"] = np.sqrt(np.mean(np.array(absolute_errors) ** 2))
            
            if "normalized_rmse" in self.config["error_metrics"] and absolute_errors and hw_values:
                # Normalize by range of hardware values
                hw_range = max(hw_values) - min(hw_values)
                if hw_range > 0:
                    metrics_comparison[metric]["normalized_rmse"] = (
                        np.sqrt(np.mean(np.array(absolute_errors) ** 2)) / hw_range
                    )
            
            # Run statistical tests if enabled
            if self.config["statistical_tests"]["enabled"]:
                self._add_group_statistical_tests(
                    metrics_comparison[metric],
                    sim_values,
                    hw_values
                )
        
        # Generate additional metrics
        additional_metrics = self._calculate_group_advanced_metrics(
            simulation_group,
            hardware_group,
            metrics_comparison
        )
        
        # Create validation result
        validation_result = ValidationResult(
            simulation_result=sim_result,
            hardware_result=hw_result,
            metrics_comparison=metrics_comparison,
            validation_version=f"{self.config['validation_version']}_group",
            additional_metrics=additional_metrics
        )
        
        return validation_result
    
    def _add_group_statistical_tests(
        self,
        metrics_comparison: Dict[str, float],
        sim_values: List[float],
        hw_values: List[float]
    ) -> None:
        """
        Add statistical test results to metrics comparison for grouped data.
        
        Args:
            metrics_comparison: Dictionary to add test results to
            sim_values: List of simulation values
            hw_values: List of hardware values
        """
        # Skip if we don't have scipy for statistical tests
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available for statistical tests")
            return
        
        # Ensure we have enough data points
        if len(sim_values) < 2 or len(hw_values) < 2:
            return
        
        tests = self.config["statistical_tests"]["tests"]
        significance_level = self.config["statistical_tests"]["significance_level"]
        
        # Trim arrays to the same length
        min_len = min(len(sim_values), len(hw_values))
        sim_values = sim_values[:min_len]
        hw_values = hw_values[:min_len]
        
        # T-test for comparing means
        if "t_test" in tests:
            try:
                t_stat, p_value = stats.ttest_ind(sim_values, hw_values, equal_var=False)
                
                metrics_comparison["t_test"] = {
                    "statistic": float(t_stat),
                    "p_value": float(p_value),
                    "is_significant": p_value < significance_level,
                    "interpretation": (
                        "Significant difference between means" 
                        if p_value < significance_level 
                        else "No significant difference between means"
                    )
                }
            except Exception as e:
                logger.warning(f"Error calculating t-test: {e}")
        
        # Mann-Whitney U test for comparing distributions
        if "mann_whitney" in tests:
            try:
                u_stat, p_value = stats.mannwhitneyu(sim_values, hw_values)
                
                metrics_comparison["mann_whitney"] = {
                    "statistic": float(u_stat),
                    "p_value": float(p_value),
                    "is_significant": p_value < significance_level,
                    "interpretation": (
                        "Significant difference between distributions" 
                        if p_value < significance_level 
                        else "No significant difference between distributions"
                    )
                }
            except Exception as e:
                logger.warning(f"Error calculating Mann-Whitney U test: {e}")
        
        # Kolmogorov-Smirnov test for comparing distributions
        if "ks_test" in tests:
            try:
                ks_stat, p_value = stats.ks_2samp(sim_values, hw_values)
                
                metrics_comparison["ks_test"] = {
                    "statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "is_significant": p_value < significance_level,
                    "interpretation": (
                        "Significant difference between distributions" 
                        if p_value < significance_level 
                        else "No significant difference between distributions"
                    )
                }
            except Exception as e:
                logger.warning(f"Error calculating Kolmogorov-Smirnov test: {e}")
        
        # Pearson correlation
        if "pearson_correlation" in tests:
            try:
                r, p_value = stats.pearsonr(sim_values, hw_values)
                
                metrics_comparison["pearson_correlation"] = {
                    "coefficient": float(r),
                    "p_value": float(p_value),
                    "is_significant": p_value < significance_level,
                    "interpretation": (
                        "Very strong correlation" if abs(r) >= 0.9 else
                        "Strong correlation" if abs(r) >= 0.7 else
                        "Moderate correlation" if abs(r) >= 0.5 else
                        "Weak correlation" if abs(r) >= 0.3 else
                        "Negligible correlation"
                    )
                }
            except Exception as e:
                logger.warning(f"Error calculating Pearson correlation: {e}")
        
        # Spearman rank correlation
        if "spearman_correlation" in tests:
            try:
                rho, p_value = stats.spearmanr(sim_values, hw_values)
                
                metrics_comparison["spearman_correlation"] = {
                    "coefficient": float(rho),
                    "p_value": float(p_value),
                    "is_significant": p_value < significance_level,
                    "interpretation": (
                        "Very strong correlation" if abs(rho) >= 0.9 else
                        "Strong correlation" if abs(rho) >= 0.7 else
                        "Moderate correlation" if abs(rho) >= 0.5 else
                        "Weak correlation" if abs(rho) >= 0.3 else
                        "Negligible correlation"
                    )
                }
            except Exception as e:
                logger.warning(f"Error calculating Spearman correlation: {e}")
    
    def _calculate_group_advanced_metrics(
        self,
        simulation_group: List[SimulationResult],
        hardware_group: List[HardwareResult],
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate advanced metrics for grouped validation.
        
        Args:
            simulation_group: Group of simulation results
            hardware_group: Group of hardware results
            metrics_comparison: Basic metrics comparison
            
        Returns:
            Dictionary with advanced metrics
        """
        if not self.config["advanced_metrics"]["enabled"]:
            return {}
        
        advanced_metrics = {}
        
        # Calculate overall MAPE
        mape_values = []
        for metric, comparison in metrics_comparison.items():
            if "mape" in comparison and not np.isnan(comparison["mape"]):
                mape_values.append(comparison["mape"])
        
        if mape_values:
            advanced_metrics["overall_mape"] = np.mean(mape_values)
            advanced_metrics["mape_std_dev"] = np.std(mape_values)
        
        # Add more advanced metrics specific to grouped validation
        
        # Sample size metrics
        advanced_metrics["sample_size"] = {
            "simulation": len(simulation_group),
            "hardware": len(hardware_group),
            "matched_pairs": min(len(simulation_group), len(hardware_group))
        }
        
        # Calculate distribution metrics for each performance metric
        distribution_metrics = {}
        for metric in self.config["metrics_to_validate"]:
            sim_values = []
            hw_values = []
            
            for result in simulation_group:
                if metric in result.metrics and result.metrics[metric] is not None:
                    sim_values.append(result.metrics[metric])
            
            for result in hardware_group:
                if metric in result.metrics and result.metrics[metric] is not None:
                    hw_values.append(result.metrics[metric])
            
            if sim_values and hw_values:
                distribution_metrics[metric] = self._calculate_distribution_metrics(sim_values, hw_values)
        
        if distribution_metrics:
            advanced_metrics["distribution_metrics"] = distribution_metrics
        
        # Calculate confidence score
        if "confidence_score" in self.config["advanced_metrics"]["metrics"]:
            # For grouped validation, confidence depends on MAPE, sample size, and statistical tests
            confidence_components = {}
            
            # MAPE component
            if "overall_mape" in advanced_metrics:
                mape = advanced_metrics["overall_mape"]
                thresholds = self.config["accuracy_thresholds"]["mape"]
                
                if mape <= thresholds["excellent"]:
                    confidence_components["mape"] = 1.0
                elif mape <= thresholds["good"]:
                    confidence_components["mape"] = 0.8
                elif mape <= thresholds["acceptable"]:
                    confidence_components["mape"] = 0.6
                elif mape <= thresholds["poor"]:
                    confidence_components["mape"] = 0.3
                else:
                    confidence_components["mape"] = 0.1
            
            # Sample size component
            sample_size = advanced_metrics["sample_size"]["matched_pairs"]
            confidence_components["sample_size"] = min(1.0, sample_size / 10.0)  # 10+ samples for full confidence
            
            # Statistical test component
            stat_test_scores = []
            for metric, comparison in metrics_comparison.items():
                # Check for t-test significance
                if "t_test" in comparison and "is_significant" in comparison["t_test"]:
                    if not comparison["t_test"]["is_significant"]:
                        # Not significant means the means are similar (good)
                        stat_test_scores.append(1.0)
                    else:
                        # Significant means the means are different (bad)
                        stat_test_scores.append(0.0)
                
                # Check for distribution test significance
                if "ks_test" in comparison and "is_significant" in comparison["ks_test"]:
                    if not comparison["ks_test"]["is_significant"]:
                        # Not significant means the distributions are similar (good)
                        stat_test_scores.append(1.0)
                    else:
                        # Significant means the distributions are different (bad)
                        stat_test_scores.append(0.0)
                
                # Check for correlation
                if "pearson_correlation" in comparison and "coefficient" in comparison["pearson_correlation"]:
                    r = comparison["pearson_correlation"]["coefficient"]
                    # Normalize correlation to 0-1 scale
                    correlation_score = (r + 1.0) / 2.0
                    stat_test_scores.append(correlation_score)
            
            if stat_test_scores:
                confidence_components["statistical_tests"] = np.mean(stat_test_scores)
            
            # Combine components with weights
            if confidence_components:
                weights = {
                    "mape": 0.4,
                    "sample_size": 0.3,
                    "statistical_tests": 0.3
                }
                
                # Calculate weighted average of available components
                total_weight = 0.0
                weighted_sum = 0.0
                
                for component, value in confidence_components.items():
                    if component in weights:
                        weighted_sum += value * weights[component]
                        total_weight += weights[component]
                
                if total_weight > 0:
                    confidence = weighted_sum / total_weight
                    
                    advanced_metrics["confidence_score"] = confidence
                    advanced_metrics["confidence_components"] = confidence_components
                    
                    # Add interpretation
                    if confidence >= 0.8:
                        advanced_metrics["confidence_interpretation"] = "Very high confidence"
                    elif confidence >= 0.6:
                        advanced_metrics["confidence_interpretation"] = "High confidence"
                    elif confidence >= 0.4:
                        advanced_metrics["confidence_interpretation"] = "Moderate confidence"
                    elif confidence >= 0.2:
                        advanced_metrics["confidence_interpretation"] = "Low confidence"
                    else:
                        advanced_metrics["confidence_interpretation"] = "Very low confidence"
        
        return advanced_metrics
    
    def _calculate_distribution_metrics(
        self,
        simulation_values: List[float],
        hardware_values: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate metrics comparing the distributions of simulation and hardware values.
        
        Args:
            simulation_values: List of simulation values
            hardware_values: List of hardware values
            
        Returns:
            Dictionary with distribution metrics
        """
        metrics = {}
        
        # Basic statistics
        sim_array = np.array(simulation_values)
        hw_array = np.array(hardware_values)
        
        metrics["simulation"] = {
            "mean": float(np.mean(sim_array)),
            "median": float(np.median(sim_array)),
            "std_dev": float(np.std(sim_array)),
            "min": float(np.min(sim_array)),
            "max": float(np.max(sim_array))
        }
        
        metrics["hardware"] = {
            "mean": float(np.mean(hw_array)),
            "median": float(np.median(hw_array)),
            "std_dev": float(np.std(hw_array)),
            "min": float(np.min(hw_array)),
            "max": float(np.max(hw_array))
        }
        
        # Calculate mean differences
        metrics["mean_diff"] = {
            "absolute": float(abs(metrics["simulation"]["mean"] - metrics["hardware"]["mean"])),
            "percent": float(
                abs(metrics["simulation"]["mean"] - metrics["hardware"]["mean"]) / 
                abs(metrics["hardware"]["mean"]) * 100.0
                if metrics["hardware"]["mean"] != 0 else float('nan')
            )
        }
        
        # Calculate median differences
        metrics["median_diff"] = {
            "absolute": float(abs(metrics["simulation"]["median"] - metrics["hardware"]["median"])),
            "percent": float(
                abs(metrics["simulation"]["median"] - metrics["hardware"]["median"]) / 
                abs(metrics["hardware"]["median"]) * 100.0
                if metrics["hardware"]["median"] != 0 else float('nan')
            )
        }
        
        # Calculate std dev differences
        metrics["std_dev_diff"] = {
            "absolute": float(abs(metrics["simulation"]["std_dev"] - metrics["hardware"]["std_dev"])),
            "percent": float(
                abs(metrics["simulation"]["std_dev"] - metrics["hardware"]["std_dev"]) / 
                abs(metrics["hardware"]["std_dev"]) * 100.0
                if metrics["hardware"]["std_dev"] != 0 else float('nan')
            )
        }
        
        # Calculate range differences
        sim_range = metrics["simulation"]["max"] - metrics["simulation"]["min"]
        hw_range = metrics["hardware"]["max"] - metrics["hardware"]["min"]
        
        metrics["range_diff"] = {
            "absolute": float(abs(sim_range - hw_range)),
            "percent": float(
                abs(sim_range - hw_range) / abs(hw_range) * 100.0
                if hw_range != 0 else float('nan')
            )
        }
        
        # Try to run statistical tests if scipy is available
        try:
            from scipy import stats
            
            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(sim_array, hw_array)
            metrics["ks_test"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_pvalue),
                "is_significant": ks_pvalue < 0.05
            }
            
            # Calculate KL divergence if possible
            # This requires binning the data
            try:
                # Create histograms for KL divergence calculation
                bin_count = 10  # Number of bins
                min_val = min(np.min(sim_array), np.min(hw_array))
                max_val = max(np.max(sim_array), np.max(hw_array))
                
                bins = np.linspace(min_val, max_val, bin_count + 1)
                sim_hist, _ = np.histogram(sim_array, bins=bins, density=True)
                hw_hist, _ = np.histogram(hw_array, bins=bins, density=True)
                
                # Add small constant to avoid division by zero
                sim_hist = sim_hist + 1e-10
                hw_hist = hw_hist + 1e-10
                
                # Normalize histograms
                sim_hist = sim_hist / np.sum(sim_hist)
                hw_hist = hw_hist / np.sum(hw_hist)
                
                # Calculate KL divergence in both directions
                kl_sim_to_hw = float(stats.entropy(sim_hist, hw_hist))
                kl_hw_to_sim = float(stats.entropy(hw_hist, sim_hist))
                
                # Symmetric KL divergence
                symmetric_kl = float((kl_sim_to_hw + kl_hw_to_sim) / 2)
                
                metrics["kl_divergence"] = {
                    "sim_to_hw": kl_sim_to_hw,
                    "hw_to_sim": kl_hw_to_sim,
                    "symmetric": symmetric_kl
                }
            except Exception as e:
                logger.warning(f"Error calculating KL divergence: {e}")
        
        except ImportError:
            logger.warning("scipy not available for statistical tests")
        
        return metrics


def get_statistical_validator_instance(config_path: Optional[str] = None) -> StatisticalValidator:
    """
    Get an instance of the StatisticalValidator class.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        StatisticalValidator instance
    """
    # Load configuration from file if provided
    config = None
    if config_path:
        import json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading configuration from {config_path}: {e}")
    
    return StatisticalValidator(config)