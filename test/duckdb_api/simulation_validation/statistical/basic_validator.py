#!/usr/bin/env python3
"""
Basic Statistical Validator implementation for the Simulation Accuracy and Validation Framework.

This module provides a concrete implementation of the SimulationValidator interface
with standard statistical methods for validating simulation accuracy.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("basic_statistical_validator")

# Add parent directories to path for module imports
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import base classes
from core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult,
    SimulationValidator
)


class BasicStatisticalValidator(SimulationValidator):
    """Basic implementation of a simulation validator using standard statistical methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the basic statistical validator.
        
        Args:
            config: Configuration options for the validator
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "metrics_to_validate": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w"
            ],
            "validation_methods": [
                "absolute_error",
                "relative_error",
                "mape",
                "percent_error"
            ],
            "acceptable_mape_threshold": 10.0,  # MAPE threshold for acceptable simulation (%)
            "validation_version": "basic_v1.0"
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
        metrics_comparison = self._calculate_metrics_comparison(
            simulation_result.metrics,
            hardware_result.metrics
        )
        
        # Generate additional metrics
        additional_metrics = self._generate_additional_metrics(
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
        
        # Match simulation and hardware results by model_id, hardware_id, batch_size, precision
        for sim_result in simulation_results:
            matching_hw_results = [
                hw_result for hw_result in hardware_results
                if hw_result.model_id == sim_result.model_id
                and hw_result.hardware_id == sim_result.hardware_id
                and hw_result.batch_size == sim_result.batch_size
                and hw_result.precision == sim_result.precision
            ]
            
            if matching_hw_results:
                # Use the first matching hardware result
                hw_result = matching_hw_results[0]
                validation_result = self.validate(sim_result, hw_result)
                validation_results.append(validation_result)
            else:
                logger.warning(f"No matching hardware result found for simulation: {sim_result.model_id}, {sim_result.hardware_id}")
        
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
            "overall": {}
        }
        
        # Extract all metrics being compared
        all_metrics = set()
        for val_result in validation_results:
            all_metrics.update(val_result.metrics_comparison.keys())
        
        # Calculate summary statistics for each metric
        for metric in all_metrics:
            metric_values = []
            
            for val_result in validation_results:
                if metric in val_result.metrics_comparison:
                    if "mape" in val_result.metrics_comparison[metric]:
                        metric_values.append(val_result.metrics_comparison[metric]["mape"])
            
            if metric_values:
                summary["metrics"][metric] = {
                    "mean_mape": np.mean(metric_values),
                    "median_mape": np.median(metric_values),
                    "min_mape": np.min(metric_values),
                    "max_mape": np.max(metric_values),
                    "std_dev_mape": np.std(metric_values),
                    "count": len(metric_values)
                }
        
        # Calculate summary statistics by model
        model_results = {}
        for val_result in validation_results:
            model_id = val_result.simulation_result.model_id
            
            if model_id not in model_results:
                model_results[model_id] = []
            
            # Get overall accuracy score if available
            if hasattr(val_result, "additional_metrics") and val_result.additional_metrics:
                if "overall_accuracy_score" in val_result.additional_metrics:
                    model_results[model_id].append(val_result.additional_metrics["overall_accuracy_score"])
            
            # Otherwise calculate from MAPE values
            else:
                mape_values = []
                for metric, comparison in val_result.metrics_comparison.items():
                    if "mape" in comparison:
                        mape_values.append(comparison["mape"])
                
                if mape_values:
                    model_results[model_id].append(np.mean(mape_values))
        
        # Calculate summary for each model
        for model_id, accuracy_scores in model_results.items():
            if accuracy_scores:
                summary["models"][model_id] = {
                    "mean_accuracy_score": np.mean(accuracy_scores),
                    "median_accuracy_score": np.median(accuracy_scores),
                    "min_accuracy_score": np.min(accuracy_scores),
                    "max_accuracy_score": np.max(accuracy_scores),
                    "std_dev_accuracy_score": np.std(accuracy_scores),
                    "count": len(accuracy_scores)
                }
        
        # Calculate summary statistics by hardware
        hardware_results = {}
        for val_result in validation_results:
            hardware_id = val_result.simulation_result.hardware_id
            
            if hardware_id not in hardware_results:
                hardware_results[hardware_id] = []
            
            # Get overall accuracy score if available
            if hasattr(val_result, "additional_metrics") and val_result.additional_metrics:
                if "overall_accuracy_score" in val_result.additional_metrics:
                    hardware_results[hardware_id].append(val_result.additional_metrics["overall_accuracy_score"])
            
            # Otherwise calculate from MAPE values
            else:
                mape_values = []
                for metric, comparison in val_result.metrics_comparison.items():
                    if "mape" in comparison:
                        mape_values.append(comparison["mape"])
                
                if mape_values:
                    hardware_results[hardware_id].append(np.mean(mape_values))
        
        # Calculate summary for each hardware
        for hardware_id, accuracy_scores in hardware_results.items():
            if accuracy_scores:
                summary["hardware"][hardware_id] = {
                    "mean_accuracy_score": np.mean(accuracy_scores),
                    "median_accuracy_score": np.median(accuracy_scores),
                    "min_accuracy_score": np.min(accuracy_scores),
                    "max_accuracy_score": np.max(accuracy_scores),
                    "std_dev_accuracy_score": np.std(accuracy_scores),
                    "count": len(accuracy_scores)
                }
        
        # Calculate overall statistics
        all_accuracy_scores = []
        for val_result in validation_results:
            # Get overall accuracy score if available
            if hasattr(val_result, "additional_metrics") and val_result.additional_metrics:
                if "overall_accuracy_score" in val_result.additional_metrics:
                    all_accuracy_scores.append(val_result.additional_metrics["overall_accuracy_score"])
            
            # Otherwise calculate from MAPE values
            else:
                mape_values = []
                for metric, comparison in val_result.metrics_comparison.items():
                    if "mape" in comparison:
                        mape_values.append(comparison["mape"])
                
                if mape_values:
                    all_accuracy_scores.append(np.mean(mape_values))
        
        if all_accuracy_scores:
            summary["overall"] = {
                "mean_accuracy_score": np.mean(all_accuracy_scores),
                "median_accuracy_score": np.median(all_accuracy_scores),
                "min_accuracy_score": np.min(all_accuracy_scores),
                "max_accuracy_score": np.max(all_accuracy_scores),
                "std_dev_accuracy_score": np.std(all_accuracy_scores),
                "count": len(all_accuracy_scores)
            }
        
        # Evaluate overall status based on mean accuracy score
        if "mean_accuracy_score" in summary["overall"]:
            mean_score = summary["overall"]["mean_accuracy_score"]
            
            if mean_score <= 5.0:
                summary["overall"]["status"] = "excellent"
            elif mean_score <= 10.0:
                summary["overall"]["status"] = "good"
            elif mean_score <= 15.0:
                summary["overall"]["status"] = "acceptable"
            elif mean_score <= 25.0:
                summary["overall"]["status"] = "poor"
            else:
                summary["overall"]["status"] = "unacceptable"
        
        return summary
    
    def _calculate_metrics_comparison(
        self,
        simulation_metrics: Dict[str, float],
        hardware_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comparison metrics between simulation and hardware results.
        
        Args:
            simulation_metrics: Metrics from simulation result
            hardware_metrics: Metrics from hardware result
            
        Returns:
            Dictionary mapping metric names to dictionaries of comparison metrics
        """
        metrics_comparison = {}
        
        # Determine which metrics to compare
        metrics_to_compare = self.config["metrics_to_validate"]
        
        for metric_name in metrics_to_compare:
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
            
            # Calculate comparison metrics
            validation_methods = self.config["validation_methods"]
            
            if "absolute_error" in validation_methods:
                metrics_comparison[metric_name]["absolute_error"] = abs(sim_value - hw_value)
            
            if "relative_error" in validation_methods:
                if hw_value != 0:
                    metrics_comparison[metric_name]["relative_error"] = abs(sim_value - hw_value) / abs(hw_value)
                else:
                    # Handle division by zero
                    metrics_comparison[metric_name]["relative_error"] = float('nan')
            
            if "mape" in validation_methods:
                if hw_value != 0:
                    metrics_comparison[metric_name]["mape"] = abs(sim_value - hw_value) / abs(hw_value) * 100.0
                else:
                    # Handle division by zero
                    metrics_comparison[metric_name]["mape"] = float('nan')
            
            if "percent_error" in validation_methods:
                if hw_value != 0:
                    metrics_comparison[metric_name]["percent_error"] = (sim_value - hw_value) / abs(hw_value) * 100.0
                else:
                    # Handle division by zero
                    metrics_comparison[metric_name]["percent_error"] = float('nan')
        
        return metrics_comparison
    
    def _generate_additional_metrics(
        self,
        simulation_result: SimulationResult,
        hardware_result: HardwareResult,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Generate additional validation metrics beyond the basic comparisons.
        
        Args:
            simulation_result: The simulation result
            hardware_result: The hardware result
            metrics_comparison: Basic metrics comparison
            
        Returns:
            Dictionary of additional metrics
        """
        additional_metrics = {}
        
        # Calculate overall accuracy score (average MAPE across all metrics)
        mape_values = []
        for metric, comparison in metrics_comparison.items():
            if "mape" in comparison and not np.isnan(comparison["mape"]):
                mape_values.append(comparison["mape"])
        
        if mape_values:
            additional_metrics["overall_accuracy_score"] = np.mean(mape_values)
            additional_metrics["overall_accuracy_std_dev"] = np.std(mape_values)
        
        # Determine if simulation is acceptable based on MAPE threshold
        if "overall_accuracy_score" in additional_metrics:
            threshold = self.config["acceptable_mape_threshold"]
            additional_metrics["is_acceptable"] = additional_metrics["overall_accuracy_score"] <= threshold
        
        # Calculate prediction bias (average of percent errors)
        percent_errors = []
        for metric, comparison in metrics_comparison.items():
            if "percent_error" in comparison and not np.isnan(comparison["percent_error"]):
                percent_errors.append(comparison["percent_error"])
        
        if percent_errors:
            additional_metrics["prediction_bias"] = np.mean(percent_errors)
            additional_metrics["prediction_bias_std_dev"] = np.std(percent_errors)
        
        # Calculate correlation between simulation and hardware metrics
        sim_values = []
        hw_values = []
        
        for metric in metrics_comparison.keys():
            if metric in simulation_result.metrics and metric in hardware_result.metrics:
                sim_value = simulation_result.metrics[metric]
                hw_value = hardware_result.metrics[metric]
                
                if sim_value is not None and hw_value is not None:
                    sim_values.append(sim_value)
                    hw_values.append(hw_value)
        
        if len(sim_values) >= 2:  # Need at least 2 points for correlation
            try:
                correlation_coefficient = np.corrcoef(sim_values, hw_values)[0, 1]
                additional_metrics["correlation_coefficient"] = correlation_coefficient
            except Exception as e:
                logger.warning(f"Error calculating correlation coefficient: {e}")
        
        return additional_metrics