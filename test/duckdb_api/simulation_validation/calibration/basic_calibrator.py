#!/usr/bin/env python3
"""
Basic Calibrator implementation for the Simulation Accuracy and Validation Framework.

This module provides a concrete implementation of the SimulationCalibrator interface
that uses simple regression techniques to calibrate simulation parameters.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("basic_calibrator")

# Add parent directories to path for module imports
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import base classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult,
    SimulationCalibrator
)


class BasicSimulationCalibrator(SimulationCalibrator):
    """
    Basic implementation of a simulation calibrator using regression techniques.
    
    This calibrator uses linear regression to adjust simulation parameters
    based on validation results. It learns correction factors for different
    metrics to improve simulation accuracy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the basic simulation calibrator.
        
        Args:
            config: Configuration options for the calibrator
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "metrics_to_calibrate": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w"
            ],
            "calibration_method": "linear_scaling",  # linear_scaling, additive_adjustment, or regression
            "min_samples_per_hardware": 3,           # Minimum number of samples needed for calibration
            "min_samples_per_model": 3,              # Minimum number of samples needed for calibration
            "calibration_version": "basic_v1.0",
            "learning_rate": 0.5                    # Learning rate for parameter updates (0-1)
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def calibrate(
        self,
        validation_results: List[ValidationResult],
        simulation_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calibrate simulation parameters based on validation results.
        
        Args:
            validation_results: List of validation results
            simulation_parameters: Current simulation parameters
            
        Returns:
            Updated simulation parameters
        """
        if not validation_results:
            logger.warning("No validation results provided for calibration")
            return simulation_parameters
        
        # Group validation results by hardware type and model type
        validation_by_hardware_model = self._group_validation_results(validation_results)
        
        # Clone the simulation parameters for updates
        updated_parameters = simulation_parameters.copy()
        
        # Create correction factors structure if it doesn't exist
        if "correction_factors" not in updated_parameters:
            updated_parameters["correction_factors"] = {}
        
        # Perform calibration for each hardware-model combination
        for (hardware_id, model_id), results in validation_by_hardware_model.items():
            if len(results) < self.config["min_samples_per_hardware"]:
                logger.info(f"Skipping calibration for {hardware_id}/{model_id}: insufficient samples ({len(results)})")
                continue
            
            # Calculate correction factors
            correction_factors = self._calculate_correction_factors(results)
            
            # Create nested dictionaries if they don't exist
            if hardware_id not in updated_parameters["correction_factors"]:
                updated_parameters["correction_factors"][hardware_id] = {}
            
            if model_id not in updated_parameters["correction_factors"][hardware_id]:
                updated_parameters["correction_factors"][hardware_id][model_id] = {}
            
            # Update correction factors with learning rate
            for metric, factor in correction_factors.items():
                current_factor = updated_parameters["correction_factors"][hardware_id][model_id].get(metric, 1.0)
                
                # Apply learning rate to smooth updates
                lr = self.config["learning_rate"]
                updated_factor = current_factor * (1 - lr) + factor * lr
                
                updated_parameters["correction_factors"][hardware_id][model_id][metric] = updated_factor
                
                logger.info(f"Updated correction factor for {hardware_id}/{model_id}/{metric}: {current_factor:.4f} -> {updated_factor:.4f}")
        
        # Update calibration metadata
        updated_parameters["calibration_version"] = self.config["calibration_version"]
        updated_parameters["calibration_method"] = self.config["calibration_method"]
        updated_parameters["num_samples_used"] = len(validation_results)
        
        return updated_parameters
    
    def evaluate_calibration(
        self,
        before_calibration: List[ValidationResult],
        after_calibration: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of a calibration.
        
        Args:
            before_calibration: Validation results before calibration
            after_calibration: Validation results after calibration
            
        Returns:
            Metrics quantifying the calibration improvement
        """
        if not before_calibration or not after_calibration:
            return {"status": "error", "message": "Missing validation results"}
        
        # Get metrics
        metrics_to_evaluate = self.config["metrics_to_calibrate"]
        
        # Calculate statistics before calibration
        before_stats = self._calculate_error_statistics(before_calibration, metrics_to_evaluate)
        
        # Calculate statistics after calibration
        after_stats = self._calculate_error_statistics(after_calibration, metrics_to_evaluate)
        
        # Calculate improvement for each metric
        improvement = {}
        for metric in metrics_to_evaluate:
            if metric in before_stats and metric in after_stats:
                before_mape = before_stats[metric]["mean_mape"]
                after_mape = after_stats[metric]["mean_mape"]
                
                if before_mape > 0:
                    relative_improvement = (before_mape - after_mape) / before_mape * 100.0
                else:
                    relative_improvement = 0.0
                
                absolute_improvement = before_mape - after_mape
                
                improvement[metric] = {
                    "before_mape": before_mape,
                    "after_mape": after_mape,
                    "absolute_improvement": absolute_improvement,
                    "relative_improvement": relative_improvement
                }
        
        # Calculate overall improvement
        before_overall = np.mean([stats["mean_mape"] for stats in before_stats.values()])
        after_overall = np.mean([stats["mean_mape"] for stats in after_stats.values()])
        
        if before_overall > 0:
            overall_relative_improvement = (before_overall - after_overall) / before_overall * 100.0
        else:
            overall_relative_improvement = 0.0
        
        overall_absolute_improvement = before_overall - after_overall
        
        # Prepare evaluation results
        evaluation = {
            "metrics": improvement,
            "overall": {
                "before_mape": before_overall,
                "after_mape": after_overall,
                "absolute_improvement": overall_absolute_improvement,
                "relative_improvement": overall_relative_improvement
            },
            "num_samples": len(before_calibration),
            "calibration_method": self.config["calibration_method"],
            "calibration_version": self.config["calibration_version"]
        }
        
        return evaluation
    
    def apply_calibration(
        self,
        simulation_result: SimulationResult,
        simulation_parameters: Dict[str, Any]
    ) -> SimulationResult:
        """
        Apply calibration to a simulation result.
        
        Args:
            simulation_result: The simulation result to calibrate
            simulation_parameters: The simulation parameters with correction factors
            
        Returns:
            Calibrated simulation result
        """
        # Check if correction factors exist
        if "correction_factors" not in simulation_parameters:
            logger.warning("No correction factors found in simulation parameters")
            return simulation_result
        
        # Get hardware and model IDs
        hardware_id = simulation_result.hardware_id
        model_id = simulation_result.model_id
        
        # Check if correction factors exist for this hardware-model combination
        correction_factors = simulation_parameters["correction_factors"]
        if hardware_id not in correction_factors or model_id not in correction_factors[hardware_id]:
            logger.info(f"No correction factors for {hardware_id}/{model_id}")
            
            # Try to use generic correction factors
            if "generic" in correction_factors and model_id in correction_factors["generic"]:
                logger.info(f"Using generic hardware correction factors for {model_id}")
                hardware_id = "generic"
            elif hardware_id in correction_factors and "generic" in correction_factors[hardware_id]:
                logger.info(f"Using generic model correction factors for {hardware_id}")
                model_id = "generic"
            elif "generic" in correction_factors and "generic" in correction_factors["generic"]:
                logger.info("Using fully generic correction factors")
                hardware_id = "generic"
                model_id = "generic"
            else:
                logger.warning("No applicable correction factors found")
                return simulation_result
        
        # Clone the simulation result for calibration
        calibrated_metrics = simulation_result.metrics.copy()
        
        # Get the calibration method
        calibration_method = simulation_parameters.get("calibration_method", self.config["calibration_method"])
        
        # Apply correction factors to metrics
        for metric in self.config["metrics_to_calibrate"]:
            if metric not in calibrated_metrics:
                continue
            
            if metric not in correction_factors[hardware_id][model_id]:
                continue
            
            # Get the correction factor
            factor = correction_factors[hardware_id][model_id][metric]
            
            # Apply calibration based on method
            if calibration_method == "linear_scaling":
                # Apply multiplicative scaling
                if isinstance(factor, (list, tuple)):
                    # If factor is a list or tuple, it might be from a more advanced calibrator
                    if len(factor) == 2:
                        # Assume it's [slope, intercept] from regression
                        slope, intercept = factor
                        calibrated_metrics[metric] = calibrated_metrics[metric] * slope + intercept
                    else:
                        logger.warning(f"Unexpected list format for factor: {factor}, using first element")
                        calibrated_metrics[metric] = calibrated_metrics[metric] * factor[0]
                elif isinstance(factor, dict):
                    # If factor is a dict, it might be from a more advanced calibrator
                    if "base_factor" in factor:
                        # Use base_factor from incremental learning
                        calibrated_metrics[metric] = calibrated_metrics[metric] * factor["base_factor"]
                    elif "ensemble_factors" in factor and "linear_scaling" in factor["ensemble_factors"]:
                        # Use linear_scaling factor from ensemble
                        calibrated_metrics[metric] = calibrated_metrics[metric] * factor["ensemble_factors"]["linear_scaling"]
                    else:
                        logger.warning(f"Unsupported dict format for factor: {factor}, using 1.0")
                        calibrated_metrics[metric] = calibrated_metrics[metric] * 1.0
                else:
                    # Normal scalar factor
                    calibrated_metrics[metric] = calibrated_metrics[metric] * factor
                
            elif calibration_method == "additive_adjustment":
                # Apply additive adjustment (original + (1 - factor) * original)
                if isinstance(factor, (list, tuple, dict)):
                    logger.warning(f"Complex factor not supported for additive_adjustment: {factor}, using linear_scaling instead")
                    if isinstance(factor, (list, tuple)) and len(factor) == 2:
                        slope, intercept = factor
                        calibrated_metrics[metric] = calibrated_metrics[metric] * slope + intercept
                    elif isinstance(factor, dict) and "base_factor" in factor:
                        calibrated_metrics[metric] = calibrated_metrics[metric] * factor["base_factor"]
                    else:
                        calibrated_metrics[metric] = calibrated_metrics[metric] * 1.0
                else:
                    calibrated_metrics[metric] = calibrated_metrics[metric] * (2 - factor)
                
            elif calibration_method == "regression":
                # Apply linear regression (assumes factor is [slope, intercept])
                if isinstance(factor, (list, tuple)) and len(factor) == 2:
                    slope, intercept = factor
                    calibrated_metrics[metric] = calibrated_metrics[metric] * slope + intercept
                elif isinstance(factor, dict):
                    if "ensemble_factors" in factor and "regression" in factor["ensemble_factors"]:
                        # Use regression factor from ensemble
                        regression_factor = factor["ensemble_factors"]["regression"]
                        if isinstance(regression_factor, (list, tuple)) and len(regression_factor) == 2:
                            slope, intercept = regression_factor
                            calibrated_metrics[metric] = calibrated_metrics[metric] * slope + intercept
                        else:
                            logger.warning(f"Invalid regression factor in ensemble: {regression_factor}, using 1.0")
                            calibrated_metrics[metric] = calibrated_metrics[metric] * 1.0
                    else:
                        logger.warning(f"Unsupported dict format for regression: {factor}, using 1.0")
                        calibrated_metrics[metric] = calibrated_metrics[metric] * 1.0
                else:
                    logger.warning(f"Invalid regression factor: {factor}, using as multiplicative factor")
                    calibrated_metrics[metric] = calibrated_metrics[metric] * factor
        
        # Create calibrated result
        calibrated_result = SimulationResult(
            model_id=simulation_result.model_id,
            hardware_id=simulation_result.hardware_id,
            metrics=calibrated_metrics,
            batch_size=simulation_result.batch_size,
            precision=simulation_result.precision,
            timestamp=simulation_result.timestamp,
            simulation_version=f"{simulation_result.simulation_version}_calibrated",
            additional_metadata={
                **(simulation_result.additional_metadata or {}),
                "calibration_applied": True,
                "calibration_version": simulation_parameters.get("calibration_version", self.config["calibration_version"]),
                "calibration_method": calibration_method
            }
        )
        
        return calibrated_result
    
    def _group_validation_results(self, validation_results: List[ValidationResult]) -> Dict[Tuple[str, str], List[ValidationResult]]:
        """
        Group validation results by hardware type and model type.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary mapping (hardware_id, model_id) tuples to lists of validation results
        """
        grouped_results = {}
        
        for val_result in validation_results:
            hardware_id = val_result.hardware_result.hardware_id
            model_id = val_result.hardware_result.model_id
            
            key = (hardware_id, model_id)
            
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].append(val_result)
        
        return grouped_results
    
    def _calculate_correction_factors(self, validation_results: List[ValidationResult]) -> Dict[str, float]:
        """
        Calculate correction factors for metrics based on validation results.
        
        Args:
            validation_results: List of validation results for a specific hardware-model combination
            
        Returns:
            Dictionary mapping metric names to correction factors
        """
        correction_factors = {}
        
        # Collect simulation and hardware metrics
        for metric in self.config["metrics_to_calibrate"]:
            sim_values = []
            hw_values = []
            
            for val_result in validation_results:
                # Check if metric exists in both simulation and hardware results
                if (metric in val_result.simulation_result.metrics and 
                    metric in val_result.hardware_result.metrics):
                    
                    sim_value = val_result.simulation_result.metrics[metric]
                    hw_value = val_result.hardware_result.metrics[metric]
                    
                    # Skip if either value is None or zero
                    if sim_value is None or hw_value is None or sim_value == 0:
                        continue
                    
                    sim_values.append(sim_value)
                    hw_values.append(hw_value)
            
            if not sim_values or not hw_values:
                logger.warning(f"No valid data for metric {metric}")
                continue
            
            # Calculate correction factor based on calibration method
            if self.config["calibration_method"] == "linear_scaling":
                # Calculate mean ratio of hardware to simulation
                ratios = [hw / sim for hw, sim in zip(hw_values, sim_values)]
                correction_factor = np.mean(ratios)
                
            elif self.config["calibration_method"] == "additive_adjustment":
                # Calculate mean relative error
                rel_errors = [(hw - sim) / sim for hw, sim in zip(hw_values, sim_values)]
                mean_rel_error = np.mean(rel_errors)
                correction_factor = 1 + mean_rel_error
                
            elif self.config["calibration_method"] == "regression":
                # Perform linear regression
                try:
                    from sklearn.linear_model import LinearRegression
                    
                    X = np.array(sim_values).reshape(-1, 1)
                    y = np.array(hw_values)
                    
                    model = LinearRegression().fit(X, y)
                    slope = model.coef_[0]
                    intercept = model.intercept_
                    
                    correction_factor = [slope, intercept]
                except ImportError:
                    logger.warning("sklearn not available for regression, falling back to linear scaling")
                    ratios = [hw / sim for hw, sim in zip(hw_values, sim_values)]
                    correction_factor = np.mean(ratios)
            else:
                logger.warning(f"Unknown calibration method: {self.config['calibration_method']}")
                ratios = [hw / sim for hw, sim in zip(hw_values, sim_values)]
                correction_factor = np.mean(ratios)
            
            correction_factors[metric] = correction_factor
        
        return correction_factors
    
    def _calculate_error_statistics(
        self,
        validation_results: List[ValidationResult],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate error statistics for validation results.
        
        Args:
            validation_results: List of validation results
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary mapping metric names to error statistics
        """
        statistics = {}
        
        for metric in metrics:
            mape_values = []
            
            for val_result in validation_results:
                if metric in val_result.metrics_comparison:
                    if "mape" in val_result.metrics_comparison[metric]:
                        mape = val_result.metrics_comparison[metric]["mape"]
                        if not np.isnan(mape):
                            mape_values.append(mape)
            
            if mape_values:
                statistics[metric] = {
                    "mean_mape": np.mean(mape_values),
                    "median_mape": np.median(mape_values),
                    "min_mape": np.min(mape_values),
                    "max_mape": np.max(mape_values),
                    "std_dev_mape": np.std(mape_values),
                    "count": len(mape_values)
                }
        
        return statistics