#!/usr/bin/env python3
"""
Advanced Calibrator implementation for the Simulation Accuracy and Validation Framework.

This module provides a more sophisticated implementation of the SimulationCalibrator interface
that uses advanced optimization techniques to calibrate simulation parameters.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import json
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("advanced_calibrator")

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

# Import basic calibrator for fallback and utilities
from duckdb_api.simulation_validation.calibration.basic_calibrator import BasicSimulationCalibrator

# Optional imports for advanced methods
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score, KFold
    sklearn_available = True
except ImportError:
    logger.warning("scikit-learn not available, some advanced calibration methods will be limited")
    sklearn_available = False

try:
    from skopt import gp_minimize
    from skopt.space import Real
    skopt_available = True
except ImportError:
    logger.warning("scikit-optimize not available, Bayesian optimization will be limited")
    skopt_available = False


class AdvancedSimulationCalibrator(SimulationCalibrator):
    """
    Advanced implementation of a simulation calibrator using sophisticated optimization techniques.
    
    This calibrator uses multiple strategies including:
    1. Bayesian optimization for parameter tuning
    2. Neural networks for complex relationship modeling
    3. Ensemble methods for combining multiple calibration techniques
    4. Incremental learning for continuous improvement
    5. Hardware-specific optimization profiles
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced simulation calibrator.
        
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
            "calibration_method": "ensemble",  # ensemble, bayesian, network, incremental
            "min_samples_per_hardware": 5,     # Minimum number of samples needed for calibration
            "min_samples_per_model": 5,        # Minimum number of samples needed for calibration
            "calibration_version": "advanced_v1.0",
            "learning_rate": 0.5,              # Learning rate for parameter updates (0-1)
            "use_hardware_profiles": True,     # Whether to use hardware-specific profiles
            "ensemble_weights": {              # Weights for ensemble methods
                "linear_scaling": 0.3,
                "bayesian": 0.4,
                "network": 0.3
            },
            "bayesian_iterations": 50,         # Number of iterations for Bayesian optimization
            "network_hidden_layers": [10, 5],  # Architecture for neural network
            "network_epochs": 100,             # Training epochs for neural network
            "calibration_history_size": 10,    # Number of past calibrations to keep
            "enable_cross_validation": True,   # Whether to use cross-validation
            "cross_validation_folds": 3,       # Number of folds for cross-validation
            "parameter_constraints": {         # Constraints for parameter values
                "min_scale_factor": 0.5,
                "max_scale_factor": 2.0
            }
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Create a basic calibrator for fallback
        self.basic_calibrator = BasicSimulationCalibrator({
            "metrics_to_calibrate": self.config["metrics_to_calibrate"],
            "calibration_method": "linear_scaling",
            "min_samples_per_hardware": self.config["min_samples_per_hardware"],
            "min_samples_per_model": self.config["min_samples_per_model"],
            "calibration_version": f"{self.config['calibration_version']}_basic_fallback",
            "learning_rate": self.config["learning_rate"]
        })
        
        # Initialize hardware profiles
        self.hardware_profiles = self._initialize_hardware_profiles()
        
        # Initialize calibration history
        self.calibration_history = []
    
    def calibrate(
        self,
        validation_results: List[ValidationResult],
        simulation_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calibrate simulation parameters based on validation results using advanced techniques.
        
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
        
        # Create calibration history if it doesn't exist
        if "calibration_history" not in updated_parameters:
            updated_parameters["calibration_history"] = []
        
        # Update calibration history with this calibration event
        calibration_event = {
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid.uuid4()),
            "num_validation_results": len(validation_results),
            "hardware_models": list(validation_by_hardware_model.keys()),
            "method": self.config["calibration_method"]
        }
        
        # Keep only the most recent calibration events
        updated_parameters["calibration_history"] = (
            updated_parameters["calibration_history"][-self.config["calibration_history_size"]+1:] + 
            [calibration_event]
        )
        
        # Perform calibration for each hardware-model combination
        for (hardware_id, model_id), results in validation_by_hardware_model.items():
            if len(results) < self.config["min_samples_per_hardware"]:
                logger.info(f"Skipping calibration for {hardware_id}/{model_id}: insufficient samples ({len(results)})")
                continue
            
            # Choose calibration method based on configuration and data characteristics
            calibration_method = self._select_calibration_method(hardware_id, model_id, results)
            
            # Apply the selected calibration method
            correction_factors = self._apply_calibration_method(
                calibration_method, hardware_id, model_id, results, updated_parameters
            )
            
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
                
                if isinstance(factor, (list, tuple)) and isinstance(current_factor, (list, tuple)):
                    # For complex factors like regression parameters [slope, intercept]
                    updated_factor = [
                        current * (1 - lr) + new * lr
                        for current, new in zip(current_factor, factor)
                    ]
                elif isinstance(factor, dict) and isinstance(current_factor, dict):
                    # For dictionary factors like neural network weights
                    updated_factor = {}
                    for k, v in factor.items():
                        if k in current_factor:
                            if isinstance(current_factor[k], dict) or isinstance(v, dict):
                                # For nested dictionaries, just use the new value
                                updated_factor[k] = v
                            else:
                                # For simple values, apply learning rate
                                try:
                                    updated_factor[k] = float(current_factor[k]) * (1 - lr) + float(v) * lr
                                except (TypeError, ValueError):
                                    # If conversion fails, just use the new value
                                    updated_factor[k] = v
                        else:
                            # If key doesn't exist in current_factor, use the new value
                            updated_factor[k] = v
                else:
                    # For simple scalar factors
                    # Handle the case where factor might be a sequence or dict
                    if isinstance(factor, (list, tuple)):
                        if isinstance(current_factor, (list, tuple)) and len(current_factor) == len(factor):
                            updated_factor = [
                                curr * (1 - lr) + new * lr
                                for curr, new in zip(current_factor, factor)
                            ]
                        else:
                            # If current_factor isn't compatible, just use new factor
                            updated_factor = factor
                    elif isinstance(factor, dict):
                        # Handle dictionary case
                        if isinstance(current_factor, dict):
                            updated_factor = {}
                            for k, v in factor.items():
                                if k in current_factor:
                                    if isinstance(current_factor[k], dict) or isinstance(v, dict):
                                        # For nested dictionaries, just use the new value
                                        updated_factor[k] = v
                                    else:
                                        # For simple values, apply learning rate
                                        try:
                                            updated_factor[k] = float(current_factor[k]) * (1 - lr) + float(v) * lr
                                        except (TypeError, ValueError):
                                            # If conversion fails, just use the new value
                                            updated_factor[k] = v
                                else:
                                    # If key doesn't exist in current_factor, use the new value
                                    updated_factor[k] = v
                        else:
                            updated_factor = factor
                    else:
                        updated_factor = current_factor * (1 - lr) + factor * lr
                
                # Apply constraints to prevent extreme values
                if not isinstance(updated_factor, (dict, list, tuple)):
                    updated_factor = max(
                        self.config["parameter_constraints"]["min_scale_factor"],
                        min(self.config["parameter_constraints"]["max_scale_factor"], updated_factor)
                    )
                
                updated_parameters["correction_factors"][hardware_id][model_id][metric] = updated_factor
                
                logger.info(f"Updated correction factor for {hardware_id}/{model_id}/{metric}: {current_factor} -> {updated_factor}")
        
        # Update calibration metadata
        updated_parameters["calibration_version"] = self.config["calibration_version"]
        updated_parameters["calibration_method"] = self.config["calibration_method"]
        updated_parameters["num_samples_used"] = len(validation_results)
        updated_parameters["last_calibration_timestamp"] = datetime.now().isoformat()
        
        return updated_parameters
    
    def evaluate_calibration(
        self,
        before_calibration: List[ValidationResult],
        after_calibration: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of a calibration with enhanced metrics.
        
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
        before_stats = self._calculate_enhanced_error_statistics(before_calibration, metrics_to_evaluate)
        
        # Calculate statistics after calibration
        after_stats = self._calculate_enhanced_error_statistics(after_calibration, metrics_to_evaluate)
        
        # Calculate improvement for each metric
        improvement = {}
        for metric in metrics_to_evaluate:
            if metric in before_stats and metric in after_stats:
                metric_improvement = {
                    "before": before_stats[metric],
                    "after": after_stats[metric],
                    "absolute_improvement": {}
                }
                
                # Calculate improvements for each error metric
                for error_type in ["mape", "rmse", "correlation", "rank_preservation"]:
                    if error_type in before_stats[metric] and error_type in after_stats[metric]:
                        before_value = before_stats[metric][error_type]
                        after_value = after_stats[metric][error_type]
                        
                        # For correlation and rank preservation, higher is better
                        if error_type in ["correlation", "rank_preservation"]:
                            diff = after_value - before_value
                            rel_imp = diff / (1 - before_value) if before_value < 1 else 0
                        else:
                            # For MAPE and RMSE, lower is better
                            diff = before_value - after_value
                            rel_imp = diff / before_value if before_value > 0 else 0
                        
                        metric_improvement["absolute_improvement"][error_type] = diff
                        metric_improvement["relative_improvement_pct"] = rel_imp * 100
                
                improvement[metric] = metric_improvement
        
        # Calculate overall improvement
        overall_before_mape = np.mean([stats.get("mape", 0) for stats in before_stats.values()])
        overall_after_mape = np.mean([stats.get("mape", 0) for stats in after_stats.values()])
        
        overall_before_rmse = np.mean([stats.get("rmse", 0) for stats in before_stats.values()])
        overall_after_rmse = np.mean([stats.get("rmse", 0) for stats in after_stats.values()])
        
        if overall_before_mape > 0:
            overall_mape_relative_improvement = (overall_before_mape - overall_after_mape) / overall_before_mape * 100.0
        else:
            overall_mape_relative_improvement = 0.0
        
        if overall_before_rmse > 0:
            overall_rmse_relative_improvement = (overall_before_rmse - overall_after_rmse) / overall_before_rmse * 100.0
        else:
            overall_rmse_relative_improvement = 0.0
        
        # Prepare evaluation results
        evaluation = {
            "metrics": improvement,
            "overall": {
                "mape": {
                    "before": overall_before_mape,
                    "after": overall_after_mape,
                    "absolute_improvement": overall_before_mape - overall_after_mape,
                    "relative_improvement_pct": overall_mape_relative_improvement
                },
                "rmse": {
                    "before": overall_before_rmse,
                    "after": overall_after_rmse,
                    "absolute_improvement": overall_before_rmse - overall_after_rmse,
                    "relative_improvement_pct": overall_rmse_relative_improvement
                }
            },
            "num_samples": len(before_calibration),
            "calibration_method": self.config["calibration_method"],
            "calibration_version": self.config["calibration_version"],
            "timestamp": datetime.now().isoformat()
        }
        
        return evaluation
    
    def apply_calibration(
        self,
        simulation_result: SimulationResult,
        simulation_parameters: Dict[str, Any]
    ) -> SimulationResult:
        """
        Apply advanced calibration to a simulation result.
        
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
        
        # Check if hardware profile exists
        hardware_specific_profile = False
        if self.config["use_hardware_profiles"] and hardware_id in self.hardware_profiles:
            profile = self.hardware_profiles[hardware_id]
            hardware_specific_profile = True
            logger.info(f"Using hardware-specific profile for {hardware_id}")
        
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
            
            # Apply hardware-specific adjustments if available
            if hardware_specific_profile and metric in profile.get("metric_adjustments", {}):
                metric_profile = profile["metric_adjustments"][metric]
                batch_size = simulation_result.batch_size
                precision = simulation_result.precision
                
                # Apply batch size specific adjustments
                if "batch_size_factors" in metric_profile:
                    batch_factors = metric_profile["batch_size_factors"]
                    if str(batch_size) in batch_factors:
                        # Apply batch-specific adjustment
                        batch_factor = batch_factors[str(batch_size)]
                        logger.info(f"Applying batch-specific factor for {hardware_id}/{metric}/batch={batch_size}: {batch_factor}")
                        
                        if isinstance(factor, (list, tuple)):
                            factor = [f * batch_factor for f in factor]
                        elif isinstance(factor, dict):
                            # For dictionary factors, only apply to specific fields if possible
                            if "base_factor" in factor:
                                factor["base_factor"] = factor["base_factor"] * batch_factor
                            elif "ensemble_factors" in factor and "linear_scaling" in factor["ensemble_factors"]:
                                factor["ensemble_factors"]["linear_scaling"] = factor["ensemble_factors"]["linear_scaling"] * batch_factor
                            else:
                                # Cannot apply batch factor to dictionary, skip adjustment
                                logger.warning(f"Cannot apply batch factor to dictionary factor: {factor}")
                        else:
                            factor = factor * batch_factor
                
                # Apply precision specific adjustments
                if "precision_factors" in metric_profile:
                    precision_factors = metric_profile["precision_factors"]
                    if precision in precision_factors:
                        # Apply precision-specific adjustment
                        precision_factor = precision_factors[precision]
                        logger.info(f"Applying precision-specific factor for {hardware_id}/{metric}/precision={precision}: {precision_factor}")
                        
                        if isinstance(factor, (list, tuple)):
                            factor = [f * precision_factor for f in factor]
                        elif isinstance(factor, dict):
                            # For dictionary factors, only apply to specific fields if possible
                            if "base_factor" in factor:
                                factor["base_factor"] = factor["base_factor"] * precision_factor
                            elif "ensemble_factors" in factor and "linear_scaling" in factor["ensemble_factors"]:
                                factor["ensemble_factors"]["linear_scaling"] = factor["ensemble_factors"]["linear_scaling"] * precision_factor
                            else:
                                # Cannot apply precision factor to dictionary, skip adjustment
                                logger.warning(f"Cannot apply precision factor to dictionary factor: {factor}")
                        else:
                            factor = factor * precision_factor
            
            # Apply calibration based on method
            if calibration_method == "ensemble":
                # Apply ensemble of methods
                if isinstance(factor, dict) and "ensemble_factors" in factor:
                    ensemble_factors = factor["ensemble_factors"]
                    ensemble_weights = factor.get("ensemble_weights", self.config["ensemble_weights"])
                    
                    # Calculate weighted average of ensemble methods
                    calibrated_value = 0
                    weight_sum = 0
                    
                    for method, weight in ensemble_weights.items():
                        if method in ensemble_factors:
                            method_factor = ensemble_factors[method]
                            
                            if method == "linear_scaling":
                                # Handle different factor types for linear_scaling
                                if isinstance(method_factor, (list, tuple)):
                                    if len(method_factor) == 2:
                                        # Assume it's [slope, intercept]
                                        slope, intercept = method_factor
                                        method_value = calibrated_metrics[metric] * float(slope) + float(intercept)
                                    else:
                                        # Use the first element as a multiplier
                                        logger.warning(f"Unexpected list length for linear_scaling: {method_factor}")
                                        method_value = calibrated_metrics[metric] * float(method_factor[0]) if method_factor else calibrated_metrics[metric]
                                elif isinstance(method_factor, dict):
                                    # Try to extract a usable value
                                    if "base_factor" in method_factor:
                                        method_value = calibrated_metrics[metric] * float(method_factor["base_factor"])
                                    else:
                                        logger.warning(f"Unsupported dict format for linear_scaling: {method_factor}")
                                        method_value = calibrated_metrics[metric]
                                else:
                                    # Assume it's a numeric value
                                    method_value = calibrated_metrics[metric] * float(method_factor)
                            elif method == "bayesian" or method == "network":
                                # These advanced methods might use different factor formats
                                if isinstance(method_factor, (list, tuple)) and len(method_factor) == 2:
                                    slope, intercept = method_factor
                                    method_value = calibrated_metrics[metric] * float(slope) + float(intercept)
                                elif isinstance(method_factor, (list, tuple)):
                                    logger.warning(f"Unexpected list format for {method}: {method_factor}")
                                    method_value = calibrated_metrics[metric] * float(method_factor[0]) if method_factor else calibrated_metrics[metric]
                                elif isinstance(method_factor, dict):
                                    logger.warning(f"Unsupported dict format for {method}: {method_factor}")
                                    method_value = calibrated_metrics[metric]
                                else:
                                    # Assume it's a numeric value
                                    method_value = calibrated_metrics[metric] * float(method_factor)
                            else:
                                # Default to simple scaling with type checking
                                if isinstance(method_factor, (list, tuple)) and len(method_factor) == 2:
                                    # Assume it's [slope, intercept]
                                    slope, intercept = method_factor
                                    method_value = calibrated_metrics[metric] * float(slope) + float(intercept)
                                elif isinstance(method_factor, (list, tuple)):
                                    logger.warning(f"Unexpected list format for {method}: {method_factor}")
                                    method_value = calibrated_metrics[metric] * float(method_factor[0]) if method_factor else calibrated_metrics[metric]
                                elif isinstance(method_factor, dict):
                                    logger.warning(f"Unsupported dict format for {method}: {method_factor}")
                                    method_value = calibrated_metrics[metric]
                                else:
                                    # Assume it's a numeric value
                                    method_value = calibrated_metrics[metric] * float(method_factor)
                            
                            calibrated_value += method_value * float(weight)
                            weight_sum += float(weight)
                    
                    if weight_sum > 0:
                        calibrated_metrics[metric] = calibrated_value / weight_sum
                    else:
                        # Fallback to original value
                        logger.warning(f"Zero weight sum for ensemble methods on {metric}, using original value")
                else:
                    # Fallback to linear scaling if ensemble factors not available
                    # Handle different factor types
                    if isinstance(factor, (list, tuple)):
                        if len(factor) == 2:
                            # Assume it's [slope, intercept]
                            slope, intercept = factor
                            calibrated_metrics[metric] = calibrated_metrics[metric] * float(slope) + float(intercept)
                        elif len(factor) > 0:
                            # Use the first element
                            logger.warning(f"Using first element of list factor: {factor}")
                            calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor[0])
                        else:
                            # Empty list, maintain original value
                            logger.warning(f"Empty list factor, using original value")
                    elif isinstance(factor, dict):
                        # Try to extract a usable factor
                        if "base_factor" in factor:
                            calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor["base_factor"])
                        elif "ensemble_factors" in factor and "linear_scaling" in factor["ensemble_factors"]:
                            linear_factor = factor["ensemble_factors"]["linear_scaling"]
                            if isinstance(linear_factor, (list, tuple)):
                                if len(linear_factor) == 2:
                                    slope, intercept = linear_factor
                                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(slope) + float(intercept)
                                elif len(linear_factor) > 0:
                                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(linear_factor[0])
                                else:
                                    logger.warning(f"Empty linear factor list, using original value")
                            else:
                                calibrated_metrics[metric] = calibrated_metrics[metric] * float(linear_factor)
                        else:
                            logger.warning(f"Unsupported dict factor format: {factor}, using original value")
                    else:
                        # Assume it's a numeric value
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor)
            
            elif calibration_method == "bayesian" or calibration_method == "network":
                # Apply model-based calibration (format should be [slope, intercept])
                if isinstance(factor, (list, tuple)) and len(factor) == 2:
                    slope, intercept = factor
                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(slope) + float(intercept)
                elif isinstance(factor, (list, tuple)) and len(factor) > 0:
                    # Use first element as a scalar if not the right format
                    logger.warning(f"Unexpected list length for {calibration_method}: {factor}, using first element")
                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor[0])
                elif isinstance(factor, dict):
                    if "base_factor" in factor:
                        # Try to use base_factor if available
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor["base_factor"])
                    elif "ensemble_factors" in factor and "linear_scaling" in factor["ensemble_factors"]:
                        # Use linear_scaling factor from ensemble
                        linear_factor = factor["ensemble_factors"]["linear_scaling"]
                        if isinstance(linear_factor, (list, tuple)):
                            if len(linear_factor) == 2:
                                slope, intercept = linear_factor
                                calibrated_metrics[metric] = calibrated_metrics[metric] * float(slope) + float(intercept)
                            elif len(linear_factor) > 0:
                                calibrated_metrics[metric] = calibrated_metrics[metric] * float(linear_factor[0])
                            else:
                                logger.warning(f"Empty linear factor list in ensemble, using original value")
                        else:
                            # Use linear factor directly
                            calibrated_metrics[metric] = calibrated_metrics[metric] * float(linear_factor)
                    else:
                        logger.warning(f"Unsupported dict format for {calibration_method}: {factor}, using original value")
                else:
                    try:
                        # Fallback to simple scaling with type conversion
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor)
                    except (TypeError, ValueError):
                        logger.warning(f"Could not convert {factor} to float for {calibration_method}, using original value")
            
            elif calibration_method == "incremental":
                # Apply incremental calibration with trend adjustment
                if isinstance(factor, dict) and "base_factor" in factor and "trend_factor" in factor:
                    base_factor = float(factor["base_factor"])
                    trend_factor = float(factor["trend_factor"])
                    recent_samples = float(factor.get("recent_samples", 1))
                    
                    # Apply base calibration with trend adjustment
                    calibrated_metrics[metric] = calibrated_metrics[metric] * base_factor * (1 + trend_factor * recent_samples)
                elif isinstance(factor, dict) and "ensemble_factors" in factor:
                    # Try to use ensemble factors if available
                    if "linear_scaling" in factor["ensemble_factors"]:
                        linear_factor = factor["ensemble_factors"]["linear_scaling"]
                        if isinstance(linear_factor, (list, tuple)):
                            if len(linear_factor) == 2:
                                slope, intercept = linear_factor
                                calibrated_metrics[metric] = calibrated_metrics[metric] * float(slope) + float(intercept)
                            elif len(linear_factor) > 0:
                                calibrated_metrics[metric] = calibrated_metrics[metric] * float(linear_factor[0])
                            else:
                                logger.warning(f"Empty linear factor list in ensemble, using original value")
                        else:
                            calibrated_metrics[metric] = calibrated_metrics[metric] * float(linear_factor)
                    else:
                        logger.warning(f"No usable factor in ensemble for incremental method: {factor}, using original value")
                elif isinstance(factor, (list, tuple)) and len(factor) > 0:
                    # Use first element as a scalar if not a dict
                    logger.warning(f"Expected dict for incremental method, got: {factor}, using first element")
                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor[0])
                else:
                    try:
                        # Fallback to simple scaling with type conversion
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor)
                    except (TypeError, ValueError):
                        logger.warning(f"Could not convert {factor} to float for incremental method, using original value")
            
            elif calibration_method == "linear_scaling":
                # Apply multiplicative scaling with type conversion
                if isinstance(factor, (list, tuple)) and len(factor) > 0:
                    # Use first element as a scalar
                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor[0])
                elif isinstance(factor, dict):
                    # Try to extract usable factor
                    if "base_factor" in factor:
                        # Use base_factor if available
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor["base_factor"])
                    elif "ensemble_factors" in factor and "linear_scaling" in factor["ensemble_factors"]:
                        # Use linear_scaling factor from ensemble
                        linear_factor = factor["ensemble_factors"]["linear_scaling"]
                        if isinstance(linear_factor, (list, tuple)):
                            if len(linear_factor) == 2:
                                slope, intercept = linear_factor
                                calibrated_metrics[metric] = calibrated_metrics[metric] * float(slope) + float(intercept)
                            elif len(linear_factor) > 0:
                                calibrated_metrics[metric] = calibrated_metrics[metric] * float(linear_factor[0])
                            else:
                                logger.warning(f"Empty linear factor list in ensemble, using original value")
                        else:
                            # Use linear factor directly
                            calibrated_metrics[metric] = calibrated_metrics[metric] * float(linear_factor)
                    else:
                        logger.warning(f"Unsupported dict format for linear_scaling: {factor}, using original value")
                else:
                    # Try direct conversion to float
                    try:
                        # Regular scalar conversion
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor)
                    except (TypeError, ValueError):
                        logger.warning(f"Could not convert {factor} to float for linear_scaling, using original value")
            
            elif calibration_method == "additive_adjustment":
                # Apply additive adjustment (original + (1 - factor) * original)
                if isinstance(factor, (list, tuple)) and len(factor) > 0:
                    # Use first element for the adjustment
                    calibrated_metrics[metric] = calibrated_metrics[metric] * (2 - float(factor[0]))
                elif isinstance(factor, dict):
                    # Try to extract usable factor
                    if "base_factor" in factor:
                        # Use base_factor if available
                        calibrated_metrics[metric] = calibrated_metrics[metric] * (2 - float(factor["base_factor"]))
                    elif "ensemble_factors" in factor and "linear_scaling" in factor["ensemble_factors"]:
                        # Use linear_scaling factor from ensemble
                        linear_factor = factor["ensemble_factors"]["linear_scaling"]
                        if isinstance(linear_factor, (list, tuple)):
                            if len(linear_factor) == 2:
                                # For regression, use the slope but convert to additive adjustment
                                slope = linear_factor[0]
                                calibrated_metrics[metric] = calibrated_metrics[metric] * (2 - float(slope))
                            elif len(linear_factor) > 0:
                                calibrated_metrics[metric] = calibrated_metrics[metric] * (2 - float(linear_factor[0]))
                            else:
                                logger.warning(f"Empty linear factor list in ensemble, using original value")
                        else:
                            # Use linear factor directly but convert to additive adjustment
                            calibrated_metrics[metric] = calibrated_metrics[metric] * (2 - float(linear_factor))
                    else:
                        logger.warning(f"Unsupported dict format for additive_adjustment: {factor}, using original value")
                else:
                    # Try direct conversion to float
                    try:
                        # Regular scalar conversion
                        calibrated_metrics[metric] = calibrated_metrics[metric] * (2 - float(factor))
                    except (TypeError, ValueError):
                        logger.warning(f"Could not convert {factor} to float for additive_adjustment, using original value")
            
            elif calibration_method == "regression":
                # Apply linear regression (assumes factor is [slope, intercept])
                if isinstance(factor, (list, tuple)) and len(factor) == 2:
                    slope, intercept = factor
                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(slope) + float(intercept)
                elif isinstance(factor, (list, tuple)) and len(factor) > 0:
                    # Use first element as a scalar if not the right format
                    logger.warning(f"Unexpected list length for regression: {factor}, using first element as multiplier")
                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor[0])
                elif isinstance(factor, dict):
                    if "ensemble_factors" in factor and "regression" in factor["ensemble_factors"]:
                        regression_factor = factor["ensemble_factors"]["regression"]
                        if isinstance(regression_factor, (list, tuple)) and len(regression_factor) == 2:
                            slope, intercept = regression_factor
                            calibrated_metrics[metric] = calibrated_metrics[metric] * float(slope) + float(intercept)
                        elif isinstance(regression_factor, (list, tuple)) and len(regression_factor) > 0:
                            # Use first element as a scalar if not the right format
                            logger.warning(f"Unexpected list length for regression in ensemble: {regression_factor}, using first element")
                            calibrated_metrics[metric] = calibrated_metrics[metric] * float(regression_factor[0])
                        else:
                            logger.warning(f"Invalid regression factor in ensemble: {regression_factor}, using original value")
                    elif "base_factor" in factor:
                        # Use base_factor if available (though this isn't regression, it's a fallback)
                        logger.warning(f"No regression parameters in factor dict, using base_factor as fallback")
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor["base_factor"])
                    else:
                        logger.warning(f"Invalid regression factor dict: {factor}, using original value")
                else:
                    try:
                        # Try as simple scaling as a last resort
                        logger.warning(f"Invalid regression factor: {factor}, trying as multiplicative factor")
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor)
                    except (TypeError, ValueError):
                        logger.warning(f"Could not convert {factor} to float for regression, using original value")
            
            else:
                logger.warning(f"Unknown calibration method: {calibration_method}")
                # Fallback to simple scaling with robust type handling
                if isinstance(factor, (list, tuple)) and len(factor) == 2:
                    slope, intercept = factor
                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(slope) + float(intercept)
                elif isinstance(factor, (list, tuple)) and len(factor) > 0:
                    # Use first element
                    calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor[0])
                elif isinstance(factor, dict):
                    if "base_factor" in factor:
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor["base_factor"])
                    else:
                        logger.warning(f"Unknown factor format: {factor}, using original value")
                else:
                    # Last resort - try to convert to float
                    try:
                        calibrated_metrics[metric] = calibrated_metrics[metric] * float(factor)
                    except (TypeError, ValueError):
                        logger.error(f"Could not convert factor to float: {factor}, using original value")
                        # Keep original value
        
        # Create calibrated result
        calibrated_result = SimulationResult(
            model_id=simulation_result.model_id,
            hardware_id=simulation_result.hardware_id,
            metrics=calibrated_metrics,
            batch_size=simulation_result.batch_size,
            precision=simulation_result.precision,
            timestamp=simulation_result.timestamp,
            simulation_version=f"{simulation_result.simulation_version}_advanced_calibrated",
            additional_metadata={
                **(simulation_result.additional_metadata or {}),
                "calibration_applied": True,
                "calibration_version": simulation_parameters.get("calibration_version", self.config["calibration_version"]),
                "calibration_method": calibration_method,
                "hardware_specific_profile_used": hardware_specific_profile
            }
        )
        
        return calibrated_result
    
    def _select_calibration_method(
        self,
        hardware_id: str,
        model_id: str,
        validation_results: List[ValidationResult]
    ) -> str:
        """
        Select the most appropriate calibration method based on the data characteristics.
        
        Args:
            hardware_id: Hardware ID
            model_id: Model ID
            validation_results: Validation results for this hardware-model combination
            
        Returns:
            Selected calibration method name
        """
        # Default to the configured method
        method = self.config["calibration_method"]
        
        # Check if we have hardware-specific profiles
        if self.config["use_hardware_profiles"] and hardware_id in self.hardware_profiles:
            profile = self.hardware_profiles[hardware_id]
            if "preferred_calibration_method" in profile:
                method = profile["preferred_calibration_method"]
                logger.info(f"Using hardware-specific calibration method for {hardware_id}: {method}")
                return method
        
        # Check data characteristics to determine the best method
        # For small datasets, simple methods are better
        if len(validation_results) < 10:
            logger.info(f"Small dataset for {hardware_id}/{model_id}, using linear_scaling")
            return "linear_scaling"
        
        # Check for recent drift detection results
        recent_drift = self._check_recent_drift_detection(hardware_id, model_id)
        if recent_drift and recent_drift.get("significant_drift_detected", False):
            # If significant drift was detected, use more advanced methods
            if len(validation_results) >= 15:
                logger.info(f"Significant drift detected for {hardware_id}/{model_id}, using bayesian optimization")
                return "bayesian"
            else:
                logger.info(f"Significant drift detected for {hardware_id}/{model_id}, but limited data, using ensemble")
                return "ensemble"
        
        # Check if calibration frequency should be adapted based on drift and history
        calibration_frequency = self._determine_calibration_frequency(
            hardware_id, model_id, validation_results, self.calibration_history)
        
        if calibration_frequency.get("adaptive_method_selection", False):
            suggested_method = calibration_frequency.get("suggested_method")
            if suggested_method:
                logger.info(f"Using adaptive method selection for {hardware_id}/{model_id}: {suggested_method}")
                return suggested_method
        
        # For larger datasets, check if relationships are linear or complex
        linear_fit_quality = self._check_linear_fit_quality(validation_results)
        
        if linear_fit_quality > 0.9:
            # Good linear fit, use regression
            logger.info(f"Good linear fit for {hardware_id}/{model_id}, using regression")
            return "regression"
        elif linear_fit_quality > 0.7:
            # Moderate linear fit, could use regression or ensemble
            logger.info(f"Moderate linear fit for {hardware_id}/{model_id}, using ensemble")
            return "ensemble"
        else:
            # Poor linear fit, try advanced methods if enough data
            if len(validation_results) >= 20:
                # Use neural network for complex relationships
                try:
                    # Check if we have sklearn available for advanced methods
                    import sklearn
                    logger.info(f"Complex relationship for {hardware_id}/{model_id}, using advanced methods")
                    return "network"
                except ImportError:
                    logger.info(f"sklearn not available, falling back to ensemble for {hardware_id}/{model_id}")
                    return "ensemble"
            else:
                # Not enough data for advanced methods
                logger.info(f"Not enough data for advanced methods for {hardware_id}/{model_id}, using ensemble")
                return "ensemble"
    
    def _check_recent_drift_detection(
        self, 
        hardware_id: str, 
        model_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if there have been recent drift detection results for this hardware-model combination.
        
        Args:
            hardware_id: Hardware ID
            model_id: Model ID
            
        Returns:
            Dictionary with drift detection results or None if no recent results
        """
        # This is a placeholder for integration with the drift detection system
        # In a real implementation, this would query the drift detection results storage
        
        # For now, return None to indicate no drift detection results
        return None
        
    def _determine_calibration_frequency(
        self, 
        hardware_id: str, 
        model_id: str,
        recent_validation_results: List[ValidationResult],
        calibration_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determine appropriate calibration frequency based on drift detection and history.
        
        Args:
            hardware_id: Hardware ID
            model_id: Model ID
            recent_validation_results: Recent validation results for this combination
            calibration_history: History of calibration events
            
        Returns:
            Dictionary with calibration frequency recommendations
        """
        # Default recommendation
        recommendation = {
            "should_calibrate": True,
            "recommended_frequency_days": 30,  # Default to monthly
            "reason": "Regular calibration schedule",
            "adaptive_method_selection": False
        }
        
        # If no history, recommend immediate calibration
        if not calibration_history:
            recommendation["should_calibrate"] = True
            recommendation["recommended_frequency_days"] = 7  # Start with weekly
            recommendation["reason"] = "No previous calibration"
            return recommendation
        
        # Check when the last calibration was performed
        try:
            last_calibration = calibration_history[-1]
            last_calibration_time = datetime.fromisoformat(last_calibration["timestamp"])
            days_since_last_calibration = (datetime.now() - last_calibration_time).days
            
            # Check for recent drift detection
            recent_drift = self._check_recent_drift_detection(hardware_id, model_id)
            
            if recent_drift and recent_drift.get("significant_drift_detected", False):
                # If significant drift was detected, recommend immediate calibration
                recommendation["should_calibrate"] = True
                recommendation["recommended_frequency_days"] = max(7, days_since_last_calibration // 2)
                recommendation["reason"] = "Significant drift detected"
                recommendation["drift_details"] = recent_drift
                
                # Also recommend more advanced calibration method
                recommendation["adaptive_method_selection"] = True
                if len(recent_validation_results) >= 20:
                    recommendation["suggested_method"] = "bayesian"
                elif len(recent_validation_results) >= 10:
                    recommendation["suggested_method"] = "ensemble"
                else:
                    recommendation["suggested_method"] = "regression"
                
            elif days_since_last_calibration > 90:
                # If it's been more than 3 months, recommend calibration
                recommendation["should_calibrate"] = True
                recommendation["recommended_frequency_days"] = 30
                recommendation["reason"] = "Long time since last calibration"
            elif len(calibration_history) >= 3:
                # If we have enough history, analyze calibration impact
                # Check if calibrations are making significant improvements
                calibration_improvements = [
                    event.get("improvement_metrics", {}).get("overall_improvement", 0)
                    for event in calibration_history[-3:]
                ]
                
                avg_improvement = sum(calibration_improvements) / len(calibration_improvements)
                
                if avg_improvement < 5:  # Less than 5% improvement
                    # If calibrations aren't helping much, recommend less frequent calibration
                    recommendation["should_calibrate"] = days_since_last_calibration > 60
                    recommendation["recommended_frequency_days"] = 60
                    recommendation["reason"] = "Limited benefit from recent calibrations"
                    
                    # Maybe try a different method if we're not getting good results
                    if days_since_last_calibration > 30:
                        previous_method = last_calibration.get("method", "")
                        if previous_method == "linear_scaling":
                            recommendation["adaptive_method_selection"] = True
                            recommendation["suggested_method"] = "regression"
                        elif previous_method == "regression":
                            recommendation["adaptive_method_selection"] = True
                            recommendation["suggested_method"] = "ensemble"
                        
                elif avg_improvement > 15:  # More than 15% improvement
                    # If calibrations are helping a lot, recommend more frequent calibration
                    recommendation["should_calibrate"] = days_since_last_calibration > 14
                    recommendation["recommended_frequency_days"] = 14
                    recommendation["reason"] = "Significant benefit from recent calibrations"
                    
                    # Keep using the same method if it's working well
                    recommendation["adaptive_method_selection"] = True
                    recommendation["suggested_method"] = last_calibration.get("method", self.config["calibration_method"])
                else:
                    # Otherwise, stick with monthly calibration
                    recommendation["should_calibrate"] = days_since_last_calibration > 30
                    recommendation["recommended_frequency_days"] = 30
                    recommendation["reason"] = "Normal calibration schedule"
            else:
                # If not enough history, use default recommendation
                recommendation["should_calibrate"] = days_since_last_calibration > 30
                
        except (KeyError, ValueError, IndexError) as e:
            # If there's an issue with the history, recommend calibration to be safe
            logger.warning(f"Error analyzing calibration history: {e}")
            recommendation["should_calibrate"] = True
            recommendation["reason"] = "Error analyzing calibration history"
        
        return recommendation
    
    def _apply_calibration_method(
        self,
        method: str,
        hardware_id: str,
        model_id: str,
        validation_results: List[ValidationResult],
        current_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply the selected calibration method to generate correction factors.
        
        Args:
            method: Calibration method to use
            hardware_id: Hardware ID
            model_id: Model ID
            validation_results: Validation results for this hardware-model combination
            current_parameters: Current simulation parameters
            
        Returns:
            Dictionary of correction factors
        """
        # Get current correction factors for this hardware-model combination
        current_factors = {}
        if (hardware_id in current_parameters.get("correction_factors", {}) and
            model_id in current_parameters["correction_factors"].get(hardware_id, {})):
            current_factors = current_parameters["correction_factors"][hardware_id][model_id]
        
        if method == "linear_scaling":
            # Use basic calibrator for linear scaling
            basic_config = self.basic_calibrator.config.copy()
            basic_config["calibration_method"] = "linear_scaling"
            self.basic_calibrator.config = basic_config
            
            return self.basic_calibrator._calculate_correction_factors(validation_results)
        
        elif method == "regression":
            # Use basic calibrator with regression method
            basic_config = self.basic_calibrator.config.copy()
            basic_config["calibration_method"] = "regression"
            self.basic_calibrator.config = basic_config
            
            return self.basic_calibrator._calculate_correction_factors(validation_results)
        
        elif method == "bayesian":
            # Apply Bayesian optimization for parameter tuning
            return self._apply_bayesian_optimization(hardware_id, model_id, validation_results, current_factors)
        
        elif method == "network":
            # Apply neural network for complex relationships
            return self._apply_neural_network(hardware_id, model_id, validation_results, current_factors)
        
        elif method == "incremental":
            # Apply incremental learning with trend analysis
            return self._apply_incremental_learning(hardware_id, model_id, validation_results, 
                                                   current_factors, current_parameters)
        
        elif method == "ensemble":
            # Apply ensemble of methods
            return self._apply_ensemble_methods(hardware_id, model_id, validation_results, current_factors)
        
        else:
            logger.warning(f"Unknown calibration method: {method}, falling back to linear_scaling")
            basic_config = self.basic_calibrator.config.copy()
            basic_config["calibration_method"] = "linear_scaling"
            self.basic_calibrator.config = basic_config
            
            return self.basic_calibrator._calculate_correction_factors(validation_results)
    
    def _apply_bayesian_optimization(
        self,
        hardware_id: str,
        model_id: str,
        validation_results: List[ValidationResult],
        current_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply Bayesian optimization for parameter tuning.
        
        Args:
            hardware_id: Hardware ID
            model_id: Model ID
            validation_results: Validation results for this hardware-model combination
            current_factors: Current correction factors
            
        Returns:
            Dictionary of correction factors
        """
        try:
            # Try to import scikit-optimize for Bayesian optimization
            from skopt import gp_minimize
            from skopt.space import Real
            from sklearn.model_selection import KFold
            import numpy as np
        except ImportError:
            logger.warning("scikit-optimize not available for Bayesian optimization, falling back to regression")
            # Use regression as a fallback
            basic_config = self.basic_calibrator.config.copy()
            basic_config["calibration_method"] = "regression"
            self.basic_calibrator.config = basic_config
            
            return self.basic_calibrator._calculate_correction_factors(validation_results)
        
        correction_factors = {}
        
        # Perform Bayesian optimization for each metric
        for metric in self.config["metrics_to_calibrate"]:
            # Collect simulation and hardware values
            sim_values = []
            hw_values = []
            batch_sizes = []
            precisions = []
            
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
                    
                    # Collect additional metadata for analysis
                    if hasattr(val_result.simulation_result, "batch_size") and val_result.simulation_result.batch_size is not None:
                        batch_sizes.append(val_result.simulation_result.batch_size)
                    else:
                        batch_sizes.append(None)
                        
                    if hasattr(val_result.simulation_result, "precision") and val_result.simulation_result.precision is not None:
                        precisions.append(val_result.simulation_result.precision)
                    else:
                        precisions.append(None)
            
            if not sim_values or not hw_values:
                logger.warning(f"No valid data for metric {metric}")
                continue
            
            # Convert to numpy arrays
            X = np.array(sim_values).reshape(-1, 1)
            y = np.array(hw_values)
            
            # Set up cross-validation
            use_cross_validation = self.config.get("enable_cross_validation", True)
            n_folds = self.config.get("cross_validation_folds", 3)
            has_enough_samples = len(sim_values) >= n_folds
            
            # If cross-validation is enabled and we have enough samples, set up folds
            if use_cross_validation and has_enough_samples:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                fold_indices = list(cv.split(X))
            else:
                fold_indices = None
            
            # Define the objective function with cross-validation if available
            def objective(params):
                slope, intercept = params
                
                if fold_indices is not None:
                    # Use cross-validation
                    fold_errors = []
                    for train_idx, test_idx in fold_indices:
                        # Predict on test fold
                        X_test = X[test_idx]
                        y_test = y[test_idx]
                        y_pred = X_test * slope + intercept
                        
                        # Calculate error
                        fold_rmse = np.sqrt(np.mean((y_test - y_pred.flatten()) ** 2))
                        fold_errors.append(fold_rmse)
                    
                    # Return mean RMSE across folds
                    return np.mean(fold_errors)
                else:
                    # No cross-validation
                    y_pred = X * slope + intercept
                    rmse = np.sqrt(np.mean((y - y_pred.flatten()) ** 2))
                    return rmse
            
            # Define the parameter space
            # Adjust bounds based on the relationship - if using existing parameters
            min_slope = 0.1
            max_slope = 3.0
            min_intercept = -100
            max_intercept = 100
            
            # If we have current factors, use them to adjust the search space
            if metric in current_factors:
                current_value = current_factors[metric]
                if isinstance(current_value, (list, tuple)) and len(current_value) == 2:
                    current_slope, current_intercept = current_value
                    min_slope = max(0.1, current_slope * 0.5)
                    max_slope = min(5.0, current_slope * 1.5)
                    min_intercept = current_intercept - abs(current_intercept) * 0.5
                    max_intercept = current_intercept + abs(current_intercept) * 0.5
                elif isinstance(current_value, (int, float)):
                    # If we just have a simple scaling factor
                    current_slope = current_value
                    min_slope = max(0.1, current_slope * 0.5)
                    max_slope = min(5.0, current_slope * 1.5)
            
            space = [
                Real(min_slope, max_slope, name='slope'),
                Real(min_intercept, max_intercept, name='intercept')
            ]
            
            # Get current factors as starting point if available
            x0 = None
            if metric in current_factors:
                if isinstance(current_factors[metric], (list, tuple)) and len(current_factors[metric]) == 2:
                    x0 = current_factors[metric]
                elif isinstance(current_factors[metric], (int, float)):
                    # If we just have a simple scaling factor, use it as slope with zero intercept
                    x0 = [current_factors[metric], 0.0]
            
            # Run Bayesian optimization
            res = gp_minimize(
                objective, 
                space, 
                n_calls=self.config["bayesian_iterations"],
                x0=x0,
                random_state=42
            )
            
            # Get the best parameters
            slope, intercept = res.x
            
            # Calculate uncertainty metrics using all data
            y_pred = X * slope + intercept
            residuals = y - y_pred.flatten()
            rmse = np.sqrt(np.mean(residuals ** 2))
            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / y)) * 100 if np.all(y != 0) else None
            
            # For uncertainty quantification
            residual_std = np.std(residuals)
            prediction_interval_95 = 1.96 * residual_std  # 95% prediction interval
            
            # Additional analysis for batch sizes and precisions if available
            batch_analysis = {}
            if len(batch_sizes) > 0 and not all(bs is None for bs in batch_sizes):
                unique_batches = sorted(set(bs for bs in batch_sizes if bs is not None))
                if len(unique_batches) > 1:
                    batch_analysis = {
                        "unique_batch_sizes": unique_batches,
                        "errors_by_batch": {}
                    }
                    
                    for batch in unique_batches:
                        # Get indices for this batch size
                        batch_indices = [i for i, bs in enumerate(batch_sizes) if bs == batch]
                        
                        # Calculate error metrics for this batch
                        batch_residuals = residuals[batch_indices]
                        batch_y = y[batch_indices]
                        
                        batch_rmse = np.sqrt(np.mean(batch_residuals ** 2))
                        batch_mae = np.mean(np.abs(batch_residuals))
                        batch_mape = np.mean(np.abs(batch_residuals / batch_y)) * 100 if np.all(batch_y != 0) else None
                        
                        batch_analysis["errors_by_batch"][str(batch)] = {
                            "rmse": float(batch_rmse),
                            "mae": float(batch_mae)
                        }
                        if batch_mape is not None:
                            batch_analysis["errors_by_batch"][str(batch)]["mape"] = float(batch_mape)
            
            precision_analysis = {}
            if len(precisions) > 0 and not all(p is None for p in precisions):
                unique_precisions = sorted(set(p for p in precisions if p is not None))
                if len(unique_precisions) > 1:
                    precision_analysis = {
                        "unique_precisions": unique_precisions,
                        "errors_by_precision": {}
                    }
                    
                    for precision in unique_precisions:
                        # Get indices for this precision
                        precision_indices = [i for i, p in enumerate(precisions) if p == precision]
                        
                        # Calculate error metrics for this precision
                        precision_residuals = residuals[precision_indices]
                        precision_y = y[precision_indices]
                        
                        precision_rmse = np.sqrt(np.mean(precision_residuals ** 2))
                        precision_mae = np.mean(np.abs(precision_residuals))
                        precision_mape = np.mean(np.abs(precision_residuals / precision_y)) * 100 if np.all(precision_y != 0) else None
                        
                        precision_analysis["errors_by_precision"][precision] = {
                            "rmse": float(precision_rmse),
                            "mae": float(precision_mae)
                        }
                        if precision_mape is not None:
                            precision_analysis["errors_by_precision"][precision]["mape"] = float(precision_mape)
            
            # Create comprehensive result with parameters and uncertainty data
            result = {
                "parameters": [float(slope), float(intercept)],
                "model_type": "bayesian_optimization",
                "error_metrics": {
                    "rmse": float(rmse),
                    "mae": float(mae)
                },
                "uncertainty": {
                    "residual_std": float(residual_std),
                    "prediction_interval_95": float(prediction_interval_95)
                },
                "num_samples": len(sim_values)
            }
            
            # Add batch analysis if available
            if batch_analysis:
                result["batch_analysis"] = batch_analysis
                
            # Add precision analysis if available
            if precision_analysis:
                result["precision_analysis"] = precision_analysis
                
            # Add MAPE if available
            if mape is not None:
                result["error_metrics"]["mape"] = float(mape)
            
            # Add cross-validation information if used
            if fold_indices is not None:
                result["cross_validation"] = {
                    "folds": n_folds,
                    "used": True
                }
            
            correction_factors[metric] = result
            
            logger.info(f"Bayesian optimization for {hardware_id}/{model_id}/{metric}: "
                       f"slope={slope:.4f}, intercept={intercept:.4f}, "
                       f"RMSE={rmse:.4f}, 95% PI=±{prediction_interval_95:.4f}")
        
        return correction_factors
    
    def _apply_neural_network(
        self,
        hardware_id: str,
        model_id: str,
        validation_results: List[ValidationResult],
        current_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply neural network for complex relationships.
        
        Args:
            hardware_id: Hardware ID
            model_id: Model ID
            validation_results: Validation results for this hardware-model combination
            current_factors: Current correction factors
            
        Returns:
            Dictionary of correction factors
        """
        try:
            # Try to import scikit-learn for neural network
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score, KFold
            import numpy as np
        except ImportError:
            logger.warning("scikit-learn not available for neural network, falling back to regression")
            # Use regression as a fallback
            basic_config = self.basic_calibrator.config.copy()
            basic_config["calibration_method"] = "regression"
            self.basic_calibrator.config = basic_config
            
            return self.basic_calibrator._calculate_correction_factors(validation_results)
        
        correction_factors = {}
        
        # Create neural network for each metric
        for metric in self.config["metrics_to_calibrate"]:
            # Collect simulation and hardware values
            data = []
            
            for val_result in validation_results:
                # Check if metric exists in both simulation and hardware results
                if (metric in val_result.simulation_result.metrics and 
                    metric in val_result.hardware_result.metrics):
                    
                    sim_value = val_result.simulation_result.metrics[metric]
                    hw_value = val_result.hardware_result.metrics[metric]
                    
                    # Skip if either value is None or zero
                    if sim_value is None or hw_value is None or sim_value == 0:
                        continue
                    
                    # Collect additional features if available
                    features = [sim_value]
                    
                    if hasattr(val_result.simulation_result, "batch_size") and val_result.simulation_result.batch_size is not None:
                        features.append(float(val_result.simulation_result.batch_size))
                    
                    if hasattr(val_result.hardware_result, "test_environment"):
                        test_env = val_result.hardware_result.test_environment or {}
                        if "temperature_c" in test_env:
                            features.append(float(test_env["temperature_c"]))
                        if "background_load" in test_env:
                            features.append(float(test_env["background_load"]))
                    
                    data.append((features, hw_value))
            
            if not data:
                logger.warning(f"No valid data for metric {metric}")
                continue
            
            # Split data into features and target
            X = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create neural network model
            # Use hidden layer sizes from config
            nn = MLPRegressor(
                hidden_layer_sizes=tuple(self.config["network_hidden_layers"]),
                max_iter=self.config["network_epochs"],
                random_state=42
            )
            
            # Use cross-validation to evaluate model performance and prevent overfitting
            model_uncertainty = {}
            if self.config.get("enable_cross_validation", True) and len(data) >= self.config.get("cross_validation_folds", 3):
                cv = KFold(n_splits=self.config.get("cross_validation_folds", 3), shuffle=True, random_state=42)
                cv_scores = cross_val_score(nn, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
                
                # Calculate mean and std of MSE
                mean_mse = -np.mean(cv_scores)
                std_mse = np.std(cv_scores)
                model_uncertainty = {
                    "cross_val_mean_mse": float(mean_mse),
                    "cross_val_std_mse": float(std_mse),
                    "cross_val_coefficient_of_variation": float(std_mse / mean_mse) if mean_mse > 0 else 0.0
                }
                
                logger.info(f"Cross-validation for {hardware_id}/{model_id}/{metric}: "
                           f"MSE = {mean_mse:.4f} ± {std_mse:.4f}, "
                           f"CoV = {model_uncertainty['cross_val_coefficient_of_variation']:.4f}")
            
            # Train the final model on all data
            nn.fit(X_scaled, y)
            
            # For simplicity in the calibration application, we'll compute
            # equivalent linear parameters (slope, intercept) by fitting a line
            # to predictions across the observed range
            
            # Generate values across the observed range
            min_sim = np.min(X[:, 0])
            max_sim = np.max(X[:, 0])
            test_sim = np.linspace(min_sim, max_sim, 100).reshape(-1, 1)
            
            # Complete with mean values for other features
            if X.shape[1] > 1:
                mean_features = np.mean(X[:, 1:], axis=0)
                test_features = np.hstack([test_sim, np.tile(mean_features, (100, 1))])
            else:
                test_features = test_sim
            
            # Scale the test features
            test_features_scaled = scaler.transform(test_features)
            
            # Get predictions
            test_predictions = nn.predict(test_features_scaled)
            
            # Also get uncertainty estimates if we did cross-validation
            prediction_uncertainty = {}
            if model_uncertainty:
                # Estimate prediction uncertainty based on residuals
                residuals = nn.predict(X_scaled) - y
                residual_std = np.std(residuals)
                prediction_uncertainty = {
                    "residual_std": float(residual_std),
                    "prediction_interval_95": float(1.96 * residual_std)  # 95% prediction interval
                }
            
            # Fit a line to the predictions
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(test_sim, test_predictions)
            
            slope = lr.coef_[0]
            intercept = lr.intercept_
            
            # Store correction factors as [slope, intercept] with uncertainty information
            result = {
                "parameters": [float(slope), float(intercept)],
                "model_type": "neural_network",
                "num_samples": len(data)
            }
            
            # Add uncertainty information if available
            if model_uncertainty:
                result["model_uncertainty"] = model_uncertainty
            
            if prediction_uncertainty:
                result["prediction_uncertainty"] = prediction_uncertainty
            
            correction_factors[metric] = result
            
            logger.info(f"Neural network for {hardware_id}/{model_id}/{metric}: "
                       f"approximated as slope={slope:.4f}, intercept={intercept:.4f}")
        
        return correction_factors
    
    def _apply_incremental_learning(
        self,
        hardware_id: str,
        model_id: str,
        validation_results: List[ValidationResult],
        current_factors: Dict[str, Any],
        current_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply incremental learning with trend analysis.
        
        Args:
            hardware_id: Hardware ID
            model_id: Model ID
            validation_results: Validation results for this hardware-model combination
            current_factors: Current correction factors
            current_parameters: Current simulation parameters
            
        Returns:
            Dictionary of correction factors
        """
        correction_factors = {}
        
        # Get calibration history
        calibration_history = current_parameters.get("calibration_history", [])
        
        # Perform incremental learning for each metric
        for metric in self.config["metrics_to_calibrate"]:
            # Calculate base correction factor using linear scaling
            basic_config = self.basic_calibrator.config.copy()
            basic_config["calibration_method"] = "linear_scaling"
            self.basic_calibrator.config = basic_config
            
            base_factors = self.basic_calibrator._calculate_correction_factors(validation_results)
            
            if metric not in base_factors:
                continue
            
            base_factor = base_factors[metric]
            
            # Calculate trend if we have history
            trend_factor = 0.0
            if len(calibration_history) > 1 and metric in current_factors:
                # Get previous incremental factors
                previous_factors = []
                for i, event in enumerate(calibration_history[:-1]):  # Skip most recent
                    if (i > 0 and  # Skip the first one since we need at least two for trend
                        "hardware_models" in event and
                        (hardware_id, model_id) in event["hardware_models"]):
                        
                        # Find corresponding factor in history
                        if (hardware_id in current_parameters.get("correction_factors", {}) and
                            model_id in current_parameters["correction_factors"].get(hardware_id, {}) and
                            metric in current_parameters["correction_factors"][hardware_id][model_id]):
                            
                            factor = current_parameters["correction_factors"][hardware_id][model_id][metric]
                            
                            if isinstance(factor, dict) and "base_factor" in factor:
                                previous_factors.append(factor["base_factor"])
                            else:
                                previous_factors.append(factor)
                
                if len(previous_factors) >= 2:
                    # Calculate trend as the average change per calibration
                    changes = [previous_factors[i] - previous_factors[i-1] for i in range(1, len(previous_factors))]
                    trend_factor = sum(changes) / len(changes)
                    
                    logger.info(f"Calculated trend for {hardware_id}/{model_id}/{metric}: {trend_factor:.6f} per calibration")
            
            # Store as an incremental factor
            correction_factors[metric] = {
                "base_factor": base_factor,
                "trend_factor": trend_factor,
                "recent_samples": len(validation_results),
                "calibration_timestamp": datetime.now().isoformat()
            }
        
        return correction_factors
    
    def _apply_ensemble_methods(
        self,
        hardware_id: str,
        model_id: str,
        validation_results: List[ValidationResult],
        current_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply ensemble of multiple calibration methods.
        
        Args:
            hardware_id: Hardware ID
            model_id: Model ID
            validation_results: Validation results for this hardware-model combination
            current_factors: Current correction factors
            
        Returns:
            Dictionary of correction factors
        """
        correction_factors = {}
        
        # Get ensemble weights
        ensemble_weights = self.config["ensemble_weights"]
        
        # Apply each method and collect factors
        methods = ["linear_scaling", "regression"]
        
        # Add advanced methods if available
        try:
            import sklearn
            methods.extend(["bayesian", "network"])
        except ImportError:
            logger.warning("sklearn not available, ensemble will only use basic methods")
        
        # Collect factors for each method and metric
        for metric in self.config["metrics_to_calibrate"]:
            ensemble_factors = {}
            
            for method in methods:
                if method == "linear_scaling":
                    # Use basic calibrator for linear scaling
                    basic_config = self.basic_calibrator.config.copy()
                    basic_config["calibration_method"] = "linear_scaling"
                    self.basic_calibrator.config = basic_config
                    
                    factors = self.basic_calibrator._calculate_correction_factors(validation_results)
                    if metric in factors:
                        ensemble_factors["linear_scaling"] = factors[metric]
                
                elif method == "regression":
                    # Use basic calibrator with regression method
                    basic_config = self.basic_calibrator.config.copy()
                    basic_config["calibration_method"] = "regression"
                    self.basic_calibrator.config = basic_config
                    
                    factors = self.basic_calibrator._calculate_correction_factors(validation_results)
                    if metric in factors:
                        ensemble_factors["regression"] = factors[metric]
                
                elif method == "bayesian":
                    # Apply Bayesian optimization for this metric only
                    method_factors = self._apply_bayesian_optimization(
                        hardware_id, model_id, validation_results, {metric: current_factors.get(metric, 1.0)}
                    )
                    if metric in method_factors:
                        ensemble_factors["bayesian"] = method_factors[metric]
                
                elif method == "network":
                    # Apply neural network for this metric only
                    method_factors = self._apply_neural_network(
                        hardware_id, model_id, validation_results, {metric: current_factors.get(metric, 1.0)}
                    )
                    if metric in method_factors:
                        ensemble_factors["network"] = method_factors[metric]
            
            if not ensemble_factors:
                logger.warning(f"No ensemble factors could be calculated for {metric}")
                continue
            
            # Calculate weighted average for simple factors
            simple_factors = {}
            for method, factor in ensemble_factors.items():
                if not isinstance(factor, (list, tuple, dict)):
                    simple_factors[method] = factor
            
            if simple_factors:
                weighted_simple = 0
                weight_sum = 0
                
                for method, factor in simple_factors.items():
                    weight = ensemble_weights.get(method, 1.0)
                    weighted_simple += factor * weight
                    weight_sum += weight
                
                if weight_sum > 0:
                    weighted_simple /= weight_sum
                    
                    # Use as base factor
                    ensemble_factors["weighted_simple"] = weighted_simple
            
            # Store ensemble factors
            correction_factors[metric] = {
                "ensemble_factors": ensemble_factors,
                "ensemble_weights": {k: v for k, v in ensemble_weights.items() if k in ensemble_factors}
            }
        
        return correction_factors
    
    def _check_linear_fit_quality(self, validation_results: List[ValidationResult]) -> float:
        """
        Check the quality of linear fits for the data.
        
        Args:
            validation_results: Validation results
            
        Returns:
            Average R^2 score across metrics
        """
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np
            from sklearn.metrics import r2_score
        except ImportError:
            logger.warning("sklearn not available for linear fit quality check")
            return 0.8  # Assume moderate fit
        
        r2_scores = []
        
        for metric in self.config["metrics_to_calibrate"]:
            # Collect simulation and hardware values
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
            
            if len(sim_values) < 3:
                continue
            
            # Fit linear model
            X = np.array(sim_values).reshape(-1, 1)
            y = np.array(hw_values)
            
            model = LinearRegression().fit(X, y)
            
            # Calculate R^2 score
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            r2_scores.append(r2)
        
        if not r2_scores:
            return 0.8  # Assume moderate fit
        
        return np.mean(r2_scores)
    
    def _initialize_hardware_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize hardware-specific profiles for calibration.
        
        Returns:
            Dictionary of hardware profiles
        """
        # Default profiles for common hardware types
        profiles = {
            # CPU profiles
            "cpu_intel_xeon": {
                "preferred_calibration_method": "regression",
                "metric_adjustments": {
                    "throughput_items_per_second": {
                        "batch_size_factors": {"1": 1.0, "4": 0.9, "8": 0.85, "16": 0.8, "32": 0.75},
                        "precision_factors": {"fp32": 1.0, "fp16": 1.2}
                    },
                    "memory_peak_mb": {
                        "batch_size_factors": {"1": 1.0, "4": 0.95, "8": 0.9, "16": 0.85, "32": 0.8},
                        "precision_factors": {"fp32": 1.0, "fp16": 0.6}
                    }
                }
            },
            "cpu_amd_epyc": {
                "preferred_calibration_method": "regression",
                "metric_adjustments": {
                    "throughput_items_per_second": {
                        "batch_size_factors": {"1": 1.0, "4": 0.92, "8": 0.88, "16": 0.84, "32": 0.8},
                        "precision_factors": {"fp32": 1.0, "fp16": 1.15}
                    }
                }
            },
            
            # GPU profiles
            "gpu_rtx3080": {
                "preferred_calibration_method": "ensemble",
                "metric_adjustments": {
                    "throughput_items_per_second": {
                        "batch_size_factors": {"1": 1.0, "4": 1.3, "8": 1.5, "16": 1.7, "32": 1.8},
                        "precision_factors": {"fp32": 1.0, "fp16": 1.8}
                    },
                    "memory_peak_mb": {
                        "precision_factors": {"fp32": 1.0, "fp16": 0.55}
                    }
                }
            },
            "gpu_a100": {
                "preferred_calibration_method": "ensemble",
                "metric_adjustments": {
                    "throughput_items_per_second": {
                        "batch_size_factors": {"1": 1.0, "4": 1.4, "8": 1.7, "16": 1.9, "32": 2.1},
                        "precision_factors": {"fp32": 1.0, "fp16": 1.9, "int8": 3.2}
                    }
                }
            },
            
            # WebGPU profiles
            "webgpu_chrome": {
                "preferred_calibration_method": "incremental",
                "metric_adjustments": {
                    "throughput_items_per_second": {
                        "batch_size_factors": {"1": 1.0, "4": 1.2, "8": 1.3, "16": 1.4},
                        "precision_factors": {"fp32": 1.0, "fp16": 1.6}
                    }
                }
            }
        }
        
        return profiles
    
    def _calculate_enhanced_error_statistics(
        self,
        validation_results: List[ValidationResult],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate enhanced error statistics for validation results.
        
        Args:
            validation_results: List of validation results
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary mapping metric names to error statistics
        """
        import numpy as np
        from scipy.stats import pearsonr, spearmanr
        
        statistics = {}
        
        for metric in metrics:
            # Collect pairs of simulation and hardware values
            pairs = []
            
            for val_result in validation_results:
                if (metric in val_result.simulation_result.metrics and 
                    metric in val_result.hardware_result.metrics):
                    
                    sim_value = val_result.simulation_result.metrics[metric]
                    hw_value = val_result.hardware_result.metrics[metric]
                    
                    if sim_value is not None and hw_value is not None:
                        pairs.append((sim_value, hw_value))
            
            if not pairs:
                continue
            
            # Split into simulation and hardware values
            sim_values = [p[0] for p in pairs]
            hw_values = [p[1] for p in pairs]
            
            # Calculate various error metrics
            
            # Mean Absolute Percentage Error (MAPE)
            mapes = [abs(hw - sim) / abs(hw) * 100 for sim, hw in pairs if hw != 0]
            mape = np.mean(mapes) if mapes else np.nan
            
            # Root Mean Square Error (RMSE)
            se = [(hw - sim) ** 2 for sim, hw in pairs]
            rmse = np.sqrt(np.mean(se)) if se else np.nan
            
            # Correlation coefficient
            correlation = None
            if len(pairs) >= 3:
                try:
                    correlation = pearsonr(sim_values, hw_values)[0]
                except:
                    correlation = np.nan
            else:
                correlation = np.nan
            
            # Rank preservation (Spearman correlation)
            rank_preservation = None
            if len(pairs) >= 3:
                try:
                    rank_preservation = spearmanr(sim_values, hw_values)[0]
                except:
                    rank_preservation = np.nan
            else:
                rank_preservation = np.nan
            
            # Store statistics
            statistics[metric] = {
                "mape": mape,
                "rmse": rmse,
                "correlation": correlation,
                "rank_preservation": rank_preservation,
                "count": len(pairs),
                "min_sim": min(sim_values) if sim_values else np.nan,
                "max_sim": max(sim_values) if sim_values else np.nan,
                "min_hw": min(hw_values) if hw_values else np.nan,
                "max_hw": max(hw_values) if hw_values else np.nan,
                "mean_sim": np.mean(sim_values) if sim_values else np.nan,
                "mean_hw": np.mean(hw_values) if hw_values else np.nan,
                "std_sim": np.std(sim_values) if sim_values else np.nan,
                "std_hw": np.std(hw_values) if hw_values else np.nan
            }
        
        return statistics
    
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