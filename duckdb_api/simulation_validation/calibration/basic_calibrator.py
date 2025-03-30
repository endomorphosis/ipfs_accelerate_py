"""
Basic calibration implementation for the Simulation Accuracy and Validation Framework.

This module provides a simple calibration system that adjusts simulation 
parameters based on observed errors between simulation and real hardware results.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional

# Setup logger
logger = logging.getLogger(__name__)

class BasicCalibrator:
    """
    Basic implementation of a calibration system for simulation parameters.
    
    This calibrator uses simple scaling and offset adjustments to improve 
    simulation accuracy based on observed errors between simulation and 
    real hardware measurements.
    """
    
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 100):
        """
        Initialize the BasicCalibrator.
        
        Args:
            learning_rate: Learning rate for parameter adjustments (default: 0.1)
            max_iterations: Maximum number of calibration iterations (default: 100)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        logger.info(f"Initialized BasicCalibrator with learning_rate={learning_rate}, "
                   f"max_iterations={max_iterations}")
    
    def calibrate(self, 
                 simulation_results: List[Dict[str, Any]], 
                 hardware_results: List[Dict[str, Any]],
                 simulation_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calibrate simulation parameters based on observed errors.
        
        Args:
            simulation_results: List of simulation result dictionaries
            hardware_results: List of hardware result dictionaries
            simulation_parameters: Current simulation parameters
            
        Returns:
            Updated simulation parameters
        """
        logger.info("Starting basic calibration process")
        
        if not simulation_results or not hardware_results:
            logger.warning("Empty simulation or hardware results provided")
            return simulation_parameters
        
        # Create a copy of the parameters to modify
        updated_parameters = simulation_parameters.copy()
        
        # Extract metrics from results
        sim_metrics = self._extract_metrics(simulation_results)
        hw_metrics = self._extract_metrics(hardware_results)
        
        # Perform calibration for each metric
        for metric_name in set(sim_metrics.keys()).intersection(hw_metrics.keys()):
            logger.info(f"Calibrating for metric: {metric_name}")
            
            sim_values = sim_metrics[metric_name]
            hw_values = hw_metrics[metric_name]
            
            # Calculate scaling factor and offset for this metric
            scaling, offset = self._calculate_adjustments(sim_values, hw_values)
            
            # Update parameters related to this metric
            param_key = f"{metric_name}_scale"
            if param_key in updated_parameters:
                updated_parameters[param_key] *= scaling
                logger.info(f"Updated {param_key}: {updated_parameters[param_key]}")
            
            param_key = f"{metric_name}_offset"
            if param_key in updated_parameters:
                updated_parameters[param_key] += offset
                logger.info(f"Updated {param_key}: {updated_parameters[param_key]}")
        
        # Apply general calibration parameters (if any)
        self._apply_general_calibration(updated_parameters, sim_metrics, hw_metrics)
        
        logger.info("Basic calibration completed")
        return updated_parameters
    
    def _extract_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Extract metrics from result dictionaries.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary mapping metric names to lists of values
        """
        metrics = {}
        
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not key.startswith("_"):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(float(value))
        
        return metrics
    
    def _calculate_adjustments(self, 
                             sim_values: List[float], 
                             hw_values: List[float]) -> Tuple[float, float]:
        """
        Calculate scaling factor and offset adjustments based on observed errors.
        
        Args:
            sim_values: List of simulation metric values
            hw_values: List of hardware metric values
            
        Returns:
            Tuple of (scaling_factor, offset)
        """
        if len(sim_values) != len(hw_values):
            logger.warning(f"Length mismatch: sim_values={len(sim_values)}, hw_values={len(hw_values)}")
            # Use the minimum length
            length = min(len(sim_values), len(hw_values))
            sim_values = sim_values[:length]
            hw_values = hw_values[:length]
        
        if not sim_values:
            logger.warning("No values to calculate adjustments")
            return 1.0, 0.0
        
        # Convert to numpy arrays for easier calculations
        sim_array = np.array(sim_values)
        hw_array = np.array(hw_values)
        
        # Calculate mean values
        sim_mean = np.mean(sim_array)
        hw_mean = np.mean(hw_array)
        
        # Calculate scaling factor (avoid division by zero)
        if abs(sim_mean) < 1e-10:
            scaling = 1.0
        else:
            scaling = hw_mean / sim_mean
        
        # Limit scaling factor to reasonable range
        scaling = max(0.1, min(10.0, scaling))
        
        # Calculate offset
        offset = hw_mean - (sim_mean * scaling)
        
        logger.info(f"Calculated scaling={scaling:.4f}, offset={offset:.4f}")
        return scaling, offset
    
    def _apply_general_calibration(self, 
                                 parameters: Dict[str, Any],
                                 sim_metrics: Dict[str, List[float]],
                                 hw_metrics: Dict[str, List[float]]) -> None:
        """
        Apply general calibration parameters that affect multiple metrics.
        
        Args:
            parameters: Parameters dictionary to update
            sim_metrics: Dictionary of simulation metrics
            hw_metrics: Dictionary of hardware metrics
        """
        # Example: Adjust global scale factor based on overall error
        if "global_scale" in parameters:
            total_sim = sum(np.mean(values) for values in sim_metrics.values())
            total_hw = sum(np.mean(values) for values in hw_metrics.values())
            
            if abs(total_sim) > 1e-10:
                global_scale = total_hw / total_sim
                # Apply with learning rate and limit to reasonable range
                adjusted_scale = parameters["global_scale"] * (1.0 + self.learning_rate * (global_scale - 1.0))
                parameters["global_scale"] = max(0.1, min(10.0, adjusted_scale))
                logger.info(f"Updated global_scale: {parameters['global_scale']}")
    
    def evaluate_calibration(self, 
                           pre_calibration_results: List[Dict[str, Any]],
                           post_calibration_results: List[Dict[str, Any]],
                           hardware_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of calibration.
        
        Args:
            pre_calibration_results: Simulation results before calibration
            post_calibration_results: Simulation results after calibration
            hardware_results: Real hardware results
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating calibration effectiveness")
        
        # Extract metrics
        pre_metrics = self._extract_metrics(pre_calibration_results)
        post_metrics = self._extract_metrics(post_calibration_results)
        hw_metrics = self._extract_metrics(hardware_results)
        
        # Calculate errors for each metric
        evaluation = {
            "metrics": {},
            "overall": {}
        }
        
        for metric_name in set(pre_metrics.keys()).intersection(post_metrics.keys()).intersection(hw_metrics.keys()):
            pre_values = np.array(pre_metrics[metric_name])
            post_values = np.array(post_metrics[metric_name])
            hw_values = np.array(hw_metrics[metric_name])
            
            # Calculate Mean Absolute Percentage Error (MAPE)
            pre_mape = self._calculate_mape(pre_values, hw_values)
            post_mape = self._calculate_mape(post_values, hw_values)
            
            # Calculate improvement
            improvement = (pre_mape - post_mape) / pre_mape * 100 if pre_mape > 0 else 0
            
            evaluation["metrics"][metric_name] = {
                "pre_calibration_mape": pre_mape,
                "post_calibration_mape": post_mape,
                "improvement_percentage": improvement
            }
        
        # Calculate overall improvement
        if evaluation["metrics"]:
            avg_pre_mape = np.mean([m["pre_calibration_mape"] for m in evaluation["metrics"].values()])
            avg_post_mape = np.mean([m["post_calibration_mape"] for m in evaluation["metrics"].values()])
            overall_improvement = (avg_pre_mape - avg_post_mape) / avg_pre_mape * 100 if avg_pre_mape > 0 else 0
            
            evaluation["overall"] = {
                "avg_pre_calibration_mape": avg_pre_mape,
                "avg_post_calibration_mape": avg_post_mape,
                "overall_improvement_percentage": overall_improvement
            }
        
        logger.info(f"Calibration evaluation completed: {evaluation['overall']}")
        return evaluation
    
    def _calculate_mape(self, sim_values: np.ndarray, hw_values: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            sim_values: Array of simulation values
            hw_values: Array of hardware values
            
        Returns:
            MAPE value
        """
        # Avoid division by zero
        mask = hw_values != 0
        if not np.any(mask):
            return 0.0
        
        # Calculate MAPE only for non-zero hardware values
        mape = np.mean(np.abs((hw_values[mask] - sim_values[mask]) / hw_values[mask])) * 100
        return mape