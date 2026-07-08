"""
Advanced calibration implementation for the Simulation Accuracy and Validation Framework.

This module provides sophisticated calibration systems that use various optimization 
techniques to improve simulation accuracy through parameter tuning.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime
import uuid
import warnings

# Import basic calibrator as a fallback
from .basic_calibrator import BasicCalibrator

# Setup logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import scipy
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Some advanced calibration features will be disabled.")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Bayesian optimization calibration will be disabled.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Neural network calibration will be disabled.")

# Base class for all advanced calibrators
class AdvancedCalibrator(ABC):
    """
    Abstract base class for advanced calibration implementations.
    
    This class defines the interface for all advanced calibration methods.
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.1, 
        max_iterations: int = 100,
        history_file: Optional[str] = None,
        fallback_to_basic: bool = True
    ):
        """
        Initialize the AdvancedCalibrator.
        
        Args:
            learning_rate: Learning rate for parameter adjustments
            max_iterations: Maximum number of calibration iterations
            history_file: Optional path to file for storing calibration history
            fallback_to_basic: Whether to fall back to BasicCalibrator if advanced methods fail
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.history_file = history_file
        self.fallback_to_basic = fallback_to_basic
        self.history = []
        self.basic_calibrator = BasicCalibrator(learning_rate, max_iterations)
        self._load_history()
        
        logger.info(f"Initialized {self.__class__.__name__} with learning_rate={learning_rate}, "
                   f"max_iterations={max_iterations}")
    
    def calibrate(
        self, 
        simulation_results: List[Dict[str, Any]], 
        hardware_results: List[Dict[str, Any]],
        simulation_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Calibrate simulation parameters based on observed errors.
        
        Args:
            simulation_results: List of simulation result dictionaries
            hardware_results: List of hardware result dictionaries
            simulation_parameters: Current simulation parameters
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
            
        Returns:
            Updated simulation parameters
        """
        logger.info(f"Starting {self.__class__.__name__} calibration")
        
        if not simulation_results or not hardware_results:
            logger.warning("Empty simulation or hardware results provided")
            return simulation_parameters
        
        try:
            # Extract metrics from results
            sim_metrics = self._extract_metrics(simulation_results)
            hw_metrics = self._extract_metrics(hardware_results)
            
            # Perform advanced calibration
            updated_parameters = self._perform_calibration(
                sim_metrics, hw_metrics, simulation_parameters, parameter_bounds
            )
            
            # Add calibration event to history
            self._add_to_history(simulation_parameters, updated_parameters, sim_metrics, hw_metrics)
            
            logger.info(f"{self.__class__.__name__} calibration completed successfully")
            return updated_parameters
            
        except Exception as e:
            logger.error(f"Error during advanced calibration: {str(e)}")
            if self.fallback_to_basic:
                logger.info("Falling back to basic calibration")
                return self.basic_calibrator.calibrate(
                    simulation_results, hardware_results, simulation_parameters
                )
            else:
                # If no fallback, return original parameters
                logger.info("Returning original parameters due to calibration failure")
                return simulation_parameters
    
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
    
    @abstractmethod
    def _perform_calibration(
        self, 
        sim_metrics: Dict[str, List[float]], 
        hw_metrics: Dict[str, List[float]],
        simulation_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Perform advanced calibration (to be implemented by subclasses).
        
        Args:
            sim_metrics: Dictionary mapping metric names to lists of simulation values
            hw_metrics: Dictionary mapping metric names to lists of hardware values
            simulation_parameters: Current simulation parameters
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
            
        Returns:
            Updated simulation parameters
        """
        pass
    
    def evaluate_calibration(
        self, 
        pre_calibration_results: List[Dict[str, Any]],
        post_calibration_results: List[Dict[str, Any]],
        hardware_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of calibration.
        
        Args:
            pre_calibration_results: Simulation results before calibration
            post_calibration_results: Simulation results after calibration
            hardware_results: Real hardware results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Use the basic calibrator's evaluation method
        return self.basic_calibrator.evaluate_calibration(
            pre_calibration_results, post_calibration_results, hardware_results
        )
    
    def _add_to_history(
        self, 
        original_parameters: Dict[str, Any],
        updated_parameters: Dict[str, Any],
        sim_metrics: Dict[str, List[float]],
        hw_metrics: Dict[str, List[float]]
    ) -> None:
        """
        Add a calibration event to history.
        
        Args:
            original_parameters: Parameters before calibration
            updated_parameters: Parameters after calibration
            sim_metrics: Simulation metrics
            hw_metrics: Hardware metrics
        """
        if not self.history_file:
            return
            
        # Create a calibration event record
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "calibrator": self.__class__.__name__,
            "original_parameters": original_parameters,
            "updated_parameters": updated_parameters,
            "parameter_changes": {
                k: {
                    "original": original_parameters.get(k),
                    "updated": updated_parameters.get(k),
                    "diff": updated_parameters.get(k) - original_parameters.get(k) 
                        if isinstance(original_parameters.get(k), (int, float)) and 
                           isinstance(updated_parameters.get(k), (int, float)) else None
                }
                for k in set(original_parameters.keys()).union(updated_parameters.keys())
            },
            "metrics_summary": {
                "sim_metrics_count": {k: len(v) for k, v in sim_metrics.items()},
                "hw_metrics_count": {k: len(v) for k, v in hw_metrics.items()},
                "sim_metrics_mean": {k: float(np.mean(v)) for k, v in sim_metrics.items()},
                "hw_metrics_mean": {k: float(np.mean(v)) for k, v in hw_metrics.items()}
            }
        }
        
        # Add to history
        self.history.append(event)
        
        # Save history to file
        self._save_history()
    
    def _load_history(self) -> None:
        """Load calibration history from file if it exists."""
        if not self.history_file:
            return
            
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} calibration events from history")
        except Exception as e:
            logger.error(f"Error loading calibration history: {str(e)}")
            self.history = []
    
    def _save_history(self) -> None:
        """Save calibration history to file."""
        if not self.history_file:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.history_file)), exist_ok=True)
            
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
                
            logger.info(f"Saved {len(self.history)} calibration events to history")
        except Exception as e:
            logger.error(f"Error saving calibration history: {str(e)}")
    
    def get_calibration_history(self) -> List[Dict[str, Any]]:
        """
        Get the calibration history.
        
        Returns:
            List of calibration event dictionaries
        """
        return self.history
    
    def analyze_history_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in calibration history.
        
        Returns:
            Dictionary with trend analysis results
        """
        if not self.history:
            return {"error": "No calibration history available"}
        
        try:
            # Get list of parameters that have been calibrated
            all_params = set()
            for event in self.history:
                all_params.update(event.get("parameter_changes", {}).keys())
            
            # Track parameter changes over time
            param_trends = {}
            for param in all_params:
                values = []
                timestamps = []
                
                for event in self.history:
                    changes = event.get("parameter_changes", {}).get(param, {})
                    if "updated" in changes and changes["updated"] is not None:
                        values.append(changes["updated"])
                        timestamps.append(event["timestamp"])
                
                if values:
                    param_trends[param] = {
                        "values": values,
                        "timestamps": timestamps,
                        "trend": "stable" if len(values) < 3 else self._detect_trend(values)
                    }
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "num_calibration_events": len(self.history),
                "parameter_trends": param_trends,
                "summary": {
                    "increasing_params": [p for p, data in param_trends.items() 
                                         if data["trend"] == "increasing"],
                    "decreasing_params": [p for p, data in param_trends.items() 
                                         if data["trend"] == "decreasing"],
                    "stable_params": [p for p, data in param_trends.items() 
                                     if data["trend"] == "stable"],
                    "oscillating_params": [p for p, data in param_trends.items() 
                                          if data["trend"] == "oscillating"]
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing calibration history trends: {str(e)}")
            return {"error": f"Error analyzing trends: {str(e)}"}
    
    def _detect_trend(self, values: List[float]) -> str:
        """
        Detect trend in a series of values.
        
        Args:
            values: List of parameter values over time
            
        Returns:
            Trend description: "increasing", "decreasing", "stable", or "oscillating"
        """
        if len(values) < 3:
            return "stable"
        
        # Convert to numpy array
        vals = np.array(values)
        
        # Calculate differences between consecutive values
        diffs = np.diff(vals)
        
        # Check for consistent sign in differences
        if np.all(diffs > 0):
            return "increasing"
        elif np.all(diffs < 0):
            return "decreasing"
        
        # Check for oscillation (sign changes)
        sign_changes = np.sum(np.diff(np.signbit(diffs)))
        if sign_changes > len(diffs) / 3:  # If more than 1/3 of differences change sign
            return "oscillating"
        
        # Default to stable
        return "stable"
    
    def clear_history(self) -> None:
        """Clear the calibration history."""
        self.history = []
        if self.history_file and os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
                logger.info(f"Removed history file: {self.history_file}")
            except Exception as e:
                logger.error(f"Error removing history file: {str(e)}")


# Implementation of multi-parameter optimization calibrator
class MultiParameterCalibrator(AdvancedCalibrator):
    """
    Advanced calibrator that optimizes multiple parameters simultaneously.
    
    Uses numerical optimization techniques to find the optimal parameter values
    that minimize the error between simulation and hardware results.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the MultiParameterCalibrator.
        
        Args:
            **kwargs: Arguments to pass to AdvancedCalibrator.__init__
        """
        super().__init__(**kwargs)
        
        # Check for SciPy availability
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available. MultiParameterCalibrator will use basic methods.")
    
    def _perform_calibration(
        self, 
        sim_metrics: Dict[str, List[float]], 
        hw_metrics: Dict[str, List[float]],
        simulation_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Perform multi-parameter optimization for calibration.
        
        Args:
            sim_metrics: Dictionary mapping metric names to lists of simulation values
            hw_metrics: Dictionary mapping metric names to lists of hardware values
            simulation_parameters: Current simulation parameters
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
            
        Returns:
            Updated simulation parameters
        """
        # Create a copy of parameters to modify
        updated_parameters = simulation_parameters.copy()
        
        # Filter parameters to optimize (only numeric parameters)
        params_to_optimize = [k for k, v in updated_parameters.items() 
                             if isinstance(v, (int, float))]
        
        if not params_to_optimize:
            logger.warning("No numeric parameters found for optimization")
            return updated_parameters
        
        logger.info(f"Optimizing {len(params_to_optimize)} parameters: {params_to_optimize}")
        
        # Setup parameter bounds if not provided
        if parameter_bounds is None:
            parameter_bounds = {}
        
        bounds = []
        for param in params_to_optimize:
            if param in parameter_bounds:
                bounds.append(parameter_bounds[param])
            else:
                # Default bounds: 0.1x to 10x current value, or -10 to +10 if near zero
                current_value = updated_parameters[param]
                if abs(current_value) < 1e-6:
                    bounds.append((-10.0, 10.0))
                else:
                    bounds.append((0.1 * current_value, 10.0 * current_value))
        
        # Initial parameter values
        initial_values = [updated_parameters[param] for param in params_to_optimize]
        
        # Define the error function to minimize
        def error_function(params):
            # Create a temporary parameter set with the current optimization values
            temp_params = updated_parameters.copy()
            for i, param in enumerate(params_to_optimize):
                temp_params[param] = params[i]
            
            # Calculate the error between simulation and hardware results
            total_error = 0.0
            for metric in set(sim_metrics.keys()).intersection(hw_metrics.keys()):
                # Apply the current parameters to adjust simulation values
                adjusted_sim_values = self._apply_parameters_to_metric(
                    sim_metrics[metric], temp_params, metric)
                
                # Calculate Mean Squared Error
                hw_values = np.array(hw_metrics[metric])
                mse = np.mean((np.array(adjusted_sim_values) - hw_values) ** 2)
                total_error += mse
            
            return total_error
        
        try:
            if SCIPY_AVAILABLE:
                # Use SciPy for optimization
                logger.info("Using SciPy optimization")
                result = scipy.optimize.minimize(
                    error_function,
                    initial_values,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self.max_iterations}
                )
                
                # Update parameters with optimized values
                for i, param in enumerate(params_to_optimize):
                    updated_parameters[param] = result.x[i]
                    logger.info(f"Optimized {param}: {updated_parameters[param]}")
                
                logger.info(f"Optimization completed with final error: {result.fun}")
            else:
                # Fallback to simple grid search
                logger.info("Using simple grid search optimization")
                self._grid_search_optimization(
                    error_function, params_to_optimize, updated_parameters, 
                    initial_values, bounds
                )
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            # Keep original parameters for the ones that failed
            logger.info("Keeping original parameter values due to optimization failure")
        
        return updated_parameters
    
    def _apply_parameters_to_metric(
        self, 
        sim_values: List[float], 
        parameters: Dict[str, Any], 
        metric_name: str
    ) -> List[float]:
        """
        Apply parameters to adjust simulation values for a metric.
        
        Args:
            sim_values: List of simulation values for the metric
            parameters: Current parameter dictionary
            metric_name: Name of the metric
            
        Returns:
            List of adjusted simulation values
        """
        adjusted_values = np.array(sim_values).copy()
        
        # Apply metric-specific parameters
        scale_param = f"{metric_name}_scale"
        offset_param = f"{metric_name}_offset"
        
        if scale_param in parameters:
            adjusted_values *= parameters[scale_param]
        
        if offset_param in parameters:
            adjusted_values += parameters[offset_param]
        
        # Apply global parameters
        if "global_scale" in parameters:
            adjusted_values *= parameters["global_scale"]
        
        if "global_offset" in parameters:
            adjusted_values += parameters["global_offset"]
        
        return adjusted_values.tolist()
    
    def _grid_search_optimization(
        self, 
        error_function: Callable, 
        params_to_optimize: List[str],
        parameters: Dict[str, Any],
        initial_values: List[float],
        bounds: List[Tuple[float, float]]
    ) -> None:
        """
        Perform a simple grid search optimization.
        
        Args:
            error_function: Function to minimize
            params_to_optimize: List of parameter names to optimize
            parameters: Dictionary of parameters to update
            initial_values: Initial parameter values
            bounds: List of (min, max) bounds for each parameter
        """
        # Start with initial error
        best_params = initial_values.copy()
        best_error = error_function(best_params)
        
        logger.info(f"Initial error: {best_error}")
        
        # Number of steps per dimension (limited by max_iterations)
        n_dims = len(params_to_optimize)
        steps_per_dim = max(2, min(5, int(self.max_iterations ** (1/n_dims))))
        
        logger.info(f"Grid search with {steps_per_dim} steps per dimension")
        
        # Simple grid search
        for iteration in range(self.max_iterations):
            # Randomly select a parameter to adjust
            param_idx = np.random.randint(0, len(params_to_optimize))
            param_name = params_to_optimize[param_idx]
            
            # Current value and bounds
            current_value = best_params[param_idx]
            lower_bound, upper_bound = bounds[param_idx]
            
            # Try different values within bounds
            step_size = (upper_bound - lower_bound) / steps_per_dim
            
            for step in range(steps_per_dim + 1):
                # Calculate new value
                new_value = lower_bound + step * step_size
                
                # Skip if it's the current value
                if abs(new_value - current_value) < 1e-6:
                    continue
                
                # Create new parameter set
                new_params = best_params.copy()
                new_params[param_idx] = new_value
                
                # Calculate error
                new_error = error_function(new_params)
                
                # Update if better
                if new_error < best_error:
                    best_params = new_params.copy()
                    best_error = new_error
                    logger.info(f"Improved error to {best_error} with {param_name}={new_value}")
        
        # Update parameters with the best values
        for i, param in enumerate(params_to_optimize):
            parameters[param] = best_params[i]
            logger.info(f"Optimized {param}: {parameters[param]}")
        
        logger.info(f"Grid search completed with final error: {best_error}")


# Bayesian Optimization Calibrator
class BayesianOptimizationCalibrator(AdvancedCalibrator):
    """
    Advanced calibrator that uses Bayesian Optimization for parameter tuning.
    
    Uses Gaussian Process Regression to build a surrogate model of the error
    function and efficiently explores the parameter space to find optimal values.
    """
    
    def __init__(self, n_initial_points: int = 5, exploration_factor: float = 0.1, **kwargs):
        """
        Initialize the BayesianOptimizationCalibrator.
        
        Args:
            n_initial_points: Number of initial random points to evaluate
            exploration_factor: Factor controlling exploration vs. exploitation (0-1)
            **kwargs: Arguments to pass to AdvancedCalibrator.__init__
        """
        super().__init__(**kwargs)
        self.n_initial_points = n_initial_points
        self.exploration_factor = exploration_factor
        
        # Check for scikit-learn availability
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. BayesianOptimizationCalibrator will use fallback methods.")
    
    def _perform_calibration(
        self, 
        sim_metrics: Dict[str, List[float]], 
        hw_metrics: Dict[str, List[float]],
        simulation_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization for calibration.
        
        Args:
            sim_metrics: Dictionary mapping metric names to lists of simulation values
            hw_metrics: Dictionary mapping metric names to lists of hardware values
            simulation_parameters: Current simulation parameters
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
            
        Returns:
            Updated simulation parameters
        """
        # If scikit-learn is not available, fall back to MultiParameterCalibrator
        if not SKLEARN_AVAILABLE:
            logger.info("Falling back to MultiParameterCalibrator due to missing scikit-learn")
            return MultiParameterCalibrator(
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations,
                history_file=self.history_file,
                fallback_to_basic=self.fallback_to_basic
            )._perform_calibration(sim_metrics, hw_metrics, simulation_parameters, parameter_bounds)
        
        # Create a copy of parameters to modify
        updated_parameters = simulation_parameters.copy()
        
        # Filter parameters to optimize (only numeric parameters)
        params_to_optimize = [k for k, v in updated_parameters.items() 
                             if isinstance(v, (int, float))]
        
        if not params_to_optimize:
            logger.warning("No numeric parameters found for optimization")
            return updated_parameters
        
        logger.info(f"Optimizing {len(params_to_optimize)} parameters using Bayesian Optimization: {params_to_optimize}")
        
        # Setup parameter bounds if not provided
        if parameter_bounds is None:
            parameter_bounds = {}
        
        bounds = []
        for param in params_to_optimize:
            if param in parameter_bounds:
                bounds.append(parameter_bounds[param])
            else:
                # Default bounds: 0.1x to 10x current value, or -10 to +10 if near zero
                current_value = updated_parameters[param]
                if abs(current_value) < 1e-6:
                    bounds.append((-10.0, 10.0))
                else:
                    bounds.append((0.1 * current_value, 10.0 * current_value))
        
        # Initial parameter values
        initial_values = [updated_parameters[param] for param in params_to_optimize]
        
        # Define the error function to minimize
        def error_function(params):
            # Create a temporary parameter set with the current optimization values
            temp_params = updated_parameters.copy()
            for i, param in enumerate(params_to_optimize):
                temp_params[param] = params[i]
            
            # Calculate the error between simulation and hardware results
            total_error = 0.0
            for metric in set(sim_metrics.keys()).intersection(hw_metrics.keys()):
                # Apply the current parameters to adjust simulation values
                adjusted_sim_values = self._apply_parameters_to_metric(
                    sim_metrics[metric], temp_params, metric)
                
                # Calculate Mean Squared Error
                hw_values = np.array(hw_metrics[metric])
                mse = np.mean((np.array(adjusted_sim_values) - hw_values) ** 2)
                total_error += mse
            
            return total_error
        
        try:
            # Bayesian Optimization process
            X_samples = []
            y_samples = []
            
            # 1. Initial random sampling
            logger.info(f"Performing {self.n_initial_points} initial random evaluations")
            for _ in range(self.n_initial_points):
                # Generate random point within bounds
                x = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(params_to_optimize))]
                y = error_function(x)
                
                X_samples.append(x)
                y_samples.append(y)
            
            # Setup Gaussian Process Regressor
            kernel = C(1.0, (0.01, 100)) * RBF(0.1, (0.01, 100))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,  # Noise level
                n_restarts_optimizer=5,
                normalize_y=True
            )
            
            # Best parameters so far
            best_idx = np.argmin(y_samples)
            best_params = X_samples[best_idx].copy()
            best_error = y_samples[best_idx]
            
            logger.info(f"Initial best error: {best_error}")
            
            # 2. Sequential optimization
            for iteration in range(self.max_iterations - self.n_initial_points):
                # Fit GP to the data
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(np.array(X_samples), np.array(y_samples))
                
                # Find the next point to evaluate using acquisition function
                next_point = self._acquisition_function(gp, np.array(X_samples), bounds)
                
                # Evaluate the point
                next_error = error_function(next_point)
                
                # Update the data
                X_samples.append(next_point)
                y_samples.append(next_error)
                
                # Update the best parameters if improved
                if next_error < best_error:
                    best_params = next_point.copy()
                    best_error = next_error
                    logger.info(f"Improved error to {best_error} at iteration {iteration+1}")
            
            # Update parameters with the best values
            for i, param in enumerate(params_to_optimize):
                updated_parameters[param] = best_params[i]
                logger.info(f"Optimized {param}: {updated_parameters[param]}")
            
            logger.info(f"Bayesian optimization completed with final error: {best_error}")
            
        except Exception as e:
            logger.error(f"Error during Bayesian optimization: {str(e)}")
            # Keep original parameters for the ones that failed
            logger.info("Keeping original parameter values due to optimization failure")
        
        return updated_parameters
    
    def _acquisition_function(self, gp, X_samples, bounds):
        """
        Expected Improvement acquisition function for Bayesian Optimization.
        
        Args:
            gp: Fitted Gaussian Process model
            X_samples: Previously sampled points
            bounds: Parameter bounds
            
        Returns:
            Next point to evaluate
        """
        # Current best value
        y_best = np.min(gp.y_train_)
        
        # Random search over parameter space for the best acquisition value
        n_random_points = 1000
        dim = X_samples.shape[1]
        X_random = np.random.uniform(
            low=[bounds[i][0] for i in range(dim)],
            high=[bounds[i][1] for i in range(dim)],
            size=(n_random_points, dim)
        )
        
        # Predict mean and std at each point
        mu, sigma = gp.predict(X_random, return_std=True)
        
        # Calculate expected improvement
        with np.errstate(divide='ignore'):
            Z = (y_best - mu) / sigma
            ei = (y_best - mu) * scipy.stats.norm.cdf(Z) + sigma * scipy.stats.norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        # Add exploration bonus
        ei += self.exploration_factor * sigma
        
        # Find best point
        best_idx = np.argmax(ei)
        next_point = X_random[best_idx]
        
        return next_point
    
    def _apply_parameters_to_metric(
        self, 
        sim_values: List[float], 
        parameters: Dict[str, Any], 
        metric_name: str
    ) -> List[float]:
        """
        Apply parameters to adjust simulation values for a metric.
        
        Args:
            sim_values: List of simulation values for the metric
            parameters: Current parameter dictionary
            metric_name: Name of the metric
            
        Returns:
            List of adjusted simulation values
        """
        # Reuse the implementation from MultiParameterCalibrator
        return MultiParameterCalibrator._apply_parameters_to_metric(
            self, sim_values, parameters, metric_name
        )


# Neural Network Calibrator
class NeuralNetworkCalibrator(AdvancedCalibrator):
    """
    Advanced calibrator that uses a Neural Network for parameter tuning.
    
    Trains a neural network to predict the error between simulation and hardware results
    based on parameter values, then uses this model to find optimal parameters.
    """
    
    def __init__(
        self, 
        hidden_layers: List[int] = [32, 16], 
        epochs: int = 100,
        batch_size: int = 16,
        training_points: int = 100,
        **kwargs
    ):
        """
        Initialize the NeuralNetworkCalibrator.
        
        Args:
            hidden_layers: List of neurons in each hidden layer
            epochs: Number of training epochs
            batch_size: Batch size for training
            training_points: Number of random points to generate for training
            **kwargs: Arguments to pass to AdvancedCalibrator.__init__
        """
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.training_points = training_points
        
        # Check for TensorFlow availability
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. NeuralNetworkCalibrator will use fallback methods.")
    
    def _perform_calibration(
        self, 
        sim_metrics: Dict[str, List[float]], 
        hw_metrics: Dict[str, List[float]],
        simulation_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Perform neural network-based calibration.
        
        Args:
            sim_metrics: Dictionary mapping metric names to lists of simulation values
            hw_metrics: Dictionary mapping metric names to lists of hardware values
            simulation_parameters: Current simulation parameters
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
            
        Returns:
            Updated simulation parameters
        """
        # If TensorFlow is not available, fall back to MultiParameterCalibrator
        if not TF_AVAILABLE:
            logger.info("Falling back to MultiParameterCalibrator due to missing TensorFlow")
            return MultiParameterCalibrator(
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations,
                history_file=self.history_file,
                fallback_to_basic=self.fallback_to_basic
            )._perform_calibration(sim_metrics, hw_metrics, simulation_parameters, parameter_bounds)
        
        # Create a copy of parameters to modify
        updated_parameters = simulation_parameters.copy()
        
        # Filter parameters to optimize (only numeric parameters)
        params_to_optimize = [k for k, v in updated_parameters.items() 
                             if isinstance(v, (int, float))]
        
        if not params_to_optimize:
            logger.warning("No numeric parameters found for optimization")
            return updated_parameters
        
        logger.info(f"Optimizing {len(params_to_optimize)} parameters using Neural Network: {params_to_optimize}")
        
        # Setup parameter bounds if not provided
        if parameter_bounds is None:
            parameter_bounds = {}
        
        bounds = []
        for param in params_to_optimize:
            if param in parameter_bounds:
                bounds.append(parameter_bounds[param])
            else:
                # Default bounds: 0.1x to 10x current value, or -10 to +10 if near zero
                current_value = updated_parameters[param]
                if abs(current_value) < 1e-6:
                    bounds.append((-10.0, 10.0))
                else:
                    bounds.append((0.1 * current_value, 10.0 * current_value))
        
        # Initial parameter values
        initial_values = [updated_parameters[param] for param in params_to_optimize]
        
        # Define the error function to minimize
        def error_function(params):
            # Create a temporary parameter set with the current optimization values
            temp_params = updated_parameters.copy()
            for i, param in enumerate(params_to_optimize):
                temp_params[param] = params[i]
            
            # Calculate the error between simulation and hardware results
            total_error = 0.0
            for metric in set(sim_metrics.keys()).intersection(hw_metrics.keys()):
                # Apply the current parameters to adjust simulation values
                adjusted_sim_values = self._apply_parameters_to_metric(
                    sim_metrics[metric], temp_params, metric)
                
                # Calculate Mean Squared Error
                hw_values = np.array(hw_metrics[metric])
                mse = np.mean((np.array(adjusted_sim_values) - hw_values) ** 2)
                total_error += mse
            
            return total_error
        
        try:
            # Neural Network-based optimization process
            
            # 1. Generate training data
            logger.info(f"Generating {self.training_points} training points")
            X_train = []
            y_train = []
            
            for _ in range(self.training_points):
                # Generate random point within bounds
                x = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(params_to_optimize))]
                y = error_function(x)
                
                X_train.append(x)
                y_train.append(y)
            
            # Normalize the training data
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0)
            X_std[X_std < 1e-10] = 1.0  # Avoid division by zero
            
            y_mean = np.mean(y_train)
            y_std = np.std(y_train)
            if y_std < 1e-10:
                y_std = 1.0  # Avoid division by zero
            
            X_train_norm = (np.array(X_train) - X_mean) / X_std
            y_train_norm = (np.array(y_train) - y_mean) / y_std
            
            # 2. Build and train neural network
            logger.info("Building and training neural network")
            input_dim = len(params_to_optimize)
            
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
            
            # Add hidden layers
            for units in self.hidden_layers:
                model.add(tf.keras.layers.Dense(units, activation='relu'))
            
            # Output layer
            model.add(tf.keras.layers.Dense(1))
            
            # Compile model
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            model.fit(
                X_train_norm, 
                y_train_norm, 
                epochs=self.epochs, 
                batch_size=self.batch_size,
                verbose=0
            )
            
            # 3. Use the neural network to guide optimization
            logger.info("Using neural network for optimization")
            
            # Current best parameters (from training data)
            best_idx = np.argmin(y_train)
            best_params = np.array(X_train[best_idx]).copy()
            best_error = y_train[best_idx]
            
            # Normalize the initial values
            initial_values_norm = (np.array(initial_values) - X_mean) / X_std
            
            # Use gradient-based optimization on the neural network
            def nn_predict(x_norm):
                x_norm_tensor = tf.convert_to_tensor(x_norm.reshape(1, -1), dtype=tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(x_norm_tensor)
                    prediction = model(x_norm_tensor)
                
                gradients = tape.gradient(prediction, x_norm_tensor)
                return prediction.numpy()[0, 0], gradients.numpy()[0]
            
            # Start from initial values
            current_point_norm = initial_values_norm.copy()
            
            # Perform gradient descent on the neural network
            for iteration in range(self.max_iterations):
                # Get prediction and gradient
                pred, grad = nn_predict(current_point_norm)
                
                # Update point (gradient descent because we're minimizing)
                current_point_norm -= self.learning_rate * grad
                
                # Clip to normalized bounds
                bounds_norm = []
                for i, (lower, upper) in enumerate(bounds):
                    lower_norm = (lower - X_mean[i]) / X_std[i]
                    upper_norm = (upper - X_mean[i]) / X_std[i]
                    bounds_norm.append((lower_norm, upper_norm))
                
                for i in range(len(current_point_norm)):
                    current_point_norm[i] = max(bounds_norm[i][0], 
                                               min(bounds_norm[i][1], current_point_norm[i]))
                
                # Convert back to original scale
                current_point = current_point_norm * X_std + X_mean
                
                # Evaluate with the true error function
                current_error = error_function(current_point)
                
                # Update best parameters if improved
                if current_error < best_error:
                    best_params = current_point.copy()
                    best_error = current_error
                    logger.info(f"Improved error to {best_error} at iteration {iteration+1}")
            
            # Update parameters with the best values
            for i, param in enumerate(params_to_optimize):
                updated_parameters[param] = best_params[i]
                logger.info(f"Optimized {param}: {updated_parameters[param]}")
            
            logger.info(f"Neural network optimization completed with final error: {best_error}")
            
        except Exception as e:
            logger.error(f"Error during neural network optimization: {str(e)}")
            # Keep original parameters for the ones that failed
            logger.info("Keeping original parameter values due to optimization failure")
        
        return updated_parameters
    
    def _apply_parameters_to_metric(
        self, 
        sim_values: List[float], 
        parameters: Dict[str, Any], 
        metric_name: str
    ) -> List[float]:
        """
        Apply parameters to adjust simulation values for a metric.
        
        Args:
            sim_values: List of simulation values for the metric
            parameters: Current parameter dictionary
            metric_name: Name of the metric
            
        Returns:
            List of adjusted simulation values
        """
        # Reuse the implementation from MultiParameterCalibrator
        return MultiParameterCalibrator._apply_parameters_to_metric(
            self, sim_values, parameters, metric_name
        )


# Ensemble Calibrator
class EnsembleCalibrator(AdvancedCalibrator):
    """
    Advanced calibrator that combines multiple calibration methods.
    
    Uses an ensemble of different calibration methods and combines their
    results to produce more robust parameter recommendations.
    """
    
    def __init__(
        self, 
        ensemble_methods: List[str] = None,
        ensemble_weights: List[float] = None,
        **kwargs
    ):
        """
        Initialize the EnsembleCalibrator.
        
        Args:
            ensemble_methods: List of calibration methods to use in the ensemble
            ensemble_weights: List of weights for each method (must match length of ensemble_methods)
            **kwargs: Arguments to pass to AdvancedCalibrator.__init__
        """
        super().__init__(**kwargs)
        
        # Default methods are all available calibrators
        self.available_methods = {
            "basic": BasicCalibrator,
            "multi_parameter": MultiParameterCalibrator
        }
        
        # Add optional methods if dependencies are available
        if SKLEARN_AVAILABLE:
            self.available_methods["bayesian"] = BayesianOptimizationCalibrator
        
        if TF_AVAILABLE:
            self.available_methods["neural_network"] = NeuralNetworkCalibrator
        
        # Set up ensemble methods
        if ensemble_methods is None:
            # Use all available methods
            self.ensemble_methods = list(self.available_methods.keys())
        else:
            # Filter to only available methods
            self.ensemble_methods = [m for m in ensemble_methods if m in self.available_methods]
            if len(self.ensemble_methods) < len(ensemble_methods):
                logger.warning(f"Some requested ensemble methods are not available: " 
                               f"{set(ensemble_methods) - set(self.ensemble_methods)}")
        
        # Set up ensemble weights
        if ensemble_weights is None:
            # Equal weights
            self.ensemble_weights = [1.0 / len(self.ensemble_methods)] * len(self.ensemble_methods)
        else:
            # Use provided weights
            if len(ensemble_weights) != len(self.ensemble_methods):
                logger.warning(f"Length of ensemble_weights ({len(ensemble_weights)}) does not match "
                               f"length of ensemble_methods ({len(self.ensemble_methods)}). "
                               f"Using equal weights.")
                self.ensemble_weights = [1.0 / len(self.ensemble_methods)] * len(self.ensemble_methods)
            else:
                # Normalize weights
                total = sum(ensemble_weights)
                self.ensemble_weights = [w / total for w in ensemble_weights]
        
        logger.info(f"Ensemble calibrator initialized with methods: {self.ensemble_methods}")
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
        
        # Initialize calibrators
        self.calibrators = {}
        for method in self.ensemble_methods:
            calibrator_class = self.available_methods[method]
            self.calibrators[method] = calibrator_class(
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations // len(self.ensemble_methods),  # Divide iterations
                history_file=None,  # Only the ensemble calibrator maintains history
                fallback_to_basic=False  # No need for fallback in individual calibrators
            )
    
    def _perform_calibration(
        self, 
        sim_metrics: Dict[str, List[float]], 
        hw_metrics: Dict[str, List[float]],
        simulation_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Perform ensemble calibration by combining multiple methods.
        
        Args:
            sim_metrics: Dictionary mapping metric names to lists of simulation values
            hw_metrics: Dictionary mapping metric names to lists of hardware values
            simulation_parameters: Current simulation parameters
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
            
        Returns:
            Updated simulation parameters
        """
        logger.info("Starting ensemble calibration")
        
        # Filter parameters to optimize (only numeric parameters)
        params_to_optimize = [k for k, v in simulation_parameters.items() 
                             if isinstance(v, (int, float))]
        
        if not params_to_optimize:
            logger.warning("No numeric parameters found for optimization")
            return simulation_parameters.copy()
        
        # Run each calibrator
        calibrated_params = {}
        errors = {}
        
        for i, method in enumerate(self.ensemble_methods):
            logger.info(f"Running calibrator: {method}")
            
            try:
                # Run the calibrator
                calibrator = self.calibrators[method]
                calibrated = calibrator._perform_calibration(
                    sim_metrics, hw_metrics, simulation_parameters, parameter_bounds
                )
                
                # Calculate the error with these parameters
                error = self._calculate_error(calibrated, sim_metrics, hw_metrics)
                
                calibrated_params[method] = calibrated
                errors[method] = error
                
                logger.info(f"Calibrator {method} completed with error: {error}")
                
            except Exception as e:
                logger.error(f"Error in calibrator {method}: {str(e)}")
                # Skip this method
                calibrated_params[method] = simulation_parameters.copy()
                errors[method] = float('inf')
        
        # Create ensemble parameter set
        if not calibrated_params:
            logger.warning("No calibrator succeeded - returning original parameters")
            return simulation_parameters.copy()
        
        # Create weighted ensemble for each parameter
        ensemble_parameters = simulation_parameters.copy()
        
        # Track which methods were actually used (error not inf)
        valid_methods = [m for m in self.ensemble_methods if errors[m] < float('inf')]
        if not valid_methods:
            logger.warning("No valid calibration results - returning original parameters")
            return simulation_parameters.copy()
        
        # Normalize weights for valid methods
        valid_weights = [self.ensemble_weights[self.ensemble_methods.index(m)] for m in valid_methods]
        total_weight = sum(valid_weights)
        if total_weight == 0:
            normalized_weights = [1.0 / len(valid_methods)] * len(valid_methods)
        else:
            normalized_weights = [w / total_weight for w in valid_weights]
        
        # Advanced ensemble: weight inversely by error (lower error = higher weight)
        inverse_errors = [1.0 / max(errors[m], 1e-10) for m in valid_methods]
        total_inverse = sum(inverse_errors)
        if total_inverse == 0:
            error_weights = normalized_weights
        else:
            error_weights = [w * (e / total_inverse) for w, e in zip(normalized_weights, inverse_errors)]
            # Renormalize
            total = sum(error_weights)
            error_weights = [w / total for w in error_weights]
        
        logger.info(f"Error-weighted ensemble weights: {dict(zip(valid_methods, error_weights))}")
        
        # Compute weighted parameters
        for param in params_to_optimize:
            ensemble_value = 0.0
            for i, method in enumerate(valid_methods):
                ensemble_value += calibrated_params[method][param] * error_weights[i]
            
            ensemble_parameters[param] = ensemble_value
            logger.info(f"Ensemble value for {param}: {ensemble_value}")
        
        # Calculate the error with ensemble parameters
        ensemble_error = self._calculate_error(ensemble_parameters, sim_metrics, hw_metrics)
        logger.info(f"Ensemble calibration completed with error: {ensemble_error}")
        
        # Compare with best individual method
        best_method = min(valid_methods, key=lambda m: errors[m])
        logger.info(f"Best individual method: {best_method} with error: {errors[best_method]}")
        
        # Return the better of ensemble and best individual
        if ensemble_error <= errors[best_method]:
            logger.info("Using ensemble parameters")
            return ensemble_parameters
        else:
            logger.info(f"Using parameters from best method: {best_method}")
            return calibrated_params[best_method]
    
    def _calculate_error(
        self, 
        parameters: Dict[str, Any], 
        sim_metrics: Dict[str, List[float]], 
        hw_metrics: Dict[str, List[float]]
    ) -> float:
        """
        Calculate the error between simulation and hardware results.
        
        Args:
            parameters: Parameter dictionary to evaluate
            sim_metrics: Dictionary of simulation metrics
            hw_metrics: Dictionary of hardware metrics
            
        Returns:
            Total error value
        """
        total_error = 0.0
        
        for metric in set(sim_metrics.keys()).intersection(hw_metrics.keys()):
            # Apply the parameters to adjust simulation values
            adjusted_sim_values = self._apply_parameters_to_metric(
                sim_metrics[metric], parameters, metric)
            
            # Calculate Mean Squared Error
            hw_values = np.array(hw_metrics[metric])
            mse = np.mean((np.array(adjusted_sim_values) - hw_values) ** 2)
            total_error += mse
        
        return total_error
    
    def _apply_parameters_to_metric(
        self, 
        sim_values: List[float], 
        parameters: Dict[str, Any], 
        metric_name: str
    ) -> List[float]:
        """
        Apply parameters to adjust simulation values for a metric.
        
        Args:
            sim_values: List of simulation values for the metric
            parameters: Current parameter dictionary
            metric_name: Name of the metric
            
        Returns:
            List of adjusted simulation values
        """
        # Reuse the implementation from MultiParameterCalibrator
        return MultiParameterCalibrator._apply_parameters_to_metric(
            self, sim_values, parameters, metric_name
        )