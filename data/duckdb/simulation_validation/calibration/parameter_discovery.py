"""
Parameter discovery and sensitivity analysis for calibration system.

This module provides functionality for automatic parameter discovery 
and sensitivity analysis to improve calibration effectiveness.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Set, Union
import json
from datetime import datetime
import os
import warnings

# Setup logger
logger = logging.getLogger(__name__)

class ParameterDiscovery:
    """
    Discovers and analyzes parameters that affect simulation accuracy.
    
    This class provides methods for discovering simulation parameters that
    have a significant effect on accuracy, and analyzing their sensitivity.
    """
    
    def __init__(
        self, 
        sensitivity_threshold: float = 0.01,
        discovery_iterations: int = 100,
        exploration_range: float = 0.5,
        result_file: Optional[str] = None
    ):
        """
        Initialize the ParameterDiscovery.
        
        Args:
            sensitivity_threshold: Threshold for considering a parameter sensitive
            discovery_iterations: Number of iterations for parameter discovery
            exploration_range: Range to explore parameter values (0-1)
            result_file: Optional path to file for storing discovery results
        """
        self.sensitivity_threshold = sensitivity_threshold
        self.discovery_iterations = discovery_iterations
        self.exploration_range = exploration_range
        self.result_file = result_file
        self.results = []
        
        if self.result_file and os.path.exists(self.result_file):
            try:
                with open(self.result_file, 'r') as f:
                    self.results = json.load(f)
                logger.info(f"Loaded {len(self.results)} discovery results from file")
            except Exception as e:
                logger.error(f"Error loading discovery results: {str(e)}")
                self.results = []
    
    def discover_parameters(
        self, 
        error_function: callable,
        initial_parameters: Dict[str, Any],
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Discover parameters that significantly affect simulation accuracy.
        
        Args:
            error_function: Function that takes parameters and returns error value
            initial_parameters: Initial parameter dictionary
            parameter_ranges: Optional dictionary mapping parameter names to (min, max) ranges
            
        Returns:
            Dictionary with parameter sensitivity analysis results
        """
        logger.info("Starting parameter discovery")
        
        # Identify numeric parameters
        numeric_params = {}
        for k, v in initial_parameters.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_params[k] = v
        
        if not numeric_params:
            logger.warning("No numeric parameters found for discovery")
            return {"sensitive_parameters": [], "analysis": {}}
        
        logger.info(f"Analyzing {len(numeric_params)} numeric parameters: {list(numeric_params.keys())}")
        
        # Set up parameter ranges if not provided
        if parameter_ranges is None:
            parameter_ranges = {}
        
        # Fill in missing ranges with default values
        for param, value in numeric_params.items():
            if param not in parameter_ranges:
                if abs(value) < 1e-10:
                    # For zero or near-zero values
                    parameter_ranges[param] = (-1.0, 1.0)
                else:
                    # For non-zero values, use percentage-based range
                    spread = abs(value) * self.exploration_range
                    parameter_ranges[param] = (value - spread, value + spread)
                    
                logger.info(f"Setting default range for {param}: {parameter_ranges[param]}")
        
        # Calculate baseline error
        baseline_error = error_function(initial_parameters)
        logger.info(f"Baseline error: {baseline_error}")
        
        # Perform parameter sensitivity analysis
        sensitivity_results = {}
        
        for param in numeric_params:
            param_range = parameter_ranges[param]
            current_value = numeric_params[param]
            
            # Sample points within the range
            sample_points = np.linspace(param_range[0], param_range[1], num=10)
            
            # Evaluate each sample point
            errors = []
            for point in sample_points:
                test_params = initial_parameters.copy()
                test_params[param] = point
                error = error_function(test_params)
                errors.append(error)
            
            # Calculate statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            min_error = np.min(errors)
            max_error = np.max(errors)
            
            # Calculate sensitivity (normalized std / mean)
            # Higher value means more sensitive
            if abs(mean_error) < 1e-10:
                sensitivity = 0.0
            else:
                sensitivity = std_error / mean_error
            
            # Calculate potential improvement
            improvement = (baseline_error - min_error) / baseline_error if baseline_error > 0 else 0.0
            
            # Store results
            sensitivity_results[param] = {
                "current_value": current_value,
                "range": param_range,
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "min_error": float(min_error),
                "max_error": float(max_error),
                "sensitivity": float(sensitivity),
                "potential_improvement": float(improvement),
                "is_sensitive": sensitivity > self.sensitivity_threshold
            }
            
            logger.info(f"Parameter {param}: sensitivity = {sensitivity}, " 
                       f"potential improvement = {improvement * 100:.2f}%")
        
        # Find the optimal parameter values within the tested ranges
        if self.discovery_iterations > 0:
            logger.info(f"Performing {self.discovery_iterations} parameter discovery iterations")
            optimal_params = self._optimize_parameters(
                error_function, initial_parameters, sensitivity_results
            )
        else:
            optimal_params = initial_parameters.copy()
        
        # Identify sensitive parameters
        sensitive_params = [p for p, data in sensitivity_results.items() 
                          if data["is_sensitive"]]
        
        # Create result summary
        result = {
            "timestamp": datetime.now().isoformat(),
            "sensitive_parameters": sensitive_params,
            "analysis": sensitivity_results,
            "baseline_error": float(baseline_error),
            "optimal_parameters": optimal_params
        }
        
        # Calculate error with optimal parameters
        optimal_error = error_function(optimal_params)
        improvement = (baseline_error - optimal_error) / baseline_error if baseline_error > 0 else 0.0
        
        result["optimal_error"] = float(optimal_error)
        result["improvement"] = float(improvement)
        
        logger.info(f"Parameter discovery completed with {len(sensitive_params)} sensitive parameters")
        logger.info(f"Optimal error: {optimal_error}, improvement: {improvement * 100:.2f}%")
        
        # Save result
        self.results.append(result)
        self._save_results()
        
        return result
    
    def _optimize_parameters(
        self, 
        error_function: callable,
        initial_parameters: Dict[str, Any],
        sensitivity_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize parameters based on sensitivity analysis.
        
        Args:
            error_function: Function that takes parameters and returns error value
            initial_parameters: Initial parameter dictionary
            sensitivity_results: Results of sensitivity analysis
            
        Returns:
            Optimized parameter dictionary
        """
        # Select sensitive parameters to optimize
        sensitive_params = [p for p, data in sensitivity_results.items() 
                          if data["is_sensitive"]]
        
        if not sensitive_params:
            logger.info("No sensitive parameters found for optimization")
            return initial_parameters.copy()
        
        logger.info(f"Optimizing {len(sensitive_params)} sensitive parameters: {sensitive_params}")
        
        # Current best parameters and error
        best_params = initial_parameters.copy()
        best_error = error_function(best_params)
        
        # Random search optimization
        for iteration in range(self.discovery_iterations):
            # Create a new parameter set
            test_params = best_params.copy()
            
            # Modify sensitive parameters
            for param in sensitive_params:
                range_data = sensitivity_results[param]["range"]
                # Sample from parameter range
                test_params[param] = np.random.uniform(range_data[0], range_data[1])
            
            # Evaluate error
            error = error_function(test_params)
            
            # Update if better
            if error < best_error:
                best_params = test_params.copy()
                best_error = error
                logger.info(f"Improved error to {best_error} at iteration {iteration+1}")
        
        return best_params
    
    def _save_results(self) -> None:
        """Save discovery results to file."""
        if not self.result_file:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.result_file)), exist_ok=True)
            
            with open(self.result_file, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            logger.info(f"Saved {len(self.results)} discovery results to file")
        except Exception as e:
            logger.error(f"Error saving discovery results: {str(e)}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get the parameter discovery results.
        
        Returns:
            List of result dictionaries
        """
        return self.results
    
    def analyze_parameters_over_time(self) -> Dict[str, Any]:
        """
        Analyze how parameter sensitivity has changed over time.
        
        Returns:
            Dictionary with time-based analysis results
        """
        if not self.results:
            return {"error": "No parameter discovery results available"}
        
        try:
            # Get all parameters ever analyzed
            all_params = set()
            for result in self.results:
                all_params.update(result.get("analysis", {}).keys())
            
            # Track sensitivity over time
            param_history = {}
            for param in all_params:
                values = []
                sensitivities = []
                improvements = []
                timestamps = []
                
                for result in self.results:
                    analysis = result.get("analysis", {}).get(param)
                    if analysis:
                        if "current_value" in analysis:
                            values.append(analysis["current_value"])
                        else:
                            values.append(None)
                            
                        if "sensitivity" in analysis:
                            sensitivities.append(analysis["sensitivity"])
                        else:
                            sensitivities.append(None)
                            
                        if "potential_improvement" in analysis:
                            improvements.append(analysis["potential_improvement"])
                        else:
                            improvements.append(None)
                            
                        timestamps.append(result["timestamp"])
                
                param_history[param] = {
                    "values": values,
                    "sensitivities": sensitivities,
                    "potential_improvements": improvements,
                    "timestamps": timestamps
                }
                
                # Calculate trend in sensitivity
                if len(sensitivities) >= 3 and all(s is not None for s in sensitivities):
                    param_history[param]["sensitivity_trend"] = self._detect_trend(sensitivities)
                else:
                    param_history[param]["sensitivity_trend"] = "insufficient_data"
            
            # Analyze which parameters have been consistently sensitive
            consistently_sensitive = []
            occasionally_sensitive = []
            rarely_sensitive = []
            
            for param, history in param_history.items():
                # Calculate how often the parameter was sensitive
                sensitive_count = sum(1 for result in self.results 
                                    if param in result.get("sensitive_parameters", []))
                
                if len(self.results) > 0:
                    sensitive_ratio = sensitive_count / len(self.results)
                    
                    if sensitive_ratio > 0.8:
                        consistently_sensitive.append(param)
                    elif sensitive_ratio > 0.3:
                        occasionally_sensitive.append(param)
                    else:
                        rarely_sensitive.append(param)
                    
                    param_history[param]["sensitive_ratio"] = sensitive_ratio
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "num_analyses": len(self.results),
                "parameter_history": param_history,
                "summary": {
                    "consistently_sensitive_parameters": consistently_sensitive,
                    "occasionally_sensitive_parameters": occasionally_sensitive,
                    "rarely_sensitive_parameters": rarely_sensitive
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing parameters over time: {str(e)}")
            return {"error": f"Error analyzing parameters: {str(e)}"}
    
    def _detect_trend(self, values: List[float]) -> str:
        """
        Detect trend in a series of values.
        
        Args:
            values: List of values over time
            
        Returns:
            Trend description: "increasing", "decreasing", "stable", or "oscillating"
        """
        if len(values) < 3:
            return "insufficient_data"
        
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
    
    def clear_results(self) -> None:
        """Clear the parameter discovery results."""
        self.results = []
        if self.result_file and os.path.exists(self.result_file):
            try:
                os.remove(self.result_file)
                logger.info(f"Removed result file: {self.result_file}")
            except Exception as e:
                logger.error(f"Error removing result file: {str(e)}")


class AdaptiveCalibrationScheduler:
    """
    Schedules calibration operations based on drift detection.
    
    This class determines when calibration should be performed and
    with what frequency based on detected drift and simulation accuracy.
    """
    
    def __init__(
        self, 
        min_interval_hours: float = 24.0,
        max_interval_hours: float = 168.0,  # 1 week
        error_threshold: float = 0.1,  # 10% error
        drift_threshold: float = 0.05,  # 5% drift
        schedule_file: Optional[str] = None
    ):
        """
        Initialize the AdaptiveCalibrationScheduler.
        
        Args:
            min_interval_hours: Minimum interval between calibrations in hours
            max_interval_hours: Maximum interval between calibrations in hours
            error_threshold: Error threshold to trigger calibration
            drift_threshold: Drift threshold to trigger calibration
            schedule_file: Optional path to file for storing schedule data
        """
        self.min_interval_hours = min_interval_hours
        self.max_interval_hours = max_interval_hours
        self.error_threshold = error_threshold
        self.drift_threshold = drift_threshold
        self.schedule_file = schedule_file
        self.schedule_data = {
            "last_calibration": None,
            "next_scheduled": None,
            "current_interval_hours": min_interval_hours,
            "calibrations": []
        }
        
        if self.schedule_file and os.path.exists(self.schedule_file):
            try:
                with open(self.schedule_file, 'r') as f:
                    self.schedule_data = json.load(f)
                logger.info(f"Loaded calibration schedule from file")
            except Exception as e:
                logger.error(f"Error loading calibration schedule: {str(e)}")
    
    def should_calibrate(
        self, 
        current_error: float = None,
        drift_value: float = None
    ) -> Tuple[bool, str]:
        """
        Determine if calibration should be performed.
        
        Args:
            current_error: Optional current error value
            drift_value: Optional drift value
            
        Returns:
            Tuple of (should_calibrate, reason)
        """
        now = datetime.now()
        now_str = now.isoformat()
        
        # Check if there's a scheduled calibration
        next_scheduled = self.schedule_data["next_scheduled"]
        if next_scheduled:
            next_time = datetime.fromisoformat(next_scheduled)
            if now >= next_time:
                return True, f"Scheduled calibration due (scheduled: {next_scheduled})"
        
        # Check error threshold
        if current_error is not None and current_error > self.error_threshold:
            return True, f"Error ({current_error:.4f}) exceeds threshold ({self.error_threshold:.4f})"
        
        # Check drift threshold
        if drift_value is not None and drift_value > self.drift_threshold:
            return True, f"Drift ({drift_value:.4f}) exceeds threshold ({self.drift_threshold:.4f})"
        
        # Otherwise, don't calibrate
        return False, "No calibration needed"
    
    def record_calibration(
        self, 
        error_before: float = None,
        error_after: float = None,
        drift_value: float = None,
        calibration_time: float = None
    ) -> None:
        """
        Record that a calibration was performed.
        
        Args:
            error_before: Optional error value before calibration
            error_after: Optional error value after calibration
            drift_value: Optional drift value that triggered calibration
            calibration_time: Optional time taken for calibration in seconds
        """
        now = datetime.now()
        now_str = now.isoformat()
        
        # Record calibration
        calibration = {
            "timestamp": now_str,
            "error_before": error_before,
            "error_after": error_after,
            "improvement": ((error_before - error_after) / error_before 
                           if error_before and error_after and error_before > 0 else None),
            "drift_value": drift_value,
            "calibration_time": calibration_time
        }
        
        self.schedule_data["calibrations"].append(calibration)
        self.schedule_data["last_calibration"] = now_str
        
        # Adjust calibration interval based on effectiveness
        self._adjust_calibration_interval(calibration)
        
        # Schedule next calibration
        next_time = now + np.timedelta64(int(self.schedule_data["current_interval_hours"] * 3600), 's')
        self.schedule_data["next_scheduled"] = next_time.isoformat()
        
        logger.info(f"Calibration recorded at {now_str}")
        logger.info(f"Next calibration scheduled for {self.schedule_data['next_scheduled']}")
        
        # Save schedule
        self._save_schedule()
    
    def _adjust_calibration_interval(self, calibration: Dict[str, Any]) -> None:
        """
        Adjust the calibration interval based on calibration effectiveness.
        
        Args:
            calibration: Calibration record
        """
        current_interval = self.schedule_data["current_interval_hours"]
        
        # Default: keep the current interval
        new_interval = current_interval
        
        # Adjust based on improvement (if available)
        if calibration.get("improvement") is not None:
            improvement = calibration["improvement"]
            
            if improvement < 0.01:  # Less than 1% improvement
                # Not very effective, increase interval (less frequent)
                new_interval = min(current_interval * 1.5, self.max_interval_hours)
                logger.info(f"Minimal improvement ({improvement:.2%}), increasing interval to {new_interval:.1f} hours")
            elif improvement > 0.1:  # More than 10% improvement
                # Very effective, decrease interval (more frequent)
                new_interval = max(current_interval * 0.7, self.min_interval_hours)
                logger.info(f"Significant improvement ({improvement:.2%}), decreasing interval to {new_interval:.1f} hours")
        
        # Adjust based on drift (if available)
        if calibration.get("drift_value") is not None:
            drift = calibration["drift_value"]
            
            if drift > self.drift_threshold * 2:  # Drift is more than 2x threshold
                # High drift, decrease interval (more frequent)
                new_interval = max(new_interval * 0.7, self.min_interval_hours)
                logger.info(f"High drift ({drift:.4f}), decreasing interval to {new_interval:.1f} hours")
        
        # Update interval
        self.schedule_data["current_interval_hours"] = new_interval
    
    def _save_schedule(self) -> None:
        """Save calibration schedule to file."""
        if not self.schedule_file:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.schedule_file)), exist_ok=True)
            
            with open(self.schedule_file, 'w') as f:
                json.dump(self.schedule_data, f, indent=2)
                
            logger.info(f"Saved calibration schedule to file")
        except Exception as e:
            logger.error(f"Error saving calibration schedule: {str(e)}")
    
    def get_calibration_history(self) -> List[Dict[str, Any]]:
        """
        Get the calibration history.
        
        Returns:
            List of calibration records
        """
        return self.schedule_data.get("calibrations", [])
    
    def get_next_scheduled_calibration(self) -> Optional[str]:
        """
        Get the timestamp of the next scheduled calibration.
        
        Returns:
            Timestamp string or None if no calibration is scheduled
        """
        return self.schedule_data.get("next_scheduled")
    
    def analyze_calibration_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of calibrations over time.
        
        Returns:
            Dictionary with analysis results
        """
        calibrations = self.schedule_data.get("calibrations", [])
        
        if not calibrations:
            return {"error": "No calibration history available"}
        
        try:
            # Extract data for analysis
            timestamps = []
            errors_before = []
            errors_after = []
            improvements = []
            drift_values = []
            
            for cal in calibrations:
                if cal.get("timestamp"):
                    timestamps.append(cal["timestamp"])
                else:
                    continue
                    
                if cal.get("error_before") is not None:
                    errors_before.append(cal["error_before"])
                else:
                    errors_before.append(None)
                    
                if cal.get("error_after") is not None:
                    errors_after.append(cal["error_after"])
                else:
                    errors_after.append(None)
                    
                if cal.get("improvement") is not None:
                    improvements.append(cal["improvement"])
                else:
                    improvements.append(None)
                    
                if cal.get("drift_value") is not None:
                    drift_values.append(cal["drift_value"])
                else:
                    drift_values.append(None)
            
            # Calculate averages (excluding None values)
            avg_error_before = np.mean([e for e in errors_before if e is not None]) if errors_before else None
            avg_error_after = np.mean([e for e in errors_after if e is not None]) if errors_after else None
            avg_improvement = np.mean([i for i in improvements if i is not None]) if improvements else None
            avg_drift = np.mean([d for d in drift_values if d is not None]) if drift_values else None
            
            # Calculate trends
            improvement_trend = self._detect_trend(
                [i for i in improvements if i is not None]) if len(improvements) >= 3 else "insufficient_data"
            
            # Calculate calibration intervals
            intervals = []
            for i in range(1, len(timestamps)):
                t1 = datetime.fromisoformat(timestamps[i-1])
                t2 = datetime.fromisoformat(timestamps[i])
                interval_hours = (t2 - t1).total_seconds() / 3600
                intervals.append(interval_hours)
            
            avg_interval = np.mean(intervals) if intervals else None
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "num_calibrations": len(calibrations),
                "avg_error_before": float(avg_error_before) if avg_error_before is not None else None,
                "avg_error_after": float(avg_error_after) if avg_error_after is not None else None,
                "avg_improvement": float(avg_improvement) if avg_improvement is not None else None,
                "avg_drift": float(avg_drift) if avg_drift is not None else None,
                "improvement_trend": improvement_trend,
                "avg_interval_hours": float(avg_interval) if avg_interval is not None else None,
                "current_interval_hours": self.schedule_data["current_interval_hours"],
                "calibration_data": {
                    "timestamps": timestamps,
                    "errors_before": errors_before,
                    "errors_after": errors_after,
                    "improvements": improvements,
                    "drift_values": drift_values
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing calibration effectiveness: {str(e)}")
            return {"error": f"Error analyzing effectiveness: {str(e)}"}
    
    def _detect_trend(self, values: List[float]) -> str:
        """
        Detect trend in a series of values.
        
        Args:
            values: List of values over time
            
        Returns:
            Trend description: "improving", "degrading", "stable", or "oscillating"
        """
        if len(values) < 3:
            return "insufficient_data"
        
        # Convert to numpy array
        vals = np.array(values)
        
        # Calculate differences between consecutive values
        diffs = np.diff(vals)
        
        # Check for consistent sign in differences
        if np.all(diffs > 0):
            return "improving"
        elif np.all(diffs < 0):
            return "degrading"
        
        # Check for oscillation (sign changes)
        sign_changes = np.sum(np.diff(np.signbit(diffs)))
        if sign_changes > len(diffs) / 3:  # If more than 1/3 of differences change sign
            return "oscillating"
        
        # Default to stable
        return "stable"
    
    def reset_schedule(self) -> None:
        """Reset the calibration schedule."""
        self.schedule_data = {
            "last_calibration": None,
            "next_scheduled": None,
            "current_interval_hours": self.min_interval_hours,
            "calibrations": []
        }
        
        if self.schedule_file and os.path.exists(self.schedule_file):
            try:
                os.remove(self.schedule_file)
                logger.info(f"Removed schedule file: {self.schedule_file}")
            except Exception as e:
                logger.error(f"Error removing schedule file: {str(e)}")
                
        logger.info("Calibration schedule reset")