#!/usr/bin/env python3
"""
Trend Projection for the Simulation Accuracy and Validation Framework.

This module provides specialized trend projection capabilities for simulation accuracy,
allowing for long-term forecasting and trend analysis of simulation performance.
The module includes:
- Long-term trend analysis and forecasting
- Scenario-based projections (best/worst/expected cases)
- Cyclical pattern detection and projection
- Simulation accuracy trajectory analysis
- Convergence and stability projections
"""

import logging
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analysis.trend_projection")

# Import base class
from duckdb_api.simulation_validation.analysis.base import AnalysisMethod
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

class TrendProjection(AnalysisMethod):
    """
    Trend projection for simulation accuracy metrics.
    
    This class extends the basic AnalysisMethod to provide specialized trend
    projection techniques for long-term forecasting of simulation accuracy:
    - Long-term trend analysis and forecasting
    - Scenario-based projections (best/worst/expected cases)
    - Cyclical pattern detection and projection
    - Simulation accuracy trajectory analysis
    - Convergence and stability projections
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trend projection method.
        
        Args:
            config: Configuration options for the analysis method
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            # Common metrics to analyze
            "metrics_to_analyze": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w"
            ],
            
            # Target metrics for projection
            "target_metrics": [
                "mape",  # Mean Absolute Percentage Error
                "overall_accuracy_score"  # Overall accuracy score
            ],
            
            # Long-term trend configuration
            "long_term_trend": {
                "enabled": True,
                "horizon": 30,  # Number of periods to forecast
                "confidence_level": 0.95,  # Confidence level for intervals
                "trend_detection_window": 5,  # Window for trend detection
                "smoothing": True,  # Whether to apply smoothing
                "min_history_periods": 10  # Minimum periods required for trend analysis
            },
            
            # Scenario-based projection configuration
            "scenario_projection": {
                "enabled": True,
                "scenarios": ["best_case", "expected_case", "worst_case"],
                "percentiles": {  # Percentiles for scenario bounds
                    "best_case": 0.25,  # 25th percentile (optimistic)
                    "expected_case": 0.5,  # 50th percentile (median, realistic)
                    "worst_case": 0.75  # 75th percentile (pessimistic)
                },
                "min_samples": 8  # Minimum samples required for scenario projection
            },
            
            # Cyclical pattern configuration
            "cyclical_pattern": {
                "enabled": True,
                "max_cycles": 3,  # Maximum cycles to detect
                "min_periods_per_cycle": 4,  # Minimum periods per cycle
                "significance_threshold": 0.05,  # Threshold for significant cycles
                "min_samples": 12  # Minimum samples required for cyclical analysis
            },
            
            # Trajectory analysis configuration
            "trajectory_analysis": {
                "enabled": True,
                "smoothing_window": 3,  # Window size for smoothing
                "change_point_threshold": 0.1,  # Threshold for change point detection
                "min_samples": 8  # Minimum samples required for trajectory analysis
            },
            
            # Convergence projection configuration
            "convergence_projection": {
                "enabled": True,
                "convergence_threshold": 0.05,  # Threshold for convergence
                "stability_window": 5,  # Window for stability assessment
                "min_samples": 10  # Minimum samples required for convergence analysis
            }
        }
        
        # Apply default config values if not specified
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict) and isinstance(self.config[key], dict):
                # Merge nested dictionaries
                for nested_key, nested_value in value.items():
                    if nested_key not in self.config[key]:
                        self.config[key][nested_key] = nested_value
    
    def analyze(
        self, 
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Perform trend projection on validation results.
        
        Args:
            validation_results: List of validation results to analyze
            
        Returns:
            Dictionary containing trend projection results and insights
        """
        # Check requirements
        meets_req, error_msg = self.check_requirements(validation_results)
        if not meets_req:
            logger.warning(f"Requirements not met for trend projection: {error_msg}")
            return {"status": "error", "message": error_msg}
        
        # Initialize results dictionary
        analysis_results = {
            "status": "success",
            "timestamp": datetime.datetime.now().isoformat(),
            "num_validation_results": len(validation_results),
            "metrics_analyzed": self.config["metrics_to_analyze"],
            "target_metrics": self.config["target_metrics"],
            "projections": {},
            "insights": {
                "key_findings": [],
                "recommendations": []
            }
        }
        
        # Extract time series data for analysis
        time_series_data = self._extract_time_series_data(validation_results)
        
        if not time_series_data:
            return {
                "status": "error",
                "message": "Failed to extract time series data for trend projection"
            }
        
        # Perform long-term trend analysis if enabled
        if self.config["long_term_trend"]["enabled"]:
            try:
                # Check if we have enough data points
                min_periods = self.config["long_term_trend"]["min_history_periods"]
                long_term_trends = {}
                
                for metric, data in time_series_data.items():
                    if len(data["values"]) >= min_periods:
                        # Perform long-term trend analysis
                        trend_results = self._analyze_long_term_trend(
                            data["values"], data["timestamps"], metric)
                        
                        if trend_results:
                            long_term_trends[metric] = trend_results
                
                # Add to results if we have any trends
                if long_term_trends:
                    analysis_results["projections"]["long_term_trend"] = long_term_trends
                else:
                    analysis_results["projections"]["long_term_trend"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for long-term trend analysis. "
                                 f"Required: {min_periods} per metric"
                    }
            except Exception as e:
                logger.error(f"Error in long-term trend analysis: {e}")
                analysis_results["projections"]["long_term_trend"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Perform scenario-based projection if enabled
        if self.config["scenario_projection"]["enabled"]:
            try:
                # Check if we have enough data points
                min_samples = self.config["scenario_projection"]["min_samples"]
                scenario_projections = {}
                
                for metric, data in time_series_data.items():
                    if len(data["values"]) >= min_samples:
                        # Perform scenario-based projection
                        scenario_results = self._generate_scenario_projections(
                            data["values"], data["timestamps"], metric)
                        
                        if scenario_results:
                            scenario_projections[metric] = scenario_results
                
                # Add to results if we have any projections
                if scenario_projections:
                    analysis_results["projections"]["scenario_projection"] = scenario_projections
                else:
                    analysis_results["projections"]["scenario_projection"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for scenario-based projection. "
                                 f"Required: {min_samples} per metric"
                    }
            except Exception as e:
                logger.error(f"Error in scenario-based projection: {e}")
                analysis_results["projections"]["scenario_projection"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Perform cyclical pattern analysis if enabled
        if self.config["cyclical_pattern"]["enabled"]:
            try:
                # Check if we have enough data points
                min_samples = self.config["cyclical_pattern"]["min_samples"]
                cyclical_patterns = {}
                
                for metric, data in time_series_data.items():
                    if len(data["values"]) >= min_samples:
                        # Perform cyclical pattern analysis
                        pattern_results = self._analyze_cyclical_patterns(
                            data["values"], data["timestamps"], metric)
                        
                        if pattern_results:
                            cyclical_patterns[metric] = pattern_results
                
                # Add to results if we have any patterns
                if cyclical_patterns:
                    analysis_results["projections"]["cyclical_pattern"] = cyclical_patterns
                else:
                    analysis_results["projections"]["cyclical_pattern"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for cyclical pattern analysis. "
                                 f"Required: {min_samples} per metric"
                    }
            except Exception as e:
                logger.error(f"Error in cyclical pattern analysis: {e}")
                analysis_results["projections"]["cyclical_pattern"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Perform trajectory analysis if enabled
        if self.config["trajectory_analysis"]["enabled"]:
            try:
                # Check if we have enough data points
                min_samples = self.config["trajectory_analysis"]["min_samples"]
                trajectories = {}
                
                for metric, data in time_series_data.items():
                    if len(data["values"]) >= min_samples:
                        # Perform trajectory analysis
                        trajectory_results = self._analyze_trajectory(
                            data["values"], data["timestamps"], metric)
                        
                        if trajectory_results:
                            trajectories[metric] = trajectory_results
                
                # Add to results if we have any trajectories
                if trajectories:
                    analysis_results["projections"]["trajectory_analysis"] = trajectories
                else:
                    analysis_results["projections"]["trajectory_analysis"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for trajectory analysis. "
                                 f"Required: {min_samples} per metric"
                    }
            except Exception as e:
                logger.error(f"Error in trajectory analysis: {e}")
                analysis_results["projections"]["trajectory_analysis"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Perform convergence projection if enabled
        if self.config["convergence_projection"]["enabled"]:
            try:
                # Check if we have enough data points
                min_samples = self.config["convergence_projection"]["min_samples"]
                convergence_results = {}
                
                for metric, data in time_series_data.items():
                    if len(data["values"]) >= min_samples:
                        # Perform convergence analysis
                        convergence = self._project_convergence(
                            data["values"], data["timestamps"], metric)
                        
                        if convergence:
                            convergence_results[metric] = convergence
                
                # Add to results if we have any convergence projections
                if convergence_results:
                    analysis_results["projections"]["convergence_projection"] = convergence_results
                else:
                    analysis_results["projections"]["convergence_projection"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for convergence projection. "
                                 f"Required: {min_samples} per metric"
                    }
            except Exception as e:
                logger.error(f"Error in convergence projection: {e}")
                analysis_results["projections"]["convergence_projection"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Generate key findings
        analysis_results["insights"]["key_findings"] = self._generate_key_findings(
            analysis_results["projections"],
            time_series_data
        )
        
        # Generate recommendations
        analysis_results["insights"]["recommendations"] = self._generate_recommendations(
            analysis_results["projections"],
            analysis_results["insights"]["key_findings"]
        )
        
        return analysis_results
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the capabilities of the trend projection.
        
        Returns:
            Dictionary describing the capabilities
        """
        return {
            "name": "Trend Projection",
            "description": "Projects long-term trends and patterns in simulation accuracy",
            "methods": [
                {
                    "name": "Long-term Trend Analysis",
                    "description": "Analyzes and forecasts long-term trends in accuracy metrics",
                    "enabled": self.config["long_term_trend"]["enabled"],
                    "horizon": self.config["long_term_trend"]["horizon"]
                },
                {
                    "name": "Scenario-based Projection",
                    "description": "Generates projections for best, expected, and worst case scenarios",
                    "enabled": self.config["scenario_projection"]["enabled"],
                    "scenarios": self.config["scenario_projection"]["scenarios"]
                },
                {
                    "name": "Cyclical Pattern Analysis",
                    "description": "Detects and projects cyclical patterns in accuracy metrics",
                    "enabled": self.config["cyclical_pattern"]["enabled"]
                },
                {
                    "name": "Trajectory Analysis",
                    "description": "Analyzes the trajectory of accuracy metrics over time",
                    "enabled": self.config["trajectory_analysis"]["enabled"]
                },
                {
                    "name": "Convergence Projection",
                    "description": "Projects when accuracy metrics will converge to stable values",
                    "enabled": self.config["convergence_projection"]["enabled"]
                }
            ],
            "output_format": {
                "projections": "Detailed projections for each analysis method",
                "insights": "Key findings and recommendations based on projections"
            }
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """
        Get information about the requirements of this analysis method.
        
        Returns:
            Dictionary describing the requirements
        """
        # Define minimum requirements
        requirements = {
            "min_validation_results": self.config["long_term_trend"]["min_history_periods"],
            "required_metrics": self.config["metrics_to_analyze"],
            "optimal_validation_results": 30,
            "time_series_required": True,
            "min_time_span_days": 7,  # Minimum time span for meaningful trend analysis
            "long_term_requirements": {
                "min_samples": self.config["long_term_trend"]["min_history_periods"]
            },
            "scenario_requirements": {
                "min_samples": self.config["scenario_projection"]["min_samples"]
            },
            "cyclical_requirements": {
                "min_samples": self.config["cyclical_pattern"]["min_samples"]
            },
            "trajectory_requirements": {
                "min_samples": self.config["trajectory_analysis"]["min_samples"]
            },
            "convergence_requirements": {
                "min_samples": self.config["convergence_projection"]["min_samples"]
            }
        }
        
        return requirements
    
    def _extract_time_series_data(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract time series data from validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary mapping metric names to time series data
        """
        # Initialize data structure
        time_series_data = {}
        
        # Get metrics to analyze
        metrics_to_analyze = self.config["metrics_to_analyze"]
        target_metrics = self.config["target_metrics"]
        
        # Process validation results
        for result in validation_results:
            # Skip if no validation timestamp
            if not hasattr(result, "validation_timestamp") or not result.validation_timestamp:
                continue
            
            # Parse timestamp
            try:
                if isinstance(result.validation_timestamp, str):
                    timestamp = datetime.datetime.fromisoformat(result.validation_timestamp)
                else:
                    timestamp = result.validation_timestamp
            except Exception as e:
                logger.warning(f"Error parsing timestamp: {e}")
                continue
            
            # Process metrics
            for performance_metric in metrics_to_analyze:
                if performance_metric in result.metrics_comparison:
                    for target_metric in target_metrics:
                        if target_metric in result.metrics_comparison[performance_metric]:
                            # Create key for this metric combination
                            metric_key = f"{performance_metric}_{target_metric}"
                            
                            # Initialize data structure if needed
                            if metric_key not in time_series_data:
                                time_series_data[metric_key] = {
                                    "values": [],
                                    "timestamps": []
                                }
                            
                            # Add data point
                            value = result.metrics_comparison[performance_metric][target_metric]
                            
                            # Skip NaN values
                            if value is None or np.isnan(value):
                                continue
                            
                            time_series_data[metric_key]["values"].append(value)
                            time_series_data[metric_key]["timestamps"].append(timestamp)
            
            # Process additional metrics
            if hasattr(result, "additional_metrics") and result.additional_metrics:
                for metric, value in result.additional_metrics.items():
                    if metric in target_metrics:
                        # Skip NaN values
                        if value is None or np.isnan(value):
                            continue
                        
                        # Initialize data structure if needed
                        if metric not in time_series_data:
                            time_series_data[metric] = {
                                "values": [],
                                "timestamps": []
                            }
                        
                        # Add data point
                        time_series_data[metric]["values"].append(value)
                        time_series_data[metric]["timestamps"].append(timestamp)
        
        # Sort time series by timestamp
        for metric, data in time_series_data.items():
            # Sort by timestamp
            sorted_indices = np.argsort(data["timestamps"])
            data["values"] = [data["values"][i] for i in sorted_indices]
            data["timestamps"] = [data["timestamps"][i] for i in sorted_indices]
        
        return time_series_data
    
    def _analyze_long_term_trend(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        metric: str
    ) -> Dict[str, Any]:
        """
        Analyze long-term trend in time series data.
        
        Args:
            values: List of metric values
            timestamps: List of timestamps
            metric: Metric name
            
        Returns:
            Dictionary with long-term trend analysis results
        """
        # Get configuration
        horizon = self.config["long_term_trend"]["horizon"]
        confidence_level = self.config["long_term_trend"]["confidence_level"]
        smoothing = self.config["long_term_trend"]["smoothing"]
        trend_detection_window = self.config["long_term_trend"]["trend_detection_window"]
        
        # Initialize results
        results = {
            "trend": {},
            "forecast": {},
            "slope": None,
            "p_value": None,
            "trend_direction": "stable",
            "significant": False
        }
        
        # Apply smoothing if enabled
        smoothed_values = self._smooth_time_series(values, trend_detection_window) if smoothing else values
        
        # Convert to numpy arrays
        values_array = np.array(smoothed_values)
        
        # Fit linear regression
        try:
            from scipy import stats
            
            # Create time index (days since first timestamp)
            first_timestamp = timestamps[0]
            time_index = [(ts - first_timestamp).total_seconds() / (24 * 3600) for ts in timestamps]
            time_array = np.array(time_index)
            
            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_array, values_array)
            
            # Store trend information
            results["slope"] = float(slope)
            results["intercept"] = float(intercept)
            results["r_squared"] = float(r_value ** 2)
            results["p_value"] = float(p_value)
            results["std_error"] = float(std_err)
            
            # Determine if trend is significant
            results["significant"] = p_value < 0.05
            
            # Determine trend direction
            if results["significant"]:
                if slope > 0:
                    results["trend_direction"] = "increasing"
                elif slope < 0:
                    results["trend_direction"] = "decreasing"
                else:
                    results["trend_direction"] = "stable"
            else:
                results["trend_direction"] = "stable"
            
            # Calculate average daily change
            if len(values) > 1:
                avg_daily_change = slope
                
                # Calculate percent change (relative to average value)
                avg_value = np.mean(values_array)
                if avg_value != 0:
                    percent_daily_change = (avg_daily_change / abs(avg_value)) * 100
                else:
                    percent_daily_change = 0.0
                
                results["trend"]["avg_daily_change"] = float(avg_daily_change)
                results["trend"]["percent_daily_change"] = float(percent_daily_change)
                
                # Calculate cumulative changes
                cumulative_changes = {
                    "30_day": float(avg_daily_change * 30),
                    "90_day": float(avg_daily_change * 90),
                    "180_day": float(avg_daily_change * 180),
                    "365_day": float(avg_daily_change * 365)
                }
                
                # Calculate percent cumulative changes
                percent_cumulative_changes = {}
                if avg_value != 0:
                    for period, change in cumulative_changes.items():
                        percent_cumulative_changes[period] = float((change / abs(avg_value)) * 100)
                else:
                    for period in cumulative_changes:
                        percent_cumulative_changes[period] = 0.0
                
                results["trend"]["cumulative_changes"] = cumulative_changes
                results["trend"]["percent_cumulative_changes"] = percent_cumulative_changes
            
            # Generate forecast
            if horizon > 0:
                # Create forecast time index (days after last timestamp)
                last_timestamp = timestamps[-1]
                last_day = time_array[-1]
                forecast_days = np.arange(1, horizon + 1) + last_day
                
                # Generate forecast dates
                forecast_dates = [
                    last_timestamp + datetime.timedelta(days=int(day - last_day))
                    for day in forecast_days
                ]
                
                # Predict values
                forecast_values = intercept + slope * forecast_days
                
                # Calculate prediction intervals
                # Using the formula: y_hat ± t * se * sqrt(1 + 1/n + (x - x_mean)²/ssx)
                t_value = stats.t.ppf((1 + confidence_level) / 2, len(time_array) - 2)
                se = np.sqrt(np.sum((values_array - (intercept + slope * time_array)) ** 2) / (len(time_array) - 2))
                x_mean = np.mean(time_array)
                ssx = np.sum((time_array - x_mean) ** 2)
                
                prediction_intervals = []
                for x in forecast_days:
                    margin = t_value * se * np.sqrt(1 + 1/len(time_array) + ((x - x_mean) ** 2) / ssx)
                    prediction_intervals.append(float(margin))
                
                # Create lower and upper bounds
                lower_bound = forecast_values - prediction_intervals
                upper_bound = forecast_values + prediction_intervals
                
                # Store forecast
                results["forecast"] = {
                    "horizon": horizon,
                    "values": forecast_values.tolist(),
                    "lower_bound": lower_bound.tolist(),
                    "upper_bound": upper_bound.tolist(),
                    "confidence_level": confidence_level,
                    "dates": [d.isoformat() for d in forecast_dates]
                }
        except Exception as e:
            logger.warning(f"Error in long-term trend analysis: {e}")
            return {}
        
        return results
    
    def _smooth_time_series(
        self,
        values: List[float],
        window_size: int
    ) -> List[float]:
        """
        Apply smoothing to a time series.
        
        Args:
            values: List of values to smooth
            window_size: Window size for moving average
            
        Returns:
            List of smoothed values
        """
        if window_size <= 1 or len(values) <= window_size:
            return values
        
        # Apply simple moving average
        smoothed = []
        for i in range(len(values)):
            if i < window_size - 1:
                # For the first few points, use available data
                window = values[:i+1]
            else:
                # Use full window
                window = values[i-window_size+1:i+1]
            
            smoothed.append(sum(window) / len(window))
        
        return smoothed
    
    def _generate_scenario_projections(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        metric: str
    ) -> Dict[str, Any]:
        """
        Generate scenario-based projections (best, expected, worst cases).
        
        Args:
            values: List of metric values
            timestamps: List of timestamps
            metric: Metric name
            
        Returns:
            Dictionary with scenario-based projections
        """
        # Get configuration
        scenarios = self.config["scenario_projection"]["scenarios"]
        percentiles = self.config["scenario_projection"]["percentiles"]
        horizon = self.config["long_term_trend"]["horizon"]  # Use same horizon as long-term trend
        
        # Initialize results
        results = {
            "scenarios": {},
            "projection_method": "bootstrap",
            "horizon": horizon
        }
        
        # Check if we have enough data
        if len(values) < 5:
            return {}
        
        try:
            # Generate forecast dates
            last_timestamp = timestamps[-1]
            forecast_dates = [
                last_timestamp + datetime.timedelta(days=i+1)
                for i in range(horizon)
            ]
            
            # Calculate historical changes
            changes = []
            for i in range(1, len(values)):
                change = values[i] - values[i-1]
                changes.append(change)
            
            # Calculate scenario bounds
            scenario_bounds = {}
            for scenario in scenarios:
                if scenario in percentiles:
                    percentile = percentiles[scenario]
                    if changes:
                        bound = np.percentile(changes, percentile * 100)
                        scenario_bounds[scenario] = bound
            
            # Generate scenario projections
            for scenario, bound in scenario_bounds.items():
                # Start from last value
                start_value = values[-1]
                
                # Generate projection
                projection = [start_value]
                for i in range(horizon):
                    next_value = projection[-1] + bound
                    projection.append(next_value)
                
                # Remove start value (which is the last actual value)
                projection = projection[1:]
                
                # Store scenario projection
                results["scenarios"][scenario] = {
                    "values": [float(v) for v in projection],
                    "dates": [d.isoformat() for d in forecast_dates],
                    "percentile": percentiles[scenario]
                }
        except Exception as e:
            logger.warning(f"Error generating scenario projections: {e}")
            return {}
        
        return results
    
    def _analyze_cyclical_patterns(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        metric: str
    ) -> Dict[str, Any]:
        """
        Analyze cyclical patterns in time series data.
        
        Args:
            values: List of metric values
            timestamps: List of timestamps
            metric: Metric name
            
        Returns:
            Dictionary with cyclical pattern analysis results
        """
        # Get configuration
        max_cycles = self.config["cyclical_pattern"]["max_cycles"]
        min_periods_per_cycle = self.config["cyclical_pattern"]["min_periods_per_cycle"]
        significance_threshold = self.config["cyclical_pattern"]["significance_threshold"]
        
        # Initialize results
        results = {
            "detected_cycles": [],
            "has_cyclical_pattern": False,
            "dominant_cycle": None,
            "projected_cycles": {}
        }
        
        # Check if we have enough data
        min_required = min_periods_per_cycle * 2  # Need at least 2 full cycles
        if len(values) < min_required:
            return {}
        
        try:
            # Convert to numpy array
            values_array = np.array(values)
            
            # Detrend the data to isolate cycles
            from scipy import signal
            detrended = signal.detrend(values_array)
            
            # Calculate autocorrelation
            autocorr = self._calculate_autocorrelation(detrended)
            
            # Find peaks in autocorrelation (potential cycles)
            peaks, properties = signal.find_peaks(autocorr, height=0.1, distance=min_periods_per_cycle)
            
            # Filter peaks and sort by prominence
            if len(peaks) > 0:
                # Get peak properties
                peak_heights = properties["peak_heights"]
                
                # Create peak data
                peak_data = []
                for i, peak in enumerate(peaks):
                    # Skip peak at lag 0
                    if peak == 0:
                        continue
                    
                    # Calculate p-value (simplified using Fisher's g-test approximation)
                    p_value = np.exp(-peak_heights[i])
                    
                    peak_data.append({
                        "lag": int(peak),
                        "height": float(peak_heights[i]),
                        "p_value": float(p_value),
                        "significant": p_value < significance_threshold
                    })
                
                # Sort by height (descending)
                peak_data.sort(key=lambda x: x["height"], reverse=True)
                
                # Keep only significant peaks
                significant_peaks = [p for p in peak_data if p["significant"]]
                
                # Store detected cycles
                detected_cycles = significant_peaks[:max_cycles]
                results["detected_cycles"] = detected_cycles
                results["has_cyclical_pattern"] = len(detected_cycles) > 0
                
                # Set dominant cycle
                if detected_cycles:
                    results["dominant_cycle"] = detected_cycles[0]
                
                # Project cycles
                if detected_cycles:
                    horizon = self.config["long_term_trend"]["horizon"]
                    last_timestamp = timestamps[-1]
                    forecast_dates = [
                        last_timestamp + datetime.timedelta(days=i+1)
                        for i in range(horizon)
                    ]
                    
                    # Project dominant cycle
                    dominant_cycle = detected_cycles[0]
                    cycle_length = dominant_cycle["lag"]
                    
                    # Calculate average cycle amplitude
                    cycle_points = []
                    for i in range(0, len(values) - cycle_length, cycle_length):
                        cycle = values[i:i+cycle_length]
                        cycle_points.append(cycle)
                    
                    if cycle_points:
                        # Calculate average cycle
                        avg_cycle = np.mean(cycle_points, axis=0)
                        
                        # Project cycle forward
                        last_values = values[-cycle_length:]
                        projection = list(last_values)
                        
                        for i in range(horizon):
                            # Use average cycle pattern
                            cycle_pos = i % cycle_length
                            next_value = projection[-cycle_length] * (avg_cycle[cycle_pos] / avg_cycle[0])
                            projection.append(next_value)
                        
                        # Remove initial values
                        projection = projection[cycle_length:]
                        
                        # Store projection
                        results["projected_cycles"]["dominant"] = {
                            "cycle_length": cycle_length,
                            "values": [float(v) for v in projection[:horizon]],
                            "dates": [d.isoformat() for d in forecast_dates]
                        }
            
            # Store autocorrelation
            results["autocorrelation"] = autocorr.tolist()
        
        except Exception as e:
            logger.warning(f"Error analyzing cyclical patterns: {e}")
            return {}
        
        return results
    
    def _calculate_autocorrelation(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate autocorrelation of a time series.
        
        Args:
            x: Time series data
            
        Returns:
            Autocorrelation values
        """
        # Ensure input is a numpy array
        x = np.array(x)
        
        # Remove mean
        x_mean = x - np.mean(x)
        
        # Calculate autocorrelation
        result = np.correlate(x_mean, x_mean, mode='full')
        result = result[len(result)//2:]  # Take only the positive lags
        
        # Normalize
        result = result / result[0]
        
        return result
    
    def _analyze_trajectory(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        metric: str
    ) -> Dict[str, Any]:
        """
        Analyze trajectory of time series data.
        
        Args:
            values: List of metric values
            timestamps: List of timestamps
            metric: Metric name
            
        Returns:
            Dictionary with trajectory analysis results
        """
        # Get configuration
        smoothing_window = self.config["trajectory_analysis"]["smoothing_window"]
        change_point_threshold = self.config["trajectory_analysis"]["change_point_threshold"]
        
        # Initialize results
        results = {
            "trajectory_type": "unknown",
            "change_points": [],
            "segments": [],
            "overall_trend": "stable"
        }
        
        # Check if we have enough data
        if len(values) < smoothing_window * 2:
            return {}
        
        try:
            # Apply smoothing
            smoothed_values = self._smooth_time_series(values, smoothing_window)
            
            # Calculate changes
            changes = []
            for i in range(1, len(smoothed_values)):
                change = smoothed_values[i] - smoothed_values[i-1]
                changes.append(change)
            
            # Detect change points
            change_points = []
            for i in range(1, len(changes)):
                # Check if direction changed
                if (changes[i] > 0 and changes[i-1] < 0) or (changes[i] < 0 and changes[i-1] > 0):
                    # Check if change is significant
                    if abs(changes[i] - changes[i-1]) > change_point_threshold * np.std(changes):
                        change_points.append(i + 1)  # +1 to account for offset in changes array
            
            # Add first and last points as segment boundaries
            segment_boundaries = [0] + change_points + [len(smoothed_values) - 1]
            
            # Analyze segments
            segments = []
            for i in range(len(segment_boundaries) - 1):
                start_idx = segment_boundaries[i]
                end_idx = segment_boundaries[i + 1]
                
                # Skip segments that are too short
                if end_idx - start_idx < smoothing_window:
                    continue
                
                # Get segment values
                segment_values = smoothed_values[start_idx:end_idx+1]
                segment_timestamps = timestamps[start_idx:end_idx+1]
                
                # Calculate trend in segment
                trend = self._calculate_segment_trend(segment_values, segment_timestamps)
                
                # Add segment info
                segments.append({
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "length": end_idx - start_idx + 1,
                    "start_value": float(segment_values[0]),
                    "end_value": float(segment_values[-1]),
                    "start_date": segment_timestamps[0].isoformat(),
                    "end_date": segment_timestamps[-1].isoformat(),
                    "trend": trend
                })
            
            # Store change points and segments
            results["change_points"] = change_points
            results["segments"] = segments
            
            # Determine overall trend from the last segment
            if segments:
                last_segment = segments[-1]
                results["overall_trend"] = last_segment["trend"]
            
            # Determine trajectory type
            if not change_points:
                # No change points, single segment
                results["trajectory_type"] = f"consistent_{results['overall_trend']}"
            elif len(change_points) == 1:
                # One change point, two segments
                if segments[0]["trend"] != segments[1]["trend"]:
                    results["trajectory_type"] = f"change_from_{segments[0]['trend']}_to_{segments[1]['trend']}"
                else:
                    results["trajectory_type"] = f"interrupted_{segments[0]['trend']}"
            else:
                # Multiple change points
                trend_counts = {
                    "increasing": 0,
                    "decreasing": 0,
                    "stable": 0
                }
                
                for segment in segments:
                    trend_counts[segment["trend"]] += 1
                
                if trend_counts["increasing"] > trend_counts["decreasing"] and trend_counts["increasing"] > trend_counts["stable"]:
                    results["trajectory_type"] = "predominantly_increasing"
                elif trend_counts["decreasing"] > trend_counts["increasing"] and trend_counts["decreasing"] > trend_counts["stable"]:
                    results["trajectory_type"] = "predominantly_decreasing"
                elif trend_counts["stable"] > trend_counts["increasing"] and trend_counts["stable"] > trend_counts["decreasing"]:
                    results["trajectory_type"] = "predominantly_stable"
                else:
                    results["trajectory_type"] = "fluctuating"
            
            # Add smoothed values
            results["smoothed_values"] = [float(v) for v in smoothed_values]
        
        except Exception as e:
            logger.warning(f"Error analyzing trajectory: {e}")
            return {}
        
        return results
    
    def _calculate_segment_trend(
        self,
        values: List[float],
        timestamps: List[datetime.datetime]
    ) -> str:
        """
        Calculate trend in a segment.
        
        Args:
            values: List of segment values
            timestamps: List of segment timestamps
            
        Returns:
            Trend direction ("increasing", "decreasing", or "stable")
        """
        # Convert to numpy arrays
        values_array = np.array(values)
        
        # Create time index (days since first timestamp)
        first_timestamp = timestamps[0]
        time_index = [(ts - first_timestamp).total_seconds() / (24 * 3600) for ts in timestamps]
        time_array = np.array(time_index)
        
        # Fit linear regression
        try:
            from scipy import stats
            slope, _, _, p_value, _ = stats.linregress(time_array, values_array)
            
            # Determine if trend is significant
            significant = p_value < 0.05
            
            # Determine trend direction
            if significant:
                if slope > 0:
                    return "increasing"
                elif slope < 0:
                    return "decreasing"
                else:
                    return "stable"
            else:
                return "stable"
        except:
            # Fallback to simple comparison
            if values[-1] > values[0] * 1.05:  # 5% increase
                return "increasing"
            elif values[-1] < values[0] * 0.95:  # 5% decrease
                return "decreasing"
            else:
                return "stable"
    
    def _project_convergence(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        metric: str
    ) -> Dict[str, Any]:
        """
        Project convergence of time series data.
        
        Args:
            values: List of metric values
            timestamps: List of timestamps
            metric: Metric name
            
        Returns:
            Dictionary with convergence projection results
        """
        # Get configuration
        convergence_threshold = self.config["convergence_projection"]["convergence_threshold"]
        stability_window = self.config["convergence_projection"]["stability_window"]
        
        # Initialize results
        results = {
            "is_converging": False,
            "is_stable": False,
            "convergence_estimate": None,
            "stability_metrics": {},
            "projected_convergence": {}
        }
        
        # Check if we have enough data
        if len(values) < stability_window:
            return {}
        
        try:
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(values, stability_window)
            results["stability_metrics"] = stability_metrics
            
            # Check if already stable
            if stability_metrics["relative_range"] <= convergence_threshold:
                results["is_stable"] = True
                results["convergence_estimate"] = 0  # Already converged
            
            # Check for convergence
            if len(values) >= stability_window * 2:
                # Compare variability in first and last windows
                first_window = values[:stability_window]
                last_window = values[-stability_window:]
                
                first_range = max(first_window) - min(first_window)
                last_range = max(last_window) - min(last_window)
                
                if first_range > 0 and last_range < first_range:
                    # Calculate convergence rate
                    results["is_converging"] = True
                    
                    # Fit exponential decay model to predict convergence
                    try:
                        # Convert to numpy arrays
                        values_array = np.array(values)
                        
                        # Create time index (days since first timestamp)
                        first_timestamp = timestamps[0]
                        time_index = [(ts - first_timestamp).total_seconds() / (24 * 3600) for ts in timestamps]
                        time_array = np.array(time_index)
                        
                        # Calculate moving range
                        moving_range = []
                        for i in range(len(values) - stability_window + 1):
                            window = values[i:i+stability_window]
                            window_range = max(window) - min(window)
                            moving_range.append(window_range)
                        
                        # Fit exponential decay: range(t) = a * exp(-b * t) + c
                        from scipy.optimize import curve_fit
                        
                        # Define exponential decay function
                        def exp_decay(t, a, b, c):
                            return a * np.exp(-b * t) + c
                        
                        # Fit model
                        window_time = time_array[:len(moving_range)]
                        params, _ = curve_fit(
                            exp_decay, window_time, moving_range,
                            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                            maxfev=10000
                        )
                        
                        a, b, c = params
                        
                        # Project convergence
                        days_ahead = np.arange(1, 366)  # Project up to 1 year ahead
                        last_day = time_array[-1]
                        projection_days = last_day + days_ahead
                        
                        projected_range = exp_decay(projection_days, a, b, c)
                        
                        # Find days until convergence
                        avg_value = np.mean(values)
                        threshold_value = avg_value * convergence_threshold
                        
                        convergence_day = None
                        for i, day in enumerate(days_ahead):
                            if projected_range[i] <= threshold_value:
                                convergence_day = day
                                break
                        
                        # Store results
                        if convergence_day is not None:
                            results["convergence_estimate"] = int(convergence_day)
                            results["projected_convergence"]["days"] = int(convergence_day)
                            results["projected_convergence"]["date"] = (
                                timestamps[-1] + datetime.timedelta(days=int(convergence_day))
                            ).isoformat()
                        
                        # Store model parameters
                        results["projected_convergence"]["model"] = {
                            "type": "exponential_decay",
                            "parameters": {
                                "a": float(a),
                                "b": float(b),
                                "c": float(c)
                            }
                        }
                    except Exception as e:
                        logger.warning(f"Error fitting convergence model: {e}")
            
            # Check for divergence
            if len(values) >= stability_window * 2:
                # Compare variability in first and last windows
                first_window = values[:stability_window]
                last_window = values[-stability_window:]
                
                first_range = max(first_window) - min(first_window)
                last_range = max(last_window) - min(last_window)
                
                if last_range > first_range * 1.5:  # 50% increase in range
                    results["is_diverging"] = True
        
        except Exception as e:
            logger.warning(f"Error projecting convergence: {e}")
            return {}
        
        return results
    
    def _calculate_stability_metrics(
        self,
        values: List[float],
        window_size: int
    ) -> Dict[str, float]:
        """
        Calculate stability metrics for a time series.
        
        Args:
            values: List of values
            window_size: Window size for stability assessment
            
        Returns:
            Dictionary with stability metrics
        """
        # Use the most recent window
        window = values[-window_size:]
        
        # Calculate statistics
        mean = np.mean(window)
        std_dev = np.std(window)
        min_val = min(window)
        max_val = max(window)
        range_val = max_val - min_val
        
        # Calculate normalized metrics
        if mean != 0:
            cv = std_dev / abs(mean)  # Coefficient of variation
            relative_range = range_val / abs(mean)  # Range relative to mean
        else:
            cv = 0.0
            relative_range = 0.0
        
        return {
            "mean": float(mean),
            "std_dev": float(std_dev),
            "min": float(min_val),
            "max": float(max_val),
            "range": float(range_val),
            "cv": float(cv),
            "relative_range": float(relative_range)
        }
    
    def _generate_key_findings(
        self,
        projections: Dict[str, Dict[str, Any]],
        time_series_data: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Generate key findings based on trend projections.
        
        Args:
            projections: Dictionary with projection results
            time_series_data: Dictionary with time series data
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Skip if no projections
        if not projections:
            findings.append("Insufficient data for trend projection analysis")
            return findings
        
        # Process long-term trend projections
        if "long_term_trend" in projections:
            long_term_trends = projections["long_term_trend"]
            
            if isinstance(long_term_trends, dict) and "status" not in long_term_trends:
                for metric, trend in long_term_trends.items():
                    if "trend_direction" in trend and trend.get("significant", False):
                        direction = trend["trend_direction"]
                        
                        # Get percent change info
                        if "trend" in trend and "percent_cumulative_changes" in trend["trend"]:
                            changes = trend["trend"]["percent_cumulative_changes"]
                            if "90_day" in changes:
                                percent_change = changes["90_day"]
                                
                                findings.append(
                                    f"{metric} shows a significant {direction} trend "
                                    f"({percent_change:.1f}% change projected over 90 days)"
                                )
        
        # Process scenario projections
        if "scenario_projection" in projections:
            scenario_projections = projections["scenario_projection"]
            
            if isinstance(scenario_projections, dict) and "status" not in scenario_projections:
                for metric, scenarios in scenario_projections.items():
                    if "scenarios" in scenarios:
                        for scenario_name, scenario in scenarios["scenarios"].items():
                            if "values" in scenario and len(scenario["values"]) > 0:
                                # Calculate percent change
                                first_value = scenario["values"][0]
                                last_value = scenario["values"][-1]
                                
                                if first_value != 0:
                                    percent_change = (last_value - first_value) / abs(first_value) * 100
                                    
                                    # Only add significant changes
                                    if abs(percent_change) >= 10:  # 10% change or more
                                        scenario_display = scenario_name.replace("_", " ")
                                        findings.append(
                                            f"{scenario_display} scenario for {metric} projects "
                                            f"a {percent_change:.1f}% change over the forecast period"
                                        )
        
        # Process cyclical pattern analysis
        if "cyclical_pattern" in projections:
            cyclical_patterns = projections["cyclical_pattern"]
            
            if isinstance(cyclical_patterns, dict) and "status" not in cyclical_patterns:
                for metric, pattern in cyclical_patterns.items():
                    if pattern.get("has_cyclical_pattern", False) and "dominant_cycle" in pattern:
                        dominant_cycle = pattern["dominant_cycle"]
                        lag = dominant_cycle.get("lag", 0)
                        
                        findings.append(
                            f"{metric} shows a cyclical pattern with a {lag}-day cycle"
                        )
        
        # Process trajectory analysis
        if "trajectory_analysis" in projections:
            trajectories = projections["trajectory_analysis"]
            
            if isinstance(trajectories, dict) and "status" not in trajectories:
                for metric, trajectory in trajectories.items():
                    if "trajectory_type" in trajectory and "overall_trend" in trajectory:
                        trajectory_type = trajectory["trajectory_type"]
                        overall_trend = trajectory["overall_trend"]
                        
                        findings.append(
                            f"{metric} shows a {trajectory_type} trajectory "
                            f"with a current {overall_trend} trend"
                        )
        
        # Process convergence projections
        if "convergence_projection" in projections:
            convergence_results = projections["convergence_projection"]
            
            if isinstance(convergence_results, dict) and "status" not in convergence_results:
                for metric, convergence in convergence_results.items():
                    if convergence.get("is_converging", False) and "convergence_estimate" in convergence:
                        days = convergence["convergence_estimate"]
                        
                        if days > 0:
                            findings.append(
                                f"{metric} is converging and projected to stabilize "
                                f"in approximately {days} days"
                            )
                        else:
                            findings.append(
                                f"{metric} has already converged to a stable value"
                            )
                    elif convergence.get("is_stable", False):
                        findings.append(
                            f"{metric} is already stable with minimal variation"
                        )
                    elif convergence.get("is_diverging", False):
                        findings.append(
                            f"{metric} is showing increasing variability and may be diverging"
                        )
        
        # If no specific findings, add general finding about data
        if not findings:
            # Count available metrics
            metric_count = len(time_series_data)
            avg_data_points = np.mean([len(data["values"]) for _, data in time_series_data.items()])
            
            findings.append(
                f"Analysis completed on {metric_count} metrics with an average of "
                f"{avg_data_points:.1f} data points per metric"
            )
        
        return findings
    
    def _generate_recommendations(
        self,
        projections: Dict[str, Dict[str, Any]],
        key_findings: List[str]
    ) -> List[str]:
        """
        Generate recommendations based on trend projections.
        
        Args:
            projections: Dictionary with projection results
            key_findings: List of key findings
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Skip if no projections
        if not projections:
            recommendations.append("Collect more validation data for trend projection analysis")
            return recommendations
        
        # Track metrics with concerning trends
        concerning_metrics = {}
        
        # Process long-term trend projections
        if "long_term_trend" in projections:
            long_term_trends = projections["long_term_trend"]
            
            if isinstance(long_term_trends, dict) and "status" not in long_term_trends:
                for metric, trend in long_term_trends.items():
                    if "trend_direction" in trend and trend.get("significant", False):
                        direction = trend["trend_direction"]
                        
                        # Check for concerning trends (increasing error metrics)
                        if direction == "increasing" and "mape" in metric.lower():
                            # Get percent change info
                            if "trend" in trend and "percent_cumulative_changes" in trend["trend"]:
                                changes = trend["trend"]["percent_cumulative_changes"]
                                if "90_day" in changes:
                                    percent_change = changes["90_day"]
                                    
                                    # Only consider significant increases
                                    if percent_change > 10:  # More than 10% increase
                                        concerning_metrics[metric] = {
                                            "trend": "increasing",
                                            "percent_change": percent_change
                                        }
        
        # Process convergence projections
        if "convergence_projection" in projections:
            convergence_results = projections["convergence_projection"]
            
            if isinstance(convergence_results, dict) and "status" not in convergence_results:
                for metric, convergence in convergence_results.items():
                    if convergence.get("is_diverging", False):
                        # Track diverging metrics
                        concerning_metrics[metric] = {
                            "trend": "diverging"
                        }
        
        # Add recommendations based on concerning metrics
        if concerning_metrics:
            # Sort by percent change (highest first)
            sorted_metrics = sorted(
                concerning_metrics.items(),
                key=lambda x: x[1].get("percent_change", 0),
                reverse=True
            )
            
            # Add recommendation for most concerning metric
            worst_metric, worst_info = sorted_metrics[0]
            if worst_info["trend"] == "increasing":
                recommendations.append(
                    f"Prioritize improvement of {worst_metric} as it shows a significant "
                    f"increasing trend ({worst_info['percent_change']:.1f}% projected increase)"
                )
            elif worst_info["trend"] == "diverging":
                recommendations.append(
                    f"Investigate {worst_metric} immediately as it shows increasing "
                    f"variability and may be unstable"
                )
            
            # Add general recommendation if multiple metrics are concerning
            if len(concerning_metrics) > 1:
                metrics_list = ", ".join([m[0] for m in sorted_metrics])
                recommendations.append(
                    f"Establish a monitoring system for these metrics with concerning trends: {metrics_list}"
                )
        
        # Add recommendations based on cyclical patterns
        if "cyclical_pattern" in projections:
            cyclical_patterns = projections["cyclical_pattern"]
            
            if isinstance(cyclical_patterns, dict) and "status" not in cyclical_patterns:
                cyclical_metrics = []
                
                for metric, pattern in cyclical_patterns.items():
                    if pattern.get("has_cyclical_pattern", False) and "dominant_cycle" in pattern:
                        dominant_cycle = pattern["dominant_cycle"]
                        lag = dominant_cycle.get("lag", 0)
                        
                        if lag > 0:
                            cyclical_metrics.append((metric, lag))
                
                if cyclical_metrics:
                    metric, lag = cyclical_metrics[0]
                    recommendations.append(
                        f"Align validation data collection with the {lag}-day cycle "
                        f"detected in {metric} for more consistent results"
                    )
        
        # Add recommendations based on convergence projections
        if "convergence_projection" in projections:
            convergence_results = projections["convergence_projection"]
            
            if isinstance(convergence_results, dict) and "status" not in convergence_results:
                converging_metrics = []
                
                for metric, convergence in convergence_results.items():
                    if convergence.get("is_converging", False) and "convergence_estimate" in convergence:
                        days = convergence["convergence_estimate"]
                        
                        if days > 0:
                            converging_metrics.append((metric, days))
                
                if converging_metrics:
                    # Sort by days to convergence (lowest first)
                    converging_metrics.sort(key=lambda x: x[1])
                    
                    metric, days = converging_metrics[0]
                    recommendations.append(
                        f"Continue current validation approach for {metric} as it is "
                        f"projected to stabilize in {days} days"
                    )
        
        # Add general recommendations
        if "long_term_trend" in projections:
            recommendations.append(
                "Establish regular trend analysis (monthly or quarterly) "
                "to track accuracy metrics over time"
            )
        
        if "scenario_projection" in projections:
            recommendations.append(
                "Develop contingency plans for the worst-case scenarios "
                "identified in the projection analysis"
            )
        
        # If no specific recommendations, add general recommendation
        if not recommendations:
            recommendations.append(
                "Continue collecting validation data to enable more "
                "detailed trend projection analysis"
            )
        
        return recommendations