#!/usr/bin/env python3
"""
Advanced Statistical Analysis for the Simulation Accuracy and Validation Framework.

This module provides advanced statistical analysis methods beyond the basic metrics
already available in the statistical validator, including:
- Bayesian analysis for simulation accuracy assessment
- Advanced time series analysis for accuracy trends
- Multivariate analysis for complex relationships
- Effect size calculations and more precise statistical tests
- Advanced correlation analysis and non-linear relationships
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analysis.advanced_statistical")

# Import base class
from duckdb_api.simulation_validation.analysis.base import AnalysisMethod
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

class AdvancedStatisticalAnalysis(AnalysisMethod):
    """
    Advanced statistical analysis methods for simulation validation results.
    
    This class extends the basic AnalysisMethod to provide sophisticated
    statistical techniques for analyzing simulation accuracy, including:
    - Bayesian analysis for more nuanced accuracy assessment
    - Advanced time series analysis for identifying accuracy trends
    - Multivariate analysis to understand complex relationships between metrics
    - Effect size calculations beyond simple error metrics
    - Advanced correlation analysis including non-linear relationships
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced statistical analysis method.
        
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
            
            # Bayesian analysis configuration
            "bayesian_analysis": {
                "enabled": True,
                "prior_type": "uninformative",  # uninformative, informative, or custom
                "num_samples": 1000,           # MCMC samples for Bayesian analysis
                "credible_interval": 0.95      # 95% credible interval
            },
            
            # Time series analysis configuration
            "time_series_analysis": {
                "enabled": True,
                "smoothing_method": "exponential",  # none, moving_average, exponential
                "smoothing_window": 5,          # Window size for smoothing
                "detrending": False,           # Whether to remove trend
                "seasonal_decomposition": True, # Whether to decompose seasonal components
                "anomaly_detection": True,      # Whether to detect anomalies in time series
                "min_points_required": 5       # Minimum points required for time series analysis
            },
            
            # Multivariate analysis configuration
            "multivariate_analysis": {
                "enabled": True,
                "methods": ["pca", "factor_analysis", "cluster_analysis"],
                "correlation_threshold": 0.7,   # Threshold for highlighting strong correlations
                "min_points_required": 5       # Minimum points required for multivariate analysis
            },
            
            # Effect size analysis configuration
            "effect_size_analysis": {
                "enabled": True,
                "methods": ["cohen_d", "hedges_g", "glass_delta"],
                "interpretation_thresholds": {
                    "small": 0.2,
                    "medium": 0.5,
                    "large": 0.8
                }
            },
            
            # Advanced correlation analysis
            "correlation_analysis": {
                "enabled": True,
                "methods": ["pearson", "spearman", "kendall", "distance_correlation"],
                "non_linear_methods": ["mutual_information", "maximal_information_coefficient"],
                "visualization": True
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
        Perform advanced statistical analysis on validation results.
        
        Args:
            validation_results: List of validation results to analyze
            
        Returns:
            Dictionary containing statistical analysis results and insights
        """
        # Check requirements
        meets_req, error_msg = self.check_requirements(validation_results)
        if not meets_req:
            logger.warning(f"Requirements not met for advanced statistical analysis: {error_msg}")
            return {"status": "error", "message": error_msg}
        
        # Initialize results dictionary
        analysis_results = {
            "status": "success",
            "timestamp": datetime.datetime.now().isoformat(),
            "num_validation_results": len(validation_results),
            "metrics_analyzed": self.config["metrics_to_analyze"],
            "analysis_methods": {},
            "insights": {
                "key_findings": [],
                "recommendations": []
            }
        }
        
        # Extract metrics from validation results
        metrics_data = self._extract_metrics_data(validation_results)
        
        # Perform Bayesian analysis if enabled
        if self.config["bayesian_analysis"]["enabled"]:
            try:
                bayesian_results = self._perform_bayesian_analysis(
                    validation_results, metrics_data)
                analysis_results["analysis_methods"]["bayesian_analysis"] = bayesian_results
                
                # Add key findings from Bayesian analysis
                for metric, result in bayesian_results.items():
                    if "credible_interval" in result:
                        ci = result["credible_interval"]
                        finding = (f"Bayesian analysis shows {metric} has a "
                                 f"{self.config['bayesian_analysis']['credible_interval']*100:.0f}% "
                                 f"credible interval of {ci['lower']:.2f} to {ci['upper']:.2f}")
                        analysis_results["insights"]["key_findings"].append(finding)
            except Exception as e:
                logger.error(f"Error performing Bayesian analysis: {e}")
                analysis_results["analysis_methods"]["bayesian_analysis"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Perform time series analysis if enabled
        if self.config["time_series_analysis"]["enabled"]:
            try:
                # Check if we have enough data points for time series analysis
                min_points = self.config["time_series_analysis"]["min_points_required"]
                if len(validation_results) >= min_points:
                    time_series_results = self._perform_time_series_analysis(
                        validation_results, metrics_data)
                    analysis_results["analysis_methods"]["time_series_analysis"] = time_series_results
                    
                    # Add key findings from time series analysis
                    if "trends" in time_series_results:
                        for metric, trend in time_series_results["trends"].items():
                            if trend["significant"]:
                                direction = "improving" if trend["direction"] == "decreasing" else "degrading"
                                finding = (f"Time series analysis shows a significant {trend['direction']} "
                                         f"trend in {metric}, indicating {direction} simulation accuracy")
                                analysis_results["insights"]["key_findings"].append(finding)
                else:
                    analysis_results["analysis_methods"]["time_series_analysis"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for time series analysis. "
                                f"Required: {min_points}, Provided: {len(validation_results)}"
                    }
            except Exception as e:
                logger.error(f"Error performing time series analysis: {e}")
                analysis_results["analysis_methods"]["time_series_analysis"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Perform multivariate analysis if enabled
        if self.config["multivariate_analysis"]["enabled"]:
            try:
                # Check if we have enough data points for multivariate analysis
                min_points = self.config["multivariate_analysis"]["min_points_required"]
                if len(validation_results) >= min_points:
                    multivariate_results = self._perform_multivariate_analysis(
                        validation_results, metrics_data)
                    analysis_results["analysis_methods"]["multivariate_analysis"] = multivariate_results
                    
                    # Add key findings from multivariate analysis
                    if "correlations" in multivariate_results:
                        for corr in multivariate_results["correlations"]:
                            if corr["strength"] == "strong":
                                finding = (f"Strong correlation ({corr['coefficient']:.2f}) "
                                         f"detected between {corr['metrics'][0]} and {corr['metrics'][1]}")
                                analysis_results["insights"]["key_findings"].append(finding)
                else:
                    analysis_results["analysis_methods"]["multivariate_analysis"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for multivariate analysis. "
                                f"Required: {min_points}, Provided: {len(validation_results)}"
                    }
            except Exception as e:
                logger.error(f"Error performing multivariate analysis: {e}")
                analysis_results["analysis_methods"]["multivariate_analysis"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Perform effect size analysis if enabled
        if self.config["effect_size_analysis"]["enabled"]:
            try:
                effect_size_results = self._perform_effect_size_analysis(
                    validation_results, metrics_data)
                analysis_results["analysis_methods"]["effect_size_analysis"] = effect_size_results
                
                # Add key findings from effect size analysis
                for metric, result in effect_size_results.items():
                    if "cohen_d" in result and result["cohen_d"]["magnitude"] == "large":
                        finding = (f"Large effect size (Cohen's d = {result['cohen_d']['value']:.2f}) "
                                 f"detected for {metric}, indicating substantial difference "
                                 f"between simulation and hardware")
                        analysis_results["insights"]["key_findings"].append(finding)
            except Exception as e:
                logger.error(f"Error performing effect size analysis: {e}")
                analysis_results["analysis_methods"]["effect_size_analysis"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Perform correlation analysis if enabled
        if self.config["correlation_analysis"]["enabled"]:
            try:
                correlation_results = self._perform_correlation_analysis(
                    validation_results, metrics_data)
                analysis_results["analysis_methods"]["correlation_analysis"] = correlation_results
                
                # Add key findings from correlation analysis
                if "non_linear_relationships" in correlation_results:
                    for rel in correlation_results["non_linear_relationships"]:
                        if rel["strength"] == "strong":
                            finding = (f"Strong non-linear relationship detected between "
                                     f"{rel['metrics'][0]} and {rel['metrics'][1]} "
                                     f"(MI = {rel['mutual_information']:.2f})")
                            analysis_results["insights"]["key_findings"].append(finding)
            except Exception as e:
                logger.error(f"Error performing correlation analysis: {e}")
                analysis_results["analysis_methods"]["correlation_analysis"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Generate overall recommendations based on analysis results
        try:
            recommendations = self._generate_recommendations(analysis_results)
            analysis_results["insights"]["recommendations"] = recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            analysis_results["insights"]["recommendations"] = [
                "Error generating recommendations: " + str(e)
            ]
        
        return analysis_results
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the capabilities of the advanced statistical analysis.
        
        Returns:
            Dictionary describing the capabilities
        """
        return {
            "name": "Advanced Statistical Analysis",
            "description": "Provides advanced statistical methods beyond basic metrics",
            "methods": [
                {
                    "name": "Bayesian Analysis",
                    "description": "Uses Bayesian statistics for more nuanced accuracy assessment",
                    "enabled": self.config["bayesian_analysis"]["enabled"]
                },
                {
                    "name": "Time Series Analysis",
                    "description": "Analyzes accuracy trends and patterns over time",
                    "enabled": self.config["time_series_analysis"]["enabled"]
                },
                {
                    "name": "Multivariate Analysis",
                    "description": "Uncovers complex relationships between multiple metrics",
                    "enabled": self.config["multivariate_analysis"]["enabled"]
                },
                {
                    "name": "Effect Size Analysis",
                    "description": "Calculates standardized effect sizes for better comparison",
                    "enabled": self.config["effect_size_analysis"]["enabled"]
                },
                {
                    "name": "Advanced Correlation Analysis",
                    "description": "Detects linear and non-linear relationships between metrics",
                    "enabled": self.config["correlation_analysis"]["enabled"]
                }
            ],
            "output_format": {
                "analysis_results": "Dictionary with analysis results for each method",
                "insights": "Key findings and recommendations based on the analysis"
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
            "min_validation_results": 1,
            "required_metrics": self.config["metrics_to_analyze"],
            "optimal_validation_results": 10,
            "time_series_requirements": {
                "min_points": self.config["time_series_analysis"]["min_points_required"],
                "time_series_required": self.config["time_series_analysis"]["enabled"]
            },
            "multivariate_requirements": {
                "min_points": self.config["multivariate_analysis"]["min_points_required"],
                "min_metrics": 2
            }
        }
        
        return requirements
    
    def _extract_metrics_data(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract metrics data from validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary mapping metric names to dictionaries of simulation and hardware values
        """
        metrics_data = {}
        
        # Define which metrics to extract
        metrics_to_extract = self.config["metrics_to_analyze"]
        
        # Initialize data structure
        for metric in metrics_to_extract:
            metrics_data[metric] = {
                "simulation": [],
                "hardware": [],
                "error": [],
                "timestamp": []
            }
        
        # Extract metrics from validation results
        for result in validation_results:
            for metric in metrics_to_extract:
                # Skip if metric is not present in both simulation and hardware results
                if (metric not in result.simulation_result.metrics or
                    metric not in result.hardware_result.metrics):
                    continue
                
                # Get values
                sim_val = result.simulation_result.metrics[metric]
                hw_val = result.hardware_result.metrics[metric]
                
                # Skip if either value is None
                if sim_val is None or hw_val is None:
                    continue
                
                # Store values
                metrics_data[metric]["simulation"].append(sim_val)
                metrics_data[metric]["hardware"].append(hw_val)
                
                # Calculate and store error
                error = abs(sim_val - hw_val)
                metrics_data[metric]["error"].append(error)
                
                # Store timestamp if available
                if hasattr(result, "validation_timestamp") and result.validation_timestamp:
                    try:
                        metrics_data[metric]["timestamp"].append(result.validation_timestamp)
                    except Exception:
                        # Use index as timestamp if parsing fails
                        metrics_data[metric]["timestamp"].append(len(metrics_data[metric]["timestamp"]))
                else:
                    # Use index as timestamp if not available
                    metrics_data[metric]["timestamp"].append(len(metrics_data[metric]["timestamp"]))
        
        return metrics_data
    
    def _perform_bayesian_analysis(
        self,
        validation_results: List[ValidationResult],
        metrics_data: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Any]:
        """
        Perform Bayesian analysis on validation results.
        
        Args:
            validation_results: List of validation results
            metrics_data: Extracted metrics data
            
        Returns:
            Dictionary with Bayesian analysis results
        """
        bayesian_results = {}
        
        # Try to use scipy for statistics
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available for Bayesian analysis")
            return {"status": "error", "message": "scipy not available"}
        
        # Process each metric
        for metric, data in metrics_data.items():
            if not data["error"]:
                continue
            
            # Calculate mean and standard deviation of errors
            mean_error = np.mean(data["error"])
            std_error = np.std(data["error"])
            
            # Get credible interval level
            credible_level = self.config["bayesian_analysis"]["credible_interval"]
            
            # Calculate credible interval for error
            lower = stats.norm.ppf((1 - credible_level) / 2, loc=mean_error, scale=std_error)
            upper = stats.norm.ppf((1 + credible_level) / 2, loc=mean_error, scale=std_error)
            
            # Store results
            bayesian_results[metric] = {
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "credible_interval": {
                    "level": float(credible_level),
                    "lower": float(max(0, lower)),  # Error can't be negative
                    "upper": float(upper)
                }
            }
            
            # For more sophisticated Bayesian analysis, additional libraries like PyMC3
            # or ArviZ would be needed, but we're implementing a simplified version here
        
        return bayesian_results
    
    def _perform_time_series_analysis(
        self,
        validation_results: List[ValidationResult],
        metrics_data: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Any]:
        """
        Perform time series analysis on validation results.
        
        Args:
            validation_results: List of validation results
            metrics_data: Extracted metrics data
            
        Returns:
            Dictionary with time series analysis results
        """
        time_series_results = {
            "trends": {},
            "seasonality": {},
            "anomalies": {}
        }
        
        # Process each metric
        for metric, data in metrics_data.items():
            # Skip if not enough data points
            if len(data["error"]) < self.config["time_series_analysis"]["min_points_required"]:
                continue
            
            # Check if we have timestamps
            has_timestamps = (
                "timestamp" in data and 
                len(data["timestamp"]) == len(data["error"]) and
                not all(isinstance(ts, int) for ts in data["timestamp"])
            )
            
            # Convert timestamps to datetime objects if they are strings
            if has_timestamps:
                try:
                    timestamps = []
                    for ts in data["timestamp"]:
                        if isinstance(ts, str):
                            timestamps.append(datetime.datetime.fromisoformat(ts))
                        else:
                            timestamps.append(ts)
                    
                    # Sort data by timestamp
                    sorted_indices = np.argsort(timestamps)
                    errors = np.array(data["error"])[sorted_indices]
                except Exception as e:
                    logger.warning(f"Error processing timestamps: {e}")
                    # Fall back to using errors in original order
                    errors = np.array(data["error"])
            else:
                # Use errors in original order
                errors = np.array(data["error"])
            
            # Apply smoothing if enabled
            smoothing_method = self.config["time_series_analysis"]["smoothing_method"]
            if smoothing_method != "none":
                smoothed_errors = self._apply_smoothing(
                    errors, 
                    smoothing_method, 
                    self.config["time_series_analysis"]["smoothing_window"]
                )
            else:
                smoothed_errors = errors
            
            # Analyze trend
            trend_analysis = self._analyze_trend(smoothed_errors)
            time_series_results["trends"][metric] = trend_analysis
            
            # Analyze seasonality if enough data points and enabled
            if (len(errors) >= 3 * self.config["time_series_analysis"]["smoothing_window"] and
                self.config["time_series_analysis"]["seasonal_decomposition"]):
                seasonality_analysis = self._analyze_seasonality(smoothed_errors)
                time_series_results["seasonality"][metric] = seasonality_analysis
            
            # Detect anomalies if enabled
            if self.config["time_series_analysis"]["anomaly_detection"]:
                anomaly_analysis = self._detect_anomalies(errors, smoothed_errors)
                time_series_results["anomalies"][metric] = anomaly_analysis
        
        return time_series_results
    
    def _apply_smoothing(
        self,
        data: np.ndarray,
        method: str,
        window: int
    ) -> np.ndarray:
        """
        Apply smoothing to a time series.
        
        Args:
            data: Input data array
            method: Smoothing method ('moving_average' or 'exponential')
            window: Window size for smoothing
            
        Returns:
            Smoothed data array
        """
        if method == "moving_average":
            # Simple moving average
            smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
            # Pad with original values to maintain array size
            padding = len(data) - len(smoothed)
            if padding > 0:
                pad_left = padding // 2
                pad_right = padding - pad_left
                smoothed = np.concatenate([data[:pad_left], smoothed, data[-pad_right:]])
            return smoothed
        
        elif method == "exponential":
            # Exponential smoothing
            alpha = 2 / (window + 1)  # Calculate alpha from window size
            smoothed = np.zeros_like(data)
            smoothed[0] = data[0]
            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            return smoothed
        
        else:
            # No smoothing
            return data
    
    def _analyze_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze trend in a time series.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with trend analysis results
        """
        # Fit linear regression to identify trend
        try:
            from scipy import stats
            
            # Create time index
            time_idx = np.arange(len(data))
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, data)
            
            # Determine if trend is significant
            is_significant = p_value < 0.05
            
            # Determine direction of trend
            if is_significant:
                if slope > 0:
                    direction = "increasing"
                elif slope < 0:
                    direction = "decreasing"
                else:
                    direction = "stable"
            else:
                direction = "stable"
            
            # Store results
            trend_analysis = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "std_error": float(std_err),
                "significant": bool(is_significant),
                "direction": direction
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing trend: {e}")
            return {
                "error": str(e),
                "significant": False,
                "direction": "unknown"
            }
    
    def _analyze_seasonality(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze seasonality in a time series.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with seasonality analysis results
        """
        # This is a simplified seasonality analysis
        # For more advanced analysis, libraries like statsmodels would be needed
        
        # Calculate autocorrelation to detect seasonality
        try:
            from scipy import signal
            
            # Calculate autocorrelation
            autocorr = signal.correlate(data - np.mean(data), data - np.mean(data), mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Use only the positive lags
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find peaks in autocorrelation (potential seasonal periods)
            peaks, _ = signal.find_peaks(autocorr, height=0.3)  # Threshold of 0.3
            
            # Remove peak at lag 0
            peaks = peaks[peaks > 0]
            
            # Determine if seasonality is present
            has_seasonality = len(peaks) > 0
            
            # Identify primary seasonal period
            primary_period = int(peaks[0]) if has_seasonality else None
            
            # Store results
            seasonality_analysis = {
                "has_seasonality": bool(has_seasonality),
                "primary_period": primary_period,
                "all_periods": peaks.tolist() if has_seasonality else [],
                "autocorrelation": autocorr.tolist()
            }
            
            return seasonality_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing seasonality: {e}")
            return {
                "error": str(e),
                "has_seasonality": False
            }
    
    def _detect_anomalies(
        self,
        original_data: np.ndarray,
        smoothed_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect anomalies in a time series.
        
        Args:
            original_data: Original input data array
            smoothed_data: Smoothed data array
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Simple anomaly detection using z-scores
        try:
            # Calculate residuals (difference between original and smoothed)
            residuals = original_data - smoothed_data
            
            # Calculate z-scores of residuals
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            
            if std_residual == 0:
                # No variation in residuals
                z_scores = np.zeros_like(residuals)
            else:
                z_scores = (residuals - mean_residual) / std_residual
            
            # Identify anomalies (points with absolute z-score > 2)
            threshold = 2.0
            anomaly_indices = np.where(np.abs(z_scores) > threshold)[0]
            
            # Store results
            anomaly_detection = {
                "anomaly_indices": anomaly_indices.tolist(),
                "anomaly_values": original_data[anomaly_indices].tolist(),
                "z_scores": z_scores[anomaly_indices].tolist(),
                "threshold": float(threshold),
                "num_anomalies": len(anomaly_indices)
            }
            
            return anomaly_detection
            
        except Exception as e:
            logger.warning(f"Error detecting anomalies: {e}")
            return {
                "error": str(e),
                "num_anomalies": 0
            }
    
    def _perform_multivariate_analysis(
        self,
        validation_results: List[ValidationResult],
        metrics_data: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Any]:
        """
        Perform multivariate analysis on validation results.
        
        Args:
            validation_results: List of validation results
            metrics_data: Extracted metrics data
            
        Returns:
            Dictionary with multivariate analysis results
        """
        multivariate_results = {
            "correlations": [],
            "pca": {},
            "cluster_analysis": {}
        }
        
        # Check if we have enough metrics for multivariate analysis
        metrics_with_data = [
            metric for metric, data in metrics_data.items() 
            if len(data["error"]) >= self.config["multivariate_analysis"]["min_points_required"]
        ]
        
        if len(metrics_with_data) < 2:
            return {
                "status": "skipped",
                "message": "Insufficient metrics for multivariate analysis"
            }
        
        # Calculate correlation matrix
        correlation_matrix = {}
        for i, metric1 in enumerate(metrics_with_data):
            correlation_matrix[metric1] = {}
            
            for j, metric2 in enumerate(metrics_with_data):
                if i == j:
                    # Same metric, correlation = 1
                    correlation_matrix[metric1][metric2] = 1.0
                    continue
                
                # Calculate correlation between error values
                try:
                    # Filter to use only points where both metrics have data
                    error1 = np.array(metrics_data[metric1]["error"])
                    error2 = np.array(metrics_data[metric2]["error"])
                    
                    # Ensure arrays have the same length
                    min_length = min(len(error1), len(error2))
                    error1 = error1[:min_length]
                    error2 = error2[:min_length]
                    
                    # Calculate Pearson correlation
                    correlation = np.corrcoef(error1, error2)[0, 1]
                    
                    correlation_matrix[metric1][metric2] = float(correlation)
                    
                    # Add to correlations list if significant
                    if abs(correlation) >= self.config["multivariate_analysis"]["correlation_threshold"]:
                        strength = "strong" if abs(correlation) >= 0.7 else "moderate"
                        
                        multivariate_results["correlations"].append({
                            "metrics": [metric1, metric2],
                            "coefficient": float(correlation),
                            "strength": strength,
                            "direction": "positive" if correlation > 0 else "negative"
                        })
                    
                except Exception as e:
                    logger.warning(f"Error calculating correlation between {metric1} and {metric2}: {e}")
                    correlation_matrix[metric1][metric2] = None
        
        # Store correlation matrix
        multivariate_results["correlation_matrix"] = correlation_matrix
        
        # Try to perform PCA if enabled and we have numpy
        if "pca" in self.config["multivariate_analysis"]["methods"]:
            try:
                # Create data matrix for PCA
                data_matrix = []
                for metric in metrics_with_data:
                    # Normalize errors for better PCA
                    errors = np.array(metrics_data[metric]["error"])
                    if len(errors) > 0 and np.std(errors) > 0:
                        normalized_errors = (errors - np.mean(errors)) / np.std(errors)
                        data_matrix.append(normalized_errors)
                
                # Transpose to have metrics as columns
                data_matrix = np.array(data_matrix).T
                
                if data_matrix.shape[0] > 0 and data_matrix.shape[1] > 0:
                    # Calculate covariance matrix
                    cov_matrix = np.cov(data_matrix, rowvar=False)
                    
                    # Calculate eigenvalues and eigenvectors
                    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                    
                    # Sort by eigenvalues
                    idx = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    
                    # Calculate explained variance
                    total_variance = np.sum(eigenvalues)
                    explained_variance = eigenvalues / total_variance
                    
                    # Store PCA results
                    multivariate_results["pca"] = {
                        "eigenvalues": eigenvalues.tolist(),
                        "explained_variance": explained_variance.tolist(),
                        "cumulative_variance": np.cumsum(explained_variance).tolist(),
                        "principal_components": eigenvectors.tolist(),
                        "metrics": metrics_with_data
                    }
                    
                    # Extract insights from PCA
                    multivariate_results["pca"]["insights"] = []
                    
                    # Find metrics with highest contribution to PC1
                    pc1_contributions = np.abs(eigenvectors[:, 0])
                    pc1_indices = np.argsort(pc1_contributions)[::-1]
                    
                    top_metric_idx = pc1_indices[0]
                    top_metric = metrics_with_data[top_metric_idx]
                    top_contribution = pc1_contributions[top_metric_idx]
                    
                    multivariate_results["pca"]["insights"].append(
                        f"{top_metric} has the highest contribution ({top_contribution:.2f}) "
                        f"to the first principal component, explaining {explained_variance[0]*100:.1f}% "
                        f"of the variance"
                    )
            except Exception as e:
                logger.warning(f"Error performing PCA: {e}")
                multivariate_results["pca"] = {"status": "error", "message": str(e)}
        
        # Cluster analysis would be added here for a more complete implementation
        
        return multivariate_results
    
    def _perform_effect_size_analysis(
        self,
        validation_results: List[ValidationResult],
        metrics_data: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Any]:
        """
        Perform effect size analysis on validation results.
        
        Args:
            validation_results: List of validation results
            metrics_data: Extracted metrics data
            
        Returns:
            Dictionary with effect size analysis results
        """
        effect_size_results = {}
        
        # Get thresholds for interpretation
        thresholds = self.config["effect_size_analysis"]["interpretation_thresholds"]
        
        # Process each metric
        for metric, data in metrics_data.items():
            # Skip if not enough data
            if (len(data["simulation"]) < 1 or len(data["hardware"]) < 1 or
                len(data["simulation"]) != len(data["hardware"])):
                continue
            
            # Get simulation and hardware values
            sim_values = np.array(data["simulation"])
            hw_values = np.array(data["hardware"])
            
            # Calculate effect sizes
            effect_size_results[metric] = {}
            
            # Cohen's d
            if "cohen_d" in self.config["effect_size_analysis"]["methods"]:
                try:
                    # Calculate pooled standard deviation
                    n1, n2 = len(sim_values), len(hw_values)
                    s1, s2 = np.std(sim_values, ddof=1), np.std(hw_values, ddof=1)
                    
                    # Pooled standard deviation
                    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                    
                    # Cohen's d
                    if pooled_std > 0:
                        cohen_d = (np.mean(sim_values) - np.mean(hw_values)) / pooled_std
                    else:
                        cohen_d = 0
                    
                    # Interpret magnitude
                    abs_d = abs(cohen_d)
                    if abs_d < thresholds["small"]:
                        magnitude = "negligible"
                    elif abs_d < thresholds["medium"]:
                        magnitude = "small"
                    elif abs_d < thresholds["large"]:
                        magnitude = "medium"
                    else:
                        magnitude = "large"
                    
                    effect_size_results[metric]["cohen_d"] = {
                        "value": float(cohen_d),
                        "magnitude": magnitude,
                        "interpretation": f"{magnitude.capitalize()} effect size"
                    }
                except Exception as e:
                    logger.warning(f"Error calculating Cohen's d for {metric}: {e}")
                    effect_size_results[metric]["cohen_d"] = {"error": str(e)}
            
            # Hedges' g
            if "hedges_g" in self.config["effect_size_analysis"]["methods"]:
                try:
                    # Calculate pooled standard deviation
                    n1, n2 = len(sim_values), len(hw_values)
                    s1, s2 = np.std(sim_values, ddof=1), np.std(hw_values, ddof=1)
                    
                    # Pooled standard deviation
                    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                    
                    # Correction factor for small sample size
                    correction = 1 - 3 / (4 * (n1 + n2) - 9)
                    
                    # Hedges' g
                    if pooled_std > 0:
                        hedges_g = correction * (np.mean(sim_values) - np.mean(hw_values)) / pooled_std
                    else:
                        hedges_g = 0
                    
                    # Interpret magnitude (same thresholds as Cohen's d)
                    abs_g = abs(hedges_g)
                    if abs_g < thresholds["small"]:
                        magnitude = "negligible"
                    elif abs_g < thresholds["medium"]:
                        magnitude = "small"
                    elif abs_g < thresholds["large"]:
                        magnitude = "medium"
                    else:
                        magnitude = "large"
                    
                    effect_size_results[metric]["hedges_g"] = {
                        "value": float(hedges_g),
                        "magnitude": magnitude,
                        "interpretation": f"{magnitude.capitalize()} effect size (bias corrected)"
                    }
                except Exception as e:
                    logger.warning(f"Error calculating Hedges' g for {metric}: {e}")
                    effect_size_results[metric]["hedges_g"] = {"error": str(e)}
            
            # Glass's delta
            if "glass_delta" in self.config["effect_size_analysis"]["methods"]:
                try:
                    # Glass's delta uses only the standard deviation of the hardware group
                    s2 = np.std(hw_values, ddof=1)
                    
                    # Glass's delta
                    if s2 > 0:
                        glass_delta = (np.mean(sim_values) - np.mean(hw_values)) / s2
                    else:
                        glass_delta = 0
                    
                    # Interpret magnitude (same thresholds as Cohen's d)
                    abs_delta = abs(glass_delta)
                    if abs_delta < thresholds["small"]:
                        magnitude = "negligible"
                    elif abs_delta < thresholds["medium"]:
                        magnitude = "small"
                    elif abs_delta < thresholds["large"]:
                        magnitude = "medium"
                    else:
                        magnitude = "large"
                    
                    effect_size_results[metric]["glass_delta"] = {
                        "value": float(glass_delta),
                        "magnitude": magnitude,
                        "interpretation": f"{magnitude.capitalize()} effect size (relative to hardware)"
                    }
                except Exception as e:
                    logger.warning(f"Error calculating Glass's delta for {metric}: {e}")
                    effect_size_results[metric]["glass_delta"] = {"error": str(e)}
        
        return effect_size_results
    
    def _perform_correlation_analysis(
        self,
        validation_results: List[ValidationResult],
        metrics_data: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Any]:
        """
        Perform advanced correlation analysis on validation results.
        
        Args:
            validation_results: List of validation results
            metrics_data: Extracted metrics data
            
        Returns:
            Dictionary with correlation analysis results
        """
        correlation_results = {
            "linear_correlations": [],
            "non_linear_relationships": []
        }
        
        # Try to use scipy for statistics
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available for correlation analysis")
            return {"status": "error", "message": "scipy not available"}
        
        # Process each metric
        for metric, data in metrics_data.items():
            # Skip if not enough data
            if len(data["simulation"]) < 3 or len(data["hardware"]) < 3:
                continue
            
            # Get simulation and hardware values
            sim_values = np.array(data["simulation"])
            hw_values = np.array(data["hardware"])
            
            # Linear correlation methods
            linear_methods = [m for m in self.config["correlation_analysis"]["methods"]
                             if m in ["pearson", "spearman", "kendall"]]
            
            for method in linear_methods:
                try:
                    if method == "pearson":
                        # Pearson correlation
                        r, p_value = stats.pearsonr(sim_values, hw_values)
                        method_name = "Pearson"
                    elif method == "spearman":
                        # Spearman rank correlation
                        r, p_value = stats.spearmanr(sim_values, hw_values)
                        method_name = "Spearman rank"
                    elif method == "kendall":
                        # Kendall's tau
                        r, p_value = stats.kendalltau(sim_values, hw_values)
                        method_name = "Kendall's tau"
                    else:
                        continue
                    
                    # Determine strength of correlation
                    abs_r = abs(r)
                    if abs_r < 0.3:
                        strength = "weak"
                    elif abs_r < 0.7:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    
                    # Determine significance
                    significant = p_value < 0.05
                    
                    # Add to results
                    correlation_results["linear_correlations"].append({
                        "metric": metric,
                        "method": method,
                        "method_name": method_name,
                        "coefficient": float(r),
                        "p_value": float(p_value),
                        "significant": bool(significant),
                        "strength": strength,
                        "direction": "positive" if r > 0 else "negative"
                    })
                except Exception as e:
                    logger.warning(f"Error calculating {method} correlation for {metric}: {e}")
            
            # Non-linear methods
            # For this implementation, we'll use a simplified approach for mutual information
            if "mutual_information" in self.config["correlation_analysis"]["non_linear_methods"]:
                try:
                    # Simplified mutual information estimation
                    # Bin the data
                    bins = 10  # Number of bins
                    sim_bins = np.linspace(min(sim_values), max(sim_values), bins + 1)
                    hw_bins = np.linspace(min(hw_values), max(hw_values), bins + 1)
                    
                    sim_digitized = np.digitize(sim_values, sim_bins)
                    hw_digitized = np.digitize(hw_values, hw_bins)
                    
                    # Create joint histogram
                    joint_hist, _, _ = np.histogram2d(sim_values, hw_values, bins=[sim_bins, hw_bins])
                    
                    # Add small constant to avoid log(0)
                    joint_hist = joint_hist + 1e-10
                    
                    # Normalize to get probabilities
                    joint_probs = joint_hist / np.sum(joint_hist)
                    
                    # Calculate marginal probabilities
                    sim_probs = np.sum(joint_probs, axis=1)
                    hw_probs = np.sum(joint_probs, axis=0)
                    
                    # Calculate entropies
                    sim_entropy = -np.sum(sim_probs * np.log2(sim_probs))
                    hw_entropy = -np.sum(hw_probs * np.log2(hw_probs))
                    joint_entropy = -np.sum(joint_probs * np.log2(joint_probs))
                    
                    # Calculate mutual information
                    mutual_info = sim_entropy + hw_entropy - joint_entropy
                    
                    # Normalize to [0, 1]
                    if sim_entropy > 0 and hw_entropy > 0:
                        normalized_mi = mutual_info / np.sqrt(sim_entropy * hw_entropy)
                    else:
                        normalized_mi = 0
                    
                    # Determine strength
                    if normalized_mi < 0.3:
                        strength = "weak"
                    elif normalized_mi < 0.7:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    
                    # Add to results
                    correlation_results["non_linear_relationships"].append({
                        "metrics": [metric],
                        "method": "mutual_information",
                        "mutual_information": float(mutual_info),
                        "normalized_mi": float(normalized_mi),
                        "strength": strength
                    })
                except Exception as e:
                    logger.warning(f"Error calculating mutual information for {metric}: {e}")
        
        return correlation_results
    
    def _generate_recommendations(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            analysis_results: Analysis results dictionary
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check if we have key findings
        if "insights" not in analysis_results or "key_findings" not in analysis_results["insights"]:
            return ["Insufficient data for generating recommendations"]
        
        # Extract components from analysis
        bayesian_results = analysis_results["analysis_methods"].get("bayesian_analysis", {})
        time_series_results = analysis_results["analysis_methods"].get("time_series_analysis", {})
        multivariate_results = analysis_results["analysis_methods"].get("multivariate_analysis", {})
        effect_size_results = analysis_results["analysis_methods"].get("effect_size_analysis", {})
        
        # Recommendations based on time series trends
        if "trends" in time_series_results:
            # Check for degrading trends
            degrading_metrics = []
            for metric, trend in time_series_results["trends"].items():
                if trend.get("significant", False) and trend.get("direction") == "increasing":
                    degrading_metrics.append(metric)
            
            if degrading_metrics:
                if len(degrading_metrics) == 1:
                    recommendations.append(
                        f"Monitor {degrading_metrics[0]} closely as it shows a significant increasing trend, "
                        f"indicating potential degradation in simulation accuracy"
                    )
                else:
                    metrics_str = ", ".join(degrading_metrics)
                    recommendations.append(
                        f"Focus on improving simulation accuracy for these metrics showing degradation: {metrics_str}"
                    )
        
        # Recommendations based on effect size analysis
        large_effect_metrics = []
        if effect_size_results:
            for metric, results in effect_size_results.items():
                if ("cohen_d" in results and 
                    results["cohen_d"].get("magnitude") == "large"):
                    large_effect_metrics.append(metric)
        
        if large_effect_metrics:
            if len(large_effect_metrics) == 1:
                recommendations.append(
                    f"Prioritize improvements to the {large_effect_metrics[0]} simulation model "
                    f"as it shows a large discrepancy from hardware measurements"
                )
            else:
                metrics_str = ", ".join(large_effect_metrics)
                recommendations.append(
                    f"Focus calibration efforts on these metrics with large discrepancies: {metrics_str}"
                )
        
        # Recommendations based on multivariate analysis
        if "correlations" in multivariate_results:
            strong_correlations = [
                corr for corr in multivariate_results["correlations"]
                if corr.get("strength") == "strong"
            ]
            
            if strong_correlations and len(strong_correlations) > 0:
                corr = strong_correlations[0]  # Just use the first one for recommendation
                metrics = corr.get("metrics", ["unknown", "unknown"])
                
                if corr.get("direction") == "positive":
                    recommendations.append(
                        f"Consider joint optimization of {metrics[0]} and {metrics[1]} "
                        f"as they show a strong positive correlation"
                    )
                else:
                    recommendations.append(
                        f"Investigate the trade-off relationship between {metrics[0]} and {metrics[1]} "
                        f"as they show a strong negative correlation"
                    )
        
        # Recommendations based on Bayesian analysis
        high_uncertainty_metrics = []
        for metric, results in bayesian_results.items():
            if "credible_interval" in results:
                ci = results["credible_interval"]
                ci_width = ci.get("upper", 0) - ci.get("lower", 0)
                
                # Check if ci_width is greater than 20% of the mean error
                if "mean_error" in results and results["mean_error"] > 0:
                    if ci_width > 0.2 * results["mean_error"]:
                        high_uncertainty_metrics.append(metric)
        
        if high_uncertainty_metrics:
            if len(high_uncertainty_metrics) == 1:
                recommendations.append(
                    f"Collect more validation data for {high_uncertainty_metrics[0]} "
                    f"to reduce uncertainty in simulation accuracy assessment"
                )
            else:
                metrics_str = ", ".join(high_uncertainty_metrics)
                recommendations.append(
                    f"Prioritize additional validation for these metrics with high uncertainty: {metrics_str}"
                )
        
        # Add more general recommendations if specific ones are limited
        if len(recommendations) < 3:
            if "anomalies" in time_series_results and any(
                anomaly.get("num_anomalies", 0) > 0 
                for anomaly in time_series_results["anomalies"].values()
            ):
                recommendations.append(
                    "Investigate anomalous validation results detected in the time series analysis, "
                    "as they may indicate sporadic simulation errors or hardware measurement issues"
                )
            
            if len(analysis_results.get("num_validation_results", 0)) < 10:
                recommendations.append(
                    "Increase validation sample size to improve statistical power and confidence "
                    "in the analysis results"
                )
        
        # Ensure we return at least one recommendation
        if not recommendations:
            recommendations.append(
                "Continue monitoring simulation accuracy with regular validation tests "
                "to maintain high quality of simulation results"
            )
        
        return recommendations