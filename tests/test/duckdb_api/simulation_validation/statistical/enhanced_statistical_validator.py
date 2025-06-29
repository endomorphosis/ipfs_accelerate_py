#!/usr/bin/env python3
"""
Enhanced Statistical Validation Tools for the Simulation Accuracy and Validation Framework.

This module extends the StatisticalValidator with advanced statistical methods including:
- Additional error metrics beyond MAPE
- Confidence interval calculations
- Distribution comparison utilities
- Statistical significance testing
- Bland-Altman analysis for method comparison
- Statistical power analysis
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
logger = logging.getLogger("enhanced_statistical_validator")

# Add parent directories to path for module imports
import sys
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import base classes and statistical validator
from duckdb_api.simulation_validation.statistical.statistical_validator import StatisticalValidator
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

class EnhancedStatisticalValidator(StatisticalValidator):
    """
    Enhanced statistical validator with advanced methods for simulation validation.
    
    This class extends the StatisticalValidator with additional statistical methods:
    - Confidence interval calculation for validation metrics
    - Enhanced distribution comparison utilities
    - Statistical significance testing for validation results
    - Bland-Altman analysis for method comparison
    - Statistical power analysis for validation confidence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced statistical validator.
        
        Args:
            config: Configuration options for the validator
        """
        # Initialize the base statistical validator
        super().__init__(config)
        
        # Add enhanced configuration defaults
        enhanced_config = {
            # Additional error metrics
            "additional_error_metrics": [
                "r_squared",              # Coefficient of determination
                "concordance_correlation", # Lin's concordance correlation coefficient
                "mae",                     # Mean Absolute Error
                "mse",                     # Mean Squared Error
                "bias",                    # Systematic bias
                "ratio_metrics"            # Ratio between simulation and hardware
            ],
            
            # Confidence interval configuration
            "confidence_intervals": {
                "enabled": True,
                "confidence_level": 0.95,  # 95% confidence intervals
                "methods": [
                    "normal_approximation", # Normal approximation method
                    "bootstrap",           # Bootstrap method
                    "student_t"            # Student's t-distribution
                ]
            },
            
            # Distribution comparison configuration
            "distribution_comparison": {
                "enabled": True,
                "tests": [
                    "anderson_darling",    # Anderson-Darling test
                    "shapiro_wilk",        # Shapiro-Wilk test for normality
                    "chi_squared",         # Chi-squared test
                    "levene",              # Levene's test for equality of variances
                    "anderson_darling"     # Anderson-Darling test
                ],
                "visualization_methods": [
                    "qq_plot",             # Quantile-Quantile plot
                    "pp_plot",             # Probability-Probability plot
                    "ecdf_comparison",     # Empirical CDF comparison
                    "histogram_comparison" # Histogram comparison
                ]
            },
            
            # Bland-Altman analysis configuration
            "bland_altman": {
                "enabled": True,
                "confidence_level": 0.95,  # 95% confidence intervals for limits of agreement
                "log_transform": False,    # Option to use log transformation
                "proportional_bias": True  # Test for proportional bias
            },
            
            # Statistical power analysis
            "power_analysis": {
                "enabled": True,
                "effect_sizes": [0.2, 0.5, 0.8], # Small, medium, large effect sizes
                "alpha": 0.05,            # Significance level
                "power_threshold": 0.8    # Desired power level
            }
        }
        
        # Update config with enhanced defaults if not already present
        for key, value in enhanced_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def validate(
        self,
        simulation_result: SimulationResult,
        hardware_result: HardwareResult
    ) -> ValidationResult:
        """
        Validate a simulation result against real hardware measurements with enhanced statistics.
        
        Args:
            simulation_result: The simulation result to validate
            hardware_result: The real hardware result to compare against
            
        Returns:
            A ValidationResult with enhanced comparison metrics
        """
        # First, perform the basic validation using the parent class
        validation_result = super().validate(simulation_result, hardware_result)
        
        # Add enhanced metrics
        enhanced_metrics = self._calculate_enhanced_metrics(
            simulation_result,
            hardware_result,
            validation_result.metrics_comparison
        )
        
        # Add confidence intervals
        if self.config["confidence_intervals"]["enabled"]:
            ci_metrics = self._calculate_confidence_intervals(
                simulation_result,
                hardware_result,
                validation_result.metrics_comparison
            )
            
            # Merge confidence intervals into enhanced metrics
            if enhanced_metrics is None:
                enhanced_metrics = {}
            enhanced_metrics["confidence_intervals"] = ci_metrics
        
        # Add Bland-Altman analysis
        if self.config["bland_altman"]["enabled"]:
            ba_metrics = self._perform_bland_altman_analysis(
                simulation_result,
                hardware_result
            )
            
            # Merge Bland-Altman metrics into enhanced metrics
            if enhanced_metrics is None:
                enhanced_metrics = {}
            enhanced_metrics["bland_altman"] = ba_metrics
        
        # Add the enhanced metrics to the validation result
        if enhanced_metrics and validation_result.additional_metrics:
            validation_result.additional_metrics.update(enhanced_metrics)
        elif enhanced_metrics:
            validation_result.additional_metrics = enhanced_metrics
        
        return validation_result
    
    def validate_batch(
        self,
        simulation_results: List[SimulationResult],
        hardware_results: List[HardwareResult]
    ) -> List[ValidationResult]:
        """
        Validate multiple simulation results with enhanced statistics.
        
        Args:
            simulation_results: The simulation results to validate
            hardware_results: The real hardware results to compare against
            
        Returns:
            A list of ValidationResults with enhanced comparison metrics
        """
        # First, perform the basic batch validation using the parent class
        validation_results = super().validate_batch(simulation_results, hardware_results)
        
        # Add power analysis for the batch validation
        if self.config["power_analysis"]["enabled"] and len(validation_results) > 1:
            power_analysis = self._perform_power_analysis(validation_results)
            
            # Add power analysis to each validation result
            for result in validation_results:
                if result.additional_metrics is None:
                    result.additional_metrics = {}
                
                result.additional_metrics["power_analysis"] = power_analysis
        
        # Add enhanced distribution comparison
        if (self.config["distribution_comparison"]["enabled"] and 
            len(simulation_results) > 1 and 
            len(hardware_results) > 1):
            
            # Group results by model_id, hardware_id, batch_size, precision
            grouped_sim_results = self._group_results(simulation_results)
            grouped_hw_results = self._group_results(hardware_results)
            
            # Perform distribution comparisons for each group
            for i, result in enumerate(validation_results):
                if (hasattr(result.simulation_result, 'model_id') and 
                    hasattr(result.simulation_result, 'hardware_id')):
                    
                    group_key = f"{result.simulation_result.model_id}_{result.simulation_result.hardware_id}_"
                    group_key += f"{result.simulation_result.batch_size}_{result.simulation_result.precision}"
                    
                    if (group_key in grouped_sim_results and 
                        group_key in grouped_hw_results and
                        len(grouped_sim_results[group_key]) > 1 and
                        len(grouped_hw_results[group_key]) > 1):
                        
                        distribution_comparison = self._compare_distributions(
                            grouped_sim_results[group_key],
                            grouped_hw_results[group_key]
                        )
                        
                        if distribution_comparison:
                            if result.additional_metrics is None:
                                result.additional_metrics = {}
                            
                            result.additional_metrics["distribution_comparison"] = distribution_comparison
        
        return validation_results
    
    def summarize_validation(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Generate an enhanced summary of validation results with additional statistics.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            A dictionary with enhanced summary statistics
        """
        # First, perform the basic summary using the parent class
        summary = super().summarize_validation(validation_results)
        
        # Add enhanced summary statistics
        enhanced_summary = {
            "enhanced_metrics": {},
            "confidence_intervals": {},
            "distribution_tests": {},
            "bland_altman": {},
            "power_analysis": {}
        }
        
        # Extract enhanced metrics from validation results
        for val_result in validation_results:
            if not val_result.additional_metrics:
                continue
            
            # Process enhanced metrics
            if "enhanced_metrics" in val_result.additional_metrics:
                for metric, value in val_result.additional_metrics["enhanced_metrics"].items():
                    if metric not in enhanced_summary["enhanced_metrics"]:
                        enhanced_summary["enhanced_metrics"][metric] = []
                    
                    enhanced_summary["enhanced_metrics"][metric].append(value)
            
            # Process confidence intervals
            if "confidence_intervals" in val_result.additional_metrics:
                for metric, ci_data in val_result.additional_metrics["confidence_intervals"].items():
                    if metric not in enhanced_summary["confidence_intervals"]:
                        enhanced_summary["confidence_intervals"][metric] = {
                            "lower_bounds": [],
                            "upper_bounds": [],
                            "width": []
                        }
                    
                    if "lower_bound" in ci_data:
                        enhanced_summary["confidence_intervals"][metric]["lower_bounds"].append(ci_data["lower_bound"])
                    
                    if "upper_bound" in ci_data:
                        enhanced_summary["confidence_intervals"][metric]["upper_bounds"].append(ci_data["upper_bound"])
                    
                    if "lower_bound" in ci_data and "upper_bound" in ci_data:
                        width = ci_data["upper_bound"] - ci_data["lower_bound"]
                        enhanced_summary["confidence_intervals"][metric]["width"].append(width)
            
            # Process Bland-Altman metrics
            if "bland_altman" in val_result.additional_metrics:
                for metric, ba_data in val_result.additional_metrics["bland_altman"].items():
                    if metric not in enhanced_summary["bland_altman"]:
                        enhanced_summary["bland_altman"][metric] = {
                            "bias": [],
                            "lower_loa": [],
                            "upper_loa": [],
                            "proportional_bias": []
                        }
                    
                    if "bias" in ba_data:
                        enhanced_summary["bland_altman"][metric]["bias"].append(ba_data["bias"])
                    
                    if "lower_loa" in ba_data:
                        enhanced_summary["bland_altman"][metric]["lower_loa"].append(ba_data["lower_loa"])
                    
                    if "upper_loa" in ba_data:
                        enhanced_summary["bland_altman"][metric]["upper_loa"].append(ba_data["upper_loa"])
                    
                    if "proportional_bias" in ba_data and "is_significant" in ba_data["proportional_bias"]:
                        enhanced_summary["bland_altman"][metric]["proportional_bias"].append(
                            ba_data["proportional_bias"]["is_significant"]
                        )
            
            # Process power analysis
            if "power_analysis" in val_result.additional_metrics:
                for effect_size, power_data in val_result.additional_metrics["power_analysis"].items():
                    if effect_size not in enhanced_summary["power_analysis"]:
                        enhanced_summary["power_analysis"][effect_size] = []
                    
                    if "power" in power_data:
                        enhanced_summary["power_analysis"][effect_size].append(power_data["power"])
        
        # Calculate summary statistics for enhanced metrics
        for metric, values in enhanced_summary["enhanced_metrics"].items():
            if values:
                enhanced_summary["enhanced_metrics"][metric] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std_dev": np.std(values),
                    "count": len(values)
                }
        
        # Calculate summary statistics for confidence intervals
        for metric, ci_data in enhanced_summary["confidence_intervals"].items():
            for ci_metric, values in ci_data.items():
                if values:
                    ci_data[ci_metric] = {
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "std_dev": np.std(values),
                        "count": len(values)
                    }
        
        # Calculate summary statistics for Bland-Altman metrics
        for metric, ba_data in enhanced_summary["bland_altman"].items():
            for ba_metric, values in ba_data.items():
                if values:
                    if ba_metric == "proportional_bias":
                        # For boolean values, calculate proportion of True
                        ba_data[ba_metric] = {
                            "proportion_significant": np.mean([1 if v else 0 for v in values]),
                            "count": len(values)
                        }
                    else:
                        ba_data[ba_metric] = {
                            "mean": np.mean(values),
                            "median": np.median(values),
                            "min": np.min(values),
                            "max": np.max(values),
                            "std_dev": np.std(values),
                            "count": len(values)
                        }
        
        # Calculate summary statistics for power analysis
        for effect_size, values in enhanced_summary["power_analysis"].items():
            if values:
                enhanced_summary["power_analysis"][effect_size] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std_dev": np.std(values),
                    "count": len(values),
                    "proportion_sufficient": np.mean([1 if v >= self.config["power_analysis"]["power_threshold"] else 0 for v in values])
                }
        
        # Add enhanced summary to the overall summary
        summary["enhanced"] = enhanced_summary
        
        return summary
    
    def _calculate_enhanced_metrics(
        self,
        simulation_result: SimulationResult,
        hardware_result: HardwareResult,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate enhanced metrics beyond the basic error metrics.
        
        Args:
            simulation_result: Simulation result to validate
            hardware_result: Hardware result to compare against
            metrics_comparison: Basic metrics comparison
            
        Returns:
            Dictionary with enhanced metrics
        """
        enhanced_metrics = {}
        
        # Check if we need to calculate additional error metrics
        if not self.config.get("additional_error_metrics"):
            return enhanced_metrics
        
        # Determine which metrics to validate
        metrics_to_validate = self.config["metrics_to_validate"]
        additional_error_metrics = self.config["additional_error_metrics"]
        
        # Calculate additional metrics for each performance metric
        for metric_name in metrics_to_validate:
            # Skip if either simulation or hardware metric is missing
            if metric_name not in simulation_result.metrics or metric_name not in hardware_result.metrics:
                continue
            
            # Get metric values
            sim_value = simulation_result.metrics[metric_name]
            hw_value = hardware_result.metrics[metric_name]
            
            # Skip if either value is None
            if sim_value is None or hw_value is None:
                continue
            
            # Initialize enhanced metrics for this performance metric
            if metric_name not in enhanced_metrics:
                enhanced_metrics[metric_name] = {}
            
            # R-squared (coefficient of determination)
            if "r_squared" in additional_error_metrics:
                # For a single point, we can't really calculate R-squared
                # Set to NaN for individual points
                enhanced_metrics[metric_name]["r_squared"] = float('nan')
            
            # Concordance correlation coefficient
            if "concordance_correlation" in additional_error_metrics:
                # For a single point, we can't calculate concordance correlation
                # Set to NaN for individual points
                enhanced_metrics[metric_name]["concordance_correlation"] = float('nan')
            
            # Mean Absolute Error (MAE)
            if "mae" in additional_error_metrics:
                enhanced_metrics[metric_name]["mae"] = abs(sim_value - hw_value)
            
            # Mean Squared Error (MSE)
            if "mse" in additional_error_metrics:
                enhanced_metrics[metric_name]["mse"] = (sim_value - hw_value) ** 2
            
            # Bias
            if "bias" in additional_error_metrics:
                enhanced_metrics[metric_name]["bias"] = sim_value - hw_value
            
            # Ratio metrics
            if "ratio_metrics" in additional_error_metrics:
                if hw_value != 0:
                    enhanced_metrics[metric_name]["ratio"] = sim_value / hw_value
                else:
                    enhanced_metrics[metric_name]["ratio"] = float('nan')
        
        return {"enhanced_metrics": enhanced_metrics}
    
    def _calculate_confidence_intervals(
        self,
        simulation_result: SimulationResult,
        hardware_result: HardwareResult,
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for validation metrics.
        
        Args:
            simulation_result: Simulation result to validate
            hardware_result: Hardware result to compare against
            metrics_comparison: Basic metrics comparison
            
        Returns:
            Dictionary mapping metric names to confidence interval data
        """
        # Initialize confidence intervals dictionary
        confidence_intervals = {}
        
        # Skip if confidence intervals are not enabled
        if not self.config["confidence_intervals"]["enabled"]:
            return confidence_intervals
        
        # For a single data point, we can't calculate confidence intervals
        # We need either multiple simulation runs or bootstrapping
        
        # Try to use scipy for statistical functions
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available for confidence interval calculations")
            return confidence_intervals
        
        # For single points, we can only generate confidence intervals with assumptions
        # Assuming some error distribution around the point estimates
        
        # Get confidence level
        confidence_level = self.config["confidence_intervals"]["confidence_level"]
        
        # Determine which metrics to validate
        metrics_to_validate = self.config["metrics_to_validate"]
        
        # For each metric, calculate confidence intervals for MAPE
        for metric_name in metrics_to_validate:
            # Skip if metric comparison not available
            if metric_name not in metrics_comparison:
                continue
            
            # Skip if MAPE is not available
            if "mape" not in metrics_comparison[metric_name]:
                continue
            
            # Get MAPE value
            mape = metrics_comparison[metric_name]["mape"]
            
            # Skip if MAPE is NaN
            if np.isnan(mape):
                continue
            
            # For a single point, we can assume some standard error
            # This is a simple approximation based on typical MAPE variability
            
            # Assumed standard error (could be made more sophisticated)
            # Using a percentage of the MAPE as the standard error
            standard_error = mape * 0.2  # 20% of MAPE
            
            # For normal approximation (z-score)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = z_score * standard_error
            
            # Calculate confidence interval
            lower_bound = max(0, mape - margin_of_error)  # MAPE can't be negative
            upper_bound = mape + margin_of_error
            
            # Store confidence interval
            confidence_intervals[metric_name] = {
                "mape": {
                    "value": mape,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "confidence_level": confidence_level,
                    "method": "normal_approximation",
                    "note": "Approximation for single data point using assumed standard error"
                }
            }
        
        return confidence_intervals
    
    def _compare_distributions(
        self,
        simulation_group: List[SimulationResult],
        hardware_group: List[HardwareResult]
    ) -> Dict[str, Any]:
        """
        Compare distributions of simulation and hardware results.
        
        Args:
            simulation_group: Group of simulation results
            hardware_group: Group of hardware results
            
        Returns:
            Dictionary with distribution comparison metrics
        """
        # Initialize distribution comparison dictionary
        distribution_comparison = {}
        
        # Skip if distribution comparison is not enabled
        if not self.config["distribution_comparison"]["enabled"]:
            return distribution_comparison
        
        # Check if we have enough data points
        if len(simulation_group) < 2 or len(hardware_group) < 2:
            return distribution_comparison
        
        # Try to use scipy for statistical tests
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available for distribution comparison")
            return distribution_comparison
        
        # Get distribution comparison tests
        distribution_tests = self.config["distribution_comparison"]["tests"]
        
        # Get significance level
        alpha = self.config["power_analysis"]["alpha"]
        
        # Collect metrics for each performance metric
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
            
            if len(sim_values) >= 3 and len(hw_values) >= 3:  # Most tests need at least 3 data points
                metrics_data[metric] = {
                    "simulation": np.array(sim_values),
                    "hardware": np.array(hw_values)
                }
        
        # Perform distribution tests for each metric
        for metric, data in metrics_data.items():
            sim_values = data["simulation"]
            hw_values = data["hardware"]
            
            # Initialize tests for this metric
            distribution_comparison[metric] = {"tests": {}}
            
            # Normality tests
            
            # Shapiro-Wilk test for normality
            if "shapiro_wilk" in distribution_tests:
                try:
                    # Test simulation values
                    sim_shapiro_stat, sim_shapiro_p = stats.shapiro(sim_values)
                    # Test hardware values
                    hw_shapiro_stat, hw_shapiro_p = stats.shapiro(hw_values)
                    
                    distribution_comparison[metric]["tests"]["shapiro_wilk"] = {
                        "simulation": {
                            "statistic": float(sim_shapiro_stat),
                            "p_value": float(sim_shapiro_p),
                            "is_normal": sim_shapiro_p > alpha,
                            "interpretation": (
                                "Normal distribution" if sim_shapiro_p > alpha 
                                else "Not normally distributed"
                            )
                        },
                        "hardware": {
                            "statistic": float(hw_shapiro_stat),
                            "p_value": float(hw_shapiro_p),
                            "is_normal": hw_shapiro_p > alpha,
                            "interpretation": (
                                "Normal distribution" if hw_shapiro_p > alpha 
                                else "Not normally distributed"
                            )
                        }
                    }
                except Exception as e:
                    logger.warning(f"Error calculating Shapiro-Wilk test: {e}")
            
            # Anderson-Darling test
            if "anderson_darling" in distribution_tests:
                try:
                    # Test simulation values
                    sim_ad_result = stats.anderson(sim_values, 'norm')
                    sim_ad_stat = sim_ad_result.statistic
                    
                    # Find the appropriate critical value and significance level
                    sim_ad_sig_level = None
                    sim_ad_is_normal = False
                    for i, size in enumerate(sim_ad_result.critical_values):
                        if sim_ad_stat < size:
                            sim_ad_is_normal = True
                            sim_ad_sig_level = sim_ad_result.significance_level[i] / 100.0
                            break
                    
                    # Test hardware values
                    hw_ad_result = stats.anderson(hw_values, 'norm')
                    hw_ad_stat = hw_ad_result.statistic
                    
                    # Find the appropriate critical value and significance level
                    hw_ad_sig_level = None
                    hw_ad_is_normal = False
                    for i, size in enumerate(hw_ad_result.critical_values):
                        if hw_ad_stat < size:
                            hw_ad_is_normal = True
                            hw_ad_sig_level = hw_ad_result.significance_level[i] / 100.0
                            break
                    
                    distribution_comparison[metric]["tests"]["anderson_darling"] = {
                        "simulation": {
                            "statistic": float(sim_ad_stat),
                            "significance_level": sim_ad_sig_level,
                            "is_normal": sim_ad_is_normal,
                            "interpretation": (
                                "Normal distribution" if sim_ad_is_normal 
                                else "Not normally distributed"
                            )
                        },
                        "hardware": {
                            "statistic": float(hw_ad_stat),
                            "significance_level": hw_ad_sig_level,
                            "is_normal": hw_ad_is_normal,
                            "interpretation": (
                                "Normal distribution" if hw_ad_is_normal 
                                else "Not normally distributed"
                            )
                        }
                    }
                except Exception as e:
                    logger.warning(f"Error calculating Anderson-Darling test: {e}")
            
            # Test for equality of variances
            
            # Levene's test
            if "levene" in distribution_tests:
                try:
                    levene_stat, levene_p = stats.levene(sim_values, hw_values)
                    
                    distribution_comparison[metric]["tests"]["levene"] = {
                        "statistic": float(levene_stat),
                        "p_value": float(levene_p),
                        "equal_variances": levene_p > alpha,
                        "interpretation": (
                            "Equal variances" if levene_p > alpha 
                            else "Unequal variances"
                        )
                    }
                except Exception as e:
                    logger.warning(f"Error calculating Levene's test: {e}")
            
            # Test for equality of distributions
            
            # Chi-squared test
            if "chi_squared" in distribution_tests:
                try:
                    # We need to bin the data for chi-squared test
                    # Using the same bins for both distributions
                    bins = 10  # Number of bins (adjust as needed)
                    min_val = min(np.min(sim_values), np.min(hw_values))
                    max_val = max(np.max(sim_values), np.max(hw_values))
                    
                    # Create histogram bins
                    bin_edges = np.linspace(min_val, max_val, bins + 1)
                    
                    # Get histogram counts
                    sim_hist, _ = np.histogram(sim_values, bins=bin_edges)
                    hw_hist, _ = np.histogram(hw_values, bins=bin_edges)
                    
                    # Perform chi-squared test
                    chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(
                        np.vstack([sim_hist, hw_hist])
                    )
                    
                    distribution_comparison[metric]["tests"]["chi_squared"] = {
                        "statistic": float(chi2_stat),
                        "p_value": float(chi2_p),
                        "degrees_of_freedom": int(dof),
                        "same_distribution": chi2_p > alpha,
                        "interpretation": (
                            "Same distribution" if chi2_p > alpha 
                            else "Different distributions"
                        )
                    }
                except Exception as e:
                    logger.warning(f"Error calculating Chi-squared test: {e}")
            
            # Calculate distribution metrics
            
            # Mean and standard deviation
            sim_mean = np.mean(sim_values)
            sim_std = np.std(sim_values)
            hw_mean = np.mean(hw_values)
            hw_std = np.std(hw_values)
            
            # Coefficient of variation
            sim_cv = sim_std / sim_mean if sim_mean != 0 else float('nan')
            hw_cv = hw_std / hw_mean if hw_mean != 0 else float('nan')
            
            # Percentiles
            sim_percentiles = np.percentile(sim_values, [25, 50, 75])
            hw_percentiles = np.percentile(hw_values, [25, 50, 75])
            
            # Interquartile range
            sim_iqr = sim_percentiles[2] - sim_percentiles[0]
            hw_iqr = hw_percentiles[2] - hw_percentiles[0]
            
            # Add metrics to distribution comparison
            distribution_comparison[metric]["metrics"] = {
                "simulation": {
                    "mean": float(sim_mean),
                    "std_dev": float(sim_std),
                    "cv": float(sim_cv),
                    "q1": float(sim_percentiles[0]),
                    "median": float(sim_percentiles[1]),
                    "q3": float(sim_percentiles[2]),
                    "iqr": float(sim_iqr),
                    "min": float(np.min(sim_values)),
                    "max": float(np.max(sim_values)),
                    "n": len(sim_values)
                },
                "hardware": {
                    "mean": float(hw_mean),
                    "std_dev": float(hw_std),
                    "cv": float(hw_cv),
                    "q1": float(hw_percentiles[0]),
                    "median": float(hw_percentiles[1]),
                    "q3": float(hw_percentiles[2]),
                    "iqr": float(hw_iqr),
                    "min": float(np.min(hw_values)),
                    "max": float(np.max(hw_values)),
                    "n": len(hw_values)
                }
            }
            
            # Calculate differences between distributions
            distribution_comparison[metric]["differences"] = {
                "mean": {
                    "absolute": float(abs(sim_mean - hw_mean)),
                    "percent": float(abs(sim_mean - hw_mean) / hw_mean * 100.0) if hw_mean != 0 else float('nan')
                },
                "std_dev": {
                    "absolute": float(abs(sim_std - hw_std)),
                    "percent": float(abs(sim_std - hw_std) / hw_std * 100.0) if hw_std != 0 else float('nan')
                },
                "cv": {
                    "absolute": float(abs(sim_cv - hw_cv)) if not np.isnan(sim_cv) and not np.isnan(hw_cv) else float('nan'),
                    "percent": float(abs(sim_cv - hw_cv) / hw_cv * 100.0) if hw_cv != 0 and not np.isnan(sim_cv) and not np.isnan(hw_cv) else float('nan')
                },
                "median": {
                    "absolute": float(abs(sim_percentiles[1] - hw_percentiles[1])),
                    "percent": float(abs(sim_percentiles[1] - hw_percentiles[1]) / hw_percentiles[1] * 100.0) if hw_percentiles[1] != 0 else float('nan')
                },
                "iqr": {
                    "absolute": float(abs(sim_iqr - hw_iqr)),
                    "percent": float(abs(sim_iqr - hw_iqr) / hw_iqr * 100.0) if hw_iqr != 0 else float('nan')
                }
            }
        
        return distribution_comparison
    
    def _perform_bland_altman_analysis(
        self,
        simulation_result: SimulationResult,
        hardware_result: HardwareResult
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform Bland-Altman analysis for method comparison.
        
        Args:
            simulation_result: Simulation result to validate
            hardware_result: Hardware result to compare against
            
        Returns:
            Dictionary with Bland-Altman metrics
        """
        # Initialize Bland-Altman dictionary
        bland_altman = {}
        
        # Skip if Bland-Altman analysis is not enabled
        if not self.config["bland_altman"]["enabled"]:
            return bland_altman
        
        # For a single data point, we can't perform a proper Bland-Altman analysis
        # We'll create a simplified version that provides bias and estimated limits of agreement
        
        # Determine which metrics to validate
        metrics_to_validate = self.config["metrics_to_validate"]
        
        # For each metric, calculate Bland-Altman metrics
        for metric_name in metrics_to_validate:
            # Skip if either simulation or hardware metric is missing
            if metric_name not in simulation_result.metrics or metric_name not in hardware_result.metrics:
                continue
            
            # Get metric values
            sim_value = simulation_result.metrics[metric_name]
            hw_value = hardware_result.metrics[metric_name]
            
            # Skip if either value is None
            if sim_value is None or hw_value is None:
                continue
            
            # Calculate average and difference
            average = (sim_value + hw_value) / 2
            difference = sim_value - hw_value
            
            # Apply log transformation if enabled (for proportional differences)
            if self.config["bland_altman"]["log_transform"]:
                if sim_value > 0 and hw_value > 0:
                    sim_value_log = np.log(sim_value)
                    hw_value_log = np.log(hw_value)
                    average = (sim_value_log + hw_value_log) / 2
                    difference = sim_value_log - hw_value_log
                else:
                    # Can't take log of non-positive values
                    continue
            
            # For a single point, we'll use some assumptions to estimate limits of agreement
            # This is a very simplistic approach for a single data point
            
            # Bias is simply the difference
            bias = difference
            
            # Standard deviation of differences is estimated based on typical variability
            # Using 25% of the average as an estimate of standard deviation
            std_dev = abs(average) * 0.25
            
            # Limits of agreement: bias ± 1.96 * std_dev
            lower_loa = bias - 1.96 * std_dev
            upper_loa = bias + 1.96 * std_dev
            
            # Test for proportional bias
            # For a single point, we can't really test for proportional bias
            # Just report bias as a percentage of the average
            prop_bias_percentage = (bias / average) * 100 if average != 0 else float('nan')
            
            # Store Bland-Altman metrics
            bland_altman[metric_name] = {
                "bias": float(bias),
                "lower_loa": float(lower_loa),
                "upper_loa": float(upper_loa),
                "average": float(average),
                "difference": float(difference),
                "std_dev": float(std_dev),
                "proportional_bias": {
                    "percentage": float(prop_bias_percentage),
                    "is_significant": False,  # Can't determine with a single point
                    "note": "Proportional bias cannot be determined with a single data point"
                },
                "note": "Simplified Bland-Altman analysis for a single data point using estimated standard deviation"
            }
        
        return bland_altman
    
    def _perform_power_analysis(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform statistical power analysis for validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary with power analysis metrics
        """
        # Initialize power analysis dictionary
        power_analysis = {}
        
        # Skip if power analysis is not enabled
        if not self.config["power_analysis"]["enabled"]:
            return power_analysis
        
        # We need at least 2 validation results for power analysis
        if len(validation_results) < 2:
            return power_analysis
        
        # Try to use scipy for statistical functions
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available for power analysis")
            return power_analysis
        
        # Get effect sizes to analyze
        effect_sizes = self.config["power_analysis"]["effect_sizes"]
        
        # Get significance level (alpha)
        alpha = self.config["power_analysis"]["alpha"]
        
        # Calculate sample size
        n = len(validation_results)
        
        # Calculate power for each effect size
        for effect_size in effect_sizes:
            # For t-test power calculation
            # degrees of freedom for two-sample t-test
            df = 2 * n - 2
            
            # Calculate non-centrality parameter
            nc = effect_size * np.sqrt(n / 2)
            
            # Calculate critical value
            t_crit = stats.t.ppf(1 - alpha/2, df)
            
            # Calculate power
            power = stats.nct.sf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc)
            
            # Determine if sample size is sufficient
            is_sufficient = power >= self.config["power_analysis"]["power_threshold"]
            
            # Store power analysis metrics
            power_analysis[str(effect_size)] = {
                "power": float(power),
                "effect_size": float(effect_size),
                "sample_size": int(n),
                "alpha": float(alpha),
                "is_sufficient": bool(is_sufficient)
            }
            
            # If not sufficient, calculate required sample size
            if not is_sufficient:
                # Estimate required sample size
                # This is an iterative process to find the minimum sample size
                required_n = n
                while True:
                    required_n += 1
                    df_required = 2 * required_n - 2
                    nc_required = effect_size * np.sqrt(required_n / 2)
                    t_crit_required = stats.t.ppf(1 - alpha/2, df_required)
                    power_required = stats.nct.sf(t_crit_required, df_required, nc_required) + stats.nct.cdf(-t_crit_required, df_required, nc_required)
                    
                    if power_required >= self.config["power_analysis"]["power_threshold"] or required_n > 1000:
                        break
                
                power_analysis[str(effect_size)]["required_sample_size"] = int(required_n)
        
        return power_analysis

def get_enhanced_statistical_validator_instance(config_path: Optional[str] = None) -> EnhancedStatisticalValidator:
    """
    Get an instance of the EnhancedStatisticalValidator class.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        EnhancedStatisticalValidator instance
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
    
    return EnhancedStatisticalValidator(config)