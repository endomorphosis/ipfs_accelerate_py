#!/usr/bin/env python3
"""
Simulation Comparison Pipeline for the Simulation Accuracy and Validation Framework.

This module provides a pipeline for comparing simulation results with real hardware
measurements, including data collection, preprocessing, alignment, and comparison.
It focuses on efficient and accurate comparison of simulation vs. real hardware results.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simulation_comparison_pipeline")

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

class ComparisonPipeline:
    """
    Pipeline for comparing simulation results with real hardware measurements.
    
    This class provides functionality for collecting, preprocessing, aligning,
    and comparing simulation and hardware results to produce detailed validation
    insights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the comparison pipeline.
        
        Args:
            config: Configuration options for the pipeline
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            # Core comparison settings
            "metrics_to_compare": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w",
                "initialization_time_ms",
                "warmup_time_ms"
            ],
            
            # Statistical methods
            "comparison_methods": [
                "absolute_error",
                "relative_error",
                "mape",
                "percent_error",
                "rmse",
                "pearson_correlation",
                "spearman_correlation"
            ],
            
            # Additional analysis options
            "distribution_analysis": {
                "enabled": True,
                "methods": ["ks_test", "kl_divergence"],
                "bin_count": 10
            },
            
            # Ranking analysis options
            "ranking_analysis": {
                "enabled": True,
                "metrics": ["kendall_tau", "spearman_rho", "percentage_same_top_n"]
            },
            
            # Alignment options
            "alignment": {
                "require_exact_match": True,
                "time_window_seconds": 3600,  # For time-based matching
                "interpolation_method": "linear"  # For unaligned data points
            },
            
            # Processing options
            "preprocessing": {
                "remove_warmup_iterations": True,
                "warmup_iterations_count": 1,
                "outlier_detection": {
                    "enabled": True,
                    "method": "iqr",  # iqr, zscore, or dbscan
                    "threshold": 2.0
                },
                "normalization": {
                    "enabled": False,
                    "method": "min_max"  # min_max, standard, or robust
                }
            }
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def collect_data(
        self,
        simulation_results: List[SimulationResult],
        hardware_results: List[HardwareResult]
    ) -> Tuple[List[SimulationResult], List[HardwareResult]]:
        """
        Collect and filter simulation and hardware results for comparison.
        
        Args:
            simulation_results: List of simulation results
            hardware_results: List of hardware results
            
        Returns:
            Tuple of filtered simulation and hardware results
        """
        # Filter out None or empty results
        filtered_simulation_results = [r for r in simulation_results if r and r.metrics]
        filtered_hardware_results = [r for r in hardware_results if r and r.metrics]
        
        if not filtered_simulation_results:
            logger.warning("No valid simulation results provided")
        
        if not filtered_hardware_results:
            logger.warning("No valid hardware results provided")
        
        return filtered_simulation_results, filtered_hardware_results
    
    def preprocess_data(
        self,
        simulation_results: List[SimulationResult],
        hardware_results: List[HardwareResult]
    ) -> Tuple[List[SimulationResult], List[HardwareResult]]:
        """
        Preprocess simulation and hardware results to improve comparison accuracy.
        
        Args:
            simulation_results: List of simulation results
            hardware_results: List of hardware results
            
        Returns:
            Tuple of preprocessed simulation and hardware results
        """
        # Apply preprocessing steps based on configuration
        preprocessed_simulation_results = simulation_results.copy()
        preprocessed_hardware_results = hardware_results.copy()
        
        # Remove warmup iterations if configured
        if self.config["preprocessing"]["remove_warmup_iterations"]:
            warmup_count = self.config["preprocessing"]["warmup_iterations_count"]
            
            if len(preprocessed_simulation_results) > warmup_count:
                preprocessed_simulation_results = preprocessed_simulation_results[warmup_count:]
            
            if len(preprocessed_hardware_results) > warmup_count:
                preprocessed_hardware_results = preprocessed_hardware_results[warmup_count:]
        
        # Perform outlier detection if enabled
        if self.config["preprocessing"]["outlier_detection"]["enabled"]:
            method = self.config["preprocessing"]["outlier_detection"]["method"]
            threshold = self.config["preprocessing"]["outlier_detection"]["threshold"]
            
            preprocessed_simulation_results = self._remove_outliers(preprocessed_simulation_results, method, threshold)
            preprocessed_hardware_results = self._remove_outliers(preprocessed_hardware_results, method, threshold)
        
        return preprocessed_simulation_results, preprocessed_hardware_results
    
    def align_data(
        self,
        simulation_results: List[SimulationResult],
        hardware_results: List[HardwareResult]
    ) -> List[Tuple[SimulationResult, HardwareResult]]:
        """
        Align simulation and hardware results for direct comparison.
        
        Args:
            simulation_results: List of simulation results
            hardware_results: List of hardware results
            
        Returns:
            List of aligned (simulation_result, hardware_result) pairs
        """
        aligned_pairs = []
        
        # Determine alignment method based on configuration
        require_exact_match = self.config["alignment"]["require_exact_match"]
        
        # Create a mapping of hardware results for faster lookup
        hardware_mapping = {}
        for hw_result in hardware_results:
            key = self._create_alignment_key(hw_result)
            if key not in hardware_mapping:
                hardware_mapping[key] = []
            hardware_mapping[key].append(hw_result)
        
        # Match simulation results to hardware results
        for sim_result in simulation_results:
            key = self._create_alignment_key(sim_result)
            
            if key in hardware_mapping:
                # Find closest hardware result by timestamp
                matching_hw_results = hardware_mapping[key]
                
                if matching_hw_results:
                    if len(matching_hw_results) == 1:
                        # Only one match, use it directly
                        aligned_pairs.append((sim_result, matching_hw_results[0]))
                    else:
                        # Multiple matches, find closest by timestamp
                        sim_time = datetime.datetime.fromisoformat(sim_result.timestamp)
                        closest_hw_result = None
                        closest_time_diff = None
                        
                        for hw_result in matching_hw_results:
                            hw_time = datetime.datetime.fromisoformat(hw_result.timestamp)
                            time_diff = abs((sim_time - hw_time).total_seconds())
                            
                            if closest_time_diff is None or time_diff < closest_time_diff:
                                closest_time_diff = time_diff
                                closest_hw_result = hw_result
                        
                        # Check if within time window if specified
                        time_window = self.config["alignment"]["time_window_seconds"]
                        if time_window > 0 and closest_time_diff > time_window:
                            logger.warning(f"Closest match for {key} exceeds time window ({closest_time_diff} > {time_window} seconds)")
                        
                        aligned_pairs.append((sim_result, closest_hw_result))
            else:
                logger.warning(f"No matching hardware result found for simulation result: {key}")
        
        return aligned_pairs
    
    def compare_results(
        self,
        aligned_pairs: List[Tuple[SimulationResult, HardwareResult]]
    ) -> List[ValidationResult]:
        """
        Compare aligned simulation and hardware results to generate validation results.
        
        Args:
            aligned_pairs: List of aligned (simulation_result, hardware_result) pairs
            
        Returns:
            List of validation results
        """
        validation_results = []
        
        for sim_result, hw_result in aligned_pairs:
            # Calculate metrics comparison
            metrics_comparison = self._calculate_metrics_comparison(sim_result.metrics, hw_result.metrics)
            
            # Generate additional metrics
            additional_metrics = self._generate_additional_metrics(sim_result, hw_result, metrics_comparison)
            
            # Create validation result
            validation_result = ValidationResult(
                simulation_result=sim_result,
                hardware_result=hw_result,
                metrics_comparison=metrics_comparison,
                validation_timestamp=datetime.datetime.now().isoformat(),
                validation_version="comparison_pipeline_v1.0",
                additional_metrics=additional_metrics
            )
            
            validation_results.append(validation_result)
        
        return validation_results
    
    def run_pipeline(
        self,
        simulation_results: List[SimulationResult],
        hardware_results: List[HardwareResult]
    ) -> List[ValidationResult]:
        """
        Run the complete comparison pipeline from data collection to comparison.
        
        Args:
            simulation_results: List of simulation results
            hardware_results: List of hardware results
            
        Returns:
            List of validation results
        """
        # Step 1: Collect and filter data
        collected_sim_results, collected_hw_results = self.collect_data(
            simulation_results, hardware_results)
        
        if not collected_sim_results or not collected_hw_results:
            logger.warning("No valid results to compare")
            return []
        
        # Step 2: Preprocess data
        preprocessed_sim_results, preprocessed_hw_results = self.preprocess_data(
            collected_sim_results, collected_hw_results)
        
        # Step 3: Align data
        aligned_pairs = self.align_data(preprocessed_sim_results, preprocessed_hw_results)
        
        if not aligned_pairs:
            logger.warning("No aligned pairs found for comparison")
            return []
        
        # Step 4: Compare results
        validation_results = self.compare_results(aligned_pairs)
        
        return validation_results
    
    def analyze_distribution(
        self,
        simulation_values: List[float],
        hardware_values: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze the distributions of simulation and hardware values.
        
        Args:
            simulation_values: List of values from simulation results
            hardware_values: List of values from hardware results
            
        Returns:
            Dictionary with distribution analysis results
        """
        if not simulation_values or not hardware_values:
            return {"status": "error", "message": "Insufficient data for distribution analysis"}
        
        analysis = {}
        
        # Convert to numpy arrays for analysis
        sim_array = np.array(simulation_values)
        hw_array = np.array(hardware_values)
        
        # Basic statistics
        analysis["simulation"] = {
            "mean": np.mean(sim_array),
            "median": np.median(sim_array),
            "std_dev": np.std(sim_array),
            "min": np.min(sim_array),
            "max": np.max(sim_array),
            "p25": np.percentile(sim_array, 25),
            "p75": np.percentile(sim_array, 75),
            "p95": np.percentile(sim_array, 95)
        }
        
        analysis["hardware"] = {
            "mean": np.mean(hw_array),
            "median": np.median(hw_array),
            "std_dev": np.std(hw_array),
            "min": np.min(hw_array),
            "max": np.max(hw_array),
            "p25": np.percentile(hw_array, 25),
            "p75": np.percentile(hw_array, 75),
            "p95": np.percentile(hw_array, 95)
        }
        
        # Statistical tests
        if "ks_test" in self.config["distribution_analysis"]["methods"]:
            try:
                from scipy import stats
                ks_stat, ks_pvalue = stats.ks_2samp(sim_array, hw_array)
                
                analysis["ks_test"] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_pvalue),
                    "is_significant": ks_pvalue < 0.05,
                    "interpretation": (
                        "Distributions are significantly different" 
                        if ks_pvalue < 0.05 
                        else "No significant difference between distributions"
                    )
                }
            except ImportError:
                logger.warning("scipy not available for KS test")
                analysis["ks_test"] = {"status": "error", "message": "scipy not available"}
        
        # KL divergence
        if "kl_divergence" in self.config["distribution_analysis"]["methods"]:
            try:
                from scipy import stats
                
                # Create histograms for KL divergence calculation
                bin_count = self.config["distribution_analysis"]["bin_count"]
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
                
                analysis["kl_divergence"] = {
                    "sim_to_hw": kl_sim_to_hw,
                    "hw_to_sim": kl_hw_to_sim,
                    "symmetric": symmetric_kl,
                    "interpretation": (
                        "High divergence between distributions" 
                        if symmetric_kl > 1.0 
                        else "Moderate divergence between distributions" 
                        if symmetric_kl > 0.5 
                        else "Low divergence between distributions"
                    )
                }
            except ImportError:
                logger.warning("scipy not available for KL divergence")
                analysis["kl_divergence"] = {"status": "error", "message": "scipy not available"}
        
        return analysis
    
    def analyze_rankings(
        self,
        simulation_values: Dict[str, float],
        hardware_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze how well simulation preserves rankings compared to hardware.
        
        Args:
            simulation_values: Dictionary mapping keys to simulation values
            hardware_values: Dictionary mapping keys to hardware values
            
        Returns:
            Dictionary with ranking analysis results
        """
        if not simulation_values or not hardware_values:
            return {"status": "error", "message": "Insufficient data for ranking analysis"}
        
        # Get common keys
        common_keys = set(simulation_values.keys()) & set(hardware_values.keys())
        
        if len(common_keys) < 2:
            return {"status": "error", "message": "Insufficient common data points for ranking analysis"}
        
        analysis = {}
        
        # Extract values for common keys
        common_sim_values = {k: simulation_values[k] for k in common_keys}
        common_hw_values = {k: hardware_values[k] for k in common_keys}
        
        # Create rankings
        sim_ranking = {k: rank for rank, (k, _) in enumerate(sorted(common_sim_values.items(), key=lambda x: x[1], reverse=True), 1)}
        hw_ranking = {k: rank for rank, (k, _) in enumerate(sorted(common_hw_values.items(), key=lambda x: x[1], reverse=True), 1)}
        
        # Calculate rank correlation metrics
        if "kendall_tau" in self.config["ranking_analysis"]["metrics"]:
            try:
                from scipy import stats
                
                sim_ranks = [sim_ranking[k] for k in common_keys]
                hw_ranks = [hw_ranking[k] for k in common_keys]
                
                tau, p_value = stats.kendalltau(sim_ranks, hw_ranks)
                
                analysis["kendall_tau"] = {
                    "coefficient": float(tau),
                    "p_value": float(p_value),
                    "is_significant": p_value < 0.05,
                    "interpretation": (
                        "Strong rank correlation" if abs(tau) > 0.7 else
                        "Moderate rank correlation" if abs(tau) > 0.4 else
                        "Weak rank correlation"
                    )
                }
            except ImportError:
                logger.warning("scipy not available for Kendall's Tau calculation")
                analysis["kendall_tau"] = {"status": "error", "message": "scipy not available"}
        
        if "spearman_rho" in self.config["ranking_analysis"]["metrics"]:
            try:
                from scipy import stats
                
                sim_ranks = [sim_ranking[k] for k in common_keys]
                hw_ranks = [hw_ranking[k] for k in common_keys]
                
                rho, p_value = stats.spearmanr(sim_ranks, hw_ranks)
                
                analysis["spearman_rho"] = {
                    "coefficient": float(rho),
                    "p_value": float(p_value),
                    "is_significant": p_value < 0.05,
                    "interpretation": (
                        "Strong rank correlation" if abs(rho) > 0.7 else
                        "Moderate rank correlation" if abs(rho) > 0.4 else
                        "Weak rank correlation"
                    )
                }
            except ImportError:
                logger.warning("scipy not available for Spearman's Rho calculation")
                analysis["spearman_rho"] = {"status": "error", "message": "scipy not available"}
        
        if "percentage_same_top_n" in self.config["ranking_analysis"]["metrics"]:
            # Calculate percentage of items that are in the top N in both rankings
            for n in [1, 3, 5, 10]:
                if n <= len(common_keys):
                    top_n_sim = {k for k, rank in sim_ranking.items() if rank <= n}
                    top_n_hw = {k for k, rank in hw_ranking.items() if rank <= n}
                    
                    common_top_n = top_n_sim & top_n_hw
                    percentage = len(common_top_n) / n * 100
                    
                    analysis[f"percentage_same_top_{n}"] = {
                        "percentage": percentage,
                        "common_items": len(common_top_n),
                        "total_items": n,
                        "interpretation": (
                            "Excellent ranking preservation" if percentage >= 80 else
                            "Good ranking preservation" if percentage >= 60 else
                            "Moderate ranking preservation" if percentage >= 40 else
                            "Poor ranking preservation"
                        )
                    }
        
        return analysis
    
    def _create_alignment_key(self, result: Union[SimulationResult, HardwareResult]) -> str:
        """
        Create a key for aligning simulation and hardware results.
        
        Args:
            result: SimulationResult or HardwareResult
            
        Returns:
            Alignment key string
        """
        return f"{result.model_id}_{result.hardware_id}_{result.batch_size}_{result.precision}"
    
    def _remove_outliers(
        self,
        results: List[Union[SimulationResult, HardwareResult]],
        method: str,
        threshold: float
    ) -> List[Union[SimulationResult, HardwareResult]]:
        """
        Remove outliers from simulation or hardware results.
        
        Args:
            results: List of results to process
            method: Outlier detection method (iqr, zscore, or dbscan)
            threshold: Threshold for outlier detection
            
        Returns:
            List with outliers removed
        """
        if not results:
            return []
        
        if method == "iqr":
            return self._remove_outliers_iqr(results, threshold)
        elif method == "zscore":
            return self._remove_outliers_zscore(results, threshold)
        elif method == "dbscan":
            return self._remove_outliers_dbscan(results, threshold)
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return results
    
    def _remove_outliers_iqr(
        self,
        results: List[Union[SimulationResult, HardwareResult]],
        threshold: float
    ) -> List[Union[SimulationResult, HardwareResult]]:
        """
        Remove outliers using the Interquartile Range (IQR) method.
        
        Args:
            results: List of results to process
            threshold: Multiplier for IQR
            
        Returns:
            List with outliers removed
        """
        if not results:
            return []
        
        filtered_results = []
        
        # Extract data for each metric
        metrics_data = {}
        for metric in self.config["metrics_to_compare"]:
            values = []
            for result in results:
                if metric in result.metrics and result.metrics[metric] is not None:
                    values.append(result.metrics[metric])
            
            if values:
                metrics_data[metric] = values
        
        # Calculate IQR and bounds for each metric
        metric_bounds = {}
        for metric, values in metrics_data.items():
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            metric_bounds[metric] = (lower_bound, upper_bound)
        
        # Filter results
        for result in results:
            is_outlier = False
            
            for metric, (lower_bound, upper_bound) in metric_bounds.items():
                if (metric in result.metrics and 
                    result.metrics[metric] is not None and 
                    (result.metrics[metric] < lower_bound or result.metrics[metric] > upper_bound)):
                    is_outlier = True
                    break
            
            if not is_outlier:
                filtered_results.append(result)
        
        return filtered_results
    
    def _remove_outliers_zscore(
        self,
        results: List[Union[SimulationResult, HardwareResult]],
        threshold: float
    ) -> List[Union[SimulationResult, HardwareResult]]:
        """
        Remove outliers using the Z-score method.
        
        Args:
            results: List of results to process
            threshold: Z-score threshold
            
        Returns:
            List with outliers removed
        """
        if not results:
            return []
        
        filtered_results = []
        
        # Extract data for each metric
        metrics_data = {}
        for metric in self.config["metrics_to_compare"]:
            values = []
            for result in results:
                if metric in result.metrics and result.metrics[metric] is not None:
                    values.append(result.metrics[metric])
            
            if values:
                metrics_data[metric] = values
        
        # Calculate mean and standard deviation for each metric
        metric_stats = {}
        for metric, values in metrics_data.items():
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                # Cannot calculate z-scores if std is 0, skip this metric
                continue
            
            metric_stats[metric] = (mean, std)
        
        # Filter results
        for result in results:
            is_outlier = False
            
            for metric, (mean, std) in metric_stats.items():
                if (metric in result.metrics and 
                    result.metrics[metric] is not None):
                    z_score = abs((result.metrics[metric] - mean) / std)
                    
                    if z_score > threshold:
                        is_outlier = True
                        break
            
            if not is_outlier:
                filtered_results.append(result)
        
        return filtered_results
    
    def _remove_outliers_dbscan(
        self,
        results: List[Union[SimulationResult, HardwareResult]],
        eps: float
    ) -> List[Union[SimulationResult, HardwareResult]]:
        """
        Remove outliers using the DBSCAN clustering method.
        
        Args:
            results: List of results to process
            eps: DBSCAN eps parameter (max distance between points in a cluster)
            
        Returns:
            List with outliers removed
        """
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("sklearn not available for DBSCAN outlier detection")
            return results
        
        if not results:
            return []
        
        # Extract features for clustering
        features = []
        for result in results:
            result_features = []
            
            for metric in self.config["metrics_to_compare"]:
                if metric in result.metrics and result.metrics[metric] is not None:
                    result_features.append(result.metrics[metric])
                else:
                    # Use 0 for missing values (will be scaled anyway)
                    result_features.append(0)
            
            features.append(result_features)
        
        if not features:
            return results
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=3)
        labels = dbscan.fit_predict(scaled_features)
        
        # Filter out outliers (points labeled as -1)
        filtered_results = []
        for i, result in enumerate(results):
            if labels[i] != -1:  # -1 indicates an outlier
                filtered_results.append(result)
        
        return filtered_results
    
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
        metrics_to_compare = self.config["metrics_to_compare"]
        
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
            comparison_methods = self.config["comparison_methods"]
            
            if "absolute_error" in comparison_methods:
                metrics_comparison[metric_name]["absolute_error"] = abs(sim_value - hw_value)
            
            if "relative_error" in comparison_methods:
                if hw_value != 0:
                    metrics_comparison[metric_name]["relative_error"] = abs(sim_value - hw_value) / abs(hw_value)
                else:
                    # Handle division by zero
                    metrics_comparison[metric_name]["relative_error"] = float('nan')
            
            if "mape" in comparison_methods:
                if hw_value != 0:
                    metrics_comparison[metric_name]["mape"] = abs(sim_value - hw_value) / abs(hw_value) * 100.0
                else:
                    # Handle division by zero
                    metrics_comparison[metric_name]["mape"] = float('nan')
            
            if "percent_error" in comparison_methods:
                if hw_value != 0:
                    metrics_comparison[metric_name]["percent_error"] = (sim_value - hw_value) / abs(hw_value) * 100.0
                else:
                    # Handle division by zero
                    metrics_comparison[metric_name]["percent_error"] = float('nan')
            
            if "rmse" in comparison_methods:
                # For a single data point, RMSE is the same as absolute error
                metrics_comparison[metric_name]["rmse"] = abs(sim_value - hw_value)
        
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
                
                # Add interpretation
                if correlation_coefficient >= 0.9:
                    additional_metrics["correlation_interpretation"] = "Very strong correlation"
                elif correlation_coefficient >= 0.7:
                    additional_metrics["correlation_interpretation"] = "Strong correlation"
                elif correlation_coefficient >= 0.5:
                    additional_metrics["correlation_interpretation"] = "Moderate correlation"
                elif correlation_coefficient >= 0.3:
                    additional_metrics["correlation_interpretation"] = "Weak correlation"
                else:
                    additional_metrics["correlation_interpretation"] = "Very weak or no correlation"
                
            except Exception as e:
                logger.warning(f"Error calculating correlation coefficient: {e}")
        
        return additional_metrics


def get_comparison_pipeline_instance(config_path: Optional[str] = None) -> ComparisonPipeline:
    """
    Get an instance of the ComparisonPipeline class.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ComparisonPipeline instance
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
    
    return ComparisonPipeline(config)