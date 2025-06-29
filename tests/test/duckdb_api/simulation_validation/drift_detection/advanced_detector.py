#!/usr/bin/env python3
"""
Advanced Drift Detector implementation for the Simulation Accuracy and Validation Framework.

This module provides a more sophisticated implementation of the DriftDetector interface
that uses advanced statistical methods and multi-dimensional analysis to detect drift in simulation accuracy.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("advanced_drift_detector")

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
    DriftDetector
)

# Import basic detector for fallback and utilities
from duckdb_api.simulation_validation.drift_detection.basic_detector import BasicDriftDetector

# Optional imports for advanced methods
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    import scipy.stats as stats
    sklearn_available = True
except ImportError:
    logger.warning("scikit-learn not available, some advanced drift detection methods will be limited")
    sklearn_available = False


class AdvancedDriftDetector(DriftDetector):
    """
    Advanced implementation of a drift detector using sophisticated statistical methods.
    
    This detector provides enhanced capabilities including:
    1. Multi-dimensional drift analysis across metrics
    2. Time-series trend analysis for detecting gradual drift
    3. Distribution-based drift detection with multiple statistical tests
    4. Feature correlation analysis to identify drift in relationships
    5. Anomaly detection for early warning of potential drift
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced drift detector.
        
        Args:
            config: Configuration options for the detector
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "metrics_to_monitor": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w"
            ],
            "drift_thresholds": {
                # Default thresholds for MAPE change to trigger drift detection
                "throughput_items_per_second": 3.0,  # 3% absolute change in MAPE
                "average_latency_ms": 3.0,
                "memory_peak_mb": 3.0,
                "power_consumption_w": 3.0,
                "overall": 2.0  # 2% absolute change in overall MAPE
            },
            "use_multi_dimensional_analysis": True,
            "use_distribution_tests": True,
            "use_correlation_analysis": True,
            "use_time_series_analysis": True,
            "use_anomaly_detection": True,
            "significance_level": 0.05,      # p-value threshold for significance
            "min_samples": 5,                # Minimum number of samples for drift detection
            "distribution_tests": ["ks_2samp", "anderson_ksamp", "epps_singleton_2samp"],
            "anomaly_detection_method": "isolation_forest",  # isolation_forest, local_outlier_factor, or dbscan
            "anomaly_detection_contamination": 0.05,  # Expected proportion of anomalies
            "trend_analysis_window": 10,     # Number of points for trend analysis
            "correlation_significance_threshold": 0.05,  # Threshold for correlation change significance
            "multi_dimensional_manifold_method": "pca"  # PCA for dimension reduction
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Initialize drift thresholds
        self.drift_thresholds = self.config["drift_thresholds"]
        
        # Initialize basic detector for fallback
        self.basic_detector = BasicDriftDetector(self.config)
        
        # Initialize drift status
        self.current_drift_status = {
            "is_drifting": False,
            "drift_metrics": {},
            "last_check_time": None,
            "significant_metrics": [],
            "multi_dimensional_drift": {
                "detected": False,
                "p_value": None,
                "method": None
            },
            "distribution_drift": {
                "detected": False,
                "metrics": {}
            },
            "correlation_drift": {
                "detected": False,
                "changes": {}
            },
            "time_series_drift": {
                "detected": False,
                "trends": {}
            },
            "anomaly_detection": {
                "anomalies_detected": False,
                "metrics": {}
            }
        }
    
    def detect_drift(
        self,
        historical_validation_results: List[ValidationResult],
        new_validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Detect drift in simulation accuracy using advanced methods.
        
        Args:
            historical_validation_results: Historical validation results
            new_validation_results: New validation results
            
        Returns:
            Drift detection results with comprehensive metrics and significance
        """
        if not historical_validation_results or not new_validation_results:
            return {
                "status": "error",
                "message": "Insufficient validation results for drift detection",
                "is_significant": False,
                "drift_metrics": {}
            }
        
        # Check if we have enough samples
        if len(historical_validation_results) < self.config["min_samples"] or len(new_validation_results) < self.config["min_samples"]:
            return {
                "status": "error",
                "message": f"Insufficient samples for drift detection (need at least {self.config['min_samples']})",
                "is_significant": False,
                "drift_metrics": {}
            }
        
        # Start with basic drift detection
        basic_drift_results = self.basic_detector.detect_drift(
            historical_validation_results, new_validation_results
        )
        
        # Initialize comprehensive drift results
        drift_results = basic_drift_results.copy()
        drift_results["analysis_types"] = ["basic"]
        drift_results["multi_dimensional_analysis"] = None
        drift_results["distribution_analysis"] = None
        drift_results["correlation_analysis"] = None
        drift_results["time_series_analysis"] = None
        drift_results["anomaly_detection"] = None
        
        # Run additional drift detection methods based on configuration
        # 1. Multi-dimensional analysis
        if self.config["use_multi_dimensional_analysis"] and sklearn_available:
            drift_results["analysis_types"].append("multi_dimensional")
            multi_dim_results = self._perform_multi_dimensional_analysis(
                historical_validation_results, new_validation_results
            )
            drift_results["multi_dimensional_analysis"] = multi_dim_results
            
            # Update overall significance if multi-dimensional drift is detected
            if multi_dim_results["is_significant"]:
                drift_results["is_significant"] = True
                if "multi_dimensional" not in drift_results.get("significant_methods", []):
                    if "significant_methods" not in drift_results:
                        drift_results["significant_methods"] = []
                    drift_results["significant_methods"].append("multi_dimensional")
        
        # 2. Distribution analysis
        if self.config["use_distribution_tests"]:
            drift_results["analysis_types"].append("distribution")
            dist_results = self._perform_distribution_analysis(
                historical_validation_results, new_validation_results
            )
            drift_results["distribution_analysis"] = dist_results
            
            # Update overall significance if distribution drift is detected
            if dist_results["is_significant"]:
                drift_results["is_significant"] = True
                if "distribution" not in drift_results.get("significant_methods", []):
                    if "significant_methods" not in drift_results:
                        drift_results["significant_methods"] = []
                    drift_results["significant_methods"].append("distribution")
        
        # 3. Correlation analysis
        if self.config["use_correlation_analysis"]:
            drift_results["analysis_types"].append("correlation")
            corr_results = self._perform_correlation_analysis(
                historical_validation_results, new_validation_results
            )
            drift_results["correlation_analysis"] = corr_results
            
            # Update overall significance if correlation drift is detected
            if corr_results["is_significant"]:
                drift_results["is_significant"] = True
                if "correlation" not in drift_results.get("significant_methods", []):
                    if "significant_methods" not in drift_results:
                        drift_results["significant_methods"] = []
                    drift_results["significant_methods"].append("correlation")
        
        # 4. Time-series analysis
        if self.config["use_time_series_analysis"]:
            drift_results["analysis_types"].append("time_series")
            ts_results = self._perform_time_series_analysis(
                historical_validation_results, new_validation_results
            )
            drift_results["time_series_analysis"] = ts_results
            
            # Update overall significance if time-series drift is detected
            if ts_results["is_significant"]:
                drift_results["is_significant"] = True
                if "time_series" not in drift_results.get("significant_methods", []):
                    if "significant_methods" not in drift_results:
                        drift_results["significant_methods"] = []
                    drift_results["significant_methods"].append("time_series")
        
        # 5. Anomaly detection
        if self.config["use_anomaly_detection"] and sklearn_available:
            drift_results["analysis_types"].append("anomaly")
            anomaly_results = self._perform_anomaly_detection(
                historical_validation_results, new_validation_results
            )
            drift_results["anomaly_detection"] = anomaly_results
            
            # Update overall significance if anomalies are detected
            if anomaly_results["is_significant"]:
                drift_results["is_significant"] = True
                if "anomaly" not in drift_results.get("significant_methods", []):
                    if "significant_methods" not in drift_results:
                        drift_results["significant_methods"] = []
                    drift_results["significant_methods"].append("anomaly")
        
        # Update drift status
        self.current_drift_status = {
            "is_drifting": drift_results["is_significant"],
            "drift_metrics": drift_results.get("drift_metrics", {}),
            "last_check_time": drift_results.get("timestamp", datetime.now().isoformat()),
            "significant_metrics": drift_results.get("significant_metrics", []),
            "multi_dimensional_drift": {
                "detected": drift_results.get("multi_dimensional_analysis", {}).get("is_significant", False),
                "p_value": drift_results.get("multi_dimensional_analysis", {}).get("p_value", None),
                "method": drift_results.get("multi_dimensional_analysis", {}).get("method", None)
            },
            "distribution_drift": {
                "detected": drift_results.get("distribution_analysis", {}).get("is_significant", False),
                "metrics": drift_results.get("distribution_analysis", {}).get("metrics", {})
            },
            "correlation_drift": {
                "detected": drift_results.get("correlation_analysis", {}).get("is_significant", False),
                "changes": drift_results.get("correlation_analysis", {}).get("correlation_changes", {})
            },
            "time_series_drift": {
                "detected": drift_results.get("time_series_analysis", {}).get("is_significant", False),
                "trends": drift_results.get("time_series_analysis", {}).get("trends", {})
            },
            "anomaly_detection": {
                "anomalies_detected": drift_results.get("anomaly_detection", {}).get("is_significant", False),
                "metrics": drift_results.get("anomaly_detection", {}).get("anomalous_metrics", {})
            }
        }
        
        return drift_results
    
    def set_drift_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set thresholds for drift detection.
        
        Args:
            thresholds: Dictionary mapping metric names to threshold values
        """
        for metric, threshold in thresholds.items():
            self.drift_thresholds[metric] = threshold
        
        # Update basic detector thresholds as well
        self.basic_detector.set_drift_thresholds(thresholds)
        
        logger.info(f"Updated drift thresholds: {self.drift_thresholds}")
    
    def get_drift_status(self) -> Dict[str, Any]:
        """
        Get the current drift status.
        
        Returns:
            Dictionary with comprehensive drift status information
        """
        return self.current_drift_status
    
    def _perform_multi_dimensional_analysis(
        self,
        historical_validation_results: List[ValidationResult],
        new_validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Perform multi-dimensional analysis for drift detection.
        
        Uses dimension reduction and statistical tests to detect drift
        in the joint distribution of multiple metrics.
        
        Args:
            historical_validation_results: Historical validation results
            new_validation_results: New validation results
            
        Returns:
            Dictionary with multi-dimensional drift analysis results
        """
        if not sklearn_available:
            return {
                "status": "error",
                "message": "sklearn not available for multi-dimensional analysis",
                "is_significant": False
            }
        
        try:
            # Extract metrics from validation results
            historical_metrics = self._extract_metric_vectors(historical_validation_results)
            new_metrics = self._extract_metric_vectors(new_validation_results)
            
            if not historical_metrics or not new_metrics:
                return {
                    "status": "error",
                    "message": "Failed to extract metrics for multi-dimensional analysis",
                    "is_significant": False
                }
            
            # Convert to numpy arrays for processing
            historical_array = np.array(historical_metrics)
            new_array = np.array(new_metrics)
            
            # Apply dimension reduction if the data has more than 2 dimensions
            if historical_array.shape[1] > 2:
                method = self.config["multi_dimensional_manifold_method"]
                
                if method == "pca":
                    # Apply PCA to reduce dimensions while retaining most variance
                    pca = PCA(n_components=2)
                    pca.fit(np.vstack((historical_array, new_array)))
                    
                    historical_reduced = pca.transform(historical_array)
                    new_reduced = pca.transform(new_array)
                    
                    # Store variance explained for reporting
                    variance_explained = pca.explained_variance_ratio_.sum()
                else:
                    # Fallback to first two dimensions
                    logger.warning(f"Unknown dimension reduction method: {method}, using first two dimensions")
                    historical_reduced = historical_array[:, :2]
                    new_reduced = new_array[:, :2]
                    variance_explained = None
            else:
                # Already 2D or less, no need for reduction
                historical_reduced = historical_array
                new_reduced = new_array
                variance_explained = 1.0
            
            # Perform statistical tests on the reduced data
            # 1. Hotelling's T-squared test (multivariate extension of t-test)
            try:
                from scipy.stats import hotelling_t2
                t2_stat, p_value = hotelling_t2(historical_reduced, new_reduced)
                hotelling_significant = p_value < self.config["significance_level"]
            except (ImportError, AttributeError):
                # Hotelling's T2 might not be available in older scipy versions
                t2_stat, p_value = None, None
                hotelling_significant = False
                logger.warning("Hotelling's T2 test not available, skipping")
            
            # 2. Compute Energy Distance between distributions
            # This is a non-parametric test for equality of multivariate distributions
            try:
                from scipy.spatial.distance import pdist, squareform
                
                # Compute pairwise distances within and between groups
                historical_dist = squareform(pdist(historical_reduced))
                new_dist = squareform(pdist(new_reduced))
                
                # Compute energy distance
                h_mean = np.mean(historical_dist)
                n_mean = np.mean(new_dist)
                
                # Compute cross-distances
                cross_dists = []
                for h_point in historical_reduced:
                    for n_point in new_reduced:
                        cross_dists.append(np.linalg.norm(h_point - n_point))
                cross_mean = np.mean(cross_dists)
                
                # Energy distance formula: 2*E(cross) - E(historical) - E(new)
                energy_distance = 2 * cross_mean - h_mean - n_mean
                
                # Permutation test for significance
                num_permutations = 1000
                energy_distances_perm = []
                
                all_data = np.vstack((historical_reduced, new_reduced))
                for _ in range(num_permutations):
                    np.random.shuffle(all_data)
                    perm_historical = all_data[:len(historical_reduced)]
                    perm_new = all_data[len(historical_reduced):]
                    
                    perm_h_dist = squareform(pdist(perm_historical))
                    perm_n_dist = squareform(pdist(perm_new))
                    
                    perm_h_mean = np.mean(perm_h_dist)
                    perm_n_mean = np.mean(perm_n_dist)
                    
                    perm_cross_dists = []
                    for h_point in perm_historical:
                        for n_point in perm_new:
                            perm_cross_dists.append(np.linalg.norm(h_point - n_point))
                    perm_cross_mean = np.mean(perm_cross_dists)
                    
                    perm_energy_distance = 2 * perm_cross_mean - perm_h_mean - perm_n_mean
                    energy_distances_perm.append(perm_energy_distance)
                
                # Compute p-value as proportion of permutations with energy distance >= observed
                energy_p_value = np.mean(np.array(energy_distances_perm) >= energy_distance)
                energy_significant = energy_p_value < self.config["significance_level"]
            except Exception as e:
                logger.error(f"Error computing energy distance: {e}")
                energy_distance, energy_p_value, energy_significant = None, None, False
            
            # Determine overall significance
            is_significant = hotelling_significant or energy_significant
            
            # Create result
            result = {
                "status": "success",
                "is_significant": is_significant,
                "method": "multi_dimensional",
                "hotelling_t2": {
                    "statistic": t2_stat,
                    "p_value": p_value,
                    "is_significant": hotelling_significant
                },
                "energy_distance": {
                    "distance": energy_distance,
                    "p_value": energy_p_value,
                    "is_significant": energy_significant
                },
                "dimension_reduction": {
                    "method": self.config["multi_dimensional_manifold_method"] if historical_array.shape[1] > 2 else "none",
                    "variance_explained": variance_explained,
                    "original_dimensions": historical_array.shape[1],
                    "reduced_dimensions": historical_reduced.shape[1]
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-dimensional analysis: {e}")
            return {
                "status": "error",
                "message": f"Error in multi-dimensional analysis: {e}",
                "is_significant": False
            }
    
    def _perform_distribution_analysis(
        self,
        historical_validation_results: List[ValidationResult],
        new_validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Perform distribution analysis for drift detection.
        
        Uses multiple statistical tests to compare distributions of validation metrics.
        
        Args:
            historical_validation_results: Historical validation results
            new_validation_results: New validation results
            
        Returns:
            Dictionary with distribution drift analysis results
        """
        try:
            # Extract MAPE values for each metric
            metrics_results = {}
            is_significant = False
            
            for metric in self.config["metrics_to_monitor"]:
                # Get MAPE values for historical and new results
                historical_mape = self._extract_mape_values(historical_validation_results, metric)
                new_mape = self._extract_mape_values(new_validation_results, metric)
                
                if len(historical_mape) < 5 or len(new_mape) < 5:
                    metrics_results[metric] = {
                        "status": "skipped",
                        "message": "Insufficient samples for distribution tests",
                        "is_significant": False
                    }
                    continue
                
                # Apply configured distribution tests
                test_results = {}
                metric_significant = False
                
                # 1. Kolmogorov-Smirnov two-sample test
                if "ks_2samp" in self.config["distribution_tests"]:
                    try:
                        from scipy import stats
                        ks_stat, ks_pvalue = stats.ks_2samp(historical_mape, new_mape)
                        ks_significant = ks_pvalue < self.config["significance_level"]
                        
                        test_results["ks_2samp"] = {
                            "statistic": float(ks_stat),
                            "p_value": float(ks_pvalue),
                            "is_significant": ks_significant
                        }
                        
                        metric_significant = metric_significant or ks_significant
                    except Exception as e:
                        logger.warning(f"Error in KS test for {metric}: {e}")
                        test_results["ks_2samp"] = {"status": "error", "message": str(e)}
                
                # 2. Anderson-Darling k-sample test
                if "anderson_ksamp" in self.config["distribution_tests"]:
                    try:
                        from scipy import stats
                        ad_result = stats.anderson_ksamp([historical_mape, new_mape])
                        ad_stat, ad_critical_values, ad_significance_level = ad_result
                        ad_significant = ad_stat > ad_critical_values[0]  # Most conservative level
                        
                        test_results["anderson_ksamp"] = {
                            "statistic": float(ad_stat),
                            "critical_values": ad_critical_values.tolist(),
                            "significance_level": ad_significance_level,
                            "is_significant": ad_significant
                        }
                        
                        metric_significant = metric_significant or ad_significant
                    except Exception as e:
                        logger.warning(f"Error in Anderson-Darling test for {metric}: {e}")
                        test_results["anderson_ksamp"] = {"status": "error", "message": str(e)}
                
                # 3. Epps-Singleton two-sample test
                if "epps_singleton_2samp" in self.config["distribution_tests"]:
                    try:
                        from scipy import stats
                        es_stat, es_pvalue = stats.epps_singleton_2samp(historical_mape, new_mape)
                        es_significant = es_pvalue < self.config["significance_level"]
                        
                        test_results["epps_singleton_2samp"] = {
                            "statistic": float(es_stat),
                            "p_value": float(es_pvalue),
                            "is_significant": es_significant
                        }
                        
                        metric_significant = metric_significant or es_significant
                    except Exception as e:
                        logger.warning(f"Error in Epps-Singleton test for {metric}: {e}")
                        test_results["epps_singleton_2samp"] = {"status": "error", "message": str(e)}
                
                # 4. Mann-Whitney U test (non-parametric test for difference in medians)
                if "mann_whitney" in self.config["distribution_tests"]:
                    try:
                        from scipy import stats
                        mw_stat, mw_pvalue = stats.mannwhitneyu(historical_mape, new_mape)
                        mw_significant = mw_pvalue < self.config["significance_level"]
                        
                        test_results["mann_whitney"] = {
                            "statistic": float(mw_stat),
                            "p_value": float(mw_pvalue),
                            "is_significant": mw_significant
                        }
                        
                        metric_significant = metric_significant or mw_significant
                    except Exception as e:
                        logger.warning(f"Error in Mann-Whitney test for {metric}: {e}")
                        test_results["mann_whitney"] = {"status": "error", "message": str(e)}
                
                # Store results for this metric
                metrics_results[metric] = {
                    "status": "success",
                    "is_significant": metric_significant,
                    "tests": test_results,
                    "sample_counts": {
                        "historical": len(historical_mape),
                        "new": len(new_mape)
                    }
                }
                
                is_significant = is_significant or metric_significant
            
            # Create overall result
            result = {
                "status": "success",
                "is_significant": is_significant,
                "method": "distribution",
                "metrics": metrics_results
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in distribution analysis: {e}")
            return {
                "status": "error",
                "message": f"Error in distribution analysis: {e}",
                "is_significant": False
            }
    
    def _perform_correlation_analysis(
        self,
        historical_validation_results: List[ValidationResult],
        new_validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Perform correlation analysis for drift detection.
        
        Analyzes changes in correlations between different metrics.
        
        Args:
            historical_validation_results: Historical validation results
            new_validation_results: New validation results
            
        Returns:
            Dictionary with correlation drift analysis results
        """
        try:
            # Extract raw metric values for correlation analysis
            historical_metrics_by_result = self._extract_metrics_by_result(historical_validation_results)
            new_metrics_by_result = self._extract_metrics_by_result(new_validation_results)
            
            if not historical_metrics_by_result or not new_metrics_by_result:
                return {
                    "status": "error",
                    "message": "Failed to extract metrics for correlation analysis",
                    "is_significant": False
                }
            
            # Calculate correlations for historical data
            historical_correlations = self._calculate_metric_correlations(historical_metrics_by_result)
            
            # Calculate correlations for new data
            new_correlations = self._calculate_metric_correlations(new_metrics_by_result)
            
            # Compare correlations
            correlation_changes = {}
            is_significant = False
            
            for pair, historical_corr in historical_correlations.items():
                if pair in new_correlations:
                    new_corr = new_correlations[pair]
                    
                    # Calculate absolute change in correlation
                    abs_change = abs(new_corr - historical_corr)
                    
                    # Determine significance based on Fisher's Z transformation
                    historical_n = len(historical_metrics_by_result)
                    new_n = len(new_metrics_by_result)
                    
                    # Fisher's Z transformation
                    from math import log, sqrt
                    z_historical = 0.5 * log((1 + historical_corr) / (1 - historical_corr)) if abs(historical_corr) < 1 else 0
                    z_new = 0.5 * log((1 + new_corr) / (1 - new_corr)) if abs(new_corr) < 1 else 0
                    
                    # Standard error
                    se = sqrt(1/(historical_n - 3) + 1/(new_n - 3))
                    
                    # Z-statistic
                    z_stat = abs(z_historical - z_new) / se
                    
                    # P-value
                    from scipy import stats
                    p_value = 2 * (1 - stats.norm.cdf(z_stat))  # Two-tailed test
                    
                    is_pair_significant = p_value < self.config["correlation_significance_threshold"]
                    is_significant = is_significant or is_pair_significant
                    
                    correlation_changes[pair] = {
                        "historical_correlation": historical_corr,
                        "new_correlation": new_corr,
                        "absolute_change": abs_change,
                        "z_statistic": z_stat,
                        "p_value": p_value,
                        "is_significant": is_pair_significant,
                        "sample_counts": {
                            "historical": historical_n,
                            "new": new_n
                        }
                    }
            
            # Create result
            result = {
                "status": "success",
                "is_significant": is_significant,
                "method": "correlation",
                "correlation_changes": correlation_changes
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {
                "status": "error",
                "message": f"Error in correlation analysis: {e}",
                "is_significant": False
            }
    
    def _perform_time_series_analysis(
        self,
        historical_validation_results: List[ValidationResult],
        new_validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Perform time-series analysis for drift detection.
        
        Analyzes trends over time to detect gradual drift.
        
        Args:
            historical_validation_results: Historical validation results
            new_validation_results: New validation results
            
        Returns:
            Dictionary with time-series drift analysis results
        """
        try:
            # Sort validation results by timestamp
            historical_sorted = sorted(historical_validation_results, key=lambda x: x.validation_timestamp)
            new_sorted = sorted(new_validation_results, key=lambda x: x.validation_timestamp)
            
            # Combine and extract metrics in time order
            metrics_by_time = {}
            
            # Initialize with metrics from historical validation results
            for result in historical_sorted:
                timestamp = result.validation_timestamp
                for metric, comparison in result.metrics_comparison.items():
                    if metric in self.config["metrics_to_monitor"] and "mape" in comparison:
                        if metric not in metrics_by_time:
                            metrics_by_time[metric] = []
                        metrics_by_time[metric].append({
                            "timestamp": timestamp,
                            "mape": comparison["mape"],
                            "dataset": "historical"
                        })
            
            # Add metrics from new validation results
            for result in new_sorted:
                timestamp = result.validation_timestamp
                for metric, comparison in result.metrics_comparison.items():
                    if metric in self.config["metrics_to_monitor"] and "mape" in comparison:
                        if metric not in metrics_by_time:
                            metrics_by_time[metric] = []
                        metrics_by_time[metric].append({
                            "timestamp": timestamp,
                            "mape": comparison["mape"],
                            "dataset": "new"
                        })
            
            # Analyze trends for each metric
            trends = {}
            is_significant = False
            
            for metric, time_series in metrics_by_time.items():
                # Sort by timestamp
                time_series.sort(key=lambda x: x["timestamp"])
                
                # Need enough points for trend analysis
                if len(time_series) < self.config["trend_analysis_window"]:
                    trends[metric] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for trend analysis (need {self.config['trend_analysis_window']})",
                        "is_significant": False
                    }
                    continue
                
                # Convert to numpy arrays for calculations
                mape_values = np.array([point["mape"] for point in time_series])
                
                # Simple linear trend analysis
                x = np.arange(len(mape_values))
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, mape_values)
                
                # Mann-Kendall test for trend significance
                try:
                    from scipy.stats import kendalltau
                    tau, mk_p_value = kendalltau(x, mape_values)
                except Exception:
                    tau, mk_p_value = None, None
                
                # Check if trend is significant
                linear_significant = p_value < self.config["significance_level"] and abs(slope) > 0.05  # At least 0.05 MAPE change per point
                mk_significant = mk_p_value is not None and mk_p_value < self.config["significance_level"] and abs(tau) > 0.3  # At least moderate correlation
                
                trend_significant = linear_significant or mk_significant
                is_significant = is_significant or trend_significant
                
                # Identify change point - when does trend significantly change?
                change_point = None
                change_point_confidence = None
                
                try:
                    # Simple approach: compare slope of first half vs second half
                    mid_point = len(mape_values) // 2
                    
                    if mid_point > 5:  # Need enough points for meaningful regression
                        first_half_x = np.arange(mid_point)
                        first_half_y = mape_values[:mid_point]
                        slope_first, _, _, p_first, _ = stats.linregress(first_half_x, first_half_y)
                        
                        second_half_x = np.arange(mid_point)
                        second_half_y = mape_values[mid_point:mid_point*2]
                        slope_second, _, _, p_second, _ = stats.linregress(second_half_x, second_half_y)
                        
                        # Check if slopes are significantly different
                        if p_first < 0.1 and p_second < 0.1:  # Both slopes are significant
                            slope_change = abs(slope_second - slope_first)
                            if slope_change > 0.1:  # At least 0.1 difference in MAPE change per point
                                change_point = mid_point
                                change_point_confidence = 1.0 - max(p_first, p_second)  # Higher confidence with lower p-values
                except Exception as e:
                    logger.warning(f"Error in change point detection for {metric}: {e}")
                
                # Create trend result
                trends[metric] = {
                    "status": "success",
                    "is_significant": trend_significant,
                    "linear_regression": {
                        "slope": slope,
                        "intercept": intercept,
                        "r_value": r_value,
                        "p_value": p_value,
                        "std_err": std_err,
                        "is_significant": linear_significant
                    },
                    "mann_kendall": {
                        "tau": tau,
                        "p_value": mk_p_value,
                        "is_significant": mk_significant
                    },
                    "change_point": {
                        "index": change_point,
                        "confidence": change_point_confidence
                    },
                    "data_points": len(time_series)
                }
            
            # Create overall result
            result = {
                "status": "success",
                "is_significant": is_significant,
                "method": "time_series",
                "trends": trends
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in time-series analysis: {e}")
            return {
                "status": "error",
                "message": f"Error in time-series analysis: {e}",
                "is_significant": False
            }
    
    def _perform_anomaly_detection(
        self,
        historical_validation_results: List[ValidationResult],
        new_validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Perform anomaly detection for drift identification.
        
        Identifies outliers in new validation results that might indicate drift.
        
        Args:
            historical_validation_results: Historical validation results
            new_validation_results: New validation results
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not sklearn_available:
            return {
                "status": "error",
                "message": "sklearn not available for anomaly detection",
                "is_significant": False
            }
        
        try:
            # Extract metrics from validation results
            metric_results = {}
            is_significant = False
            
            for metric in self.config["metrics_to_monitor"]:
                # Get MAPE values for historical and new results
                historical_mape = self._extract_mape_values(historical_validation_results, metric)
                new_mape = self._extract_mape_values(new_validation_results, metric)
                
                if len(historical_mape) < 10 or len(new_mape) < 5:
                    metric_results[metric] = {
                        "status": "skipped",
                        "message": "Insufficient samples for anomaly detection",
                        "is_significant": False
                    }
                    continue
                
                # Train anomaly detector on historical data
                X_historical = np.array(historical_mape).reshape(-1, 1)
                
                method = self.config["anomaly_detection_method"]
                anomaly_detector = None
                
                if method == "isolation_forest":
                    anomaly_detector = IsolationForest(
                        contamination=self.config["anomaly_detection_contamination"],
                        random_state=42
                    )
                elif method == "local_outlier_factor":
                    anomaly_detector = LocalOutlierFactor(
                        n_neighbors=min(20, len(historical_mape) - 1),
                        contamination=self.config["anomaly_detection_contamination"]
                    )
                elif method == "dbscan":
                    # Estimate epsilon based on nearest neighbor distances
                    from sklearn.neighbors import NearestNeighbors
                    neigh = NearestNeighbors(n_neighbors=2)
                    nbrs = neigh.fit(X_historical)
                    distances, indices = nbrs.kneighbors(X_historical)
                    distances = np.sort(distances[:, 1])
                    
                    # Use knee/elbow point for epsilon
                    eps = np.percentile(distances, 90)  # Conservative estimate
                    
                    anomaly_detector = DBSCAN(eps=eps, min_samples=3)
                else:
                    logger.warning(f"Unknown anomaly detection method: {method}")
                    continue
                
                # Fit the detector
                anomaly_detector.fit(X_historical)
                
                # Predict anomalies in new data
                X_new = np.array(new_mape).reshape(-1, 1)
                
                if method == "local_outlier_factor":
                    # LOF has separate predict method
                    anomaly_scores = -anomaly_detector.negative_outlier_factor_
                    predictions = anomaly_detector.fit_predict(X_new)
                elif method == "dbscan":
                    # For DBSCAN, predict new data
                    new_clusters = anomaly_detector.fit_predict(X_new)
                    predictions = [1 if c == -1 else -1 for c in new_clusters]  # -1 means outlier in DBSCAN
                    anomaly_scores = [1 if c == -1 else 0 for c in new_clusters]
                else:
                    # For other methods, use decision_function
                    predictions = anomaly_detector.predict(X_new)
                    anomaly_scores = -anomaly_detector.decision_function(X_new)
                
                # Count anomalies
                anomaly_count = np.sum(np.array(predictions) == -1)  # -1 indicates anomaly
                anomaly_ratio = anomaly_count / len(new_mape)
                
                # Determine if anomalies are significant
                anomaly_significant = anomaly_ratio > self.config["anomaly_detection_contamination"] * 1.5
                is_significant = is_significant or anomaly_significant
                
                # Store results
                metric_results[metric] = {
                    "status": "success",
                    "is_significant": anomaly_significant,
                    "method": method,
                    "anomaly_count": int(anomaly_count),
                    "anomaly_ratio": float(anomaly_ratio),
                    "anomaly_threshold": float(self.config["anomaly_detection_contamination"]),
                    "sample_counts": {
                        "historical": len(historical_mape),
                        "new": len(new_mape)
                    }
                }
                
                # If we have extreme anomalies, report them
                if anomaly_significant:
                    # Find top 3 anomalies
                    top_anomalies = []
                    for i in range(len(new_mape)):
                        if predictions[i] == -1:  # If it's an anomaly
                            top_anomalies.append({
                                "index": i,
                                "mape": float(new_mape[i]),
                                "score": float(anomaly_scores[i])
                            })
                    
                    # Sort by anomaly score
                    top_anomalies.sort(key=lambda x: x["score"], reverse=True)
                    
                    # Keep top 3
                    metric_results[metric]["top_anomalies"] = top_anomalies[:3]
            
            # Create overall result
            result = {
                "status": "success",
                "is_significant": is_significant,
                "method": "anomaly_detection",
                "anomalous_metrics": metric_results
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {
                "status": "error",
                "message": f"Error in anomaly detection: {e}",
                "is_significant": False
            }
    
    def _extract_mape_values(self, validation_results: List[ValidationResult], metric: str) -> List[float]:
        """
        Extract MAPE values for a specific metric from validation results.
        
        Args:
            validation_results: List of validation results
            metric: Metric name to extract
            
        Returns:
            List of MAPE values
        """
        mape_values = []
        
        for val_result in validation_results:
            if metric in val_result.metrics_comparison:
                if "mape" in val_result.metrics_comparison[metric]:
                    mape = val_result.metrics_comparison[metric]["mape"]
                    if not np.isnan(mape):
                        mape_values.append(mape)
        
        return mape_values
    
    def _extract_metric_vectors(self, validation_results: List[ValidationResult]) -> List[List[float]]:
        """
        Extract metric vectors for multi-dimensional analysis.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            List of metric vectors (each vector contains MAPE values for multiple metrics)
        """
        metric_vectors = []
        
        for val_result in validation_results:
            # Extract MAPE values for each monitored metric
            vector = []
            
            for metric in self.config["metrics_to_monitor"]:
                if metric in val_result.metrics_comparison and "mape" in val_result.metrics_comparison[metric]:
                    mape = val_result.metrics_comparison[metric]["mape"]
                    if not np.isnan(mape):
                        vector.append(mape)
                    else:
                        # Skip this result if any metric has NaN MAPE
                        vector = []
                        break
                else:
                    # Skip this result if any metric is missing
                    vector = []
                    break
            
            if vector and len(vector) == len(self.config["metrics_to_monitor"]):
                metric_vectors.append(vector)
        
        return metric_vectors
    
    def _extract_metrics_by_result(self, validation_results: List[ValidationResult]) -> List[Dict[str, float]]:
        """
        Extract raw metric values by validation result for correlation analysis.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            List of dictionaries mapping metric names to raw values
        """
        metrics_by_result = []
        
        for val_result in validation_results:
            metrics = {}
            
            # Extract raw metric values from simulation_result
            for metric in self.config["metrics_to_monitor"]:
                if metric in val_result.simulation_result.metrics:
                    metrics[f"sim_{metric}"] = val_result.simulation_result.metrics[metric]
                
                # Also extract from hardware_result for correlation analysis
                if metric in val_result.hardware_result.metrics:
                    metrics[f"hw_{metric}"] = val_result.hardware_result.metrics[metric]
            
            # Only include results that have values for all metrics
            if len(metrics) == 2 * len(self.config["metrics_to_monitor"]):
                metrics_by_result.append(metrics)
        
        return metrics_by_result
    
    def _calculate_metric_correlations(self, metrics_by_result: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate correlations between pairs of metrics.
        
        Args:
            metrics_by_result: List of dictionaries with metric values by result
            
        Returns:
            Dictionary mapping metric pairs to correlation coefficients
        """
        correlations = {}
        
        if not metrics_by_result:
            return correlations
        
        # Get list of all metric names
        metric_names = list(metrics_by_result[0].keys())
        
        # Calculate correlation for each pair of metrics
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric1 = metric_names[i]
                metric2 = metric_names[j]
                
                # Skip correlations between simulation and hardware versions of same metric
                if (metric1.startswith("sim_") and metric2.startswith("hw_") and 
                    metric1[4:] == metric2[3:]):
                    continue
                
                # Extract values for both metrics
                values1 = []
                values2 = []
                
                for metrics in metrics_by_result:
                    if metric1 in metrics and metric2 in metrics:
                        val1 = metrics[metric1]
                        val2 = metrics[metric2]
                        
                        if val1 is not None and val2 is not None:
                            values1.append(val1)
                            values2.append(val2)
                
                # Calculate correlation if we have enough values
                if len(values1) >= 5:
                    try:
                        from scipy import stats
                        corr, _ = stats.pearsonr(values1, values2)
                        correlations[f"{metric1}/{metric2}"] = corr
                    except Exception as e:
                        logger.warning(f"Error calculating correlation for {metric1}/{metric2}: {e}")
        
        return correlations