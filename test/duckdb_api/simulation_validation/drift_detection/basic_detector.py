#!/usr/bin/env python3
"""
Basic Drift Detector implementation for the Simulation Accuracy and Validation Framework.

This module provides a concrete implementation of the DriftDetector interface
that detects drift in simulation accuracy over time.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("basic_drift_detector")

# Add parent directories to path for module imports
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import base classes
from data.duckdb.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult,
    DriftDetector
)


class BasicDriftDetector(DriftDetector):
    """
    Basic implementation of a drift detector using statistical methods.
    
    This detector compares historical validation results with new validation
    results to detect significant drift in simulation accuracy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the basic drift detector.
        
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
            "significance_test": "t_test",  # t_test, mann_whitney, or bootstrap
            "significance_level": 0.05,     # p-value threshold for significance
            "min_samples": 5                # Minimum number of samples for drift detection
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Initialize drift thresholds
        self.drift_thresholds = self.config["drift_thresholds"]
        
        # Initialize drift status
        self.current_drift_status = {
            "is_drifting": False,
            "drift_metrics": {},
            "last_check_time": None,
            "significant_metrics": []
        }
    
    def detect_drift(
        self,
        historical_validation_results: List[ValidationResult],
        new_validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Detect drift in simulation accuracy.
        
        Args:
            historical_validation_results: Historical validation results
            new_validation_results: New validation results
            
        Returns:
            Drift detection results with metrics and significance
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
        
        # Calculate MAPE statistics for historical results
        historical_stats = self._calculate_mape_statistics(historical_validation_results)
        
        # Calculate MAPE statistics for new results
        new_stats = self._calculate_mape_statistics(new_validation_results)
        
        # Calculate drift metrics
        drift_metrics = {}
        significant_metrics = []
        
        for metric in self.config["metrics_to_monitor"]:
            if metric in historical_stats and metric in new_stats:
                historical_mape = historical_stats[metric]["mean_mape"]
                new_mape = new_stats[metric]["mean_mape"]
                
                absolute_change = new_mape - historical_mape
                if historical_mape > 0:
                    relative_change = absolute_change / historical_mape * 100.0
                else:
                    relative_change = float('inf') if absolute_change > 0 else float('-inf')
                
                # Check for statistical significance
                significance = self._test_significance(
                    historical_stats[metric]["mape_values"],
                    new_stats[metric]["mape_values"]
                )
                
                is_significant = (
                    abs(absolute_change) >= self.drift_thresholds.get(metric, self.drift_thresholds["overall"]) and
                    significance["is_significant"]
                )
                
                drift_metrics[metric] = {
                    "historical_mape": historical_mape,
                    "new_mape": new_mape,
                    "absolute_change": absolute_change,
                    "relative_change": relative_change,
                    "p_value": significance["p_value"],
                    "is_significant": is_significant
                }
                
                if is_significant:
                    significant_metrics.append(metric)
        
        # Calculate overall drift
        overall_historical_mape = historical_stats["overall"]["mean_mape"]
        overall_new_mape = new_stats["overall"]["mean_mape"]
        
        overall_absolute_change = overall_new_mape - overall_historical_mape
        if overall_historical_mape > 0:
            overall_relative_change = overall_absolute_change / overall_historical_mape * 100.0
        else:
            overall_relative_change = float('inf') if overall_absolute_change > 0 else float('-inf')
        
        # Check for statistical significance
        overall_significance = self._test_significance(
            historical_stats["overall"]["mape_values"],
            new_stats["overall"]["mape_values"]
        )
        
        overall_is_significant = (
            abs(overall_absolute_change) >= self.drift_thresholds["overall"] and
            overall_significance["is_significant"]
        )
        
        drift_metrics["overall"] = {
            "historical_mape": overall_historical_mape,
            "new_mape": overall_new_mape,
            "absolute_change": overall_absolute_change,
            "relative_change": overall_relative_change,
            "p_value": overall_significance["p_value"],
            "is_significant": overall_is_significant
        }
        
        if overall_is_significant:
            significant_metrics.append("overall")
        
        # Get time ranges for historical and new results
        historical_start = min(val.validation_timestamp for val in historical_validation_results)
        historical_end = max(val.validation_timestamp for val in historical_validation_results)
        new_start = min(val.validation_timestamp for val in new_validation_results)
        new_end = max(val.validation_timestamp for val in new_validation_results)
        
        # Prepare drift detection results
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        drift_result = {
            "status": "success",
            "timestamp": timestamp,
            "historical_window_start": historical_start,
            "historical_window_end": historical_end,
            "new_window_start": new_start,
            "new_window_end": new_end,
            "historical_sample_count": len(historical_validation_results),
            "new_sample_count": len(new_validation_results),
            "drift_metrics": drift_metrics,
            "significant_metrics": significant_metrics,
            "is_significant": overall_is_significant or len(significant_metrics) > 0,
            "thresholds_used": self.drift_thresholds
        }
        
        # Update drift status
        self.current_drift_status = {
            "is_drifting": drift_result["is_significant"],
            "drift_metrics": drift_metrics,
            "last_check_time": timestamp,
            "significant_metrics": significant_metrics
        }
        
        return drift_result
    
    def set_drift_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set thresholds for drift detection.
        
        Args:
            thresholds: Dictionary mapping metric names to threshold values
        """
        for metric, threshold in thresholds.items():
            self.drift_thresholds[metric] = threshold
        
        logger.info(f"Updated drift thresholds: {self.drift_thresholds}")
    
    def get_drift_status(self) -> Dict[str, Any]:
        """
        Get the current drift status.
        
        Returns:
            Dictionary with drift status information
        """
        return self.current_drift_status
    
    def _calculate_mape_statistics(self, validation_results: List[ValidationResult]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate MAPE statistics for validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary with MAPE statistics by metric
        """
        statistics = {}
        
        # Collect MAPE values for each metric
        for metric in self.config["metrics_to_monitor"]:
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
                    "count": len(mape_values),
                    "mape_values": mape_values
                }
        
        # Calculate overall MAPE statistics
        all_mape_values = []
        
        for val_result in validation_results:
            result_mape_values = []
            
            for metric, comparison in val_result.metrics_comparison.items():
                if metric in self.config["metrics_to_monitor"] and "mape" in comparison:
                    mape = comparison["mape"]
                    if not np.isnan(mape):
                        result_mape_values.append(mape)
            
            if result_mape_values:
                # Add average MAPE for this validation result
                all_mape_values.append(np.mean(result_mape_values))
        
        if all_mape_values:
            statistics["overall"] = {
                "mean_mape": np.mean(all_mape_values),
                "median_mape": np.median(all_mape_values),
                "min_mape": np.min(all_mape_values),
                "max_mape": np.max(all_mape_values),
                "std_dev_mape": np.std(all_mape_values),
                "count": len(all_mape_values),
                "mape_values": all_mape_values
            }
        
        return statistics
    
    def _test_significance(self, historical_values: List[float], new_values: List[float]) -> Dict[str, Any]:
        """
        Test the statistical significance of differences between historical and new values.
        
        Args:
            historical_values: MAPE values from historical validation results
            new_values: MAPE values from new validation results
            
        Returns:
            Dictionary with significance test results
        """
        test_type = self.config["significance_test"]
        alpha = self.config["significance_level"]
        
        if len(historical_values) < 2 or len(new_values) < 2:
            return {
                "is_significant": False,
                "p_value": 1.0,
                "test_type": "none",
                "message": "Insufficient samples for significance testing"
            }
        
        try:
            if test_type == "t_test":
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(historical_values, new_values, equal_var=False)
                
                return {
                    "is_significant": p_value < alpha,
                    "p_value": p_value,
                    "test_type": "t_test",
                    "statistic": t_stat
                }
                
            elif test_type == "mann_whitney":
                from scipy import stats
                u_stat, p_value = stats.mannwhitneyu(historical_values, new_values)
                
                return {
                    "is_significant": p_value < alpha,
                    "p_value": p_value,
                    "test_type": "mann_whitney",
                    "statistic": u_stat
                }
                
            elif test_type == "bootstrap":
                # Implement bootstrap test for significance
                # This is a simple implementation; a more sophisticated version would use a proper bootstrap library
                num_bootstrap_samples = 1000
                combined = historical_values + new_values
                n_historical = len(historical_values)
                
                # Calculate observed difference in means
                observed_diff = np.mean(new_values) - np.mean(historical_values)
                
                # Perform bootstrap resampling
                bootstrap_diffs = []
                for _ in range(num_bootstrap_samples):
                    np.random.shuffle(combined)
                    bootstrap_historical = combined[:n_historical]
                    bootstrap_new = combined[n_historical:]
                    bootstrap_diff = np.mean(bootstrap_new) - np.mean(bootstrap_historical)
                    bootstrap_diffs.append(bootstrap_diff)
                
                # Calculate p-value (two-tailed)
                p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
                
                return {
                    "is_significant": p_value < alpha,
                    "p_value": p_value,
                    "test_type": "bootstrap",
                    "statistic": observed_diff
                }
                
            else:
                logger.warning(f"Unknown significance test type: {test_type}")
                return {
                    "is_significant": False,
                    "p_value": 1.0,
                    "test_type": "none",
                    "message": f"Unknown test type: {test_type}"
                }
                
        except Exception as e:
            logger.error(f"Error in significance testing: {e}")
            return {
                "is_significant": False,
                "p_value": 1.0,
                "test_type": test_type,
                "message": f"Error in significance testing: {e}"
            }