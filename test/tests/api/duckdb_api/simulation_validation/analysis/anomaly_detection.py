#!/usr/bin/env python3
"""
Anomaly Detection for the Simulation Accuracy and Validation Framework.

This module provides advanced anomaly detection capabilities for identifying unusual
or unexpected validation results that may indicate issues with simulations or hardware
measurements. The module includes:
- Statistical anomaly detection based on distributions
- Proximity-based anomaly detection using clustering
- Deviation-based anomaly detection for time series
- Classification-based anomaly detection
- Ensemble methods combining multiple detection techniques
"""

import logging
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analysis.anomaly_detection")

# Import base class
from data.duckdb.simulation_validation.analysis.base import AnalysisMethod
from data.duckdb.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

class AnomalyDetection(AnalysisMethod):
    """
    Anomaly detection for simulation validation results.
    
    This class extends the basic AnalysisMethod to provide advanced anomaly
    detection techniques for identifying unusual validation results:
    - Statistical anomaly detection based on distributions
    - Proximity-based anomaly detection using clustering
    - Deviation-based anomaly detection for time series
    - Classification-based anomaly detection
    - Ensemble methods combining multiple detection techniques
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the anomaly detection method.
        
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
            
            # Statistical anomaly detection
            "statistical_detection": {
                "enabled": True,
                "methods": ["z_score", "modified_z_score", "iqr"],
                "z_score_threshold": 3.0,  # Standard deviations for Z-score
                "modified_z_score_threshold": 3.5,  # Threshold for modified Z-score
                "iqr_scale": 1.5,  # Scale factor for IQR-based detection
                "min_samples": 5  # Minimum samples needed for statistical detection
            },
            
            # Proximity-based anomaly detection
            "proximity_detection": {
                "enabled": True,
                "methods": ["dbscan", "isolation_forest", "local_outlier_factor"],
                "dbscan_eps": 0.5,  # Maximum distance for DBSCAN
                "dbscan_min_samples": 3,  # Minimum samples for core point in DBSCAN
                "isolation_forest_contamination": 0.1,  # Expected proportion of anomalies
                "lof_n_neighbors": 5,  # Number of neighbors for LOF
                "lof_contamination": 0.1,  # Expected proportion of anomalies for LOF
                "min_samples": 5  # Minimum samples needed for proximity detection
            },
            
            # Deviation-based anomaly detection for time series
            "deviation_detection": {
                "enabled": True,
                "methods": ["moving_average", "ewma", "seasonal_decomposition"],
                "window_size": 5,  # Window size for moving average
                "alpha": 0.2,  # Smoothing factor for EWMA
                "threshold_scale": 2.0,  # Scale factor for thresholds
                "min_samples": 8  # Minimum samples needed for deviation detection
            },
            
            # Ensemble anomaly detection
            "ensemble_detection": {
                "enabled": True,
                "voting_threshold": 0.5,  # Proportion of methods that must agree
                "min_methods": 2  # Minimum methods required for ensemble detection
            },
            
            # Anomaly context analysis
            "context_analysis": {
                "enabled": True,
                "correlation_threshold": 0.7,  # Threshold for strong correlations
                "cluster_analysis": True  # Whether to analyze anomaly clusters
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
        Perform anomaly detection on validation results.
        
        Args:
            validation_results: List of validation results to analyze
            
        Returns:
            Dictionary containing anomaly detection results and insights
        """
        # Check requirements
        meets_req, error_msg = self.check_requirements(validation_results)
        if not meets_req:
            logger.warning(f"Requirements not met for anomaly detection: {error_msg}")
            return {"status": "error", "message": error_msg}
        
        # Initialize results dictionary
        analysis_results = {
            "status": "success",
            "timestamp": datetime.datetime.now().isoformat(),
            "num_validation_results": len(validation_results),
            "metrics_analyzed": self.config["metrics_to_analyze"],
            "analysis_methods": {},
            "anomalies_detected": [],
            "anomaly_insights": {
                "summary": {},
                "key_findings": [],
                "recommendations": []
            }
        }
        
        # Extract features and metadata for analysis
        features, feature_names, metadata = self._extract_features(validation_results)
        
        if features.size == 0 or len(feature_names) == 0:
            return {
                "status": "error", 
                "message": "Failed to extract features for anomaly detection"
            }
        
        # Normalize features
        normalized_features = self._normalize_features(features)
        
        # Apply statistical anomaly detection if enabled
        if self.config["statistical_detection"]["enabled"]:
            try:
                # Check if we have enough data points
                min_samples = self.config["statistical_detection"]["min_samples"]
                if features.shape[0] >= min_samples:
                    stat_anomalies = self._detect_statistical_anomalies(
                        features, feature_names, metadata)
                    analysis_results["analysis_methods"]["statistical_detection"] = stat_anomalies
                    
                    # Add detected anomalies to the overall list
                    if "anomalies" in stat_anomalies:
                        for anomaly in stat_anomalies["anomalies"]:
                            analysis_results["anomalies_detected"].append(anomaly)
                else:
                    analysis_results["analysis_methods"]["statistical_detection"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for statistical detection. "
                                 f"Required: {min_samples}, Provided: {features.shape[0]}"
                    }
            except Exception as e:
                logger.error(f"Error in statistical anomaly detection: {e}")
                analysis_results["analysis_methods"]["statistical_detection"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Apply proximity-based anomaly detection if enabled
        if self.config["proximity_detection"]["enabled"]:
            try:
                # Check if we have enough data points
                min_samples = self.config["proximity_detection"]["min_samples"]
                if features.shape[0] >= min_samples:
                    prox_anomalies = self._detect_proximity_anomalies(
                        normalized_features, feature_names, metadata)
                    analysis_results["analysis_methods"]["proximity_detection"] = prox_anomalies
                    
                    # Add detected anomalies to the overall list
                    if "anomalies" in prox_anomalies:
                        for anomaly in prox_anomalies["anomalies"]:
                            analysis_results["anomalies_detected"].append(anomaly)
                else:
                    analysis_results["analysis_methods"]["proximity_detection"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for proximity detection. "
                                 f"Required: {min_samples}, Provided: {features.shape[0]}"
                    }
            except Exception as e:
                logger.error(f"Error in proximity-based anomaly detection: {e}")
                analysis_results["analysis_methods"]["proximity_detection"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Apply deviation-based anomaly detection for time series if enabled
        if self.config["deviation_detection"]["enabled"]:
            try:
                # Check if we have enough data points and timestamps
                min_samples = self.config["deviation_detection"]["min_samples"]
                has_timestamps = all(hasattr(result, "validation_timestamp") 
                                    for result in validation_results)
                
                if features.shape[0] >= min_samples and has_timestamps:
                    dev_anomalies = self._detect_deviation_anomalies(
                        validation_results, features, feature_names, metadata)
                    analysis_results["analysis_methods"]["deviation_detection"] = dev_anomalies
                    
                    # Add detected anomalies to the overall list
                    if "anomalies" in dev_anomalies:
                        for anomaly in dev_anomalies["anomalies"]:
                            analysis_results["anomalies_detected"].append(anomaly)
                else:
                    skip_reason = ("Insufficient data points" if features.shape[0] < min_samples
                                  else "Missing timestamps")
                    analysis_results["analysis_methods"]["deviation_detection"] = {
                        "status": "skipped",
                        "message": f"{skip_reason} for deviation detection. "
                                 f"Required: {min_samples} samples with timestamps"
                    }
            except Exception as e:
                logger.error(f"Error in deviation-based anomaly detection: {e}")
                analysis_results["analysis_methods"]["deviation_detection"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Apply ensemble anomaly detection if enabled
        if self.config["ensemble_detection"]["enabled"]:
            try:
                # Count how many methods detected each result as anomalous
                anomaly_counts = self._ensemble_detection(
                    analysis_results["analysis_methods"], validation_results)
                
                # Apply voting threshold
                voting_threshold = self.config["ensemble_detection"]["voting_threshold"]
                min_methods = self.config["ensemble_detection"]["min_methods"]
                
                ensemble_anomalies = {
                    "method": "ensemble",
                    "voting_threshold": voting_threshold,
                    "min_methods": min_methods,
                    "anomalies": []
                }
                
                for idx, count_data in anomaly_counts.items():
                    methods_count = count_data["count"]
                    total_methods = count_data["total"]
                    
                    if methods_count >= min_methods and methods_count / total_methods >= voting_threshold:
                        # Create ensemble anomaly
                        anomaly = {
                            "index": idx,
                            "confidence": methods_count / total_methods,
                            "voting_methods": count_data["methods"],
                            "metadata": metadata[idx] if idx < len(metadata) else {},
                            "description": f"Anomaly detected by {methods_count}/{total_methods} methods"
                        }
                        
                        ensemble_anomalies["anomalies"].append(anomaly)
                
                analysis_results["analysis_methods"]["ensemble_detection"] = ensemble_anomalies
                
                # Update anomalies_detected with ensemble results
                analysis_results["anomalies_detected"] = ensemble_anomalies["anomalies"]
                
            except Exception as e:
                logger.error(f"Error in ensemble anomaly detection: {e}")
                analysis_results["analysis_methods"]["ensemble_detection"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Analyze anomaly context if enabled and anomalies detected
        if (self.config["context_analysis"]["enabled"] and 
            analysis_results["anomalies_detected"]):
            try:
                context_analysis = self._analyze_anomaly_context(
                    analysis_results["anomalies_detected"], 
                    validation_results, 
                    features, 
                    feature_names,
                    metadata
                )
                
                analysis_results["anomaly_insights"]["context_analysis"] = context_analysis
                
                # Add context information to anomalies
                if "contexts" in context_analysis:
                    for anomaly in analysis_results["anomalies_detected"]:
                        idx = anomaly.get("index")
                        if idx is not None and idx in context_analysis["contexts"]:
                            anomaly["context"] = context_analysis["contexts"][idx]
                
            except Exception as e:
                logger.error(f"Error in anomaly context analysis: {e}")
                analysis_results["anomaly_insights"]["context_analysis"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Generate anomaly summary
        analysis_results["anomaly_insights"]["summary"] = self._generate_anomaly_summary(
            analysis_results["anomalies_detected"], validation_results)
        
        # Generate key findings
        analysis_results["anomaly_insights"]["key_findings"] = self._generate_key_findings(
            analysis_results["anomalies_detected"], 
            analysis_results["anomaly_insights"].get("context_analysis", {}),
            analysis_results["anomaly_insights"]["summary"]
        )
        
        # Generate recommendations
        analysis_results["anomaly_insights"]["recommendations"] = self._generate_recommendations(
            analysis_results["anomalies_detected"],
            analysis_results["anomaly_insights"]
        )
        
        return analysis_results
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the capabilities of the anomaly detection.
        
        Returns:
            Dictionary describing the capabilities
        """
        return {
            "name": "Anomaly Detection",
            "description": "Identifies unusual or unexpected validation results using multiple techniques",
            "methods": [
                {
                    "name": "Statistical Anomaly Detection",
                    "description": "Identifies anomalies based on statistical distributions",
                    "enabled": self.config["statistical_detection"]["enabled"],
                    "techniques": self.config["statistical_detection"]["methods"]
                },
                {
                    "name": "Proximity-based Anomaly Detection",
                    "description": "Identifies anomalies using distance and density measures",
                    "enabled": self.config["proximity_detection"]["enabled"],
                    "techniques": self.config["proximity_detection"]["methods"]
                },
                {
                    "name": "Deviation-based Anomaly Detection",
                    "description": "Identifies anomalies in time series data",
                    "enabled": self.config["deviation_detection"]["enabled"],
                    "techniques": self.config["deviation_detection"]["methods"]
                },
                {
                    "name": "Ensemble Anomaly Detection",
                    "description": "Combines multiple detection methods for robust results",
                    "enabled": self.config["ensemble_detection"]["enabled"]
                },
                {
                    "name": "Context Analysis",
                    "description": "Analyzes the context of detected anomalies",
                    "enabled": self.config["context_analysis"]["enabled"]
                }
            ],
            "output_format": {
                "anomalies_detected": "List of detected anomalies with confidence scores",
                "anomaly_insights": "Summary, key findings, and recommendations"
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
            "min_validation_results": 3,
            "required_metrics": self.config["metrics_to_analyze"],
            "optimal_validation_results": 10,
            "statistical_requirements": {
                "min_samples": self.config["statistical_detection"]["min_samples"]
            },
            "proximity_requirements": {
                "min_samples": self.config["proximity_detection"]["min_samples"]
            },
            "deviation_requirements": {
                "min_samples": self.config["deviation_detection"]["min_samples"],
                "time_series_required": self.config["deviation_detection"]["enabled"]
            }
        }
        
        return requirements
    
    def _extract_features(
        self,
        validation_results: List[ValidationResult]
    ) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """
        Extract features from validation results for anomaly detection.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Tuple containing:
                - Feature matrix (samples x features)
                - List of feature names
                - List of metadata dictionaries for each sample
        """
        # Define metrics to analyze
        metrics_to_analyze = self.config["metrics_to_analyze"]
        
        # Initialize lists for features, feature names, and metadata
        features_list = []
        feature_names = []
        metadata = []
        
        # Extract features from validation results
        for i, result in enumerate(validation_results):
            row = []
            meta = {
                "index": i,
                "hardware_id": result.hardware_result.hardware_id,
                "model_id": result.hardware_result.model_id,
                "batch_size": result.hardware_result.batch_size,
                "precision": result.hardware_result.precision,
                "timestamp": result.validation_timestamp
            }
            
            # Extract error metrics for each performance metric
            for metric in metrics_to_analyze:
                if metric in result.metrics_comparison:
                    # Extract error metrics from comparison
                    comparison = result.metrics_comparison[metric]
                    
                    # Add feature names the first time
                    if not feature_names:
                        for error_metric in ["mape", "absolute_error", "relative_error"]:
                            if error_metric in comparison:
                                feature_names.append(f"{metric}_{error_metric}")
                    
                    # Add error metrics to features
                    for error_metric in ["mape", "absolute_error", "relative_error"]:
                        if error_metric in comparison:
                            # Skip NaN values
                            if comparison[error_metric] is None or np.isnan(comparison[error_metric]):
                                row.append(0.0)  # Use 0 as default for missing values
                            else:
                                row.append(comparison[error_metric])
            
            # Add additional metrics from enhanced analysis
            if hasattr(result, "additional_metrics") and result.additional_metrics:
                additional_metrics = result.additional_metrics
                
                # Add overall metrics if available
                for key in ["overall_mape", "bias_score", "precision_score", "confidence_score"]:
                    if key in additional_metrics:
                        if not feature_names or key not in feature_names:
                            feature_names.append(key)
                        
                        value = additional_metrics[key]
                        if value is None or np.isnan(value):
                            row.append(0.0)
                        else:
                            row.append(value)
            
            # Only add row if we have features
            if row and len(row) > 0:
                # Make sure all rows have the same number of features
                if not features_list:
                    # First row defines the number of features
                    features_list.append(row)
                    metadata.append(meta)
                elif len(row) == len(features_list[0]):
                    features_list.append(row)
                    metadata.append(meta)
                else:
                    logger.warning(f"Skipping result with mismatched feature count: {len(row)} vs {len(features_list[0])}")
        
        # Convert to numpy array
        if features_list:
            features = np.array(features_list)
            
            # Ensure feature_names matches feature count
            if len(feature_names) != features.shape[1]:
                # Adjust feature names if needed
                if len(feature_names) < features.shape[1]:
                    # Add generic names for missing features
                    for i in range(len(feature_names), features.shape[1]):
                        feature_names.append(f"feature_{i}")
                else:
                    # Truncate feature names to match feature count
                    feature_names = feature_names[:features.shape[1]]
            
            return features, feature_names, metadata
        else:
            return np.array([]), [], []
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features for anomaly detection.
        
        Args:
            features: Feature matrix (samples x features)
            
        Returns:
            Normalized feature matrix
        """
        if features.size == 0:
            return features
        
        # Replace NaN values with 0
        features = np.nan_to_num(features, nan=0.0)
        
        # Normalize each feature to [0, 1] range
        normalized = np.zeros_like(features, dtype=float)
        
        for j in range(features.shape[1]):
            col = features[:, j]
            col_min = np.min(col)
            col_max = np.max(col)
            
            if col_max > col_min:
                normalized[:, j] = (col - col_min) / (col_max - col_min)
            else:
                normalized[:, j] = 0.0  # Default when all values are the same
        
        return normalized
    
    def _detect_statistical_anomalies(
        self,
        features: np.ndarray,
        feature_names: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods.
        
        Args:
            features: Feature matrix (samples x features)
            feature_names: List of feature names
            metadata: List of metadata dictionaries for each sample
            
        Returns:
            Dictionary with anomaly detection results
        """
        results = {
            "methods_used": [],
            "anomalies": []
        }
        
        # Get methods to use
        methods = self.config["statistical_detection"]["methods"]
        
        # Apply Z-score method if enabled
        if "z_score" in methods:
            try:
                # Calculate Z-scores for each feature
                z_scores = np.zeros_like(features, dtype=float)
                
                for j in range(features.shape[1]):
                    col = features[:, j]
                    # Skip features with no variation
                    if np.std(col) > 0:
                        z_scores[:, j] = (col - np.mean(col)) / np.std(col)
                
                # Get threshold
                threshold = self.config["statistical_detection"]["z_score_threshold"]
                
                # Find anomalies (samples with absolute Z-score > threshold)
                anomaly_flags = np.abs(z_scores) > threshold
                
                # Aggregate anomalies across features
                for i in range(features.shape[0]):
                    # Check if sample has any feature with Z-score > threshold
                    anomalous_features = np.where(anomaly_flags[i])[0]
                    
                    if len(anomalous_features) > 0:
                        # Create anomaly details
                        anomaly_feature_details = []
                        for j in anomalous_features:
                            feature_name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
                            z_score_val = z_scores[i, j]
                            
                            anomaly_feature_details.append({
                                "feature": feature_name,
                                "z_score": float(z_score_val),
                                "value": float(features[i, j])
                            })
                        
                        # Sort by absolute Z-score
                        anomaly_feature_details.sort(
                            key=lambda x: abs(x["z_score"]), reverse=True)
                        
                        # Create anomaly record
                        max_z_score = max(abs(z_scores[i, j]) for j in anomalous_features)
                        confidence = min(1.0, (max_z_score - threshold) / threshold)
                        
                        anomaly = {
                            "method": "z_score",
                            "index": i,
                            "confidence": float(confidence),
                            "features": anomaly_feature_details,
                            "metadata": metadata[i] if i < len(metadata) else {},
                            "description": f"Z-score anomaly with {len(anomalous_features)} " 
                                          f"anomalous features (max |Z| = {max_z_score:.2f})"
                        }
                        
                        results["anomalies"].append(anomaly)
                
                # Add method info
                results["methods_used"].append({
                    "name": "z_score",
                    "threshold": threshold,
                    "anomalies_found": sum(1 for a in results["anomalies"] if a["method"] == "z_score")
                })
                
            except Exception as e:
                logger.warning(f"Error in Z-score anomaly detection: {e}")
        
        # Apply modified Z-score method if enabled
        if "modified_z_score" in methods:
            try:
                # Calculate modified Z-scores for each feature
                mod_z_scores = np.zeros_like(features, dtype=float)
                
                for j in range(features.shape[1]):
                    col = features[:, j]
                    # Calculate median and median absolute deviation (MAD)
                    median = np.median(col)
                    mad = np.median(np.abs(col - median))
                    
                    # Skip features with no variation
                    if mad > 0:
                        # Modified Z-score = 0.6745 * (x - median) / MAD
                        mod_z_scores[:, j] = 0.6745 * (col - median) / mad
                
                # Get threshold
                threshold = self.config["statistical_detection"]["modified_z_score_threshold"]
                
                # Find anomalies (samples with absolute modified Z-score > threshold)
                anomaly_flags = np.abs(mod_z_scores) > threshold
                
                # Aggregate anomalies across features
                for i in range(features.shape[0]):
                    # Check if sample has any feature with modified Z-score > threshold
                    anomalous_features = np.where(anomaly_flags[i])[0]
                    
                    if len(anomalous_features) > 0:
                        # Create anomaly details
                        anomaly_feature_details = []
                        for j in anomalous_features:
                            feature_name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
                            mod_z_score_val = mod_z_scores[i, j]
                            
                            anomaly_feature_details.append({
                                "feature": feature_name,
                                "modified_z_score": float(mod_z_score_val),
                                "value": float(features[i, j])
                            })
                        
                        # Sort by absolute modified Z-score
                        anomaly_feature_details.sort(
                            key=lambda x: abs(x["modified_z_score"]), reverse=True)
                        
                        # Create anomaly record
                        max_mod_z_score = max(abs(mod_z_scores[i, j]) for j in anomalous_features)
                        confidence = min(1.0, (max_mod_z_score - threshold) / threshold)
                        
                        anomaly = {
                            "method": "modified_z_score",
                            "index": i,
                            "confidence": float(confidence),
                            "features": anomaly_feature_details,
                            "metadata": metadata[i] if i < len(metadata) else {},
                            "description": f"Modified Z-score anomaly with {len(anomalous_features)} " 
                                          f"anomalous features (max |Z'| = {max_mod_z_score:.2f})"
                        }
                        
                        results["anomalies"].append(anomaly)
                
                # Add method info
                results["methods_used"].append({
                    "name": "modified_z_score",
                    "threshold": threshold,
                    "anomalies_found": sum(1 for a in results["anomalies"] if a["method"] == "modified_z_score")
                })
                
            except Exception as e:
                logger.warning(f"Error in modified Z-score anomaly detection: {e}")
        
        # Apply IQR method if enabled
        if "iqr" in methods:
            try:
                # Calculate IQR for each feature
                iqr_values = np.zeros_like(features, dtype=float)
                iqr_thresholds_lower = np.zeros(features.shape[1], dtype=float)
                iqr_thresholds_upper = np.zeros(features.shape[1], dtype=float)
                
                for j in range(features.shape[1]):
                    col = features[:, j]
                    q1 = np.percentile(col, 25)
                    q3 = np.percentile(col, 75)
                    iqr = q3 - q1
                    
                    # Get IQR scale factor
                    scale = self.config["statistical_detection"]["iqr_scale"]
                    
                    # Calculate thresholds
                    lower_threshold = q1 - scale * iqr
                    upper_threshold = q3 + scale * iqr
                    
                    # Store thresholds
                    iqr_thresholds_lower[j] = lower_threshold
                    iqr_thresholds_upper[j] = upper_threshold
                    
                    # Calculate how many IQRs away from the median
                    median = np.median(col)
                    if iqr > 0:
                        iqr_values[:, j] = (col - median) / iqr
                
                # Find anomalies (samples outside IQR thresholds)
                anomaly_flags_lower = features < iqr_thresholds_lower
                anomaly_flags_upper = features > iqr_thresholds_upper
                anomaly_flags = np.logical_or(anomaly_flags_lower, anomaly_flags_upper)
                
                # Aggregate anomalies across features
                for i in range(features.shape[0]):
                    # Check if sample has any feature outside IQR thresholds
                    anomalous_features = np.where(anomaly_flags[i])[0]
                    
                    if len(anomalous_features) > 0:
                        # Create anomaly details
                        anomaly_feature_details = []
                        for j in anomalous_features:
                            feature_name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
                            value = features[i, j]
                            iqr_value = iqr_values[i, j]
                            
                            # Determine if below or above threshold
                            if value < iqr_thresholds_lower[j]:
                                direction = "below"
                                threshold = iqr_thresholds_lower[j]
                            else:
                                direction = "above"
                                threshold = iqr_thresholds_upper[j]
                            
                            anomaly_feature_details.append({
                                "feature": feature_name,
                                "value": float(value),
                                "iqr_distance": float(iqr_value),
                                "direction": direction,
                                "threshold": float(threshold)
                            })
                        
                        # Sort by absolute IQR distance
                        anomaly_feature_details.sort(
                            key=lambda x: abs(x["iqr_distance"]), reverse=True)
                        
                        # Create anomaly record
                        max_iqr_dist = max(abs(iqr_values[i, j]) for j in anomalous_features)
                        confidence = min(1.0, max_iqr_dist / 5.0)  # Normalize confidence
                        
                        anomaly = {
                            "method": "iqr",
                            "index": i,
                            "confidence": float(confidence),
                            "features": anomaly_feature_details,
                            "metadata": metadata[i] if i < len(metadata) else {},
                            "description": f"IQR-based anomaly with {len(anomalous_features)} " 
                                          f"anomalous features (max IQR distance = {max_iqr_dist:.2f})"
                        }
                        
                        results["anomalies"].append(anomaly)
                
                # Add method info
                results["methods_used"].append({
                    "name": "iqr",
                    "scale": self.config["statistical_detection"]["iqr_scale"],
                    "anomalies_found": sum(1 for a in results["anomalies"] if a["method"] == "iqr")
                })
                
            except Exception as e:
                logger.warning(f"Error in IQR anomaly detection: {e}")
        
        return results
    
    def _detect_proximity_anomalies(
        self,
        features: np.ndarray,
        feature_names: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect anomalies using proximity-based methods.
        
        Args:
            features: Normalized feature matrix (samples x features)
            feature_names: List of feature names
            metadata: List of metadata dictionaries for each sample
            
        Returns:
            Dictionary with anomaly detection results
        """
        results = {
            "methods_used": [],
            "anomalies": []
        }
        
        # Get methods to use
        methods = self.config["proximity_detection"]["methods"]
        
        # Apply DBSCAN if enabled
        if "dbscan" in methods:
            try:
                from sklearn.cluster import DBSCAN
                
                # Get parameters
                eps = self.config["proximity_detection"]["dbscan_eps"]
                min_samples = self.config["proximity_detection"]["dbscan_min_samples"]
                
                # Apply DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(features)
                
                # Anomalies are labeled as -1
                anomaly_indices = np.where(labels == -1)[0]
                
                # Calculate distances to nearest core point for confidence score
                core_sample_indices = dbscan.core_sample_indices_
                
                if len(core_sample_indices) > 0 and len(anomaly_indices) > 0:
                    # Calculate distances from anomalies to nearest core points
                    from sklearn.metrics import pairwise_distances
                    
                    # Get core samples
                    core_samples = features[core_sample_indices]
                    
                    # Calculate distances from each anomaly to each core sample
                    distances = pairwise_distances(
                        features[anomaly_indices], core_samples, metric='euclidean')
                    
                    # Find minimum distance for each anomaly
                    min_distances = np.min(distances, axis=1)
                    
                    # Normalize distances to confidence scores
                    max_dist = np.max(min_distances) if min_distances.size > 0 else 1.0
                    confidence_scores = np.minimum(min_distances / max_dist, np.ones_like(min_distances))
                    
                    # Create anomaly records
                    for i, idx in enumerate(anomaly_indices):
                        anomaly = {
                            "method": "dbscan",
                            "index": int(idx),
                            "confidence": float(confidence_scores[i]),
                            "distance": float(min_distances[i]) if i < len(min_distances) else None,
                            "metadata": metadata[idx] if idx < len(metadata) else {},
                            "description": f"DBSCAN identified this point as an outlier "
                                        f"(distance to nearest core point: {min_distances[i]:.2f})"
                        }
                        
                        results["anomalies"].append(anomaly)
                
                # Add method info
                results["methods_used"].append({
                    "name": "dbscan",
                    "eps": eps,
                    "min_samples": min_samples,
                    "anomalies_found": len(anomaly_indices)
                })
                
            except Exception as e:
                logger.warning(f"Error in DBSCAN anomaly detection: {e}")
        
        # Apply Isolation Forest if enabled
        if "isolation_forest" in methods:
            try:
                from sklearn.ensemble import IsolationForest
                
                # Get parameters
                contamination = self.config["proximity_detection"]["isolation_forest_contamination"]
                
                # Apply Isolation Forest
                iso_forest = IsolationForest(
                    contamination=contamination, random_state=42)
                iso_forest.fit(features)
                
                # Get anomaly scores (-1 for anomalies, 1 for normal)
                y_pred = iso_forest.predict(features)
                
                # Get decision scores (negative values are anomalies, more negative = more anomalous)
                decision_scores = iso_forest.decision_function(features)
                
                # Anomalies are labeled as -1
                anomaly_indices = np.where(y_pred == -1)[0]
                
                # Create anomaly records
                for idx in anomaly_indices:
                    # Convert decision score to confidence (more negative = higher confidence)
                    # decision_scores are centered around 0, with most values in [-0.5, 0.5]
                    # Convert to [0, 1] range where 1 is highest confidence
                    score = decision_scores[idx]
                    confidence = min(1.0, max(0.0, -score * 2))  # Scale and clamp to [0, 1]
                    
                    anomaly = {
                        "method": "isolation_forest",
                        "index": int(idx),
                        "confidence": float(confidence),
                        "decision_score": float(score),
                        "metadata": metadata[idx] if idx < len(metadata) else {},
                        "description": f"Isolation Forest identified this point as an outlier "
                                      f"(decision score: {score:.2f})"
                    }
                    
                    results["anomalies"].append(anomaly)
                
                # Add method info
                results["methods_used"].append({
                    "name": "isolation_forest",
                    "contamination": contamination,
                    "anomalies_found": len(anomaly_indices)
                })
                
            except Exception as e:
                logger.warning(f"Error in Isolation Forest anomaly detection: {e}")
        
        # Apply Local Outlier Factor if enabled
        if "local_outlier_factor" in methods:
            try:
                from sklearn.neighbors import LocalOutlierFactor
                
                # Get parameters
                n_neighbors = self.config["proximity_detection"]["lof_n_neighbors"]
                contamination = self.config["proximity_detection"]["lof_contamination"]
                
                # Apply Local Outlier Factor
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors, 
                    contamination=contamination)
                y_pred = lof.fit_predict(features)
                
                # Get negative outlier factor (higher value = more anomalous)
                # Note: LOF doesn't expose decision_function, but provides negative_outlier_factor_
                # which is only available after fitting
                neg_outlier_factors = lof.negative_outlier_factor_
                
                # Anomalies are labeled as -1
                anomaly_indices = np.where(y_pred == -1)[0]
                
                # Create anomaly records
                for idx in anomaly_indices:
                    # Convert outlier factor to confidence (more negative = higher confidence)
                    # Usually in range [-2, -1] for outliers, where -2 is more anomalous
                    factor = neg_outlier_factors[idx]
                    # Scale to [0, 1] where 1 is highest confidence
                    confidence = min(1.0, max(0.0, (-factor - 1) / 1.0))
                    
                    anomaly = {
                        "method": "local_outlier_factor",
                        "index": int(idx),
                        "confidence": float(confidence),
                        "outlier_factor": float(factor),
                        "metadata": metadata[idx] if idx < len(metadata) else {},
                        "description": f"Local Outlier Factor identified this point as an outlier "
                                      f"(negative outlier factor: {factor:.2f})"
                    }
                    
                    results["anomalies"].append(anomaly)
                
                # Add method info
                results["methods_used"].append({
                    "name": "local_outlier_factor",
                    "n_neighbors": n_neighbors,
                    "contamination": contamination,
                    "anomalies_found": len(anomaly_indices)
                })
                
            except Exception as e:
                logger.warning(f"Error in Local Outlier Factor anomaly detection: {e}")
        
        return results
    
    def _detect_deviation_anomalies(
        self,
        validation_results: List[ValidationResult],
        features: np.ndarray,
        feature_names: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect anomalies based on deviations in time series.
        
        Args:
            validation_results: List of validation results
            features: Feature matrix (samples x features)
            feature_names: List of feature names
            metadata: List of metadata dictionaries for each sample
            
        Returns:
            Dictionary with anomaly detection results
        """
        results = {
            "methods_used": [],
            "anomalies": []
        }
        
        # Check if we have timestamps
        if not all(hasattr(result, "validation_timestamp") for result in validation_results):
            return {
                "status": "skipped",
                "message": "Missing timestamps for deviation detection"
            }
        
        # Sort validation results by timestamp
        try:
            # Convert timestamps to datetime objects
            timestamps = []
            for result in validation_results:
                if isinstance(result.validation_timestamp, str):
                    timestamps.append(datetime.datetime.fromisoformat(result.validation_timestamp))
                else:
                    timestamps.append(result.validation_timestamp)
            
            # Create sorted indices
            sorted_indices = np.argsort(timestamps)
            
            # Sort features and metadata
            sorted_features = features[sorted_indices]
            sorted_metadata = [metadata[i] for i in sorted_indices]
            
            # Store original indices for reference
            original_indices = sorted_indices
            
        except Exception as e:
            logger.warning(f"Error sorting by timestamp: {e}")
            sorted_features = features
            sorted_metadata = metadata
            original_indices = np.arange(len(features))
        
        # Get methods to use
        methods = self.config["deviation_detection"]["methods"]
        
        # Apply moving average method if enabled
        if "moving_average" in methods:
            try:
                # Get parameters
                window_size = self.config["deviation_detection"]["window_size"]
                threshold_scale = self.config["deviation_detection"]["threshold_scale"]
                
                # Skip if not enough data points
                if sorted_features.shape[0] < window_size + 1:
                    logger.warning(f"Insufficient data points for moving average: "
                                  f"{sorted_features.shape[0]} < {window_size + 1}")
                else:
                    # Calculate moving average for each feature
                    anomalies = []
                    
                    for j in range(sorted_features.shape[1]):
                        # Get feature values
                        values = sorted_features[:, j]
                        
                        # Calculate moving average
                        ma_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                        
                        # Calculate residuals for points where MA is available
                        residuals = np.zeros_like(values)
                        residuals[window_size-1:window_size-1+len(ma_values)] = values[window_size-1:window_size-1+len(ma_values)] - ma_values
                        
                        # Calculate standard deviation of residuals
                        residual_std = np.std(residuals[~np.isnan(residuals)])
                        
                        # Set threshold for anomalies
                        threshold = threshold_scale * residual_std
                        
                        # Detect anomalies (residuals > threshold)
                        anomaly_flags = np.abs(residuals) > threshold
                        anomaly_indices = np.where(anomaly_flags)[0]
                        
                        for idx in anomaly_indices:
                            # Skip NaN values
                            if np.isnan(residuals[idx]):
                                continue
                            
                            # Get original index
                            orig_idx = original_indices[idx]
                            
                            # Calculate confidence score
                            deviation = abs(residuals[idx])
                            confidence = min(1.0, deviation / (2 * threshold))
                            
                            # Get feature name
                            feature_name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
                            
                            # Create anomaly record
                            anomaly = {
                                "method": "moving_average",
                                "index": int(orig_idx),
                                "feature_index": j,
                                "feature_name": feature_name,
                                "value": float(values[idx]),
                                "expected_value": float(values[idx] - residuals[idx]),
                                "deviation": float(residuals[idx]),
                                "threshold": float(threshold),
                                "confidence": float(confidence),
                                "metadata": metadata[orig_idx] if orig_idx < len(metadata) else {},
                                "description": f"Moving average anomaly in {feature_name} "
                                              f"(deviation: {residuals[idx]:.2f}, threshold: {threshold:.2f})"
                            }
                            
                            anomalies.append(anomaly)
                    
                    # Add anomalies to results
                    results["anomalies"].extend(anomalies)
                    
                    # Add method info
                    results["methods_used"].append({
                        "name": "moving_average",
                        "window_size": window_size,
                        "threshold_scale": threshold_scale,
                        "anomalies_found": len(anomalies)
                    })
                    
            except Exception as e:
                logger.warning(f"Error in moving average anomaly detection: {e}")
        
        # Apply exponentially weighted moving average method if enabled
        if "ewma" in methods:
            try:
                # Get parameters
                alpha = self.config["deviation_detection"]["alpha"]
                threshold_scale = self.config["deviation_detection"]["threshold_scale"]
                
                # Skip if not enough data points
                if sorted_features.shape[0] < 3:
                    logger.warning(f"Insufficient data points for EWMA: {sorted_features.shape[0]} < 3")
                else:
                    # Calculate EWMA for each feature
                    anomalies = []
                    
                    for j in range(sorted_features.shape[1]):
                        # Get feature values
                        values = sorted_features[:, j]
                        
                        # Calculate EWMA
                        ewma_values = np.zeros_like(values)
                        ewma_values[0] = values[0]  # Initialize with first value
                        
                        for i in range(1, len(values)):
                            ewma_values[i] = alpha * values[i] + (1 - alpha) * ewma_values[i-1]
                        
                        # Calculate residuals
                        residuals = values - ewma_values
                        
                        # Calculate standard deviation of residuals (excluding the first few points)
                        burn_in = min(5, len(residuals) // 3)  # Skip burn-in period
                        residual_std = np.std(residuals[burn_in:])
                        
                        # Set threshold for anomalies
                        threshold = threshold_scale * residual_std
                        
                        # Detect anomalies (residuals > threshold), excluding burn-in period
                        anomaly_flags = np.abs(residuals) > threshold
                        anomaly_flags[:burn_in] = False  # Exclude burn-in period
                        anomaly_indices = np.where(anomaly_flags)[0]
                        
                        for idx in anomaly_indices:
                            # Get original index
                            orig_idx = original_indices[idx]
                            
                            # Calculate confidence score
                            deviation = abs(residuals[idx])
                            confidence = min(1.0, deviation / (2 * threshold))
                            
                            # Get feature name
                            feature_name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
                            
                            # Create anomaly record
                            anomaly = {
                                "method": "ewma",
                                "index": int(orig_idx),
                                "feature_index": j,
                                "feature_name": feature_name,
                                "value": float(values[idx]),
                                "expected_value": float(ewma_values[idx]),
                                "deviation": float(residuals[idx]),
                                "threshold": float(threshold),
                                "confidence": float(confidence),
                                "metadata": metadata[orig_idx] if orig_idx < len(metadata) else {},
                                "description": f"EWMA anomaly in {feature_name} "
                                              f"(deviation: {residuals[idx]:.2f}, threshold: {threshold:.2f})"
                            }
                            
                            anomalies.append(anomaly)
                    
                    # Add anomalies to results
                    results["anomalies"].extend(anomalies)
                    
                    # Add method info
                    results["methods_used"].append({
                        "name": "ewma",
                        "alpha": alpha,
                        "threshold_scale": threshold_scale,
                        "anomalies_found": len(anomalies)
                    })
                    
            except Exception as e:
                logger.warning(f"Error in EWMA anomaly detection: {e}")
        
        # Seasonal decomposition would be implemented here for a more complete solution
        
        return results
    
    def _ensemble_detection(
        self,
        method_results: Dict[str, Dict[str, Any]],
        validation_results: List[ValidationResult]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Combine anomaly detection results from multiple methods.
        
        Args:
            method_results: Results from different anomaly detection methods
            validation_results: List of validation results
            
        Returns:
            Dictionary mapping sample indices to ensemble voting results
        """
        # Count how many methods detected each result as anomalous
        anomaly_counts = {}
        total_methods = 0
        
        # Process each method's results
        for method_name, result in method_results.items():
            # Skip if method didn't detect any anomalies
            if ("anomalies" not in result or 
                not isinstance(result["anomalies"], list) or
                not result["anomalies"]):
                continue
            
            # Count this as a valid method
            total_methods += 1
            
            # Process anomalies detected by this method
            for anomaly in result["anomalies"]:
                # Get sample index
                idx = anomaly.get("index")
                if idx is None:
                    continue
                
                # Initialize entry if needed
                if idx not in anomaly_counts:
                    anomaly_counts[idx] = {
                        "count": 0,
                        "total": total_methods,
                        "methods": []
                    }
                
                # Update count and add method
                anomaly_counts[idx]["count"] += 1
                anomaly_counts[idx]["methods"].append({
                    "method": method_name,
                    "confidence": anomaly.get("confidence", 0.5)
                })
        
        # Update total methods for all entries
        for idx in anomaly_counts:
            anomaly_counts[idx]["total"] = total_methods
        
        return anomaly_counts
    
    def _analyze_anomaly_context(
        self,
        anomalies: List[Dict[str, Any]],
        validation_results: List[ValidationResult],
        features: np.ndarray,
        feature_names: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the context of detected anomalies.
        
        Args:
            anomalies: List of detected anomalies
            validation_results: List of validation results
            features: Feature matrix
            feature_names: List of feature names
            metadata: List of metadata dictionaries for each sample
            
        Returns:
            Dictionary with anomaly context analysis
        """
        # Skip if no anomalies detected
        if not anomalies:
            return {"status": "skipped", "message": "No anomalies detected"}
        
        context_analysis = {
            "summary": {},
            "patterns": [],
            "contexts": {}
        }
        
        # Get anomaly indices
        anomaly_indices = [a["index"] for a in anomalies if "index" in a]
        
        # Analyze metadata patterns
        metadata_patterns = {}
        
        # Get all metadata fields
        metadata_fields = set()
        for meta in metadata:
            metadata_fields.update(meta.keys())
        
        # Count occurrences of each metadata value in anomalies
        for field in metadata_fields:
            if field == "index" or field == "timestamp":
                continue
                
            value_counts = defaultdict(int)
            total_with_field = 0
            
            for idx in anomaly_indices:
                if idx < len(metadata) and field in metadata[idx]:
                    value = metadata[idx][field]
                    value_counts[value] += 1
                    total_with_field += 1
            
            if total_with_field > 0:
                # Calculate frequency of each value
                frequencies = {}
                for value, count in value_counts.items():
                    frequencies[str(value)] = count / total_with_field
                
                # Find most common value
                most_common = max(value_counts.items(), key=lambda x: x[1])
                frequency = most_common[1] / total_with_field
                
                # Add to patterns if frequency is high enough
                if frequency >= 0.5:  # Only include if at least 50% have this value
                    metadata_patterns[field] = {
                        "most_common": str(most_common[0]),
                        "frequency": frequency,
                        "distribution": frequencies
                    }
        
        # Find correlated features for anomalies
        if features.shape[0] > 0:
            try:
                # Get correlation threshold
                correlation_threshold = self.config["context_analysis"]["correlation_threshold"]
                
                # Calculate correlations between features
                correlation_matrix = np.corrcoef(features.T)
                
                # Find strong correlations
                strong_correlations = []
                
                for i in range(correlation_matrix.shape[0]):
                    for j in range(i+1, correlation_matrix.shape[1]):
                        if abs(correlation_matrix[i, j]) >= correlation_threshold:
                            strong_correlations.append({
                                "features": [
                                    feature_names[i] if i < len(feature_names) else f"feature_{i}",
                                    feature_names[j] if j < len(feature_names) else f"feature_{j}"
                                ],
                                "correlation": float(correlation_matrix[i, j])
                            })
                
                # Sort by absolute correlation
                strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                
                # Add to context analysis
                context_analysis["feature_correlations"] = strong_correlations
                
            except Exception as e:
                logger.warning(f"Error calculating feature correlations: {e}")
        
        # Add metadata patterns to analysis
        context_analysis["metadata_patterns"] = metadata_patterns
        
        # Calculate temporal context (time since last anomaly, frequency)
        if len(anomaly_indices) > 1:
            try:
                # Sort anomalies by timestamp
                anomaly_timestamps = []
                for idx in anomaly_indices:
                    if idx < len(validation_results):
                        ts = validation_results[idx].validation_timestamp
                        if isinstance(ts, str):
                            anomaly_timestamps.append(datetime.datetime.fromisoformat(ts))
                        else:
                            anomaly_timestamps.append(ts)
                
                # Skip if no valid timestamps
                if anomaly_timestamps:
                    # Sort timestamps
                    sorted_timestamps = sorted(anomaly_timestamps)
                    
                    # Calculate time differences
                    time_diffs = []
                    for i in range(1, len(sorted_timestamps)):
                        diff = sorted_timestamps[i] - sorted_timestamps[i-1]
                        time_diffs.append(diff.total_seconds() / 3600)  # Convert to hours
                    
                    # Calculate temporal statistics
                    if time_diffs:
                        context_analysis["temporal_context"] = {
                            "avg_hours_between_anomalies": np.mean(time_diffs),
                            "max_hours_between_anomalies": np.max(time_diffs),
                            "min_hours_between_anomalies": np.min(time_diffs),
                            "anomalies_per_day": 24 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else None
                        }
            except Exception as e:
                logger.warning(f"Error calculating temporal context: {e}")
        
        # Create context for each anomaly
        for anomaly in anomalies:
            idx = anomaly.get("index")
            if idx is None:
                continue
                
            # Initialize context
            context_analysis["contexts"][idx] = {
                "metadata": metadata[idx] if idx < len(metadata) else {},
                "related_patterns": []
            }
            
            # Add metadata patterns that match this anomaly
            for field, pattern in metadata_patterns.items():
                if (idx < len(metadata) and field in metadata[idx] and
                    str(metadata[idx][field]) == pattern["most_common"]):
                    context_analysis["contexts"][idx]["related_patterns"].append({
                        "type": "metadata",
                        "field": field,
                        "value": pattern["most_common"],
                        "frequency": pattern["frequency"]
                    })
        
        # Analyze clusters if requested
        if self.config["context_analysis"]["cluster_analysis"] and len(anomaly_indices) >= 3:
            try:
                from sklearn.cluster import KMeans
                
                # Get anomaly features
                anomaly_features = np.zeros((len(anomaly_indices), features.shape[1]))
                for i, idx in enumerate(anomaly_indices):
                    if idx < features.shape[0]:
                        anomaly_features[i] = features[idx]
                
                # Normalize features
                anomaly_features = self._normalize_features(anomaly_features)
                
                # Determine optimal number of clusters (up to 5)
                max_clusters = min(5, len(anomaly_indices) // 2)
                
                if max_clusters >= 2:
                    # Calculate silhouette scores for different numbers of clusters
                    from sklearn.metrics import silhouette_score
                    silhouette_scores = []
                    
                    for n_clusters in range(2, max_clusters + 1):
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = kmeans.fit_predict(anomaly_features)
                        
                        # Skip if any cluster has only one sample
                        if min(np.bincount(cluster_labels)) < 2:
                            silhouette_scores.append(-1)
                            continue
                        
                        # Calculate silhouette score
                        score = silhouette_score(anomaly_features, cluster_labels)
                        silhouette_scores.append(score)
                    
                    # Select optimal number of clusters
                    if silhouette_scores and max(silhouette_scores) > 0:
                        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 since we start from 2
                    else:
                        optimal_clusters = 2  # Default to 2 clusters
                    
                    # Apply KMeans with optimal number of clusters
                    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(anomaly_features)
                    
                    # Add cluster info to anomaly contexts
                    for i, idx in enumerate(anomaly_indices):
                        if i < len(cluster_labels) and idx in context_analysis["contexts"]:
                            cluster_id = int(cluster_labels[i])
                            context_analysis["contexts"][idx]["cluster"] = cluster_id
                    
                    # Add cluster info to analysis
                    context_analysis["cluster_analysis"] = {
                        "num_clusters": optimal_clusters,
                        "silhouette_score": float(max(silhouette_scores)) if silhouette_scores else None,
                        "cluster_sizes": [int(count) for count in np.bincount(cluster_labels)],
                        "clusters": {}
                    }
                    
                    # Analyze each cluster
                    for cluster_id in range(optimal_clusters):
                        # Get anomalies in this cluster
                        cluster_indices = [anomaly_indices[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                        
                        # Analyze cluster characteristics
                        cluster_metadata = {}
                        
                        for field in metadata_fields:
                            if field == "index" or field == "timestamp":
                                continue
                                
                            value_counts = defaultdict(int)
                            total_with_field = 0
                            
                            for idx in cluster_indices:
                                if idx < len(metadata) and field in metadata[idx]:
                                    value = metadata[idx][field]
                                    value_counts[value] += 1
                                    total_with_field += 1
                            
                            if total_with_field > 0:
                                # Find most common value
                                most_common = max(value_counts.items(), key=lambda x: x[1])
                                frequency = most_common[1] / total_with_field
                                
                                # Add to cluster metadata if frequency is high enough
                                if frequency >= 0.7:  # Only include if at least 70% have this value
                                    cluster_metadata[field] = {
                                        "value": str(most_common[0]),
                                        "frequency": frequency
                                    }
                        
                        # Add cluster info
                        context_analysis["cluster_analysis"]["clusters"][str(cluster_id)] = {
                            "size": len(cluster_indices),
                            "common_metadata": cluster_metadata,
                            "indices": cluster_indices
                        }
                
            except Exception as e:
                logger.warning(f"Error in cluster analysis: {e}")
        
        return context_analysis
    
    def _generate_anomaly_summary(
        self,
        anomalies: List[Dict[str, Any]],
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Generate a summary of detected anomalies.
        
        Args:
            anomalies: List of detected anomalies
            validation_results: List of validation results
            
        Returns:
            Dictionary with anomaly summary
        """
        summary = {
            "total_anomalies": len(anomalies),
            "total_results": len(validation_results),
            "anomaly_percentage": len(anomalies) / len(validation_results) * 100 if validation_results else 0,
            "methods": {},
            "features": {},
            "metadata": {}
        }
        
        # Skip if no anomalies
        if not anomalies:
            return summary
        
        # Count anomalies by method
        method_counts = defaultdict(int)
        for anomaly in anomalies:
            method = anomaly.get("method", "unknown")
            method_counts[method] += 1
        
        summary["methods"] = {method: count for method, count in method_counts.items()}
        
        # Count anomalies by feature
        feature_counts = defaultdict(int)
        for anomaly in anomalies:
            # Check if anomaly has feature-specific information
            if "features" in anomaly and isinstance(anomaly["features"], list):
                for feature_info in anomaly["features"]:
                    feature = feature_info.get("feature", "unknown")
                    feature_counts[feature] += 1
            elif "feature_name" in anomaly:
                feature = anomaly["feature_name"]
                feature_counts[feature] += 1
        
        summary["features"] = {feature: count for feature, count in feature_counts.items()}
        
        # Analyze metadata patterns
        metadata_patterns = {}
        
        # Get metadata fields
        metadata_fields = set()
        for anomaly in anomalies:
            if "metadata" in anomaly and isinstance(anomaly["metadata"], dict):
                metadata_fields.update(anomaly["metadata"].keys())
        
        # Count occurrences of each metadata value
        for field in metadata_fields:
            if field == "index" or field == "timestamp":
                continue
                
            value_counts = defaultdict(int)
            total_with_field = 0
            
            for anomaly in anomalies:
                if ("metadata" in anomaly and isinstance(anomaly["metadata"], dict) and
                    field in anomaly["metadata"]):
                    value = anomaly["metadata"][field]
                    value_counts[value] += 1
                    total_with_field += 1
            
            if total_with_field > 0:
                # Calculate frequency of each value
                frequencies = {}
                for value, count in value_counts.items():
                    frequencies[str(value)] = count / total_with_field
                
                # Add to patterns
                metadata_patterns[field] = {
                    "distribution": frequencies,
                    "total": total_with_field
                }
        
        summary["metadata"] = metadata_patterns
        
        # Analyze confidence distribution
        confidence_values = [anomaly.get("confidence", 0.5) for anomaly in anomalies]
        
        if confidence_values:
            summary["confidence"] = {
                "mean": float(np.mean(confidence_values)),
                "median": float(np.median(confidence_values)),
                "min": float(np.min(confidence_values)),
                "max": float(np.max(confidence_values)),
                "high_confidence_count": sum(1 for c in confidence_values if c >= 0.7)
            }
        
        return summary
    
    def _generate_key_findings(
        self,
        anomalies: List[Dict[str, Any]],
        context_analysis: Dict[str, Any],
        summary: Dict[str, Any]
    ) -> List[str]:
        """
        Generate key findings based on anomaly detection results.
        
        Args:
            anomalies: List of detected anomalies
            context_analysis: Anomaly context analysis
            summary: Anomaly summary
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Skip if no anomalies
        if not anomalies:
            findings.append("No anomalies detected in the validation results")
            return findings
        
        # Add general finding about anomaly percentage
        anomaly_percentage = summary.get("anomaly_percentage", 0)
        findings.append(f"Detected {len(anomalies)} anomalies "
                       f"({anomaly_percentage:.1f}% of validation results)")
        
        # Add finding about high confidence anomalies
        if "confidence" in summary and "high_confidence_count" in summary["confidence"]:
            high_conf_count = summary["confidence"]["high_confidence_count"]
            if high_conf_count > 0:
                findings.append(f"Found {high_conf_count} high-confidence anomalies "
                              f"that require attention")
        
        # Add findings about metadata patterns
        if "metadata_patterns" in context_analysis:
            patterns = context_analysis["metadata_patterns"]
            
            for field, pattern in patterns.items():
                frequency = pattern.get("frequency", 0)
                value = pattern.get("most_common", "unknown")
                
                if frequency >= 0.7:  # Only add if at least 70% have this value
                    findings.append(f"{frequency*100:.0f}% of anomalies occurred with "
                                  f"{field}={value}")
        
        # Add findings about feature correlations
        if "feature_correlations" in context_analysis:
            correlations = context_analysis["feature_correlations"]
            
            if correlations:
                # Take top correlation
                corr = correlations[0]
                features = corr["features"]
                coefficient = corr["correlation"]
                
                direction = "positive" if coefficient > 0 else "negative"
                findings.append(f"Strong {direction} correlation ({abs(coefficient):.2f}) "
                              f"between {features[0]} and {features[1]} in anomalies")
        
        # Add findings about common features in anomalies
        if "features" in summary:
            feature_counts = summary["features"]
            
            if feature_counts:
                # Find most common feature
                most_common = max(feature_counts.items(), key=lambda x: x[1])
                feature, count = most_common
                
                findings.append(f"{count} anomalies ({count/len(anomalies)*100:.0f}%) "
                              f"involve the {feature} metric")
        
        # Add findings about cluster analysis
        if "cluster_analysis" in context_analysis:
            cluster_analysis = context_analysis["cluster_analysis"]
            
            if "clusters" in cluster_analysis:
                clusters = cluster_analysis["clusters"]
                
                if len(clusters) > 1:
                    findings.append(f"Anomalies form {len(clusters)} distinct clusters, "
                                  f"suggesting multiple root causes")
                
                # Find largest cluster
                largest_cluster = None
                max_size = 0
                
                for cluster_id, cluster in clusters.items():
                    if cluster["size"] > max_size:
                        max_size = cluster["size"]
                        largest_cluster = (cluster_id, cluster)
                
                if largest_cluster:
                    cluster_id, cluster = largest_cluster
                    
                    if "common_metadata" in cluster and cluster["common_metadata"]:
                        # Find most distinctive metadata
                        metadata_str = ", ".join([
                            f"{field}={data['value']}" 
                            for field, data in cluster["common_metadata"].items()
                        ])
                        
                        findings.append(f"Cluster {cluster_id} ({cluster['size']} anomalies) "
                                     f"is characterized by {metadata_str}")
        
        # Add finding about temporal patterns
        if "temporal_context" in context_analysis:
            temporal = context_analysis["temporal_context"]
            
            if "anomalies_per_day" in temporal and temporal["anomalies_per_day"]:
                rate = temporal["anomalies_per_day"]
                findings.append(f"Anomalies occur at a rate of approximately "
                              f"{rate:.1f} per day")
        
        # Ensure at least one finding
        if not findings:
            findings.append("Analysis complete, but no clear patterns identified in anomalies")
        
        return findings
    
    def _generate_recommendations(
        self,
        anomalies: List[Dict[str, Any]],
        anomaly_insights: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on anomaly detection results.
        
        Args:
            anomalies: List of detected anomalies
            anomaly_insights: Insights from anomaly analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Skip if no anomalies
        if not anomalies:
            recommendations.append("Continue monitoring validation results for potential anomalies")
            return recommendations
        
        # Get summary and context analysis
        summary = anomaly_insights.get("summary", {})
        context_analysis = anomaly_insights.get("context_analysis", {})
        
        # Add recommendation about high confidence anomalies
        if "confidence" in summary and "high_confidence_count" in summary["confidence"]:
            high_conf_count = summary["confidence"]["high_confidence_count"]
            if high_conf_count > 0:
                recommendations.append(f"Investigate the {high_conf_count} high-confidence anomalies "
                                     f"as a priority")
        
        # Add recommendations based on metadata patterns
        if "metadata_patterns" in context_analysis:
            patterns = context_analysis["metadata_patterns"]
            
            for field, pattern in patterns.items():
                frequency = pattern.get("frequency", 0)
                value = pattern.get("most_common", "unknown")
                
                if frequency >= 0.7:  # Only add if at least 70% have this value
                    recommendations.append(f"Review simulation configuration for {field}={value} "
                                        f"as it's associated with {frequency*100:.0f}% of anomalies")
        
        # Add recommendations based on cluster analysis
        if "cluster_analysis" in context_analysis:
            cluster_analysis = context_analysis["cluster_analysis"]
            
            if "clusters" in cluster_analysis:
                clusters = cluster_analysis["clusters"]
                
                if len(clusters) > 1:
                    recommendations.append(f"Address each of the {len(clusters)} anomaly clusters "
                                         f"separately as they likely have different root causes")
                
                # Find largest cluster
                largest_cluster = None
                max_size = 0
                
                for cluster_id, cluster in clusters.items():
                    if cluster["size"] > max_size:
                        max_size = cluster["size"]
                        largest_cluster = (cluster_id, cluster)
                
                if largest_cluster:
                    cluster_id, cluster = largest_cluster
                    
                    if "common_metadata" in cluster and cluster["common_metadata"]:
                        # Find most distinctive metadata
                        metadata_items = list(cluster["common_metadata"].items())
                        if metadata_items:
                            field, data = metadata_items[0]
                            recommendations.append(f"Prioritize investigating issues with "
                                               f"{field}={data['value']} configurations")
        
        # Add recommendations based on feature analysis
        if "features" in summary:
            feature_counts = summary["features"]
            
            if feature_counts:
                # Find most common feature
                most_common = max(feature_counts.items(), key=lambda x: x[1])
                feature, count = most_common
                
                recommendations.append(f"Focus on improving simulation accuracy for the "
                                     f"{feature} metric which appears in {count} anomalies")
        
        # Add recommendation based on temporal patterns
        if "temporal_context" in context_analysis:
            temporal = context_analysis["temporal_context"]
            
            if "avg_hours_between_anomalies" in temporal:
                hours = temporal["avg_hours_between_anomalies"]
                if hours < 24:
                    recommendations.append(f"Set up more frequent monitoring with alerts, "
                                        f"as anomalies occur approximately every {hours:.1f} hours")
        
        # Add general recommendations
        if len(anomalies) / summary.get("total_results", 1) > 0.2:
            # High anomaly rate
            recommendations.append("Review and recalibrate the simulation model as the "
                                 "anomaly rate is unusually high")
        
        # Ensure at least one recommendation
        if not recommendations:
            recommendations.append("Continue monitoring validation results and collect more "
                                "data to identify clear patterns")
        
        return recommendations