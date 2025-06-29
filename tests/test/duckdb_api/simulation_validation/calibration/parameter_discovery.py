#!/usr/bin/env python3
"""
Parameter discovery and sensitivity analysis for the Simulation Accuracy and Validation Framework.

This module provides utilities for automatically discovering which simulation parameters 
need calibration and analyzing their sensitivity to different conditions.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("parameter_discovery")

# Import from parent modules
import os
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import base classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

class AutomaticParameterDiscovery:
    """
    Discovers which simulation parameters need calibration and analyzes their sensitivity.
    
    This class provides methods to:
    1. Identify parameters with significant impact on simulation accuracy
    2. Analyze parameter sensitivity to different conditions (batch size, precision, etc.)
    3. Calculate importance scores for parameters
    4. Recommend a priority list for parameter calibration
    5. Generate insights about parameter relationships
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the automatic parameter discovery system.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "metrics_to_analyze": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w"
            ],
            "min_samples_for_analysis": 5,
            "sensitivity_threshold": 0.05,  # 5% threshold for sensitivity detection
            "importance_calculation_method": "permutation",  # 'permutation', 'correlation', or 'mutual_info'
            "cross_parameter_analysis": True,  # Whether to analyze interactions between parameters
            "batch_size_analysis": True,  # Whether to analyze batch size sensitivity
            "precision_analysis": True,  # Whether to analyze precision sensitivity
            "model_size_analysis": True,  # Whether to analyze model size sensitivity
            "random_state": 42  # Random state for reproducibility
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Store for parameter importance scores
        self.parameter_importance = {}
        
        # Store for parameter sensitivity results
        self.parameter_sensitivity = {}
        
        # Store for discovered parameters
        self.discovered_parameters = {}
    
    def discover_parameters(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Discover important parameters and their sensitivity.
        
        Args:
            validation_results: List of validation results for analysis
            
        Returns:
            Dictionary of discovered parameters and their importance
        """
        if not validation_results:
            logger.warning("No validation results provided for parameter discovery")
            return {}
        
        # Extract data for analysis
        analysis_data = self._prepare_analysis_data(validation_results)
        
        # Check if we have enough data
        if len(analysis_data) < self.config["min_samples_for_analysis"]:
            logger.warning(f"Not enough samples for analysis (need at least {self.config['min_samples_for_analysis']})")
            return {}
        
        # Discover parameters with significant impact
        self.discovered_parameters = self._discover_significant_parameters(analysis_data)
        
        # Analyze parameter sensitivity
        self.parameter_sensitivity = self._analyze_parameter_sensitivity(analysis_data)
        
        # Calculate parameter importance
        self.parameter_importance = self._calculate_parameter_importance(analysis_data)
        
        # Generate parameter recommendations
        parameter_recommendations = self._generate_parameter_recommendations()
        
        return parameter_recommendations
    
    def analyze_parameter_sensitivity(
        self,
        validation_results: List[ValidationResult],
        parameter_name: str
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity of a specific parameter in detail.
        
        Args:
            validation_results: List of validation results
            parameter_name: Name of the parameter to analyze
            
        Returns:
            Dictionary of sensitivity analysis results
        """
        # Extract data for analysis
        analysis_data = self._prepare_analysis_data(validation_results)
        
        # Check if we have enough data
        if len(analysis_data) < self.config["min_samples_for_analysis"]:
            logger.warning(f"Not enough samples for analysis (need at least {self.config['min_samples_for_analysis']})")
            return {}
        
        # Check if parameter exists in the data
        if parameter_name not in analysis_data.columns:
            logger.warning(f"Parameter {parameter_name} not found in analysis data")
            return {}
        
        # Perform detailed sensitivity analysis
        sensitivity_results = self._analyze_single_parameter_sensitivity(
            analysis_data, parameter_name
        )
        
        return sensitivity_results
    
    def get_parameter_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get the importance scores for parameters.
        
        Returns:
            Dictionary mapping metrics to parameter importance scores
        """
        return self.parameter_importance
    
    def get_parameter_sensitivity(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get the sensitivity analysis results for parameters.
        
        Returns:
            Dictionary of sensitivity analysis results
        """
        return self.parameter_sensitivity
    
    def generate_insight_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report with insights about parameters.
        
        Returns:
            Dictionary with detailed insights and recommendations
        """
        insights = {
            "parameter_importance": self.parameter_importance,
            "parameter_sensitivity": self.parameter_sensitivity,
            "discovered_parameters": self.discovered_parameters,
            "recommendations": self._generate_parameter_recommendations(),
            "insights": {
                "key_findings": self._generate_key_findings(),
                "optimization_opportunities": self._identify_optimization_opportunities(),
                "parameter_relationships": self._identify_parameter_relationships()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return insights
    
    def _prepare_analysis_data(
        self, 
        validation_results: List[ValidationResult]
    ) -> pd.DataFrame:
        """
        Prepare data for analysis by extracting relevant features and metrics.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            DataFrame with features and metrics for analysis
        """
        # Try to import pandas, needed for analysis
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required for parameter discovery. Please install it with 'pip install pandas'.")
            return pd.DataFrame()
        
        # Extract data for each validation result
        data_records = []
        
        for val_result in validation_results:
            sim_result = val_result.simulation_result
            hw_result = val_result.hardware_result
            
            # Skip if either result is None
            if sim_result is None or hw_result is None:
                continue
            
            # Extract metrics
            metrics_data = {}
            for metric in self.config["metrics_to_analyze"]:
                # Skip if metric doesn't exist in both results
                if (metric not in sim_result.metrics or 
                    metric not in hw_result.metrics or
                    sim_result.metrics[metric] is None or
                    hw_result.metrics[metric] is None or
                    hw_result.metrics[metric] == 0):
                    continue
                
                # Calculate error metrics
                sim_value = sim_result.metrics[metric]
                hw_value = hw_result.metrics[metric]
                
                absolute_error = abs(sim_value - hw_value)
                relative_error = absolute_error / hw_value
                percentage_error = relative_error * 100
                
                metrics_data[f"{metric}_sim"] = sim_value
                metrics_data[f"{metric}_hw"] = hw_value
                metrics_data[f"{metric}_abs_error"] = absolute_error
                metrics_data[f"{metric}_rel_error"] = relative_error
                metrics_data[f"{metric}_pct_error"] = percentage_error
            
            # Skip if no metrics data was extracted
            if not metrics_data:
                continue
            
            # Extract features from simulation result
            feature_data = {}
            
            # Hardware and model IDs
            feature_data["hardware_id"] = sim_result.hardware_id
            feature_data["model_id"] = sim_result.model_id
            
            # Batch size and precision
            if hasattr(sim_result, "batch_size") and sim_result.batch_size is not None:
                feature_data["batch_size"] = sim_result.batch_size
            
            if hasattr(sim_result, "precision") and sim_result.precision is not None:
                feature_data["precision"] = sim_result.precision
            
            # Additional metadata
            if sim_result.additional_metadata:
                for key, value in sim_result.additional_metadata.items():
                    # Only include scalar values
                    if isinstance(value, (int, float, str, bool)):
                        feature_data[f"metadata_{key}"] = value
            
            # Hardware info
            if hasattr(hw_result, "hardware_info") and hw_result.hardware_info:
                for key, value in hw_result.hardware_info.items():
                    # Only include scalar values
                    if isinstance(value, (int, float, str, bool)):
                        feature_data[f"hw_info_{key}"] = value
            
            # Test environment
            if hasattr(hw_result, "test_environment") and hw_result.test_environment:
                for key, value in hw_result.test_environment.items():
                    # Only include scalar values
                    if isinstance(value, (int, float, str, bool)):
                        feature_data[f"env_{key}"] = value
            
            # Combine all data
            record = {**feature_data, **metrics_data}
            data_records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(data_records)
        
        # Handle categorical variables
        # For hardware_id and model_id, we'll create one-hot encoding
        categorical_columns = ['hardware_id', 'model_id', 'precision']
        for col in categorical_columns:
            if col in df.columns:
                # Create one-hot encoding
                try:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(columns=[col], inplace=True)
                except Exception as e:
                    logger.warning(f"Could not one-hot encode column {col}: {e}")
        
        return df
    
    def _discover_significant_parameters(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Discover parameters with significant impact on simulation accuracy.
        
        Args:
            data: Analysis data
            
        Returns:
            Dictionary mapping metrics to lists of significant parameters
        """
        significant_parameters = {}
        
        # Get all feature columns (excluding metric columns)
        metric_prefixes = [f"{m}_" for m in self.config["metrics_to_analyze"]]
        feature_cols = [col for col in data.columns if not any(col.startswith(prefix) for prefix in metric_prefixes)]
        
        # Analyze each metric
        for metric in self.config["metrics_to_analyze"]:
            error_col = f"{metric}_pct_error"
            if error_col not in data.columns:
                continue
            
            # Find parameters that correlate with error
            correlated_params = []
            for feature in feature_cols:
                # Skip non-numeric features
                if not pd.api.types.is_numeric_dtype(data[feature]):
                    continue
                
                # Calculate correlation
                try:
                    correlation = data[feature].corr(data[error_col])
                    if abs(correlation) >= self.config["sensitivity_threshold"]:
                        correlated_params.append((feature, abs(correlation)))
                except Exception as e:
                    logger.debug(f"Could not calculate correlation for {feature}: {e}")
            
            # Sort by correlation strength
            correlated_params.sort(key=lambda x: x[1], reverse=True)
            
            # Store significant parameters
            significant_parameters[metric] = [p[0] for p in correlated_params]
        
        return significant_parameters
    
    def _analyze_parameter_sensitivity(self, data: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Analyze parameter sensitivity to different conditions.
        
        Args:
            data: Analysis data
            
        Returns:
            Dictionary of sensitivity analysis results
        """
        sensitivity_results = {}
        
        # Get discovered parameters for each metric
        for metric, parameters in self.discovered_parameters.items():
            error_col = f"{metric}_pct_error"
            if error_col not in data.columns:
                continue
            
            metric_sensitivity = {}
            
            # Analyze sensitivity for each parameter
            for parameter in parameters:
                # Skip non-numeric parameters
                if parameter not in data.columns or not pd.api.types.is_numeric_dtype(data[parameter]):
                    continue
                
                param_sensitivity = self._analyze_single_parameter_sensitivity(
                    data, parameter, error_col
                )
                
                if param_sensitivity:
                    metric_sensitivity[parameter] = param_sensitivity
            
            if metric_sensitivity:
                sensitivity_results[metric] = metric_sensitivity
        
        return sensitivity_results
    
    def _analyze_single_parameter_sensitivity(
        self, 
        data: pd.DataFrame, 
        parameter: str,
        error_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity of a single parameter in detail.
        
        Args:
            data: Analysis data
            parameter: Parameter to analyze
            error_col: Error column to use (if None, analyze for all metrics)
            
        Returns:
            Dictionary of sensitivity analysis results
        """
        sensitivity_results = {}
        
        # Skip non-numeric parameters
        if parameter not in data.columns or not pd.api.types.is_numeric_dtype(data[parameter]):
            return sensitivity_results
        
        # Handle the case where no specific error column is specified
        metrics_to_analyze = []
        if error_col is None:
            # Analyze all metrics
            for metric in self.config["metrics_to_analyze"]:
                temp_error_col = f"{metric}_pct_error"
                if temp_error_col in data.columns:
                    metrics_to_analyze.append((metric, temp_error_col))
        else:
            # Extract metric name from error column
            for metric in self.config["metrics_to_analyze"]:
                if error_col == f"{metric}_pct_error":
                    metrics_to_analyze.append((metric, error_col))
                    break
            
            if not metrics_to_analyze:
                logger.warning(f"Could not extract metric from error column: {error_col}")
                return sensitivity_results
        
        # Analyze sensitivity for each metric
        for metric, error_col in metrics_to_analyze:
            # Calculate overall correlation
            correlation = data[parameter].corr(data[error_col])
            
            # Create bins for parameter values
            try:
                # Use qcut to create quantile-based bins
                data['parameter_bin'] = pd.qcut(data[parameter], 5, duplicates='drop')
                
                # Calculate error statistics by bin
                bin_stats = data.groupby('parameter_bin')[error_col].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ]).reset_index()
                
                # Convert to records for easier serialization
                bin_stats_records = bin_stats.to_dict('records')
                
                # Format bin values for better readability
                for record in bin_stats_records:
                    bin_obj = record['parameter_bin']
                    record['parameter_bin'] = [float(str(bin_obj.left)), float(str(bin_obj.right))]
                
                # Store results
                metric_sensitivity = {
                    "correlation": correlation,
                    "bin_statistics": bin_stats_records
                }
                
                # Additional analysis based on configuration
                
                # Batch size analysis
                if self.config["batch_size_analysis"] and "batch_size" in data.columns:
                    batch_sensitivity = self._analyze_parameter_by_factor(
                        data, parameter, error_col, "batch_size")
                    if batch_sensitivity:
                        metric_sensitivity["batch_size_sensitivity"] = batch_sensitivity
                
                # Precision analysis
                if self.config["precision_analysis"]:
                    precision_cols = [col for col in data.columns if col.startswith('precision_')]
                    if precision_cols:
                        precision_sensitivity = {}
                        for col in precision_cols:
                            precision_value = col.split('_')[1]  # Extract precision value from column name
                            precision_data = data[data[col] == 1]  # Filter data for this precision
                            if len(precision_data) >= self.config["min_samples_for_analysis"]:
                                precision_correlation = precision_data[parameter].corr(precision_data[error_col])
                                precision_sensitivity[precision_value] = precision_correlation
                        
                        if precision_sensitivity:
                            metric_sensitivity["precision_sensitivity"] = precision_sensitivity
                
                sensitivity_results[metric] = metric_sensitivity
                
            except Exception as e:
                logger.warning(f"Could not analyze sensitivity for parameter {parameter}: {e}")
                continue
            
            # Clean up temporary column
            if 'parameter_bin' in data.columns:
                data.drop(columns=['parameter_bin'], inplace=True)
        
        return sensitivity_results
    
    def _analyze_parameter_by_factor(
        self, 
        data: pd.DataFrame, 
        parameter: str, 
        error_col: str, 
        factor_col: str
    ) -> Dict[str, float]:
        """
        Analyze parameter sensitivity by a specific factor (e.g., batch size).
        
        Args:
            data: Analysis data
            parameter: Parameter to analyze
            error_col: Error column to use
            factor_col: Factor column to analyze by
            
        Returns:
            Dictionary mapping factor values to parameter sensitivity
        """
        # Check if factor column exists and is numeric
        if factor_col not in data.columns:
            return {}
        
        factor_values = sorted(data[factor_col].unique())
        
        factor_sensitivity = {}
        for value in factor_values:
            # Filter data for this factor value
            factor_data = data[data[factor_col] == value]
            if len(factor_data) >= self.config["min_samples_for_analysis"]:
                # Calculate correlation for this factor value
                factor_correlation = factor_data[parameter].corr(factor_data[error_col])
                factor_sensitivity[str(value)] = factor_correlation
        
        return factor_sensitivity
    
    def _calculate_parameter_importance(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate importance scores for parameters using the selected method.
        
        Args:
            data: Analysis data
            
        Returns:
            Dictionary mapping metrics to parameter importance scores
        """
        importance_scores = {}
        
        # Determine method to use
        method = self.config["importance_calculation_method"]
        
        # Analyze each metric
        for metric in self.config["metrics_to_analyze"]:
            error_col = f"{metric}_pct_error"
            if error_col not in data.columns:
                continue
            
            # Get parameters for this metric
            parameters = self.discovered_parameters.get(metric, [])
            
            # Skip if no parameters were discovered
            if not parameters:
                continue
            
            # Filter to include only numeric parameters
            numeric_params = [p for p in parameters if p in data.columns and pd.api.types.is_numeric_dtype(data[p])]
            
            if not numeric_params:
                continue
            
            # Calculate importance scores based on selected method
            if method == "correlation":
                # Calculate importance based on correlation with error
                scores = {}
                for param in numeric_params:
                    correlation = abs(data[param].corr(data[error_col]))
                    scores[param] = correlation
                
                # Normalize scores
                total = sum(scores.values())
                if total > 0:
                    for param in scores:
                        scores[param] /= total
                
                importance_scores[metric] = scores
                
            elif method == "permutation":
                # Try to import scikit-learn for permutation importance
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.inspection import permutation_importance
                except ImportError:
                    logger.warning("scikit-learn is required for permutation importance. Falling back to correlation.")
                    # Fallback to correlation
                    scores = {}
                    for param in numeric_params:
                        correlation = abs(data[param].corr(data[error_col]))
                        scores[param] = correlation
                    
                    # Normalize scores
                    total = sum(scores.values())
                    if total > 0:
                        for param in scores:
                            scores[param] /= total
                    
                    importance_scores[metric] = scores
                    continue
                
                try:
                    # Create feature matrix and target vector
                    X = data[numeric_params]
                    y = data[error_col]
                    
                    # Train a random forest model
                    model = RandomForestRegressor(
                        n_estimators=100, 
                        random_state=self.config["random_state"]
                    )
                    model.fit(X, y)
                    
                    # Calculate permutation importance
                    result = permutation_importance(
                        model, X, y, 
                        n_repeats=10, 
                        random_state=self.config["random_state"]
                    )
                    
                    # Extract importance scores
                    scores = {}
                    for i, param in enumerate(numeric_params):
                        scores[param] = result.importances_mean[i]
                    
                    # Normalize scores
                    total = sum(scores.values())
                    if total > 0:
                        for param in scores:
                            scores[param] /= total
                    
                    importance_scores[metric] = scores
                    
                except Exception as e:
                    logger.warning(f"Could not calculate permutation importance: {e}")
                    # Fallback to correlation
                    scores = {}
                    for param in numeric_params:
                        correlation = abs(data[param].corr(data[error_col]))
                        scores[param] = correlation
                    
                    # Normalize scores
                    total = sum(scores.values())
                    if total > 0:
                        for param in scores:
                            scores[param] /= total
                    
                    importance_scores[metric] = scores
            
            elif method == "mutual_info":
                # Try to import scikit-learn for mutual information
                try:
                    from sklearn.feature_selection import mutual_info_regression
                except ImportError:
                    logger.warning("scikit-learn is required for mutual information. Falling back to correlation.")
                    # Fallback to correlation
                    scores = {}
                    for param in numeric_params:
                        correlation = abs(data[param].corr(data[error_col]))
                        scores[param] = correlation
                    
                    # Normalize scores
                    total = sum(scores.values())
                    if total > 0:
                        for param in scores:
                            scores[param] /= total
                    
                    importance_scores[metric] = scores
                    continue
                
                try:
                    # Create feature matrix and target vector
                    X = data[numeric_params]
                    y = data[error_col]
                    
                    # Calculate mutual information
                    mi_scores = mutual_info_regression(
                        X, y, 
                        random_state=self.config["random_state"]
                    )
                    
                    # Extract importance scores
                    scores = {}
                    for i, param in enumerate(numeric_params):
                        scores[param] = mi_scores[i]
                    
                    # Normalize scores
                    total = sum(scores.values())
                    if total > 0:
                        for param in scores:
                            scores[param] /= total
                    
                    importance_scores[metric] = scores
                    
                except Exception as e:
                    logger.warning(f"Could not calculate mutual information: {e}")
                    # Fallback to correlation
                    scores = {}
                    for param in numeric_params:
                        correlation = abs(data[param].corr(data[error_col]))
                        scores[param] = correlation
                    
                    # Normalize scores
                    total = sum(scores.values())
                    if total > 0:
                        for param in scores:
                            scores[param] /= total
                    
                    importance_scores[metric] = scores
            
            else:
                logger.warning(f"Unknown importance calculation method: {method}")
                # Fallback to correlation
                scores = {}
                for param in numeric_params:
                    correlation = abs(data[param].corr(data[error_col]))
                    scores[param] = correlation
                
                # Normalize scores
                total = sum(scores.values())
                if total > 0:
                    for param in scores:
                        scores[param] /= total
                
                importance_scores[metric] = scores
        
        return importance_scores
    
    def _generate_parameter_recommendations(self) -> Dict[str, Any]:
        """
        Generate parameter recommendations based on discovery results.
        
        Returns:
            Dictionary of parameter recommendations
        """
        recommendations = {
            "parameters_by_metric": {},
            "overall_priority_list": [],
            "sensitivity_insights": {},
            "optimization_recommendations": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Create parameters by metric
        for metric, importance in self.parameter_importance.items():
            # Sort parameters by importance
            sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            recommendations["parameters_by_metric"][metric] = [
                {"parameter": p, "importance": float(i)} for p, i in sorted_params
            ]
        
        # Create overall priority list
        # Combine importance scores across metrics
        overall_importance = {}
        for metric, importance in self.parameter_importance.items():
            for param, score in importance.items():
                if param not in overall_importance:
                    overall_importance[param] = 0
                
                # Add weighted by the number of metrics
                overall_importance[param] += score / len(self.parameter_importance)
        
        # Sort by overall importance
        sorted_overall = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True)
        recommendations["overall_priority_list"] = [
            {"parameter": p, "importance": float(i)} for p, i in sorted_overall
        ]
        
        # Add sensitivity insights
        for metric, sensitivity in self.parameter_sensitivity.items():
            metric_insights = {}
            
            for param, sensitivity_data in sensitivity.items():
                # Basic insights
                insights = []
                
                # Check correlation
                correlation = sensitivity_data.get("correlation")
                if correlation is not None:
                    if correlation > 0.7:
                        insights.append(f"Strong positive correlation with {metric} error")
                    elif correlation > 0.4:
                        insights.append(f"Moderate positive correlation with {metric} error")
                    elif correlation < -0.7:
                        insights.append(f"Strong negative correlation with {metric} error")
                    elif correlation < -0.4:
                        insights.append(f"Moderate negative correlation with {metric} error")
                
                # Check batch size sensitivity
                batch_sensitivity = sensitivity_data.get("batch_size_sensitivity", {})
                if batch_sensitivity:
                    batch_values = sorted([int(k) for k in batch_sensitivity.keys()])
                    if batch_values:
                        # Check if sensitivity increases with batch size
                        if all(batch_sensitivity[str(batch_values[i])] <= batch_sensitivity[str(batch_values[i+1])] 
                               for i in range(len(batch_values)-1)):
                            insights.append(f"Sensitivity increases with batch size")
                        # Check if sensitivity decreases with batch size
                        elif all(batch_sensitivity[str(batch_values[i])] >= batch_sensitivity[str(batch_values[i+1])] 
                                for i in range(len(batch_values)-1)):
                            insights.append(f"Sensitivity decreases with batch size")
                        # Check for non-monotonic sensitivity
                        else:
                            insights.append(f"Non-monotonic sensitivity to batch size")
                
                # Check precision sensitivity
                precision_sensitivity = sensitivity_data.get("precision_sensitivity", {})
                if precision_sensitivity:
                    precisions = list(precision_sensitivity.keys())
                    if "fp16" in precisions and "fp32" in precisions:
                        fp16_sens = precision_sensitivity["fp16"]
                        fp32_sens = precision_sensitivity["fp32"]
                        if abs(fp16_sens) > abs(fp32_sens) * 1.5:
                            insights.append(f"Higher sensitivity in fp16 precision")
                        elif abs(fp32_sens) > abs(fp16_sens) * 1.5:
                            insights.append(f"Higher sensitivity in fp32 precision")
                
                if insights:
                    metric_insights[param] = insights
            
            if metric_insights:
                recommendations["sensitivity_insights"][metric] = metric_insights
        
        # Add optimization recommendations
        for metric, importance in self.parameter_importance.items():
            metric_recommendations = []
            
            # Get top parameters by importance
            top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for param, score in top_params:
                # Get sensitivity data if available
                sensitivity_data = self.parameter_sensitivity.get(metric, {}).get(param, {})
                
                param_recommendations = []
                
                # Basic recommendation based on correlation
                correlation = sensitivity_data.get("correlation")
                if correlation is not None:
                    if correlation > 0:
                        param_recommendations.append(f"Consider reducing {param} to improve {metric}")
                    else:
                        param_recommendations.append(f"Consider increasing {param} to improve {metric}")
                
                # Recommendations based on batch size sensitivity
                batch_sensitivity = sensitivity_data.get("batch_size_sensitivity", {})
                if batch_sensitivity:
                    # Find batch sizes with highest sensitivity
                    highest_batch = max(batch_sensitivity.items(), key=lambda x: abs(float(x[1])))
                    param_recommendations.append(f"Focus on calibrating for batch size {highest_batch[0]}")
                
                # Recommendations based on precision sensitivity
                precision_sensitivity = sensitivity_data.get("precision_sensitivity", {})
                if precision_sensitivity:
                    # Find precision with highest sensitivity
                    highest_precision = max(precision_sensitivity.items(), key=lambda x: abs(float(x[1])))
                    param_recommendations.append(f"Focus on calibrating for {highest_precision[0]} precision")
                
                if param_recommendations:
                    metric_recommendations.append({
                        "parameter": param,
                        "importance": float(score),
                        "recommendations": param_recommendations
                    })
            
            if metric_recommendations:
                recommendations["optimization_recommendations"][metric] = metric_recommendations
        
        return recommendations
    
    def _generate_key_findings(self) -> List[str]:
        """
        Generate a list of key findings from the analysis.
        
        Returns:
            List of key findings
        """
        findings = []
        
        # Finding 1: Most important parameters overall
        if self.parameter_importance:
            # Get top 3 parameters across all metrics
            overall_importance = {}
            for metric, importance in self.parameter_importance.items():
                for param, score in importance.items():
                    if param not in overall_importance:
                        overall_importance[param] = 0
                    
                    # Add weighted by the number of metrics
                    overall_importance[param] += score / len(self.parameter_importance)
            
            # Get top 3 parameters
            top_params = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_params:
                param_list = ", ".join([p[0] for p in top_params])
                findings.append(f"The most important parameters overall are: {param_list}")
        
        # Finding 2: Parameters with metric-specific importance
        for metric, importance in self.parameter_importance.items():
            # Get top parameter for this metric
            top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:1]
            if top_params:
                param, score = top_params[0]
                # Only include if importance is significant
                if score > 0.25:
                    findings.append(f"For {metric}, the most important parameter is {param} with {score:.2f} importance score")
        
        # Finding 3: Batch size sensitivity
        batch_size_findings = []
        for metric, sensitivity in self.parameter_sensitivity.items():
            for param, sensitivity_data in sensitivity.items():
                batch_sensitivity = sensitivity_data.get("batch_size_sensitivity", {})
                if batch_sensitivity:
                    # Check if sensitivity changes significantly across batch sizes
                    batch_values = sorted([int(k) for k in batch_sensitivity.keys()])
                    if len(batch_values) >= 2:
                        min_sens = min(batch_sensitivity.values())
                        max_sens = max(batch_sensitivity.values())
                        if max_sens > min_sens * 2:  # At least double the sensitivity
                            batch_size_findings.append(f"{param} sensitivity to {metric} varies significantly with batch size")
        
        if batch_size_findings:
            findings.extend(batch_size_findings[:3])  # Add up to 3 batch size findings
        
        # Finding 4: Precision sensitivity
        precision_findings = []
        for metric, sensitivity in self.parameter_sensitivity.items():
            for param, sensitivity_data in sensitivity.items():
                precision_sensitivity = sensitivity_data.get("precision_sensitivity", {})
                if precision_sensitivity:
                    # Check if sensitivity changes significantly across precisions
                    if len(precision_sensitivity) >= 2:
                        min_sens = min(precision_sensitivity.values())
                        max_sens = max(precision_sensitivity.values())
                        if max_sens > min_sens * 2:  # At least double the sensitivity
                            precision_findings.append(f"{param} sensitivity to {metric} varies significantly with precision type")
        
        if precision_findings:
            findings.extend(precision_findings[:3])  # Add up to 3 precision findings
        
        return findings
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify optimization opportunities based on parameter analysis.
        
        Returns:
            List of optimization opportunities
        """
        opportunities = []
        
        # Opportunity 1: Parameters with high importance across multiple metrics
        if len(self.parameter_importance) >= 2:
            # Count parameters that are important across metrics
            param_count = {}
            for metric, importance in self.parameter_importance.items():
                # Get top 3 parameters for each metric
                top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                for param, _ in top_params:
                    if param not in param_count:
                        param_count[param] = 0
                    param_count[param] += 1
            
            # Get parameters important across multiple metrics
            cross_metric_params = [p for p, count in param_count.items() if count >= 2]
            if cross_metric_params:
                opportunities.append({
                    "type": "cross_metric_optimization",
                    "description": "Optimize parameters important across multiple metrics",
                    "parameters": cross_metric_params,
                    "benefit": "Improve performance across multiple metrics with fewer calibration efforts"
                })
        
        # Opportunity 2: Batch size specific optimizations
        batch_size_opportunities = []
        for metric, sensitivity in self.parameter_sensitivity.items():
            metric_opportunities = {}
            
            for param, sensitivity_data in sensitivity.items():
                batch_sensitivity = sensitivity_data.get("batch_size_sensitivity", {})
                if batch_sensitivity:
                    # Find batch sizes with highest sensitivity
                    high_sensitivity_batches = [(k, v) for k, v in batch_sensitivity.items() 
                                                if abs(v) > self.config["sensitivity_threshold"]]
                    
                    if high_sensitivity_batches:
                        # Sort by sensitivity
                        high_sensitivity_batches.sort(key=lambda x: abs(float(x[1])), reverse=True)
                        
                        for batch, sens in high_sensitivity_batches:
                            if batch not in metric_opportunities:
                                metric_opportunities[batch] = []
                            metric_opportunities[batch].append((param, float(sens)))
            
            # Create opportunities for each batch size
            for batch, params in metric_opportunities.items():
                if len(params) >= 2:  # Only if multiple parameters are sensitive
                    batch_size_opportunities.append({
                        "type": "batch_size_optimization",
                        "metric": metric,
                        "batch_size": batch,
                        "description": f"Optimize for batch size {batch} to improve {metric}",
                        "parameters": [{"parameter": p, "sensitivity": s} for p, s in params],
                        "benefit": f"Improved accuracy for {metric} at batch size {batch}"
                    })
        
        if batch_size_opportunities:
            # Sort by the number of parameters
            batch_size_opportunities.sort(key=lambda x: len(x["parameters"]), reverse=True)
            opportunities.extend(batch_size_opportunities[:3])  # Add up to 3 batch size opportunities
        
        # Opportunity 3: Precision specific optimizations
        precision_opportunities = []
        for metric, sensitivity in self.parameter_sensitivity.items():
            precision_params = {}
            
            for param, sensitivity_data in sensitivity.items():
                precision_sensitivity = sensitivity_data.get("precision_sensitivity", {})
                if precision_sensitivity:
                    # Find precisions with highest sensitivity
                    high_sensitivity_precisions = [(k, v) for k, v in precision_sensitivity.items() 
                                                  if abs(v) > self.config["sensitivity_threshold"]]
                    
                    if high_sensitivity_precisions:
                        # Sort by sensitivity
                        high_sensitivity_precisions.sort(key=lambda x: abs(float(x[1])), reverse=True)
                        
                        for precision, sens in high_sensitivity_precisions:
                            if precision not in precision_params:
                                precision_params[precision] = []
                            precision_params[precision].append((param, float(sens)))
            
            # Create opportunities for each precision
            for precision, params in precision_params.items():
                if len(params) >= 2:  # Only if multiple parameters are sensitive
                    precision_opportunities.append({
                        "type": "precision_optimization",
                        "metric": metric,
                        "precision": precision,
                        "description": f"Optimize for {precision} precision to improve {metric}",
                        "parameters": [{"parameter": p, "sensitivity": s} for p, s in params],
                        "benefit": f"Improved accuracy for {metric} in {precision} precision"
                    })
        
        if precision_opportunities:
            # Sort by the number of parameters
            precision_opportunities.sort(key=lambda x: len(x["parameters"]), reverse=True)
            opportunities.extend(precision_opportunities[:3])  # Add up to 3 precision opportunities
        
        return opportunities
    
    def _identify_parameter_relationships(self) -> List[Dict[str, Any]]:
        """
        Identify relationships between parameters.
        
        Returns:
            List of parameter relationships
        """
        relationships = []
        
        # Skip if cross-parameter analysis is disabled
        if not self.config["cross_parameter_analysis"]:
            return relationships
        
        # Get all parameters discovered for any metric
        all_params = set()
        for params in self.discovered_parameters.values():
            all_params.update(params)
        
        # This will likely need more advanced statistical methods
        # For now, let's implement a simple correlation-based approach
        
        # Relationship 1: Highly correlated parameters
        from itertools import combinations
        
        # Try to get analysis data from cache
        analysis_data = getattr(self, '_analysis_data', None)
        if analysis_data is None or len(analysis_data) == 0:
            return relationships  # No data available
        
        # Get numeric parameters only
        numeric_params = [p for p in all_params if p in analysis_data.columns and 
                          pd.api.types.is_numeric_dtype(analysis_data[p])]
        
        if len(numeric_params) < 2:
            return relationships  # Need at least 2 parameters for relationships
        
        # Check correlations between parameters
        correlated_pairs = []
        for param1, param2 in combinations(numeric_params, 2):
            correlation = analysis_data[param1].corr(analysis_data[param2])
            if abs(correlation) > 0.7:  # Strong correlation
                correlated_pairs.append((param1, param2, correlation))
        
        # Sort by correlation strength
        correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Add top correlations
        for param1, param2, correlation in correlated_pairs[:5]:  # Top 5
            relationship_type = "positive" if correlation > 0 else "negative"
            relationships.append({
                "type": "parameter_correlation",
                "parameters": [param1, param2],
                "correlation": float(correlation),
                "relationship_type": relationship_type,
                "description": f"Strong {relationship_type} correlation between {param1} and {param2}"
            })
        
        # Relationship 2: Complementary parameters for metrics
        if len(self.parameter_importance) >= 2:
            for metric1, metric2 in combinations(self.parameter_importance.keys(), 2):
                # Get top parameters for each metric
                top_params1 = set(sorted(self.parameter_importance[metric1].items(), 
                                         key=lambda x: x[1], reverse=True)[:3])
                top_params2 = set(sorted(self.parameter_importance[metric2].items(), 
                                         key=lambda x: x[1], reverse=True)[:3])
                
                # Find parameters unique to each metric
                unique_to_metric1 = {p[0] for p in top_params1 if p[0] not in {p[0] for p in top_params2}}
                unique_to_metric2 = {p[0] for p in top_params2 if p[0] not in {p[0] for p in top_params1}}
                
                if unique_to_metric1 and unique_to_metric2:
                    relationships.append({
                        "type": "complementary_parameters",
                        "metrics": [metric1, metric2],
                        "parameters_for_metric1": list(unique_to_metric1),
                        "parameters_for_metric2": list(unique_to_metric2),
                        "description": f"Different parameters affect {metric1} and {metric2}"
                    })
        
        return relationships