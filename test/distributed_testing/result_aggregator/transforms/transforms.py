#!/usr/bin/env python3
"""
Data Transformation Module for Result Aggregator

This module provides specialized data transformation functions for processing
distributed testing results, including data cleaning, normalization, feature
extraction, and preprocessing for analysis and reporting.

Usage:
    from result_aggregator.transforms.transforms import (
        normalize_metrics, extract_features, clean_outliers, transform_for_analysis
    )
    
    # Normalize metrics in a dataset
    normalized_data = normalize_metrics(data, metrics=['latency', 'throughput'])
    
    # Extract features for analysis or machine learning
    features = extract_features(data, feature_set='performance')
    
    # Clean outliers from a dataset
    cleaned_data = clean_outliers(data, columns=['latency', 'throughput'], method='zscore')
    
    # Apply a complete transformation pipeline for analysis
    analysis_data = transform_for_analysis(data, target='performance_analysis')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import re
import logging
from pathlib import Path
import math
import json

logger = logging.getLogger(__name__)


def normalize_metrics(
    data: pd.DataFrame,
    metrics: List[str],
    method: str = 'min-max',
    target_range: Optional[Tuple[float, float]] = None,
    group_by: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Normalize metrics in a dataset.
    
    Args:
        data: Input DataFrame
        metrics: List of metrics to normalize
        method: Normalization method ('min-max', 'z-score', 'robust', 'log')
        target_range: Target range for min-max normalization (default: (0, 1))
        group_by: Optional column(s) to group by before normalizing
    
    Returns:
        DataFrame with normalized metrics
    
    Raises:
        ValueError: If invalid parameters or data are provided
    """
    # Validate method
    valid_methods = ['min-max', 'z-score', 'robust', 'log']
    if method not in valid_methods:
        raise ValueError(f"Invalid normalization method: {method}")
    
    # Check if metrics exist
    missing_metrics = [m for m in metrics if m not in data.columns]
    if missing_metrics:
        raise ValueError(f"Metrics not found in data: {missing_metrics}")
    
    # Create a copy of the input data
    result = data.copy()
    
    # Apply normalization within groups if specified
    if group_by:
        group_cols = [group_by] if isinstance(group_by, str) else group_by
        missing_groups = [g for g in group_cols if g not in data.columns]
        if missing_groups:
            raise ValueError(f"Group columns not found in data: {missing_groups}")
        
        groups = result.groupby(group_cols)
        
        for _, group in groups:
            group_idx = group.index
            
            for metric in metrics:
                result.loc[group_idx, metric] = _normalize_series(
                    group[metric], method, target_range
                )
        
        return result
    
    # Apply normalization to the entire dataset
    for metric in metrics:
        result[metric] = _normalize_series(data[metric], method, target_range)
    
    return result


def _normalize_series(
    series: pd.Series,
    method: str,
    target_range: Optional[Tuple[float, float]] = None
) -> pd.Series:
    """
    Normalize a single series using the specified method.
    
    Args:
        series: Input Series
        method: Normalization method
        target_range: Target range for min-max normalization
    
    Returns:
        Normalized Series
    """
    if method == 'min-max':
        # Default target range is (0, 1)
        min_val, max_val = target_range or (0, 1)
        
        series_min = series.min()
        series_max = series.max()
        
        if series_max == series_min:
            # Handle constant series
            return pd.Series(np.ones(len(series)) * (min_val + max_val) / 2, index=series.index)
        
        return min_val + (max_val - min_val) * (series - series_min) / (series_max - series_min)
    
    elif method == 'z-score':
        # Z-score normalization
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            # Handle constant series
            return pd.Series(np.zeros(len(series)), index=series.index)
        
        return (series - mean) / std
    
    elif method == 'robust':
        # Robust scaling using median and IQR
        median = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            # Handle constant series
            return pd.Series(np.zeros(len(series)), index=series.index)
        
        return (series - median) / iqr
    
    elif method == 'log':
        # Log transformation (add 1 to handle zeros)
        return np.log1p(series)
    
    # Shouldn't reach here if method validation is done properly
    return series


def extract_features(
    data: pd.DataFrame,
    feature_set: str = 'performance',
    include_derived: bool = True,
    include_statistical: bool = True,
    time_window: Optional[int] = None,
    time_column: str = 'timestamp'
) -> pd.DataFrame:
    """
    Extract features for analysis or machine learning.
    
    Args:
        data: Input DataFrame
        feature_set: Type of features to extract ('performance', 'hardware', 'error', 'all')
        include_derived: Whether to include derived features
        include_statistical: Whether to include statistical features
        time_window: Optional time window in hours for time-based features
        time_column: Column containing timestamps
    
    Returns:
        DataFrame with extracted features
    
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Validate feature_set
    valid_feature_sets = ['performance', 'hardware', 'error', 'all']
    if feature_set not in valid_feature_sets:
        raise ValueError(f"Invalid feature set: {feature_set}")
    
    # Create a copy of the input data
    result = data.copy()
    
    if feature_set == 'all':
        feature_sets = ['performance', 'hardware', 'error']
    else:
        feature_sets = [feature_set]
    
    # Extract each type of features
    for fs in feature_sets:
        if fs == 'performance':
            result = _extract_performance_features(
                result, include_derived, include_statistical
            )
        elif fs == 'hardware':
            result = _extract_hardware_features(result, include_derived)
        elif fs == 'error':
            result = _extract_error_features(result, include_statistical)
    
    # Add time-based features if requested
    if time_window is not None:
        if time_column not in result.columns:
            logger.warning(f"Time column '{time_column}' not found, skipping time-based features")
        else:
            result = _add_time_features(result, time_column, time_window)
    
    return result


def _extract_performance_features(
    data: pd.DataFrame,
    include_derived: bool = True,
    include_statistical: bool = True
) -> pd.DataFrame:
    """
    Extract performance-related features.
    
    Args:
        data: Input DataFrame
        include_derived: Whether to include derived features
        include_statistical: Whether to include statistical features
    
    Returns:
        DataFrame with performance features
    """
    result = data.copy()
    
    # Check for key performance metrics
    performance_metrics = ['latency', 'throughput', 'memory_usage', 'cpu_usage', 'gpu_usage']
    available_metrics = [m for m in performance_metrics if m in data.columns]
    
    if not available_metrics:
        logger.warning("No standard performance metrics found in data")
        return result
    
    # Add derived performance features
    if include_derived:
        # Efficiency (throughput / resource usage)
        if 'throughput' in available_metrics:
            if 'latency' in available_metrics:
                result['throughput_latency_ratio'] = result['throughput'] / result['latency'].replace(0, np.nan)
            
            if 'memory_usage' in available_metrics:
                result['memory_efficiency'] = result['throughput'] / result['memory_usage'].replace(0, np.nan)
            
            if 'cpu_usage' in available_metrics:
                result['cpu_efficiency'] = result['throughput'] / result['cpu_usage'].replace(0, np.nan)
            
            if 'gpu_usage' in available_metrics:
                result['gpu_efficiency'] = result['throughput'] / result['gpu_usage'].replace(0, np.nan)
        
        # Resource usage ratio
        if 'cpu_usage' in available_metrics and 'gpu_usage' in available_metrics:
            total_usage = result['cpu_usage'] + result['gpu_usage']
            result['cpu_ratio'] = result['cpu_usage'] / total_usage.replace(0, np.nan)
            result['gpu_ratio'] = result['gpu_usage'] / total_usage.replace(0, np.nan)
    
    # Add statistical features
    if include_statistical and len(data) >= 3:
        for metric in available_metrics:
            # Calculate z-scores
            mean = result[metric].mean()
            std = result[metric].std()
            
            if std > 0:
                result[f'{metric}_zscore'] = (result[metric] - mean) / std
            
            # Calculate percentiles relative to the whole dataset
            result[f'{metric}_percentile'] = result[metric].rank(pct=True) * 100
            
            # Calculate rolling metrics if timestamps are available
            if 'timestamp' in result.columns and len(result) >= 5:
                try:
                    ts_result = result.sort_values('timestamp')
                    rolling = ts_result[metric].rolling(window=3, min_periods=1)
                    ts_result[f'{metric}_trend'] = rolling.mean().diff()
                    ts_result[f'{metric}_volatility'] = rolling.std()
                    result = ts_result
                except Exception as e:
                    logger.warning(f"Error calculating rolling metrics: {str(e)}")
    
    return result


def _extract_hardware_features(
    data: pd.DataFrame,
    include_derived: bool = True
) -> pd.DataFrame:
    """
    Extract hardware-related features.
    
    Args:
        data: Input DataFrame
        include_derived: Whether to include derived features
    
    Returns:
        DataFrame with hardware features
    """
    result = data.copy()
    
    # Check for key hardware columns
    hardware_columns = [
        'hardware_type', 'compute_units', 'memory_capacity', 'clock_speed',
        'architecture', 'driver_version'
    ]
    available_columns = [c for c in hardware_columns if c in data.columns]
    
    if not available_columns:
        logger.warning("No standard hardware columns found in data")
        return result
    
    # Encode categorical hardware features
    if 'hardware_type' in available_columns:
        # Create one-hot encoding for hardware type
        try:
            hardware_dummies = pd.get_dummies(
                result['hardware_type'], prefix='hw', dummy_na=False
            )
            result = pd.concat([result, hardware_dummies], axis=1)
        except Exception as e:
            logger.warning(f"Error encoding hardware_type: {str(e)}")
    
    if 'architecture' in available_columns:
        # Create one-hot encoding for architecture
        try:
            arch_dummies = pd.get_dummies(
                result['architecture'], prefix='arch', dummy_na=False
            )
            result = pd.concat([result, arch_dummies], axis=1)
        except Exception as e:
            logger.warning(f"Error encoding architecture: {str(e)}")
    
    # Add derived hardware features
    if include_derived:
        # Memory per compute unit
        if 'compute_units' in available_columns and 'memory_capacity' in available_columns:
            result['memory_per_cu'] = result['memory_capacity'] / result['compute_units'].replace(0, np.nan)
        
        # Parse driver version for feature extraction if it exists
        if 'driver_version' in available_columns:
            try:
                # Extract major and minor version numbers
                result['driver_major'] = result['driver_version'].str.extract(r'(\d+)\.').astype(float)
                result['driver_minor'] = result['driver_version'].str.extract(r'\d+\.(\d+)').astype(float)
            except Exception as e:
                logger.warning(f"Error parsing driver version: {str(e)}")
    
    return result


def _extract_error_features(
    data: pd.DataFrame,
    include_statistical: bool = True
) -> pd.DataFrame:
    """
    Extract error-related features.
    
    Args:
        data: Input DataFrame
        include_statistical: Whether to include statistical features
    
    Returns:
        DataFrame with error features
    """
    result = data.copy()
    
    # Check for key error columns
    error_columns = ['status', 'error_message', 'error_type', 'success']
    available_columns = [c for c in error_columns if c in data.columns]
    
    if not available_columns:
        logger.warning("No standard error columns found in data")
        return result
    
    # Add binary success/failure indicator if not present
    if 'success' not in available_columns and 'status' in available_columns:
        result['success'] = result['status'].str.lower() == 'success'
    
    # Categorize errors if error_message exists
    if 'error_message' in available_columns and 'error_category' not in result.columns:
        # Define common error categories and their keywords
        error_categories = {
            'timeout': ['timeout', 'timed out', 'deadline exceeded'],
            'connection': ['connection', 'network', 'unreachable', 'refused', 'reset'],
            'permission': ['permission', 'access denied', 'unauthorized'],
            'resource': ['resource', 'memory', 'disk', 'storage', 'out of memory'],
            'validation': ['validation', 'invalid', 'schema', 'constraint'],
            'not_found': ['not found', '404', 'missing'],
            'runtime': ['runtime', 'execution', 'exception', 'error']
        }
        
        # Categorize errors
        def categorize_error(error_msg):
            if pd.isna(error_msg) or error_msg == '':
                return None
            
            error_msg = str(error_msg).lower()
            for category, keywords in error_categories.items():
                if any(kw in error_msg for kw in keywords):
                    return category
            
            return 'other'
        
        result['error_category'] = result['error_message'].apply(categorize_error)
        
        # Create binary indicators for error categories
        try:
            if 'error_category' in result.columns:
                cat_dummies = pd.get_dummies(
                    result['error_category'], prefix='err', dummy_na=False
                )
                result = pd.concat([result, cat_dummies], axis=1)
        except Exception as e:
            logger.warning(f"Error encoding error categories: {str(e)}")
    
    # Add statistical features for errors
    if include_statistical and len(data) >= 3:
        if 'success' in result.columns:
            # Calculate overall success rate
            if 'timestamp' in result.columns:
                try:
                    # Sort by timestamp
                    ts_result = result.sort_values('timestamp')
                    
                    # Calculate rolling success rate
                    rolling = ts_result['success'].astype(float).rolling(window=5, min_periods=1)
                    ts_result['success_rate'] = rolling.mean()
                    
                    # Calculate success rate trend
                    ts_result['success_rate_trend'] = ts_result['success_rate'].diff()
                    
                    result = ts_result
                except Exception as e:
                    logger.warning(f"Error calculating success rate: {str(e)}")
            else:
                # Calculate global success rate
                result['success_rate'] = result['success'].mean()
    
    return result


def _add_time_features(
    data: pd.DataFrame,
    time_column: str,
    time_window: int
) -> pd.DataFrame:
    """
    Add time-based features.
    
    Args:
        data: Input DataFrame
        time_column: Column containing timestamps
        time_window: Time window in hours
    
    Returns:
        DataFrame with time-based features
    """
    result = data.copy()
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[time_column]):
        result[time_column] = pd.to_datetime(result[time_column], errors='coerce')
    
    # Extract basic time components
    result['hour_of_day'] = result[time_column].dt.hour
    result['day_of_week'] = result[time_column].dt.dayofweek
    result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate time since first data point (in hours)
    min_time = result[time_column].min()
    if pd.notna(min_time):
        result['hours_since_start'] = (result[time_column] - min_time).dt.total_seconds() / 3600
    
    # Calculate time window features
    try:
        # Sort by timestamp
        result = result.sort_values(time_column)
        
        # Create a time window indicator
        max_time = result[time_column].max()
        cutoff_time = max_time - pd.Timedelta(hours=time_window)
        result['in_recent_window'] = (result[time_column] >= cutoff_time).astype(int)
        
        # Calculate recency score (1.0 for most recent, 0.0 for oldest)
        time_range = (result[time_column].max() - result[time_column].min()).total_seconds()
        if time_range > 0:
            result['recency_score'] = (result[time_column] - result[time_column].min()).dt.total_seconds() / time_range
        else:
            result['recency_score'] = 1.0
    except Exception as e:
        logger.warning(f"Error calculating time window features: {str(e)}")
    
    return result


def clean_outliers(
    data: pd.DataFrame,
    columns: List[str],
    method: str = 'zscore',
    threshold: float = 3.0,
    replace_with: str = 'mean'
) -> pd.DataFrame:
    """
    Clean outliers from a dataset.
    
    Args:
        data: Input DataFrame
        columns: Columns to check for outliers
        method: Outlier detection method ('zscore', 'iqr', 'percentile')
        threshold: Threshold for outlier detection
            For 'zscore': z-score threshold (default: 3.0)
            For 'iqr': IQR multiplier (default: 1.5)
            For 'percentile': percentile threshold (0-100, default: 99)
        replace_with: How to handle outliers ('mean', 'median', 'mode', 'drop', 'clip')
    
    Returns:
        DataFrame with outliers cleaned
    
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Validate method
    valid_methods = ['zscore', 'iqr', 'percentile']
    if method not in valid_methods:
        raise ValueError(f"Invalid outlier detection method: {method}")
    
    # Validate replace_with
    valid_replacements = ['mean', 'median', 'mode', 'drop', 'clip']
    if replace_with not in valid_replacements:
        raise ValueError(f"Invalid outlier replacement method: {replace_with}")
    
    # Check if columns exist
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in data: {missing_columns}")
    
    # Create a copy of the input data
    result = data.copy()
    
    # Detect and clean outliers for each column
    for col in columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(result[col]):
            logger.warning(f"Column '{col}' is not numeric, skipping outlier detection")
            continue
        
        # Detect outliers
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs((result[col] - result[col].mean()) / result[col].std())
            is_outlier = z_scores > threshold
        
        elif method == 'iqr':
            # IQR method
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            is_outlier = (result[col] < lower_bound) | (result[col] > upper_bound)
        
        elif method == 'percentile':
            # Percentile method
            if threshold <= 0 or threshold >= 100:
                threshold = 99  # Default to 99th percentile
            
            lower_bound = result[col].quantile(1 - threshold/100)
            upper_bound = result[col].quantile(threshold/100)
            is_outlier = (result[col] < lower_bound) | (result[col] > upper_bound)
        
        # Skip if no outliers detected
        if not is_outlier.any():
            logger.info(f"No outliers detected in column '{col}'")
            continue
        
        # Handle outliers
        outlier_count = is_outlier.sum()
        logger.info(f"Detected {outlier_count} outliers in column '{col}' ({outlier_count/len(result)*100:.1f}%)")
        
        if replace_with == 'drop':
            # Drop rows with outliers
            result = result[~is_outlier]
        
        elif replace_with == 'clip':
            # Clip outliers to the threshold
            if method == 'zscore':
                mean = result[col].mean()
                std = result[col].std()
                lower = mean - threshold * std
                upper = mean + threshold * std
                result.loc[is_outlier, col] = result.loc[is_outlier, col].clip(lower, upper)
            
            elif method == 'iqr':
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                result.loc[is_outlier, col] = result.loc[is_outlier, col].clip(lower, upper)
            
            elif method == 'percentile':
                lower = result[col].quantile(1 - threshold/100)
                upper = result[col].quantile(threshold/100)
                result.loc[is_outlier, col] = result.loc[is_outlier, col].clip(lower, upper)
        
        else:
            # Replace with a central tendency measure
            if replace_with == 'mean':
                replacement = result[~is_outlier][col].mean()
            elif replace_with == 'median':
                replacement = result[~is_outlier][col].median()
            elif replace_with == 'mode':
                replacement = result[~is_outlier][col].mode()[0]
            
            result.loc[is_outlier, col] = replacement
    
    return result


def transform_for_analysis(
    data: pd.DataFrame,
    target: str,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Apply a complete transformation pipeline for a specific analysis target.
    
    Args:
        data: Input DataFrame
        target: Analysis target ('performance_analysis', 'anomaly_detection',
                                'hardware_comparison', 'error_analysis')
        config: Optional configuration for the transformation pipeline
    
    Returns:
        Transformed DataFrame ready for the specified analysis
    
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Validate target
    valid_targets = [
        'performance_analysis', 'anomaly_detection',
        'hardware_comparison', 'error_analysis'
    ]
    if target not in valid_targets:
        raise ValueError(f"Invalid analysis target: {target}")
    
    # Use default config if not provided
    if config is None:
        config = {}
    
    # Create a copy of the input data
    result = data.copy()
    
    # Apply target-specific transformations
    if target == 'performance_analysis':
        # Configuration for performance analysis
        metrics = config.get('metrics', ['latency', 'throughput', 'memory_usage'])
        group_by = config.get('group_by', None)
        time_window = config.get('time_window', 24)
        
        # Extract features for performance analysis
        result = extract_features(
            result,
            feature_set='performance',
            include_derived=True,
            include_statistical=True,
            time_window=time_window
        )
        
        # Clean outliers
        if config.get('clean_outliers', True):
            result = clean_outliers(
                result,
                columns=metrics,
                method=config.get('outlier_method', 'zscore'),
                threshold=config.get('outlier_threshold', 3.0),
                replace_with=config.get('outlier_replacement', 'clip')
            )
        
        # Normalize metrics
        if config.get('normalize', True):
            result = normalize_metrics(
                result,
                metrics=metrics,
                method=config.get('normalization_method', 'min-max'),
                group_by=group_by
            )
    
    elif target == 'anomaly_detection':
        # Configuration for anomaly detection
        metrics = config.get('metrics', ['latency', 'throughput', 'memory_usage'])
        include_errors = config.get('include_errors', True)
        
        # Extract both performance and error features if needed
        feature_sets = ['performance']
        if include_errors:
            feature_sets.append('error')
        
        for fs in feature_sets:
            result = extract_features(
                result,
                feature_set=fs,
                include_derived=True,
                include_statistical=True,
                time_window=config.get('time_window', 24)
            )
        
        # For anomaly detection, we don't clean outliers or normalize,
        # as these might remove the anomalies we're trying to detect
    
    elif target == 'hardware_comparison':
        # Configuration for hardware comparison
        metrics = config.get('metrics', ['latency', 'throughput', 'memory_usage'])
        
        # Extract hardware and performance features
        result = extract_features(
            result,
            feature_set='hardware',
            include_derived=True
        )
        
        result = extract_features(
            result,
            feature_set='performance',
            include_derived=True,
            include_statistical=False
        )
        
        # Clean outliers
        if config.get('clean_outliers', True):
            result = clean_outliers(
                result,
                columns=metrics,
                method=config.get('outlier_method', 'iqr'),
                threshold=config.get('outlier_threshold', 1.5),
                replace_with=config.get('outlier_replacement', 'clip')
            )
        
        # Normalize metrics by hardware type
        if config.get('normalize', True):
            result = normalize_metrics(
                result,
                metrics=metrics,
                method=config.get('normalization_method', 'min-max'),
                group_by='hardware_type'
            )
    
    elif target == 'error_analysis':
        # Configuration for error analysis
        time_window = config.get('time_window', 168)  # Default: 1 week
        
        # Extract error features
        result = extract_features(
            result,
            feature_set='error',
            include_statistical=True,
            time_window=time_window
        )
        
        # Extract performance features if requested
        if config.get('include_performance', False):
            result = extract_features(
                result,
                feature_set='performance',
                include_derived=True,
                include_statistical=True
            )
    
    return result


def prepare_for_visualization(
    data: pd.DataFrame,
    viz_type: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare data for visualization.
    
    Args:
        data: Input DataFrame
        viz_type: Visualization type ('time_series', 'comparison', 'distribution', 
                                     'correlation', 'anomaly', 'heatmap')
        config: Optional configuration for visualization preparation
    
    Returns:
        Dictionary with data and metadata ready for visualization
    
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Validate viz_type
    valid_viz_types = [
        'time_series', 'comparison', 'distribution', 
        'correlation', 'anomaly', 'heatmap'
    ]
    if viz_type not in valid_viz_types:
        raise ValueError(f"Invalid visualization type: {viz_type}")
    
    # Use default config if not provided
    if config is None:
        config = {}
    
    # Prepare result dictionary
    result = {
        'data': None,
        'metadata': {
            'viz_type': viz_type,
            'config': config,
            'data_summary': {
                'row_count': len(data),
                'column_count': len(data.columns)
            }
        }
    }
    
    # Apply visualization-specific transformations
    if viz_type == 'time_series':
        result = _prepare_time_series_viz(data, config, result)
    
    elif viz_type == 'comparison':
        result = _prepare_comparison_viz(data, config, result)
    
    elif viz_type == 'distribution':
        result = _prepare_distribution_viz(data, config, result)
    
    elif viz_type == 'correlation':
        result = _prepare_correlation_viz(data, config, result)
    
    elif viz_type == 'anomaly':
        result = _prepare_anomaly_viz(data, config, result)
    
    elif viz_type == 'heatmap':
        result = _prepare_heatmap_viz(data, config, result)
    
    return result


def _prepare_time_series_viz(
    data: pd.DataFrame,
    config: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare data for time series visualization.
    
    Args:
        data: Input DataFrame
        config: Visualization configuration
        result: Initial result dictionary
    
    Returns:
        Updated result dictionary
    """
    # Get configuration
    time_column = config.get('time_column', 'timestamp')
    metrics = config.get('metrics', [col for col in data.columns if col != time_column])
    group_by = config.get('group_by', None)
    resample = config.get('resample', None)
    
    # Validate time column
    if time_column not in data.columns:
        raise ValueError(f"Time column '{time_column}' not found in data")
    
    # Ensure time column is datetime
    data_copy = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(data_copy[time_column]):
        data_copy[time_column] = pd.to_datetime(data_copy[time_column], errors='coerce')
    
    # Sort by time
    data_copy = data_copy.sort_values(time_column)
    
    # Filter metrics to include only those in the data
    available_metrics = [m for m in metrics if m in data_copy.columns]
    if not available_metrics:
        logger.warning(f"None of the specified metrics {metrics} found in data")
        available_metrics = [col for col in data_copy.columns 
                           if col != time_column and pd.api.types.is_numeric_dtype(data_copy[col])]
    
    # Prepare data based on grouping
    if group_by:
        group_cols = [group_by] if isinstance(group_by, str) else group_by
        missing_groups = [g for g in group_cols if g not in data_copy.columns]
        if missing_groups:
            raise ValueError(f"Group columns not found in data: {missing_groups}")
        
        # Group and resample data
        viz_data = []
        
        for group_name, group_df in data_copy.groupby(group_cols):
            group_name = group_name if not isinstance(group_name, tuple) else '_'.join(map(str, group_name))
            
            # Resample if requested
            if resample:
                group_df = group_df.set_index(time_column)
                group_df = group_df[available_metrics].resample(resample).mean().reset_index()
            
            # Create series for each metric
            for metric in available_metrics:
                if metric in group_df.columns:
                    viz_data.append({
                        'name': f"{group_name}_{metric}",
                        'group': group_name,
                        'metric': metric,
                        'x': group_df[time_column].tolist(),
                        'y': group_df[metric].tolist()
                    })
        
        result['data'] = viz_data
        result['metadata']['groups'] = list(data_copy.groupby(group_cols).groups.keys())
    
    else:
        # Resample if requested
        if resample:
            data_copy = data_copy.set_index(time_column)
            data_copy = data_copy[available_metrics].resample(resample).mean().reset_index()
        
        # Create series for each metric
        viz_data = []
        for metric in available_metrics:
            if metric in data_copy.columns:
                viz_data.append({
                    'name': metric,
                    'metric': metric,
                    'x': data_copy[time_column].tolist(),
                    'y': data_copy[metric].tolist()
                })
        
        result['data'] = viz_data
    
    # Add time range metadata
    result['metadata']['time_range'] = {
        'start': data_copy[time_column].min().isoformat(),
        'end': data_copy[time_column].max().isoformat(),
        'duration_hours': (data_copy[time_column].max() - data_copy[time_column].min()).total_seconds() / 3600
    }
    
    result['metadata']['metrics'] = available_metrics
    
    return result


def _prepare_comparison_viz(
    data: pd.DataFrame,
    config: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare data for comparison visualization.
    
    Args:
        data: Input DataFrame
        config: Visualization configuration
        result: Initial result dictionary
    
    Returns:
        Updated result dictionary
    """
    # Get configuration
    categories = config.get('categories')
    if not categories:
        raise ValueError("'categories' must be specified for comparison visualization")
    
    metrics = config.get('metrics')
    if not metrics:
        raise ValueError("'metrics' must be specified for comparison visualization")
    
    sort_by = config.get('sort_by', None)
    limit = config.get('limit', None)
    
    # Validate categories and metrics
    if not isinstance(categories, str) and not isinstance(categories, list):
        raise ValueError("'categories' must be a string or list")
    
    cat_cols = [categories] if isinstance(categories, str) else categories
    missing_cats = [c for c in cat_cols if c not in data.columns]
    if missing_cats:
        raise ValueError(f"Category columns not found in data: {missing_cats}")
    
    if not isinstance(metrics, str) and not isinstance(metrics, list):
        raise ValueError("'metrics' must be a string or list")
    
    metric_cols = [metrics] if isinstance(metrics, str) else metrics
    missing_metrics = [m for m in metric_cols if m not in data.columns]
    if missing_metrics:
        raise ValueError(f"Metric columns not found in data: {missing_metrics}")
    
    # Group by categories and calculate metrics
    data_copy = data.copy()
    grouped = data_copy.groupby(cat_cols)[metric_cols].agg(['mean', 'std', 'count'])
    
    # Flatten multi-level columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    # Reset index to convert categories back to columns
    grouped = grouped.reset_index()
    
    # Sort if requested
    if sort_by:
        if sort_by in grouped.columns:
            grouped = grouped.sort_values(sort_by, ascending=False)
        else:
            logger.warning(f"Sort column '{sort_by}' not found, skipping sort")
    
    # Limit rows if requested
    if limit and limit > 0:
        grouped = grouped.head(limit)
    
    # Prepare data for visualization
    categories_flat = '_'.join(cat_cols) if len(cat_cols) > 1 else cat_cols[0]
    
    viz_data = []
    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        
        if mean_col in grouped.columns:
            metric_data = {
                'name': metric,
                'categories': grouped[categories_flat].tolist(),
                'values': grouped[mean_col].tolist()
            }
            
            # Add error bars if available
            if std_col in grouped.columns:
                metric_data['errors'] = grouped[std_col].tolist()
            
            # Add sample counts if available
            if count_col in grouped.columns:
                metric_data['counts'] = grouped[count_col].tolist()
            
            viz_data.append(metric_data)
    
    result['data'] = viz_data
    result['metadata']['categories'] = cat_cols
    result['metadata']['metrics'] = metric_cols
    
    return result


def _prepare_distribution_viz(
    data: pd.DataFrame,
    config: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare data for distribution visualization.
    
    Args:
        data: Input DataFrame
        config: Visualization configuration
        result: Initial result dictionary
    
    Returns:
        Updated result dictionary
    """
    # Get configuration
    columns = config.get('columns')
    if not columns:
        # Use all numeric columns by default
        columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        if not columns:
            raise ValueError("No numeric columns found in data")
    
    # Validate columns
    if not isinstance(columns, list):
        columns = [columns]
    
    missing_cols = [c for c in columns if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")
    
    # Gather distribution data for each column
    viz_data = []
    
    for col in columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(data[col]):
            logger.warning(f"Column '{col}' is not numeric, skipping")
            continue
        
        # Calculate distribution statistics
        col_data = data[col].dropna()
        
        if len(col_data) == 0:
            logger.warning(f"Column '{col}' has no valid data, skipping")
            continue
        
        # Calculate bins for histogram
        bin_count = min(max(10, int(np.sqrt(len(col_data)))), 50)
        hist, bin_edges = np.histogram(col_data, bins=bin_count)
        
        # Calculate basic statistics
        stats = {
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'q1': float(col_data.quantile(0.25)),
            'q3': float(col_data.quantile(0.75))
        }
        
        # Create data for column
        viz_data.append({
            'name': col,
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'statistics': stats,
            'raw_values': col_data.tolist() if config.get('include_raw', False) else None
        })
    
    result['data'] = viz_data
    result['metadata']['columns'] = columns
    
    return result


def _prepare_correlation_viz(
    data: pd.DataFrame,
    config: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare data for correlation visualization.
    
    Args:
        data: Input DataFrame
        config: Visualization configuration
        result: Initial result dictionary
    
    Returns:
        Updated result dictionary
    """
    # Get configuration
    columns = config.get('columns')
    if not columns:
        # Use all numeric columns by default
        columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        if not columns:
            raise ValueError("No numeric columns found in data")
    
    method = config.get('method', 'pearson')
    valid_methods = ['pearson', 'spearman', 'kendall']
    if method not in valid_methods:
        logger.warning(f"Invalid correlation method: {method}, using 'pearson'")
        method = 'pearson'
    
    # Validate columns
    if not isinstance(columns, list):
        columns = [columns]
    
    missing_cols = [c for c in columns if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")
    
    # Calculate correlation matrix
    data_copy = data[columns].copy()
    corr_matrix = data_copy.corr(method=method)
    
    # Convert to the format needed for visualization
    matrix_data = []
    for i, row_col in enumerate(corr_matrix.index):
        for j, col_col in enumerate(corr_matrix.columns):
            matrix_data.append({
                'row': row_col,
                'column': col_col,
                'value': float(corr_matrix.iloc[i, j])
            })
    
    result['data'] = {
        'matrix': matrix_data,
        'columns': list(corr_matrix.columns)
    }
    
    result['metadata']['correlation_method'] = method
    result['metadata']['columns'] = columns
    
    # Add correlation statistics if requested
    if config.get('include_stats', False):
        # Calculate p-values for correlations
        import scipy.stats as stats
        
        p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                               index=corr_matrix.index, 
                               columns=corr_matrix.columns)
        
        for i, row_col in enumerate(corr_matrix.index):
            for j, col_col in enumerate(corr_matrix.columns):
                if i != j:  # Skip diagonal (self-correlation)
                    if method == 'pearson':
                        corr, p_value = stats.pearsonr(data_copy[row_col].dropna(), 
                                                     data_copy[col_col].dropna())
                    elif method == 'spearman':
                        corr, p_value = stats.spearmanr(data_copy[row_col].dropna(), 
                                                      data_copy[col_col].dropna())
                    elif method == 'kendall':
                        corr, p_value = stats.kendalltau(data_copy[row_col].dropna(), 
                                                       data_copy[col_col].dropna())
                    
                    p_values.iloc[i, j] = p_value
        
        # Add p-values to the result
        p_value_data = []
        for i, row_col in enumerate(p_values.index):
            for j, col_col in enumerate(p_values.columns):
                p_value_data.append({
                    'row': row_col,
                    'column': col_col,
                    'value': float(p_values.iloc[i, j])
                })
        
        result['data']['p_values'] = p_value_data
    
    return result


def _prepare_anomaly_viz(
    data: pd.DataFrame,
    config: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare data for anomaly visualization.
    
    Args:
        data: Input DataFrame
        config: Visualization configuration
        result: Initial result dictionary
    
    Returns:
        Updated result dictionary
    """
    # Get configuration
    anomaly_column = config.get('anomaly_column')
    if not anomaly_column:
        logger.warning("No anomaly column specified, attempting to detect")
        # Try to find an anomaly column
        anomaly_candidates = [col for col in data.columns 
                             if 'anomaly' in col.lower() or 'outlier' in col.lower()]
        
        if anomaly_candidates:
            anomaly_column = anomaly_candidates[0]
            logger.info(f"Using detected anomaly column: {anomaly_column}")
        else:
            raise ValueError("No anomaly column found in data")
    
    metrics = config.get('metrics')
    if not metrics:
        # Use all numeric columns that aren't the anomaly column
        metrics = [col for col in data.columns 
                  if col != anomaly_column and pd.api.types.is_numeric_dtype(data[col])]
        
        if not metrics:
            raise ValueError("No metric columns found in data")
    
    time_column = config.get('time_column', 'timestamp')
    
    # Validate columns
    if anomaly_column not in data.columns:
        raise ValueError(f"Anomaly column '{anomaly_column}' not found in data")
    
    if not isinstance(metrics, list):
        metrics = [metrics]
    
    missing_metrics = [m for m in metrics if m not in data.columns]
    if missing_metrics:
        raise ValueError(f"Metric columns not found in data: {missing_metrics}")
    
    has_time = time_column in data.columns
    
    # Prepare data for visualization
    data_copy = data.copy()
    
    # Ensure anomaly column is boolean or numeric
    if not pd.api.types.is_bool_dtype(data_copy[anomaly_column]) and not pd.api.types.is_numeric_dtype(data_copy[anomaly_column]):
        logger.warning(f"Anomaly column '{anomaly_column}' is not boolean or numeric, attempting to convert")
        # Try to convert to boolean
        if data_copy[anomaly_column].isin([0, 1, True, False, 'True', 'False', 'true', 'false']).all():
            data_copy[anomaly_column] = data_copy[anomaly_column].map(
                {0: False, 1: True, True: True, False: False, 
                 'True': True, 'False': False, 'true': True, 'false': False}
            )
        else:
            raise ValueError(f"Anomaly column '{anomaly_column}' cannot be converted to boolean")
    
    # Convert anomaly column to boolean if numeric
    if pd.api.types.is_numeric_dtype(data_copy[anomaly_column]):
        data_copy[anomaly_column] = data_copy[anomaly_column].astype(bool)
    
    # Sort by time if available
    if has_time:
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data_copy[time_column]):
            data_copy[time_column] = pd.to_datetime(data_copy[time_column], errors='coerce')
        
        data_copy = data_copy.sort_values(time_column)
    
    # Separate normal and anomalous points
    normal = data_copy[~data_copy[anomaly_column]]
    anomalies = data_copy[data_copy[anomaly_column]]
    
    # Prepare visualization data
    viz_data = []
    
    for metric in metrics:
        metric_data = {
            'name': metric,
            'normal': {},
            'anomalies': {}
        }
        
        if has_time:
            # Prepare time series data
            metric_data['normal']['x'] = normal[time_column].tolist()
            metric_data['normal']['y'] = normal[metric].tolist()
            
            metric_data['anomalies']['x'] = anomalies[time_column].tolist()
            metric_data['anomalies']['y'] = anomalies[metric].tolist()
        else:
            # Prepare distribution data
            metric_data['normal']['values'] = normal[metric].tolist()
            metric_data['anomalies']['values'] = anomalies[metric].tolist()
        
        viz_data.append(metric_data)
    
    result['data'] = viz_data
    result['metadata']['metrics'] = metrics
    result['metadata']['anomaly_column'] = anomaly_column
    result['metadata']['has_time'] = has_time
    
    if has_time:
        result['metadata']['time_range'] = {
            'start': data_copy[time_column].min().isoformat(),
            'end': data_copy[time_column].max().isoformat()
        }
    
    result['metadata']['anomaly_stats'] = {
        'total_points': len(data_copy),
        'normal_points': len(normal),
        'anomaly_points': len(anomalies),
        'anomaly_percentage': len(anomalies) / len(data_copy) * 100 if len(data_copy) > 0 else 0
    }
    
    return result


def _prepare_heatmap_viz(
    data: pd.DataFrame,
    config: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare data for heatmap visualization.
    
    Args:
        data: Input DataFrame
        config: Visualization configuration
        result: Initial result dictionary
    
    Returns:
        Updated result dictionary
    """
    # Get configuration
    x_column = config.get('x_column')
    if not x_column:
        raise ValueError("'x_column' must be specified for heatmap visualization")
    
    y_column = config.get('y_column')
    if not y_column:
        raise ValueError("'y_column' must be specified for heatmap visualization")
    
    value_column = config.get('value_column')
    if not value_column:
        raise ValueError("'value_column' must be specified for heatmap visualization")
    
    # Validate columns
    if x_column not in data.columns:
        raise ValueError(f"X column '{x_column}' not found in data")
    
    if y_column not in data.columns:
        raise ValueError(f"Y column '{y_column}' not found in data")
    
    if value_column not in data.columns:
        raise ValueError(f"Value column '{value_column}' not found in data")
    
    # Create pivot table for heatmap
    try:
        pivot_data = data.pivot_table(
            index=y_column,
            columns=x_column,
            values=value_column,
            aggfunc=config.get('aggfunc', 'mean')
        )
        
        # Convert to lists for visualization
        heatmap_data = {
            'x': list(pivot_data.columns),
            'y': list(pivot_data.index),
            'z': pivot_data.values.tolist()
        }
        
        result['data'] = heatmap_data
        result['metadata']['x_column'] = x_column
        result['metadata']['y_column'] = y_column
        result['metadata']['value_column'] = value_column
        result['metadata']['aggfunc'] = config.get('aggfunc', 'mean')
        
        # Add value range
        result['metadata']['value_range'] = {
            'min': float(pivot_data.values.min()),
            'max': float(pivot_data.values.max()),
            'mean': float(pivot_data.values.mean())
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error creating heatmap data: {str(e)}")
        raise ValueError(f"Failed to create heatmap: {str(e)}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    try:
        # Create sample data
        np.random.seed(42)
        
        # Sample test results data
        n_samples = 100
        
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'),
            'model_name': np.random.choice(['bert', 'gpt', 't5'], n_samples),
            'hardware_type': np.random.choice(['cpu', 'gpu', 'tpu'], n_samples),
            'batch_size': np.random.choice([1, 2, 4, 8, 16], n_samples),
            'latency': np.random.normal(100, 20, n_samples),
            'throughput': np.random.normal(200, 40, n_samples),
            'memory_usage': np.random.normal(1000, 200, n_samples),
            'success': np.random.choice([True, False], n_samples, p=[0.9, 0.1])
        })
        
        # Add some outliers
        data.loc[10, 'latency'] = 500
        data.loc[20, 'throughput'] = 10
        data.loc[30, 'memory_usage'] = 5000
        
        # Add error messages for failed tests
        data.loc[~data['success'], 'error_message'] = np.random.choice(
            ['Timeout error', 'Connection refused', 'Out of memory', 'Invalid input'],
            sum(~data['success'])
        )
        
        print("\nExample 1: Normalize metrics")
        normalized_data = normalize_metrics(
            data, 
            metrics=['latency', 'throughput', 'memory_usage'],
            method='min-max'
        )
        print(normalized_data[['latency', 'throughput', 'memory_usage']].describe())
        
        print("\nExample 2: Extract features")
        features = extract_features(
            data,
            feature_set='performance',
            include_derived=True
        )
        print("Extracted features:", list(features.columns))
        
        print("\nExample 3: Clean outliers")
        cleaned_data = clean_outliers(
            data,
            columns=['latency', 'throughput', 'memory_usage'],
            method='zscore',
            threshold=3.0,
            replace_with='median'
        )
        print("Original data:")
        print(data[['latency', 'throughput', 'memory_usage']].describe())
        print("Cleaned data:")
        print(cleaned_data[['latency', 'throughput', 'memory_usage']].describe())
        
        print("\nExample 4: Transform for analysis")
        analysis_data = transform_for_analysis(
            data,
            target='performance_analysis',
            config={
                'metrics': ['latency', 'throughput'],
                'clean_outliers': True,
                'normalize': True
            }
        )
        print("Analysis data columns:", list(analysis_data.columns))
        
        print("\nExample 5: Prepare for visualization")
        viz_data = prepare_for_visualization(
            data,
            viz_type='time_series',
            config={
                'time_column': 'timestamp',
                'metrics': ['latency', 'throughput'],
                'group_by': 'hardware_type'
            }
        )
        print("Visualization data structure:", json.dumps(viz_data['metadata'], indent=2))
        
    except Exception as e:
        logging.error(f"Error in example: {str(e)}")