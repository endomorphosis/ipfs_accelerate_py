#!/usr/bin/env python3
"""
ML-based Anomaly Detection Module for Result Aggregator

This module provides specialized machine learning algorithms for detecting anomalies
in distributed testing results, including Isolation Forest, DBSCAN clustering,
and One-Class SVM approaches.

Usage:
    from result_aggregator.ml_detection.ml_anomaly_detector import MLAnomalyDetector
    
    detector = MLAnomalyDetector(db_path='path/to/benchmark_db.duckdb')
    
    # Detect anomalies in performance metrics
    anomalies = detector.detect_anomalies(
        model_name='bert-base-uncased',
        hardware_type='cuda',
        metrics=['latency', 'throughput'],
        time_period_days=30
    )
    
    # Train a model for anomaly detection
    detector.train_anomaly_model(
        model_name='bert-base-uncased',
        hardware_type='cuda',
        metrics=['latency', 'throughput'],
        method='isolation_forest'
    )
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import pickle
import warnings

# Conditional imports for ML libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import DBSCAN
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, 
        roc_auc_score, average_precision_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def _is_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules)


def _log_optional_dependency(message: str) -> None:
    if _is_pytest():
        logger.info(message)
    else:
        logger.warning(message)


class MLAnomalyDetector:
    """
    Machine Learning based anomaly detection for distributed testing results.
    
    This class provides methods for detecting anomalies in test results using:
    - Isolation Forest
    - DBSCAN Clustering
    - Local Outlier Factor (LOF)
    - One-Class SVM
    - Z-score based detection
    - Feature-based anomaly detection
    """
    
    def __init__(self, db_path: str = None, connection = None):
        """
        Initialize the ML Anomaly Detector.
        
        Args:
            db_path: Path to the DuckDB database file
            connection: Existing DuckDB connection (optional)
        
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If neither db_path nor connection is provided
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is required for the MLAnomalyDetector module")
        
        if connection is None and db_path is None:
            raise ValueError("Either db_path or connection must be provided")
        
        self.db_path = db_path
        self._conn = connection if connection else duckdb.connect(db_path)
        self._trained_models = {}
        
        # Log the availability of optional dependencies
        if not SKLEARN_AVAILABLE:
            _log_optional_dependency("Scikit-learn not available. ML-based anomaly detection will be disabled.")
            _log_optional_dependency("Install scikit-learn using: pip install scikit-learn")
        if not SCIPY_AVAILABLE:
            _log_optional_dependency("SciPy not available. Some statistical functions will be limited.")
        if not PLOTTING_AVAILABLE:
            _log_optional_dependency("Matplotlib/Seaborn not available. Visualization will be disabled.")
    
    def detect_anomalies(self,
                         model_name: Optional[str] = None,
                         hardware_type: Optional[str] = None,
                         metrics: List[str] = ['latency', 'throughput'],
                         time_period_days: int = 30,
                         method: str = 'isolation_forest',
                         contamination: float = 0.05,
                         sensitivity: float = 1.0,
                         feature_engineering: bool = True,
                         n_estimators: int = 100,
                         visualize: bool = False,
                         output_path: Optional[str] = None,
                         filter_criteria: Dict[str, Any] = None) -> Dict:
        """
        Detect anomalies in test results using the specified ML method.
        
        Args:
            model_name: Name of the model (None for all models)
            hardware_type: Type of hardware (None for all hardware types)
            metrics: List of metrics to include in anomaly detection
            time_period_days: Number of days to look back
            method: Anomaly detection method ('isolation_forest', 'dbscan', 'lof', 'ocsvm', 'zscore')
            contamination: Expected proportion of anomalies (used for all methods except 'zscore')
            sensitivity: Sensitivity multiplier for threshold-based methods
            feature_engineering: Whether to perform feature engineering
            n_estimators: Number of estimators for ensemble methods
            visualize: Whether to generate visualization
            output_path: Path to save visualization
            filter_criteria: Additional filtering criteria for test results
        
        Returns:
            Dictionary with detected anomalies
        
        Raises:
            ImportError: If required ML libraries are not available
            ValueError: If invalid parameters are provided
            RuntimeError: If anomaly detection fails
        """
        if not SKLEARN_AVAILABLE and method != 'zscore':
            raise ImportError("Scikit-learn is required for ML-based anomaly detection")
        
        # Validate method
        valid_methods = ['isolation_forest', 'dbscan', 'lof', 'ocsvm', 'zscore']
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Valid methods are: {', '.join(valid_methods)}")
        
        # Validate contamination
        if contamination <= 0 or contamination >= 0.5:
            raise ValueError("Contamination must be between 0 and 0.5")
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # Build the query
        query = """
        SELECT r.result_id, r.model_name, r.hardware_type, r.batch_size, r.timestamp,
               r.test_name, r.status, m.metric_name, m.metric_value
        FROM test_results r
        JOIN performance_metrics m ON r.result_id = m.result_id
        WHERE r.timestamp >= ?
        """
        
        params = [cutoff_str]
        
        # Add filters for model and hardware
        if model_name:
            query += " AND r.model_name = ?"
            params.append(model_name)
        
        if hardware_type:
            query += " AND r.hardware_type = ?"
            params.append(hardware_type)
        
        # Add filters for metrics
        if metrics:
            metrics_str = ', '.join([f"'{m}'" for m in metrics])
            query += f" AND m.metric_name IN ({metrics_str})"
        
        # Add additional filters
        if filter_criteria:
            for key, value in filter_criteria.items():
                if isinstance(value, list):
                    placeholders = ', '.join(['?' for _ in value])
                    query += f" AND r.{key} IN ({placeholders})"
                    params.extend(value)
                else:
                    query += f" AND r.{key} = ?"
                    params.append(value)
        
        try:
            # Execute the query
            results = self._conn.execute(query, params).fetchdf()
            
            if results.empty:
                logger.warning("No data found for the specified criteria")
                return {
                    'status': 'warning',
                    'message': 'No data found for the specified criteria',
                    'anomalies': []
                }
            
            # Ensure timestamp is datetime
            results['timestamp'] = pd.to_datetime(results['timestamp'])
            
            # Pivot the data to create a matrix of metrics by result
            # First, create a unique identifier for each test result
            results['test_result_id'] = (
                results['model_name'] + '_' +
                results['hardware_type'] + '_' +
                results['batch_size'].astype(str) + '_' +
                results['test_name'] + '_' +
                results['timestamp'].dt.strftime('%Y%m%d%H%M%S') + '_' +
                results['result_id'].astype(str)
            )
            
            # Create the pivot table
            pivot = results.pivot(index='test_result_id', columns='metric_name', values='metric_value')
            
            # Drop rows with missing values
            pivot = pivot.dropna()
            
            if pivot.empty:
                logger.warning("No complete data rows found after pivoting")
                return {
                    'status': 'warning',
                    'message': 'No complete data rows found after pivoting',
                    'anomalies': []
                }
            
            # Create mapping to original data
            id_mapping = results.drop_duplicates('test_result_id')[
                ['test_result_id', 'result_id', 'model_name', 'hardware_type', 
                 'batch_size', 'test_name', 'timestamp']
            ].set_index('test_result_id')
            
            # Perform feature engineering if requested
            if feature_engineering:
                pivot = self._feature_engineering(pivot)
            
            # Standardize the data
            scaler = StandardScaler()
            X = scaler.fit_transform(pivot)
            
            # Initialize the anomalies list
            anomalies = []
            anomaly_indices = None
            anomaly_scores = None
            
            # Apply the selected anomaly detection method
            if method == 'isolation_forest':
                # Use Isolation Forest
                model = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42
                )
                anomaly_labels = model.fit_predict(X)
                anomaly_scores = model.decision_function(X)
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                
            elif method == 'dbscan':
                # Use DBSCAN clustering
                # Determine epsilon (distance threshold) dynamically
                from sklearn.neighbors import NearestNeighbors
                neighbors = NearestNeighbors(n_neighbors=2)
                neighbors_fit = neighbors.fit(X)
                distances, _ = neighbors_fit.kneighbors(X)
                distances = np.sort(distances[:, 1])
                
                # Find the "elbow" point as a heuristic for epsilon
                from sklearn.cluster import DBSCAN
                elbow = int(len(distances) * 0.05)  # Simple heuristic based on contamination
                epsilon = distances[elbow] * sensitivity
                
                # Apply DBSCAN
                dbscan = DBSCAN(eps=epsilon, min_samples=5)
                labels = dbscan.fit_predict(X)
                
                # Points labeled as -1 are considered outliers
                anomaly_indices = np.where(labels == -1)[0]
                
                # Calculate anomaly scores based on distance to nearest core point
                core_samples_mask = np.zeros_like(labels, dtype=bool)
                core_samples_mask[dbscan.core_sample_indices_] = True
                
                if len(dbscan.core_sample_indices_) > 0:
                    # If there are core points, calculate distance to nearest core point
                    from scipy.spatial.distance import cdist
                    core_points = X[core_samples_mask]
                    distances_to_core = cdist(X, core_points).min(axis=1)
                    anomaly_scores = distances_to_core
                else:
                    # If no core points, use average distance to all points
                    from scipy.spatial.distance import pdist, squareform
                    dist_matrix = squareform(pdist(X))
                    anomaly_scores = np.mean(dist_matrix, axis=1)
                
            elif method == 'lof':
                # Use Local Outlier Factor
                lof = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=contamination
                )
                anomaly_labels = lof.fit_predict(X)
                anomaly_scores = -lof.negative_outlier_factor_
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                
            elif method == 'ocsvm':
                # Use One-Class SVM
                ocsvm = OneClassSVM(
                    kernel='rbf',
                    gamma='auto',
                    nu=contamination
                )
                anomaly_labels = ocsvm.fit_predict(X)
                anomaly_scores = -ocsvm.decision_function(X)
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                
            elif method == 'zscore':
                # Use Z-score based detection
                if not SCIPY_AVAILABLE:
                    # Manually calculate Z-scores
                    means = np.mean(X, axis=0)
                    stds = np.std(X, axis=0)
                    z_scores = np.abs((X - means) / (stds + 1e-10))
                else:
                    # Use scipy for Z-scores
                    z_scores = np.abs(zscore(X, axis=0, nan_policy='omit'))
                
                # Consider a point anomalous if any feature has Z-score > threshold
                threshold = 3.0 * sensitivity  # Default threshold is 3 sigma
                anomaly_mask = np.any(z_scores > threshold, axis=1)
                anomaly_indices = np.where(anomaly_mask)[0]
                
                # Use max Z-score across features as the anomaly score
                anomaly_scores = np.max(z_scores, axis=1)
            
            # Process detected anomalies
            anomaly_info = []
            
            for idx in anomaly_indices:
                test_result_id = pivot.index[idx]
                original_data = id_mapping.loc[test_result_id]
                
                # Get the feature values for this anomaly
                feature_values = pivot.iloc[idx].to_dict()
                
                # Calculate the anomaly score for reporting
                score = float(anomaly_scores[idx]) if anomaly_scores is not None else None
                
                # Get the most anomalous metrics (highest z-scores)
                metric_z_scores = {}
                for col in pivot.columns:
                    value = pivot.iloc[idx][col]
                    mean = pivot[col].mean()
                    std = pivot[col].std()
                    if std > 0:
                        z = abs((value - mean) / std)
                        metric_z_scores[col] = float(z)
                    else:
                        metric_z_scores[col] = 0.0
                
                # Sort metrics by z-score
                sorted_metrics = sorted(
                    metric_z_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Get top 3 most anomalous metrics
                top_metrics = []
                for metric, z in sorted_metrics[:3]:
                    direction = 'high' if pivot.iloc[idx][metric] > pivot[metric].mean() else 'low'
                    top_metrics.append({
                        'metric': metric,
                        'value': float(pivot.iloc[idx][metric]),
                        'z_score': z,
                        'expected_range': [
                            float(pivot[metric].mean() - 2 * pivot[metric].std()),
                            float(pivot[metric].mean() + 2 * pivot[metric].std())
                        ],
                        'direction': direction
                    })
                
                # Create anomaly record
                anomaly_record = {
                    'result_id': str(original_data['result_id']),
                    'model_name': original_data['model_name'],
                    'hardware_type': original_data['hardware_type'],
                    'batch_size': int(original_data['batch_size']),
                    'test_name': original_data['test_name'],
                    'timestamp': original_data['timestamp'].isoformat(),
                    'anomaly_score': float(score) if score is not None else None,
                    'detection_method': method,
                    'feature_values': feature_values,
                    'top_anomalous_metrics': top_metrics
                }
                
                anomaly_info.append(anomaly_record)
            
            # Generate visualization if requested
            if visualize and PLOTTING_AVAILABLE and len(pivot) > 1:
                self._visualize_anomalies(pivot, anomaly_indices, method, output_path)
            
            # Create the final result
            result = {
                'status': 'success',
                'method': method,
                'total_points': len(pivot),
                'anomalies_found': len(anomaly_indices),
                'anomaly_percentage': len(anomaly_indices) / len(pivot) * 100 if len(pivot) > 0 else 0,
                'anomalies': anomaly_info,
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'model_name': model_name,
                    'hardware_type': hardware_type,
                    'metrics': metrics,
                    'time_period_days': time_period_days,
                    'method': method,
                    'contamination': contamination,
                    'sensitivity': sensitivity,
                    'feature_engineering': feature_engineering
                }
            }
            
            return result
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to detect anomalies: {str(e)}")
    
    def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the input data.
        
        Args:
            data: Input DataFrame with metrics as columns
        
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        enhanced = data.copy()
        
        # Add basic statistical features
        for col in data.columns:
            # Add rolling statistics if enough data points
            if len(data) >= 5:
                # Skip if the column is already a derived feature
                if col.startswith('derived_'):
                    continue
                    
                # Calculate metrics relative to group means
                mean_name = f"derived_{col}_rel_mean"
                enhanced[mean_name] = (data[col] - data[col].mean()) / (data[col].std() + 1e-10)
        
        # Compute ratios between key metrics if they exist
        metric_pairs = [
            ('latency', 'throughput'),
            ('memory_usage', 'throughput'),
            ('latency', 'memory_usage')
        ]
        
        for m1, m2 in metric_pairs:
            if m1 in data.columns and m2 in data.columns:
                ratio_name = f"derived_ratio_{m1}_{m2}"
                # Add small epsilon to avoid division by zero
                enhanced[ratio_name] = data[m1] / (data[m2] + 1e-10)
        
        # Compute derived metrics based on domain knowledge
        if 'latency' in data.columns and 'throughput' in data.columns:
            enhanced['derived_efficiency'] = 1 / (data['latency'] * (1 / data['throughput'] + 1e-10) + 1e-10)
        
        if 'memory_usage' in data.columns and 'throughput' in data.columns:
            enhanced['derived_memory_efficiency'] = data['throughput'] / (data['memory_usage'] + 1e-10)
        
        # Handle NaN values created during feature engineering
        enhanced = enhanced.fillna(0)
        
        return enhanced
    
    def _visualize_anomalies(self, 
                           data: pd.DataFrame, 
                           anomaly_indices: np.ndarray,
                           method: str,
                           output_path: Optional[str] = None):
        """
        Visualize detected anomalies.
        
        Args:
            data: Input DataFrame with metrics as columns
            anomaly_indices: Indices of detected anomalies
            method: Anomaly detection method used
            output_path: Path to save visualization
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available. Skipping visualization.")
            return
        
        # Apply PCA for dimensionality reduction if we have more than 2 features
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(StandardScaler().fit_transform(data))
            
            # Create a DataFrame with PCA results
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=data.index)
            
            # Add anomaly labels
            pca_df['anomaly'] = 0
            pca_df.iloc[anomaly_indices, -1] = 1
            
            # Create the scatter plot
            plt.figure(figsize=(10, 8))
            
            # Plot normal points
            normal = pca_df[pca_df['anomaly'] == 0]
            plt.scatter(normal['PC1'], normal['PC2'], c='blue', marker='o', 
                       alpha=0.5, label='Normal')
            
            # Plot anomalies
            anomalies = pca_df[pca_df['anomaly'] == 1]
            plt.scatter(anomalies['PC1'], anomalies['PC2'], c='red', marker='x', 
                       s=100, label='Anomaly')
            
            plt.title(f"Anomaly Detection using {method.replace('_', ' ').title()}")
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add information about top contributing features
            feature_weights = pd.DataFrame(
                pca.components_.T, 
                columns=['PC1', 'PC2'], 
                index=data.columns
            )
            
            # Top 3 features for PC1
            top_pc1 = feature_weights['PC1'].abs().sort_values(ascending=False).head(3)
            top_pc2 = feature_weights['PC2'].abs().sort_values(ascending=False).head(3)
            
            pc1_text = "PC1 top features:\n" + "\n".join([
                f"{feat}: {feature_weights.loc[feat, 'PC1']:.3f}" 
                for feat in top_pc1.index
            ])
            
            pc2_text = "PC2 top features:\n" + "\n".join([
                f"{feat}: {feature_weights.loc[feat, 'PC2']:.3f}" 
                for feat in top_pc2.index
            ])
            
            plt.text(0.02, 0.98, pc1_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
            
            plt.text(0.02, 0.75, pc2_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
            
            # Add anomaly statistics
            anomaly_text = (
                f"Total points: {len(data)}\n"
                f"Anomalies: {len(anomaly_indices)} ({len(anomaly_indices)/len(data):.1%})"
            )
            
            plt.text(0.75, 0.02, anomaly_text, transform=plt.gca().transAxes, 
                   verticalalignment='bottom', horizontalalignment='center',
                   bbox=dict(boxstyle='round', alpha=0.1))
            
        else:
            # For 1-2 features, create a direct scatter plot
            plt.figure(figsize=(10, 8))
            
            # Create a copy with anomaly labels
            viz_data = data.copy()
            viz_data['anomaly'] = 0
            viz_data.iloc[anomaly_indices, -1] = 1
            
            if data.shape[1] == 1:
                # 1D case: scatter plot with anomalies highlighted
                feature = data.columns[0]
                plt.scatter(viz_data.index, viz_data[feature], 
                          c=viz_data['anomaly'].map({0: 'blue', 1: 'red'}),
                          alpha=0.5)
                plt.title(f"Anomaly Detection for {feature} using {method.replace('_', ' ').title()}")
                plt.ylabel(feature)
                plt.xlabel("Data Point Index")
                
            else:
                # 2D case: scatter plot with two features
                feat1, feat2 = data.columns[:2]
                
                # Plot normal points
                normal = viz_data[viz_data['anomaly'] == 0]
                plt.scatter(normal[feat1], normal[feat2], c='blue', marker='o', 
                          alpha=0.5, label='Normal')
                
                # Plot anomalies
                anomalies = viz_data[viz_data['anomaly'] == 1]
                plt.scatter(anomalies[feat1], anomalies[feat2], c='red', marker='x', 
                          s=100, label='Anomaly')
                
                plt.title(f"Anomaly Detection using {method.replace('_', ' ').title()}")
                plt.xlabel(feat1)
                plt.ylabel(feat2)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add anomaly statistics
                anomaly_text = (
                    f"Total points: {len(data)}\n"
                    f"Anomalies: {len(anomaly_indices)} ({len(anomaly_indices)/len(data):.1%})"
                )
                
                plt.text(0.75, 0.02, anomaly_text, transform=plt.gca().transAxes, 
                       verticalalignment='bottom', horizontalalignment='center',
                       bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.tight_layout()
        
        # Save or show the plot
        if output_path:
            # Create a specific filename
            file_name = f"anomalies_{method}.png"
            file_path = os.path.join(output_path, file_name) if os.path.isdir(output_path) else output_path
            
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved anomaly visualization to {file_path}")
        
        plt.close()
        
        # Create feature distributions plot with anomalies highlighted
        self._visualize_feature_distributions(data, anomaly_indices, method, output_path)
    
    def _visualize_feature_distributions(self,
                                       data: pd.DataFrame,
                                       anomaly_indices: np.ndarray,
                                       method: str,
                                       output_path: Optional[str] = None):
        """
        Visualize feature distributions with anomalies highlighted.
        
        Args:
            data: Input DataFrame with metrics as columns
            anomaly_indices: Indices of detected anomalies
            method: Anomaly detection method used
            output_path: Path to save visualization
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available. Skipping visualization.")
            return
        
        # Create a copy with anomaly labels
        viz_data = data.copy()
        viz_data['anomaly'] = 0
        viz_data.iloc[anomaly_indices, -1] = 1
        
        # Select at most 8 features to display
        features = [col for col in data.columns if not col.startswith('derived_')][:8]
        
        if not features:
            features = list(data.columns)[:8]
        
        # Determine the grid layout
        n_features = len(features)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create the figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        fig.suptitle(f"Feature Distributions with Anomalies ({method.replace('_', ' ').title()})", 
                   fontsize=16)
        
        # Convert to 2D array if needed
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # Plot each feature distribution
        for i, feature in enumerate(features):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Get normal and anomaly data
            normal_data = viz_data[viz_data['anomaly'] == 0][feature]
            anomaly_data = viz_data[viz_data['anomaly'] == 1][feature]
            
            # Plot distributions
            sns.histplot(normal_data, ax=ax, color='blue', alpha=0.5, label='Normal', kde=True)
            
            if not anomaly_data.empty:
                # Add anomaly points as rug plot
                sns.rugplot(anomaly_data, ax=ax, color='red', label='Anomalies', height=0.1)
                
                # Add vertical lines for each anomaly
                for value in anomaly_data:
                    ax.axvline(x=value, color='red', linestyle='--', alpha=0.5)
            
            ax.set_title(feature)
            ax.grid(True, alpha=0.3)
            
            # Add z-score ranges
            mean = normal_data.mean()
            std = normal_data.std()
            
            # Draw mean and std bounds
            ax.axvline(x=mean, color='green', linestyle='-', label='Mean')
            ax.axvline(x=mean + 2*std, color='orange', linestyle=':', label='+2σ')
            ax.axvline(x=mean - 2*std, color='orange', linestyle=':', label='-2σ')
            
            # Add legend only to the first plot
            if i == 0:
                ax.legend()
        
        # Hide unused subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the plot
        if output_path:
            # Create a specific filename
            file_name = f"feature_distributions_{method}.png"
            file_path = os.path.join(output_path, file_name) if os.path.isdir(output_path) else output_path
            
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature distributions visualization to {file_path}")
        
        plt.close()
    
    def train_anomaly_model(self,
                           model_name: str,
                           hardware_type: str,
                           metrics: List[str] = ['latency', 'throughput'],
                           time_period_days: int = 30,
                           method: str = 'isolation_forest',
                           contamination: float = 0.05,
                           n_estimators: int = 100,
                           feature_engineering: bool = True,
                           model_path: Optional[str] = None,
                           filter_criteria: Dict[str, Any] = None) -> Dict:
        """
        Train an anomaly detection model for future use.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            metrics: List of metrics to include in the model
            time_period_days: Number of days of data to use for training
            method: Anomaly detection method to use
            contamination: Expected proportion of anomalies
            n_estimators: Number of estimators for ensemble methods
            feature_engineering: Whether to perform feature engineering
            model_path: Path to save the trained model
            filter_criteria: Additional filtering criteria for test results
        
        Returns:
            Dictionary with model training information
        
        Raises:
            ImportError: If required ML libraries are not available
            ValueError: If invalid parameters are provided
            RuntimeError: If model training fails
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for ML model training")
        
        # Validate method
        valid_methods = ['isolation_forest', 'lof', 'ocsvm']
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Valid methods are: {', '.join(valid_methods)}")
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # Build the query
        query = """
        SELECT r.result_id, r.model_name, r.hardware_type, r.batch_size, r.timestamp,
               r.test_name, r.status, m.metric_name, m.metric_value
        FROM test_results r
        JOIN performance_metrics m ON r.result_id = m.result_id
        WHERE r.timestamp >= ?
        AND r.model_name = ?
        AND r.hardware_type = ?
        """
        
        params = [cutoff_str, model_name, hardware_type]
        
        # Add filters for metrics
        if metrics:
            metrics_str = ', '.join([f"'{m}'" for m in metrics])
            query += f" AND m.metric_name IN ({metrics_str})"
        
        # Add additional filters
        if filter_criteria:
            for key, value in filter_criteria.items():
                if isinstance(value, list):
                    placeholders = ', '.join(['?' for _ in value])
                    query += f" AND r.{key} IN ({placeholders})"
                    params.extend(value)
                else:
                    query += f" AND r.{key} = ?"
                    params.append(value)
        
        try:
            # Execute the query
            results = self._conn.execute(query, params).fetchdf()
            
            if results.empty:
                msg = "No data found for the specified criteria"
                logger.warning(msg)
                return {
                    'status': 'error',
                    'message': msg,
                    'model_info': None
                }
            
            # Ensure timestamp is datetime
            results['timestamp'] = pd.to_datetime(results['timestamp'])
            
            # Create the pivot table
            pivot = results.pivot_table(
                index=['result_id', 'timestamp'], 
                columns='metric_name', 
                values='metric_value'
            ).reset_index()
            
            # Sort by timestamp
            pivot = pivot.sort_values('timestamp')
            
            # Drop rows with missing values
            data = pivot.drop(['result_id', 'timestamp'], axis=1).dropna()
            
            if data.empty:
                msg = "No complete data rows found after pivoting"
                logger.warning(msg)
                return {
                    'status': 'error',
                    'message': msg,
                    'model_info': None
                }
            
            # Perform feature engineering if requested
            if feature_engineering:
                data = self._feature_engineering(data)
            
            # Standardize the data
            scaler = StandardScaler()
            X = scaler.fit_transform(data)
            
            # Train the model
            model = None
            
            if method == 'isolation_forest':
                model = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42
                )
                model.fit(X)
                
            elif method == 'lof':
                model = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=contamination,
                    novelty=True  # Required for prediction on new data
                )
                model.fit(X)
                
            elif method == 'ocsvm':
                model = OneClassSVM(
                    kernel='rbf',
                    gamma='auto',
                    nu=contamination
                )
                model.fit(X)
            
            # Store the model information
            model_info = {
                'ml_model': model,
                'scaler': scaler,
                'feature_names': list(data.columns),
                'feature_engineering': feature_engineering,
                'method': method,
                'model_name': model_name,
                'hardware_type': hardware_type,
                'metrics': metrics,
                'contamination': contamination,
                'n_estimators': n_estimators,
                'training_samples': len(data),
                'training_time': datetime.now().isoformat()
            }
            
            # Save the model to file if requested
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info, f)
                logger.info(f"Saved anomaly detection model to {model_path}")
            
            # Store the model in memory
            model_key = f"{model_name}_{hardware_type}_{method}"
            self._trained_models[model_key] = model_info
            
            # Create the result
            result = {
                'status': 'success',
                'method': method,
                'model_name': model_name,
                'hardware_type': hardware_type,
                'metrics': metrics,
                'training_samples': len(data),
                'feature_count': len(data.columns),
                'training_time': datetime.now().isoformat(),
                'model_key': model_key,
                'model_saved': model_path is not None,
                'model_path': model_path
            }
            
            return result
                
        except Exception as e:
            logger.error(f"Error training anomaly model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to train anomaly model: {str(e)}")
    
    def predict_anomalies(self,
                         data: pd.DataFrame,
                         model_name: str,
                         hardware_type: str,
                         method: str = 'isolation_forest',
                         feature_engineering: bool = True,
                         model_path: Optional[str] = None) -> Dict:
        """
        Predict anomalies using a previously trained model.
        
        Args:
            data: DataFrame with metrics as columns
            model_name: Name of the model
            hardware_type: Type of hardware
            method: Anomaly detection method
            feature_engineering: Whether to perform feature engineering
            model_path: Path to load the model from (if not already loaded)
        
        Returns:
            Dictionary with anomaly predictions
        
        Raises:
            ImportError: If required ML libraries are not available
            ValueError: If the model is not found
            RuntimeError: If prediction fails
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for ML-based anomaly detection")
        
        # Try to find the model
        model_key = f"{model_name}_{hardware_type}_{method}"
        model_info = self._trained_models.get(model_key)
        
        # Load the model from file if not found and path is provided
        if model_info is None and model_path:
            try:
                with open(model_path, 'rb') as f:
                    model_info = pickle.load(f)
                self._trained_models[model_key] = model_info
                logger.info(f"Loaded anomaly detection model from {model_path}")
            except Exception as e:
                raise ValueError(f"Failed to load model from {model_path}: {str(e)}")
        
        if model_info is None:
            raise ValueError(f"No trained model found for {model_name} on {hardware_type} using {method}")
        
        # Check if the data has the expected features
        expected_features = set(model_info['feature_names'])
        provided_features = set(data.columns)
        
        # Check for missing features
        missing_features = expected_features - provided_features
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        try:
            # Prepare the data
            # First, ensure we have all required features
            required_features = model_info['feature_names']
            
            # Perform feature engineering if the model was trained with it
            if model_info['feature_engineering'] and feature_engineering:
                data = self._feature_engineering(data)
            
            # Select only the features used during training
            data = data[required_features]
            
            # Scale the data using the model's scaler
            X = model_info['scaler'].transform(data)
            
            # Make predictions
            ml_model = model_info['ml_model']
            predictions = ml_model.predict(X)
            
            # Get anomaly scores
            if method == 'isolation_forest':
                scores = ml_model.decision_function(X)
            elif method == 'lof':
                scores = -ml_model.score_samples(X)
            elif method == 'ocsvm':
                scores = -ml_model.decision_function(X)
            else:
                scores = None
            
            # Find anomalies (predictions == -1)
            anomaly_indices = np.where(predictions == -1)[0]
            
            # Process anomalies
            anomalies = []
            for idx in anomaly_indices:
                anomaly = {
                    'index': int(idx),
                    'features': {feat: float(data.iloc[idx][feat]) for feat in required_features},
                    'score': float(scores[idx]) if scores is not None else None
                }
                anomalies.append(anomaly)
            
            # Create the result
            result = {
                'status': 'success',
                'method': method,
                'model_name': model_name,
                'hardware_type': hardware_type,
                'total_points': len(data),
                'anomalies_found': len(anomaly_indices),
                'anomaly_percentage': len(anomaly_indices) / len(data) * 100 if len(data) > 0 else 0,
                'anomalies': anomalies,
                'timestamp': datetime.now().isoformat(),
                'model_key': model_key
            }
            
            return result
                
        except Exception as e:
            logger.error(f"Error predicting anomalies: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to predict anomalies: {str(e)}")
    
    def load_model(self, model_path: str) -> Dict:
        """
        Load an anomaly detection model from a file.
        
        Args:
            model_path: Path to the model file
        
        Returns:
            Dictionary with model information
        
        Raises:
            ValueError: If the model file is not found or invalid
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)
            
            # Store the model in memory
            model_key = f"{model_info['model_name']}_{model_info['hardware_type']}_{model_info['method']}"
            self._trained_models[model_key] = model_info
            
            # Return model metadata (excluding the actual model)
            result = {
                'status': 'success',
                'method': model_info['method'],
                'model_name': model_info['model_name'],
                'hardware_type': model_info['hardware_type'],
                'metrics': model_info['metrics'],
                'feature_count': len(model_info['feature_names']),
                'training_samples': model_info['training_samples'],
                'training_time': model_info['training_time'],
                'model_key': model_key,
                'model_loaded': True
            }
            
            return result
                
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def save_model(self, model_name: str, hardware_type: str, method: str, model_path: str) -> Dict:
        """
        Save a trained model to a file.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            method: Anomaly detection method
            model_path: Path to save the model
        
        Returns:
            Dictionary with save operation result
        
        Raises:
            ValueError: If the model is not found
        """
        # Try to find the model
        model_key = f"{model_name}_{hardware_type}_{method}"
        model_info = self._trained_models.get(model_key)
        
        if model_info is None:
            raise ValueError(f"No trained model found for {model_name} on {hardware_type} using {method}")
        
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            
            logger.info(f"Saved anomaly detection model to {model_path}")
            
            return {
                'status': 'success',
                'message': f"Model saved to {model_path}",
                'model_key': model_key,
                'model_path': model_path
            }
                
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def evaluate_model(self,
                      model_name: str,
                      hardware_type: str,
                      metrics: List[str] = ['latency', 'throughput'],
                      time_period_days: int = 30,
                      method: str = 'isolation_forest',
                      contamination: float = 0.05,
                      feature_engineering: bool = True,
                      test_size: float = 0.3,
                      n_estimators: int = 100,
                      filter_criteria: Dict[str, Any] = None) -> Dict:
        """
        Evaluate an anomaly detection model using train/test split.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            metrics: List of metrics to include in the model
            time_period_days: Number of days of data to use
            method: Anomaly detection method to use
            contamination: Expected proportion of anomalies
            feature_engineering: Whether to perform feature engineering
            test_size: Proportion of data to use for testing
            n_estimators: Number of estimators for ensemble methods
            filter_criteria: Additional filtering criteria for test results
        
        Returns:
            Dictionary with model evaluation metrics
        
        Raises:
            ImportError: If required ML libraries are not available
            ValueError: If invalid parameters are provided
            RuntimeError: If evaluation fails
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for ML model evaluation")
        
        # Validate method
        valid_methods = ['isolation_forest', 'lof', 'ocsvm']
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Valid methods are: {', '.join(valid_methods)}")
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # Build the query
        query = """
        SELECT r.result_id, r.model_name, r.hardware_type, r.batch_size, r.timestamp,
               r.test_name, r.status, m.metric_name, m.metric_value
        FROM test_results r
        JOIN performance_metrics m ON r.result_id = m.result_id
        WHERE r.timestamp >= ?
        AND r.model_name = ?
        AND r.hardware_type = ?
        """
        
        params = [cutoff_str, model_name, hardware_type]
        
        # Add filters for metrics
        if metrics:
            metrics_str = ', '.join([f"'{m}'" for m in metrics])
            query += f" AND m.metric_name IN ({metrics_str})"
        
        # Add additional filters
        if filter_criteria:
            for key, value in filter_criteria.items():
                if isinstance(value, list):
                    placeholders = ', '.join(['?' for _ in value])
                    query += f" AND r.{key} IN ({placeholders})"
                    params.extend(value)
                else:
                    query += f" AND r.{key} = ?"
                    params.append(value)
        
        try:
            # Execute the query
            results = self._conn.execute(query, params).fetchdf()
            
            if results.empty:
                msg = "No data found for the specified criteria"
                logger.warning(msg)
                return {
                    'status': 'error',
                    'message': msg,
                    'evaluation': None
                }
            
            # Ensure timestamp is datetime
            results['timestamp'] = pd.to_datetime(results['timestamp'])
            
            # Create the pivot table
            pivot = results.pivot_table(
                index=['result_id', 'timestamp'], 
                columns='metric_name', 
                values='metric_value'
            ).reset_index()
            
            # Sort by timestamp
            pivot = pivot.sort_values('timestamp')
            
            # Drop rows with missing values
            data = pivot.drop(['result_id', 'timestamp'], axis=1).dropna()
            
            if data.empty:
                msg = "No complete data rows found after pivoting"
                logger.warning(msg)
                return {
                    'status': 'error',
                    'message': msg,
                    'evaluation': None
                }
            
            # Perform feature engineering if requested
            if feature_engineering:
                data = self._feature_engineering(data)
            
            # Split the data into normal and anomalous
            # For evaluation, we'll introduce some anomalies using extreme values
            
            # Compute z-scores for each feature
            z_scores = np.abs(zscore(data, axis=0, nan_policy='omit'))
            
            # Consider a point anomalous if any feature has z-score > 3
            anomaly_mask = np.any(z_scores > 3, axis=1)
            
            # Create labels (1 for normal, -1 for anomaly)
            y_true = np.ones(len(data))
            y_true[anomaly_mask] = -1
            
            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                data, 
                y_true, 
                test_size=test_size, 
                random_state=42,
                stratify=y_true
            )
            
            # Standardize the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train the model
            model = None
            
            if method == 'isolation_forest':
                model = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42
                )
                model.fit(X_train_scaled)
                
            elif method == 'lof':
                model = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=contamination,
                    novelty=True
                )
                model.fit(X_train_scaled)
                
            elif method == 'ocsvm':
                model = OneClassSVM(
                    kernel='rbf',
                    gamma='auto',
                    nu=contamination
                )
                model.fit(X_train_scaled)
            
            # Evaluate the model
            y_pred = model.predict(X_test_scaled)
            
            # Calculate evaluation metrics
            # Convert labels to binary format for sklearn metrics
            y_test_binary = np.where(y_test == -1, 1, 0)
            y_pred_binary = np.where(y_pred == -1, 1, 0)
            
            precision = precision_score(y_test_binary, y_pred_binary)
            recall = recall_score(y_test_binary, y_pred_binary)
            f1 = f1_score(y_test_binary, y_pred_binary)
            
            # Calculate confusion matrix
            tp = np.sum((y_test_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_test_binary == 0) & (y_pred_binary == 1))
            tn = np.sum((y_test_binary == 0) & (y_pred_binary == 0))
            fn = np.sum((y_test_binary == 1) & (y_pred_binary == 0))
            
            # Create the evaluation result
            evaluation = {
                'method': method,
                'model_name': model_name,
                'hardware_type': hardware_type,
                'metrics': metrics,
                'feature_count': len(data.columns),
                'total_samples': len(data),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_anomalies': np.sum(y_train == -1),
                'test_anomalies': np.sum(y_test == -1),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': {
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn)
                },
                'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
                'timestamp': datetime.now().isoformat()
            }
            
            # Create the result
            result = {
                'status': 'success',
                'evaluation': evaluation
            }
            
            return result
                
        except Exception as e:
            logger.error(f"Error evaluating anomaly model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to evaluate anomaly model: {str(e)}")
    
    def list_trained_models(self) -> List[Dict]:
        """
        List all trained anomaly detection models.
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        
        for model_key, model_info in self._trained_models.items():
            model_data = {
                'model_key': model_key,
                'model_name': model_info['model_name'],
                'hardware_type': model_info['hardware_type'],
                'method': model_info['method'],
                'metrics': model_info['metrics'],
                'feature_count': len(model_info['feature_names']),
                'training_samples': model_info['training_samples'],
                'training_time': model_info['training_time']
            }
            models.append(model_data)
        
        return models
    
    def delete_model(self, model_name: str, hardware_type: str, method: str) -> Dict:
        """
        Delete a trained model from memory.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            method: Anomaly detection method
        
        Returns:
            Dictionary with deletion result
        
        Raises:
            ValueError: If the model is not found
        """
        model_key = f"{model_name}_{hardware_type}_{method}"
        
        if model_key not in self._trained_models:
            raise ValueError(f"No trained model found for {model_name} on {hardware_type} using {method}")
        
        del self._trained_models[model_key]
        
        return {
            'status': 'success',
            'message': f"Model {model_key} deleted from memory"
        }
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
    
    def __del__(self):
        """Destructor to ensure the database connection is closed."""
        self.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    try:
        detector = MLAnomalyDetector(db_path="benchmark_db.duckdb")
        
        # Example anomaly detection
        anomalies = detector.detect_anomalies(
            model_name='bert-base-uncased',
            hardware_type='cuda',
            metrics=['latency', 'throughput', 'memory_usage'],
            method='isolation_forest',
            visualize=True,
            output_path='anomalies.png'
        )
        
        print(f"Found {len(anomalies['anomalies'])} anomalies")
        
        # Example model training
        model_info = detector.train_anomaly_model(
            model_name='bert-base-uncased',
            hardware_type='cuda',
            metrics=['latency', 'throughput', 'memory_usage'],
            method='isolation_forest',
            model_path='models/anomaly_detector.pkl'
        )
        
        print(f"Trained model with {model_info['training_samples']} samples")
        
        # Example model evaluation
        evaluation = detector.evaluate_model(
            model_name='bert-base-uncased',
            hardware_type='cuda',
            metrics=['latency', 'throughput', 'memory_usage'],
            method='isolation_forest'
        )
        
        print(f"Model evaluation: F1 score = {evaluation['evaluation']['f1_score']:.2f}")
        
    except Exception as e:
        logging.error(f"Error in example: {str(e)}")
    finally:
        if 'detector' in locals():
            detector.close()