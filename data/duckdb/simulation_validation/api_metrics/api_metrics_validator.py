"""
API Metrics Validator for the DuckDB API in IPFS Accelerate Framework.

This module provides validation tools for API performance metrics, including
time series validation, anomaly detection validation, and predictive analytics validation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .api_metrics_repository import DuckDBAPIMetricsRepository

logger = logging.getLogger(__name__)

class APIMetricsValidator:
    """
    Validator for API performance metrics and predictions.
    
    This class provides methods for validating API metrics data quality,
    prediction accuracy, anomaly detection effectiveness, and recommendation relevance.
    """
    
    def __init__(
        self,
        repository: Optional[DuckDBAPIMetricsRepository] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the API Metrics Validator.
        
        Args:
            repository: Optional DuckDBAPIMetricsRepository instance
            config: Additional configuration options
        """
        self.repository = repository
        self.config = config or {}
        
        # Initialize configuration with defaults
        self.validation_thresholds = self.config.get('validation_thresholds', {
            'completeness': 0.95,  # 95% of expected fields should be present
            'accuracy': 0.8,       # 80% accuracy for predictions
            'anomaly_precision': 0.7,  # 70% precision for anomaly detection
            'recommendation_relevance': 0.6  # 60% relevance for recommendations
        })
    
    def validate_data_quality(
        self,
        metrics: Optional[List[Dict[str, Any]]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate the quality of API metrics data.
        
        Args:
            metrics: Optional list of metrics to validate (if None, fetched from repository)
            start_time: Optional start time filter for fetching metrics
            end_time: Optional end time filter for fetching metrics
            endpoint: Optional endpoint filter for fetching metrics
            model: Optional model filter for fetching metrics
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Fetch metrics if not provided
            if metrics is None and self.repository is not None:
                metrics = self.repository.get_metrics(
                    start_time=start_time,
                    end_time=end_time,
                    endpoint=endpoint,
                    model=model,
                    limit=10000
                )
            
            if not metrics:
                return {
                    'status': 'error',
                    'message': 'No metrics available for validation',
                    'completeness': 0.0,
                    'consistency': 0.0,
                    'validity': 0.0,
                    'timeliness': 0.0,
                    'overall_quality': 0.0
                }
            
            # Check completeness (required fields present)
            required_fields = ['timestamp', 'endpoint', 'model', 'response_time', 'status_code', 'success']
            completeness_scores = []
            
            for metric in metrics:
                present_fields = sum(1 for field in required_fields if field in metric and metric[field] is not None)
                completeness_scores.append(present_fields / len(required_fields))
            
            avg_completeness = sum(completeness_scores) / len(completeness_scores)
            
            # Check consistency (values within expected ranges)
            consistency_scores = []
            
            for metric in metrics:
                score = 1.0
                
                # Response time should be positive
                if 'response_time' in metric and metric['response_time'] is not None:
                    if metric['response_time'] < 0:
                        score -= 0.2
                    elif metric['response_time'] > 60:  # Unusually high response time
                        score -= 0.1
                
                # Status code should be valid
                if 'status_code' in metric and metric['status_code'] is not None:
                    if not (100 <= metric['status_code'] < 600):
                        score -= 0.2
                
                # Token count should be non-negative
                if 'tokens' in metric and metric['tokens'] is not None:
                    if metric['tokens'] < 0:
                        score -= 0.1
                
                consistency_scores.append(max(0.0, score))
            
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            
            # Check validity (logical consistency between fields)
            validity_scores = []
            
            for metric in metrics:
                score = 1.0
                
                # If success is False, should have a non-200 status code
                if 'success' in metric and 'status_code' in metric:
                    if metric['success'] is False and metric['status_code'] == 200:
                        score -= 0.5
                    elif metric['success'] is True and (metric['status_code'] < 200 or metric['status_code'] >= 300):
                        score -= 0.5
                
                # If error_message is present, success should be False
                if 'error_message' in metric and 'success' in metric:
                    if metric['error_message'] and metric['success'] is True:
                        score -= 0.3
                
                validity_scores.append(max(0.0, score))
            
            avg_validity = sum(validity_scores) / len(validity_scores)
            
            # Check timeliness (timestamps are reasonable)
            timeliness_scores = []
            now = datetime.now()
            max_age = timedelta(days=365)  # Max 1 year old
            future_threshold = timedelta(hours=1)  # Max 1 hour in future (allowing for clock skew)
            
            for metric in metrics:
                score = 1.0
                
                if 'timestamp' in metric and metric['timestamp'] is not None:
                    timestamp = metric['timestamp']
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except:
                            score -= 0.5
                            timestamp = None
                    
                    if timestamp:
                        age = now - timestamp
                        if age > max_age:
                            score -= 0.5
                        elif timestamp > now + future_threshold:
                            score -= 0.8
                else:
                    score -= 0.5
                
                timeliness_scores.append(max(0.0, score))
            
            avg_timeliness = sum(timeliness_scores) / len(timeliness_scores)
            
            # Calculate overall quality score (weighted average)
            weights = {
                'completeness': 0.35,
                'consistency': 0.25,
                'validity': 0.25,
                'timeliness': 0.15
            }
            
            overall_quality = (
                avg_completeness * weights['completeness'] +
                avg_consistency * weights['consistency'] +
                avg_validity * weights['validity'] +
                avg_timeliness * weights['timeliness']
            )
            
            return {
                'status': 'success',
                'completeness': avg_completeness,
                'consistency': avg_consistency,
                'validity': avg_validity,
                'timeliness': avg_timeliness,
                'overall_quality': overall_quality,
                'sample_size': len(metrics),
                'threshold_met': overall_quality >= self.validation_thresholds['completeness'],
                'recommendations': self._generate_data_quality_recommendations(
                    avg_completeness, avg_consistency, avg_validity, avg_timeliness
                )
            }
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error validating data quality: {str(e)}",
                'completeness': 0.0,
                'consistency': 0.0,
                'validity': 0.0,
                'timeliness': 0.0,
                'overall_quality': 0.0
            }
    
    def _generate_data_quality_recommendations(
        self,
        completeness: float,
        consistency: float,
        validity: float,
        timeliness: float
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for improving data quality.
        
        Args:
            completeness: Completeness score
            consistency: Consistency score
            validity: Validity score
            timeliness: Timeliness score
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if completeness < 0.9:
            recommendations.append({
                'issue': 'Low Data Completeness',
                'description': 'Required fields are missing in many records',
                'recommendation': 'Ensure all API calls include required fields: timestamp, endpoint, model, response_time, status_code, success',
                'priority': 'high' if completeness < 0.7 else 'medium'
            })
        
        if consistency < 0.9:
            recommendations.append({
                'issue': 'Data Consistency Issues',
                'description': 'Some field values are outside expected ranges',
                'recommendation': 'Validate field values: response_time should be positive, status_code should be valid HTTP code, tokens should be non-negative',
                'priority': 'high' if consistency < 0.7 else 'medium'
            })
        
        if validity < 0.9:
            recommendations.append({
                'issue': 'Logical Inconsistencies',
                'description': 'Logical relationships between fields are inconsistent',
                'recommendation': 'Ensure logical consistency: success=false should have status_code outside 200-299, error_message should only be present with success=false',
                'priority': 'high' if validity < 0.7 else 'medium'
            })
        
        if timeliness < 0.9:
            recommendations.append({
                'issue': 'Timestamp Issues',
                'description': 'Some records have unusually old or future timestamps',
                'recommendation': 'Check system clock synchronization and timestamp formatting',
                'priority': 'medium' if timeliness < 0.7 else 'low'
            })
        
        return recommendations
    
    def validate_prediction_accuracy(
        self,
        predictions: Optional[List[Dict[str, Any]]] = None,
        actual_metrics: Optional[List[Dict[str, Any]]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        prediction_type: str = 'response_time'
    ) -> Dict[str, Any]:
        """
        Validate the accuracy of API performance predictions.
        
        Args:
            predictions: Optional list of predictions to validate (if None, fetched from repository)
            actual_metrics: Optional list of actual metrics to compare against
            start_time: Optional start time filter for fetching metrics
            end_time: Optional end time filter for fetching metrics
            endpoint: Optional endpoint filter for fetching metrics
            model: Optional model filter for fetching metrics
            prediction_type: Type of prediction to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            if self.repository is None:
                return {
                    'status': 'error',
                    'message': 'Repository not available for validation',
                    'mae': None,
                    'rmse': None,
                    'r2': None,
                    'accuracy': 0.0
                }
            
            # Fetch predictions if not provided
            if predictions is None:
                predictions = self.repository.get_predictions(
                    start_time=start_time,
                    end_time=end_time,
                    endpoint=endpoint,
                    model=model,
                    prediction_type=prediction_type,
                    limit=1000
                )
            
            if not predictions:
                return {
                    'status': 'error',
                    'message': 'No predictions available for validation',
                    'mae': None,
                    'rmse': None,
                    'r2': None,
                    'accuracy': 0.0
                }
            
            # Group predictions by endpoint and model
            prediction_groups = {}
            for pred in predictions:
                key = (pred.get('endpoint', ''), pred.get('model', ''))
                if key not in prediction_groups:
                    prediction_groups[key] = []
                prediction_groups[key].append(pred)
            
            # Validate each group separately
            results = []
            for (pred_endpoint, pred_model), preds in prediction_groups.items():
                # Fetch actual metrics for comparison
                if actual_metrics is None:
                    # Determine time range from predictions
                    pred_timestamps = [p['timestamp'] for p in preds if 'timestamp' in p]
                    earliest_pred = min(pred_timestamps) if pred_timestamps else datetime.now() - timedelta(days=30)
                    latest_pred = max(pred_timestamps) if pred_timestamps else datetime.now()
                    
                    # Get actual metrics from earlier period for comparison
                    actual_metrics = self.repository.get_metrics_aggregated(
                        start_time=earliest_pred - timedelta(days=30),
                        end_time=latest_pred,
                        endpoint=pred_endpoint,
                        model=pred_model,
                        group_by='hour',
                        metric_type=prediction_type
                    )
                
                if not actual_metrics:
                    continue
                
                # Convert to pandas DataFrames for easier comparison
                pred_df = pd.DataFrame([{
                    'timestamp': p['timestamp'],
                    'endpoint': p.get('endpoint', ''),
                    'model': p.get('model', ''),
                    'predicted_value': p.get('predicted_value', 0.0),
                    'horizon': p.get('horizon', 24)
                } for p in preds])
                
                actual_df = pd.DataFrame([{
                    'timestamp': a['timestamp'],
                    'endpoint': a.get('endpoint', ''),
                    'model': a.get('model', ''),
                    'actual_value': a.get('avg_value', 0.0)
                } for a in actual_metrics])
                
                # Merge predictions with actuals based on timestamp + horizon
                pred_df['target_timestamp'] = pred_df.apply(
                    lambda row: row['timestamp'] + timedelta(hours=row['horizon']), axis=1
                )
                
                # Round timestamps to nearest hour for matching
                pred_df['target_timestamp'] = pred_df['target_timestamp'].dt.floor('H')
                actual_df['timestamp'] = actual_df['timestamp'].dt.floor('H')
                
                # Merge on target timestamp
                merged_df = pd.merge(
                    pred_df,
                    actual_df,
                    left_on=['target_timestamp', 'endpoint', 'model'],
                    right_on=['timestamp', 'endpoint', 'model'],
                    suffixes=('_pred', '_actual'),
                    how='inner'
                )
                
                if len(merged_df) == 0:
                    continue
                
                # Calculate error metrics
                y_true = merged_df['actual_value'].values
                y_pred = merged_df['predicted_value'].values
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
                
                # Calculate accuracy (1 - normalized MAE)
                max_value = max(y_true)
                min_value = min(y_true)
                value_range = max_value - min_value if max_value > min_value else 1.0
                normalized_mae = mae / value_range
                accuracy = max(0.0, 1.0 - normalized_mae)
                
                results.append({
                    'endpoint': pred_endpoint,
                    'model': pred_model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'accuracy': accuracy,
                    'sample_size': len(merged_df),
                    'threshold_met': accuracy >= self.validation_thresholds['accuracy']
                })
            
            if not results:
                return {
                    'status': 'error',
                    'message': 'No matching actual metrics found for predictions',
                    'mae': None,
                    'rmse': None,
                    'r2': None,
                    'accuracy': 0.0
                }
            
            # Calculate aggregated results
            avg_mae = sum(r['mae'] for r in results) / len(results)
            avg_rmse = sum(r['rmse'] for r in results) / len(results)
            avg_r2 = sum(r['r2'] for r in results) / len(results)
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            total_samples = sum(r['sample_size'] for r in results)
            
            # Generate recommendations
            recommendations = self._generate_prediction_recommendations(results)
            
            return {
                'status': 'success',
                'mae': avg_mae,
                'rmse': avg_rmse,
                'r2': avg_r2,
                'accuracy': avg_accuracy,
                'sample_size': total_samples,
                'threshold_met': avg_accuracy >= self.validation_thresholds['accuracy'],
                'group_results': results,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Error validating prediction accuracy: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error validating prediction accuracy: {str(e)}",
                'mae': None,
                'rmse': None,
                'r2': None,
                'accuracy': 0.0
            }
    
    def _generate_prediction_recommendations(
        self,
        group_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for improving prediction accuracy.
        
        Args:
            group_results: Accuracy results for different endpoint/model groups
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Identify groups with poor accuracy
        low_accuracy_groups = [g for g in group_results if g['accuracy'] < self.validation_thresholds['accuracy']]
        
        if low_accuracy_groups:
            for group in low_accuracy_groups:
                endpoint = group['endpoint']
                model = group['model']
                accuracy = group['accuracy']
                
                recommendations.append({
                    'issue': f'Low Prediction Accuracy for {endpoint}/{model}',
                    'description': f'Prediction accuracy is {accuracy:.2f}, below the threshold of {self.validation_thresholds["accuracy"]:.2f}',
                    'recommendation': 'Consider retraining the prediction model with more recent data or using a different algorithm',
                    'priority': 'high' if accuracy < 0.5 else 'medium'
                })
        
        # Check for overfitting (high R² but poor accuracy)
        potential_overfitting = [g for g in group_results if g['r2'] > 0.9 and g['accuracy'] < 0.7]
        
        if potential_overfitting:
            recommendations.append({
                'issue': 'Potential Model Overfitting',
                'description': 'High R² values combined with lower prediction accuracy suggests overfitting',
                'recommendation': 'Simplify the prediction model or use regularization techniques to reduce overfitting',
                'priority': 'medium'
            })
        
        # Check for small sample sizes
        small_samples = [g for g in group_results if g['sample_size'] < 30]
        
        if small_samples:
            recommendations.append({
                'issue': 'Insufficient Validation Samples',
                'description': 'Some endpoint/model combinations have too few samples for reliable validation',
                'recommendation': 'Collect more historical data or adjust the prediction horizon to ensure adequate validation samples',
                'priority': 'medium'
            })
        
        # General recommendations
        if len(group_results) > 0:
            avg_accuracy = sum(g['accuracy'] for g in group_results) / len(group_results)
            
            if avg_accuracy < 0.7:
                recommendations.append({
                    'issue': 'Generally Low Prediction Accuracy',
                    'description': f'Average prediction accuracy across all groups is {avg_accuracy:.2f}',
                    'recommendation': 'Review the overall prediction methodology, consider using ensemble methods or different features',
                    'priority': 'high' if avg_accuracy < 0.5 else 'medium'
                })
        
        return recommendations
    
    def validate_anomaly_detection(
        self,
        anomalies: Optional[List[Dict[str, Any]]] = None,
        actual_metrics: Optional[List[Dict[str, Any]]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        metric_type: Optional[str] = None,
        min_severity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate the effectiveness of anomaly detection.
        
        Args:
            anomalies: Optional list of anomalies to validate (if None, fetched from repository)
            actual_metrics: Optional list of actual metrics to compare against
            start_time: Optional start time filter for fetching metrics
            end_time: Optional end time filter for fetching metrics
            endpoint: Optional endpoint filter for fetching metrics
            model: Optional model filter for fetching metrics
            metric_type: Optional metric type filter for fetching metrics
            min_severity: Minimum severity threshold for anomalies
            
        Returns:
            Dictionary with validation results
        """
        try:
            if self.repository is None:
                return {
                    'status': 'error',
                    'message': 'Repository not available for validation',
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'effectiveness': 0.0
                }
            
            # Fetch anomalies if not provided
            if anomalies is None:
                anomalies = self.repository.get_anomalies(
                    start_time=start_time,
                    end_time=end_time,
                    endpoint=endpoint,
                    model=model,
                    metric_type=metric_type,
                    min_severity=min_severity,
                    limit=1000
                )
            
            if not anomalies:
                return {
                    'status': 'error',
                    'message': 'No anomalies available for validation',
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'effectiveness': 0.0
                }
            
            # Since we don't have ground truth for anomalies, we'll use statistical validation
            # This is an approximation - in a real system, you'd want human-labeled validation data
            
            # Group anomalies by endpoint, model, and metric_type
            anomaly_groups = {}
            for anomaly in anomalies:
                key = (
                    anomaly.get('endpoint', ''),
                    anomaly.get('model', ''),
                    anomaly.get('metric_type', '')
                )
                if key not in anomaly_groups:
                    anomaly_groups[key] = []
                anomaly_groups[key].append(anomaly)
            
            # Validate each group separately
            results = []
            for (anom_endpoint, anom_model, anom_metric_type), anoms in anomaly_groups.items():
                # Fetch actual metrics for comparison if not provided
                if actual_metrics is None:
                    # Determine time range from anomalies
                    anom_timestamps = [a['timestamp'] for a in anoms if 'timestamp' in a]
                    earliest_anom = min(anom_timestamps) if anom_timestamps else datetime.now() - timedelta(days=30)
                    latest_anom = max(anom_timestamps) if anom_timestamps else datetime.now()
                    
                    # Get actual metrics spanning the anomaly period
                    buffer = timedelta(days=1)  # Add buffer around anomaly period
                    metrics = self.repository.get_metrics(
                        start_time=earliest_anom - buffer,
                        end_time=latest_anom + buffer,
                        endpoint=anom_endpoint,
                        model=anom_model,
                        limit=10000
                    )
                else:
                    # Filter provided metrics for this group
                    metrics = [
                        m for m in actual_metrics
                        if m.get('endpoint') == anom_endpoint and m.get('model') == anom_model
                    ]
                
                if not metrics:
                    continue
                
                # Convert metrics to pandas DataFrame
                metrics_df = pd.DataFrame([{
                    'timestamp': m['timestamp'],
                    'value': m.get(anom_metric_type, m.get('response_time', 0.0))
                } for m in metrics if 'timestamp' in m])
                
                # Sort by timestamp
                metrics_df = metrics_df.sort_values('timestamp')
                
                # Use statistical methods to identify actual anomalies
                # Here we use a simple Z-score method, but more sophisticated methods could be used
                metrics_df['z_score'] = (metrics_df['value'] - metrics_df['value'].mean()) / metrics_df['value'].std()
                metrics_df['is_statistical_anomaly'] = abs(metrics_df['z_score']) > 3.0
                
                # Mark time windows containing detected anomalies
                anom_windows = []
                window_size = timedelta(minutes=30)  # Time window around each anomaly
                
                for anomaly in anoms:
                    if 'timestamp' in anomaly:
                        anom_windows.append((
                            anomaly['timestamp'] - window_size,
                            anomaly['timestamp'] + window_size
                        ))
                
                # Check if each statistical anomaly is within a detected anomaly window
                true_positives = 0
                for idx, row in metrics_df[metrics_df['is_statistical_anomaly']].iterrows():
                    timestamp = row['timestamp']
                    for window_start, window_end in anom_windows:
                        if window_start <= timestamp <= window_end:
                            true_positives += 1
                            break
                
                # Calculate precision, recall, and F1 score
                predicted_positives = len(anoms)
                actual_positives = metrics_df['is_statistical_anomaly'].sum()
                
                if predicted_positives > 0:
                    precision = true_positives / predicted_positives
                else:
                    precision = 0.0
                
                if actual_positives > 0:
                    recall = true_positives / actual_positives
                else:
                    recall = 0.0
                
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0.0
                
                # Calculate overall effectiveness score (weighted F1)
                effectiveness = f1_score
                
                results.append({
                    'endpoint': anom_endpoint,
                    'model': anom_model,
                    'metric_type': anom_metric_type,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'effectiveness': effectiveness,
                    'detected_anomalies': len(anoms),
                    'statistical_anomalies': int(actual_positives),
                    'true_positives': true_positives,
                    'threshold_met': precision >= self.validation_thresholds['anomaly_precision']
                })
            
            if not results:
                return {
                    'status': 'error',
                    'message': 'No matching actual metrics found for anomalies',
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'effectiveness': 0.0
                }
            
            # Calculate aggregated results
            avg_precision = sum(r['precision'] for r in results) / len(results)
            avg_recall = sum(r['recall'] for r in results) / len(results)
            avg_f1 = sum(r['f1_score'] for r in results) / len(results)
            avg_effectiveness = sum(r['effectiveness'] for r in results) / len(results)
            
            # Generate recommendations
            recommendations = self._generate_anomaly_recommendations(results)
            
            return {
                'status': 'success',
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'effectiveness': avg_effectiveness,
                'threshold_met': avg_precision >= self.validation_thresholds['anomaly_precision'],
                'group_results': results,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Error validating anomaly detection: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error validating anomaly detection: {str(e)}",
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'effectiveness': 0.0
            }
    
    def _generate_anomaly_recommendations(
        self,
        group_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for improving anomaly detection.
        
        Args:
            group_results: Effectiveness results for different endpoint/model/metric groups
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Identify groups with low precision (too many false positives)
        low_precision_groups = [g for g in group_results if g['precision'] < 0.6]
        
        if low_precision_groups:
            for group in low_precision_groups:
                endpoint = group['endpoint']
                model = group['model']
                metric_type = group['metric_type']
                
                recommendations.append({
                    'issue': f'Low Anomaly Detection Precision for {endpoint}/{model}/{metric_type}',
                    'description': f'Precision is {group["precision"]:.2f}, indicating many false positives',
                    'recommendation': 'Increase anomaly detection thresholds or use more contextual information to reduce false positives',
                    'priority': 'high' if group['precision'] < 0.4 else 'medium'
                })
        
        # Identify groups with low recall (missing many anomalies)
        low_recall_groups = [g for g in group_results if g['recall'] < 0.6]
        
        if low_recall_groups:
            for group in low_recall_groups:
                endpoint = group['endpoint']
                model = group['model']
                metric_type = group['metric_type']
                
                recommendations.append({
                    'issue': f'Low Anomaly Detection Recall for {endpoint}/{model}/{metric_type}',
                    'description': f'Recall is {group["recall"]:.2f}, indicating many missed anomalies',
                    'recommendation': 'Lower anomaly detection thresholds or use multiple detection algorithms to identify different types of anomalies',
                    'priority': 'high' if group['recall'] < 0.4 else 'medium'
                })
        
        # Check for statistical anomalies vs. detected anomalies discrepancy
        for group in group_results:
            if group['statistical_anomalies'] > 0 and group['detected_anomalies'] > 0:
                ratio = group['detected_anomalies'] / group['statistical_anomalies']
                
                if ratio > 3.0:
                    recommendations.append({
                        'issue': f'Over-detection of Anomalies for {group["endpoint"]}/{group["model"]}/{group["metric_type"]}',
                        'description': f'Detected {group["detected_anomalies"]} anomalies but only {group["statistical_anomalies"]} statistical anomalies',
                        'recommendation': 'Review anomaly detection sensitivity; current settings may be too sensitive',
                        'priority': 'medium'
                    })
                elif ratio < 0.3:
                    recommendations.append({
                        'issue': f'Under-detection of Anomalies for {group["endpoint"]}/{group["model"]}/{group["metric_type"]}',
                        'description': f'Detected only {group["detected_anomalies"]} anomalies but found {group["statistical_anomalies"]} statistical anomalies',
                        'recommendation': 'Consider using more sensitive anomaly detection settings',
                        'priority': 'medium'
                    })
        
        # General recommendations
        if len(group_results) > 0:
            avg_effectiveness = sum(g['effectiveness'] for g in group_results) / len(group_results)
            
            if avg_effectiveness < 0.6:
                recommendations.append({
                    'issue': 'Generally Low Anomaly Detection Effectiveness',
                    'description': f'Average effectiveness across all groups is {avg_effectiveness:.2f}',
                    'recommendation': 'Consider using multiple complementary anomaly detection techniques, such as combining statistical methods with machine learning approaches',
                    'priority': 'high' if avg_effectiveness < 0.4 else 'medium'
                })
        
        return recommendations
    
    def validate_recommendation_relevance(
        self,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        recommendation_type: Optional[str] = None,
        min_priority: int = 0
    ) -> Dict[str, Any]:
        """
        Validate the relevance of API performance recommendations.
        
        Args:
            recommendations: Optional list of recommendations to validate (if None, fetched from repository)
            endpoint: Optional endpoint filter for fetching recommendations
            model: Optional model filter for fetching recommendations
            recommendation_type: Optional recommendation type filter
            min_priority: Minimum priority threshold
            
        Returns:
            Dictionary with validation results
        """
        try:
            if self.repository is None:
                return {
                    'status': 'error',
                    'message': 'Repository not available for validation',
                    'relevance_score': 0.0,
                    'actionability_score': 0.0,
                    'impact_coverage': 0.0,
                    'overall_quality': 0.0
                }
            
            # Fetch recommendations if not provided
            if recommendations is None:
                recommendations = self.repository.get_recommendations(
                    endpoint=endpoint,
                    model=model,
                    recommendation_type=recommendation_type,
                    min_priority=min_priority,
                    limit=100
                )
            
            if not recommendations:
                return {
                    'status': 'error',
                    'message': 'No recommendations available for validation',
                    'relevance_score': 0.0,
                    'actionability_score': 0.0,
                    'impact_coverage': 0.0,
                    'overall_quality': 0.0
                }
            
            # Validate relevance based on metrics data
            # For each recommendation, check if there's supporting evidence in the metrics
            
            # Group recommendations by endpoint and model
            recommendation_groups = {}
            for rec in recommendations:
                key = (rec.get('endpoint', ''), rec.get('model', ''))
                if key not in recommendation_groups:
                    recommendation_groups[key] = []
                recommendation_groups[key].append(rec)
            
            # Evaluate each group
            results = []
            for (rec_endpoint, rec_model), recs in recommendation_groups.items():
                # Fetch recent metrics for this endpoint/model
                recent_metrics = None
                if self.repository is not None:
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=30)
                    
                    recent_metrics = self.repository.get_metrics_aggregated(
                        start_time=start_time,
                        end_time=end_time,
                        endpoint=rec_endpoint,
                        model=rec_model,
                        group_by='day'
                    )
                
                # Fetch recent anomalies for this endpoint/model
                recent_anomalies = None
                if self.repository is not None:
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=30)
                    
                    recent_anomalies = self.repository.get_anomalies(
                        start_time=start_time,
                        end_time=end_time,
                        endpoint=rec_endpoint,
                        model=rec_model,
                        limit=100
                    )
                
                # Evaluate relevance based on available data
                relevance_scores = []
                actionability_scores = []
                impact_scores = []
                
                for rec in recs:
                    # Initialize scores
                    relevance = 0.5  # Start at neutral relevance
                    actionability = 0.5  # Start at neutral actionability
                    
                    # Relevance factors
                    rec_type = rec.get('recommendation_type', '')
                    
                    # Check if metrics data supports this recommendation
                    if recent_metrics:
                        # For response time issues
                        if rec_type in ['caching', 'model_switching', 'performance_optimization']:
                            # Check if response times are high or increasing
                            response_times = [m.get('avg_value', 0) for m in recent_metrics]
                            if response_times and max(response_times) > 2.0:
                                relevance += 0.2
                            
                            # Check for trend
                            if len(response_times) > 5:
                                if response_times[-1] > response_times[0] * 1.2:  # 20% increase
                                    relevance += 0.1
                        
                        # For reliability issues
                        if rec_type in ['reliability_improvement', 'rate_limiting', 'error_handling']:
                            # Check success rates
                            success_rates = [m.get('success_rate', 1.0) for m in recent_metrics]
                            if success_rates and min(success_rates) < 0.95:
                                relevance += 0.2
                            
                            # Check for trend
                            if len(success_rates) > 5:
                                if success_rates[-1] < success_rates[0] * 0.9:  # 10% decrease
                                    relevance += 0.1
                    
                    # Check if anomalies support this recommendation
                    if recent_anomalies:
                        # Count relevant anomalies
                        relevant_anomalies = [
                            a for a in recent_anomalies
                            if (rec_type == 'performance_optimization' and a.get('metric_type') == 'response_time') or
                               (rec_type == 'reliability_improvement' and a.get('metric_type') == 'error_rate') or
                               (rec_type == 'rate_limiting' and a.get('metric_type') == 'throughput')
                        ]
                        
                        if len(relevant_anomalies) > 0:
                            relevance += 0.1 * min(len(relevant_anomalies), 3) / 3
                    
                    # Actionability factors
                    if 'implementation_cost' in rec:
                        # Lower cost = higher actionability
                        cost = rec['implementation_cost']
                        if isinstance(cost, (int, float)):
                            # Normalize to 0-1 scale (assumes costs are 1-5)
                            normalized_cost = max(0, min(1, (5 - cost) / 4))
                            actionability += 0.2 * normalized_cost
                    
                    if 'description' in rec:
                        # More detailed descriptions are more actionable
                        desc = rec['description']
                        if isinstance(desc, str):
                            if len(desc) > 100:
                                actionability += 0.1
                            if 'how to' in desc.lower() or 'steps' in desc.lower():
                                actionability += 0.1
                    
                    # Impact score
                    impact = rec.get('impact_score', 0.5)
                    
                    # Normalize scores to 0-1 range
                    relevance = max(0.0, min(1.0, relevance))
                    actionability = max(0.0, min(1.0, actionability))
                    
                    relevance_scores.append(relevance)
                    actionability_scores.append(actionability)
                    impact_scores.append(impact)
                
                # Calculate averages
                avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
                avg_actionability = sum(actionability_scores) / len(actionability_scores) if actionability_scores else 0.0
                
                # Calculate impact coverage (% of issues addressed by recommendations)
                impact_coverage = 0.0
                if recent_anomalies and recent_metrics:
                    # Count distinct issues in metrics and anomalies
                    distinct_issues = set()
                    
                    for anomaly in recent_anomalies:
                        metric_type = anomaly.get('metric_type', '')
                        if metric_type:
                            distinct_issues.add(f"anomaly_{metric_type}")
                    
                    # Check response time issues in metrics
                    avg_response_times = [m.get('avg_value', 0) for m in recent_metrics if m.get('metric_type') == 'response_time']
                    if avg_response_times and max(avg_response_times) > 2.0:
                        distinct_issues.add('high_response_time')
                    
                    # Check success rate issues in metrics
                    success_rates = [m.get('success_rate', 1.0) for m in recent_metrics]
                    if success_rates and min(success_rates) < 0.95:
                        distinct_issues.add('low_success_rate')
                    
                    # Count issues addressed by recommendations
                    addressed_issues = set()
                    
                    for rec in recs:
                        rec_type = rec.get('recommendation_type', '')
                        
                        if rec_type in ['caching', 'performance_optimization', 'model_switching']:
                            addressed_issues.add('high_response_time')
                            addressed_issues.add('anomaly_response_time')
                        
                        if rec_type in ['reliability_improvement', 'error_handling']:
                            addressed_issues.add('low_success_rate')
                            addressed_issues.add('anomaly_error_rate')
                        
                        if rec_type in ['rate_limiting', 'throughput_optimization']:
                            addressed_issues.add('anomaly_throughput')
                    
                    # Calculate coverage
                    if distinct_issues:
                        impact_coverage = len(addressed_issues.intersection(distinct_issues)) / len(distinct_issues)
                
                # Overall quality score
                weights = {
                    'relevance': 0.4,
                    'actionability': 0.3,
                    'impact_coverage': 0.3
                }
                
                overall_quality = (
                    avg_relevance * weights['relevance'] +
                    avg_actionability * weights['actionability'] +
                    impact_coverage * weights['impact_coverage']
                )
                
                results.append({
                    'endpoint': rec_endpoint,
                    'model': rec_model,
                    'relevance_score': avg_relevance,
                    'actionability_score': avg_actionability,
                    'impact_coverage': impact_coverage,
                    'overall_quality': overall_quality,
                    'recommendation_count': len(recs),
                    'threshold_met': overall_quality >= self.validation_thresholds['recommendation_relevance']
                })
            
            if not results:
                return {
                    'status': 'error',
                    'message': 'No evaluation results available',
                    'relevance_score': 0.0,
                    'actionability_score': 0.0,
                    'impact_coverage': 0.0,
                    'overall_quality': 0.0
                }
            
            # Calculate aggregated results
            avg_relevance = sum(r['relevance_score'] for r in results) / len(results)
            avg_actionability = sum(r['actionability_score'] for r in results) / len(results)
            avg_impact_coverage = sum(r['impact_coverage'] for r in results) / len(results)
            avg_overall_quality = sum(r['overall_quality'] for r in results) / len(results)
            
            # Generate recommendations for improving recommendations
            meta_recommendations = self._generate_recommendation_improvement_suggestions(results)
            
            return {
                'status': 'success',
                'relevance_score': avg_relevance,
                'actionability_score': avg_actionability,
                'impact_coverage': avg_impact_coverage,
                'overall_quality': avg_overall_quality,
                'threshold_met': avg_overall_quality >= self.validation_thresholds['recommendation_relevance'],
                'group_results': results,
                'recommendations': meta_recommendations
            }
        except Exception as e:
            logger.error(f"Error validating recommendation relevance: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error validating recommendation relevance: {str(e)}",
                'relevance_score': 0.0,
                'actionability_score': 0.0,
                'impact_coverage': 0.0,
                'overall_quality': 0.0
            }
    
    def _generate_recommendation_improvement_suggestions(
        self,
        group_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate meta-recommendations for improving the recommendation system.
        
        Args:
            group_results: Evaluation results for different endpoint/model groups
            
        Returns:
            List of meta-recommendations
        """
        recommendations = []
        
        # Identify groups with low relevance
        low_relevance_groups = [g for g in group_results if g['relevance_score'] < 0.6]
        
        if low_relevance_groups:
            recommendations.append({
                'issue': 'Low Recommendation Relevance',
                'description': f'{len(low_relevance_groups)} endpoint/model groups have recommendations with low relevance scores',
                'recommendation': 'Improve relevance by tying recommendations more closely to observed metrics and anomalies',
                'priority': 'high' if any(g['relevance_score'] < 0.4 for g in low_relevance_groups) else 'medium'
            })
        
        # Identify groups with low actionability
        low_actionability_groups = [g for g in group_results if g['actionability_score'] < 0.6]
        
        if low_actionability_groups:
            recommendations.append({
                'issue': 'Low Recommendation Actionability',
                'description': f'{len(low_actionability_groups)} endpoint/model groups have recommendations that are difficult to act upon',
                'recommendation': 'Improve actionability by providing more specific steps, estimating implementation costs, and prioritizing recommendations',
                'priority': 'high' if any(g['actionability_score'] < 0.4 for g in low_actionability_groups) else 'medium'
            })
        
        # Identify groups with low impact coverage
        low_coverage_groups = [g for g in group_results if g['impact_coverage'] < 0.6]
        
        if low_coverage_groups:
            recommendations.append({
                'issue': 'Insufficient Issue Coverage',
                'description': f'{len(low_coverage_groups)} endpoint/model groups have recommendations that don\'t address all observed issues',
                'recommendation': 'Expand recommendation generation to cover all observed issues in metrics and anomalies',
                'priority': 'high' if any(g['impact_coverage'] < 0.4 for g in low_coverage_groups) else 'medium'
            })
        
        # Check for overall quality
        if len(group_results) > 0:
            avg_quality = sum(g['overall_quality'] for g in group_results) / len(group_results)
            
            if avg_quality < self.validation_thresholds['recommendation_relevance']:
                recommendations.append({
                    'issue': 'Overall Recommendation Quality Below Threshold',
                    'description': f'Average quality score is {avg_quality:.2f}, below the threshold of {self.validation_thresholds["recommendation_relevance"]:.2f}',
                    'recommendation': 'Review and enhance the recommendation generation algorithm, focusing on relevance, actionability, and comprehensive issue coverage',
                    'priority': 'high' if avg_quality < self.validation_thresholds['recommendation_relevance'] * 0.7 else 'medium'
                })
        
        return recommendations
    
    def generate_validation_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report for all aspects of API metrics.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            endpoint: Optional endpoint filter
            model: Optional model filter
            
        Returns:
            Dictionary with comprehensive validation results
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        
        if end_time is None:
            end_time = datetime.now()
        
        # Validate data quality
        data_quality = self.validate_data_quality(
            start_time=start_time,
            end_time=end_time,
            endpoint=endpoint,
            model=model
        )
        
        # Validate prediction accuracy
        prediction_accuracy = self.validate_prediction_accuracy(
            start_time=start_time,
            end_time=end_time,
            endpoint=endpoint,
            model=model
        )
        
        # Validate anomaly detection
        anomaly_detection = self.validate_anomaly_detection(
            start_time=start_time,
            end_time=end_time,
            endpoint=endpoint,
            model=model
        )
        
        # Validate recommendation relevance
        recommendation_relevance = self.validate_recommendation_relevance(
            endpoint=endpoint,
            model=model
        )
        
        # Calculate overall validation score
        weights = {
            'data_quality': 0.3,
            'prediction_accuracy': 0.25,
            'anomaly_detection': 0.25,
            'recommendation_relevance': 0.2
        }
        
        overall_score = (
            data_quality.get('overall_quality', 0.0) * weights['data_quality'] +
            prediction_accuracy.get('accuracy', 0.0) * weights['prediction_accuracy'] +
            anomaly_detection.get('effectiveness', 0.0) * weights['anomaly_detection'] +
            recommendation_relevance.get('overall_quality', 0.0) * weights['recommendation_relevance']
        )
        
        # Collect all recommendations
        all_recommendations = []
        
        if 'recommendations' in data_quality and data_quality['recommendations']:
            for rec in data_quality['recommendations']:
                rec['category'] = 'Data Quality'
                all_recommendations.append(rec)
        
        if 'recommendations' in prediction_accuracy and prediction_accuracy['recommendations']:
            for rec in prediction_accuracy['recommendations']:
                rec['category'] = 'Prediction Accuracy'
                all_recommendations.append(rec)
        
        if 'recommendations' in anomaly_detection and anomaly_detection['recommendations']:
            for rec in anomaly_detection['recommendations']:
                rec['category'] = 'Anomaly Detection'
                all_recommendations.append(rec)
        
        if 'recommendations' in recommendation_relevance and recommendation_relevance['recommendations']:
            for rec in recommendation_relevance['recommendations']:
                rec['category'] = 'Recommendation Quality'
                all_recommendations.append(rec)
        
        # Sort recommendations by priority
        priority_map = {'high': 3, 'medium': 2, 'low': 1}
        all_recommendations.sort(
            key=lambda r: priority_map.get(r.get('priority', 'low'), 0),
            reverse=True
        )
        
        return {
            'timestamp': datetime.now(),
            'start_time': start_time,
            'end_time': end_time,
            'endpoint': endpoint,
            'model': model,
            'overall_score': overall_score,
            'data_quality': data_quality,
            'prediction_accuracy': prediction_accuracy,
            'anomaly_detection': anomaly_detection,
            'recommendation_relevance': recommendation_relevance,
            'all_recommendations': all_recommendations,
            'status': 'success' if overall_score >= 0.7 else 'needs_improvement'
        }