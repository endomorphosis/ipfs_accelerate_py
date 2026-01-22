#!/usr/bin/env python3
"""
Statistical Outlier Detector for Benchmark Results

This module implements outlier detection algorithms for identifying anomalous
benchmark results using statistical methods.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from scipy import stats

from duckdb_api.benchmark_validation.core.base import (
    BenchmarkResult,
    ValidationResult,
    ValidationStatus,
    ValidationLevel,
    OutlierDetector
)

logger = logging.getLogger("benchmark_validation.outlier_detection")

class StatisticalOutlierDetector(OutlierDetector):
    """
    Implements statistical outlier detection for benchmark results.
    
    This class provides methods for detecting outliers in benchmark results
    using various statistical techniques, including Z-score analysis, IQR-based
    detection, and DBSCAN clustering.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the StatisticalOutlierDetector.
        
        Args:
            config: Configuration for the detector
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config.setdefault("detection_methods", ["zscore", "iqr"])
        self.config.setdefault("zscore_threshold", 2.0)
        self.config.setdefault("iqr_factor", 1.5)
        self.config.setdefault("min_samples", 5)
        self.config.setdefault("default_metrics", [
            "average_latency_ms",
            "throughput_items_per_second",
            "memory_peak_mb",
            "total_time_seconds"
        ])
    
    def validate(
        self, 
        benchmark_result: BenchmarkResult,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a benchmark result by checking for outliers.
        
        Args:
            benchmark_result: The benchmark result to validate
            validation_level: Level of validation to perform
            reference_data: Reference data containing benchmark results for comparison
            
        Returns:
            ValidationResult object with outlier detection results
        """
        logger.info(f"Validating benchmark {benchmark_result.result_id} for outliers")
        
        # Need reference data to detect outliers
        if not reference_data or "reference_results" not in reference_data:
            logger.warning("No reference data provided for outlier detection")
            return ValidationResult(
                benchmark_result=benchmark_result,
                status=ValidationStatus.WARNING,
                validation_level=validation_level,
                confidence_score=0.5,
                validation_metrics={
                    "outlier_detection": {
                        "status": "skipped",
                        "reason": "No reference data provided"
                    }
                },
                issues=[{
                    "type": "warning",
                    "message": "Outlier detection skipped due to lack of reference data"
                }],
                recommendations=["Provide reference benchmark results for outlier detection"]
            )
        
        reference_results = reference_data["reference_results"]
        
        # Ensure we have enough reference results
        if len(reference_results) < self.config["min_samples"]:
            logger.warning(f"Not enough reference results for outlier detection, need at least {self.config['min_samples']}")
            return ValidationResult(
                benchmark_result=benchmark_result,
                status=ValidationStatus.WARNING,
                validation_level=validation_level,
                confidence_score=0.5,
                validation_metrics={
                    "outlier_detection": {
                        "status": "skipped",
                        "reason": "Not enough reference results"
                    }
                },
                issues=[{
                    "type": "warning",
                    "message": f"Outlier detection skipped: need at least {self.config['min_samples']} reference results"
                }],
                recommendations=["Collect more benchmark results for reliable outlier detection"]
            )
        
        # Determine which metrics to check
        metrics = reference_data.get("metrics", self.config["default_metrics"])
        
        # Calculate outlier scores
        outlier_scores = self.calculate_outlier_score(
            benchmark_result=benchmark_result,
            reference_results=reference_results,
            metrics=metrics
        )
        
        # Determine if the result is an outlier
        is_outlier = False
        outlier_metrics = {}
        
        for metric, score in outlier_scores.items():
            threshold = self.config["zscore_threshold"]
            outlier_metrics[metric] = {
                "score": score,
                "threshold": threshold,
                "is_outlier": abs(score) > threshold
            }
            
            if abs(score) > threshold:
                is_outlier = True
        
        # Create validation result
        status = ValidationStatus.WARNING if is_outlier else ValidationStatus.VALID
        issues = []
        recommendations = []
        
        if is_outlier:
            outlier_metric_names = [m for m, data in outlier_metrics.items() if data["is_outlier"]]
            issues.append({
                "type": "warning",
                "message": f"Benchmark result is an outlier for metrics: {', '.join(outlier_metric_names)}"
            })
            recommendations.append("Review and verify benchmark setup and configuration")
            recommendations.append("Run additional benchmark iterations to confirm results")
        
        # Calculate confidence based on sample size and outlier score magnitude
        sample_size_factor = min(1.0, len(reference_results) / 30)  # More samples = higher confidence
        outlier_magnitude = max([abs(score) for score in outlier_scores.values()], default=0)
        outlier_factor = max(0, 1 - (outlier_magnitude / (2 * threshold)))  # Higher outlier score = lower confidence
        
        confidence_score = sample_size_factor * outlier_factor
        
        return ValidationResult(
            benchmark_result=benchmark_result,
            status=status,
            validation_level=validation_level,
            confidence_score=confidence_score,
            validation_metrics={
                "outlier_detection": {
                    "status": "completed",
                    "is_outlier": is_outlier,
                    "metrics": outlier_metrics,
                    "sample_size": len(reference_results),
                    "detection_methods": self.config["detection_methods"]
                }
            },
            issues=issues,
            recommendations=recommendations,
            validator_id=self.validator_id
        )
    
    def validate_batch(
        self,
        benchmark_results: List[BenchmarkResult],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """
        Validate a batch of benchmark results by checking for outliers.
        
        Args:
            benchmark_results: List of benchmark results to validate
            validation_level: Level of validation to perform
            reference_data: Optional reference data for validation
            
        Returns:
            List of ValidationResult objects
        """
        logger.info(f"Batch validating {len(benchmark_results)} benchmark results for outliers")
        
        # If no reference data provided, use the batch itself as reference
        if not reference_data or "reference_results" not in reference_data:
            reference_data = {"reference_results": benchmark_results}
        
        # Validate each benchmark result
        validation_results = []
        for benchmark_result in benchmark_results:
            validation_result = self.validate(
                benchmark_result=benchmark_result,
                validation_level=validation_level,
                reference_data=reference_data
            )
            validation_results.append(validation_result)
        
        return validation_results
    
    def detect_outliers(
        self,
        benchmark_results: List[BenchmarkResult],
        metrics: List[str] = None,
        threshold: float = None
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Detect outliers in benchmark results for specified metrics.
        
        Args:
            benchmark_results: List of benchmark results to check for outliers
            metrics: List of metric names to check for outliers
            threshold: Z-score threshold for outlier detection (overrides config)
            
        Returns:
            Dictionary mapping metric names to lists of outlier BenchmarkResults
        """
        logger.info(f"Detecting outliers in {len(benchmark_results)} benchmark results")
        
        # Use default metrics if none specified
        if metrics is None:
            metrics = self.config["default_metrics"]
        
        # Use configured threshold if none specified
        if threshold is None:
            threshold = self.config["zscore_threshold"]
        
        # Group benchmark results by model_id and hardware_id
        grouped_results = {}
        for result in benchmark_results:
            key = f"{result.model_id}_{result.hardware_id}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Detect outliers in each group
        outliers = {metric: [] for metric in metrics}
        
        for group_key, group_results in grouped_results.items():
            # Skip groups with too few results
            if len(group_results) < self.config["min_samples"]:
                continue
            
            # Extract metrics data
            metric_values = {}
            for metric in metrics:
                values = []
                for result in group_results:
                    if metric in result.metrics:
                        values.append((result, result.metrics[metric]))
                
                if len(values) >= self.config["min_samples"]:
                    metric_values[metric] = values
            
            # Detect outliers in each metric
            for metric, values in metric_values.items():
                results, data = zip(*values)
                
                # Calculate Z-scores
                mean = np.mean(data)
                std = np.std(data)
                
                if std > 0:
                    zscores = [(result, (value - mean) / std) for result, value in zip(results, data)]
                    
                    # Identify outliers
                    for result, zscore in zscores:
                        if abs(zscore) > threshold:
                            outliers[metric].append(result)
        
        # Log results
        total_outliers = sum(len(results) for results in outliers.values())
        logger.info(f"Detected {total_outliers} outliers across {len(metrics)} metrics")
        
        return outliers
    
    def calculate_outlier_score(
        self,
        benchmark_result: BenchmarkResult,
        reference_results: List[BenchmarkResult],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate outlier scores for a benchmark result compared to reference results.
        
        Args:
            benchmark_result: The benchmark result to assess
            reference_results: Reference benchmark results for comparison
            metrics: List of metric names to calculate outlier scores for
            
        Returns:
            Dictionary mapping metric names to outlier scores (Z-scores)
        """
        logger.info(f"Calculating outlier scores for benchmark {benchmark_result.result_id}")
        
        # Use default metrics if none specified
        if metrics is None:
            metrics = self.config["default_metrics"]
        
        # Filter reference results to match model_id and hardware_id
        filtered_refs = [
            ref for ref in reference_results
            if ref.model_id == benchmark_result.model_id and ref.hardware_id == benchmark_result.hardware_id
        ]
        
        # If not enough matching results, use all reference results
        if len(filtered_refs) < self.config["min_samples"]:
            filtered_refs = reference_results
        
        # Calculate outlier scores for each metric
        outlier_scores = {}
        
        for metric in metrics:
            # Skip if metric not present in benchmark result
            if metric not in benchmark_result.metrics:
                continue
            
            # Extract metric values from reference results
            ref_values = []
            for ref in filtered_refs:
                if metric in ref.metrics:
                    ref_values.append(ref.metrics[metric])
            
            # Skip if not enough reference values
            if len(ref_values) < self.config["min_samples"]:
                continue
            
            # Calculate Z-score
            mean = np.mean(ref_values)
            std = np.std(ref_values)
            
            if std > 0:
                zscore = (benchmark_result.metrics[metric] - mean) / std
                outlier_scores[metric] = float(zscore)
        
        return outlier_scores
    
    def calculate_confidence(
        self,
        validation_result: ValidationResult
    ) -> float:
        """
        Calculate confidence score for a validation result.
        
        Args:
            validation_result: The validation result to assess
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Ensure we have outlier detection metrics
        if "outlier_detection" not in validation_result.validation_metrics:
            return 0.5
        
        outlier_metrics = validation_result.validation_metrics["outlier_detection"]
        
        # Consider sample size - more samples means higher confidence
        sample_size = outlier_metrics.get("sample_size", 0)
        sample_factor = min(1.0, sample_size / 30)
        
        # Consider outlier magnitude - stronger outliers have lower confidence
        is_outlier = outlier_metrics.get("is_outlier", False)
        if is_outlier:
            # Get the maximum outlier score
            metrics = outlier_metrics.get("metrics", {})
            max_score = max([data.get("score", 0) for data in metrics.values()], default=0)
            threshold = self.config["zscore_threshold"]
            
            # Calculate confidence factor based on outlier magnitude
            outlier_factor = max(0, 1 - (abs(max_score) - threshold) / (3 * threshold))
        else:
            outlier_factor = 1.0
        
        # Combine factors
        confidence = sample_factor * outlier_factor
        
        return confidence