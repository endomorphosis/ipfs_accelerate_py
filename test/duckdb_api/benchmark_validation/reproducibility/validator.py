#!/usr/bin/env python3
"""
Reproducibility Validator for Benchmark Results

This module implements validation of benchmark reproducibility across multiple runs,
helping ensure that benchmark results are consistent and reliable.
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
    ReproducibilityValidator as BaseReproducibilityValidator
)

logger = logging.getLogger("benchmark_validation.reproducibility")

class ReproducibilityValidator(BaseReproducibilityValidator):
    """
    Implements reproducibility validation for benchmark results.
    
    This class provides methods for assessing the reproducibility of benchmark
    results across multiple runs, identifying inconsistencies, and calculating
    reproducibility scores.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ReproducibilityValidator.
        
        Args:
            config: Configuration for the validator
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config.setdefault("min_runs", 3)
        self.config.setdefault("cv_threshold", 5.0)  # Coefficient of variation threshold (%)
        self.config.setdefault("default_metrics", [
            "average_latency_ms",
            "throughput_items_per_second",
            "memory_peak_mb",
            "total_time_seconds"
        ])
        self.config.setdefault("error_metrics", [
            "average_error",
            "std_deviation",
            "coefficient_of_variation",
            "min_max_range"
        ])
    
    def validate(
        self, 
        benchmark_result: BenchmarkResult,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a single benchmark result.
        
        For reproducibility validation, a single result doesn't provide enough information.
        This method returns a warning indicating that multiple runs are needed.
        
        Args:
            benchmark_result: The benchmark result to validate
            validation_level: Level of validation to perform
            reference_data: Optional reference data for validation
            
        Returns:
            ValidationResult object with validation results
        """
        logger.warning("Reproducibility validation requires multiple benchmark results")
        return ValidationResult(
            benchmark_result=benchmark_result,
            status=ValidationStatus.WARNING,
            validation_level=validation_level,
            confidence_score=0.0,
            validation_metrics={
                "reproducibility": {
                    "status": "skipped",
                    "reason": "Reproducibility validation requires multiple benchmark results"
                }
            },
            issues=[{
                "type": "warning",
                "message": "Cannot validate reproducibility with a single benchmark result"
            }],
            recommendations=["Perform multiple benchmark runs to assess reproducibility"],
            validator_id=self.validator_id
        )
    
    def validate_batch(
        self,
        benchmark_results: List[BenchmarkResult],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """
        Validate a batch of benchmark results for reproducibility.
        
        This method groups benchmark results by model and hardware,
        then assesses reproducibility for each group.
        
        Args:
            benchmark_results: List of benchmark results to validate
            validation_level: Level of validation to perform
            reference_data: Optional reference data for validation
            
        Returns:
            List of ValidationResult objects
        """
        logger.info(f"Batch validating {len(benchmark_results)} benchmark results for reproducibility")
        
        # Group benchmark results by model_id and hardware_id
        grouped_results = {}
        for result in benchmark_results:
            key = f"{result.model_id}_{result.hardware_id}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Validate each group
        validation_results = []
        for group_key, group_results in grouped_results.items():
            # Skip groups with too few results
            if len(group_results) < self.config["min_runs"]:
                for result in group_results:
                    validation_result = ValidationResult(
                        benchmark_result=result,
                        status=ValidationStatus.WARNING,
                        validation_level=validation_level,
                        confidence_score=0.0,
                        validation_metrics={
                            "reproducibility": {
                                "status": "skipped",
                                "reason": f"Not enough runs (have {len(group_results)}, need {self.config['min_runs']})"
                            }
                        },
                        issues=[{
                            "type": "warning",
                            "message": f"Need at least {self.config['min_runs']} runs to assess reproducibility"
                        }],
                        recommendations=[f"Perform at least {self.config['min_runs']} benchmark runs"],
                        validator_id=self.validator_id
                    )
                    validation_results.append(validation_result)
                continue
            
            # Assess reproducibility for the group
            group_validation = self.validate_reproducibility(
                benchmark_results=group_results,
                validation_level=validation_level,
                reference_data=reference_data
            )
            
            # Create individual validation results based on group validation
            for result in group_results:
                validation_result = ValidationResult(
                    benchmark_result=result,
                    status=group_validation.status,
                    validation_level=validation_level,
                    confidence_score=group_validation.confidence_score,
                    validation_metrics=group_validation.validation_metrics,
                    issues=group_validation.issues,
                    recommendations=group_validation.recommendations,
                    validator_id=self.validator_id
                )
                validation_results.append(validation_result)
        
        return validation_results
    
    def validate_reproducibility(
        self,
        benchmark_results: List[BenchmarkResult],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate reproducibility of benchmark results.
        
        Args:
            benchmark_results: List of benchmark results to validate for reproducibility
            validation_level: Level of validation to perform
            reference_data: Optional reference data for validation
            
        Returns:
            ValidationResult for reproducibility assessment
        """
        logger.info(f"Validating reproducibility of {len(benchmark_results)} benchmark results")
        
        # Ensure we have enough results
        if len(benchmark_results) < self.config["min_runs"]:
            logger.warning(f"Not enough benchmark results for reproducibility validation")
            return ValidationResult(
                benchmark_result=benchmark_results[0],
                status=ValidationStatus.WARNING,
                validation_level=validation_level,
                confidence_score=0.0,
                validation_metrics={
                    "reproducibility": {
                        "status": "skipped",
                        "reason": f"Not enough runs (have {len(benchmark_results)}, need {self.config['min_runs']})"
                    }
                },
                issues=[{
                    "type": "warning",
                    "message": f"Need at least {self.config['min_runs']} runs to assess reproducibility"
                }],
                recommendations=[f"Perform at least {self.config['min_runs']} benchmark runs"],
                validator_id=self.validator_id
            )
        
        # Calculate reproducibility metrics
        reproducibility_score = self.calculate_reproducibility_score(benchmark_results)
        
        # Determine which metrics to analyze
        metrics = reference_data.get("metrics", self.config["default_metrics"]) if reference_data else self.config["default_metrics"]
        
        # Extract metric values for each benchmark result
        metric_values = {}
        for metric in metrics:
            values = []
            for result in benchmark_results:
                if metric in result.metrics:
                    values.append(result.metrics[metric])
            if len(values) >= self.config["min_runs"]:
                metric_values[metric] = values
        
        # Calculate statistics for each metric
        metric_stats = {}
        for metric, values in metric_values.items():
            # Basic statistics
            mean = np.mean(values)
            std = np.std(values)
            min_val = min(values)
            max_val = max(values)
            cv = (std / mean) * 100 if mean > 0 else float('inf')  # Coefficient of variation
            
            metric_stats[metric] = {
                "mean": float(mean),
                "std_deviation": float(std),
                "min_value": float(min_val),
                "max_value": float(max_val),
                "range": float(max_val - min_val),
                "range_percent": float((max_val - min_val) / mean * 100) if mean > 0 else float('inf'),
                "coefficient_of_variation": float(cv),
                "is_reproducible": cv <= self.config["cv_threshold"],
                "sample_size": len(values)
            }
        
        # Determine overall reproducibility
        is_reproducible = all(stats["is_reproducible"] for stats in metric_stats.values())
        
        # Create validation metrics
        validation_metrics = {
            "reproducibility": {
                "status": "completed",
                "is_reproducible": is_reproducible,
                "reproducibility_score": reproducibility_score,
                "sample_size": len(benchmark_results),
                "metrics": metric_stats,
                "cv_threshold": self.config["cv_threshold"]
            }
        }
        
        # Determine status and issues
        issues = []
        recommendations = []
        
        if is_reproducible:
            status = ValidationStatus.VALID
        else:
            status = ValidationStatus.WARNING
            non_reproducible_metrics = [
                metric for metric, stats in metric_stats.items()
                if not stats["is_reproducible"]
            ]
            
            issues.append({
                "type": "warning",
                "message": f"Benchmark is not reproducible for metrics: {', '.join(non_reproducible_metrics)}"
            })
            
            recommendations.append("Review benchmark environment for sources of variability")
            recommendations.append("Increase warm-up iterations to stabilize performance")
            recommendations.append("Ensure consistent system state between benchmark runs")
        
        # Calculate confidence score
        confidence_score = reproducibility_score
        
        # Create validation result
        return ValidationResult(
            benchmark_result=benchmark_results[0],  # Use first result as reference
            status=status,
            validation_level=validation_level,
            confidence_score=confidence_score,
            validation_metrics=validation_metrics,
            issues=issues,
            recommendations=recommendations,
            validator_id=self.validator_id
        )
    
    def calculate_reproducibility_score(
        self,
        benchmark_results: List[BenchmarkResult]
    ) -> float:
        """
        Calculate reproducibility score for benchmark results.
        
        Args:
            benchmark_results: List of benchmark results to assess
            
        Returns:
            Reproducibility score from 0.0 to 1.0
        """
        logger.info(f"Calculating reproducibility score for {len(benchmark_results)} results")
        
        # Ensure we have enough results
        if len(benchmark_results) < self.config["min_runs"]:
            return 0.0
        
        # Extract metric values for each benchmark result
        metrics = self.config["default_metrics"]
        metric_values = {}
        
        for metric in metrics:
            values = []
            for result in benchmark_results:
                if metric in result.metrics:
                    values.append(result.metrics[metric])
            
            if len(values) >= self.config["min_runs"]:
                metric_values[metric] = values
        
        # Calculate coefficient of variation for each metric
        cv_scores = {}
        for metric, values in metric_values.items():
            mean = np.mean(values)
            std = np.std(values)
            cv = (std / mean) * 100 if mean > 0 else float('inf')
            cv_scores[metric] = cv
        
        # Calculate reproducibility score
        if not cv_scores:
            return 0.0
        
        # Convert CV to a score between 0 and 1
        # Lower CV means higher reproducibility
        cv_threshold = self.config["cv_threshold"]
        metric_scores = {}
        
        for metric, cv in cv_scores.items():
            if cv <= cv_threshold:
                # CV below threshold gets a score between 0.8 and 1.0
                metric_scores[metric] = 0.8 + 0.2 * (1 - cv / cv_threshold)
            else:
                # CV above threshold gets a score between 0.0 and 0.8
                metric_scores[metric] = max(0.0, 0.8 * (1 - (cv - cv_threshold) / (5 * cv_threshold)))
        
        # Overall score is the average of metric scores
        overall_score = sum(metric_scores.values()) / len(metric_scores)
        
        return overall_score
    
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
        # Ensure we have reproducibility metrics
        if ("reproducibility" not in validation_result.validation_metrics or
            validation_result.validation_metrics["reproducibility"].get("status") != "completed"):
            return 0.5
        
        repro_metrics = validation_result.validation_metrics["reproducibility"]
        
        # Consider sample size - more samples means higher confidence
        sample_size = repro_metrics.get("sample_size", 0)
        min_runs = self.config["min_runs"]
        
        if sample_size < min_runs:
            return 0.0
        
        # More runs beyond the minimum increases confidence
        sample_factor = min(1.0, 0.7 + 0.3 * (sample_size - min_runs) / (10 - min_runs))
        
        # Consider metric consistency
        metric_stats = repro_metrics.get("metrics", {})
        cv_values = [stats.get("coefficient_of_variation", float('inf')) for stats in metric_stats.values()]
        
        if not cv_values:
            return 0.5 * sample_factor
        
        # Average CV normalized to a 0-1 scale (lower CV means higher confidence)
        avg_cv = np.mean(cv_values)
        cv_threshold = self.config["cv_threshold"]
        
        if avg_cv <= cv_threshold:
            cv_factor = 0.8 + 0.2 * (1 - avg_cv / cv_threshold)
        else:
            cv_factor = max(0.0, 0.8 * (1 - (avg_cv - cv_threshold) / (5 * cv_threshold)))
        
        # Combine factors
        confidence = sample_factor * cv_factor
        
        return confidence