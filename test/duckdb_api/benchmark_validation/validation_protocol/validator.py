#!/usr/bin/env python3
"""
Standard Benchmark Validator

This module implements standard validation protocols for benchmark results,
providing a comprehensive framework for validating benchmark data quality.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set

from duckdb_api.benchmark_validation.core.base import (
    BenchmarkResult,
    ValidationResult,
    ValidationStatus,
    ValidationLevel,
    BenchmarkValidator
)

logger = logging.getLogger("benchmark_validation.standard_validator")

class StandardBenchmarkValidator(BenchmarkValidator):
    """
    Implements standard benchmark validation protocols.
    
    This class provides methods for validating benchmark results against
    predefined standards, ensuring data quality and reliability.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the StandardBenchmarkValidator.
        
        Args:
            config: Configuration for the validator
        """
        super().__init__(config)
        
        # Set default configuration values
        self.config.setdefault("default_metrics", [
            "average_latency_ms",
            "throughput_items_per_second",
            "memory_peak_mb",
            "total_time_seconds"
        ])
        
        self.config.setdefault("validation_protocols", {
            ValidationLevel.MINIMAL.name: {
                "description": "Basic validation of data presence and format",
                "required_fields": ["model_id", "hardware_id", "metrics"],
                "required_metrics": ["average_latency_ms", "throughput_items_per_second"]
            },
            ValidationLevel.STANDARD.name: {
                "description": "Standard validation with basic statistical checks",
                "required_fields": ["model_id", "hardware_id", "metrics", "run_id", "timestamp"],
                "required_metrics": ["average_latency_ms", "throughput_items_per_second", "memory_peak_mb"],
                "value_constraints": {
                    "average_latency_ms": {"min": 0},
                    "throughput_items_per_second": {"min": 0},
                    "memory_peak_mb": {"min": 0}
                }
            },
            ValidationLevel.STRICT.name: {
                "description": "Strict validation with comprehensive checks",
                "required_fields": ["model_id", "hardware_id", "metrics", "run_id", "timestamp", "metadata"],
                "required_metrics": ["average_latency_ms", "throughput_items_per_second", "memory_peak_mb", "total_time_seconds"],
                "value_constraints": {
                    "average_latency_ms": {"min": 0, "max": 100000},
                    "throughput_items_per_second": {"min": 0.001},
                    "memory_peak_mb": {"min": 1},
                    "total_time_seconds": {"min": 0.001}
                },
                "metadata_fields": ["test_environment", "software_versions", "test_parameters"]
            },
            ValidationLevel.CERTIFICATION.name: {
                "description": "Comprehensive validation for certification",
                "required_fields": ["model_id", "hardware_id", "metrics", "run_id", "timestamp", "metadata"],
                "required_metrics": ["average_latency_ms", "throughput_items_per_second", "memory_peak_mb", "total_time_seconds", 
                                     "warmup_iterations", "iterations"],
                "value_constraints": {
                    "average_latency_ms": {"min": 0, "max": 100000},
                    "throughput_items_per_second": {"min": 0.001},
                    "memory_peak_mb": {"min": 1},
                    "total_time_seconds": {"min": 0.001},
                    "warmup_iterations": {"min": 5},
                    "iterations": {"min": 10}
                },
                "metadata_fields": ["test_environment", "software_versions", "test_parameters", "hardware_details", "model_details"]
            }
        })
    
    def validate(
        self, 
        benchmark_result: BenchmarkResult,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a benchmark result.
        
        Args:
            benchmark_result: The benchmark result to validate
            validation_level: Level of validation to perform
            reference_data: Optional reference data for validation
            
        Returns:
            ValidationResult object with validation results
        """
        logger.info(f"Validating benchmark {benchmark_result.result_id} at level {validation_level.name}")
        
        # Get validation protocol for the specified level
        protocol = self.config["validation_protocols"].get(
            validation_level.name, 
            self.config["validation_protocols"][ValidationLevel.STANDARD.name]
        )
        
        # Track validation issues
        issues = []
        recommendations = []
        
        # Validate required fields
        missing_fields = []
        for field in protocol["required_fields"]:
            value = getattr(benchmark_result, field, None)
            if value is None or (isinstance(value, dict) and len(value) == 0):
                missing_fields.append(field)
        
        if missing_fields:
            issues.append({
                "type": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            })
            recommendations.append(f"Provide values for required fields: {', '.join(missing_fields)}")
        
        # Validate required metrics
        missing_metrics = []
        for metric in protocol["required_metrics"]:
            if metric not in benchmark_result.metrics:
                missing_metrics.append(metric)
        
        if missing_metrics:
            issues.append({
                "type": "error",
                "message": f"Missing required metrics: {', '.join(missing_metrics)}"
            })
            recommendations.append(f"Provide values for required metrics: {', '.join(missing_metrics)}")
        
        # Validate value constraints
        constraint_violations = []
        if "value_constraints" in protocol:
            for metric, constraints in protocol["value_constraints"].items():
                if metric in benchmark_result.metrics:
                    value = benchmark_result.metrics[metric]
                    
                    if "min" in constraints and value < constraints["min"]:
                        constraint_violations.append({
                            "metric": metric,
                            "constraint": "min",
                            "expected": constraints["min"],
                            "actual": value
                        })
                    
                    if "max" in constraints and value > constraints["max"]:
                        constraint_violations.append({
                            "metric": metric,
                            "constraint": "max",
                            "expected": constraints["max"],
                            "actual": value
                        })
        
        if constraint_violations:
            for violation in constraint_violations:
                issues.append({
                    "type": "error",
                    "message": f"Metric {violation['metric']} violates {violation['constraint']} constraint: {violation['actual']} (expected {violation['constraint']} {violation['expected']})"
                })
            recommendations.append("Ensure metrics values meet the constraints")
        
        # Validate metadata fields
        if "metadata_fields" in protocol:
            missing_metadata = []
            for field in protocol["metadata_fields"]:
                if field not in benchmark_result.metadata:
                    missing_metadata.append(field)
            
            if missing_metadata:
                issues.append({
                    "type": "warning",
                    "message": f"Missing recommended metadata fields: {', '.join(missing_metadata)}"
                })
                recommendations.append(f"Include recommended metadata fields: {', '.join(missing_metadata)}")
        
        # Determine validation status
        if any(issue["type"] == "error" for issue in issues):
            status = ValidationStatus.INVALID
        elif issues:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID
        
        # Create validation metrics
        validation_metrics = {
            "standard_validation": {
                "level": validation_level.name,
                "protocol": protocol["description"],
                "missing_fields": missing_fields,
                "missing_metrics": missing_metrics,
                "constraint_violations": constraint_violations,
                "reference_data_used": reference_data is not None
            }
        }
        
        # Include references to other validations if available
        if reference_data:
            other_validations = reference_data.get("other_validations", {})
            if other_validations:
                validation_metrics["other_validations"] = {
                    "types": list(other_validations.keys()),
                    "summary": {vtype: vdata.get("status", "unknown") for vtype, vdata in other_validations.items()}
                }
        
        # Calculate confidence score
        confidence_score = self._calculate_validation_confidence(
            benchmark_result=benchmark_result,
            validation_level=validation_level,
            issues=issues
        )
        
        return ValidationResult(
            benchmark_result=benchmark_result,
            status=status,
            validation_level=validation_level,
            confidence_score=confidence_score,
            validation_metrics=validation_metrics,
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
        Validate a batch of benchmark results.
        
        Args:
            benchmark_results: List of benchmark results to validate
            validation_level: Level of validation to perform
            reference_data: Optional reference data for validation
            
        Returns:
            List of ValidationResult objects
        """
        logger.info(f"Batch validating {len(benchmark_results)} benchmark results at level {validation_level.name}")
        
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
        # Ensure we have standard validation metrics
        if "standard_validation" not in validation_result.validation_metrics:
            return 0.5
        
        standard_metrics = validation_result.validation_metrics["standard_validation"]
        
        # Consider validation level
        level_factor = {
            ValidationLevel.MINIMAL.name: 0.6,
            ValidationLevel.STANDARD.name: 0.8,
            ValidationLevel.STRICT.name: 0.9,
            ValidationLevel.CERTIFICATION.name: 1.0
        }.get(standard_metrics.get("level"), 0.7)
        
        # Consider validation issues
        issues = validation_result.issues
        error_count = sum(1 for issue in issues if issue["type"] == "error")
        warning_count = sum(1 for issue in issues if issue["type"] == "warning")
        
        # Calculate issue factor
        issue_factor = max(0, 1 - (error_count * 0.3 + warning_count * 0.1))
        
        # Calculate overall confidence
        confidence = level_factor * issue_factor
        
        return confidence
    
    def _calculate_validation_confidence(
        self,
        benchmark_result: BenchmarkResult,
        validation_level: ValidationLevel,
        issues: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score for validation.
        
        Args:
            benchmark_result: The benchmark result being validated
            validation_level: Validation level applied
            issues: List of validation issues found
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Base confidence on validation level
        level_factor = {
            ValidationLevel.MINIMAL: 0.6,
            ValidationLevel.STANDARD: 0.8,
            ValidationLevel.STRICT: 0.9,
            ValidationLevel.CERTIFICATION: 1.0
        }.get(validation_level, 0.7)
        
        # Adjust for validation issues
        error_count = sum(1 for issue in issues if issue["type"] == "error")
        warning_count = sum(1 for issue in issues if issue["type"] == "warning")
        
        # Calculate issue factor
        issue_factor = max(0, 1 - (error_count * 0.3 + warning_count * 0.1))
        
        # Calculate overall confidence
        confidence = level_factor * issue_factor
        
        return confidence