#!/usr/bin/env python3
"""
Benchmark Validation System - Core Base Module

This module defines the fundamental components and interfaces for the Benchmark Validation System.
It provides abstract base classes and data structures that form the foundation of the validation framework.
"""

import os
import sys
import logging
import datetime
import json
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, TypeVar, Generic, Set

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark_validation")

class ValidationLevel(Enum):
    """Validation levels for benchmark results."""
    MINIMAL = 1    # Basic data presence and format validation
    STANDARD = 2   # Standard statistical validation with outlier detection
    STRICT = 3     # Comprehensive validation with reproducibility checks
    CERTIFICATION = 4  # Highest level validation for certified benchmarks

class BenchmarkType(Enum):
    """Types of benchmarks that can be validated."""
    PERFORMANCE = 1     # Performance benchmarks (throughput, latency, etc.)
    COMPATIBILITY = 2   # Hardware compatibility tests
    INTEGRATION = 3     # Integration tests
    WEB_PLATFORM = 4    # Web platform-specific tests (WebGPU, WebNN)
    
class ValidationStatus(Enum):
    """Status of a validation result."""
    VALID = 1           # Benchmark passed validation
    INVALID = 2         # Benchmark failed validation
    WARNING = 3         # Benchmark passed with warnings
    ERROR = 4           # Error occurred during validation
    PENDING = 5         # Validation in progress or pending
    
class ValidationScope(Enum):
    """Scope of benchmark validation."""
    SINGLE_RUN = 1      # Validate a single benchmark run
    COMPARISON = 2      # Compare benchmark runs against each other
    TREND = 3           # Validate against historical trends
    REPRODUCIBILITY = 4 # Validate reproducibility across multiple runs

class BenchmarkResult:
    """
    Base class for benchmark results to be validated.
    
    This class represents a benchmark result from any source that will be validated
    by the framework. It provides a common interface for accessing benchmark data.
    """
    
    def __init__(
        self, 
        result_id: str,
        benchmark_type: BenchmarkType,
        model_id: Optional[int] = None,
        hardware_id: Optional[int] = None,
        metrics: Dict[str, Any] = None,
        run_id: Optional[int] = None,
        timestamp: Optional[datetime.datetime] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a BenchmarkResult object.
        
        Args:
            result_id: Unique identifier for this benchmark result
            benchmark_type: Type of benchmark (performance, compatibility, etc.)
            model_id: Identifier for the model used (optional)
            hardware_id: Identifier for the hardware used (optional)
            metrics: Dictionary of benchmark metrics
            run_id: Identifier for the test run (optional)
            timestamp: Timestamp of when the benchmark was run
            metadata: Additional metadata about the benchmark
        """
        self.result_id = result_id
        self.benchmark_type = benchmark_type
        self.model_id = model_id
        self.hardware_id = hardware_id
        self.metrics = metrics or {}
        self.run_id = run_id
        self.timestamp = timestamp or datetime.datetime.now()
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the benchmark result to a dictionary."""
        return {
            "result_id": self.result_id,
            "benchmark_type": self.benchmark_type.name,
            "model_id": self.model_id,
            "hardware_id": self.hardware_id,
            "metrics": self.metrics,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create a BenchmarkResult from a dictionary."""
        benchmark_type = BenchmarkType[data["benchmark_type"]] if isinstance(data["benchmark_type"], str) else data["benchmark_type"]
        timestamp = datetime.datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        
        return cls(
            result_id=data["result_id"],
            benchmark_type=benchmark_type,
            model_id=data.get("model_id"),
            hardware_id=data.get("hardware_id"),
            metrics=data.get("metrics", {}),
            run_id=data.get("run_id"),
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )

class ValidationResult:
    """
    Result of validating a benchmark result.
    
    This class represents the outcome of validation checks performed on a benchmark result,
    including status, confidence score, and detailed validation metrics.
    """
    
    def __init__(
        self,
        benchmark_result: BenchmarkResult,
        status: ValidationStatus,
        validation_level: ValidationLevel,
        confidence_score: float,
        validation_metrics: Dict[str, Any],
        issues: List[Dict[str, Any]] = None,
        recommendations: List[str] = None,
        validation_timestamp: Optional[datetime.datetime] = None,
        validator_id: Optional[str] = None
    ):
        """
        Initialize a ValidationResult object.
        
        Args:
            benchmark_result: The benchmark result being validated
            status: Validation status (VALID, INVALID, WARNING, ERROR)
            validation_level: Level of validation performed
            confidence_score: Confidence score for validation (0.0 to 1.0)
            validation_metrics: Detailed metrics from validation process
            issues: List of issues found during validation
            recommendations: Recommended actions based on validation
            validation_timestamp: Timestamp of when validation was performed
            validator_id: Identifier for validator that performed validation
        """
        self.id = str(uuid.uuid4())
        self.benchmark_result = benchmark_result
        self.status = status
        self.validation_level = validation_level
        self.confidence_score = confidence_score
        self.validation_metrics = validation_metrics
        self.issues = issues or []
        self.recommendations = recommendations or []
        self.validation_timestamp = validation_timestamp or datetime.datetime.now()
        self.validator_id = validator_id
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "id": self.id,
            "benchmark_result": self.benchmark_result.to_dict(),
            "status": self.status.name,
            "validation_level": self.validation_level.name,
            "confidence_score": self.confidence_score,
            "validation_metrics": self.validation_metrics,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "validator_id": self.validator_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create a ValidationResult from a dictionary."""
        validation_obj = cls(
            benchmark_result=BenchmarkResult.from_dict(data["benchmark_result"]),
            status=ValidationStatus[data["status"]] if isinstance(data["status"], str) else data["status"],
            validation_level=ValidationLevel[data["validation_level"]] if isinstance(data["validation_level"], str) else data["validation_level"],
            confidence_score=data["confidence_score"],
            validation_metrics=data["validation_metrics"],
            issues=data.get("issues", []),
            recommendations=data.get("recommendations", []),
            validation_timestamp=datetime.datetime.fromisoformat(data["validation_timestamp"]) if data.get("validation_timestamp") else None,
            validator_id=data.get("validator_id")
        )
        validation_obj.id = data["id"]
        return validation_obj

class BenchmarkValidator(ABC):
    """
    Abstract base class for benchmark validators.
    
    This class defines the interface that all benchmark validators must implement.
    A validator is responsible for validating benchmark results against specific criteria.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a BenchmarkValidator.
        
        Args:
            config: Configuration for the validator
        """
        self.config = config or {}
        self.validator_id = str(uuid.uuid4())
        self.logger = logging.getLogger(f"benchmark_validator.{self.__class__.__name__}")
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass

class ReproducibilityValidator(BenchmarkValidator):
    """
    Abstract base class for reproducibility validators.
    
    This class extends BenchmarkValidator to specifically handle reproducibility validation
    of benchmark results across multiple runs.
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass

class OutlierDetector(BenchmarkValidator):
    """
    Abstract base class for outlier detection in benchmark results.
    
    This class extends BenchmarkValidator to specifically handle outlier detection
    in benchmark results.
    """
    
    @abstractmethod
    def detect_outliers(
        self,
        benchmark_results: List[BenchmarkResult],
        metrics: List[str] = None,
        threshold: float = 2.0
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Detect outliers in benchmark results.
        
        Args:
            benchmark_results: List of benchmark results to check for outliers
            metrics: List of metric names to check for outliers
            threshold: Threshold for outlier detection (typical z-score threshold)
            
        Returns:
            Dictionary mapping metric names to lists of outlier BenchmarkResults
        """
        pass
    
    @abstractmethod
    def calculate_outlier_score(
        self,
        benchmark_result: BenchmarkResult,
        reference_results: List[BenchmarkResult],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate outlier scores for a benchmark result.
        
        Args:
            benchmark_result: The benchmark result to assess
            reference_results: Reference benchmark results for comparison
            metrics: List of metric names to calculate outlier scores for
            
        Returns:
            Dictionary mapping metric names to outlier scores
        """
        pass

class BenchmarkCertifier(BenchmarkValidator):
    """
    Abstract base class for benchmark certification.
    
    This class extends BenchmarkValidator to specifically handle certification
    of benchmark results according to defined standards.
    """
    
    @abstractmethod
    def certify(
        self,
        benchmark_result: BenchmarkResult,
        validation_results: List[ValidationResult] = None,
        certification_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Certify a benchmark result.
        
        Args:
            benchmark_result: The benchmark result to certify
            validation_results: Previous validation results (optional)
            certification_level: Level of certification to apply
            
        Returns:
            Dictionary with certification details
        """
        pass
    
    @abstractmethod
    def verify_certification(
        self,
        certification: Dict[str, Any],
        benchmark_result: BenchmarkResult
    ) -> bool:
        """
        Verify a certification against a benchmark result.
        
        Args:
            certification: Certification details to verify
            benchmark_result: The benchmark result to verify against
            
        Returns:
            True if certification is valid, False otherwise
        """
        pass

class ValidationReporter(ABC):
    """
    Abstract base class for validation reporters.
    
    This class defines the interface for components that generate reports
    based on validation results.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a ValidationReporter.
        
        Args:
            config: Configuration for the reporter
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"validation_reporter.{self.__class__.__name__}")
    
    @abstractmethod
    def generate_report(
        self,
        validation_results: List[ValidationResult],
        report_format: str = "html",
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a validation report.
        
        Args:
            validation_results: Validation results to include in the report
            report_format: Format of the report (html, markdown, json, etc.)
            include_visualizations: Whether to include visualizations
            
        Returns:
            Report content as a string
        """
        pass
    
    @abstractmethod
    def export_report(
        self,
        validation_results: List[ValidationResult],
        output_path: str,
        report_format: str = "html",
        include_visualizations: bool = True
    ) -> str:
        """
        Export a validation report to a file.
        
        Args:
            validation_results: Validation results to include in the report
            output_path: Path to save the report to
            report_format: Format of the report (html, markdown, json, etc.)
            include_visualizations: Whether to include visualizations
            
        Returns:
            Path to saved report
        """
        pass

class ValidationRepository(ABC):
    """
    Abstract base class for validation data repositories.
    
    This class defines the interface for storing and retrieving validation results
    and related data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a ValidationRepository.
        
        Args:
            config: Configuration for the repository
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"validation_repository.{self.__class__.__name__}")
    
    @abstractmethod
    def save_validation_result(
        self,
        validation_result: ValidationResult
    ) -> str:
        """
        Save a validation result.
        
        Args:
            validation_result: The validation result to save
            
        Returns:
            Identifier for the saved validation result
        """
        pass
    
    @abstractmethod
    def load_validation_result(
        self,
        result_id: str
    ) -> Optional[ValidationResult]:
        """
        Load a validation result by ID.
        
        Args:
            result_id: Identifier for the validation result
            
        Returns:
            ValidationResult if found, None otherwise
        """
        pass
    
    @abstractmethod
    def query_validation_results(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[ValidationResult]:
        """
        Query validation results using filters.
        
        Args:
            filters: Dictionary of filter criteria
            limit: Maximum number of results to return
            
        Returns:
            List of ValidationResult objects matching filters
        """
        pass
    
    @abstractmethod
    def save_benchmark_result(
        self,
        benchmark_result: BenchmarkResult
    ) -> str:
        """
        Save a benchmark result.
        
        Args:
            benchmark_result: The benchmark result to save
            
        Returns:
            Identifier for the saved benchmark result
        """
        pass
    
    @abstractmethod
    def load_benchmark_result(
        self,
        result_id: str
    ) -> Optional[BenchmarkResult]:
        """
        Load a benchmark result by ID.
        
        Args:
            result_id: Identifier for the benchmark result
            
        Returns:
            BenchmarkResult if found, None otherwise
        """
        pass

class BenchmarkValidationFramework:
    """
    Main framework class for benchmark validation.
    
    This class orchestrates the validation process using validators, reporters,
    and repositories to provide a complete validation system.
    """
    
    def __init__(
        self,
        validator: BenchmarkValidator,
        outlier_detector: Optional[OutlierDetector] = None,
        reproducibility_validator: Optional[ReproducibilityValidator] = None,
        certifier: Optional[BenchmarkCertifier] = None,
        reporter: Optional[ValidationReporter] = None,
        repository: Optional[ValidationRepository] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the BenchmarkValidationFramework.
        
        Args:
            validator: Primary validator for benchmarks
            outlier_detector: Detector for outliers in benchmark results
            reproducibility_validator: Validator for reproducibility
            certifier: Certifier for benchmark results
            reporter: Reporter for validation results
            repository: Repository for storing and retrieving data
            config: Framework configuration
        """
        self.validator = validator
        self.outlier_detector = outlier_detector
        self.reproducibility_validator = reproducibility_validator
        self.certifier = certifier
        self.reporter = reporter
        self.repository = repository
        self.config = config or {}
        self.logger = logging.getLogger("benchmark_validation_framework")
        
    def validate(
        self,
        benchmark_result: BenchmarkResult,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        reference_data: Optional[Dict[str, Any]] = None,
        store_result: bool = True
    ) -> ValidationResult:
        """
        Validate a single benchmark result.
        
        Args:
            benchmark_result: The benchmark result to validate
            validation_level: Level of validation to perform
            reference_data: Optional reference data for validation
            store_result: Whether to store the validation result
            
        Returns:
            ValidationResult object with validation results
        """
        self.logger.info(f"Validating benchmark result {benchmark_result.result_id} at level {validation_level.name}")
        
        # Perform validation
        validation_result = self.validator.validate(
            benchmark_result=benchmark_result,
            validation_level=validation_level,
            reference_data=reference_data
        )
        
        # Store validation result if requested
        if store_result and self.repository:
            self.repository.save_validation_result(validation_result)
            
        return validation_result
    
    def validate_batch(
        self,
        benchmark_results: List[BenchmarkResult],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        reference_data: Optional[Dict[str, Any]] = None,
        detect_outliers: bool = True,
        store_results: bool = True
    ) -> List[ValidationResult]:
        """
        Validate a batch of benchmark results.
        
        Args:
            benchmark_results: List of benchmark results to validate
            validation_level: Level of validation to perform
            reference_data: Optional reference data for validation
            detect_outliers: Whether to perform outlier detection
            store_results: Whether to store the validation results
            
        Returns:
            List of ValidationResult objects
        """
        self.logger.info(f"Validating batch of {len(benchmark_results)} benchmark results at level {validation_level.name}")
        
        # Detect outliers if requested
        if detect_outliers and self.outlier_detector:
            outliers = self.outlier_detector.detect_outliers(benchmark_results)
            self.logger.info(f"Detected outliers in {len(outliers)} metrics")
            
            # Add outlier information to reference data
            if reference_data is None:
                reference_data = {}
            reference_data["outliers"] = outliers
        
        # Perform batch validation
        validation_results = self.validator.validate_batch(
            benchmark_results=benchmark_results,
            validation_level=validation_level,
            reference_data=reference_data
        )
        
        # Store validation results if requested
        if store_results and self.repository:
            for validation_result in validation_results:
                self.repository.save_validation_result(validation_result)
                
        return validation_results
    
    def validate_reproducibility(
        self,
        benchmark_results: List[BenchmarkResult],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        reference_data: Optional[Dict[str, Any]] = None,
        store_result: bool = True
    ) -> ValidationResult:
        """
        Validate reproducibility of benchmark results.
        
        Args:
            benchmark_results: List of benchmark results to validate for reproducibility
            validation_level: Level of validation to perform
            reference_data: Optional reference data for validation
            store_result: Whether to store the validation result
            
        Returns:
            ValidationResult for reproducibility assessment
        """
        if not self.reproducibility_validator:
            self.logger.error("No reproducibility validator available")
            return None
            
        self.logger.info(f"Validating reproducibility of {len(benchmark_results)} benchmark results at level {validation_level.name}")
        
        # Perform reproducibility validation
        validation_result = self.reproducibility_validator.validate_reproducibility(
            benchmark_results=benchmark_results,
            validation_level=validation_level,
            reference_data=reference_data
        )
        
        # Store validation result if requested
        if store_result and self.repository:
            self.repository.save_validation_result(validation_result)
            
        return validation_result
    
    def certify_benchmark(
        self,
        benchmark_result: BenchmarkResult,
        validation_results: List[ValidationResult] = None,
        certification_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Certify a benchmark result.
        
        Args:
            benchmark_result: The benchmark result to certify
            validation_results: Previous validation results (optional)
            certification_level: Level of certification to apply
            
        Returns:
            Dictionary with certification details
        """
        if not self.certifier:
            self.logger.error("No certifier available")
            return None
            
        self.logger.info(f"Certifying benchmark result {benchmark_result.result_id} at level {certification_level}")
        
        # If no validation results provided, retrieve or generate them
        if validation_results is None and self.repository:
            # Attempt to retrieve existing validation results
            validation_results = self.repository.query_validation_results(
                filters={"benchmark_result.result_id": benchmark_result.result_id}
            )
            
            # If still no validation results, generate them
            if not validation_results:
                validation_result = self.validate(
                    benchmark_result=benchmark_result,
                    validation_level=ValidationLevel.CERTIFICATION
                )
                validation_results = [validation_result]
        
        # Perform certification
        certification = self.certifier.certify(
            benchmark_result=benchmark_result,
            validation_results=validation_results,
            certification_level=certification_level
        )
        
        return certification
    
    def generate_report(
        self,
        validation_results: List[ValidationResult],
        report_format: str = "html",
        include_visualizations: bool = True,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a validation report.
        
        Args:
            validation_results: Validation results to include in the report
            report_format: Format of the report (html, markdown, json, etc.)
            include_visualizations: Whether to include visualizations
            output_path: Path to save the report to (if None, returns report content)
            
        Returns:
            Report content or path to saved report
        """
        if not self.reporter:
            self.logger.error("No reporter available")
            return None
            
        self.logger.info(f"Generating {report_format} report for {len(validation_results)} validation results")
        
        if output_path:
            # Export report to file
            return self.reporter.export_report(
                validation_results=validation_results,
                output_path=output_path,
                report_format=report_format,
                include_visualizations=include_visualizations
            )
        else:
            # Generate report content
            return self.reporter.generate_report(
                validation_results=validation_results,
                report_format=report_format,
                include_visualizations=include_visualizations
            )