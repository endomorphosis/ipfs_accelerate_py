#!/usr/bin/env python3
"""
Simulation Validation Methodology for the Simulation Accuracy and Validation Framework.

This module defines a comprehensive methodology for validating hardware simulation accuracy
in the IPFS Accelerate system. It provides standardized processes, metrics, and evaluation
criteria to ensure simulation results closely match real hardware performance.

The methodology is designed to support:
- Consistent validation across different hardware types
- Statistical rigor in evaluating simulation accuracy
- Confidence scoring for simulation results
- Progressive validation workflows
- Integration with the calibration and drift detection subsystems
"""

import os
import logging
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simulation_validation_methodology")

# Import base classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult,
    SimulationValidator,
    SimulationCalibrator,
    DriftDetector
)

class ValidationMethodology:
    """
    Defines the methodology for validating simulation accuracy against real hardware.
    
    This class provides a framework for validation that can be applied consistently
    across different hardware types, model types, and validation scenarios. It defines
    the metrics, workflows, and evaluation criteria used throughout the framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation methodology.
        
        Args:
            config: Configuration options for the methodology
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            # Core validation metrics
            "primary_metrics": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w"
            ],
            "secondary_metrics": [
                "initialization_time_ms",
                "warmup_time_ms",
                "peak_cpu_utilization_percent",
                "sustained_cpu_utilization_percent"
            ],
            
            # Statistical metrics for evaluation
            "statistical_metrics": [
                "mape",                # Mean Absolute Percentage Error
                "rmse",                # Root Mean Square Error
                "pearson_correlation", # Correlation coefficient
                "f1_ranking_score",    # F1 score for ranking accuracy
                "distribution_kl_div"  # KL divergence between distributions
            ],
            
            # Accuracy thresholds
            "accuracy_thresholds": {
                "excellent": 5.0,      # MAPE <= 5%
                "good": 10.0,          # MAPE <= 10%
                "acceptable": 15.0,    # MAPE <= 15%
                "poor": 25.0,          # MAPE <= 25%
                "unacceptable": float('inf')  # MAPE > 25%
            },
            
            # Hardware-specific validation requirements
            "hardware_validation_requirements": {
                "default": {
                    "min_samples": 5,
                    "required_metrics": ["throughput_items_per_second", "average_latency_ms"],
                    "batch_size_validation": True,
                    "precision_validation": True
                },
                # Hardware-specific overrides can be defined here
            },
            
            # Model-specific validation requirements
            "model_validation_requirements": {
                "default": {
                    "min_samples": 5,
                    "required_metrics": ["throughput_items_per_second", "average_latency_ms"],
                    "batch_size_validation": True,
                    "precision_validation": True
                },
                # Model-specific overrides can be defined here
            },
            
            # Confidence scoring parameters
            "confidence_scoring": {
                "sample_size_weight": 0.3,    # Weight of sample size in confidence score
                "recency_weight": 0.2,        # Weight of result recency in confidence score
                "accuracy_weight": 0.5,       # Weight of accuracy in confidence score
                "min_samples_full_confidence": 30,  # Samples needed for full confidence
                "max_age_days_full_confidence": 30  # Maximum age in days for full confidence
            },
            
            # Progressive validation workflow
            "progressive_validation": {
                "stages": [
                    "basic_metrics",          # Basic performance metrics (throughput, latency)
                    "extended_metrics",       # Extended metrics (memory, power)
                    "resource_usage",         # Detailed resource usage
                    "variable_batch_size",    # Tests with varying batch sizes
                    "precision_variants",     # Tests with different precision formats
                    "stress_conditions",      # Tests under stress conditions
                    "long_running"            # Extended duration tests
                ],
                "hardware_stage_requirements": {
                    # Hardware-specific stage requirements can be defined here
                },
                "model_stage_requirements": {
                    # Model-specific stage requirements can be defined here
                }
            },
            
            # Validation protocol options
            "validation_protocols": {
                "standard": {
                    "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                    "statistical_metrics": ["mape", "rmse", "pearson_correlation"],
                    "min_samples": 5,
                    "stages": ["basic_metrics", "extended_metrics"]
                },
                "comprehensive": {
                    "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb", 
                              "power_consumption_w", "initialization_time_ms", "warmup_time_ms"],
                    "statistical_metrics": ["mape", "rmse", "pearson_correlation", "f1_ranking_score", "distribution_kl_div"],
                    "min_samples": 10,
                    "stages": ["basic_metrics", "extended_metrics", "variable_batch_size", "precision_variants"]
                },
                "minimal": {
                    "metrics": ["throughput_items_per_second", "average_latency_ms"],
                    "statistical_metrics": ["mape"],
                    "min_samples": 3,
                    "stages": ["basic_metrics"]
                }
            },
            
            # Metadata tracking for validation
            "metadata_tracking": {
                "hardware_details": ["cpu_model", "gpu_model", "memory_gb", "os_version"],
                "test_environment": ["temperature_c", "background_load", "driver_version"],
                "simulation_details": ["simulation_engine", "version", "configuration_hash"]
            },
            
            # Integration with other subsystems
            "calibration_integration": {
                "auto_calibrate_threshold": 15.0,  # MAPE threshold for automatic calibration
                "calibration_frequency": "weekly",  # How often to run calibration
                "min_samples_before_calibration": 10  # Minimum samples needed before calibration
            },
            
            "drift_detection_integration": {
                "drift_check_frequency": "daily",  # How often to check for drift
                "min_samples_before_drift_check": 10,  # Minimum samples needed before drift check
                "drift_significance_level": 0.05  # P-value threshold for significance
            }
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def get_hardware_validation_requirements(self, hardware_id: str) -> Dict[str, Any]:
        """
        Get validation requirements for a specific hardware type.
        
        Args:
            hardware_id: Identifier for the hardware
            
        Returns:
            Dictionary with validation requirements
        """
        requirements = self.config["hardware_validation_requirements"].get(
            hardware_id, 
            self.config["hardware_validation_requirements"]["default"]
        )
        return requirements
    
    def get_model_validation_requirements(self, model_id: str) -> Dict[str, Any]:
        """
        Get validation requirements for a specific model type.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Dictionary with validation requirements
        """
        requirements = self.config["model_validation_requirements"].get(
            model_id, 
            self.config["model_validation_requirements"]["default"]
        )
        return requirements
    
    def validate_requirements(
        self,
        simulation_result: SimulationResult,
        hardware_result: HardwareResult
    ) -> Dict[str, Any]:
        """
        Validate if simulation and hardware results meet requirements for validation.
        
        Args:
            simulation_result: Simulation result to validate
            hardware_result: Hardware result to compare against
            
        Returns:
            Dictionary with validation status and issues (if any)
        """
        hardware_id = hardware_result.hardware_id
        model_id = hardware_result.model_id
        
        hardware_requirements = self.get_hardware_validation_requirements(hardware_id)
        model_requirements = self.get_model_validation_requirements(model_id)
        
        # Combine requirements (taking the stricter of the two)
        combined_requirements = {
            "required_metrics": list(set(hardware_requirements["required_metrics"] + model_requirements["required_metrics"])),
            "batch_size_validation": hardware_requirements["batch_size_validation"] or model_requirements["batch_size_validation"],
            "precision_validation": hardware_requirements["precision_validation"] or model_requirements["precision_validation"]
        }
        
        issues = []
        
        # Check required metrics in simulation result
        for metric in combined_requirements["required_metrics"]:
            if metric not in simulation_result.metrics or simulation_result.metrics[metric] is None:
                issues.append(f"Missing required metric in simulation result: {metric}")
        
        # Check required metrics in hardware result
        for metric in combined_requirements["required_metrics"]:
            if metric not in hardware_result.metrics or hardware_result.metrics[metric] is None:
                issues.append(f"Missing required metric in hardware result: {metric}")
        
        # Check batch size if required
        if combined_requirements["batch_size_validation"]:
            if simulation_result.batch_size != hardware_result.batch_size:
                issues.append(f"Batch size mismatch: {simulation_result.batch_size} vs {hardware_result.batch_size}")
        
        # Check precision if required
        if combined_requirements["precision_validation"]:
            if simulation_result.precision != hardware_result.precision:
                issues.append(f"Precision mismatch: {simulation_result.precision} vs {hardware_result.precision}")
        
        return {
            "passes_requirements": len(issues) == 0,
            "issues": issues
        }
    
    def calculate_confidence_score(
        self,
        validation_results: List[ValidationResult],
        hardware_id: str,
        model_id: str
    ) -> float:
        """
        Calculate a confidence score for simulation results based on validation history.
        
        Args:
            validation_results: List of validation results to analyze
            hardware_id: Hardware identifier
            model_id: Model identifier
            
        Returns:
            Confidence score (0-1) where 1 is highest confidence
        """
        if not validation_results:
            return 0.0
        
        # Filter validation results for this hardware and model
        filtered_results = [
            result for result in validation_results
            if result.hardware_result.hardware_id == hardware_id and result.hardware_result.model_id == model_id
        ]
        
        if not filtered_results:
            return 0.0
        
        # Get confidence scoring parameters
        params = self.config["confidence_scoring"]
        
        # Calculate sample size component
        sample_count = len(filtered_results)
        sample_size_score = min(1.0, sample_count / params["min_samples_full_confidence"])
        
        # Calculate recency component
        now = datetime.datetime.now()
        timestamps = [datetime.datetime.fromisoformat(result.validation_timestamp) for result in filtered_results]
        most_recent = max(timestamps)
        age_days = (now - most_recent).days
        recency_score = max(0.0, 1.0 - (age_days / params["max_age_days_full_confidence"]))
        
        # Calculate accuracy component
        mape_values = []
        for result in filtered_results:
            for metric in self.config["primary_metrics"]:
                if metric in result.metrics_comparison and "mape" in result.metrics_comparison[metric]:
                    mape = result.metrics_comparison[metric]["mape"]
                    if not np.isnan(mape):
                        mape_values.append(mape)
        
        if mape_values:
            mean_mape = np.mean(mape_values)
            # Convert MAPE to a score (0-1) where lower MAPE means higher score
            thresholds = self.config["accuracy_thresholds"]
            if mean_mape <= thresholds["excellent"]:
                accuracy_score = 1.0
            elif mean_mape <= thresholds["good"]:
                accuracy_score = 0.8
            elif mean_mape <= thresholds["acceptable"]:
                accuracy_score = 0.6
            elif mean_mape <= thresholds["poor"]:
                accuracy_score = 0.3
            else:
                accuracy_score = 0.0
        else:
            accuracy_score = 0.0
        
        # Combine components into overall confidence score
        confidence_score = (
            params["sample_size_weight"] * sample_size_score +
            params["recency_weight"] * recency_score +
            params["accuracy_weight"] * accuracy_score
        )
        
        return confidence_score
    
    def get_validation_protocol(self, protocol_name: str = "standard") -> Dict[str, Any]:
        """
        Get validation protocol configuration by name.
        
        Args:
            protocol_name: Name of the validation protocol
            
        Returns:
            Dictionary with protocol configuration
        """
        if protocol_name not in self.config["validation_protocols"]:
            logger.warning(f"Unknown validation protocol: {protocol_name}, using standard")
            protocol_name = "standard"
        
        return self.config["validation_protocols"][protocol_name]
    
    def create_validation_plan(
        self,
        hardware_id: str,
        model_id: str,
        protocol_name: str = "standard",
        existing_validation_results: Optional[List[ValidationResult]] = None
    ) -> Dict[str, Any]:
        """
        Create a validation plan for a hardware-model combination.
        
        Args:
            hardware_id: Hardware identifier
            model_id: Model identifier
            protocol_name: Name of the validation protocol to use
            existing_validation_results: Optional list of existing validation results
            
        Returns:
            Dictionary with validation plan details
        """
        protocol = self.get_validation_protocol(protocol_name)
        
        # Get hardware and model requirements
        hardware_reqs = self.get_hardware_validation_requirements(hardware_id)
        model_reqs = self.get_model_validation_requirements(model_id)
        
        # Determine stages to include
        stages = protocol["stages"]
        
        # Check current confidence if existing results are provided
        current_confidence = 0.0
        if existing_validation_results:
            current_confidence = self.calculate_confidence_score(
                existing_validation_results, hardware_id, model_id)
        
        # Create plan
        validation_plan = {
            "hardware_id": hardware_id,
            "model_id": model_id,
            "protocol": protocol_name,
            "metrics": protocol["metrics"],
            "statistical_metrics": protocol["statistical_metrics"],
            "stages": stages,
            "current_confidence": current_confidence,
            "min_samples": max(protocol["min_samples"], hardware_reqs["min_samples"], model_reqs["min_samples"]),
            "batch_sizes_to_test": [1, 4, 16, 64] if "variable_batch_size" in stages else [1],
            "precisions_to_test": ["fp32", "fp16", "int8"] if "precision_variants" in stages else ["fp32"],
            "created_at": datetime.datetime.now().isoformat()
        }
        
        return validation_plan
    
    def evaluate_validation_result(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Evaluate a validation result against the methodology criteria.
        
        Args:
            validation_result: Validation result to evaluate
            
        Returns:
            Dictionary with evaluation metrics and assessment
        """
        evaluation = {
            "metrics": {},
            "overall": {}
        }
        
        # Evaluate each primary metric
        for metric in self.config["primary_metrics"]:
            if metric in validation_result.metrics_comparison and "mape" in validation_result.metrics_comparison[metric]:
                mape = validation_result.metrics_comparison[metric]["mape"]
                
                if np.isnan(mape):
                    status = "invalid"
                elif mape <= self.config["accuracy_thresholds"]["excellent"]:
                    status = "excellent"
                elif mape <= self.config["accuracy_thresholds"]["good"]:
                    status = "good"
                elif mape <= self.config["accuracy_thresholds"]["acceptable"]:
                    status = "acceptable"
                elif mape <= self.config["accuracy_thresholds"]["poor"]:
                    status = "poor"
                else:
                    status = "unacceptable"
                
                evaluation["metrics"][metric] = {
                    "mape": mape,
                    "status": status
                }
        
        # Calculate overall MAPE from all metrics
        mape_values = [
            m["mape"] for m in evaluation["metrics"].values()
            if "mape" in m and not np.isnan(m["mape"])
        ]
        
        if mape_values:
            overall_mape = np.mean(mape_values)
            
            if overall_mape <= self.config["accuracy_thresholds"]["excellent"]:
                overall_status = "excellent"
            elif overall_mape <= self.config["accuracy_thresholds"]["good"]:
                overall_status = "good"
            elif overall_mape <= self.config["accuracy_thresholds"]["acceptable"]:
                overall_status = "acceptable"
            elif overall_mape <= self.config["accuracy_thresholds"]["poor"]:
                overall_status = "poor"
            else:
                overall_status = "unacceptable"
            
            evaluation["overall"] = {
                "mape": overall_mape,
                "status": overall_status,
                "calibration_recommended": overall_mape > self.config["calibration_integration"]["auto_calibrate_threshold"]
            }
        else:
            evaluation["overall"] = {
                "mape": float('nan'),
                "status": "invalid",
                "calibration_recommended": False
            }
        
        return evaluation
    
    def aggregate_validation_results(
        self,
        validation_results: List[ValidationResult],
        grouping: str = "hardware_model"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate multiple validation results with grouping.
        
        Args:
            validation_results: List of validation results to aggregate
            grouping: Grouping method ('hardware', 'model', 'hardware_model', or 'all')
            
        Returns:
            Dictionary with aggregated results by group
        """
        aggregated = {}
        
        for val_result in validation_results:
            # Determine group key based on grouping method
            if grouping == "hardware":
                key = val_result.hardware_result.hardware_id
            elif grouping == "model":
                key = val_result.hardware_result.model_id
            elif grouping == "hardware_model":
                key = f"{val_result.hardware_result.hardware_id}_{val_result.hardware_result.model_id}"
            elif grouping == "all":
                key = "all"
            else:
                logger.warning(f"Unknown grouping method: {grouping}, using hardware_model")
                key = f"{val_result.hardware_result.hardware_id}_{val_result.hardware_result.model_id}"
            
            # Initialize group if it doesn't exist
            if key not in aggregated:
                aggregated[key] = {
                    "metrics": {},
                    "overall": {
                        "mape_values": [],
                        "count": 0
                    }
                }
            
            # Collect MAPE values for each primary metric
            for metric in self.config["primary_metrics"]:
                if metric in val_result.metrics_comparison and "mape" in val_result.metrics_comparison[metric]:
                    mape = val_result.metrics_comparison[metric]["mape"]
                    
                    if np.isnan(mape):
                        continue
                    
                    if metric not in aggregated[key]["metrics"]:
                        aggregated[key]["metrics"][metric] = {
                            "mape_values": [],
                            "count": 0
                        }
                    
                    aggregated[key]["metrics"][metric]["mape_values"].append(mape)
                    aggregated[key]["metrics"][metric]["count"] += 1
            
            # Collect overall MAPE
            metric_mapes = []
            for metric in self.config["primary_metrics"]:
                if metric in val_result.metrics_comparison and "mape" in val_result.metrics_comparison[metric]:
                    mape = val_result.metrics_comparison[metric]["mape"]
                    if not np.isnan(mape):
                        metric_mapes.append(mape)
            
            if metric_mapes:
                overall_mape = np.mean(metric_mapes)
                aggregated[key]["overall"]["mape_values"].append(overall_mape)
                aggregated[key]["overall"]["count"] += 1
        
        # Calculate statistics for each group
        for key, group in aggregated.items():
            # Calculate statistics for each metric
            for metric, data in group["metrics"].items():
                if data["mape_values"]:
                    data["mean_mape"] = np.mean(data["mape_values"])
                    data["median_mape"] = np.median(data["mape_values"])
                    data["min_mape"] = np.min(data["mape_values"])
                    data["max_mape"] = np.max(data["mape_values"])
                    data["std_dev_mape"] = np.std(data["mape_values"])
                    
                    if data["mean_mape"] <= self.config["accuracy_thresholds"]["excellent"]:
                        data["status"] = "excellent"
                    elif data["mean_mape"] <= self.config["accuracy_thresholds"]["good"]:
                        data["status"] = "good"
                    elif data["mean_mape"] <= self.config["accuracy_thresholds"]["acceptable"]:
                        data["status"] = "acceptable"
                    elif data["mean_mape"] <= self.config["accuracy_thresholds"]["poor"]:
                        data["status"] = "poor"
                    else:
                        data["status"] = "unacceptable"
            
            # Calculate overall statistics
            if group["overall"]["mape_values"]:
                group["overall"]["mean_mape"] = np.mean(group["overall"]["mape_values"])
                group["overall"]["median_mape"] = np.median(group["overall"]["mape_values"])
                group["overall"]["min_mape"] = np.min(group["overall"]["mape_values"])
                group["overall"]["max_mape"] = np.max(group["overall"]["mape_values"])
                group["overall"]["std_dev_mape"] = np.std(group["overall"]["mape_values"])
                
                if group["overall"]["mean_mape"] <= self.config["accuracy_thresholds"]["excellent"]:
                    group["overall"]["status"] = "excellent"
                elif group["overall"]["mean_mape"] <= self.config["accuracy_thresholds"]["good"]:
                    group["overall"]["status"] = "good"
                elif group["overall"]["mean_mape"] <= self.config["accuracy_thresholds"]["acceptable"]:
                    group["overall"]["status"] = "acceptable"
                elif group["overall"]["mean_mape"] <= self.config["accuracy_thresholds"]["poor"]:
                    group["overall"]["status"] = "poor"
                else:
                    group["overall"]["status"] = "unacceptable"
                
                group["overall"]["calibration_recommended"] = (
                    group["overall"]["mean_mape"] > self.config["calibration_integration"]["auto_calibrate_threshold"]
                )
        
        return aggregated
    
    def check_calibration_needed(
        self,
        validation_results: List[ValidationResult],
        hardware_id: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Check if calibration is needed based on validation results.
        
        Args:
            validation_results: List of validation results
            hardware_id: Hardware identifier
            model_id: Model identifier
            
        Returns:
            Dictionary with calibration recommendation
        """
        if not validation_results:
            return {
                "calibration_recommended": False,
                "reason": "No validation results available"
            }
        
        # Filter validation results for this hardware and model
        filtered_results = [
            result for result in validation_results
            if result.hardware_result.hardware_id == hardware_id and result.hardware_result.model_id == model_id
        ]
        
        if not filtered_results:
            return {
                "calibration_recommended": False,
                "reason": f"No validation results for {hardware_id}/{model_id}"
            }
        
        # Check if we have enough samples
        if len(filtered_results) < self.config["calibration_integration"]["min_samples_before_calibration"]:
            return {
                "calibration_recommended": False,
                "reason": f"Insufficient samples ({len(filtered_results)} < {self.config['calibration_integration']['min_samples_before_calibration']})"
            }
        
        # Calculate overall MAPE
        all_mape_values = []
        for result in filtered_results:
            result_mape_values = []
            for metric in self.config["primary_metrics"]:
                if metric in result.metrics_comparison and "mape" in result.metrics_comparison[metric]:
                    mape = result.metrics_comparison[metric]["mape"]
                    if not np.isnan(mape):
                        result_mape_values.append(mape)
            
            if result_mape_values:
                all_mape_values.append(np.mean(result_mape_values))
        
        if all_mape_values:
            overall_mape = np.mean(all_mape_values)
            calibration_needed = overall_mape > self.config["calibration_integration"]["auto_calibrate_threshold"]
            
            return {
                "calibration_recommended": calibration_needed,
                "overall_mape": overall_mape,
                "threshold": self.config["calibration_integration"]["auto_calibrate_threshold"],
                "reason": (
                    f"Overall MAPE ({overall_mape:.2f}%) exceeds threshold ({self.config['calibration_integration']['auto_calibrate_threshold']:.2f}%)"
                    if calibration_needed
                    else f"Overall MAPE ({overall_mape:.2f}%) below threshold ({self.config['calibration_integration']['auto_calibrate_threshold']:.2f}%)"
                )
            }
        else:
            return {
                "calibration_recommended": False,
                "reason": "No MAPE values available"
            }
    
    def check_drift_detection_needed(
        self,
        validation_results: List[ValidationResult],
        hardware_id: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Check if drift detection is needed based on validation results.
        
        Args:
            validation_results: List of validation results
            hardware_id: Hardware identifier
            model_id: Model identifier
            
        Returns:
            Dictionary with drift detection recommendation
        """
        if not validation_results:
            return {
                "drift_detection_recommended": False,
                "reason": "No validation results available"
            }
        
        # Filter validation results for this hardware and model
        filtered_results = [
            result for result in validation_results
            if result.hardware_result.hardware_id == hardware_id and result.hardware_result.model_id == model_id
        ]
        
        if not filtered_results:
            return {
                "drift_detection_recommended": False,
                "reason": f"No validation results for {hardware_id}/{model_id}"
            }
        
        # Check if we have enough samples
        if len(filtered_results) < self.config["drift_detection_integration"]["min_samples_before_drift_check"]:
            return {
                "drift_detection_recommended": False,
                "reason": f"Insufficient samples ({len(filtered_results)} < {self.config['drift_detection_integration']['min_samples_before_drift_check']})"
            }
        
        # Sort results by timestamp
        sorted_results = sorted(filtered_results, key=lambda x: x.validation_timestamp)
        
        # Check if we have enough samples in both historical and recent periods
        cutoff_point = len(sorted_results) // 2
        if cutoff_point < 3:  # Need at least 3 samples in each period for meaningful comparison
            return {
                "drift_detection_recommended": False,
                "reason": "Insufficient samples for historical/recent comparison"
            }
        
        historical_results = sorted_results[:cutoff_point]
        recent_results = sorted_results[cutoff_point:]
        
        return {
            "drift_detection_recommended": True,
            "historical_count": len(historical_results),
            "recent_count": len(recent_results),
            "historical_period": {
                "start": historical_results[0].validation_timestamp,
                "end": historical_results[-1].validation_timestamp
            },
            "recent_period": {
                "start": recent_results[0].validation_timestamp,
                "end": recent_results[-1].validation_timestamp
            },
            "reason": "Sufficient samples available for drift detection"
        }
    
    def generate_validation_report(
        self,
        validation_results: List[ValidationResult],
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None,
        report_format: str = "markdown"
    ) -> str:
        """
        Generate a validation report based on validation results.
        
        Args:
            validation_results: List of validation results
            hardware_id: Optional filter for hardware ID
            model_id: Optional filter for model ID
            report_format: Output format (markdown or json)
            
        Returns:
            Report content as a string
        """
        # Filter results if hardware_id or model_id is specified
        filtered_results = validation_results
        
        if hardware_id:
            filtered_results = [
                result for result in filtered_results
                if result.hardware_result.hardware_id == hardware_id
            ]
        
        if model_id:
            filtered_results = [
                result for result in filtered_results
                if result.hardware_result.model_id == model_id
            ]
        
        if not filtered_results:
            return "No validation results found matching the specified criteria."
        
        # Aggregate results
        if hardware_id and model_id:
            grouping = "hardware_model"
        elif hardware_id:
            grouping = "model"
        elif model_id:
            grouping = "hardware"
        else:
            grouping = "hardware_model"
        
        aggregated = self.aggregate_validation_results(filtered_results, grouping)
        
        # Generate report based on format
        if report_format == "json":
            import json
            return json.dumps(aggregated, indent=2)
        else:  # markdown
            report = []
            report.append("# Simulation Validation Report\n")
            
            # Report metadata
            report.append("## Report Overview\n")
            report.append(f"- **Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"- **Total Validation Results:** {len(filtered_results)}")
            if hardware_id:
                report.append(f"- **Hardware:** {hardware_id}")
            if model_id:
                report.append(f"- **Model:** {model_id}")
            report.append(f"- **Grouping:** {grouping}")
            report.append("")
            
            # Summary table
            report.append("## Summary\n")
            report.append("| Group | Validation Count | Overall MAPE | Status | Calibration Recommended |")
            report.append("|-------|-----------------|-------------|--------|-------------------------|")
            
            for key, group in aggregated.items():
                overall = group["overall"]
                if "mean_mape" in overall:
                    report.append(
                        f"| {key} | {overall['count']} | "
                        f"{overall['mean_mape']:.2f}% | {overall['status'].capitalize()} | "
                        f"{'Yes' if overall.get('calibration_recommended', False) else 'No'} |"
                    )
            
            report.append("")
            
            # Detailed metrics
            report.append("## Detailed Metrics\n")
            
            for key, group in aggregated.items():
                report.append(f"### {key}\n")
                
                report.append("#### Metrics\n")
                report.append("| Metric | Mean MAPE | Median MAPE | Min MAPE | Max MAPE | Std Dev | Status |")
                report.append("|--------|-----------|-------------|---------|---------|---------|--------|")
                
                for metric, data in group["metrics"].items():
                    if "mean_mape" in data:
                        report.append(
                            f"| {metric} | {data['mean_mape']:.2f}% | {data['median_mape']:.2f}% | "
                            f"{data['min_mape']:.2f}% | {data['max_mape']:.2f}% | {data['std_dev_mape']:.2f}% | "
                            f"{data['status'].capitalize()} |"
                        )
                
                report.append("")
                
                report.append("#### Overall Assessment\n")
                overall = group["overall"]
                if "mean_mape" in overall:
                    report.append(f"- **Overall MAPE:** {overall['mean_mape']:.2f}%")
                    report.append(f"- **Status:** {overall['status'].capitalize()}")
                    report.append(f"- **Calibration Recommended:** {'Yes' if overall.get('calibration_recommended', False) else 'No'}")
                    
                    if "mean_mape" in overall and overall["mean_mape"] > self.config["calibration_integration"]["auto_calibrate_threshold"]:
                        report.append(
                            f"- **Recommendation:** Calibration is recommended as the overall MAPE "
                            f"({overall['mean_mape']:.2f}%) exceeds the threshold "
                            f"({self.config['calibration_integration']['auto_calibrate_threshold']:.2f}%)."
                        )
                    else:
                        report.append("- **Recommendation:** No calibration needed at this time.")
                
                report.append("")
            
            return "\n".join(report)


def get_validation_methodology_instance(config_path: Optional[str] = None) -> ValidationMethodology:
    """
    Get an instance of the ValidationMethodology class.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ValidationMethodology instance
    """
    # Load configuration from file if provided
    config = None
    if config_path:
        import json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading configuration from {config_path}: {e}")
    
    return ValidationMethodology(config)