#!/usr/bin/env python3
"""
Base classes for advanced analysis methods in the Simulation Accuracy and Validation Framework.

This module defines the base classes and interfaces for implementing advanced analysis methods.
Each analysis method should extend the AnalysisMethod base class and implement its abstract methods.
"""

import abc
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analysis.base")

# Import base validation classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

class AnalysisMethod(abc.ABC):
    """
    Abstract base class for all advanced analysis methods.
    
    This class defines the interface that all analysis methods must implement.
    Each analysis method should extend this class and implement its abstract methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analysis method with configuration.
        
        Args:
            config: Configuration options for the analysis method
        """
        self.config = config or {}
    
    @abc.abstractmethod
    def analyze(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Analyze validation results and generate insights.
        
        Args:
            validation_results: List of validation results to analyze
            
        Returns:
            Dictionary containing analysis results and insights
        """
        pass
    
    @abc.abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the capabilities of this analysis method.
        
        Returns:
            Dictionary describing the capabilities of this analysis method
        """
        pass
    
    @abc.abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """
        Get information about the requirements of this analysis method.
        
        Returns:
            Dictionary describing the requirements of this analysis method,
            such as minimum number of validation results, required metrics, etc.
        """
        pass
    
    def check_requirements(
        self,
        validation_results: List[ValidationResult]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if the provided validation results meet the requirements for analysis.
        
        Args:
            validation_results: List of validation results to check
            
        Returns:
            Tuple containing:
                - Boolean indicating if requirements are met
                - Optional string with error message if requirements aren't met
        """
        requirements = self.get_requirements()
        
        # Check minimum number of validation results
        min_results = requirements.get("min_validation_results", 1)
        if len(validation_results) < min_results:
            return False, (f"Insufficient validation results. "
                          f"Required: {min_results}, Provided: {len(validation_results)}")
        
        # Check required metrics
        if "required_metrics" in requirements:
            for result in validation_results:
                for metric in requirements["required_metrics"]:
                    if (metric not in result.simulation_result.metrics or
                        metric not in result.hardware_result.metrics):
                        return False, f"Required metric '{metric}' not found in all validation results"
        
        # Check time series requirements if applicable
        if requirements.get("time_series_required", False):
            # Check if validation results have timestamps
            for result in validation_results:
                if not hasattr(result, "validation_timestamp") or not result.validation_timestamp:
                    return False, "Validation results must have timestamps for time series analysis"
            
            # Check minimum time span if specified
            if "min_time_span_days" in requirements:
                # Sort by timestamp
                try:
                    import datetime
                    timestamps = [datetime.datetime.fromisoformat(r.validation_timestamp) 
                                 for r in validation_results]
                    if timestamps:
                        min_time = min(timestamps)
                        max_time = max(timestamps)
                        span_days = (max_time - min_time).days
                        if span_days < requirements["min_time_span_days"]:
                            return False, (f"Insufficient time span. "
                                          f"Required: {requirements['min_time_span_days']} days, "
                                          f"Provided: {span_days} days")
                except Exception as e:
                    return False, f"Error checking time span: {e}"
        
        # All requirements are met
        return True, None
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the configuration of this analysis method.
        
        Returns:
            Tuple containing:
                - Boolean indicating if configuration is valid
                - Optional string with error message if configuration is invalid
        """
        # Default implementation that subclasses can override
        return True, None