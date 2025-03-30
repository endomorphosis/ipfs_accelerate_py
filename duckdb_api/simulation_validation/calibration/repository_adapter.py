"""
Repository adapter for integrating the calibration system with DuckDB.

This module provides adapter classes that connect the existing calibration components
with the DuckDB repository for persistent storage and retrieval.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import json

from .basic_calibrator import BasicCalibrator
from .advanced_calibrator import (
    AdvancedCalibrator, 
    MultiParameterCalibrator, 
    BayesianOptimizationCalibrator,
    NeuralNetworkCalibrator,
    EnsembleCalibrator
)
from .cross_validation import CalibrationCrossValidator
from .parameter_discovery import ParameterDiscovery, AdaptiveCalibrationScheduler
from .uncertainty_quantification import UncertaintyQuantifier
from .calibration_repository import DuckDBCalibrationRepository

# Setup logger
logger = logging.getLogger(__name__)

class CalibratorDuckDBAdapter:
    """
    Adapter for integrating calibrators with DuckDB repository.
    
    This class wraps calibrators (basic or advanced) and connects them
    to the DuckDB repository for persisting calibration results.
    """
    
    def __init__(
        self,
        calibrator: Union[BasicCalibrator, AdvancedCalibrator],
        repository: DuckDBCalibrationRepository,
        calibration_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the calibrator adapter.
        
        Args:
            calibrator: The calibrator instance to adapt
            repository: The DuckDB repository for storing results
            calibration_id: Optional calibration ID (generated if not provided)
            metadata: Optional metadata for this calibration
        """
        self.calibrator = calibrator
        self.repository = repository
        self.calibration_id = calibration_id or f"cal-{uuid.uuid4()}"
        self.metadata = metadata or {}
        self.start_time = None
        self.end_time = None
    
    def calibrate(
        self, 
        simulation_results: List[Dict[str, Any]], 
        hardware_results: List[Dict[str, Any]],
        simulation_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        dataset_id: Optional[str] = None,
        hardware_id: Optional[str] = None,
        simulation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform calibration and store results in the repository.
        
        Args:
            simulation_results: List of simulation result dictionaries
            hardware_results: List of hardware result dictionaries
            simulation_parameters: Current simulation parameters
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
            dataset_id: Optional identifier for the dataset
            hardware_id: Optional identifier for the hardware
            simulation_id: Optional identifier for the simulation configuration
            
        Returns:
            Dictionary with calibration results
        """
        # Record start time
        self.start_time = datetime.now()
        
        # Calculate error before calibration
        error_before = self._calculate_error(simulation_results, hardware_results, simulation_parameters)
        
        # Perform calibration
        calibrated_parameters = self.calibrator.calibrate(
            simulation_results, hardware_results, simulation_parameters, parameter_bounds
        )
        
        # Record end time
        self.end_time = datetime.now()
        runtime_seconds = (self.end_time - self.start_time).total_seconds()
        
        # Calculate error after calibration
        error_after = self._calculate_error(simulation_results, hardware_results, calibrated_parameters)
        
        # Store parameter values in repository
        for name, value in calibrated_parameters.items():
            if isinstance(value, (int, float)):
                parameter = {
                    'timestamp': self.end_time,
                    'name': name,
                    'value': value,
                    'uncertainty': 0.0,  # Will be updated later with uncertainty quantification
                    'description': f"Calibrated parameter {name}",
                    'source': self.calibrator.__class__.__name__,
                    'calibration_id': self.calibration_id,
                    'metadata': {
                        'original_value': simulation_parameters.get(name)
                    }
                }
                
                self.repository.store_parameter(parameter)
        
        # Store calibration result
        result = {
            'timestamp': self.end_time,
            'calibration_id': self.calibration_id,
            'error_before': error_before,
            'error_after': error_after,
            'error_reduction_percent': ((error_before - error_after) / error_before * 100) if error_before > 0 else 0.0,
            'calibrator_type': self.calibrator.__class__.__name__,
            'iterations': getattr(self.calibrator, 'max_iterations', 0),
            'converged': error_after < error_before,
            'runtime_seconds': runtime_seconds,
            'dataset_id': dataset_id,
            'hardware_id': hardware_id,
            'simulation_id': simulation_id,
            'metadata': self.metadata
        }
        
        self.repository.store_calibration_result(result)
        
        # Create and store a calibration history entry
        history_entry = {
            'timestamp': self.end_time,
            'calibration_id': self.calibration_id,
            'user_id': self.metadata.get('user_id'),
            'calibrator_type': self.calibrator.__class__.__name__,
            'dataset_size': len(simulation_results),
            'hardware_platforms': hardware_id.split(',') if hardware_id else None,
            'simulation_config': simulation_id,
            'best_parameters': calibrated_parameters,
            'final_error': error_after,
            'improvement_percent': result['error_reduction_percent'],
            'description': self.metadata.get('description', 'Calibration run'),
            'tags': self.metadata.get('tags', []),
            'status': 'completed',
            'metadata': self.metadata
        }
        
        self.repository.store_calibration_history(history_entry)
        
        # Return the calibration results along with metadata
        return {
            'calibration_id': self.calibration_id,
            'parameters': calibrated_parameters,
            'error_before': error_before,
            'error_after': error_after,
            'improvement_percent': result['error_reduction_percent'],
            'runtime_seconds': runtime_seconds
        }
    
    def _calculate_error(
        self, 
        simulation_results: List[Dict[str, Any]], 
        hardware_results: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> float:
        """
        Calculate error between adjusted simulation results and hardware results.
        
        Args:
            simulation_results: List of simulation result dictionaries
            hardware_results: List of hardware result dictionaries
            parameters: Parameter dictionary
            
        Returns:
            Error value
        """
        # Use the calibrator's built-in error calculation if available
        if hasattr(self.calibrator, '_calculate_error'):
            return self.calibrator._calculate_error(simulation_results, hardware_results, parameters)
        
        # Otherwise, implement a default error calculation
        # Extract metrics
        sim_metrics = {}
        hw_metrics = {}
        
        for result in simulation_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not key.startswith('_'):
                    if key not in sim_metrics:
                        sim_metrics[key] = []
                    sim_metrics[key].append(float(value))
        
        for result in hardware_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not key.startswith('_'):
                    if key not in hw_metrics:
                        hw_metrics[key] = []
                    hw_metrics[key].append(float(value))
        
        # Calculate total error across metrics
        total_error = 0.0
        count = 0
        
        for metric in set(sim_metrics.keys()).intersection(hw_metrics.keys()):
            sim_values = sim_metrics[metric]
            hw_values = hw_metrics[metric]
            
            # Apply parameters to adjust simulation values
            adjusted_sim_values = self._apply_parameters_to_metric(sim_values, parameters, metric)
            
            # Calculate Mean Squared Error
            try:
                import numpy as np
                mse = np.mean((np.array(adjusted_sim_values) - np.array(hw_values)) ** 2)
                total_error += mse
                count += 1
            except Exception as e:
                logger.warning(f"Error calculating MSE for metric {metric}: {str(e)}")
        
        # Return average error across metrics
        return total_error / count if count > 0 else 0.0
    
    def _apply_parameters_to_metric(
        self, 
        sim_values: List[float], 
        parameters: Dict[str, Any], 
        metric_name: str
    ) -> List[float]:
        """
        Apply parameters to adjust simulation values for a metric.
        
        Args:
            sim_values: List of simulation values for the metric
            parameters: Current parameter dictionary
            metric_name: Name of the metric
            
        Returns:
            List of adjusted simulation values
        """
        try:
            import numpy as np
            adjusted_values = np.array(sim_values).copy()
            
            # Apply metric-specific parameters
            scale_param = f"{metric_name}_scale"
            offset_param = f"{metric_name}_offset"
            
            if scale_param in parameters:
                adjusted_values *= parameters[scale_param]
            
            if offset_param in parameters:
                adjusted_values += parameters[offset_param]
            
            # Apply global parameters
            if "global_scale" in parameters:
                adjusted_values *= parameters["global_scale"]
            
            if "global_offset" in parameters:
                adjusted_values += parameters["global_offset"]
            
            return adjusted_values.tolist()
        except Exception as e:
            logger.error(f"Error applying parameters to metric: {str(e)}")
            return sim_values


class CrossValidatorDuckDBAdapter:
    """
    Adapter for integrating cross-validators with DuckDB repository.
    
    This class wraps the CalibrationCrossValidator and connects it
    to the DuckDB repository for persisting validation results.
    """
    
    def __init__(
        self,
        cross_validator: CalibrationCrossValidator,
        repository: DuckDBCalibrationRepository,
        validation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the cross-validator adapter.
        
        Args:
            cross_validator: The cross-validator instance to adapt
            repository: The DuckDB repository for storing results
            validation_id: Optional validation ID (generated if not provided)
            metadata: Optional metadata for this validation
        """
        self.cross_validator = cross_validator
        self.repository = repository
        self.validation_id = validation_id or f"val-{uuid.uuid4()}"
        self.metadata = metadata or {}
    
    def cross_validate(
        self, 
        simulation_results: List[Dict[str, Any]], 
        hardware_results: List[Dict[str, Any]],
        initial_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        group_by: Optional[str] = None,
        calibration_id: Optional[str] = None,
        dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation and store results in the repository.
        
        Args:
            simulation_results: List of simulation result dictionaries
            hardware_results: List of hardware result dictionaries
            initial_parameters: Initial parameter dictionary
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) ranges
            group_by: Optional field to group results by
            calibration_id: Optional ID to associate with a calibration
            dataset_id: Optional identifier for the dataset
            
        Returns:
            Dictionary with cross-validation results
        """
        # Run cross-validation
        result = self.cross_validator.cross_validate(
            simulation_results, hardware_results, initial_parameters, 
            parameter_bounds, group_by
        )
        
        if result.get("status") != "success":
            logger.warning(f"Cross-validation failed: {result.get('error', 'Unknown error')}")
            return result
        
        # Store cross-validation results in repository
        for fold_result in result.get("fold_results", []):
            cv_record = {
                'timestamp': datetime.now(),
                'validation_id': self.validation_id,
                'calibration_id': calibration_id,
                'fold': fold_result.get("fold"),
                'train_error': fold_result.get("train_error"),
                'validation_error': fold_result.get("val_error"),
                'generalization_gap': fold_result.get("val_error") - fold_result.get("train_error") 
                                     if "val_error" in fold_result and "train_error" in fold_result else None,
                'dataset_id': dataset_id,
                'calibrator_type': self.cross_validator.calibrator_type,
                'parameters': fold_result.get("calibrated_parameters"),
                'metadata': self.metadata
            }
            
            self.repository.store_cross_validation_result(cv_record)
        
        # Store recommended parameters if they exist
        if "recommended_parameters" in result:
            for name, value in result["recommended_parameters"].items():
                if isinstance(value, (int, float)):
                    parameter = {
                        'timestamp': datetime.now(),
                        'name': name,
                        'value': value,
                        'uncertainty': result.get("parameter_stability", {}).get(name, {}).get("std", 0.0),
                        'description': f"Cross-validated parameter {name}",
                        'source': f"CrossValidator_{self.cross_validator.calibrator_type}",
                        'calibration_id': calibration_id,
                        'metadata': {
                            'validation_id': self.validation_id,
                            'stability': result.get("parameter_stability", {}).get(name, {}).get("stability", "unknown")
                        }
                    }
                    
                    self.repository.store_parameter(parameter)
        
        # Add validation ID to result
        result["validation_id"] = self.validation_id
        
        return result


class ParameterDiscoveryDuckDBAdapter:
    """
    Adapter for integrating parameter discovery with DuckDB repository.
    
    This class wraps the ParameterDiscovery and connects it
    to the DuckDB repository for persisting sensitivity results.
    """
    
    def __init__(
        self,
        parameter_discovery: ParameterDiscovery,
        repository: DuckDBCalibrationRepository,
        analysis_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the parameter discovery adapter.
        
        Args:
            parameter_discovery: The parameter discovery instance to adapt
            repository: The DuckDB repository for storing results
            analysis_id: Optional analysis ID (generated if not provided)
            metadata: Optional metadata for this analysis
        """
        self.parameter_discovery = parameter_discovery
        self.repository = repository
        self.analysis_id = analysis_id or f"analysis-{uuid.uuid4()}"
        self.metadata = metadata or {}
    
    def discover_parameters(
        self, 
        error_function: Callable,
        initial_parameters: Dict[str, Any],
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        calibration_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform parameter discovery and store results in the repository.
        
        Args:
            error_function: Function that takes parameters and returns error value
            initial_parameters: Initial parameter dictionary
            parameter_ranges: Optional dictionary mapping parameter names to (min, max) ranges
            calibration_id: Optional ID to associate with a calibration
            
        Returns:
            Dictionary with parameter discovery results
        """
        # Run parameter discovery
        result = self.parameter_discovery.discover_parameters(
            error_function, initial_parameters, parameter_ranges
        )
        
        # Store sensitivity results in repository
        for i, param_name in enumerate(result.get("sensitive_parameters", [])):
            analysis = result.get("analysis", {}).get(param_name, {})
            
            sensitivity_record = {
                'timestamp': datetime.now(),
                'parameter_name': param_name,
                'sensitivity': analysis.get("sensitivity", 0.0),
                'relative_sensitivity': analysis.get("sensitivity", 0.0),  # May need conversion
                'non_linearity': 0.0,  # Not provided by current implementation
                'analysis_id': self.analysis_id,
                'calibration_id': calibration_id,
                'importance_rank': i + 1,
                'threshold_value': self.parameter_discovery.sensitivity_threshold,
                'metadata': {
                    **self.metadata,
                    'current_value': analysis.get("current_value"),
                    'range': analysis.get("range"),
                    'mean_error': analysis.get("mean_error"),
                    'std_error': analysis.get("std_error"),
                    'potential_improvement': analysis.get("potential_improvement")
                }
            }
            
            self.repository.store_parameter_sensitivity(sensitivity_record)
        
        # Store optimal parameters if available
        if "optimal_parameters" in result:
            for name, value in result["optimal_parameters"].items():
                if isinstance(value, (int, float)):
                    analysis = result.get("analysis", {}).get(name, {})
                    is_sensitive = name in result.get("sensitive_parameters", [])
                    
                    parameter = {
                        'timestamp': datetime.now(),
                        'name': name,
                        'value': value,
                        'uncertainty': analysis.get("std_error", 0.0) if is_sensitive else 0.0,
                        'description': f"{'Sensitive' if is_sensitive else 'Non-sensitive'} parameter {name}",
                        'source': f"ParameterDiscovery",
                        'calibration_id': calibration_id,
                        'metadata': {
                            'analysis_id': self.analysis_id,
                            'is_sensitive': is_sensitive,
                            'potential_improvement': analysis.get("potential_improvement") if is_sensitive else 0.0
                        }
                    }
                    
                    self.repository.store_parameter(parameter)
        
        # Add analysis ID to result
        result["analysis_id"] = self.analysis_id
        
        return result


class UncertaintyQuantifierDuckDBAdapter:
    """
    Adapter for integrating uncertainty quantification with DuckDB repository.
    
    This class wraps the UncertaintyQuantifier and connects it
    to the DuckDB repository for persisting uncertainty results.
    """
    
    def __init__(
        self,
        uncertainty_quantifier: UncertaintyQuantifier,
        repository: DuckDBCalibrationRepository,
        analysis_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the uncertainty quantifier adapter.
        
        Args:
            uncertainty_quantifier: The uncertainty quantifier instance to adapt
            repository: The DuckDB repository for storing results
            analysis_id: Optional analysis ID (generated if not provided)
            metadata: Optional metadata for this analysis
        """
        self.uncertainty_quantifier = uncertainty_quantifier
        self.repository = repository
        self.analysis_id = analysis_id or f"uncertainty-{uuid.uuid4()}"
        self.metadata = metadata or {}
    
    def quantify_parameter_uncertainty(
        self, 
        parameter_sets: List[Dict[str, Any]],
        calibration_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Quantify parameter uncertainty and store results in the repository.
        
        Args:
            parameter_sets: List of parameter dictionaries
            calibration_id: Optional ID to associate with a calibration
            
        Returns:
            Dictionary with uncertainty quantification results
        """
        # Run uncertainty quantification
        result = self.uncertainty_quantifier.quantify_parameter_uncertainty(parameter_sets)
        
        if result.get("status") != "success":
            logger.warning(f"Uncertainty quantification failed: {result.get('error', 'Unknown error')}")
            return result
        
        # Store uncertainty results in repository
        for param_name, metrics in result.get("parameter_uncertainty", {}).items():
            uncertainty_record = {
                'timestamp': datetime.now(),
                'parameter_name': param_name,
                'mean_value': metrics.get("mean"),
                'std_value': metrics.get("std"),
                'cv_value': metrics.get("cv"),
                'ci_lower': metrics.get("ci_lower"),
                'ci_upper': metrics.get("ci_upper"),
                'uncertainty_level': metrics.get("uncertainty_level"),
                'analysis_id': self.analysis_id,
                'calibration_id': calibration_id,
                'confidence_level': self.uncertainty_quantifier.confidence_level,
                'sample_size': metrics.get("sample_size"),
                'metadata': self.metadata
            }
            
            self.repository.store_uncertainty_quantification(uncertainty_record)
        
        # Add analysis ID to result
        result["analysis_id"] = self.analysis_id
        
        return result
    
    def sensitivity_analysis(
        self, 
        parameter_uncertainty: Dict[str, Dict[str, Any]],
        simulation_results: List[Dict[str, Any]],
        error_function: Callable[[Dict[str, Any]], float],
        perturbation_factor: float = 0.1,
        calibration_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis and store results in the repository.
        
        Args:
            parameter_uncertainty: Parameter uncertainty metrics dictionary
            simulation_results: List of simulation result dictionaries
            error_function: Function that takes parameters and returns error value
            perturbation_factor: Factor for parameter perturbation
            calibration_id: Optional ID to associate with a calibration
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        # Run sensitivity analysis
        result = self.uncertainty_quantifier.sensitivity_analysis(
            parameter_uncertainty, simulation_results, error_function, perturbation_factor
        )
        
        # Store sensitivity results in repository
        for param_name, metrics in result.get("sensitivity", {}).items():
            sensitivity_record = {
                'timestamp': datetime.now(),
                'parameter_name': param_name,
                'sensitivity': metrics.get("sensitivity"),
                'relative_sensitivity': metrics.get("relative_sensitivity"),
                'non_linearity': metrics.get("non_linearity", 0.0),
                'analysis_id': self.analysis_id,
                'calibration_id': calibration_id,
                'importance_rank': result.get("sorted_parameters", []).index(param_name) + 1 
                                  if param_name in result.get("sorted_parameters", []) else None,
                'threshold_value': metrics.get("threshold", 0.0),
                'metadata': {
                    **self.metadata,
                    'sensitivity_level': metrics.get("sensitivity_level"),
                    'mean': metrics.get("mean"),
                    'perturbation': metrics.get("perturbation"),
                    'is_critical': param_name in result.get("critical_parameters", [])
                }
            }
            
            self.repository.store_parameter_sensitivity(sensitivity_record)
        
        # Add analysis ID to result
        result["analysis_id"] = self.analysis_id
        
        return result


class SchedulerDuckDBAdapter:
    """
    Adapter for integrating calibration scheduler with DuckDB repository.
    
    This class wraps the AdaptiveCalibrationScheduler and connects it
    to the DuckDB repository for persisting drift and scheduling data.
    """
    
    def __init__(
        self,
        scheduler: AdaptiveCalibrationScheduler,
        repository: DuckDBCalibrationRepository,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the scheduler adapter.
        
        Args:
            scheduler: The calibration scheduler instance to adapt
            repository: The DuckDB repository for storing results
            metadata: Optional metadata
        """
        self.scheduler = scheduler
        self.repository = repository
        self.metadata = metadata or {}
    
    def record_drift(
        self, 
        calibration_id: str,
        drift_value: float,
        drift_type: str = 'parameter',
        threshold_value: Optional[float] = None,
        affected_parameters: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a drift measurement and determine if recalibration is needed.
        
        Args:
            calibration_id: The calibration ID associated with the drift
            drift_value: The measured drift value
            drift_type: Type of drift (parameter, data, environment)
            threshold_value: Optional threshold value for drift
            affected_parameters: Optional list of affected parameters
            description: Optional description of the drift
            
        Returns:
            Dictionary with drift record information
        """
        # Use scheduler's threshold if not provided
        if threshold_value is None:
            threshold_value = self.scheduler.drift_threshold
        
        # Determine if recalibration is needed
        requires_recalibration = drift_value > threshold_value
        
        # Create drift record
        drift_record = {
            'timestamp': datetime.now(),
            'calibration_id': calibration_id,
            'drift_value': drift_value,
            'drift_type': drift_type,
            'threshold_value': threshold_value,
            'requires_recalibration': requires_recalibration,
            'affected_parameters': affected_parameters or [],
            'description': description or f"Detected {drift_type} drift of {drift_value:.4f}",
            'metadata': self.metadata
        }
        
        # Store in repository
        drift_id = self.repository.store_calibration_drift(drift_record)
        
        # Call scheduler's should_calibrate to update internal state
        should_calibrate, reason = self.scheduler.should_calibrate(drift_value=drift_value)
        
        # Return drift information
        drift_record['id'] = drift_id
        drift_record['should_calibrate'] = should_calibrate
        drift_record['reason'] = reason
        
        return drift_record
    
    def record_calibration(
        self, 
        calibration_id: str,
        error_before: Optional[float] = None,
        error_after: Optional[float] = None,
        drift_value: Optional[float] = None,
        calibration_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Record that a calibration was performed and update scheduling.
        
        Args:
            calibration_id: The calibration ID
            error_before: Optional error value before calibration
            error_after: Optional error value after calibration
            drift_value: Optional drift value that triggered calibration
            calibration_time: Optional time taken for calibration in seconds
            
        Returns:
            Dictionary with scheduling information
        """
        # Call scheduler's record_calibration to update internal state
        self.scheduler.record_calibration(
            error_before=error_before,
            error_after=error_after,
            drift_value=drift_value,
            calibration_time=calibration_time
        )
        
        # Get next scheduled calibration from scheduler
        next_scheduled = self.scheduler.get_next_scheduled_calibration()
        
        # Return scheduling information
        return {
            'calibration_id': calibration_id,
            'recorded_timestamp': datetime.now().isoformat(),
            'next_scheduled_calibration': next_scheduled,
            'current_interval_hours': self.scheduler.schedule_data["current_interval_hours"]
        }