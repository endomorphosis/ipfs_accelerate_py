"""
Calibration system for the Simulation Accuracy and Validation Framework.

This module provides components for calibrating simulation parameters
to improve the accuracy of simulation results compared to real hardware measurements.
"""

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

# DuckDB integration components
from .calibration_repository import DuckDBCalibrationRepository
from .repository_adapter import (
    CalibratorDuckDBAdapter,
    CrossValidatorDuckDBAdapter,
    ParameterDiscoveryDuckDBAdapter,
    UncertaintyQuantifierDuckDBAdapter,
    SchedulerDuckDBAdapter
)

__all__ = [
    'BasicCalibrator',
    'AdvancedCalibrator',
    'MultiParameterCalibrator',
    'BayesianOptimizationCalibrator',
    'NeuralNetworkCalibrator',
    'EnsembleCalibrator',
    'CalibrationCrossValidator',
    'ParameterDiscovery',
    'AdaptiveCalibrationScheduler',
    'UncertaintyQuantifier',
    'DuckDBCalibrationRepository',
    'CalibratorDuckDBAdapter',
    'CrossValidatorDuckDBAdapter',
    'ParameterDiscoveryDuckDBAdapter',
    'UncertaintyQuantifierDuckDBAdapter',
    'SchedulerDuckDBAdapter',
]