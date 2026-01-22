"""
Simulation Accuracy Validation Framework for IPFS Accelerate.

This package provides tools for validating simulation accuracy,
calibrating simulations, and detecting simulation drift over time.
"""

# Import base classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult, 
    HardwareResult, 
    ValidationResult,
    SimulationValidator,
    SimulationCalibrator,
    DriftDetector,
    ValidationReporter,
    SimulationAccuracyFramework
)

# Import core classes
from duckdb_api.simulation_validation.methodology import ValidationMethodology
from duckdb_api.simulation_validation.simulation_validation_framework import (
    SimulationValidationFramework,
    get_framework_instance
)
from duckdb_api.simulation_validation.comparison.comparison_pipeline import ComparisonPipeline
from duckdb_api.simulation_validation.statistical.statistical_validator import StatisticalValidator

# Import optional modules if available
try:
    from duckdb_api.simulation_validation.calibration.basic_calibrator import BasicSimulationCalibrator
except ImportError:
    pass

try:
    from duckdb_api.simulation_validation.drift_detection.basic_detector import BasicDriftDetector
except ImportError:
    pass

try:
    from duckdb_api.simulation_validation.visualization.validation_reporter import ValidationReporterImpl
except ImportError:
    pass

# Export public classes
__all__ = [
    # Base classes
    "SimulationResult", 
    "HardwareResult", 
    "ValidationResult",
    "SimulationValidator",
    "SimulationCalibrator",
    "DriftDetector",
    "ValidationReporter",
    "SimulationAccuracyFramework",
    
    # Core classes
    "ValidationMethodology",
    "SimulationValidationFramework",
    "get_framework_instance",
    "ComparisonPipeline",
    "StatisticalValidator",
    
    # Optional classes
    "BasicSimulationCalibrator",
    "BasicDriftDetector",
    "ValidationReporterImpl"
]