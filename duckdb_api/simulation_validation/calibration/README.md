# Calibration System Documentation

## Overview

The Calibration System provides comprehensive functionality for calibrating simulation parameters to improve the accuracy of simulation results compared to real hardware measurements. It includes various calibration methods, parameter discovery and sensitivity analysis, cross-validation for parameter tuning, and uncertainty quantification.

## Key Components

### Basic Calibrator

The `BasicCalibrator` provides a simple calibration approach with scaling and offset adjustments based on observed errors between simulation and hardware results.

Key features:
- Simple parameter adjustment using scaling and offset
- Error calculation and evaluation of calibration effectiveness
- Easily customizable for different metrics and parameter types

### Advanced Calibrators

The `advanced_calibrator.py` module provides several advanced calibration methods:

#### MultiParameterCalibrator

Optimizes multiple parameters simultaneously using numerical optimization techniques:
- Leverages SciPy optimization for L-BFGS-B algorithm when available
- Falls back to custom grid search optimization when SciPy is not available
- Handles parameter bounds and constraints

#### BayesianOptimizationCalibrator

Uses Bayesian optimization with Gaussian Process Regression for efficient parameter search:
- Builds surrogate model of the error function for efficient exploration
- Balances exploration and exploitation through acquisition function
- Adapts to the shape of the error function

#### NeuralNetworkCalibrator

Uses neural networks for parameter optimization:
- Trains a neural network to predict error based on parameter values
- Uses gradient-based optimization on the neural network model
- Provides robust optimization for complex parameter landscapes

#### EnsembleCalibrator

Combines multiple calibration methods for robust results:
- Uses multiple calibration methods in parallel
- Combines results based on calibration effectiveness
- Weights methods by inverse error for optimal results

### Parameter Discovery

The `parameter_discovery.py` module provides tools for discovering and analyzing parameters that affect simulation accuracy:

#### ParameterDiscovery

Discovers parameters that significantly affect simulation accuracy:
- Explores parameter ranges to identify sensitive parameters
- Performs sensitivity analysis to quantify impact
- Provides recommendations for parameter optimization

#### AdaptiveCalibrationScheduler

Schedules calibration operations based on drift detection:
- Determines when calibration should be performed
- Adapts calibration frequency based on error and drift
- Tracks calibration effectiveness over time

### Cross-Validation

The `cross_validation.py` module provides tools for validating calibration parameters:

#### CalibrationCrossValidator

Validates calibration parameters using cross-validation:
- Splits data into training and validation sets
- Evaluates parameters on validation set for generalization
- Provides recommended parameters based on cross-validation
- Generates visualizations for parameter stability

### Uncertainty Quantification

The `uncertainty_quantification.py` module provides tools for quantifying uncertainty in calibration parameters:

#### UncertaintyQuantifier

Quantifies uncertainty in calibration parameters:
- Analyzes parameter uncertainty using statistical methods
- Propagates uncertainty to simulation results
- Estimates reliability of calibration based on uncertainty
- Performs sensitivity analysis to identify critical parameters

## Command-Line Interface

The `run_calibration.py` script provides a command-line interface for running the calibration system:

```bash
# Run basic calibration
python -m duckdb_api.simulation_validation.calibration.run_calibration \
  --sim-file simulation_results.json \
  --hw-file hardware_results.json \
  --param-file initial_parameters.json \
  --output-file calibrated_parameters.json \
  calibrate --calibrator basic

# Run parameter discovery
python -m duckdb_api.simulation_validation.calibration.run_calibration \
  --sim-file simulation_results.json \
  --hw-file hardware_results.json \
  --param-file initial_parameters.json \
  --output-file discovered_parameters.json \
  discover --discovery-iterations 200

# Run cross-validation
python -m duckdb_api.simulation_validation.calibration.run_calibration \
  --sim-file simulation_results.json \
  --hw-file hardware_results.json \
  --param-file initial_parameters.json \
  --output-file validated_parameters.json \
  cross-validate --n-splits 5 --calibrator multi_parameter

# Run uncertainty quantification
python -m duckdb_api.simulation_validation.calibration.run_calibration \
  --sim-file simulation_results.json \
  --hw-file hardware_results.json \
  --param-file initial_parameters.json \
  --parameter-sets-file parameter_sets.json \
  uncertainty --confidence-level 0.95 --report-file uncertainty_report.md
```

## Usage Examples

### Basic Calibration

```python
from duckdb_api.simulation_validation.calibration.basic_calibrator import BasicCalibrator

# Create calibrator
calibrator = BasicCalibrator(learning_rate=0.1, max_iterations=100)

# Run calibration
calibrated_parameters = calibrator.calibrate(
    simulation_results=simulation_results,
    hardware_results=hardware_results,
    simulation_parameters=initial_parameters
)

# Evaluate calibration effectiveness
evaluation = calibrator.evaluate_calibration(
    pre_calibration_results=simulation_results,
    post_calibration_results=post_simulation_results,
    hardware_results=hardware_results
)

print(f"Improvement: {evaluation['overall']['overall_improvement_percentage']:.2f}%")
```

### Advanced Calibration

```python
from duckdb_api.simulation_validation.calibration.advanced_calibrator import (
    MultiParameterCalibrator, 
    BayesianOptimizationCalibrator, 
    EnsembleCalibrator
)

# Create ensemble calibrator with multiple methods
calibrator = EnsembleCalibrator(
    ensemble_methods=["multi_parameter", "bayesian"],
    ensemble_weights=[0.6, 0.4],
    learning_rate=0.1,
    max_iterations=200,
    history_file="calibration_history.json"
)

# Run calibration
calibrated_parameters = calibrator.calibrate(
    simulation_results=simulation_results,
    hardware_results=hardware_results,
    simulation_parameters=initial_parameters,
    parameter_bounds={
        "global_scale": (0.5, 2.0),
        "throughput_scale": (0.8, 1.2)
    }
)

# Analyze calibration history trends
trend_analysis = calibrator.analyze_history_trends()
print(f"Increasing parameters: {trend_analysis['summary']['increasing_params']}")
print(f"Decreasing parameters: {trend_analysis['summary']['decreasing_params']}")
```

### Parameter Discovery

```python
from duckdb_api.simulation_validation.calibration.parameter_discovery import (
    ParameterDiscovery, AdaptiveCalibrationScheduler
)

# Create parameter discovery instance
discovery = ParameterDiscovery(
    sensitivity_threshold=0.01,
    discovery_iterations=100,
    exploration_range=0.5,
    result_file="discovery_results.json"
)

# Define error function
def error_function(params):
    # Calculate error between simulation and hardware results
    # ...
    return error

# Run discovery
result = discovery.discover_parameters(
    error_function=error_function,
    initial_parameters=initial_parameters
)

print(f"Sensitive parameters: {result['sensitive_parameters']}")
print(f"Optimal parameters: {result['optimal_parameters']}")
print(f"Improvement: {result['improvement']:.2f}%")

# Create calibration scheduler
scheduler = AdaptiveCalibrationScheduler(
    min_interval_hours=24.0,
    max_interval_hours=168.0,
    error_threshold=0.1,
    drift_threshold=0.05,
    schedule_file="calibration_schedule.json"
)

# Check if calibration should be performed
should_calibrate, reason = scheduler.should_calibrate(
    current_error=0.15,
    drift_value=0.03
)

print(f"Should calibrate: {should_calibrate}, Reason: {reason}")
```

### Cross-Validation

```python
from duckdb_api.simulation_validation.calibration.cross_validation import (
    CalibrationCrossValidator
)

# Create cross-validator
cross_validator = CalibrationCrossValidator(
    n_splits=5,
    test_size=0.2,
    random_state=42,
    result_file="cross_validation_results.json",
    calibrator_type="multi_parameter"
)

# Run cross-validation
result = cross_validator.cross_validate(
    simulation_results=simulation_results,
    hardware_results=hardware_results,
    initial_parameters=initial_parameters
)

print(f"Average train error: {result['avg_train_error']:.6f}")
print(f"Average validation error: {result['avg_val_error']:.6f}")
print(f"Generalization gap: {result['generalization_gap_percentage']:.2f}%")
print(f"Improvement: {result['improvement']:.2f}%")

# Generate visualization
cross_validator.visualize_results(
    output_file="cross_validation_visualization.png",
    result_id=result["id"]
)

# Analyze generalization
analysis = cross_validator.analyze_generalization()
print(f"Recommendations: {analysis['recommendations']}")
```

### Uncertainty Quantification

```python
from duckdb_api.simulation_validation.calibration.uncertainty_quantification import (
    UncertaintyQuantifier
)

# Create uncertainty quantifier
uncertainty_quantifier = UncertaintyQuantifier(
    confidence_level=0.95,
    n_samples=1000,
    result_file="uncertainty_results.json"
)

# Quantify parameter uncertainty
parameter_uncertainty = uncertainty_quantifier.quantify_parameter_uncertainty(
    parameter_sets=parameter_sets
)

# Define error function
def error_function(params):
    # Calculate error between simulation and hardware results
    # ...
    return error

# Propagate uncertainty
propagation_result = uncertainty_quantifier.propagate_uncertainty(
    parameter_uncertainty=parameter_uncertainty["parameter_uncertainty"],
    simulation_results=simulation_results,
    error_function=error_function
)

print(f"Mean error: {propagation_result['error_statistics']['mean']:.6f}")
print(f"Error CI: [{propagation_result['error_statistics']['ci_lower']:.6f}, "
      f"{propagation_result['error_statistics']['ci_upper']:.6f}]")

# Estimate reliability
reliability_result = uncertainty_quantifier.estimate_reliability(
    parameter_uncertainty=parameter_uncertainty["parameter_uncertainty"],
    simulation_results=simulation_results,
    error_threshold=0.1,
    error_function=error_function
)

print(f"Reliability: {reliability_result['reliability']:.4f} "
      f"({reliability_result['reliability']*100:.1f}%)")

# Run sensitivity analysis
sensitivity_result = uncertainty_quantifier.sensitivity_analysis(
    parameter_uncertainty=parameter_uncertainty["parameter_uncertainty"],
    simulation_results=simulation_results,
    error_function=error_function,
    perturbation_factor=0.1
)

print(f"Critical parameters: {sensitivity_result['critical_parameters']}")

# Generate report
report = uncertainty_quantifier.generate_report(
    result_id=sensitivity_result["id"],
    format="markdown"
)

with open("uncertainty_report.md", "w") as f:
    f.write(report)
```

## Integration with Simulation Validation Framework

The Calibration System integrates seamlessly with the rest of the Simulation Accuracy and Validation Framework. It can be used with the following components:

- **Comparison Pipeline**: For aligning simulation and hardware results before calibration
- **Statistical Validation**: For evaluating calibration effectiveness
- **Drift Detection**: For determining when calibration should be performed
- **Reporting and Visualization**: For visualizing calibration results and effectiveness

Example integration:

```python
from duckdb_api.simulation_validation import get_framework_instance
from duckdb_api.simulation_validation.calibration.advanced_calibrator import EnsembleCalibrator

# Get framework instance
framework = get_framework_instance()

# Load simulation and hardware results
simulation_results = framework.load_simulation_results(simulation_file)
hardware_results = framework.load_hardware_results(hardware_file)

# Align results
aligned_results = framework.align_results(simulation_results, hardware_results)

# Create calibrator
calibrator = EnsembleCalibrator(
    ensemble_methods=["multi_parameter", "bayesian"],
    learning_rate=0.1,
    max_iterations=200
)

# Run calibration
calibrated_parameters = calibrator.calibrate(
    simulation_results=aligned_results["simulation"],
    hardware_results=aligned_results["hardware"],
    simulation_parameters=initial_parameters
)

# Create adjusted simulation results
adjusted_results = framework.apply_parameters_to_results(
    simulation_results=simulation_results,
    parameters=calibrated_parameters
)

# Validate calibration effectiveness
validation_results = framework.validate(
    simulation_results=adjusted_results,
    hardware_results=hardware_results
)

# Generate visualization
framework.visualize_validation(
    validation_results=validation_results,
    output_path="validation_visualization.html"
)
```

## Conclusion

The Calibration System provides comprehensive capabilities for calibrating simulation parameters, with advanced features for parameter discovery, validation, and uncertainty quantification. It integrates seamlessly with the rest of the framework and includes extensive documentation and examples.