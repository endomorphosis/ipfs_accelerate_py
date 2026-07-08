# Calibration DuckDB Integration Guide

This guide explains how to use the DuckDB integration with the Simulation Calibration Framework.

## Overview

The Calibration DuckDB Integration provides persistent storage and retrieval capabilities for simulation calibration data, including:

- Calibration parameters
- Calibration results
- Cross-validation data
- Parameter sensitivity analysis
- Uncertainty quantification
- Calibration history

The integration allows tracking of calibration performance over time, comparing different calibration methods, and providing robust recommendations for simulation parameters.

## Components

The integration consists of several components:

1. **DuckDBCalibrationRepository**: Core repository class that handles database operations
2. **Adapter Classes**: Connect existing calibration components with the repository
   - `CalibratorDuckDBAdapter`: For basic and advanced calibrators
   - `CrossValidatorDuckDBAdapter`: For cross-validation
   - `ParameterDiscoveryDuckDBAdapter`: For parameter discovery and sensitivity analysis
   - `UncertaintyQuantifierDuckDBAdapter`: For uncertainty quantification
   - `SchedulerDuckDBAdapter`: For calibration scheduling and drift detection

## Getting Started

### Installation

The DuckDB integration is built into the IPFS Accelerate Python framework. Make sure DuckDB is installed:

```bash
pip install duckdb pandas numpy
```

### Basic Usage

Here's a simple example of using the calibration system with DuckDB:

```python
from duckdb_api.simulation_validation.calibration import (
    BasicCalibrator,
    DuckDBCalibrationRepository,
    CalibratorDuckDBAdapter
)

# Create repository
repository = DuckDBCalibrationRepository(db_path="calibration.duckdb")

# Create calibrator with repository integration
calibrator = BasicCalibrator()
adapter = CalibratorDuckDBAdapter(
    calibrator=calibrator,
    repository=repository,
    calibration_id="example-calibration",
    metadata={"description": "Example calibration run"}
)

# Run calibration
result = adapter.calibrate(
    simulation_results=simulation_data,
    hardware_results=hardware_data,
    simulation_parameters=initial_params,
    hardware_id="cuda"
)

print(f"Calibration completed with improvement: {result['improvement_percent']:.2f}%")
```

### Command-Line Tool

The `run_calibration_with_duckdb.py` script provides a command-line interface for using the calibration system with DuckDB:

```bash
# Run basic calibration with DuckDB integration
python test/run_calibration_with_duckdb.py \
    --sim-file simulation_results.json \
    --hw-file hardware_results.json \
    --param-file initial_parameters.json \
    --db-path calibration.duckdb \
    --hardware-id cuda \
    calibrate

# Run cross-validation with DuckDB integration
python test/run_calibration_with_duckdb.py \
    --sim-file simulation_results.json \
    --hw-file hardware_results.json \
    --param-file initial_parameters.json \
    --db-path calibration.duckdb \
    cross-validate --n-splits 5

# Generate sample data for testing
python test/run_calibration_with_duckdb.py \
    --db-path calibration.duckdb \
    generate-sample --num-calibrations 10
```

## Database Schema

The DuckDB database contains the following tables:

1. **calibration_parameters**: Stores individual parameter values
2. **calibration_results**: Stores results of calibration runs
3. **cross_validation_results**: Stores cross-validation results
4. **parameter_sensitivity**: Stores sensitivity analysis results
5. **uncertainty_quantification**: Stores uncertainty metrics
6. **calibration_history**: Records the history of calibration runs
7. **calibration_drift**: Tracks parameter drift over time

## Advanced Usage

### Cross-Validation with DuckDB

```python
from duckdb_api.simulation_validation.calibration import (
    CalibrationCrossValidator,
    DuckDBCalibrationRepository,
    CrossValidatorDuckDBAdapter
)

# Create repository
repository = DuckDBCalibrationRepository(db_path="calibration.duckdb")

# Create cross-validator with repository integration
cross_validator = CalibrationCrossValidator(n_splits=5, calibrator_type="basic")
adapter = CrossValidatorDuckDBAdapter(
    cross_validator=cross_validator,
    repository=repository,
    validation_id="example-validation"
)

# Run cross-validation
result = adapter.cross_validate(
    simulation_results=simulation_data,
    hardware_results=hardware_data,
    initial_parameters=initial_params,
    calibration_id="example-calibration",
    dataset_id="benchmark-dataset"
)

print(f"Cross-validation generalization gap: {result['generalization_gap_percentage']:.2f}%")
```

### Parameter Discovery with DuckDB

```python
from duckdb_api.simulation_validation.calibration import (
    ParameterDiscovery,
    DuckDBCalibrationRepository,
    ParameterDiscoveryDuckDBAdapter
)

# Create repository
repository = DuckDBCalibrationRepository(db_path="calibration.duckdb")

# Create parameter discovery with repository integration
discovery = ParameterDiscovery(sensitivity_threshold=0.01)
adapter = ParameterDiscoveryDuckDBAdapter(
    parameter_discovery=discovery,
    repository=repository,
    analysis_id="example-analysis"
)

# Define error function for discovery
def error_function(params):
    # Calculate error between simulation and hardware results
    # ...
    return error_value

# Run parameter discovery
result = adapter.discover_parameters(
    error_function=error_function,
    initial_parameters=initial_params,
    calibration_id="example-calibration"
)

print(f"Discovered {len(result['sensitive_parameters'])} sensitive parameters")
```

### Uncertainty Quantification with DuckDB

```python
from duckdb_api.simulation_validation.calibration import (
    UncertaintyQuantifier,
    DuckDBCalibrationRepository,
    UncertaintyQuantifierDuckDBAdapter
)

# Create repository
repository = DuckDBCalibrationRepository(db_path="calibration.duckdb")

# Create uncertainty quantifier with repository integration
quantifier = UncertaintyQuantifier(confidence_level=0.95)
adapter = UncertaintyQuantifierDuckDBAdapter(
    uncertainty_quantifier=quantifier,
    repository=repository,
    analysis_id="example-uncertainty"
)

# Run uncertainty quantification
result = adapter.quantify_parameter_uncertainty(
    parameter_sets=parameter_sets,
    calibration_id="example-calibration"
)

print("Parameter uncertainty analysis completed")
```

## Querying Calibration Data

You can query the calibration data directly using DuckDB SQL:

```python
import duckdb

# Connect to the database
conn = duckdb.connect("calibration.duckdb")

# Query calibration results
results = conn.execute("""
    SELECT calibration_id, error_before, error_after, error_reduction_percent
    FROM calibration_results
    ORDER BY timestamp DESC
    LIMIT 10
""").fetchall()

for result in results:
    print(f"Calibration {result[0]}: {result[3]:.2f}% improvement")

# Query parameter sensitivities
sensitivities = conn.execute("""
    SELECT parameter_name, sensitivity, importance_rank
    FROM parameter_sensitivity
    WHERE analysis_id = 'example-analysis'
    ORDER BY importance_rank
""").fetchall()

for sensitivity in sensitivities:
    print(f"Parameter {sensitivity[0]}: Sensitivity {sensitivity[1]:.4f}")

# Close the connection
conn.close()
```

Or you can use the repository methods to retrieve data:

```python
# Get calibration history
repository = DuckDBCalibrationRepository(db_path="calibration.duckdb")
history = repository.get_calibration_history(limit=10)

for entry in history:
    print(f"Calibration {entry['calibration_id']}: {entry['improvement_percent']:.2f}% improvement")

# Get uncertainty quantifications
uncertainties = repository.get_uncertainty_quantifications(analysis_id="example-uncertainty")

for uncertainty in uncertainties:
    print(f"Parameter {uncertainty['parameter_name']}: "
          f"μ={uncertainty['mean_value']:.4f}, "
          f"σ={uncertainty['std_value']:.4f}, "
          f"Level: {uncertainty['uncertainty_level']}")
```

## Best Practices

1. **Use Consistent IDs**: Always provide calibration_id, validation_id, and analysis_id to link related data.
2. **Include Metadata**: Add descriptive metadata to help with analysis and reporting.
3. **Store Hardware IDs**: Include hardware platform identifiers to compare calibration across hardware.
4. **Tag Calibrations**: Use tags to categorize and filter calibrations.
5. **Track Drift**: Monitor parameter drift over time to detect when recalibration is needed.

## Conclusion

The DuckDB integration provides a robust framework for storing, analyzing, and tracking simulation calibration data. By using this system, you can improve simulation accuracy, understand parameter sensitivity, and make data-driven decisions about calibration approaches.