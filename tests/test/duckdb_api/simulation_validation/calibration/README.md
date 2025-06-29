# Simulation Calibration Components

This directory contains the calibration components for the Simulation Accuracy and Validation Framework. These components are responsible for improving the accuracy of the simulation by adjusting simulation parameters based on real hardware results.

## Key Components

- `basic_calibrator.py`: Simple calibration techniques including linear scaling and basic regression
- `advanced_calibrator.py`: Advanced calibration techniques including Bayesian optimization, neural networks, and ensemble methods

## Calibration Methods

The framework supports the following calibration methods:

### Basic Methods

1. **Linear Scaling**: Simple correction factors for different metrics (throughput, latency, memory usage)
2. **Error-Based Calibration**: Adjusts parameters based on the magnitude and direction of error
3. **Basic Regression**: Simple regression models to predict correction factors

### Advanced Methods

1. **Bayesian Optimization**: Uses Bayesian optimization to find optimal calibration parameters
2. **Neural Network Calibration**: Uses neural networks to learn complex relationships between hardware, models, and correction factors
3. **Multi-Objective Optimization**: Optimizes multiple metrics simultaneously
4. **Ensemble Methods**: Combines multiple calibration methods for better results
5. **Hardware-Specific Profiles**: Specialized calibration for different hardware types
6. **Auto Parameter Discovery**: Automatically discovers which parameters need calibration

## Using the Calibration Components

### Basic Usage

```python
from duckdb_api.simulation_validation.calibration.basic_calibrator import BasicCalibrator
from duckdb_api.simulation_validation.core.base import SimulationResult, HardwareResult, ValidationResult

# Create the calibrator
calibrator = BasicCalibrator()

# Gather validation results
# (example code to create SimulationResult, HardwareResult, and ValidationResult objects)

# Calibrate simulation parameters
calibration_record = calibrator.calibrate(
    validation_results=validation_results,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased"
)

# Apply calibration to future simulations
calibrated_simulation = calibrator.apply_calibration(
    simulation_result=new_simulation,
    calibration_record=calibration_record
)
```

### Advanced Usage

```python
from duckdb_api.simulation_validation.calibration.advanced_calibrator import AdvancedCalibrator

# Create the advanced calibrator with specific methods
calibrator = AdvancedCalibrator(
    method="bayesian_optimization",
    optimization_iterations=50,
    learning_rate=0.01
)

# Calibrate with advanced method
calibration_record = calibrator.calibrate(
    validation_results=validation_results,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    calibration_targets=["throughput_items_per_second", "average_latency_ms"]
)

# Apply calibration with confidence estimates
calibrated_simulation, confidence = calibrator.apply_calibration_with_confidence(
    simulation_result=new_simulation,
    calibration_record=calibration_record
)
```

### Database Integration

```python
from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration
from duckdb_api.simulation_validation.calibration.advanced_calibrator import AdvancedCalibrator

# Create database integration
db_integration = SimulationValidationDBIntegration(db_path="./simulation_db.duckdb")

# Get validation results from database
validation_results = db_integration.get_validation_results(
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased"
)

# Create the calibrator
calibrator = AdvancedCalibrator()

# Calibrate using database results
calibration_record = calibrator.calibrate_from_db(
    db_integration=db_integration,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased"
)

# Store calibration record in database
db_integration.store_calibration_record(calibration_record)
```

## Calibration Workflow

The typical calibration workflow includes:

1. **Data Collection**: Gather simulation and hardware results for the same configurations
2. **Validation**: Calculate validation metrics (MAPE, absolute errors, etc.)
3. **Parameter Identification**: Identify which simulation parameters need adjustment
4. **Calibration**: Apply calibration algorithm to find optimal parameter values
5. **Validation**: Verify that calibrated parameters improve simulation accuracy
6. **Storage**: Store calibration records for future use
7. **Application**: Apply calibration to future simulations
8. **Monitoring**: Continuously monitor calibration effectiveness and recalibrate as needed

## Testing Calibration

The calibration components are tested through both unit tests and the end-to-end testing framework:

```bash
# Run calibration unit tests
python -m unittest duckdb_api.simulation_validation.calibration.test_basic_calibrator
python -m unittest duckdb_api.simulation_validation.calibration.test_advanced_calibrator

# Run end-to-end tests involving calibration
./run_visualization_tests.sh --test-type calibration
```

## Dependencies

The calibration components require the following dependencies:
- numpy: Numerical operations
- scipy: Scientific computing
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning algorithms
- bayesian-optimization: For Bayesian optimization methods
- PyTorch or TensorFlow (optional): For neural network-based calibration

## Integration with Visualization

The calibration results can be visualized using the visualization components:

```python
from duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector

# Create the connector
connector = ValidationVisualizerDBConnector()

# Create a calibration improvement chart
connector.create_calibration_improvement_chart_from_db(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    output_path="./calibration_improvement.html"
)

# Visualize calibration effectiveness
connector.visualize_calibration_effectiveness_from_db(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    interactive=True,
    output_path="./calibration_effectiveness.html"
)
```