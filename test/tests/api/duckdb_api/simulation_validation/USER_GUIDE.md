# Simulation Accuracy and Validation Framework User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Key Concepts](#key-concepts)
4. [Core Components](#core-components)
5. [Quickstart Examples](#quickstart-examples)
6. [Common Workflows](#common-workflows)
   - [Validation Workflow](#validation-workflow)
   - [Calibration Workflow](#calibration-workflow)
   - [Drift Detection Workflow](#drift-detection-workflow)
7. [Visualization Guide](#visualization-guide)
8. [Database Integration](#database-integration)
9. [Advanced Usage](#advanced-usage)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

## Introduction

The Simulation Accuracy and Validation Framework is a comprehensive tool for validating, calibrating, and monitoring simulation accuracy across different hardware configurations. The framework provides tools for comparing simulation predictions with real hardware measurements, calibrating simulations for better accuracy, and detecting when simulation accuracy drifts over time.

### Key Features

- **Validation**: Compare simulation predictions with real hardware measurements using various metrics
- **Calibration**: Improve simulation accuracy by calibrating simulation parameters based on validation results
- **Drift Detection**: Detect when simulation accuracy changes over time, indicating potential issues
- **Visualization**: Generate interactive visualizations to analyze validation results, calibration improvements, and drift detection
- **Database Integration**: Store and query validation results, calibration records, and drift detection results
- **Statistical Analysis**: Perform statistical analysis on validation results to understand simulation accuracy

### Use Cases

- **Simulation Development**: Validate and improve simulation accuracy during development
- **Regression Testing**: Detect regressions in simulation accuracy over time
- **Model Comparison**: Compare accuracy of different simulation models across hardware types
- **Hardware Evaluation**: Evaluate how well simulations perform across different hardware configurations
- **Performance Analysis**: Analyze performance metrics (throughput, latency, memory usage) across simulations and real hardware

## Installation

### Prerequisites

- Python 3.8 or higher
- DuckDB 0.7.0 or higher
- Pandas 1.5.0 or higher
- NumPy 1.22.0 or higher
- (Optional) Plotly 5.10.0 or higher (for interactive visualizations)
- (Optional) Matplotlib 3.5.0 or higher (fallback for static visualizations)
- (Optional) SciPy 1.8.0 or higher (for statistical analysis)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/your-organization/ipfs_accelerate_py.git
cd ipfs_accelerate_py
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install optional visualization dependencies:
```bash
pip install plotly matplotlib scipy
```

### Verifying Installation

To verify that the installation was successful, run:

```bash
python -m test.duckdb_api.simulation_validation.run_e2e_tests --run-db --skip-long-tests
```

This will run a subset of the database integration tests to ensure that the core functionality is working properly.

## Key Concepts

Before diving into the framework, it's important to understand the key concepts:

### Simulation Result

A `SimulationResult` represents the output of a simulation run on a particular hardware configuration. It includes:
- Model identifier (e.g., "bert-base-uncased")
- Hardware identifier (e.g., "gpu_rtx3080")
- Metrics (e.g., throughput, latency, memory usage)
- Simulation parameters (e.g., model parameters, hardware parameters)
- Metadata (timestamp, simulation version)

### Hardware Result

A `HardwareResult` represents the measurements from a real hardware run. It includes:
- Model identifier (matching the simulation)
- Hardware identifier (matching the simulation)
- Metrics (matching the simulation metrics)
- Hardware details (e.g., GPU type, CPU cores)
- Test environment (e.g., OS, driver versions)
- Metadata (timestamp)

### Validation Result

A `ValidationResult` compares a simulation result with a corresponding hardware result. It includes:
- Simulation result
- Hardware result
- Metrics comparison (absolute error, relative error, MAPE)
- Overall accuracy score
- Metadata (validation timestamp, version)

### Calibration Record

A `CalibrationRecord` represents the process of calibrating simulation parameters to improve accuracy. It includes:
- Previous parameters (before calibration)
- Updated parameters (after calibration)
- Validation results before calibration
- Validation results after calibration
- Improvement metrics (MAPE before/after, absolute/relative improvement)
- Metadata (calibration timestamp, version)

### Drift Detection Result

A `DriftDetectionResult` represents the detection of changes in simulation accuracy over time. It includes:
- Drift metrics (p-value, detected drift, mean change percentage)
- Significance assessment
- Time windows (historical and new)
- Thresholds used for detection
- Metadata (detection timestamp)

### MAPE (Mean Absolute Percentage Error)

MAPE is the primary metric used for measuring simulation accuracy. It's calculated as:

```
MAPE = (|simulation_value - hardware_value| / hardware_value) * 100
```

Lower MAPE values indicate better simulation accuracy:
- Excellent: MAPE < 5%
- Good: 5% ≤ MAPE < 10%
- Fair: 10% ≤ MAPE < 20%
- Poor: MAPE ≥ 20%

## Core Components

The framework consists of several core components:

### Core Data Structures

Located in `core/base.py`, these include:
- `SimulationResult`
- `HardwareResult`
- `ValidationResult`
- `CalibrationRecord`
- `DriftDetectionResult`

### Validators

- `BasicValidator`: Performs simple validation comparing simulation and hardware results
- `StatisticalValidator`: Performs statistical validation with additional metrics and analysis

### Calibrators

- `BasicCalibrator`: Performs simple calibration using correction factors
- `AdvancedCalibrator`: Performs advanced calibration with batch size and precision-specific corrections

### Drift Detectors

- `BasicDriftDetector`: Detects drift using simple statistical tests
- `AdvancedDriftDetector`: Detects drift using advanced statistical methods with more detailed metrics

### Database Integration

The `SimulationValidationDBIntegration` class provides database operations for:
- Storing and retrieving all result types
- Querying results by various criteria
- Managing the database schema

### Visualization Components

The `ValidationVisualizer` class provides various visualization tools:
- MAPE comparison charts
- Hardware comparison heatmaps
- Error distribution histograms
- Time series charts
- Comprehensive dashboards

### Framework Integration

The `SimulationValidationFramework` class integrates all components into a cohesive system, providing:
- High-level validation workflows
- Calibration workflows
- Drift detection workflows
- Reporting and visualization

## Quickstart Examples

Here are some quickstart examples to get you up and running quickly:

### Basic Validation

```python
from duckdb_api.simulation_validation.core.base import SimulationResult, HardwareResult
from duckdb_api.simulation_validation.statistical.basic_validator import BasicValidator

# Create simulation result
sim_result = SimulationResult(
    model_id="bert-base-uncased",
    hardware_id="gpu_rtx3080",
    batch_size=32,
    precision="fp16",
    timestamp="2025-03-15T12:00:00",
    simulation_version="sim_v1.0",
    metrics={
        "throughput_items_per_second": 95.0,
        "average_latency_ms": 16.0,
        "peak_memory_mb": 2000
    },
    simulation_params={
        "model_params": {"hidden_size": 768, "num_layers": 12},
        "hardware_params": {"gpu_compute_capability": "8.6", "gpu_memory": 10240}
    }
)

# Create hardware result
hw_result = HardwareResult(
    model_id="bert-base-uncased",
    hardware_id="gpu_rtx3080",
    batch_size=32,
    precision="fp16",
    timestamp="2025-03-15T12:00:00",
    metrics={
        "throughput_items_per_second": 90.0,
        "average_latency_ms": 17.0,
        "peak_memory_mb": 2200
    },
    hardware_details={
        "name": "NVIDIA RTX 3080",
        "compute_capability": "8.6",
        "vram_gb": 10
    },
    test_environment={
        "os": "Linux",
        "cuda_version": "11.4",
        "driver_version": "470.82.01"
    }
)

# Perform validation
validator = BasicValidator()
validation_result = validator.validate(sim_result, hw_result)

# Print validation results
print(f"Overall accuracy score (MAPE): {validation_result.overall_accuracy_score:.2f}%")
for metric, comparison in validation_result.metrics_comparison.items():
    print(f"{metric}:")
    print(f"  Simulation: {comparison['simulation_value']}")
    print(f"  Hardware: {comparison['hardware_value']}")
    print(f"  MAPE: {comparison['mape']:.2f}%")
```

### Basic Visualization

```python
from duckdb_api.simulation_validation.visualization.validation_visualizer import ValidationVisualizer

# Create visualizer
visualizer = ValidationVisualizer()

# Create metric comparison chart
visualizer.create_metric_comparison(
    simulation_values=[95.0, 92.0, 88.0, 90.0],
    hardware_values=[90.0, 89.0, 85.0, 88.0],
    metric_name="throughput_items_per_second",
    output_path="metric_comparison.html"
)

# Create hardware comparison heatmap
visualizer.create_hardware_comparison_heatmap(
    hardware_models=["gpu_rtx3080", "cpu_intel_xeon", "webgpu_chrome"],
    model_types=["bert-base-uncased", "vit-base-patch16-224"],
    mape_values=[[5.2, 7.8], [10.5, 8.3], [15.2, 12.1]],
    metric_name="throughput_items_per_second",
    output_path="hardware_heatmap.html"
)
```

### Using the Framework

```python
from duckdb_api.simulation_validation.simulation_validation_framework import SimulationValidationFramework
from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration

# Create database integration
db_integration = SimulationValidationDBIntegration(db_path="validation_db.duckdb")
db_integration.initialize_database()

# Create framework
framework = SimulationValidationFramework()

# Store simulation and hardware results
sim_id = db_integration.store_simulation_result(sim_result)
hw_id = db_integration.store_hardware_result(hw_result)

# Validate simulation result
validation_result = framework.validate_simulation_result(
    simulation_result_id=sim_id,
    hardware_result_id=hw_id,
    validator_type="basic"
)

# Generate validation report
framework.generate_validation_report(
    validation_result_id=validation_result.id,
    output_format="markdown",
    output_path="validation_report.md"
)

# Create visualization
framework.visualize_validation_result(
    validation_result_id=validation_result.id,
    output_path="validation_visualization.html"
)
```

## Common Workflows

### Validation Workflow

The validation workflow compares simulation predictions with real hardware measurements:

1. **Collect Data**:
   - Run simulations and collect `SimulationResult` objects
   - Run real hardware tests and collect `HardwareResult` objects

2. **Perform Validation**:
   - Use the `BasicValidator` or `StatisticalValidator` to compare results
   - Store validation results in the database

3. **Analyze Results**:
   - Generate validation reports
   - Create visualizations to analyze accuracy
   - Identify areas for improvement

#### Example:

```python
# Collect multiple validation results
validation_results = []
for sim_result, hw_result in zip(simulation_results, hardware_results):
    val_result = validator.validate(sim_result, hw_result)
    validation_results.append(val_result)
    db_integration.store_validation_result(val_result)

# Perform statistical analysis
statistical_validator = StatisticalValidator()
summary = statistical_validator.create_statistical_summary(validation_results)

# Generate report
report_path = "validation_summary_report.md"
framework.generate_validation_summary_report(
    validation_results=validation_results,
    statistical_summary=summary,
    output_format="markdown",
    output_path=report_path
)
```

### Calibration Workflow

The calibration workflow improves simulation accuracy by adjusting simulation parameters:

1. **Collect Validation Data**:
   - Generate validation results as in the validation workflow

2. **Perform Calibration**:
   - Use the `BasicCalibrator` or `AdvancedCalibrator` to calibrate simulation parameters
   - Store calibration record in the database

3. **Validate Calibration**:
   - Run simulations with the calibrated parameters
   - Compare with hardware results to verify improvement
   - Generate calibration reports and visualizations

#### Example:

```python
# Collect validation results for a specific hardware and model
validation_results = db_integration.get_validation_results_with_filters(
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased",
    limit=10
)

# Perform calibration
calibrator = BasicCalibrator()
calibration_record = calibrator.calibrate(
    validation_results=validation_results,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased"
)

# Store calibration record
cal_id = db_integration.store_calibration_record(calibration_record)

# Generate calibration report
framework.generate_calibration_report(
    calibration_record_id=cal_id,
    output_format="markdown",
    output_path="calibration_report.md"
)

# Visualize calibration improvement
framework.visualize_calibration_improvement(
    calibration_record_id=cal_id,
    output_path="calibration_improvement.html"
)
```

### Drift Detection Workflow

The drift detection workflow identifies changes in simulation accuracy over time:

1. **Collect Historical Data**:
   - Retrieve historical validation results from the database

2. **Collect Current Data**:
   - Generate new validation results for the current period

3. **Perform Drift Detection**:
   - Use the `BasicDriftDetector` or `AdvancedDriftDetector` to detect drift
   - Store drift detection result in the database

4. **Analyze Drift**:
   - Generate drift detection reports
   - Create visualizations to analyze drift
   - Take corrective action if significant drift is detected

#### Example:

```python
# Get historical validation results
historical_results = db_integration.get_validation_results_by_time_range(
    start_time="2025-01-01T00:00:00",
    end_time="2025-02-01T00:00:00",
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased"
)

# Get current validation results
current_results = db_integration.get_validation_results_by_time_range(
    start_time="2025-02-01T00:00:00",
    end_time="2025-03-01T00:00:00",
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased"
)

# Perform drift detection
detector = AdvancedDriftDetector()
drift_result = detector.detect_drift(
    historical_validation_results=historical_results,
    current_validation_results=current_results,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased"
)

# Store drift detection result
drift_id = db_integration.store_drift_detection_result(drift_result)

# Generate drift detection report
framework.generate_drift_detection_report(
    drift_detection_result_id=drift_id,
    output_format="markdown",
    output_path="drift_report.md"
)

# Visualize drift detection
framework.visualize_drift_detection(
    drift_detection_result_id=drift_id,
    output_path="drift_visualization.html"
)
```

## Visualization Guide

The framework provides a rich set of visualization tools for analyzing validation results, calibration improvements, and drift detection:

### MAPE Comparison Charts

MAPE comparison charts show the Mean Absolute Percentage Error for different hardware types and models:

```python
# Using the visualizer directly
visualizer.create_mape_comparison(
    hardware_types=["gpu_rtx3080", "cpu_intel_xeon", "webgpu_chrome"],
    model_types=["bert-base-uncased", "vit-base-patch16-224"],
    mape_values=[[5.2, 7.8], [10.5, 8.3], [15.2, 12.1]],
    metric_name="throughput_items_per_second",
    output_path="mape_comparison.html"
)

# Using the DB connector
visualization_db_connector.create_mape_comparison_chart_from_db(
    hardware_ids=["gpu_rtx3080", "cpu_intel_xeon", "webgpu_chrome"],
    model_ids=["bert-base-uncased", "vit-base-patch16-224"],
    metric_name="throughput_items_per_second",
    output_path="mape_comparison_db.html"
)
```

### Hardware Comparison Heatmaps

Hardware comparison heatmaps show MAPE values across different hardware types and models in a heatmap format:

```python
# Using the visualizer directly
visualizer.create_hardware_comparison_heatmap(
    hardware_models=["gpu_rtx3080", "cpu_intel_xeon", "webgpu_chrome"],
    model_types=["bert-base-uncased", "vit-base-patch16-224"],
    mape_values=[[5.2, 7.8], [10.5, 8.3], [15.2, 12.1]],
    metric_name="throughput_items_per_second",
    output_path="hardware_heatmap.html"
)

# Using the DB connector
visualization_db_connector.create_hardware_comparison_heatmap_from_db(
    metric_name="throughput_items_per_second",
    model_ids=["bert-base-uncased", "vit-base-patch16-224"],
    output_path="hardware_heatmap_db.html"
)
```

### Error Distribution Histograms

Error distribution histograms show the distribution of errors for a specific hardware type and model:

```python
# Using the visualizer directly
visualizer.create_error_distribution(
    errors=[2.1, 3.5, 1.8, 4.2, 2.9, 3.7, 2.5, 3.1, 4.0, 2.3],
    metric_name="throughput_items_per_second",
    output_path="error_distribution.html"
)

# Using the DB connector
visualization_db_connector.create_error_distribution_from_db(
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased",
    metric_name="throughput_items_per_second",
    output_path="error_distribution_db.html"
)
```

### Time Series Charts

Time series charts show how metrics change over time:

```python
# Using the visualizer directly
visualizer.create_time_series(
    timestamps=["2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05"],
    simulation_values=[90, 92, 88, 95, 91],
    hardware_values=[85, 90, 86, 92, 88],
    metric_name="throughput_items_per_second",
    output_path="time_series.html"
)

# Using the DB connector
visualization_db_connector.create_time_series_chart_from_db(
    metric_name="throughput_items_per_second",
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased",
    output_path="time_series_db.html"
)
```

### Drift Visualizations

Drift visualizations show changes in simulation accuracy over time:

```python
# Using the DB connector
visualization_db_connector.create_drift_visualization_from_db(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    output_path="drift_visualization.html"
)
```

### Calibration Improvement Charts

Calibration improvement charts show the improvement in simulation accuracy after calibration:

```python
# Using the DB connector
visualization_db_connector.create_calibration_improvement_chart_from_db(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    output_path="calibration_improvement.html"
)
```

### Comprehensive Dashboards

Comprehensive dashboards combine multiple visualization types into a single dashboard:

```python
# Using the DB connector
visualization_db_connector.create_comprehensive_dashboard_from_db(
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased",
    output_path="comprehensive_dashboard.html"
)
```

## Database Integration

The database integration component provides storage and querying capabilities for all data types:

### Initialization

```python
from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration

# Initialize database
db_integration = SimulationValidationDBIntegration(db_path="validation_db.duckdb")
db_integration.initialize_database()
```

### Storing Results

```python
# Store simulation result
sim_id = db_integration.store_simulation_result(sim_result)

# Store hardware result
hw_id = db_integration.store_hardware_result(hw_result)

# Store validation result
val_id = db_integration.store_validation_result(validation_result)

# Store calibration record
cal_id = db_integration.store_calibration_record(calibration_record)

# Store drift detection result
drift_id = db_integration.store_drift_detection_result(drift_result)
```

### Retrieving Results

```python
# Retrieve by ID
sim_result = db_integration.get_simulation_result_by_id(sim_id)
hw_result = db_integration.get_hardware_result_by_id(hw_id)
val_result = db_integration.get_validation_result_by_id(val_id)
cal_record = db_integration.get_calibration_record_by_id(cal_id)
drift_result = db_integration.get_drift_detection_result_by_id(drift_id)
```

### Querying Results

```python
# Query by model and hardware
hw_results = db_integration.get_hardware_results_by_model_and_hardware(
    model_id="bert-base-uncased",
    hardware_id="gpu_rtx3080"
)

# Query by time range
sim_results = db_integration.get_simulation_results_by_time_range(
    start_time="2025-03-01T00:00:00",
    end_time="2025-03-15T00:00:00"
)

# Query with filters
val_results = db_integration.get_validation_results_with_filters(
    model_id="bert-base-uncased",
    hardware_id="gpu_rtx3080",
    limit=10
)
```

## Advanced Usage

### Custom Validators

You can create custom validators by extending the `BasicValidator` class:

```python
from duckdb_api.simulation_validation.statistical.basic_validator import BasicValidator

class CustomValidator(BasicValidator):
    def __init__(self, custom_threshold=15.0):
        super().__init__()
        self.custom_threshold = custom_threshold
    
    def validate(self, simulation_result, hardware_result):
        # Call the parent's validate method to get the basic validation result
        validation_result = super().validate(simulation_result, hardware_result)
        
        # Add custom validation logic
        for metric, comparison in validation_result.metrics_comparison.items():
            # Add custom flags based on thresholds
            comparison["exceeds_threshold"] = comparison["mape"] > self.custom_threshold
        
        return validation_result
```

### Custom Calibrators

You can create custom calibrators by extending the `BasicCalibrator` class:

```python
from duckdb_api.simulation_validation.calibration.basic_calibrator import BasicCalibrator

class CustomCalibrator(BasicCalibrator):
    def __init__(self, weight_recent=0.7):
        super().__init__()
        self.weight_recent = weight_recent
    
    def calibrate(self, validation_results, hardware_type, model_type):
        # Sort validation results by timestamp
        sorted_results = sorted(validation_results, 
                               key=lambda x: x.validation_timestamp)
        
        # Split into older and more recent results
        split_idx = len(sorted_results) // 2
        older_results = sorted_results[:split_idx]
        recent_results = sorted_results[split_idx:]
        
        # Calculate correction factors with weighted approach
        correction_factors = {}
        
        for metric in ["throughput_items_per_second", "average_latency_ms", "peak_memory_mb"]:
            # Calculate for older results
            older_ratios = self._calculate_ratios(older_results, metric)
            
            # Calculate for recent results
            recent_ratios = self._calculate_ratios(recent_results, metric)
            
            # Combine with weighting
            if older_ratios and recent_ratios:
                weighted_ratio = ((1 - self.weight_recent) * sum(older_ratios) / len(older_ratios) + 
                                 self.weight_recent * sum(recent_ratios) / len(recent_ratios))
                correction_factors[metric] = round(weighted_ratio, 3)
            elif recent_ratios:
                correction_factors[metric] = round(sum(recent_ratios) / len(recent_ratios), 3)
            elif older_ratios:
                correction_factors[metric] = round(sum(older_ratios) / len(older_ratios), 3)
            else:
                correction_factors[metric] = 1.0
        
        # Create calibration record using parent's method
        return self._create_calibration_record(
            validation_results=validation_results,
            hardware_type=hardware_type,
            model_type=model_type,
            correction_factors=correction_factors
        )
    
    def _calculate_ratios(self, validation_results, metric):
        """Calculate hardware/simulation ratios for a metric."""
        ratios = []
        for val_result in validation_results:
            comparison = val_result.metrics_comparison.get(metric, {})
            if "hardware_value" in comparison and "simulation_value" in comparison:
                hw_val = comparison["hardware_value"]
                sim_val = comparison["simulation_value"]
                if sim_val != 0:
                    ratios.append(hw_val / sim_val)
        return ratios
```

### Custom Drift Detectors

You can create custom drift detectors by extending the `BasicDriftDetector` class:

```python
from duckdb_api.simulation_validation.drift_detection.basic_detector import BasicDriftDetector

class CustomDriftDetector(BasicDriftDetector):
    def __init__(self, p_threshold=0.05, change_threshold=10.0, min_sample_size=5):
        super().__init__()
        self.p_threshold = p_threshold
        self.change_threshold = change_threshold
        self.min_sample_size = min_sample_size
    
    def detect_drift(self, historical_validation_results, current_validation_results, 
                    hardware_type, model_type):
        # Check sample sizes
        if (len(historical_validation_results) < self.min_sample_size or 
            len(current_validation_results) < self.min_sample_size):
            # Not enough data for reliable drift detection
            return self._create_drift_detection_result(
                historical_validation_results=historical_validation_results,
                current_validation_results=current_validation_results,
                hardware_type=hardware_type,
                model_type=model_type,
                drift_metrics={},
                is_significant=False
            )
        
        # Use parent's detection method
        return super().detect_drift(
            historical_validation_results=historical_validation_results,
            current_validation_results=current_validation_results,
            hardware_type=hardware_type,
            model_type=model_type
        )
```

## Performance Considerations

### Large Datasets

When working with large datasets, consider the following:

1. **Query Optimization**:
   - Use filters to narrow down query results
   - Use time range filters when possible
   - Use limit parameter to control result size

2. **Batch Operations**:
   - When storing multiple results, use batch operations
   - Use database transactions for bulk operations

3. **Memory Management**:
   - Process large datasets in chunks
   - Use generators for large result sets
   - Release memory explicitly when possible

### Example for Large Datasets

```python
# Process validation results in chunks
chunk_size = 100
total_results = db_integration.count_validation_results_with_filters(
    hardware_id="gpu_rtx3080",
    model_id="bert-base-uncased"
)

# Process in chunks
for offset in range(0, total_results, chunk_size):
    results_chunk = db_integration.get_validation_results_with_filters(
        hardware_id="gpu_rtx3080",
        model_id="bert-base-uncased",
        limit=chunk_size,
        offset=offset
    )
    
    # Process chunk
    process_results_chunk(results_chunk)
    
    # Explicitly release memory
    del results_chunk
```

## Troubleshooting

### Common Issues and Solutions

#### Database Connection Issues

**Problem**: Unable to connect to the database.

**Solution**:
- Check that the database path is correct
- Ensure the directory exists and is writable
- Try creating a new database in a different location

#### Missing Dependencies

**Problem**: Import errors or missing functionality.

**Solution**:
- Check that all required dependencies are installed
- Install optional dependencies for full functionality
- Verify Python version is compatible

#### Visualization Issues

**Problem**: Visualizations are not generating or are incomplete.

**Solution**:
- Check that Plotly is installed for interactive visualizations
- Check that Matplotlib is installed for static visualizations
- Verify that the output directory exists and is writable
- Try a different output format

#### Memory Errors

**Problem**: Out of memory when processing large datasets.

**Solution**:
- Process data in smaller chunks
- Use filters to limit the amount of data processed
- Increase available memory or use a machine with more memory
- Use database queries to aggregate data before processing

### Debugging

For debugging issues, enable verbose logging:

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create framework with logging
framework = SimulationValidationFramework(enable_logging=True)
```

## FAQ

### General Questions

**Q: Can the framework work with any simulation system?**

A: Yes, the framework is designed to be simulation-agnostic. It works with any simulation system that can produce results in the format required by the framework.

**Q: Does the framework support multiple hardware types?**

A: Yes, the framework supports any hardware type. Common hardware types include GPUs, CPUs, and web browsers (WebGPU, WebNN).

**Q: Can I use the framework with my existing database?**

A: The framework currently uses DuckDB as its database backend. While it's possible to adapt it to other databases, this would require customizing the database integration component.

### Technical Questions

**Q: How much data is needed for reliable drift detection?**

A: For reliable drift detection, we recommend at least 10 validation results in each time window (historical and current). More data generally leads to more reliable drift detection.

**Q: How often should I calibrate my simulations?**

A: Calibration frequency depends on your specific needs. Common approaches include:
- Regular calibration (e.g., monthly)
- Event-driven calibration (when drift is detected)
- Version-driven calibration (when simulation code changes)

**Q: Can I use custom metrics?**

A: Yes, the framework supports any numeric metrics. Common metrics include throughput, latency, and memory usage, but you can use any custom metrics relevant to your simulation.

**Q: How can I extend the framework with custom components?**

A: The framework is designed to be extensible. You can create custom validators, calibrators, and drift detectors by extending the base classes. See the Advanced Usage section for examples.

### Troubleshooting Questions

**Q: Why are my visualizations not interactive?**

A: Interactive visualizations require Plotly. Make sure Plotly is installed (`pip install plotly`). If Plotly is not available, the framework will fall back to static visualizations using Matplotlib.

**Q: How can I improve the performance of database operations?**

A: To improve database performance:
- Use appropriate filters when querying
- Use indexes on frequently queried columns
- Use batch operations for bulk inserts
- Optimize queries to fetch only needed data

**Q: What should I do if drift detection fails?**

A: Drift detection might fail if there's not enough data or if the data is too noisy. Try:
- Increasing the sample size
- Adjusting the p-value and change thresholds
- Using the advanced drift detector for more detailed analysis
- Checking the data quality