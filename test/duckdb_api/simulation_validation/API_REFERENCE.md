# Simulation Accuracy and Validation Framework API Reference

## Table of Contents

1. [Core Data Structures](#core-data-structures)
   - [SimulationResult](#simulationresult)
   - [HardwareResult](#hardwareresult)
   - [ValidationResult](#validationresult)
   - [CalibrationRecord](#calibrationrecord)
   - [DriftDetectionResult](#driftdetectionresult)
2. [Validators](#validators)
   - [BasicValidator](#basicvalidator)
   - [StatisticalValidator](#statisticalvalidator)
3. [Calibrators](#calibrators)
   - [BasicCalibrator](#basiccalibrator)
   - [AdvancedCalibrator](#advancedcalibrator)
4. [Drift Detectors](#drift-detectors)
   - [BasicDriftDetector](#basicdriftdetector)
   - [AdvancedDriftDetector](#advanceddriftdetector)
5. [Database Integration](#database-integration)
   - [SimulationValidationDBIntegration](#simulationvalidationdbintegration)
6. [Visualization](#visualization)
   - [ValidationVisualizer](#validationvisualizer)
   - [ValidationVisualizerDBConnector](#validationvisualizerdbconnector)
7. [Framework](#framework)
   - [SimulationValidationFramework](#simulationvalidationframework)
   - [ValidationMethodology](#validationmethodology)

## Core Data Structures

### SimulationResult

A class representing the results of a simulation run on a particular hardware configuration.

#### Constructor

```python
SimulationResult(
    model_id: str,
    hardware_id: str,
    batch_size: int,
    precision: str,
    timestamp: str,
    simulation_version: str,
    metrics: Dict[str, float],
    simulation_params: Dict[str, Any],
    id: Optional[str] = None
)
```

**Parameters:**
- `model_id` (str): Identifier for the model (e.g., "bert-base-uncased")
- `hardware_id` (str): Identifier for the hardware (e.g., "gpu_rtx3080")
- `batch_size` (int): Batch size used for the simulation
- `precision` (str): Precision used for the simulation (e.g., "fp32", "fp16", "int8")
- `timestamp` (str): ISO format timestamp of when the simulation was run
- `simulation_version` (str): Version of the simulation software
- `metrics` (Dict[str, float]): Dictionary of metrics (e.g., {"throughput_items_per_second": 95.0})
- `simulation_params` (Dict[str, Any]): Dictionary of simulation parameters
- `id` (Optional[str]): Unique identifier for the result (auto-generated if not provided)

#### Methods

**to_dict()** -> Dict[str, Any]

Converts the SimulationResult to a dictionary.

**from_dict(data: Dict[str, Any])** -> SimulationResult

Creates a SimulationResult from a dictionary.

### HardwareResult

A class representing the results of running a model on real hardware.

#### Constructor

```python
HardwareResult(
    model_id: str,
    hardware_id: str,
    batch_size: int,
    precision: str,
    timestamp: str,
    metrics: Dict[str, float],
    hardware_details: Dict[str, Any],
    test_environment: Dict[str, Any],
    id: Optional[str] = None
)
```

**Parameters:**
- `model_id` (str): Identifier for the model (e.g., "bert-base-uncased")
- `hardware_id` (str): Identifier for the hardware (e.g., "gpu_rtx3080")
- `batch_size` (int): Batch size used for the hardware run
- `precision` (str): Precision used for the hardware run (e.g., "fp32", "fp16", "int8")
- `timestamp` (str): ISO format timestamp of when the hardware run was performed
- `metrics` (Dict[str, float]): Dictionary of metrics (e.g., {"throughput_items_per_second": 90.0})
- `hardware_details` (Dict[str, Any]): Dictionary of hardware details
- `test_environment` (Dict[str, Any]): Dictionary of test environment details
- `id` (Optional[str]): Unique identifier for the result (auto-generated if not provided)

#### Methods

**to_dict()** -> Dict[str, Any]

Converts the HardwareResult to a dictionary.

**from_dict(data: Dict[str, Any])** -> HardwareResult

Creates a HardwareResult from a dictionary.

### ValidationResult

A class representing the comparison of simulation and hardware results.

#### Constructor

```python
ValidationResult(
    simulation_result: SimulationResult,
    hardware_result: HardwareResult,
    metrics_comparison: Dict[str, Dict[str, float]],
    validation_timestamp: str,
    validation_version: str,
    overall_accuracy_score: float,
    id: Optional[str] = None
)
```

**Parameters:**
- `simulation_result` (SimulationResult): The simulation result being validated
- `hardware_result` (HardwareResult): The hardware result for comparison
- `metrics_comparison` (Dict[str, Dict[str, float]]): Dictionary of metric comparisons
- `validation_timestamp` (str): ISO format timestamp of when the validation was performed
- `validation_version` (str): Version of the validation software
- `overall_accuracy_score` (float): Overall accuracy score (MAPE)
- `id` (Optional[str]): Unique identifier for the result (auto-generated if not provided)

#### Methods

**to_dict()** -> Dict[str, Any]

Converts the ValidationResult to a dictionary.

**from_dict(data: Dict[str, Any], simulation_result: SimulationResult, hardware_result: HardwareResult)** -> ValidationResult

Creates a ValidationResult from a dictionary and the associated SimulationResult and HardwareResult.

### CalibrationRecord

A class representing the process of calibrating simulation parameters to improve accuracy.

#### Constructor

```python
CalibrationRecord(
    id: str,
    timestamp: str,
    hardware_type: str,
    model_type: str,
    previous_parameters: Dict[str, Any],
    updated_parameters: Dict[str, Any],
    validation_results_before: List[ValidationResult],
    validation_results_after: List[ValidationResult],
    improvement_metrics: Dict[str, Dict[str, float]],
    calibration_version: str
)
```

**Parameters:**
- `id` (str): Unique identifier for the calibration record
- `timestamp` (str): ISO format timestamp of when the calibration was performed
- `hardware_type` (str): Type of hardware being calibrated for
- `model_type` (str): Type of model being calibrated for
- `previous_parameters` (Dict[str, Any]): Parameters before calibration
- `updated_parameters` (Dict[str, Any]): Parameters after calibration
- `validation_results_before` (List[ValidationResult]): Validation results before calibration
- `validation_results_after` (List[ValidationResult]): Validation results after calibration
- `improvement_metrics` (Dict[str, Dict[str, float]]): Metrics showing improvement
- `calibration_version` (str): Version of the calibration software

#### Methods

**to_dict()** -> Dict[str, Any]

Converts the CalibrationRecord to a dictionary.

**from_dict(data: Dict[str, Any], validation_results_before: List[ValidationResult], validation_results_after: List[ValidationResult])** -> CalibrationRecord

Creates a CalibrationRecord from a dictionary and the associated validation results.

### DriftDetectionResult

A class representing the detection of changes in simulation accuracy over time.

#### Constructor

```python
DriftDetectionResult(
    id: str,
    timestamp: str,
    hardware_type: str,
    model_type: str,
    drift_metrics: Dict[str, Dict[str, Any]],
    is_significant: bool,
    historical_window_start: str,
    historical_window_end: str,
    new_window_start: str,
    new_window_end: str,
    thresholds_used: Dict[str, float]
)
```

**Parameters:**
- `id` (str): Unique identifier for the drift detection result
- `timestamp` (str): ISO format timestamp of when the drift detection was performed
- `hardware_type` (str): Type of hardware being analyzed
- `model_type` (str): Type of model being analyzed
- `drift_metrics` (Dict[str, Dict[str, Any]]): Metrics showing drift
- `is_significant` (bool): Whether the drift is significant
- `historical_window_start` (str): Start of historical window (ISO format)
- `historical_window_end` (str): End of historical window (ISO format)
- `new_window_start` (str): Start of new window (ISO format)
- `new_window_end` (str): End of new window (ISO format)
- `thresholds_used` (Dict[str, float]): Thresholds used for drift detection

#### Methods

**to_dict()** -> Dict[str, Any]

Converts the DriftDetectionResult to a dictionary.

**from_dict(data: Dict[str, Any])** -> DriftDetectionResult

Creates a DriftDetectionResult from a dictionary.

## Validators

### BasicValidator

A class for validating simulation results against hardware results.

#### Constructor

```python
BasicValidator()
```

No parameters required.

#### Methods

**validate(simulation_result: SimulationResult, hardware_result: HardwareResult)** -> ValidationResult

Validates a simulation result against a hardware result and returns a ValidationResult.

**Parameters:**
- `simulation_result` (SimulationResult): The simulation result to validate
- `hardware_result` (HardwareResult): The hardware result to compare against

**Returns:**
- `ValidationResult`: The validation result

### StatisticalValidator

A class for performing statistical validation of multiple simulation results.

#### Constructor

```python
StatisticalValidator()
```

No parameters required.

#### Methods

**validate(simulation_result: SimulationResult, hardware_result: HardwareResult)** -> ValidationResult

Validates a simulation result against a hardware result and returns a ValidationResult.

**Parameters:**
- `simulation_result` (SimulationResult): The simulation result to validate
- `hardware_result` (HardwareResult): The hardware result to compare against

**Returns:**
- `ValidationResult`: The validation result

**validate_multiple(simulation_results: List[SimulationResult], hardware_results: List[HardwareResult])** -> List[ValidationResult]

Validates multiple simulation results against hardware results.

**Parameters:**
- `simulation_results` (List[SimulationResult]): The simulation results to validate
- `hardware_results` (List[HardwareResult]): The hardware results to compare against

**Returns:**
- `List[ValidationResult]`: The validation results

**create_statistical_summary(validation_results: List[ValidationResult])** -> Dict[str, Any]

Creates a statistical summary of validation results.

**Parameters:**
- `validation_results` (List[ValidationResult]): The validation results to summarize

**Returns:**
- `Dict[str, Any]`: The statistical summary

## Calibrators

### BasicCalibrator

A class for calibrating simulation parameters to improve accuracy.

#### Constructor

```python
BasicCalibrator()
```

No parameters required.

#### Methods

**calibrate(validation_results: List[ValidationResult], hardware_type: str, model_type: str)** -> CalibrationRecord

Calibrates simulation parameters based on validation results.

**Parameters:**
- `validation_results` (List[ValidationResult]): The validation results to use for calibration
- `hardware_type` (str): The hardware type to calibrate for
- `model_type` (str): The model type to calibrate for

**Returns:**
- `CalibrationRecord`: The calibration record

### AdvancedCalibrator

A class for advanced calibration of simulation parameters.

#### Constructor

```python
AdvancedCalibrator()
```

No parameters required.

#### Methods

**calibrate(validation_results: List[ValidationResult], hardware_type: str, model_type: str)** -> CalibrationRecord

Calibrates simulation parameters based on validation results, with advanced techniques.

**Parameters:**
- `validation_results` (List[ValidationResult]): The validation results to use for calibration
- `hardware_type` (str): The hardware type to calibrate for
- `model_type` (str): The model type to calibrate for

**Returns:**
- `CalibrationRecord`: The calibration record

## Drift Detectors

### BasicDriftDetector

A class for detecting drift in simulation accuracy over time.

#### Constructor

```python
BasicDriftDetector(p_threshold: float = 0.05, change_threshold: float = 10.0)
```

**Parameters:**
- `p_threshold` (float): P-value threshold for statistical significance
- `change_threshold` (float): Percentage change threshold for practical significance

#### Methods

**detect_drift(historical_validation_results: List[ValidationResult], current_validation_results: List[ValidationResult], hardware_type: str, model_type: str)** -> DriftDetectionResult

Detects drift in simulation accuracy between historical and current validation results.

**Parameters:**
- `historical_validation_results` (List[ValidationResult]): Historical validation results
- `current_validation_results` (List[ValidationResult]): Current validation results
- `hardware_type` (str): The hardware type to analyze
- `model_type` (str): The model type to analyze

**Returns:**
- `DriftDetectionResult`: The drift detection result

### AdvancedDriftDetector

A class for advanced drift detection with more detailed metrics.

#### Constructor

```python
AdvancedDriftDetector(p_threshold: float = 0.05, change_threshold: float = 10.0)
```

**Parameters:**
- `p_threshold` (float): P-value threshold for statistical significance
- `change_threshold` (float): Percentage change threshold for practical significance

#### Methods

**detect_drift(historical_validation_results: List[ValidationResult], current_validation_results: List[ValidationResult], hardware_type: str, model_type: str)** -> DriftDetectionResult

Detects drift in simulation accuracy between historical and current validation results, with advanced metrics.

**Parameters:**
- `historical_validation_results` (List[ValidationResult]): Historical validation results
- `current_validation_results` (List[ValidationResult]): Current validation results
- `hardware_type` (str): The hardware type to analyze
- `model_type` (str): The model type to analyze

**Returns:**
- `DriftDetectionResult`: The drift detection result

## Database Integration

### SimulationValidationDBIntegration

A class for database operations for the simulation validation framework.

#### Constructor

```python
SimulationValidationDBIntegration(db_path: str)
```

**Parameters:**
- `db_path` (str): Path to the database file

#### Methods

**initialize_database()** -> None

Initializes the database schema.

**close()** -> None

Closes the database connection.

**store_simulation_result(simulation_result: SimulationResult)** -> str

Stores a simulation result in the database.

**Parameters:**
- `simulation_result` (SimulationResult): The simulation result to store

**Returns:**
- `str`: The ID of the stored simulation result

**store_hardware_result(hardware_result: HardwareResult)** -> str

Stores a hardware result in the database.

**Parameters:**
- `hardware_result` (HardwareResult): The hardware result to store

**Returns:**
- `str`: The ID of the stored hardware result

**store_validation_result(validation_result: ValidationResult)** -> str

Stores a validation result in the database.

**Parameters:**
- `validation_result` (ValidationResult): The validation result to store

**Returns:**
- `str`: The ID of the stored validation result

**store_calibration_record(calibration_record: CalibrationRecord)** -> str

Stores a calibration record in the database.

**Parameters:**
- `calibration_record` (CalibrationRecord): The calibration record to store

**Returns:**
- `str`: The ID of the stored calibration record

**store_drift_detection_result(drift_detection_result: DriftDetectionResult)** -> str

Stores a drift detection result in the database.

**Parameters:**
- `drift_detection_result` (DriftDetectionResult): The drift detection result to store

**Returns:**
- `str`: The ID of the stored drift detection result

**get_simulation_result_by_id(id: str)** -> SimulationResult

Retrieves a simulation result by ID.

**Parameters:**
- `id` (str): The ID of the simulation result

**Returns:**
- `SimulationResult`: The simulation result

**get_hardware_result_by_id(id: str)** -> HardwareResult

Retrieves a hardware result by ID.

**Parameters:**
- `id` (str): The ID of the hardware result

**Returns:**
- `HardwareResult`: The hardware result

**get_validation_result_by_id(id: str)** -> ValidationResult

Retrieves a validation result by ID.

**Parameters:**
- `id` (str): The ID of the validation result

**Returns:**
- `ValidationResult`: The validation result

**get_calibration_record_by_id(id: str)** -> CalibrationRecord

Retrieves a calibration record by ID.

**Parameters:**
- `id` (str): The ID of the calibration record

**Returns:**
- `CalibrationRecord`: The calibration record

**get_drift_detection_result_by_id(id: str)** -> DriftDetectionResult

Retrieves a drift detection result by ID.

**Parameters:**
- `id` (str): The ID of the drift detection result

**Returns:**
- `DriftDetectionResult`: The drift detection result

**get_hardware_results_by_model_and_hardware(model_id: str, hardware_id: str)** -> List[HardwareResult]

Retrieves hardware results by model ID and hardware ID.

**Parameters:**
- `model_id` (str): The model ID
- `hardware_id` (str): The hardware ID

**Returns:**
- `List[HardwareResult]`: The hardware results

**get_simulation_results_by_model_and_hardware(model_id: str, hardware_id: str)** -> List[SimulationResult]

Retrieves simulation results by model ID and hardware ID.

**Parameters:**
- `model_id` (str): The model ID
- `hardware_id` (str): The hardware ID

**Returns:**
- `List[SimulationResult]`: The simulation results

**get_validation_results_by_model_and_hardware(model_id: str, hardware_id: str)** -> List[ValidationResult]

Retrieves validation results by model ID and hardware ID.

**Parameters:**
- `model_id` (str): The model ID
- `hardware_id` (str): The hardware ID

**Returns:**
- `List[ValidationResult]`: The validation results

**get_hardware_results_by_time_range(start_time: str, end_time: str, model_id: Optional[str] = None, hardware_id: Optional[str] = None)** -> List[HardwareResult]

Retrieves hardware results within a time range.

**Parameters:**
- `start_time` (str): Start time in ISO format
- `end_time` (str): End time in ISO format
- `model_id` (Optional[str]): Model ID filter
- `hardware_id` (Optional[str]): Hardware ID filter

**Returns:**
- `List[HardwareResult]`: The hardware results

**get_simulation_results_by_time_range(start_time: str, end_time: str, model_id: Optional[str] = None, hardware_id: Optional[str] = None)** -> List[SimulationResult]

Retrieves simulation results within a time range.

**Parameters:**
- `start_time` (str): Start time in ISO format
- `end_time` (str): End time in ISO format
- `model_id` (Optional[str]): Model ID filter
- `hardware_id` (Optional[str]): Hardware ID filter

**Returns:**
- `List[SimulationResult]`: The simulation results

**get_validation_results_by_time_range(start_time: str, end_time: str, model_id: Optional[str] = None, hardware_id: Optional[str] = None)** -> List[ValidationResult]

Retrieves validation results within a time range.

**Parameters:**
- `start_time` (str): Start time in ISO format
- `end_time` (str): End time in ISO format
- `model_id` (Optional[str]): Model ID filter
- `hardware_id` (Optional[str]): Hardware ID filter

**Returns:**
- `List[ValidationResult]`: The validation results

**get_hardware_results_with_filters(model_id: Optional[str] = None, hardware_id: Optional[str] = None, batch_size: Optional[int] = None, precision: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None)** -> List[HardwareResult]

Retrieves hardware results with filters.

**Parameters:**
- `model_id` (Optional[str]): Model ID filter
- `hardware_id` (Optional[str]): Hardware ID filter
- `batch_size` (Optional[int]): Batch size filter
- `precision` (Optional[str]): Precision filter
- `limit` (Optional[int]): Maximum number of results
- `offset` (Optional[int]): Offset for pagination

**Returns:**
- `List[HardwareResult]`: The hardware results

**get_simulation_results_with_filters(model_id: Optional[str] = None, hardware_id: Optional[str] = None, batch_size: Optional[int] = None, precision: Optional[str] = None, simulation_version: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None)** -> List[SimulationResult]

Retrieves simulation results with filters.

**Parameters:**
- `model_id` (Optional[str]): Model ID filter
- `hardware_id` (Optional[str]): Hardware ID filter
- `batch_size` (Optional[int]): Batch size filter
- `precision` (Optional[str]): Precision filter
- `simulation_version` (Optional[str]): Simulation version filter
- `limit` (Optional[int]): Maximum number of results
- `offset` (Optional[int]): Offset for pagination

**Returns:**
- `List[SimulationResult]`: The simulation results

**get_validation_results_with_filters(model_id: Optional[str] = None, hardware_id: Optional[str] = None, batch_size: Optional[int] = None, precision: Optional[str] = None, validation_version: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None)** -> List[ValidationResult]

Retrieves validation results with filters.

**Parameters:**
- `model_id` (Optional[str]): Model ID filter
- `hardware_id` (Optional[str]): Hardware ID filter
- `batch_size` (Optional[int]): Batch size filter
- `precision` (Optional[str]): Precision filter
- `validation_version` (Optional[str]): Validation version filter
- `limit` (Optional[int]): Maximum number of results
- `offset` (Optional[int]): Offset for pagination

**Returns:**
- `List[ValidationResult]`: The validation results

**get_calibration_records_with_filters(hardware_type: Optional[str] = None, model_type: Optional[str] = None, calibration_version: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None)** -> List[CalibrationRecord]

Retrieves calibration records with filters.

**Parameters:**
- `hardware_type` (Optional[str]): Hardware type filter
- `model_type` (Optional[str]): Model type filter
- `calibration_version` (Optional[str]): Calibration version filter
- `limit` (Optional[int]): Maximum number of results
- `offset` (Optional[int]): Offset for pagination

**Returns:**
- `List[CalibrationRecord]`: The calibration records

**get_drift_detection_results_with_filters(hardware_type: Optional[str] = None, model_type: Optional[str] = None, is_significant: Optional[bool] = None, limit: Optional[int] = None, offset: Optional[int] = None)** -> List[DriftDetectionResult]

Retrieves drift detection results with filters.

**Parameters:**
- `hardware_type` (Optional[str]): Hardware type filter
- `model_type` (Optional[str]): Model type filter
- `is_significant` (Optional[bool]): Significance filter
- `limit` (Optional[int]): Maximum number of results
- `offset` (Optional[int]): Offset for pagination

**Returns:**
- `List[DriftDetectionResult]`: The drift detection results

**count_validation_results_with_filters(model_id: Optional[str] = None, hardware_id: Optional[str] = None)** -> int

Counts validation results with filters.

**Parameters:**
- `model_id` (Optional[str]): Model ID filter
- `hardware_id` (Optional[str]): Hardware ID filter

**Returns:**
- `int`: The count of validation results

**get_table_list()** -> List[str]

Gets the list of tables in the database.

**Returns:**
- `List[str]`: The list of tables

**get_table_schema(table_name: str)** -> Dict[str, str]

Gets the schema of a table.

**Parameters:**
- `table_name` (str): The name of the table

**Returns:**
- `Dict[str, str]`: The schema of the table

## Visualization

### ValidationVisualizer

A class for creating visualizations of validation results.

#### Constructor

```python
ValidationVisualizer()
```

No parameters required.

#### Methods

**create_metric_comparison(simulation_values: List[float], hardware_values: List[float], metric_name: str, output_path: str, interactive: bool = True)** -> str

Creates a visualization comparing simulation and hardware metrics.

**Parameters:**
- `simulation_values` (List[float]): Simulation values
- `hardware_values` (List[float]): Hardware values
- `metric_name` (str): Name of the metric
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_hardware_comparison_heatmap(hardware_models: List[str], model_types: List[str], mape_values: List[List[float]], metric_name: str, output_path: str, interactive: bool = True)** -> str

Creates a heatmap visualization comparing MAPE values across hardware models and model types.

**Parameters:**
- `hardware_models` (List[str]): List of hardware models
- `model_types` (List[str]): List of model types
- `mape_values` (List[List[float]]): MAPE values
- `metric_name` (str): Name of the metric
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_error_distribution(errors: List[float], metric_name: str, output_path: str, interactive: bool = True)** -> str

Creates a histogram visualization of error distribution.

**Parameters:**
- `errors` (List[float]): List of errors
- `metric_name` (str): Name of the metric
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_time_series(timestamps: List[str], simulation_values: List[float], hardware_values: List[float], metric_name: str, output_path: str, interactive: bool = True)** -> str

Creates a time series visualization of simulation and hardware metrics.

**Parameters:**
- `timestamps` (List[str]): List of timestamps
- `simulation_values` (List[float]): Simulation values
- `hardware_values` (List[float]): Hardware values
- `metric_name` (str): Name of the metric
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_multi_metric_comparison(metrics: List[str], simulation_values: Dict[str, List[float]], hardware_values: Dict[str, List[float]], output_path: str, interactive: bool = True)** -> str

Creates a multi-metric comparison visualization.

**Parameters:**
- `metrics` (List[str]): List of metrics
- `simulation_values` (Dict[str, List[float]]): Simulation values by metric
- `hardware_values` (Dict[str, List[float]]): Hardware values by metric
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_comprehensive_dashboard(visualizations: List[Dict[str, Any]], title: str, output_path: str)** -> str

Creates a comprehensive dashboard combining multiple visualizations.

**Parameters:**
- `visualizations` (List[Dict[str, Any]]): List of visualizations
- `title` (str): Title of the dashboard
- `output_path` (str): Path to save the dashboard

**Returns:**
- `str`: Path to the generated dashboard

### ValidationVisualizerDBConnector

A class for creating visualizations directly from database data.

#### Constructor

```python
ValidationVisualizerDBConnector(db_integration: SimulationValidationDBIntegration, visualizer: Optional[ValidationVisualizer] = None)
```

**Parameters:**
- `db_integration` (SimulationValidationDBIntegration): Database integration instance
- `visualizer` (Optional[ValidationVisualizer]): Visualizer instance (created if not provided)

#### Methods

**create_mape_comparison_chart_from_db(hardware_ids: List[str], model_ids: List[str], metric_name: str, output_path: str, interactive: bool = True)** -> str

Creates a MAPE comparison chart from database data.

**Parameters:**
- `hardware_ids` (List[str]): List of hardware IDs
- `model_ids` (List[str]): List of model IDs
- `metric_name` (str): Name of the metric
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_hardware_comparison_heatmap_from_db(metric_name: str, model_ids: List[str], output_path: str, interactive: bool = True)** -> str

Creates a hardware comparison heatmap from database data.

**Parameters:**
- `metric_name` (str): Name of the metric
- `model_ids` (List[str]): List of model IDs
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_error_distribution_from_db(hardware_id: str, model_id: str, metric_name: str, output_path: str, interactive: bool = True)** -> str

Creates an error distribution histogram from database data.

**Parameters:**
- `hardware_id` (str): Hardware ID
- `model_id` (str): Model ID
- `metric_name` (str): Name of the metric
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_time_series_chart_from_db(metric_name: str, hardware_id: str, model_id: str, output_path: str, interactive: bool = True)** -> str

Creates a time series chart from database data.

**Parameters:**
- `metric_name` (str): Name of the metric
- `hardware_id` (str): Hardware ID
- `model_id` (str): Model ID
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_drift_visualization_from_db(hardware_type: str, model_type: str, output_path: str, interactive: bool = True)** -> str

Creates a drift visualization from database data.

**Parameters:**
- `hardware_type` (str): Hardware type
- `model_type` (str): Model type
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_calibration_improvement_chart_from_db(hardware_type: str, model_type: str, output_path: str, interactive: bool = True)** -> str

Creates a calibration improvement chart from database data.

**Parameters:**
- `hardware_type` (str): Hardware type
- `model_type` (str): Model type
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**create_simulation_vs_hardware_chart_from_db(metric_name: str, hardware_id: str, model_id: str, interactive: bool = True, output_path: str)** -> str

Creates a simulation vs hardware chart from database data.

**Parameters:**
- `metric_name` (str): Name of the metric
- `hardware_id` (str): Hardware ID
- `model_id` (str): Model ID
- `interactive` (bool): Whether to create an interactive visualization
- `output_path` (str): Path to save the visualization

**Returns:**
- `str`: Path to the generated visualization

**create_comprehensive_dashboard_from_db(hardware_id: str, model_id: str, output_path: str)** -> str

Creates a comprehensive dashboard from database data.

**Parameters:**
- `hardware_id` (str): Hardware ID
- `model_id` (str): Model ID
- `output_path` (str): Path to save the dashboard

**Returns:**
- `str`: Path to the generated dashboard

**export_visualization_data_from_db(query_type: str, export_path: str, **kwargs)** -> bool

Exports visualization data from the database.

**Parameters:**
- `query_type` (str): Type of query to export data for
- `export_path` (str): Path to save the exported data
- `**kwargs`: Additional query parameters

**Returns:**
- `bool`: Whether the export was successful

## Framework

### SimulationValidationFramework

A class integrating all components of the simulation validation framework.

#### Constructor

```python
SimulationValidationFramework(
    db_integration: Optional[SimulationValidationDBIntegration] = None,
    visualizer: Optional[ValidationVisualizer] = None,
    basic_calibrator: Optional[BasicCalibrator] = None,
    advanced_calibrator: Optional[AdvancedCalibrator] = None,
    basic_drift_detector: Optional[BasicDriftDetector] = None,
    advanced_drift_detector: Optional[AdvancedDriftDetector] = None,
    basic_validator: Optional[BasicValidator] = None,
    statistical_validator: Optional[StatisticalValidator] = None,
    visualization_db_connector: Optional[ValidationVisualizerDBConnector] = None,
    methodology: Optional[ValidationMethodology] = None,
    enable_logging: bool = False
)
```

**Parameters:**
- `db_integration` (Optional[SimulationValidationDBIntegration]): Database integration instance
- `visualizer` (Optional[ValidationVisualizer]): Visualizer instance
- `basic_calibrator` (Optional[BasicCalibrator]): Basic calibrator instance
- `advanced_calibrator` (Optional[AdvancedCalibrator]): Advanced calibrator instance
- `basic_drift_detector` (Optional[BasicDriftDetector]): Basic drift detector instance
- `advanced_drift_detector` (Optional[AdvancedDriftDetector]): Advanced drift detector instance
- `basic_validator` (Optional[BasicValidator]): Basic validator instance
- `statistical_validator` (Optional[StatisticalValidator]): Statistical validator instance
- `visualization_db_connector` (Optional[ValidationVisualizerDBConnector]): Visualization DB connector instance
- `methodology` (Optional[ValidationMethodology]): Validation methodology instance
- `enable_logging` (bool): Whether to enable logging

#### Methods

**validate_simulation_result(simulation_result_id: str, hardware_result_id: str, validator_type: str = "basic")** -> ValidationResult

Validates a simulation result against a hardware result.

**Parameters:**
- `simulation_result_id` (str): ID of the simulation result
- `hardware_result_id` (str): ID of the hardware result
- `validator_type` (str): Type of validator to use ("basic" or "statistical")

**Returns:**
- `ValidationResult`: The validation result

**generate_validation_report(validation_result_id: str, output_format: str = "markdown", output_path: Optional[str] = None)** -> str

Generates a validation report.

**Parameters:**
- `validation_result_id` (str): ID of the validation result
- `output_format` (str): Format of the report ("markdown", "html", "json")
- `output_path` (Optional[str]): Path to save the report

**Returns:**
- `str`: Content of the report or path to the saved report

**generate_validation_summary_report(validation_results: List[ValidationResult], statistical_summary: Dict[str, Any], output_format: str = "markdown", output_path: Optional[str] = None)** -> str

Generates a summary report for multiple validation results.

**Parameters:**
- `validation_results` (List[ValidationResult]): List of validation results
- `statistical_summary` (Dict[str, Any]): Statistical summary
- `output_format` (str): Format of the report ("markdown", "html", "json")
- `output_path` (Optional[str]): Path to save the report

**Returns:**
- `str`: Content of the report or path to the saved report

**visualize_validation_result(validation_result_id: str, output_path: str, interactive: bool = True)** -> str

Creates a visualization of a validation result.

**Parameters:**
- `validation_result_id` (str): ID of the validation result
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**calibrate_simulation(hardware_type: str, model_type: str, calibrator_type: str = "basic")** -> CalibrationRecord

Calibrates simulation parameters for a hardware and model type.

**Parameters:**
- `hardware_type` (str): Hardware type to calibrate for
- `model_type` (str): Model type to calibrate for
- `calibrator_type` (str): Type of calibrator to use ("basic" or "advanced")

**Returns:**
- `CalibrationRecord`: The calibration record

**generate_calibration_report(calibration_record_id: str, output_format: str = "markdown", output_path: Optional[str] = None)** -> str

Generates a calibration report.

**Parameters:**
- `calibration_record_id` (str): ID of the calibration record
- `output_format` (str): Format of the report ("markdown", "html", "json")
- `output_path` (Optional[str]): Path to save the report

**Returns:**
- `str`: Content of the report or path to the saved report

**visualize_calibration_improvement(calibration_record_id: str, output_path: str, interactive: bool = True)** -> str

Creates a visualization of calibration improvement.

**Parameters:**
- `calibration_record_id` (str): ID of the calibration record
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

**detect_simulation_drift(hardware_type: str, model_type: str, historical_start_date: str, historical_end_date: str, current_start_date: str, current_end_date: str, detector_type: str = "basic")** -> DriftDetectionResult

Detects drift in simulation accuracy.

**Parameters:**
- `hardware_type` (str): Hardware type to analyze
- `model_type` (str): Model type to analyze
- `historical_start_date` (str): Start of historical window (ISO format)
- `historical_end_date` (str): End of historical window (ISO format)
- `current_start_date` (str): Start of current window (ISO format)
- `current_end_date` (str): End of current window (ISO format)
- `detector_type` (str): Type of detector to use ("basic" or "advanced")

**Returns:**
- `DriftDetectionResult`: The drift detection result

**generate_drift_detection_report(drift_detection_result_id: str, output_format: str = "markdown", output_path: Optional[str] = None)** -> str

Generates a drift detection report.

**Parameters:**
- `drift_detection_result_id` (str): ID of the drift detection result
- `output_format` (str): Format of the report ("markdown", "html", "json")
- `output_path` (Optional[str]): Path to save the report

**Returns:**
- `str`: Content of the report or path to the saved report

**visualize_drift_detection(drift_detection_result_id: str, output_path: str, interactive: bool = True)** -> str

Creates a visualization of drift detection.

**Parameters:**
- `drift_detection_result_id` (str): ID of the drift detection result
- `output_path` (str): Path to save the visualization
- `interactive` (bool): Whether to create an interactive visualization

**Returns:**
- `str`: Path to the generated visualization

### ValidationMethodology

A class defining the validation methodology.

#### Constructor

```python
ValidationMethodology()
```

No parameters required.

#### Methods

**define_validation_thresholds(metric_name: str)** -> Dict[str, float]

Defines validation thresholds for a metric.

**Parameters:**
- `metric_name` (str): Name of the metric

**Returns:**
- `Dict[str, float]`: Thresholds for the metric

**calculate_validation_quality(mape: float, metric_name: str)** -> str

Calculates validation quality based on MAPE and metric.

**Parameters:**
- `mape` (float): MAPE value
- `metric_name` (str): Name of the metric

**Returns:**
- `str`: Validation quality ("excellent", "good", "fair", "poor")

**get_validation_recommendations(validation_result: ValidationResult)** -> List[str]

Gets recommendations based on a validation result.

**Parameters:**
- `validation_result` (ValidationResult): The validation result

**Returns:**
- `List[str]`: List of recommendations