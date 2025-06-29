# Drift Detection Components

This directory contains the drift detection components for the Simulation Accuracy and Validation Framework. These components are responsible for detecting when simulation accuracy changes over time, which may indicate a need for recalibration.

## Key Components

- `basic_detector.py`: Basic drift detection using statistical tests and threshold-based detection
- `advanced_detector.py`: Advanced drift detection using machine learning, distribution comparison, and root cause analysis

## Drift Detection Methods

The framework supports the following drift detection methods:

### Basic Methods

1. **Statistical Tests**: Statistical tests (t-test, Kolmogorov-Smirnov) to detect changes in metric distributions
2. **Threshold-Based Detection**: Alert when metrics exceed predefined thresholds
3. **Moving Window Analysis**: Compare recent metrics with historical windows
4. **Trend Analysis**: Detect trends in simulation accuracy over time

### Advanced Methods

1. **Multi-Dimensional Analysis**: Consider multiple metrics and their interactions
2. **Distribution Comparison**: Compare distributions using advanced statistical methods
3. **Root Cause Analysis**: Identify the likely cause of drift when it occurs
4. **Online Drift Detection**: Continuous monitoring for drift in production
5. **Machine Learning-Based Detection**: Use ML for anomaly detection and drift prediction
6. **Confidence-Based Detection**: Account for uncertainty in drift detection

## Using the Drift Detection Components

### Basic Usage

```python
from duckdb_api.simulation_validation.drift_detection.basic_detector import BasicDriftDetector
from duckdb_api.simulation_validation.core.base import ValidationResult

# Create the detector
detector = BasicDriftDetector()

# Detect drift in validation results
historical_results = [...] # List of ValidationResult objects from historical window
recent_results = [...] # List of ValidationResult objects from recent window

drift_result = detector.detect_drift(
    historical_results=historical_results,
    recent_results=recent_results,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased"
)

# Check if significant drift was detected
if drift_result.is_significant:
    print(f"Significant drift detected in {drift_result.hardware_type} - {drift_result.model_type}")
    
    # Analyze drift by metrics
    for metric, details in drift_result.drift_metrics.items():
        if details["drift_detected"]:
            print(f"  Drift in {metric}: {details['mean_change_pct']:.2f}% change (p-value: {details['p_value']:.4f})")
```

### Advanced Usage

```python
from duckdb_api.simulation_validation.drift_detection.advanced_detector import AdvancedDriftDetector

# Create the advanced detector with specific methods
detector = AdvancedDriftDetector(
    method="multi_dimensional",
    confidence_threshold=0.95,
    use_machine_learning=True
)

# Detect drift with advanced methods
drift_result = detector.detect_drift(
    historical_results=historical_results,
    recent_results=recent_results,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    metrics=["throughput_items_per_second", "average_latency_ms"]
)

# Get root cause analysis
if drift_result.is_significant:
    root_causes = detector.analyze_root_causes(drift_result)
    for cause in root_causes:
        print(f"Root cause: {cause['cause']} (confidence: {cause['confidence']:.2f})")
        print(f"  Recommended action: {cause['recommended_action']}")
```

### Database Integration

```python
from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration
from duckdb_api.simulation_validation.drift_detection.advanced_detector import AdvancedDriftDetector

# Create database integration
db_integration = SimulationValidationDBIntegration(db_path="./simulation_db.duckdb")

# Create the detector
detector = AdvancedDriftDetector()

# Detect drift using database
drift_result = detector.detect_drift_from_db(
    db_integration=db_integration,
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    historical_window_start="2025-01-01",
    historical_window_end="2025-01-31",
    new_window_start="2025-02-01",
    new_window_end="2025-02-28"
)

# Store drift detection result in database
db_integration.store_drift_detection_result(drift_result)
```

## Drift Detection Workflow

The typical drift detection workflow includes:

1. **Data Collection**: Gather validation results over time
2. **Window Definition**: Define historical and recent time windows
3. **Metric Selection**: Choose which metrics to monitor for drift
4. **Drift Detection**: Apply drift detection algorithms to detect significant changes
5. **Significance Testing**: Determine if detected drift is statistically significant
6. **Root Cause Analysis**: Identify the likely cause of drift
7. **Action Recommendation**: Recommend actions (e.g., recalibration, model update)
8. **Storage**: Store drift detection results for future reference
9. **Visualization**: Visualize drift detection results and trends
10. **Monitoring**: Continuously monitor for drift in production

## Testing Drift Detection

The drift detection components are tested through both unit tests and the end-to-end testing framework:

```bash
# Run drift detection unit tests
python -m unittest duckdb_api.simulation_validation.drift_detection.test_basic_detector
python -m unittest duckdb_api.simulation_validation.drift_detection.test_advanced_detector

# Run end-to-end tests involving drift detection
./run_visualization_tests.sh --test-type drift
```

## Dependencies

The drift detection components require the following dependencies:
- numpy: Numerical operations
- scipy: Statistical testing
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning algorithms and anomaly detection
- statsmodels: Statistical models and tests

## Integration with Visualization

The drift detection results can be visualized using the visualization components:

```python
from duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector

# Create the connector
connector = ValidationVisualizerDBConnector()

# Create a drift visualization
connector.create_drift_visualization_from_db(
    hardware_type="gpu_rtx3080",
    model_type="bert-base-uncased",
    output_path="./drift_visualization.html"
)
```

## Drift Alerts

The framework can generate alerts when significant drift is detected:

```python
from duckdb_api.simulation_validation.drift_detection.drift_alert_manager import DriftAlertManager

# Create alert manager
alert_manager = DriftAlertManager()

# Configure alerts
alert_manager.configure(
    email_alerts=True,
    dashboard_alerts=True,
    recipients=["team@example.com"]
)

# Set up drift detection process with alerts
if drift_result.is_significant:
    alert_manager.send_alert(
        drift_result=drift_result,
        message="Significant drift detected in simulation accuracy",
        severity="high"
    )
```