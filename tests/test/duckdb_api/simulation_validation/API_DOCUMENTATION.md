# Database Predictive Analytics API Documentation

## Overview

The Database Predictive Analytics API provides advanced time series forecasting and proactive optimization capabilities for the DuckDB database used by the Simulation Accuracy and Validation Framework. It enables proactive identification of potential performance issues before they occur, helping maintain optimal database performance with minimal manual intervention. The system uses multiple forecasting methods including ARIMA, exponential smoothing, and linear regression to predict future database metrics.

## Key Features

1. **Time Series Forecasting**: Predicts future values of database performance metrics using statistical and machine learning methods.
2. **Ensemble Forecasting**: Combines multiple forecasting methods for improved accuracy.
3. **Threshold-Based Alerts**: Predicts when metrics will exceed warning/error thresholds.
4. **Proactive Recommendations**: Suggests optimization actions before problems occur.
5. **Visualization**: Generates charts of historical data and predictions.
6. **Comprehensive Analysis**: Provides a holistic view of database health forecast.

## Getting Started

### Installation

The Database Predictive Analytics module is part of the IPFS Accelerate Python Framework. For optimal functionality, ensure you have the following dependencies installed:

```bash
pip install duckdb pandas matplotlib statsmodels scipy
```

### Basic Usage

The module can be accessed through the CLI or directly through the Python API.

#### Using the CLI

```bash
# Generate a forecast for database metrics
python run_database_performance_monitoring.py predictive --action forecast --horizon medium_term

# Generate visualizations for forecasted metrics
python run_database_performance_monitoring.py predictive --action visualize --format html --output forecast_viz.html

# Get alerts for potential future threshold violations
python run_database_performance_monitoring.py predictive --action alerts

# Get proactive recommendations based on forecasts
python run_database_performance_monitoring.py predictive --action recommend

# Run a comprehensive analysis of database health forecast
python run_database_performance_monitoring.py predictive --action analyze --horizon medium_term --visualize
```

#### Using the Python API

```python
from duckdb_api.simulation_validation.db_performance_optimizer import get_db_optimizer
from duckdb_api.simulation_validation.automated_optimization_manager import get_optimization_manager
from duckdb_api.simulation_validation.database_predictive_analytics import DatabasePredictiveAnalytics

# Create database optimizer
db_optimizer = get_db_optimizer(db_path="./benchmark_db.duckdb")

# Create automated optimization manager
auto_manager = get_optimization_manager(db_optimizer=db_optimizer)

# Create predictive analytics instance
predictive = DatabasePredictiveAnalytics(
    automated_optimization_manager=auto_manager
)

# Generate forecasts
forecast_result = predictive.forecast_database_metrics(
    horizon="medium_term",
    specific_metrics=["query_time", "storage_size"]
)

# Generate visualizations
vis_result = predictive.generate_forecast_visualizations(
    forecast_results=forecast_result,
    output_format="file"
)

# Check for predicted threshold alerts
alert_result = predictive.check_predicted_thresholds(
    forecast_results=forecast_result
)

# Get proactive recommendations
rec_result = predictive.recommend_proactive_actions(
    forecast_results=forecast_result,
    threshold_alerts=alert_result
)

# Run comprehensive analysis
analysis_result = predictive.analyze_database_health_forecast(
    horizon="medium_term",
    generate_visualizations=True,
    output_format="file"
)
```

## API Reference

### CLI Options

The `predictive` command in `run_database_performance_monitoring.py` accepts the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--action` | Predictive action to perform (forecast, visualize, alerts, recommend, analyze) | analyze |
| `--horizon` | Forecast horizon (short_term, medium_term, long_term) | medium_term |
| `--metrics` | Specific metrics to analyze (space-separated list) | all monitored metrics |
| `--config` | Path to configuration file for predictive analytics | None |
| `--format` | Output format for results (text, json, markdown, html) | text |
| `--output` | Output file path for results | None (stdout) |
| `--visualize` | Generate visualizations | False |
| `--visual-format` | Visualization output format (base64, file, object) | base64 |
| `--visual-dir` | Directory to save visualizations | ./visualizations |
| `--theme` | Visualization theme (light, dark) | light |

### DatabasePredictiveAnalytics Class

#### Constructor

```python
DatabasePredictiveAnalytics(
    automated_optimization_manager,
    config: Optional[Dict[str, Any]] = None
)
```

- `automated_optimization_manager`: Instance of `AutomatedOptimizationManager`
- `config`: Optional configuration dictionary

#### Methods

##### forecast_database_metrics

```python
forecast_database_metrics(
    horizon: str = "medium_term",
    specific_metrics: Optional[List[str]] = None
) -> Dict[str, Any]
```

Forecasts database performance metrics for the specified horizon.

- `horizon`: Forecast horizon ("short_term", "medium_term", or "long_term")
- `specific_metrics`: List of specific metrics to forecast (None for all)
- Returns: Dictionary containing forecast results for each metric

##### generate_forecast_visualizations

```python
generate_forecast_visualizations(
    forecast_results: Dict[str, Any],
    output_format: str = "base64"
) -> Dict[str, Any]
```

Generates visualizations of forecast results.

- `forecast_results`: Results from `forecast_database_metrics`
- `output_format`: Format for visualization output ("base64", "file", or "object")
- Returns: Dictionary containing visualizations for each metric

##### check_predicted_thresholds

```python
check_predicted_thresholds(
    forecast_results: Dict[str, Any]
) -> Dict[str, Any]
```

Checks if forecasted values will exceed thresholds in the future.

- `forecast_results`: Results from `forecast_database_metrics`
- Returns: Dictionary containing alerts for metrics predicted to exceed thresholds

##### recommend_proactive_actions

```python
recommend_proactive_actions(
    forecast_results: Dict[str, Any],
    threshold_alerts: Dict[str, Any]
) -> Dict[str, Any]
```

Recommends proactive actions based on forecast and threshold alerts.

- `forecast_results`: Results from `forecast_database_metrics`
- `threshold_alerts`: Results from `check_predicted_thresholds`
- Returns: Dictionary containing recommended proactive actions

##### analyze_database_health_forecast

```python
analyze_database_health_forecast(
    horizon: str = "medium_term",
    specific_metrics: Optional[List[str]] = None,
    generate_visualizations: bool = True,
    output_format: str = "base64"
) -> Dict[str, Any]
```

Comprehensive analysis of database health forecast with recommendations.

- `horizon`: Forecast horizon ("short_term", "medium_term", or "long_term")
- `specific_metrics`: List of specific metrics to forecast (None for all)
- `generate_visualizations`: Whether to generate visualizations
- `output_format`: Format for visualization output
- Returns: Dictionary containing comprehensive analysis results

## Forecasting Methods

The module implements three primary forecasting methods and an ensemble approach:

1. **ARIMA** (AutoRegressive Integrated Moving Average): For time series with temporal dependencies.
2. **Exponential Smoothing**: For time series with level, trend, and seasonal components.
3. **Linear Regression**: For time series with linear trends.
4. **Ensemble Forecasting**: Combines the above methods for improved accuracy.

## Configuration

The module can be configured using a JSON configuration file with the following structure:

```json
{
  "metrics_to_forecast": [
    "query_time",
    "storage_size",
    "index_efficiency",
    "vacuum_status",
    "compression_ratio",
    "read_efficiency",
    "write_efficiency",
    "cache_performance"
  ],
  "forecasting": {
    "short_term_horizon": 7,
    "medium_term_horizon": 30,
    "long_term_horizon": 90,
    "confidence_level": 0.95,
    "min_data_points": 10,
    "forecast_methods": ["arima", "exponential_smoothing", "linear_regression"],
    "use_ensemble": true
  },
  "anomaly_detection": {
    "enabled": true,
    "lookback_window": 30,
    "z_score_threshold": 3.0,
    "sensitivity": 0.95
  },
  "thresholds": {
    "use_auto_manager_thresholds": true,
    "predicted_threshold_factor": 0.8,
    "custom_thresholds": {}
  },
  "visualization": {
    "enabled": true,
    "theme": "light",
    "show_confidence_intervals": true,
    "forecast_colors": {
      "historical": "#1f77b4",
      "forecast": "#ff7f0e",
      "lower_bound": "#2ca02c",
      "upper_bound": "#d62728",
      "anomalies": "#9467bd"
    },
    "figure_size": [10, 6],
    "dpi": 100
  }
}
```

## Example Output

### Forecast Result

```json
{
  "timestamp": "2025-03-14T22:32:23.708147",
  "horizon": "medium_term",
  "horizon_days": 30,
  "forecasts": {
    "query_time": {
      "metric": "query_time",
      "historical_values": [...],
      "historical_dates": [...],
      "forecast_values": [...],
      "forecast_dates": [...],
      "lower_bound": [...],
      "upper_bound": [...],
      "trend_analysis": {
        "direction": "increasing",
        "magnitude": "moderate",
        "percent_change": 15.3
      }
    },
    ...
  },
  "anomalies": {...},
  "status": "success"
}
```

### Threshold Alerts

```json
{
  "timestamp": "2025-03-14T22:32:23.737777",
  "alerts": {
    "query_time": [
      {
        "severity": "warning",
        "threshold": 400.0,
        "forecasted_value": 425.3,
        "forecast_date": "2025-04-02T22:32:23.708147",
        "days_until": 7,
        "message": "query_time predicted to exceed warning threshold (425.3 >= 400.0) on 2025-04-02T22:32:23.708147"
      }
    ]
  },
  "status": "warning"
}
```

### Proactive Recommendations

```json
{
  "timestamp": "2025-03-14T22:32:23.737788",
  "recommendations": [
    {
      "metric": "query_time",
      "severity": "warning",
      "urgency": "this_week",
      "days_until": 7,
      "message": "query_time predicted to exceed warning threshold (425.3 >= 400.0) on 2025-04-02T22:32:23.708147",
      "recommended_actions": [
        "optimize_queries",
        "create_indexes",
        "analyze_tables"
      ]
    }
  ],
  "proactive_actions": {
    "query_time": [
      "optimize_queries",
      "create_indexes",
      "analyze_tables"
    ]
  },
  "summary": [
    "This Week: 1 warning issues predicted"
  ],
  "status": "warning"
}
```

## Best Practices

1. **Data Collection**: Ensure you have sufficient historical data for accurate forecasting (at least 10 data points).
2. **Ensemble Forecasting**: Enable ensemble forecasting for improved accuracy.
3. **Threshold Tuning**: Adjust the `predicted_threshold_factor` to match your proactive management needs.
4. **Regular Updates**: Re-run forecasts periodically to capture the latest trends.
5. **Visualization**: Use visualizations to identify patterns and better understand forecast results.

## Limitations

1. Forecasting accuracy depends on the quality and quantity of historical data.
2. Sudden changes in workload patterns may not be captured by statistical forecasting methods.
3. Visualization features require matplotlib, which may not be available in all environments.
4. Advanced forecasting methods (ARIMA, exponential smoothing) require the statsmodels package.