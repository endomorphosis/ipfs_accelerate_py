# Predictive Performance Modeling System with DuckDB Integration

## Overview

The Predictive Performance Modeling System with DuckDB Integration provides a comprehensive solution for predicting hardware performance, managing hardware-model mappings, and analyzing prediction accuracy. It enables the persistent storage of performance predictions, actual measurements, and machine learning models using DuckDB, a lightweight analytical database.

This guide explains how to use the system to:

- Predict optimal hardware for machine learning models
- Predict performance metrics (throughput, latency, memory usage)
- Record actual performance measurements
- Analyze prediction accuracy
- Store and retrieve ML-based prediction models

## Architecture

The system consists of the following components:

1. **DuckDBPredictorRepository**: Core repository class for storing and retrieving data from DuckDB
2. **HardwareModelPredictorDuckDBAdapter**: Adapter for integrating the existing HardwareModelPredictor with DuckDB
3. **ModelPerformancePredictorDuckDBAdapter**: Adapter for integrating ML-based performance prediction with DuckDB
4. **Command-line Tool**: CLI for accessing functionality from the terminal

### Database Schema

The system uses the following tables in DuckDB:

- **performance_predictions**: Stores performance predictions
- **performance_measurements**: Stores actual performance measurements
- **hardware_model_mappings**: Stores hardware recommendations and compatibility
- **prediction_models**: Stores ML models for performance prediction
- **prediction_errors**: Tracks prediction accuracy
- **recommendation_history**: Records hardware recommendations
- **feature_importance**: Stores feature importance for prediction models

## Installation

### Prerequisites

- Python 3.8+
- DuckDB
- Required Python packages:
  - duckdb
  - numpy
  - pandas
  - joblib

### Setup

1. Install required packages:

```bash
pip install duckdb numpy pandas joblib
```

2. Make the command-line tool executable:

```bash
chmod +x run_predictive_performance_with_duckdb.py
```

## Using the Command-line Tool

### Predicting Optimal Hardware

```bash
python run_predictive_performance_with_duckdb.py predict-hardware \
  --model bert-base-uncased \
  --family embedding \
  --batch-size 8 \
  --precision fp16 \
  --hardware cuda,cpu,rocm,mps,openvino \
  --predict-performance
```

### Predicting Performance on Specific Hardware

```bash
python run_predictive_performance_with_duckdb.py predict-performance \
  --model bert-base-uncased \
  --family embedding \
  --hardware cuda,cpu \
  --batch-size 8 \
  --precision fp16
```

### Recording Actual Measurements

```bash
python run_predictive_performance_with_duckdb.py record-measurement \
  --model bert-base-uncased \
  --family embedding \
  --hardware cuda \
  --batch-size 8 \
  --precision fp16 \
  --throughput 123.45 \
  --latency 7.89 \
  --memory 1024.5
```

### Analyzing Prediction Accuracy

```bash
python run_predictive_performance_with_duckdb.py analyze-predictions \
  --model bert-base-uncased \
  --hardware cuda \
  --metric throughput \
  --days 30
```

### Recording Feedback on Hardware Recommendations

```bash
python run_predictive_performance_with_duckdb.py record-feedback \
  --recommendation-id rec-1234567890 \
  --accepted yes \
  --feedback "Works great on our production environment"
```

### Listing Hardware Recommendations

```bash
python run_predictive_performance_with_duckdb.py list-recommendations \
  --model bert-base-uncased \
  --days 30 \
  --limit 20
```

### Listing Prediction Models

```bash
python run_predictive_performance_with_duckdb.py list-models \
  --target-metric throughput \
  --hardware cuda
```

### Generating Sample Data

```bash
python run_predictive_performance_with_duckdb.py generate-sample-data \
  --num-models 10
```

## Programmatic Usage

### Basic Usage

```python
from duckdb_api.predictive_performance.predictor_repository import DuckDBPredictorRepository
from duckdb_api.predictive_performance.repository_adapter import HardwareModelPredictorDuckDBAdapter

# Create repository and adapter
repository = DuckDBPredictorRepository(db_path="predictive_performance.duckdb")
adapter = HardwareModelPredictorDuckDBAdapter(repository=repository)

# Predict optimal hardware
result = adapter.predict_optimal_hardware(
    model_name="bert-base-uncased",
    model_family="embedding",
    batch_size=8,
    precision="fp16",
    available_hardware=["cuda", "cpu", "rocm"]
)

# Predict performance
performance = adapter.predict_performance(
    model_name="bert-base-uncased",
    model_family="embedding",
    hardware="cuda",
    batch_size=8,
    precision="fp16"
)

# Record actual measurement
measurement = adapter.record_actual_performance(
    model_name="bert-base-uncased",
    model_family="embedding",
    hardware_platform="cuda",
    batch_size=8,
    precision="fp16",
    throughput=123.45,
    latency=7.89,
    memory_usage=1024.5
)
```

### ML Model Integration

```python
from duckdb_api.predictive_performance.repository_adapter import ModelPerformancePredictorDuckDBAdapter
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Create repository and adapter
repository = DuckDBPredictorRepository(db_path="predictive_performance.duckdb")
adapter = ModelPerformancePredictorDuckDBAdapter(repository=repository)

# Train a model
X = np.random.rand(100, 4)  # Features: model_size, batch_size, etc.
y = np.random.rand(100)     # Target: throughput or latency
features = ["model_size", "batch_size", "sequence_length", "precision_numeric"]
model = RandomForestRegressor().fit(X, y)

# Store the model
model_id = adapter.store_model(
    model=model,
    model_type="RandomForestRegressor",
    target_metric="throughput",
    hardware_platform="cuda",
    model_family="embedding",
    features=features,
    training_score=0.95,
    validation_score=0.92,
    test_score=0.90
)

# Load the model later
loaded_model, metadata = adapter.load_model(
    target_metric="throughput",
    hardware_platform="cuda",
    model_family="embedding"
)
```

## Integration with Existing Systems

### Integration with Hardware Model Predictor

The HardwareModelPredictorDuckDBAdapter class seamlessly integrates with the existing HardwareModelPredictor class, adding persistent storage capabilities for predictions and recommendations.

```python
from predictive_performance.hardware_model_predictor import HardwareModelPredictor
from duckdb_api.predictive_performance.repository_adapter import HardwareModelPredictorDuckDBAdapter

# Create predictor and adapter
predictor = HardwareModelPredictor()
adapter = HardwareModelPredictorDuckDBAdapter(predictor=predictor)

# Use the adapter to make predictions and store them in DuckDB
recommendation = adapter.predict_optimal_hardware(model_name="bert-base-uncased")
```

### Integration with Model Performance Predictor

The ModelPerformancePredictorDuckDBAdapter class integrates with the existing model_performance_predictor module, adding persistent storage for ML models and predictions.

```python
from duckdb_api.predictive_performance.repository_adapter import ModelPerformancePredictorDuckDBAdapter

# Create adapter
adapter = ModelPerformancePredictorDuckDBAdapter()

# Use the adapter to make predictions and store them in DuckDB
prediction = adapter.predict(
    model_name="bert-base-uncased",
    hardware_platform="cuda",
    batch_size=8
)
```

## Advanced Features

### Prediction Accuracy Analysis

The system automatically calculates prediction errors when measurements are recorded, allowing for comprehensive analysis of prediction accuracy:

```python
# Analyze prediction accuracy
stats = repository.get_prediction_accuracy_stats(
    model_name="bert-base-uncased",
    hardware_platform="cuda",
    metric="throughput"
)

# Get R² coefficient of determination
r_squared = stats["throughput"]["r_squared"]

# Get mean absolute error
mae = stats["throughput"]["mean_absolute_error"]

# Get bias (systematic error)
bias = stats["throughput"]["bias"]
```

### Feature Importance Tracking

The system automatically tracks feature importance for ML-based prediction models:

```python
# Get feature importance for a model
importance = repository.get_feature_importance(model_id="model-123456")

# Analyze which features are most important for predictions
for feature in importance:
    print(f"{feature['feature_name']}: {feature['importance_score']:.4f}")
```

### Recommendation Feedback Loop

The system supports recording user feedback on hardware recommendations, enabling continuous improvement:

```python
# Record feedback on a recommendation
adapter.record_recommendation_feedback(
    recommendation_id="rec-123456",
    was_accepted=True,
    user_feedback="Works well for our use case"
)

# Analyze acceptance rate
recommendations = repository.get_recommendations()
accepted = sum(1 for r in recommendations if r.get("was_accepted", False))
acceptance_rate = accepted / len(recommendations) if recommendations else 0
print(f"Recommendation acceptance rate: {acceptance_rate:.2%}")
```

## Performance Considerations

### Database Optimization

- The system creates indices on commonly queried fields for fast lookups
- For large datasets, consider using DuckDB's partitioning features
- For write-heavy workloads, consider batching inserts

### Memory Management

- When storing large ML models, consider using model compression techniques
- For large prediction matrices, use the batch methods to reduce memory usage

## Troubleshooting

### Common Issues

1. **Missing Tables**: If you see errors about missing tables, make sure you're connecting to the correct database file and that it was initialized with `create_if_missing=True`.

2. **Serialization Errors**: If you encounter errors when storing ML models, make sure the model is compatible with joblib serialization.

3. **Performance Issues**: If the database operations are slow, consider adding additional indices on frequently queried fields.

## Examples

### End-to-End Hardware Recommendation Workflow

```python
# 1. Predict optimal hardware
recommendation = adapter.predict_optimal_hardware(
    model_name="bert-base-uncased",
    model_family="embedding",
    batch_size=8,
    precision="fp16"
)

# 2. Record user feedback
adapter.record_recommendation_feedback(
    recommendation_id=recommendation["recommendation_id"],
    was_accepted=True
)

# 3. Predict performance on recommended hardware
performance = adapter.predict_performance(
    model_name="bert-base-uncased",
    model_family="embedding",
    hardware=recommendation["primary_recommendation"],
    batch_size=8,
    precision="fp16"
)

# 4. Record actual performance
measurement = adapter.record_actual_performance(
    model_name="bert-base-uncased",
    model_family="embedding",
    hardware_platform=recommendation["primary_recommendation"],
    batch_size=8,
    precision="fp16",
    throughput=120.5,
    latency=8.2,
    memory_usage=1050.3,
    prediction_id=performance["predictions"][recommendation["primary_recommendation"]]["prediction_id"]
)

# 5. Analyze prediction accuracy
stats = repository.get_prediction_accuracy_stats(
    model_name="bert-base-uncased",
    hardware_platform=recommendation["primary_recommendation"]
)
```

### Using ML Models for Performance Prediction

```python
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# 1. Prepare training data from historical measurements
measurements = repository.get_measurements(
    model_family="embedding",
    hardware_platform="cuda",
    limit=1000
)

# Create DataFrame from measurements
df = pd.DataFrame([
    {
        "model_name": m["model_name"],
        "batch_size": m["batch_size"],
        "sequence_length": m["sequence_length"],
        "precision_numeric": 16 if m["precision"] == "fp16" else 32,
        "throughput": m["throughput"],
        "latency": m["latency"],
        "memory_usage": m["memory_usage"]
    }
    for m in measurements
])

# 2. Train a throughput prediction model
X = df[["batch_size", "sequence_length", "precision_numeric"]]
y_throughput = df["throughput"]

model_throughput = GradientBoostingRegressor().fit(X, y_throughput)

# 3. Store the model in the repository
adapter = ModelPerformancePredictorDuckDBAdapter(repository=repository)
model_id = adapter.store_model(
    model=model_throughput,
    model_type="GradientBoostingRegressor",
    target_metric="throughput",
    hardware_platform="cuda",
    model_family="embedding",
    features=["batch_size", "sequence_length", "precision_numeric"],
    training_score=model_throughput.score(X, y_throughput)
)

# 4. Use the model for prediction
loaded_model, _ = adapter.load_model(
    target_metric="throughput",
    hardware_platform="cuda",
    model_family="embedding"
)

# Make a prediction
new_config = pd.DataFrame([{
    "batch_size": 16,
    "sequence_length": 256,
    "precision_numeric": 16
}])

predicted_throughput = loaded_model.predict(new_config)[0]
print(f"Predicted throughput: {predicted_throughput:.2f} items/sec")
```

## Conclusion

The Predictive Performance Modeling System with DuckDB Integration provides a comprehensive solution for hardware performance prediction and analysis. By storing predictions, measurements, and models in a persistent database, it enables continuous improvement of prediction accuracy and informed hardware selection decisions.

The adapter pattern used in this system demonstrates how to effectively integrate existing components with database persistence, following clean design principles and ensuring separation of concerns.