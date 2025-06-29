# Multi-Model Resource Pool Integration Guide

This guide explains how to use the Multi-Model Resource Pool Integration module, which connects the Multi-Model Execution Support predictive system with the Web Resource Pool for empirical validation and optimization of execution strategies.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Usage Examples](#usage-examples)
5. [Empirical Validation](#empirical-validation)
6. [Adaptive Optimization](#adaptive-optimization)
7. [Cross-Model Tensor Sharing](#cross-model-tensor-sharing)
8. [Integration with DuckDB](#integration-with-duckdb)
9. [Best Practices](#best-practices)
10. [API Reference](#api-reference)

## Overview

The Multi-Model Resource Pool Integration connects two powerful components:

1. **Multi-Model Execution Support**: Predicts performance metrics for concurrent execution of multiple AI models, accounting for resource contention, sharing benefits, and execution strategies.

2. **WebNN/WebGPU Resource Pool**: Enables actual execution of AI models in browsers with features like fault tolerance, connection pooling, and cross-model tensor sharing.

By connecting these components, the integration enables:

- Validation of performance predictions against actual measurements
- Selection of optimal execution strategies based on empirical data
- Dynamic refinement of prediction models
- Monitoring of prediction accuracy
- Adaptive optimization of execution parameters

## Key Features

- **Prediction-Guided Execution**: Uses performance predictions to select optimal execution strategies
- **Empirical Validation**: Measures actual performance and compares against predictions
- **Adaptive Optimization**: Adjusts strategy parameters based on real-world measurements
- **Strategy Comparison**: Directly compares different execution strategies (parallel, sequential, batched)
- **Continuous Refinement**: Updates prediction models based on observed performance
- **Performance Metrics Collection**: Tracks accuracy and impact of optimizations
- **DuckDB Integration**: Stores validation metrics and optimization impact for analysis

## Architecture

The integration consists of several key components:

```
                 ┌──────────────────┐
                 │    Prediction    │
                 │      System      │
                 └──────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────┐
│   Multi-Model Resource Pool Integration       │
│                                               │
│  ┌─────────────────┐    ┌──────────────────┐  │
│  │   Execution     │    │    Empirical     │  │
│  │    Strategy     │◄───┤    Validation    │  │
│  │   Selection     │    │                  │  │
│  └─────────────────┘    └──────────────────┘  │
│           │                      ▲            │
│           ▼                      │            │
│  ┌─────────────────┐    ┌──────────────────┐  │
│  │    Adaptive     │    │   Performance    │  │
│  │  Optimization   │    │    Metrics       │  │
│  │                 │    │    Collection    │  │
│  └─────────────────┘    └──────────────────┘  │
└───────────────────────────────────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  Web Resource    │
                 │      Pool        │
                 └──────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Browser(s)    │
                 └──────────────────┘
```

## Usage Examples

### Basic Usage

```python
from predictive_performance.multi_model_resource_pool_integration import MultiModelResourcePoolIntegration

# Create integration instance
integration = MultiModelResourcePoolIntegration(
    max_connections=4,            # Maximum browser connections
    enable_empirical_validation=True,
    validation_interval=10,       # Validate every 10 executions
    prediction_refinement=True,   # Update prediction models based on measurements
    enable_adaptive_optimization=True,
    enable_trend_analysis=True,   # Enable error trend analysis
    error_threshold=0.15,         # 15% error threshold for refinement
    verbose=True
)

# Initialize
integration.initialize()

try:
    # Define model configurations
    model_configs = [
        {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
        {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1},
        {"model_name": "whisper-tiny", "model_type": "audio", "batch_size": 1}
    ]
    
    # Execute with automatic strategy recommendation
    result = integration.execute_with_strategy(
        model_configs=model_configs,
        hardware_platform="webgpu",
        execution_strategy=None,  # Auto-select optimal strategy
        optimization_goal="latency"
    )
    
    print(f"Selected strategy: {result['execution_strategy']}")
    print(f"Predicted latency: {result['predicted_latency']:.2f} ms")
    print(f"Actual latency: {result['actual_latency']:.2f} ms")
    
finally:
    # Clean up resources
    integration.close()
```

### Using Enhanced Empirical Validation

```python
from predictive_performance.multi_model_resource_pool_integration import MultiModelResourcePoolIntegration
from predictive_performance.multi_model_empirical_validation import MultiModelEmpiricalValidator

# Create custom validator with specific settings
validator = MultiModelEmpiricalValidator(
    db_path="./validation_metrics.duckdb",
    validation_history_size=200,      # Store more validation records
    error_threshold=0.10,             # Lower threshold (10%) for refinement
    refinement_interval=5,            # Check for refinement more frequently
    enable_trend_analysis=True,
    enable_visualization=True
)

# Create integration with custom validator
integration = MultiModelResourcePoolIntegration(
    validator=validator,              # Use custom validator
    max_connections=4,
    enable_empirical_validation=True,
    prediction_refinement=True,
    enable_adaptive_optimization=True
)

# Initialize
integration.initialize()

try:
    # Run multiple executions to collect validation data
    for i in range(20):
        # Vary configurations to collect diverse validation data
        model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": i % 8 + 1},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
        
        # Execute with different strategies to validate all approaches
        strategy = ["parallel", "sequential", "batched"][i % 3]
        
        result = integration.execute_with_strategy(
            model_configs=model_configs,
            hardware_platform="webgpu",
            execution_strategy=strategy,
            optimization_goal="throughput"
        )
    
    # Get validation metrics with error trends
    metrics = integration.get_validation_metrics()
    
    print("Validation Results:")
    print(f"Validation count: {metrics['validation_count']}")
    
    if 'error_rates' in metrics:
        print("\nError Rates:")
        for metric, value in metrics['error_rates'].items():
            print(f"  {metric}: {value:.2%}")
    
    if 'error_trends' in metrics:
        print("\nError Trends:")
        for metric, trend in metrics['error_trends'].items():
            direction = trend.get('direction', 'unknown')
            strength = trend.get('strength', 0.0)
            print(f"  {metric}: {direction} (strength: {strength:.2f})")
    
    # Get refinement recommendations
    if hasattr(validator, 'get_refinement_recommendations'):
        recommendations = validator.get_refinement_recommendations()
        
        print("\nRefinement Recommendations:")
        print(f"Refinement needed: {recommendations['refinement_needed']}")
        if recommendations['refinement_needed']:
            print(f"Reason: {recommendations['reason']}")
            print(f"Recommended method: {recommendations['recommended_method']}")
    
    # Generate visualization if matplotlib is available
    if hasattr(validator, 'visualize_validation_metrics'):
        try:
            visualization = validator.visualize_validation_metrics(metric_type="error_rates")
            if visualization["success"]:
                print("\nVisualization generated successfully")
                # Save to file if needed
                # visualization["figure"].savefig("error_rates.png")
        except ImportError:
            print("\nVisualization requires matplotlib")
    
finally:
    # Clean up resources
    integration.close()
```

### Comparing Execution Strategies

```python
# Compare different execution strategies
comparison = integration.compare_strategies(
    model_configs=model_configs,
    hardware_platform="webgpu",
    optimization_goal="throughput"
)

print(f"Best strategy: {comparison['best_strategy']}")
print(f"Recommendation accuracy: {comparison['recommendation_accuracy']}")
print(f"Performance improvement: {comparison['optimization_impact']['improvement_percent']:.1f}%")

# Strategy results
for strategy, metrics in comparison["strategy_results"].items():
    print(f"\n{strategy} strategy:")
    print(f"  Throughput: {metrics['throughput']:.2f} items/sec")
    print(f"  Latency: {metrics['latency']:.2f} ms")
    print(f"  Memory: {metrics['memory']:.2f} MB")
```

### Getting Validation Metrics

```python
# Get validation metrics
metrics = integration.get_validation_metrics()

print(f"Validation count: {metrics['validation_count']}")
print(f"Execution count: {metrics['execution_count']}")

# Error rates
if 'error_rates' in metrics:
    print("\nError rates:")
    for metric, value in metrics['error_rates'].items():
        print(f"  {metric}: {value:.2%}")

# Optimization impact
if 'optimization_impact' in metrics:
    print("\nOptimization impact:")
    print(f"  Average improvement: {metrics['optimization_impact'].get('avg_improvement_percent', 0):.1f}%")
    print(f"  Recommendation accuracy: {metrics['optimization_impact'].get('recommendation_accuracy', 0):.2%}")
```

### Updating Strategy Configuration

```python
# Get adaptive configuration based on empirical measurements
adaptive_config = integration.get_adaptive_configuration("webgpu")
print(f"Adaptive configuration: {adaptive_config}")

# Update configuration with custom values
custom_config = {
    "parallel_threshold": 4,      # Use parallel for 4 or fewer models
    "sequential_threshold": 10,   # Use sequential for more than 10 models
    "batching_size": 5,           # Batch size for batched execution
    "memory_threshold": 6000      # Memory threshold in MB
}

integration.update_strategy_configuration("webgpu", custom_config)

# Or update adaptively based on measurements
integration.update_strategy_configuration("webgpu")  # No config provided = adaptive update
```

## Empirical Validation

The empirical validation system compares predicted performance metrics with actual measurements to:

1. Assess prediction accuracy
2. Identify patterns in prediction errors
3. Provide feedback for model improvement
4. Track accuracy trends over time

Validation collects three key error metrics:

- **Throughput Error**: Difference between predicted and actual throughput
- **Latency Error**: Difference between predicted and actual latency
- **Memory Error**: Difference between predicted and actual memory usage

Validation is performed at a configurable interval (default: every 10 executions) to balance overhead and data collection.

### Enhanced Empirical Validation

The system now includes an enhanced empirical validation component (`MultiModelEmpiricalValidator`) that provides sophisticated analysis and refinement capabilities:

- **Error Trend Analysis**: Detects patterns in prediction errors over time
- **Hardware-Specific Validation**: Tracks prediction accuracy by hardware platform
- **Execution Strategy Validation**: Analyzes accuracy by execution strategy
- **Model Refinement Recommendations**: Suggests when and how to refine models
- **Validation Dataset Generation**: Creates datasets for model retraining
- **Visualization**: Generates visualizations of validation metrics and trends

### How Validation Works

1. The integration predicts performance metrics for a given execution
2. It executes the models using the Web Resource Pool
3. It measures actual performance metrics
4. It calculates error rates and stores them in the validator
5. The validator analyzes error patterns and trends
6. If error rates exceed thresholds, it recommends model refinement
7. When refinement is needed, it generates a validation dataset
8. The predictor is updated using the recommended method and dataset
9. The validator records refinement results to track improvements
10. All metrics are stored in a database for long-term analysis

### Validation Methods

The validator supports three refinement methods:

1. **Incremental**: Updates existing models with new data (low overhead)
2. **Window**: Retrains on a sliding window of recent data (moderate overhead)
3. **Weighted**: Creates optimal weighted combinations of models (high accuracy)

The appropriate method is recommended based on error patterns:

- **Incremental**: Used for minor corrections and when errors are small
- **Window**: Recommended when errors show consistent bias or drift over time
- **Weighted**: Suggested when different error patterns appear across platforms

### Validation Metrics

The `get_validation_metrics()` method returns detailed validation statistics:

```python
metrics = integration.get_validation_metrics()
```

Key metrics include:

- **Average Error Rates**: Average prediction error for each metric
- **Recent Error Rates**: Error rates from recent validations
- **Error Trends**: Whether prediction accuracy is improving or worsening
- **Optimization Impact**: Effectiveness of strategy recommendations
- **Strategy Distribution**: Distribution of best strategies across workloads

## Adaptive Optimization

The adaptive optimization system uses empirical measurements to adjust execution strategy parameters:

1. **Parallel Threshold**: Maximum model count for parallel execution
2. **Sequential Threshold**: Minimum model count for sequential execution
3. **Batching Size**: Optimal batch size for batched execution
4. **Memory Threshold**: Memory threshold for contention modeling

### How Adaptive Optimization Works

1. The system analyzes validation metrics for each hardware platform
2. It identifies which strategies perform well or poorly for different workloads
3. It adjusts thresholds based on observed performance
4. It optimizes batch sizes based on efficiency metrics
5. It refines memory thresholds based on actual usage

Adaptive optimization requires sufficient validation data (at least 5 validations) to make meaningful adjustments.

## Cross-Model Tensor Sharing

The integration supports cross-model tensor sharing through the Resource Pool:

```python
# Get models from integration
text_model = integration.resource_pool.get_model(model_type="text", model_name="bert-base-uncased")
vision_model = integration.resource_pool.get_model(model_type="vision", model_name="vit-base-patch16-224")

# Run text model and get embeddings
text_result = text_model(text_input)
embeddings = text_result["embeddings"]

# Share embeddings between models
sharing_result = integration.resource_pool.share_tensor_between_models(
    tensor_data=embeddings,
    tensor_name="text_embeddings",
    producer_model=text_model,
    consumer_models=[vision_model],
    storage_type="webgpu",
    dtype="float32"
)
```

The Multi-Model Execution predictor automatically accounts for sharing benefits in its predictions.

## Integration with DuckDB

The integration can store validation metrics and optimization impact in a DuckDB database:

```python
# Create integration with database
integration = MultiModelResourcePoolIntegration(
    db_path="./benchmark_db.duckdb",
    # Other parameters...
)
```

Two tables are created:

1. **multi_model_validation_metrics**: Stores prediction validation metrics
2. **multi_model_optimization_impact**: Stores optimization impact measurements

You can analyze this data with SQL queries or export it for visualization.

## Best Practices

1. **Start with Default Configuration**: Let the system collect data before making manual adjustments
2. **Enable Empirical Validation**: Always enable validation for production workloads
3. **Use Adaptive Optimization**: Let the system optimize based on your specific workloads
4. **Compare Strategies for Critical Workloads**: Use `compare_strategies()` for important workloads
5. **Monitor Validation Metrics**: Track prediction accuracy and update models as needed
6. **Use Database Integration**: Store metrics for long-term analysis and trend tracking
7. **Custom Configuration per Hardware**: Different hardware platforms need different configurations

## API Reference

### `MultiModelResourcePoolIntegration`

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictor` | `MultiModelPredictor` | Existing predictor instance (creates new if None) |
| `resource_pool` | `ResourcePoolBridgeIntegration` | Existing resource pool (creates new if None) |
| `validator` | `MultiModelEmpiricalValidator` | Existing validator instance (creates new if None) |
| `max_connections` | `int` | Maximum browser connections for resource pool |
| `browser_preferences` | `Dict[str, str]` | Browser preferences by model type |
| `enable_empirical_validation` | `bool` | Whether to enable empirical validation |
| `validation_interval` | `int` | Interval for empirical validation in executions |
| `prediction_refinement` | `bool` | Whether to refine prediction models with empirical data |
| `db_path` | `str` | Path to database for storing results |
| `error_threshold` | `float` | Threshold for acceptable prediction error (15% by default) |
| `enable_adaptive_optimization` | `bool` | Whether to adapt optimization based on measurements |
| `enable_trend_analysis` | `bool` | Whether to analyze error trends over time |
| `verbose` | `bool` | Whether to enable verbose logging |

#### Key Methods

| Method | Description |
|--------|-------------|
| `initialize()` | Initialize the integration |
| `execute_with_strategy()` | Execute models with a specific or recommended strategy |
| `compare_strategies()` | Compare different execution strategies |
| `get_validation_metrics()` | Get validation metrics and error statistics |
| `get_adaptive_configuration()` | Get adaptive configuration based on empirical measurements |
| `update_strategy_configuration()` | Update strategy configuration for a hardware platform |
| `close()` | Close the integration and release resources |

### `MultiModelEmpiricalValidator`

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `db_path` | `str` | Path to database for storing validation metrics |
| `validation_history_size` | `int` | Maximum number of validation records to keep in memory |
| `error_threshold` | `float` | Threshold for acceptable prediction error (15% by default) |
| `refinement_interval` | `int` | Number of validations between model refinements |
| `enable_trend_analysis` | `bool` | Whether to analyze error trends over time |
| `enable_visualization` | `bool` | Whether to enable visualization of validation metrics |
| `verbose` | `bool` | Whether to enable verbose logging |

#### Key Methods

| Method | Description |
|--------|-------------|
| `validate_prediction()` | Validate a prediction against actual measurement |
| `get_validation_metrics()` | Get comprehensive validation metrics |
| `get_refinement_recommendations()` | Get recommendations for model refinement |
| `generate_validation_dataset()` | Generate a validation dataset for model refinement |
| `record_model_refinement()` | Record metrics for a model refinement |
| `visualize_validation_metrics()` | Visualize validation metrics |
| `close()` | Close the validator and release resources |