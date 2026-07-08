# Dual-Layer Result Aggregation System

This module provides a comprehensive dual-layer system for aggregating and analyzing results from distributed tests, offering powerful tools for data analysis, anomaly detection, comparative analysis, and visualization.

## Architecture Overview

The Distributed Testing Framework features a sophisticated dual-layer result aggregation system consisting of two complementary components:

1. **High-Level Aggregator Service** (`ResultAggregatorService` in `service.py`): Provides efficient aggregation and statistical analysis with a focus on performance, caching, and real-time insights.

2. **Detailed Aggregator** (`ResultAggregator` in `aggregator.py`): Offers in-depth multi-dimensional analysis with advanced visualization capabilities and comprehensive statistical processing.

These components work together to provide both high-level insights and detailed analysis of test results. The dual-layer approach allows for efficient processing of large datasets while still providing deep analytical capabilities.

## System Integration

The dual-layer result aggregation system is fully integrated with the Coordinator Server:

- Both aggregators are initialized and configured during coordinator startup
- A unified data preparation process ensures consistency across both aggregators
- Test results are processed by both aggregators simultaneously
- The coordinator provides a WebSocket API for accessing aggregated results from either aggregator
- Comprehensive error handling and fallback mechanisms ensure reliability

## High-Level Aggregator Service (`ResultAggregatorService`)

The `ResultAggregatorService` is a key component of the Distributed Testing Framework, enabling advanced analysis of test results from multiple workers. It implements a three-phase processing pipeline (preprocessing, aggregation, postprocessing) to analyze different types of results (performance, compatibility, integration, web platform) across multiple levels of aggregation.

## Key Features

- **Multi-Stage Pipeline Architecture**: Three-phase processing with customizable components at each stage
- **Multiple Result Types**: Support for performance, compatibility, integration, and web platform results
- **Flexible Aggregation Levels**: Six distinct aggregation levels (test run, model, hardware, model-hardware pair, task type, worker)
- **Advanced Statistical Analysis**: Comprehensive statistics including means, medians, percentiles, distributions
- **Z-Score Anomaly Detection**: Automatic detection of performance outliers with severity classification
- **Historical Trend Analysis**: Compare current results against historical data with significance testing
- **Metric Correlation Analysis**: Analyze relationships between different metrics with p-value significance testing
- **Intelligent Caching System**: Time-based caching with automatic invalidation for performance optimization
- **Export Capabilities**: Export results in JSON or CSV formats for sharing and visualization

## Usage

### Basic Usage

```python
from duckdb_api.distributed_testing.result_aggregator import (
    ResultAggregatorService,
    RESULT_TYPE_PERFORMANCE,
    AGGREGATION_LEVEL_MODEL
)

# Create aggregator with database manager
aggregator = ResultAggregatorService(db_manager=db_manager)

# Aggregate results by model
results = aggregator.aggregate_results(
    result_type=RESULT_TYPE_PERFORMANCE,
    aggregation_level=AGGREGATION_LEVEL_MODEL
)

# Detect anomalies
anomalies = aggregator.get_result_anomalies(
    result_type=RESULT_TYPE_PERFORMANCE,
    aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE
)

# Export results as JSON
json_output = aggregator.export_results(
    result_type=RESULT_TYPE_PERFORMANCE,
    aggregation_level=AGGREGATION_LEVEL_MODEL,
    format="json"
)
```

### Filtering and Time Range Selection

You can filter results by specific criteria and time ranges:

```python
# Filter by specific model and hardware
results = aggregator.aggregate_results(
    result_type=RESULT_TYPE_PERFORMANCE,
    aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
    filter_params={
        "model_id": "bert-base-uncased",
        "hardware_id": "nvidia_a100",
        "batch_size": 4,
        "precision": "fp16"
    }
)

# Filter by time range (last 7 days)
from datetime import datetime, timedelta
end_time = datetime.now()
start_time = end_time - timedelta(days=7)
time_range = (start_time, end_time)

results = aggregator.aggregate_results(
    result_type=RESULT_TYPE_PERFORMANCE,
    aggregation_level=AGGREGATION_LEVEL_MODEL,
    time_range=time_range
)
```

### Advanced Analysis

The service provides powerful analysis capabilities:

```python
# Get comparison report between current and historical results
comparison_report = aggregator.get_comparison_report(
    result_type=RESULT_TYPE_PERFORMANCE,
    aggregation_level=AGGREGATION_LEVEL_MODEL
)

# Analyze correlations between latency and throughput
correlation_results = aggregator.analyze_correlations(
    result_type=RESULT_TYPE_PERFORMANCE,
    metrics=["average_latency_ms", "throughput_items_per_second", "memory_peak_mb"]
)
```

### Configuration

The service can be configured with various options for fine-tuned control:

```python
aggregator.configure({
    "cache_ttl_seconds": 300,       # Cache time-to-live
    "anomaly_threshold": 2.5,       # Z-score threshold for anomalies
    "min_data_points": 5,           # Minimum data points for analysis
    "model_family_grouping": False, # Whether to group by model family
    "normalize_metrics": True,      # Whether to normalize metrics
    "comparative_lookback_days": 7, # Days to look back for comparison
    "workers_historical_limit": 10, # Maximum workers to include in historical analysis
    "deduplication_enabled": True   # Whether to deduplicate similar results
})
```

### Custom Pipeline Components

The pipeline architecture is designed for extensibility - you can extend the pipeline with custom components:

```python
# Add a custom preprocessor
def custom_preprocessor(results, context):
    """Custom data preprocessing logic."""
    # Filter results based on custom criteria
    filtered_results = [r for r in results if r.get("custom_field") == "value"]
    
    # Add context metadata
    context["metadata"]["custom_filter_count"] = len(results) - len(filtered_results)
    
    return filtered_results

aggregator.register_preprocessor(custom_preprocessor)

# Add a custom aggregator
def custom_aggregator(results, context):
    """Custom aggregation logic."""
    # Compute custom metric
    custom_metric = {}
    for group_key, group_results in context["grouped_results"].items():
        metric_values = [r.get("custom_metric", 0) for r in group_results]
        if metric_values:
            custom_metric[group_key] = sum(metric_values) / len(metric_values)
    
    return {"custom_metrics": custom_metric}

aggregator.register_aggregator(custom_aggregator)

# Add a custom postprocessor
def custom_postprocessor(aggregated_results, context):
    """Custom postprocessing logic."""
    # Add custom analysis results
    custom_analysis = {
        "analysis_timestamp": datetime.now().isoformat(),
        "analysis_type": "custom",
        "summary": "Custom analysis of aggregated results"
    }
    
    aggregated_results["results"]["custom_analysis"] = custom_analysis

aggregator.register_postprocessor(custom_postprocessor)
```

## API Reference

### Main Methods

- `aggregate_results(result_type, aggregation_level, filter_params=None, time_range=None, use_cache=True)`: Aggregates results using the full processing pipeline
- `get_result_anomalies(result_type, aggregation_level, filter_params=None, time_range=None, use_cache=True)`: Gets anomalies in results
- `get_comparison_report(result_type, aggregation_level, filter_params=None, time_range=None, use_cache=True)`: Gets a comparison report between current and historical results
- `analyze_correlations(result_type, metrics, filter_params=None, time_range=None)`: Analyzes correlations between different metrics
- `export_results(result_type, aggregation_level, filter_params=None, time_range=None, format="json", file_path=None)`: Exports aggregated results to a file
- `configure(config_updates)`: Updates the service configuration
- `register_preprocessor(func)`: Registers a preprocessing function
- `register_aggregator(func)`: Registers an aggregation function
- `register_postprocessor(func)`: Registers a postprocessing function
- `clear_cache()`: Clears all result caches

### Constants

#### Result Types

- `RESULT_TYPE_PERFORMANCE`: Performance test results
- `RESULT_TYPE_COMPATIBILITY`: Compatibility test results
- `RESULT_TYPE_INTEGRATION`: Integration test results
- `RESULT_TYPE_WEB_PLATFORM`: Web platform test results

#### Aggregation Levels

- `AGGREGATION_LEVEL_TEST_RUN`: Aggregate by test run
- `AGGREGATION_LEVEL_MODEL`: Aggregate by model
- `AGGREGATION_LEVEL_HARDWARE`: Aggregate by hardware
- `AGGREGATION_LEVEL_MODEL_HARDWARE`: Aggregate by model-hardware pair
- `AGGREGATION_LEVEL_TASK_TYPE`: Aggregate by task type
- `AGGREGATION_LEVEL_WORKER`: Aggregate by worker

## Implementation Details

### Pipeline Phases

1. **Preprocessing**:
   - `_filter_invalid_results`: Removes invalid or corrupted results
   - `_normalize_metrics`: Normalizes metrics to make them comparable across different runs
   - `_deduplicate_results`: Removes duplicate or highly similar results to avoid skewing aggregations

2. **Aggregation**:
   - `_aggregate_basic_statistics`: Calculates mean, median, min, max, std for each metric
   - `_aggregate_percentiles`: Calculates percentiles (p50, p75, p90, p95, p99) for each metric
   - `_aggregate_distributions`: Calculates distribution of categorical values

3. **Postprocessing**:
   - `_detect_anomalies`: Identifies statistical outliers based on Z-score
   - `_comparative_analysis`: Compares current results with historical data
   - `_add_context_metadata`: Enriches results with hardware and model information

### Response Structure

The response from `aggregate_results` follows this structure:

```json
{
  "aggregation_level": "model",
  "result_type": "performance",
  "results": {
    "basic_statistics": {
      "model1": {
        "average_latency_ms": {
          "count": 20,
          "mean": 95.2,
          "median": 94.5,
          "min": 80.1,
          "max": 120.3,
          "std": 8.7
        },
        "throughput_items_per_second": {
          "count": 20,
          "mean": 42.8,
          "median": 43.2,
          "min": 35.6,
          "max": 50.1,
          "std": 4.2
        },
        "result_count": 20
      }
    },
    "percentiles": {
      "model1": {
        "average_latency_ms": {
          "p50": 94.5,
          "p75": 101.2,
          "p90": 105.8,
          "p95": 110.3,
          "p99": 118.7
        }
      }
    },
    "distributions": {},
    "anomalies": {
      "model1:hw1": {
        "average_latency_ms": {
          "anomalies": [
            {
              "value": 120.3,
              "z_score": 2.89,
              "direction": "high",
              "severity": "bad"
            }
          ]
        }
      }
    },
    "comparisons": {
      "model1": {
        "average_latency_ms": {
          "current_mean": 95.2,
          "historical_mean": 105.7,
          "pct_change_mean": -9.93,
          "is_improvement": true,
          "significance": true
        }
      }
    }
  },
  "metadata": {},
  "processed_at": "2025-03-13T01:11:39.318945",
  "raw_result_count": 40,
  "processed_result_count": 38,
  "context": {
    "result_type": "performance",
    "aggregation_level": "model",
    "filter_params": {},
    "current_time": "2025-03-13T01:11:39.320125",
    "processing_time_ms": 432.65,
    "model_metadata": {
      "model1": {
        "name": "Test Model 1",
        "family": "transformer",
        "modality": "text",
        "parameters_million": 110
      }
    }
  }
}
```

### Supported Metrics

#### Performance Metrics
- `total_time_seconds`: Total execution time in seconds
- `average_latency_ms`: Average latency in milliseconds
- `throughput_items_per_second`: Throughput in items per second
- `memory_peak_mb`: Peak memory usage in megabytes

#### Compatibility Metrics
- `is_compatible`: Whether the model is compatible with the hardware
- `detection_success`: Whether hardware detection succeeded
- `initialization_success`: Whether initialization succeeded
- `compatibility_score`: Compatibility score from 0 to 1

#### Integration Metrics
- `passed`: Whether tests passed (1) or failed (0)
- `execution_time_seconds`: Execution time in seconds

#### Web Platform Metrics
- `load_time_ms`: Model load time in milliseconds
- `initialization_time_ms`: Initialization time in milliseconds
- `inference_time_ms`: Inference time in milliseconds
- `total_time_ms`: Total execution time in milliseconds
- `memory_usage_mb`: Memory usage in megabytes
- `success_value`: Whether execution succeeded (1) or failed (0)

## Examples

See [`test_basic_result_aggregator.py`](/test/duckdb_api/distributed_testing/test_basic_result_aggregator.py) for a complete example of using the ResultAggregatorService with sample data.

## Integration with Distributed Testing Framework

The `ResultAggregatorService` is a critical component of the Distributed Testing Framework, integrating seamlessly with:

1. **Coordinator**: Processes raw test results from distributed workers through the coordinator
2. **Dashboard Server**: Provides aggregated data for visualization in the testing dashboard
3. **Task Scheduler**: Uses historical performance data to optimize task scheduling
4. **Health Monitor**: Identifies anomalies that may indicate system health issues
5. **Database API**: Stores aggregated results and retrieves historical data for comparison

To use the aggregator within the larger framework:

```python
from duckdb_api.distributed_testing.coordinator import TestCoordinator
from duckdb_api.distributed_testing.result_aggregator import ResultAggregatorService
from duckdb_api.core.database_manager import BenchmarkDBManager

# Create database manager
db_manager = BenchmarkDBManager(db_path="benchmark_db.duckdb")

# Create result aggregator
aggregator = ResultAggregatorService(db_manager=db_manager)

# Create test coordinator with result aggregator
coordinator = TestCoordinator(
    result_aggregator=aggregator,
    db_manager=db_manager
)

# Run tests and process results
coordinator.run_distributed_tests(test_config)
coordinator.process_results()

# Get aggregated results from the latest test run
latest_results = aggregator.aggregate_results(
    result_type=RESULT_TYPE_PERFORMANCE,
    aggregation_level=AGGREGATION_LEVEL_MODEL,
    filter_params={"run_id": coordinator.current_run_id}
)
```

## Performance Considerations

- The service implements intelligent caching to avoid redundant calculations
- Time-based cache invalidation ensures fresh results without manual intervention
- For large result sets, consider using filtering to reduce the data volume
- The multi-stage pipeline can be extended with custom optimizations for specific use cases
- When analyzing correlations between many metrics, consider batching the analysis to avoid performance issues

## Detailed Result Aggregator

The `ResultAggregator` class in `aggregator.py` provides in-depth multi-dimensional analysis capabilities with a focus on detailed statistical processing and comprehensive visualization.

### Key Features

- **Multi-Dimensional Analysis**: Analyze results across various dimensions (test, worker, hardware, model)
- **Detailed Statistical Processing**: Comprehensive statistical analysis including means, medians, percentiles, distributions
- **Advanced Visualization**: Generate rich visualizations of performance trends and anomalies
- **Regression Detection**: Automatically detect performance regressions with statistical significance testing
- **Dimensional Grouping**: Group results by different dimensions to identify patterns and correlations
- **Export Capabilities**: Export results and visualizations in multiple formats

### Usage

```python
from duckdb_api.distributed_testing.result_aggregator.aggregator import ResultAggregator

# Create aggregator with database manager
aggregator = ResultAggregator(db_manager=db_manager)

# Configure aggregator
aggregator.configure({
    "visualization_enabled": True,
    "history_days": 30,
    "significance_level": 0.05
})

# Start the aggregator
aggregator.start()

# Get dimension analysis
dimension_analysis = aggregator.get_dimension_analysis("model")

# Get regressions
regressions = aggregator.get_regressions()

# Get anomalies
anomalies = aggregator.get_anomalies()

# Get visualizations
visualizations = aggregator.get_visualizations("hardware")

# Stop the aggregator
aggregator.stop()
```

## WebSocket API for Accessing Aggregated Results

The coordinator server provides a WebSocket API to access the results from both aggregators.

### Message Format

To request aggregated results:

```json
{
    "type": "get_aggregated_results",
    "result_type": "performance",
    "aggregation_level": "model",
    "filter_params": {
        "model": "bert-base-uncased",
        "batch_size": 32
    },
    "time_range": {
        "start": "2025-03-01T00:00:00Z",
        "end": "2025-03-15T23:59:59Z"
    },
    "use_detailed": false
}
```

Parameters:
- `result_type`: Type of results to aggregate (performance, compatibility, integration, web_platform)
- `aggregation_level`: Level of aggregation (test_run, model, hardware, model_hardware, task_type, worker)
- `filter_params`: Optional parameters to filter results
- `time_range`: Optional time range to filter results
- `use_detailed`: Whether to use the detailed aggregator (true) or the high-level service (false)

### Response Format

The response from the API varies depending on which aggregator was used:

#### High-Level Aggregator Response:

```json
{
    "type": "get_aggregated_results_result",
    "success": true,
    "results": {
        "aggregation_level": "model",
        "result_type": "performance",
        "results": {
            "basic_statistics": { ... },
            "percentiles": { ... },
            "distributions": { ... },
            "anomalies": { ... },
            "comparisons": { ... }
        },
        "metadata": {
            "aggregator": "high_level",
            "timestamp": "2025-03-15T12:34:56Z"
        }
    }
}
```

#### Detailed Aggregator Response:

```json
{
    "type": "get_aggregated_results_result",
    "success": true,
    "results": {
        "aggregation_level": "model",
        "result_type": "performance",
        "results": {
            "dimension_analysis": { ... },
            "overall_status": { ... },
            "regressions": { ... },
            "anomalies": { ... },
            "visualizations": { ... },
            "metadata": {
                "aggregator": "detailed",
                "timestamp": "2025-03-15T12:34:56Z"
            }
        }
    }
}
```

### Python Client Example

```python
import json
import websockets
import anyio

async def get_aggregated_results(coordinator_url, api_key, result_type, aggregation_level, use_detailed=False):
    async with websockets.connect(coordinator_url) as websocket:
        # Authenticate
        await websocket.send(json.dumps({
            "type": "auth_response",
            "api_key": api_key
        }))
        
        auth_result = json.loads(await websocket.recv())
        if not auth_result.get("success"):
            raise Exception("Authentication failed")
        
        # Get aggregated results
        await websocket.send(json.dumps({
            "type": "get_aggregated_results",
            "result_type": result_type,
            "aggregation_level": aggregation_level,
            "use_detailed": use_detailed
        }))
        
        result = json.loads(await websocket.recv())
        return result["results"]

# Example: Get high-level performance aggregation by model
results = anyio.run(get_aggregated_results,
    "ws://coordinator-host:8080",
    "your-api-key",
    "performance",
    "model",
    use_detailed=False
))
```