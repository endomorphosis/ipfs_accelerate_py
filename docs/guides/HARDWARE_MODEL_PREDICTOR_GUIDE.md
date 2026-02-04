# Hardware Model Predictor Guide

## Overview

The Hardware Model Predictor is an advanced component of the IPFS Accelerate Python Framework that unifies hardware selection and performance prediction capabilities. It provides intelligent hardware recommendations and accurately predicts performance metrics for model-hardware combinations based on historical benchmarks and machine learning.

This guide explains how to use the Hardware Model Predictor to make informed deployment decisions and optimize model performance across various hardware platforms.

## Key Features

- **Unified Hardware Selection**: Automatically determines optimal hardware for any model and task
- **Performance Prediction**: Accurately predicts throughput, latency, and memory usage
- **Multi-Layer Prediction**: Uses advanced ML models with fallback to simpler heuristics when needed
- **Precision-Aware**: Provides recommendations optimized for different precision formats (fp32, fp16, int8)
- **Cross-Platform Support**: Covers CUDA, ROCm, MPS, CPU, OpenVINO, WebNN, and WebGPU
- **Visualization Tools**: Generates comprehensive performance comparison visualizations
- **Robust Error Handling**: Gracefully handles missing components or data with intelligent fallbacks
- **Distributed Training Support**: Provides recommendations for distributed configurations

## Installation

The Hardware Model Predictor is included in the IPFS Accelerate Python Framework. To ensure all features are available, install the following dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib duckdb
```

## Command-Line Interface

The Hardware Model Predictor provides a comprehensive command-line interface:

```bash
python hardware_model_predictor.py --model bert-base-uncased --batch-size 8
```

### Key Parameters

- `--model`: Model name
- `--family`: Model family/category (embedding, text_generation, vision, audio, multimodal)
- `--batch-size`: Batch size (default: 1)
- `--seq-length`: Sequence length (default: 128)
- `--mode`: "inference" or "training" (default: inference)
- `--precision`: Precision format - "fp32", "fp16", or "int8" (default: fp32)
- `--hardware`: List of hardware platforms to consider
- `--benchmark-dir`: Directory containing benchmark results
- `--database`: Path to benchmark database
- `--config`: Path to configuration file
- `--debug`: Enable debug logging

### Example Commands

#### Get Hardware Recommendation

```bash
# Basic recommendation
python hardware_model_predictor.py --model bert-base-uncased --batch-size 8

# Recommendation with specific hardware options
python hardware_model_predictor.py --model t5-small --hardware cuda cpu --precision fp16

# Training mode recommendation
python hardware_model_predictor.py --model gpt2 --mode training --batch-size 32
```

#### Generate Prediction Matrix

```bash
# Generate matrix for default models and hardware
python hardware_model_predictor.py --generate-matrix --output-file matrix.json

# Generate matrix with specific hardware options
python hardware_model_predictor.py --generate-matrix --hardware cuda cpu --output-file matrix.json

# Generate matrix and visualizations
python hardware_model_predictor.py --generate-matrix --visualize --output-dir visualizations
```

#### Hardware Detection

```bash
# Detect available hardware
python hardware_model_predictor.py --detect-hardware

# Detect hardware and make recommendation
python hardware_model_predictor.py --detect-hardware --model gpt2 --batch-size 32
```

## Python API

For programmatic use, import the HardwareModelPredictor class:

```python
from hardware_model_predictor import HardwareModelPredictor

# Initialize predictor
predictor = HardwareModelPredictor(
    benchmark_dir="./benchmark_results",
    database_path="./benchmark_db.duckdb",
    config_path="./config.json"
)

# Get hardware recommendation
recommendation = predictor.predict_optimal_hardware(
    model_name="bert-base-uncased",
    model_family="embedding",
    batch_size=8,
    sequence_length=128,
    mode="inference",
    precision="fp32"
)

# Print recommendation
print(f"Primary recommendation: {recommendation['primary_recommendation']}")
print(f"Fallback options: {recommendation['fallback_options']}")
print(f"Explanation: {recommendation['explanation']}")

# Get performance prediction for specific hardware
performance = predictor.predict_performance(
    model_name="bert-base-uncased",
    model_family="embedding",
    hardware="cuda",
    batch_size=8,
    precision="fp32"
)

# Print performance metrics
hw = "cuda"
if hw in performance["predictions"]:
    pred = performance["predictions"][hw]
    print(f"Throughput: {pred.get('throughput', 'N/A'):.2f} items/sec")
    print(f"Latency: {pred.get('latency', 'N/A'):.2f} ms")
    print(f"Memory Usage: {pred.get('memory_usage', 'N/A'):.2f} MB")
```

### Creating Performance Matrices

```python
# Define custom models to test
models = [
    {"name": "bert-base-uncased", "family": "embedding"},
    {"name": "t5-small", "family": "text_generation"},
    {"name": "gpt2", "family": "text_generation"},
    {"name": "vit-base-patch16-224", "family": "vision"}
]

# Define hardware platforms and batch sizes
hardware_platforms = ["cuda", "cpu", "openvino"]
batch_sizes = [1, 16, 64]

# Generate prediction matrix
matrix = predictor.create_hardware_prediction_matrix(
    models=models,
    batch_sizes=batch_sizes,
    hardware_platforms=hardware_platforms,
    precision="fp16",
    mode="inference"
)

# Save matrix to file
import json
with open("custom_matrix.json", "w") as f:
    json.dump(matrix, f, indent=2)

# Generate visualizations
visualization_files = predictor.visualize_matrix(matrix, "visualizations")
```

## Integration Architecture

The Hardware Model Predictor integrates three key systems:

1. **Hardware Selector**: Determines optimal hardware based on model and task characteristics
2. **Performance Predictor**: Predicts performance metrics using machine learning models
3. **Hardware-Model Integration**: Provides compatibility information and fallback mechanisms

![Architecture Diagram](https://example.com/architecture.png)

### Prediction Method Hierarchy

The predictor follows a layered approach to provide the most accurate predictions:

1. **Advanced ML-Based Prediction**: First attempts to use trained ML models for accurate prediction
2. **Hardware Selector Integration**: Falls back to compatibility-based selection if ML prediction fails
3. **Hardware-Model Integration**: Uses compatibility matrices and heuristics as second fallback
4. **Simple Heuristics**: Final fallback using basic rules and known performance patterns

This approach ensures robust predictions even with limited benchmark data.

## Prediction Matrix Format

The prediction matrix JSON format contains comprehensive performance predictions:

```json
{
  "timestamp": "2025-03-02T00:00:00Z",
  "mode": "inference",
  "models": {
    "bert-base-uncased": {
      "name": "bert-base-uncased",
      "category": "embedding",
      "predictions": {
        "cuda": {
          "1": {
            "fp32": {
              "throughput": 120.5,
              "latency_mean": 8.3,
              "memory_usage": 850.2
            },
            "fp16": {
              "throughput": 180.3,
              "latency_mean": 5.5,
              "memory_usage": 425.1
            }
          },
          "8": {
            "fp32": {
              "throughput": 350.4,
              "latency_mean": 22.8,
              "memory_usage": 1250.5
            }
          }
        },
        "cpu": {
          "1": {
            "fp32": {
              "throughput": 25.3,
              "latency_mean": 39.5,
              "memory_usage": 800.1
            }
          }
        }
      }
    }
  },
  "hardware_platforms": ["cuda", "cpu", "openvino"],
  "batch_sizes": [1, 8, 32],
  "precision_options": ["fp32", "fp16"]
}
```

## Visualizations

The Hardware Model Predictor can generate several types of visualizations:

1. **Hardware Comparison**: Bar charts comparing different hardware platforms for each model
2. **Batch Size Scaling**: Line plots showing throughput/latency scaling with batch size
3. **Efficiency Comparison**: Normalized performance compared to CPU baseline
4. **Memory Usage**: Bar charts showing memory requirements by hardware and batch size
5. **Precision Impact**: Comparison of different precision formats (fp32 vs fp16 vs int8)

To generate visualizations:

```bash
python hardware_model_predictor.py --generate-matrix --visualize --output-dir visualizations
```

## Integration with Database System

The Hardware Model Predictor integrates with the benchmark database system:

1. **Data Source**: Uses benchmark database for performance prediction training
2. **Model Storage**: Stores trained prediction models in the database
3. **Result Storage**: Can store predictions in the database for validation
4. **Historical Comparison**: Compares predicted vs. actual performance

The database integration is handled through:

```python
# Connect predictor to database
predictor = HardwareModelPredictor(database_path="./benchmark_db.duckdb")

# Use database as data source for predictions
recommendation = predictor.predict_optimal_hardware(
    model_name="bert-base-uncased",
    model_family="embedding"
)
```

## Advanced Features

### Custom Prediction Models

You can train and use custom prediction models:

```python
from model_performance_predictor import train_prediction_models

# Train custom models on specific data
custom_models = train_prediction_models(
    db_path="./my_benchmark_data.parquet",
    output_dir="./my_models"
)

# Use custom models with predictor
predictor = HardwareModelPredictor()
predictor.prediction_models = custom_models

# Make predictions with custom models
recommendation = predictor.predict_optimal_hardware(...)
```

### Distributed Training Configuration

For distributed training recommendations:

```bash
python hardware_model_predictor.py --model llama-7b --mode training --distributed --gpu-count 8
```

```python
# Programmatic configuration
recommendation = predictor.predict_optimal_hardware(
    model_name="llama-7b",
    model_family="text_generation",
    mode="training",
    batch_size=32,
    is_distributed=True,
    gpu_count=8
)
```

### Web Platform Optimization

For web platform deployment:

```bash
python hardware_model_predictor.py --model bert-base-uncased --hardware webnn webgpu
```

```python
# Check compatibility for web platforms
recommendation = predictor.predict_optimal_hardware(
    model_name="bert-base-uncased",
    model_family="embedding",
    available_hardware=["webnn", "webgpu", "cpu"]
)

# Get browser-specific recommendations
performance = predictor.predict_performance(
    model_name="bert-base-uncased",
    model_family="embedding",
    hardware=["webnn", "webgpu"],
    precision="fp16"
)
```

## Best Practices

### Optimizing Predictions

1. **Use Specific Model Families**: Always specify the correct model family for more accurate predictions
2. **Provide Available Hardware**: Limit hardware options to what's actually available in your environment
3. **Consider Batch Size Impact**: Different hardware performs differently at various batch sizes
4. **Update Benchmark Database**: Regularly add new benchmark results to improve prediction accuracy
5. **Use Appropriate Precision**: Consider fp16/int8 for better performance on supported hardware

### Performance Troubleshooting

If predicted performance doesn't match actual results:

1. **Check Database Coverage**: Ensure benchmark database has similar model-hardware combinations
2. **Verify Hardware Detection**: Make sure hardware is correctly detected and identified
3. **Compare Prediction Sources**: Check if prediction comes from ML models or fallback heuristics
4. **Update Prediction Models**: Retrain models with more recent benchmark data
5. **Adjust Batch Size**: Try different batch sizes to find optimal performance

## Conclusion

The Hardware Model Predictor provides a comprehensive solution for hardware selection and performance prediction in the IPFS Accelerate framework. By leveraging machine learning, historical benchmark data, and intelligent fallback mechanisms, it enables optimal model deployment across a wide range of hardware platforms.

For more information about specific components, see:
- [Hardware Selection Guide](HARDWARE_SELECTION_GUIDE.md) - Details on hardware selection algorithm and configuration
- [Model Benchmarking Guide](MODEL_BENCHMARKING_GUIDE.md) - Guide to benchmarking models across hardware platforms
- [Hardware Model Integration Guide](HARDWARE_MODEL_INTEGRATION_GUIDE.md) - Information on hardware-model compatibility
- [Benchmark Database Guide](BENCHMARK_DATABASE_GUIDE.md) - Guide to the benchmark database system
- [Phase 16 Implementation Update](PHASE16_IMPLEMENTATION_UPDATE.md) - Latest updates on hardware selection and prediction