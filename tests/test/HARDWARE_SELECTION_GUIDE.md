# Hardware Selection and Benchmarking System Guide

## Overview

The Hardware Selection System is a key component of the IPFS Accelerate Python Framework's Phase 16 implementation, providing intelligent hardware recommendations for models based on comprehensive benchmark data, model characteristics, and task requirements. This guide explains how to use the enhanced hardware selection and benchmarking system to optimize model deployment across various hardware platforms.

> **March 2025 Update**: This guide includes information about the latest enhancements to the hardware selection system, including improved reliability, robust fallback mechanisms, and the new Enhanced Hardware Benchmark Runner.

> **QNN Integration Update (March 2025)**: Full QNN (Qualcomm Neural Networks)/Hexagon DSP support has been integrated into the hardware selection and benchmarking system. All templates have been updated to support QNN hardware acceleration. See the [QNN Integration Guide](QNN_INTEGRATION_GUIDE.md) for details.

## Key Features

- **Performance-Based Recommendations**: Select optimal hardware based on benchmarking data
- **Machine Learning-Based Prediction**: Predict performance for untested model-hardware combinations
- **Task-Specific Optimization**: Tailor hardware selection for specific tasks and workloads
- **Batch Size Awareness**: Optimize for different batch sizes from single inference to batch processing
- **Model Size Consideration**: Account for model size in hardware selection
- **Fallback Recommendations**: Provide robust fallback options when primary hardware is unavailable
- **Compatibility Checking**: Ensure hardware compatibility based on model family
- **Enhanced Reliability**: Robust error handling and fallback mechanisms
- **Automated Benchmarking**: Integrated benchmark runner with hardware auto-selection
- **Database Integration**: Direct storage of benchmark results in DuckDB database
- **Comprehensive Hardware Support**: Support for CPU, CUDA, ROCm, MPS, OpenVINO, QNN, WebNN, and WebGPU

## Installation

The Hardware Selection System is included in the IPFS Accelerate Python Framework. No additional installation is required beyond the framework's standard dependencies:

```bash
pip install numpy pandas scikit-learn
```

For optimal performance prediction, scikit-learn is recommended but not required.

## Basic Usage

### Command-Line Interface

The Hardware Selector provides a command-line interface for quick hardware recommendations:

```bash
python hardware_selector.py --model-family embedding --model-name bert-base-uncased --batch-size 8 --mode inference
```

Key parameters:
- `--model-family`: Model family (embedding, text_generation, vision, audio, multimodal)
- `--model-name`: Specific model name
- `--batch-size`: Inference or training batch size
- `--mode`: "inference" or "training"
- `--hardware`: Optional list of available hardware types
- `--task`: Alternative to mode, specifies a task like "classification" or "generation"

### Python API

For programmatic use, import the HardwareSelector class:

```python
from hardware_selector import HardwareSelector

# Initialize selector
selector = HardwareSelector(database_path="./benchmark_results")

# Get hardware recommendation
result = selector.select_hardware(
    model_family="text_generation",
    model_name="gpt2",
    batch_size=1,
    mode="inference"
)

# Print recommendation
print(f"Primary hardware: {result['primary_recommendation']}")
print(f"Fallback options: {result['fallback_options']}")
```

### Task-Specific Selection

For task-specific hardware selection:

```python
result = selector.select_hardware_for_task(
    model_family="vision",
    model_name="vit-base-patch16-224",
    task_type="classification",
    batch_size=32
)
```

Supported task types include:
- `classification`: Image/text classification
- `generation`: Text generation
- `embedding`: Embedding generation
- `speech_recognition`: Speech to text
- `fine_tuning`: Model fine-tuning
- `training`: Model training

## Hardware Selection Map

The hardware selection system can generate a comprehensive selection map for all model families, sizes, and batch sizes:

```bash
python hardware_selector.py --create-map --output hardware_map.json
```

This generates a JSON file with hardware recommendations for various combinations of:
- Model families
- Model sizes (small, medium, large)
- Batch sizes (1, 4, 16, 32, 64)
- Modes (inference, training)

Example selection map structure:

```json
{
  "timestamp": "2025-03-02T12:34:56.789Z",
  "model_families": {
    "embedding": {
      "model_sizes": {
        "small": {
          "inference": {
            "primary": "webnn",
            "fallbacks": ["cuda", "cpu"]
          },
          "training": {
            "primary": "cuda",
            "fallbacks": ["rocm", "cpu"]
          }
        }
      },
      "inference": {
        "batch_sizes": {
          "1": {
            "primary": "webnn",
            "fallbacks": ["cuda", "cpu"]
          },
          "32": {
            "primary": "cuda",
            "fallbacks": ["rocm", "cpu"]
          }
        }
      }
    }
  }
}
```

## Performance Prediction

The Hardware Selection System includes a machine learning-based performance prediction component that:

1. Trains on benchmark data to predict performance metrics
2. Provides estimated latency and throughput for untested configurations
3. Updates predictions as new benchmark data becomes available

To see prediction details in hardware selection:

```python
result = selector.select_hardware(
    model_family="vision",
    model_name="vit-base-patch16-224",
    batch_size=16,
    mode="inference"
)

# Check predictions for each hardware type
for hw_type, scores in result['all_scores'].items():
    if 'predictions' in scores:
        print(f"{hw_type} predictions:")
        print(f"  Latency: {scores['predictions'].get('latency', 'N/A')} ms")
        print(f"  Throughput: {scores['predictions'].get('throughput', 'N/A')} items/sec")
```

## Configuration Options

The Hardware Selection System can be configured with a custom configuration file:

```bash
python hardware_selector.py --config my_config.json
```

Configuration options include:

### Selection Criteria Weights

```json
"selection_criteria": {
  "inference": {
    "latency_weight": 0.4,
    "throughput_weight": 0.3,
    "memory_weight": 0.2,
    "compatibility_weight": 0.1
  },
  "training": {
    "throughput_weight": 0.4,
    "convergence_weight": 0.3,
    "memory_weight": 0.2,
    "compatibility_weight": 0.1
  }
}
```

### Model Family Preferences

```json
"model_families": {
  "embedding": {
    "batch_size_importance": "medium",
    "model_size_importance": "low"
  },
  "text_generation": {
    "batch_size_importance": "low",
    "model_size_importance": "high"
  }
}
```

### Hardware Preferences

```json
"hardware_preferences": {
  "cpu": {
    "cost_factor": 1.0,
    "availability_factor": 1.0,
    "power_factor": 0.8
  },
  "cuda": {
    "cost_factor": 0.7,
    "availability_factor": 0.8,
    "power_factor": 0.4
  }
}
```

### Batch Size and Model Size Thresholds

```json
"batch_size_thresholds": {
  "small": 1,
  "medium": 8,
  "large": 32
},
"model_size_categories": {
  "small": 100000000,  // ~100M parameters
  "medium": 1000000000,  // ~1B parameters
  "large": 10000000000  // ~10B parameters
}
```

## Hardware Compatibility Matrix

The Hardware Selection System uses a hardware compatibility matrix to determine which hardware is compatible with each model family. This matrix can be:

1. Loaded from the benchmark database
2. Parsed from the `CLAUDE.md` file
3. Created as a default if no other source is available

Example compatibility matrix:

```json
{
  "timestamp": "2025-03-01T00:00:00Z",
  "hardware_types": ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"],
  "model_families": {
    "embedding": {
      "hardware_compatibility": {
        "cpu": {"compatible": true, "performance_rating": "medium"},
        "cuda": {"compatible": true, "performance_rating": "high"},
        "webnn": {"compatible": true, "performance_rating": "high"}
      }
    }
  }
}
```

## Integration with Benchmark Database

The Hardware Selection System integrates with the benchmark database to:

1. Load performance data for various model-hardware combinations
2. Train prediction models on historical benchmark results
3. Update recommendations based on new benchmark data

The benchmark database organization:

```
benchmark_results/
├── raw_results/           # Raw benchmark result files
├── processed_results/     # Processed and aggregated results
│   └── aggregated_benchmarks.json
├── visualizations/        # Performance visualization plots
└── prediction_models/     # Trained prediction models
```

## Enhanced Hardware Benchmark Runner

The Enhanced Hardware Benchmark Runner is a new addition to the framework that provides automated benchmarking with built-in hardware selection.

### Key Features

- **Automated Hardware Selection**: Automatically selects optimal hardware for benchmarking
- **Multiple Model Support**: Run benchmarks for multiple models in a single command
- **Distributed Training Integration**: Automated configuration for distributed training benchmarks
- **Database Integration**: Direct storage of results in the benchmark database
- **Result Aggregation**: Combines results from multiple batch sizes and creates aggregate metrics
- **Performance Reporting**: Generates comprehensive performance reports

### Basic Usage

```python
from enhanced_hardware_benchmark_runner import EnhancedHardwareBenchmarkRunner

# Initialize benchmark runner
runner = EnhancedHardwareBenchmarkRunner()

# Run benchmark with automatic hardware selection
result = runner.run_benchmark(
    model_name="bert-base-uncased",
    model_family="embedding"
)
```

### Command-Line Interface

```bash
# Run benchmark with automatic hardware selection
python enhanced_hardware_benchmark_runner.py --model bert-base-uncased

# Specify model family, hardware, and batch sizes
python enhanced_hardware_benchmark_runner.py --model gpt2 --model-family text_generation --hardware cuda --batch-sizes 1,4,8

# Run in training mode
python enhanced_hardware_benchmark_runner.py --model bert-base-uncased --mode training
```

### Running Multiple Models

```bash
# Run benchmarks for multiple models and create report
python enhanced_hardware_benchmark_runner.py --multiple-models --model-families embedding,vision --create-report
```

### Distributed Training Benchmarks

```bash
# Run distributed training benchmark
python enhanced_hardware_benchmark_runner.py --model gpt2 --distributed --gpu-count 4 --max-memory-gb 16
```

### Database Integration

```bash
# Run benchmark and store results in database
python enhanced_hardware_benchmark_runner.py --model bert-base-uncased --database-path ./benchmark_db.duckdb

# Disable database storage
python enhanced_hardware_benchmark_runner.py --model bert-base-uncased --no-database
```

## Fallback and Error Handling

The system includes robust fallback mechanisms for handling various error scenarios:

### Prediction Model Fallbacks

When prediction models cannot be trained due to insufficient data:

```python
# Initialize selector with debug logging to see fallback behavior
import logging
logging.getLogger("hardware_selector").setLevel(logging.DEBUG)

selector = HardwareSelector()

# This will use fallback rules if no training data is available
result = selector.select_hardware(
    model_family="multimodal",  # May have limited training data
    model_name="clip-vit-base-patch32",
    batch_size=1,
    mode="inference"
)
```

### Hardware Unavailability

When recommended hardware is not available:

```python
# Specify available hardware options
available_hardware = ["cpu", "openvino"]  # No GPU available

# Get recommendation for available hardware only
result = selector.select_hardware(
    model_family="text_generation",
    model_name="gpt2",
    batch_size=1,
    mode="inference",
    available_hardware=available_hardware
)
```

### scikit-learn Unavailability

The system works even when scikit-learn is not available, using rule-based fallbacks for hardware selection.

## Best Practices

### When to Use Each Hardware Type

- **CPU**: 
  - Small models with low latency requirements
  - When no specialized hardware is available
  - For development and testing

- **CUDA (NVIDIA GPUs)**:
  - Large language models
  - High-throughput batch processing
  - Training and fine-tuning

- **ROCm (AMD GPUs)**:
  - Alternative to CUDA for batch processing
  - Medium to large models
  - Training with compatible frameworks

- **MPS (Apple Silicon)**:
  - Mac-based deployment
  - Medium-sized models
  - Good balance of performance and power efficiency

- **OpenVINO**:
  - Intel CPU/GPU optimization
  - Vision models
  - Edge deployment scenarios

- **QNN**:
  - Qualcomm Snapdragon devices
  - Mobile and edge deployment
  - Power-efficient inference
  - Specialized for on-device AI

- **WebNN**:
  - Browser-based deployment
  - Small to medium models
  - Embedding and vision tasks

- **WebGPU**:
  - Advanced browser-based deployment
  - GPU acceleration in web applications
  - When WebNN is not available

### Optimizing Hardware Selection

1. **Match batch size to hardware capabilities**: Large batch sizes work best on GPU, small batch sizes can be efficient on CPU
2. **Consider model size**: Larger models benefit more from GPU acceleration
3. **Balance performance and cost**: Sometimes a slightly slower but more available hardware option is preferable
4. **Test fallback options**: Ensure your application works well with fallback hardware options
5. **Update benchmark data**: Regularly run benchmarks to keep hardware recommendations current

## Troubleshooting

### Common Issues

1. **Missing Benchmark Data**:
   - Problem: No benchmark data available for specific model-hardware combination
   - Solution: Run benchmarks for the model on target hardware or use prediction-based selection

2. **Inconsistent Recommendations**:
   - Problem: Hardware recommendations change between runs
   - Solution: Check if benchmark data is being properly loaded and processed

3. **Unexpected Hardware Selection**:
   - Problem: System recommends hardware that seems suboptimal
   - Solution: Check weights in selection criteria and hardware preferences

4. **Prediction Model Errors**:
   - Problem: Errors when using performance prediction
   - Solution: Ensure scikit-learn is installed and benchmark database has sufficient data

### Debugging Tools

1. **Enable Debug Logging**:
   ```bash
   python hardware_selector.py --debug --model-family embedding --model-name bert-base-uncased
   ```

2. **Examine All Scores**:
   ```python
   result = selector.select_hardware(...)
   for hw_type, scores in result['all_scores'].items():
       print(f"{hw_type}: {scores}")
   ```

3. **Check Compatibility Matrix**:
   ```python
   print(json.dumps(selector.compatibility_matrix, indent=2))
   ```

## Advanced Usage

### Custom Hardware Types

To add support for custom hardware types:

1. Add to hardware preferences configuration
2. Update compatibility matrix
3. Add encoding in `_encode_hardware_type` method
4. Add to fallback order list

### Custom Model Size Estimation

The system estimates model size based on model name. To customize this:

1. Extend the `_estimate_model_size` method
2. Add recognition patterns for your models

### Custom Selection Logic

To implement custom selection logic:

1. Subclass the `HardwareSelector` class
2. Override the `select_hardware` method
3. Implement your custom logic while maintaining compatibility

## Conclusion

The Enhanced Hardware Selection and Benchmarking System provides a powerful, data-driven approach to choosing optimal hardware for model deployment and automating performance benchmarks. By leveraging benchmark data, model characteristics, and task requirements, it enables efficient use of available hardware resources.

The March 2025 enhancements significantly improve the reliability and robustness of the system, ensuring consistent performance even in challenging environments or when dependencies are missing.

Key improvements include:
- Robust transaction handling in the benchmark database
- Comprehensive fallback mechanisms for prediction models
- Enhanced error handling throughout the system
- New components like the Enhanced Hardware Benchmark Runner
- Extensive test coverage for all critical components

These enhancements ensure that the system can operate reliably in diverse computing environments with different hardware configurations, providing optimal performance recommendations even in edge cases.

---

For more information, see related documentation:
- [Phase 16 Improvements](PHASE16_IMPROVEMENTS.md) - Details about the recent reliability enhancements
- [Hardware Model Predictor Guide](HARDWARE_MODEL_PREDICTOR_GUIDE.md) - Unified hardware selection and prediction system
- [Hardware Benchmarking Guide](HARDWARE_BENCHMARKING_GUIDE.md) - Guide to benchmarking models on different hardware
- [Web Platform Audio Testing Guide](WEB_PLATFORM_AUDIO_TESTING_GUIDE.md) - Specialized audio model testing for web platforms
- [Training Benchmark System](HARDWARE_MODEL_INTEGRATION_GUIDE.md) - Guide to training benchmarks
- [Phase 16 Implementation Summary](PHASE16_IMPLEMENTATION_SUMMARY_UPDATED.md) - Complete summary of Phase 16 implementation

*Last updated: March 5, 2025*