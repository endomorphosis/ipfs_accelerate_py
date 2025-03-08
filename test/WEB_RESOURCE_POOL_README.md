# WebGPU/WebNN Resource Pool Integration Testing

This directory contains tools for testing and evaluating the WebGPU/WebNN Resource Pool Integration, which enables efficient execution of AI models in browsers with enhanced resource management.

## Overview

The WebGPU/WebNN Resource Pool Integration provides:

- **Concurrent Model Execution**: Run multiple models simultaneously across WebGPU and CPU backends
- **Browser Connection Pooling**: Efficiently manage browser connections with automatic lifecycle control
- **Adaptive Resource Scaling**: Dynamically adjust resources based on workload demands
- **Multi-Browser Support**: Optimal model placement across Chrome, Firefox, and Edge
- **Memory Efficiency**: Reduce memory usage through browser resource sharing

## Test Script Usage

The `test_web_resource_pool.py` script provides comprehensive testing capabilities:

### Basic Model Testing

```bash
# Test specific models
python test_web_resource_pool.py --models bert,vit,whisper

# Test with custom connection limit
python test_web_resource_pool.py --models bert,vit --max-connections 6
```

### Advanced Testing Features

```bash
# Test concurrent model execution
python test_web_resource_pool.py --concurrent-models --models bert,vit,whisper

# Test batch processing with different batch sizes
python test_web_resource_pool.py --batch-test --models bert

# Run stress test with multiple models
python test_web_resource_pool.py --stress-test --models bert,vit,whisper

# Test multi-browser support with model placement
python test_web_resource_pool.py --browser-test --browsers chrome,firefox,edge --models bert,vit,whisper

# Test memory efficiency through browser resource sharing
python test_web_resource_pool.py --memory-test --models bert,vit,t5,whisper
```

### Browser Comparison Benchmarking

```bash
# Run comprehensive browser comparison across multiple models
python test_web_resource_pool.py --browser-comparison --models bert,vit,whisper --browsers chrome,firefox,edge

# Test a single model with different browsers
python test_web_resource_pool.py --single-browser-test bert --browsers chrome,firefox,edge

# Run parallel browser benchmarks
python test_web_resource_pool.py --browser-comparison --models bert,vit,whisper,t5,clip --parallel

# Run more iterations for higher confidence
python test_web_resource_pool.py --browser-comparison --models bert,vit --num-runs 10
```

The browser comparison benchmarks will:
1. Test each model on each browser
2. Identify the optimal browser for each model family
3. Generate comprehensive reports with visualizations
4. Provide browser preference configurations you can add to your application

### Quantization Testing

```bash
# Run comprehensive quantization tests across models and browsers
python test_web_resource_pool.py --quantization-test --models bert,vit,whisper --browsers chrome,firefox

# Test quantization for a single model
python test_web_resource_pool.py --single-quant-test bert --browsers chrome 

# Run more iterations for higher confidence
python test_web_resource_pool.py --quantization-test --models bert --num-runs 5
```

The quantization testing framework:
1. Tests multiple precision levels (16-bit, 8-bit, 4-bit, mixed precision)
2. Identifies optimal precision settings for each model and browser
3. Analyzes performance and memory trade-offs
4. Creates detailed reports with visualizations
5. Provides recommended quantization settings for your application

### Performance Reporting

```bash
# Generate performance dashboard in markdown format
python test_web_resource_pool.py --perf-dashboard --models bert,vit,whisper

# Generate HTML performance dashboard with visualizations
python test_web_resource_pool.py --perf-dashboard --models bert,vit,whisper --report-format html
```

## Additional Configuration Options

- `--adaptive-scaling`: Enable adaptive resource scaling (default: True)
- `--monitoring-interval`: Set resource monitoring interval in seconds (default: 30)
- `--report-format`: Choose format for performance reports ('markdown' or 'html')

## Report Generation

The performance dashboard includes:

1. **Model Performance Summary**: Latency, throughput, and variability metrics for each model
2. **Resource Utilization**: Connection counts, utilization percentages, and execution statistics
3. **Visualizations**: Charts for latency comparison, throughput comparison, inference time variability, and resource utilization

Reports are saved in the `webgpu_reports` directory along with raw performance data and visualizations.

## Implementation Details

For implementation details, refer to:
- `resource_pool_bridge.py`: Core implementation of the WebGPU/WebNN resource pool
- `websocket_bridge.py`: Communication bridge between Python and browsers
- `WEB_RESOURCE_POOL_INTEGRATION.md`: Comprehensive documentation of the architecture

## Requirements

- Python 3.8+
- Selenium with browser drivers (Chrome, Firefox, Edge)
- Matplotlib (optional, for visualizations)
- Websockets (for real browser communication)